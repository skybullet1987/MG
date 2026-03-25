# region imports
from AlgorithmImports import *
from execution import *
from datetime import timedelta
# endregion

"""
mg2_exits.py -- Right-Tail Capture Exit Logic for Leader Breakout v0
=====================================================================

Exit framework (all with explicit exit tags):

  1. FailedBreakout  -- within FAILED_BREAKOUT_BARS, price falls back below
                        breakout_level; the setup never confirmed.
  2. HardStop        -- price drops >= effective_stop from entry.
  3. DeadTrade       -- after DEAD_TRADE_MINUTES, pnl < DEAD_TRADE_PNL_MAX.
  4. BreakevenStop   -- after partial TP taken, price retreats to breakeven.
  5. RunnerTrail     -- ATR-based trail activates after RUNNER_ACTIVATION_PNL gain.
  6. KalmanTrail     -- Kalman trail activates after KALMAN_TRAIL_MIN_PNL gain.

Partial TP action (not a full exit):
  - At PARTIAL_TP_PNL: sell PARTIAL_TP_FRACTION, let the rest ride.
  - At most once per trade.

Design notes:
  - No RSI scalp exits, no stagnation micro-exits, no volume dry-up exits.
  - No tiny interval time stops.
  - Designed to capture rare large winners.
"""

# ── Constants ─────────────────────────────────────────────────────────────────
FAILED_BREAKOUT_BARS   = 10
FAILED_BREAKOUT_GRACE  = 3

HARD_STOP_PCT          = 0.035
ATR_SL_MULT            = 2.5

DEAD_TRADE_MINUTES     = 45
DEAD_TRADE_PNL_MAX     = 0.005

RUNNER_ACTIVATION_PNL  = 0.06
RUNNER_ATR_MULT_TIGHT  = 2.0
RUNNER_ATR_MULT_MID    = 2.5
RUNNER_ATR_MULT_WIDE   = 3.5

KALMAN_TRAIL_MIN_PNL   = 0.04
KALMAN_TRAIL_BUFFER    = 0.015

PARTIAL_TP_PNL         = 0.15
PARTIAL_TP_FRACTION    = 0.30
PARTIAL_TP_BREAKEVEN   = 0.005

MIN_HOLD_MINUTES       = 3


def check_exits(algo):
    if algo.IsWarmingUp:
        return
    if algo._rate_limit_until is not None and algo.Time < algo._rate_limit_until:
        return
    for kvp in algo.Portfolio:
        if not is_invested_not_dust(algo, kvp.Key):
            algo._failed_exit_attempts.pop(kvp.Key, None)
            algo._failed_exit_counts.pop(kvp.Key, None)
            continue
        if algo._failed_exit_counts.get(kvp.Key, 0) >= 3:
            continue
        check_exit(algo, kvp.Key, algo.Securities[kvp.Key].Price, kvp.Value)

    for kvp in algo.Portfolio:
        symbol = kvp.Key
        if not is_invested_not_dust(algo, symbol):
            continue
        if symbol not in algo.entry_prices:
            algo.entry_prices[symbol]   = kvp.Value.AveragePrice
            algo.highest_prices[symbol] = kvp.Value.AveragePrice
            algo.entry_times[symbol]    = algo.Time
            algo.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")


def check_exit(algo, symbol, price, holding):
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return

    min_notional_usd = get_min_notional_usd(algo, symbol)
    if price > 0 and abs(holding.Quantity) * price < min_notional_usd * 0.3:
        try:
            algo.Liquidate(symbol)
        except Exception as e:
            algo.Debug(f"Dust liquidation failed for {symbol.Value}: {e}")
        cleanup_position(algo, symbol)
        algo._failed_exit_counts.pop(symbol, None)
        return

    actual_qty   = abs(holding.Quantity)
    rounded_sell = round_quantity(algo, symbol, actual_qty)
    if rounded_sell > actual_qty:
        cleanup_position(algo, symbol)
        algo._failed_exit_counts.pop(symbol, None)
        return

    if symbol not in algo.entry_prices:
        algo.entry_prices[symbol]   = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol]    = algo.Time

    entry   = algo.entry_prices[symbol]
    highest = algo.highest_prices.get(symbol, entry)
    if price > highest:
        algo.highest_prices[symbol] = price
        highest = price

    pnl     = (price - entry) / entry if entry > 0 else 0.0
    hours   = (algo.Time - algo.entry_times.get(symbol, algo.Time)).total_seconds() / 3600
    minutes = hours * 60

    crypto  = algo.crypto_data.get(symbol)
    atr     = crypto.get("atr") if crypto else None
    atr_val = (float(atr.Current.Value)
               if atr and hasattr(atr, "IsReady") and atr.IsReady and atr.Current.Value > 0
               else None)

    if atr_val and entry > 0:
        atr_stop       = (atr_val * ATR_SL_MULT) / entry
        effective_stop = max(atr_stop, HARD_STOP_PCT)
    else:
        effective_stop = HARD_STOP_PCT

    if minutes < MIN_HOLD_MINUTES:
        return

    # 1. Failed Breakout: within first FAILED_BREAKOUT_BARS minutes
    if crypto and FAILED_BREAKOUT_GRACE <= minutes <= FAILED_BREAKOUT_BARS:
        breakout_level = float(crypto.get("breakout_level", 0.0))
        if breakout_level > 0 and price < breakout_level * 0.998:
            _do_exit(algo, symbol, holding, "FailedBreakout", price,
                     min_notional_usd, pnl, hours)
            return

    # 2. Hard Stop / Structural Invalidation
    if pnl <= -effective_stop:
        _do_exit(algo, symbol, holding, "HardStop", price,
                 min_notional_usd, pnl, hours)
        return

    # 3. Breakeven Stop (after partial TP)
    if (algo._partial_tp_taken.get(symbol, False)
            and symbol in algo._breakeven_stops
            and price <= algo._breakeven_stops[symbol]):
        _do_exit(algo, symbol, holding, "BreakevenStop", price,
                 min_notional_usd, pnl, hours)
        return

    # 4. Dead Trade Kill
    if minutes >= DEAD_TRADE_MINUTES and pnl < DEAD_TRADE_PNL_MAX:
        _do_exit(algo, symbol, holding, "DeadTrade", price,
                 min_notional_usd, pnl, hours)
        return

    # 5. Partial TP action (at most once)
    if (not algo._partial_tp_taken.get(symbol, False)
            and pnl >= PARTIAL_TP_PNL
            and minutes >= MIN_HOLD_MINUTES):
        if partial_smart_sell(algo, symbol, PARTIAL_TP_FRACTION, "PartialTP"):
            algo._partial_tp_taken[symbol] = True
            algo._breakeven_stops[symbol]  = entry * (1.0 + PARTIAL_TP_BREAKEVEN)
            algo.Debug(
                f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} "
                f"| Sold {PARTIAL_TP_FRACTION:.0%} | Stop->entry+{PARTIAL_TP_BREAKEVEN:.1%}"
            )
        return

    # 6. Runner ATR Trail
    if pnl >= RUNNER_ACTIVATION_PNL and atr_val and entry > 0:
        if pnl > 0.12:
            mult = RUNNER_ATR_MULT_TIGHT
        elif pnl > 0.07:
            mult = RUNNER_ATR_MULT_MID
        else:
            mult = RUNNER_ATR_MULT_WIDE

        trail_level = highest - atr_val * mult
        if crypto:
            crypto["trail_stop"] = trail_level
        if (crypto and crypto.get("trail_stop") is not None
                and holding.Quantity > 0
                and price <= crypto["trail_stop"]):
            _do_exit(algo, symbol, holding, "RunnerTrail", price,
                     min_notional_usd, pnl, hours)
            return

    # 7. Kalman Trail
    if pnl >= KALMAN_TRAIL_MIN_PNL and crypto:
        kalman_est = float(crypto.get("kalman_estimate", 0.0))
        if kalman_est > 0 and price < kalman_est * (1.0 - KALMAN_TRAIL_BUFFER):
            _do_exit(algo, symbol, holding, "KalmanTrail", price,
                     min_notional_usd, pnl, hours)
            return


def _do_exit(algo, symbol, holding, tag, price, min_notional_usd, pnl, hours):
    """Execute exit, record metadata, apply cooldowns."""
    if price * abs(holding.Quantity) < min_notional_usd * 0.9:
        return

    sold = smart_liquidate(algo, symbol, tag)
    if sold:
        algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
        algo.entry_volumes.pop(symbol, None)
        algo.Debug(f"EXIT [{tag}]: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")

        diag = getattr(algo, "_diagnostics", None)
        if diag is not None:
            sp = get_spread_pct(algo, symbol)
            diag.record_exit(symbol, price, tag, spread_at_exit=sp)

        if pnl < 0:
            algo._symbol_loss_cooldowns[symbol] = algo.Time + timedelta(hours=1)
    else:
        fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
        algo._failed_exit_counts[symbol] = fail_count
        algo.Debug(
            f"EXIT FAILED ({tag}) #{fail_count}: {symbol.Value} | "
            f"PnL:{pnl:+.2%} | Held:{hours:.1f}h"
        )
        if fail_count >= 3:
            algo.Debug(f"FATAL EXIT: {symbol.Value} escalating to market")
            try:
                qty = abs(holding.Quantity)
                if qty > 0:
                    algo.MarketOrder(symbol, -qty, tag=f"ForceExit({tag})")
            except Exception as e:
                algo.Debug(f"Force market exit error for {symbol.Value}: {e}")
            algo._failed_exit_counts.pop(symbol, None)
            algo.entry_volumes.pop(symbol, None)
