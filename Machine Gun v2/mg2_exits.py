# region imports
from AlgorithmImports import *
from execution import *
from datetime import timedelta
# endregion

"""
mg2_exits.py - Right-Tail Capture Exit Logic for Machine Gun v2
"""

# Exit parameter defaults

PARTIAL_TP_PNL_THRESHOLD  = 0.08
PARTIAL_TP_FRACTION       = 0.25   # reduced from 0.40 — let 75% of position ride the runner
PARTIAL_TP_BREAKEVEN_GAIN = 0.005
TRAIL_ACTIVATION     = 0.06
TRAIL_STOP_PCT       = 0.040
TRAIL_ATR_MULT_WIDE  = 3.5
TRAIL_ATR_MULT_MID   = 2.5
TRAIL_ATR_MULT_TIGHT = 2.0   # raised from 1.5 — don't clip large winners too tightly
STAGNATION_MINUTES        = 90
STAGNATION_PNL_THRESHOLD  = 0.006
TIME_STOP_HOURS           = 4.0
TIME_STOP_PNL_MIN         = 0.005
EXTENDED_TIME_STOP_HOURS  = 6.0
EXTENDED_TIME_STOP_MAX    = 0.020
STALE_POSITION_HOURS      = 8.0
STALE_POSITION_PNL_SKIP   = 0.08   # skip stale exit if trade is up more than this
KALMAN_TRAIL_MIN_PNL      = 0.04
KALMAN_TRAIL_BUFFER       = 0.010
MIN_HOLD_MINUTES_FOR_TRAIL = 10


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
            algo.Debug(f"DUST liquidation failed for {symbol.Value}: {e}")
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
    pnl     = (price - entry) / entry if entry > 0 else 0
    dd      = (highest - price) / highest if highest > 0 else 0
    hours   = (algo.Time - algo.entry_times.get(symbol, algo.Time)).total_seconds() / 3600
    minutes = hours * 60
    crypto  = algo.crypto_data.get(symbol)
    atr     = crypto["atr"].Current.Value if crypto and crypto["atr"].IsReady else None
    if atr and entry > 0:
        sl = max((atr * algo.atr_sl_mult) / entry, algo.tight_stop_loss)
        tp = max((atr * algo.atr_tp_mult) / entry, algo.quick_take_profit)
    else:
        sl = algo.tight_stop_loss
        tp = algo.quick_take_profit
    exit_spread = get_spread_pct(algo, symbol)
    if exit_spread is not None:
        spread_cost = exit_spread * 0.5
        tp = tp + spread_cost
        sl = max(sl - spread_cost * 0.5, 0.005)
    if tp < sl * 1.5:
        tp = sl * 1.5
    if algo._choppy_regime_entries.get(symbol, False):
        tp = tp * 0.70
        effective_partial_tp = PARTIAL_TP_PNL_THRESHOLD * 0.65
    else:
        effective_partial_tp = PARTIAL_TP_PNL_THRESHOLD
    trailing_activation = getattr(algo, "trail_activation", TRAIL_ACTIVATION)
    trailing_stop_pct   = getattr(algo, "trail_stop_pct",   TRAIL_STOP_PCT)
    stagnation_minutes  = getattr(algo, "stagnation_minutes", STAGNATION_MINUTES)

    # ── RSI momentum exit: flag overbought peak, exit on RSI rollover ─────────
    if crypto and crypto["rsi"].IsReady and pnl > 0.02 and minutes >= MIN_HOLD_MINUTES_FOR_TRAIL:
        rsi_val = crypto["rsi"].Current.Value
        if rsi_val > 78:
            algo.rsi_peaked_overbought[symbol] = True
        elif algo.rsi_peaked_overbought.get(symbol, False) and rsi_val < 72:
            _do_exit(algo, symbol, holding, "RSI Momentum Exit", price, min_notional_usd, pnl, hours)
            return

    if (algo._partial_tp_taken.get(symbol, False)
            and symbol in algo._breakeven_stops
            and price <= algo._breakeven_stops[symbol]):
        _do_exit(algo, symbol, holding, "Breakeven Stop", price, min_notional_usd, pnl, hours)
        return
    if not algo._partial_tp_taken.get(symbol, False) and pnl <= -sl:
        _do_exit(algo, symbol, holding, "Stop Loss", price, min_notional_usd, pnl, hours)
        return
    if (not algo._partial_tp_taken.get(symbol, False)
            and pnl >= effective_partial_tp
            and minutes >= MIN_HOLD_MINUTES_FOR_TRAIL):
        if partial_smart_sell(algo, symbol, PARTIAL_TP_FRACTION, "Partial TP"):
            algo._partial_tp_taken[symbol] = True
            algo._breakeven_stops[symbol]  = entry * (1.0 + PARTIAL_TP_BREAKEVEN_GAIN)
            algo.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL->entry+{PARTIAL_TP_BREAKEVEN_GAIN:.1%}")
        return
    if not algo._partial_tp_taken.get(symbol, False) and pnl >= tp:
        _do_exit(algo, symbol, holding, "Take Profit", price, min_notional_usd, pnl, hours)
        return
    if minutes >= MIN_HOLD_MINUTES_FOR_TRAIL:
        if pnl > trailing_activation and dd >= trailing_stop_pct:
            _do_exit(algo, symbol, holding, "Trailing Stop", price, min_notional_usd, pnl, hours)
            return
        if atr and entry > 0 and holding.Quantity != 0:
            if pnl > 0.12:
                effective_trail_mult = TRAIL_ATR_MULT_TIGHT
            elif pnl > 0.07:
                effective_trail_mult = TRAIL_ATR_MULT_MID
            else:
                effective_trail_mult = TRAIL_ATR_MULT_WIDE
            trail_offset = atr * effective_trail_mult
            trail_level  = highest - trail_offset
            if crypto:
                crypto["trail_stop"] = trail_level
            if (crypto and crypto["trail_stop"] is not None
                    and holding.Quantity > 0
                    and price <= crypto["trail_stop"]):
                _do_exit(algo, symbol, holding, "ATR Trail", price, min_notional_usd, pnl, hours)
                return
        if pnl >= KALMAN_TRAIL_MIN_PNL and crypto:
            kalman_est = crypto.get("kalman_estimate", 0.0)
            if kalman_est > 0 and price < kalman_est * (1.0 - KALMAN_TRAIL_BUFFER):
                _do_exit(algo, symbol, holding, "Kalman Trail", price, min_notional_usd, pnl, hours)
                return
    if minutes > stagnation_minutes and pnl < STAGNATION_PNL_THRESHOLD:
        _do_exit(algo, symbol, holding, "Stagnation Exit", price, min_notional_usd, pnl, hours)
        return
    if hours >= TIME_STOP_HOURS and pnl < TIME_STOP_PNL_MIN:
        _do_exit(algo, symbol, holding, "Time Stop", price, min_notional_usd, pnl, hours)
        return
    if hours >= 2.0 and crypto and len(crypto["volume"]) >= 2:
        entry_vol = algo.entry_volumes.get(symbol, 0)
        if entry_vol > 0:
            v1 = crypto["volume"][-1]
            v2 = crypto["volume"][-2]
            if v1 < entry_vol * 0.40 and v2 < entry_vol * 0.40:
                _do_exit(algo, symbol, holding, "Volume Dry-up", price, min_notional_usd, pnl, hours)
                return
    # Extended time stop: skip if trade is meaningfully profitable (let runner run)
    if hours >= EXTENDED_TIME_STOP_HOURS and pnl < EXTENDED_TIME_STOP_MAX and pnl < 0.05:
        _do_exit(algo, symbol, holding, "Extended Time Stop", price, min_notional_usd, pnl, hours)
        return
    # Stale position exit: skip if still generating healthy profit (runner still alive)
    if hours >= STALE_POSITION_HOURS and pnl < STALE_POSITION_PNL_SKIP:
        _do_exit(algo, symbol, holding, "Stale Position Exit", price, min_notional_usd, pnl, hours)
        return


def _do_exit(algo, symbol, holding, tag, price, min_notional_usd, pnl, hours):
    if price * abs(holding.Quantity) < min_notional_usd * 0.9:
        return
    sold = smart_liquidate(algo, symbol, tag)
    if sold:
        algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
        algo.rsi_peaked_overbought.pop(symbol, None)
        algo.entry_volumes.pop(symbol, None)
        algo._choppy_regime_entries.pop(symbol, None)
        algo.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
        diag = getattr(algo, "_diagnostics", None)
        if diag is not None:
            sp = get_spread_pct(algo, symbol)
            diag.record_exit(symbol, price, tag, spread_at_exit=sp)
        if pnl < 0:
            algo._symbol_loss_cooldowns[symbol] = algo.Time + timedelta(hours=1)
    else:
        fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
        algo._failed_exit_counts[symbol] = fail_count
        algo.Debug(f"EXIT FAILED ({tag}) #{fail_count}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
        if fail_count >= 3:
            algo.Debug(f"FATAL EXIT FAILURE: {symbol.Value} escalating to market")
            try:
                qty = abs(holding.Quantity)
                if qty > 0:
                    algo.MarketOrder(symbol, -qty, tag=f"Force Exit (fail#{fail_count})")
            except Exception as e:
                algo.Debug(f"Force market exit error for {symbol.Value}: {e}")
            algo._failed_exit_counts.pop(symbol, None)
            algo.rsi_peaked_overbought.pop(symbol, None)
            algo.entry_volumes.pop(symbol, None)
            algo._choppy_regime_entries.pop(symbol, None)
