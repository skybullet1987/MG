"""
Machine Gun 3 — Exit-handling helper functions
===============================================
Plain functions that check exit conditions and execute liquidations for
open positions.

All functions receive the algorithm instance as ``algo`` so they can read and
write algorithm state without inheriting from QCAlgorithm.
"""
from datetime import timedelta

import config as MG3Config
from execution import (
    is_invested_not_dust, smart_liquidate, partial_smart_sell,
    cleanup_position, get_min_notional_usd, round_quantity,
)
from mg3_constants import (
    POSITION_STATE_FLAT, POSITION_STATE_OPEN,
    POSITION_STATE_EXITING, POSITION_STATE_RECOVERING,
)


# ---------------------------------------------------------------------------
# CheckExits — iterate all open positions and call the per-position check
# ---------------------------------------------------------------------------

def check_exits(algo):
    if algo.IsWarmingUp:
        return
    if algo._rate_limit_until is not None and algo.Time < algo._rate_limit_until:
        return

    for kvp in algo.Portfolio:
        symbol = kvp.Key
        if not is_invested_not_dust(algo, symbol):
            algo._failed_exit_attempts.pop(symbol, None)
            algo._failed_exit_counts.pop(symbol, None)
            continue

        state = algo._get_position_state(symbol)

        # RECOVERING: exit retries exhausted — use a direct market order.
        if state == POSITION_STATE_RECOVERING:
            force_market_liquidate(algo, symbol)
            continue

        # Belt-and-suspenders: if failure count reached threshold but state
        # was not updated (e.g. failures counted via OnOrderEvent Invalid),
        # escalate now.
        if algo._failed_exit_counts.get(symbol, 0) >= MG3Config.MAX_EXIT_RETRIES:
            algo._set_position_state(symbol, POSITION_STATE_RECOVERING)
            force_market_liquidate(algo, symbol)
            continue

        check_exit(algo, symbol, algo.Securities[symbol].Price, kvp.Value)

    # Orphan recovery: brokerage holdings not in entry_prices tracking
    for kvp in algo.Portfolio:
        symbol = kvp.Key
        if not is_invested_not_dust(algo, symbol):
            continue
        if symbol not in algo.entry_prices:
            algo.entry_prices[symbol]  = kvp.Value.AveragePrice
            algo.highest_prices[symbol] = kvp.Value.AveragePrice
            algo.entry_times[symbol]   = algo.Time
            algo.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")


# ---------------------------------------------------------------------------
# Force-market liquidation for RECOVERING positions
# ---------------------------------------------------------------------------

def force_market_liquidate(algo, symbol):
    """Force market liquidation for a position stuck in RECOVERING state.

    Called when the normal limit-exit path has failed MAX_EXIT_RETRIES times.
    Uses a market order so the position is closed regardless of spread.
    Does NOT call cleanup_position — OnOrderEvent handles cleanup on fill.
    """
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return  # exit already in flight

    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        algo._set_position_state(symbol, POSITION_STATE_FLAT)
        algo._failed_exit_counts.pop(symbol, None)
        return

    holding = algo.Portfolio[symbol]
    qty     = abs(holding.Quantity)
    if qty == 0:
        algo._set_position_state(symbol, POSITION_STATE_FLAT)
        return

    direction_mult = -1 if holding.Quantity > 0 else 1
    entry  = algo.entry_prices.get(symbol, holding.AveragePrice)
    price  = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    pnl    = (price - entry) / entry if entry > 0 and price > 0 else 0

    algo.Debug(
        f"⚠️ FORCE MARKET LIQUIDATE (recovering): "
        f"{symbol.Value} qty={qty:.6f} PnL:{pnl:+.2%}"
    )
    try:
        algo.MarketOrder(symbol, qty * direction_mult, tag="Force Recovery Exit")
        algo._set_position_state(symbol, POSITION_STATE_EXITING)
        algo._failed_exit_counts.pop(symbol, None)
        algo.mg3_recovery_events += 1
    except Exception as e:
        algo.Debug(f"FORCE MARKET LIQUIDATE failed for {symbol.Value}: {e}")


# ---------------------------------------------------------------------------
# Per-position exit logic
# ---------------------------------------------------------------------------

def check_exit(algo, symbol, price, holding):
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return

    min_notional_usd = get_min_notional_usd(algo, symbol)

    # Dust cleanup: position value below 30% of minimum notional
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
        algo.Debug(
            f"DUST (rounded sell > actual): {symbol.Value} | "
            f"actual={actual_qty} rounded={rounded_sell} — cleaning up"
        )
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

    pnl    = (price - entry) / entry if entry > 0 else 0
    crypto = algo.crypto_data.get(symbol)
    dd     = (highest - price) / highest if highest > 0 else 0
    hours  = (algo.Time - algo.entry_times.get(symbol, algo.Time)).total_seconds() / 3600
    minutes = hours * 60

    # ATR-based stop-loss and take-profit levels
    atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None
    if atr and entry > 0:
        sl = max((atr * algo.atr_sl_mult) / entry, algo.tight_stop_loss)
        tp = max((atr * algo.atr_tp_mult) / entry, algo.quick_take_profit)
    else:
        sl = algo.tight_stop_loss
        tp = algo.quick_take_profit

    if tp < sl * 1.5:
        tp = sl * 1.5

    # Tighten TP when entered in a choppy (ADX < 25) regime
    if algo._choppy_regime_entries.get(symbol, False):
        tp = tp * 0.65   # 35 % tighter — trend continuation unlikely

    # Tighten TP in low-volatility regime to secure profits faster
    if algo.volatility_regime == "low":
        tp = tp * 0.75   # 25 % tighter — low-vol moves mean-revert quickly

    trailing_activation = algo.trail_activation
    trailing_stop_pct   = algo.trail_stop_pct

    # Track RSI overbought peak for RSI momentum exit
    if crypto and crypto['rsi'].IsReady:
        rsi_now = crypto['rsi'].Current.Value
        if rsi_now > 85:
            algo.rsi_peaked_overbought[symbol] = True

    # ------------------------------------------------------------------
    # Partial take-profit at +2.5 %: sell 50 %, set breakeven stop
    # ------------------------------------------------------------------
    if (not algo._partial_tp_taken.get(symbol, False)
            and pnl >= algo.partial_tp_threshold):
        if partial_smart_sell(algo, symbol, 0.50, "Partial TP"):
            algo._partial_tp_taken[symbol]  = True
            algo._breakeven_stops[symbol]   = entry * 1.002
            algo.Debug(
                f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL→entry+0.2%"
            )
            return  # Don't trigger full exit this bar

    # ------------------------------------------------------------------
    # Determine exit tag
    # ------------------------------------------------------------------
    tag = ""

    if algo._partial_tp_taken.get(symbol, False):
        be_price = algo._breakeven_stops.get(symbol, entry)
        if price <= be_price:
            tag = "Breakeven Stop"
    elif pnl <= -sl:
        tag = "Stop Loss"

    if not tag and minutes > algo.stagnation_minutes and pnl < algo.stagnation_pnl_threshold:
        tag = "Stagnation Exit"

    elif not tag:
        if not algo._partial_tp_taken.get(symbol, False) and pnl >= tp:
            tag = "Take Profit"

        elif pnl > trailing_activation and dd >= trailing_stop_pct:
            tag = "Trailing Stop"

        elif atr and entry > 0 and holding.Quantity != 0:
            trail_offset = atr * algo.atr_trail_mult
            trail_level  = highest - trail_offset  # anchored to highest price since entry
            if crypto:
                crypto['trail_stop'] = trail_level
            if crypto and crypto['trail_stop'] is not None:
                if holding.Quantity > 0 and price <= crypto['trail_stop']:
                    tag = "ATR Trail"
                elif holding.Quantity < 0 and price >= crypto['trail_stop']:
                    tag = "ATR Trail"

        if not tag and crypto and crypto['rsi'].IsReady:
            rsi_now = crypto['rsi'].Current.Value
            if algo.rsi_peaked_overbought.get(symbol, False) and rsi_now < 75:
                tag = "RSI Momentum Exit"

        if not tag and hours >= 2.0 and crypto and len(crypto['volume']) >= 2:
            entry_vol = algo.entry_volumes.get(symbol, 0)
            if entry_vol > 0:
                v1 = crypto['volume'][-1]
                v2 = crypto['volume'][-2]
                if v1 < entry_vol * 0.50 and v2 < entry_vol * 0.50:
                    tag = "Volume Dry-up"

        if not tag and hours >= algo.time_stop_hours and pnl < algo.time_stop_pnl_min:
            tag = "Time Stop"

        if (not tag
                and hours >= algo.extended_time_stop_hours
                and pnl < algo.extended_time_stop_pnl_max):
            tag = "Extended Time Stop"

        if not tag and hours >= algo.stale_position_hours:
            tag = "Stale Position Exit"

    # ------------------------------------------------------------------
    # Execute exit
    # ------------------------------------------------------------------
    if tag:
        if price * abs(holding.Quantity) < min_notional_usd * 0.9:
            return
        if pnl < 0:
            algo._symbol_loss_cooldowns[symbol] = algo.Time + timedelta(hours=1)
        # MG3: transition to EXITING state before placing sell order
        algo._set_position_state(symbol, POSITION_STATE_EXITING)
        sold = smart_liquidate(algo, symbol, tag)
        if sold:
            algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
            algo.rsi_peaked_overbought.pop(symbol, None)
            algo.entry_volumes.pop(symbol, None)
            algo._choppy_regime_entries.pop(symbol, None)
            # MG3: record PnL by exit tag for post-backtest analysis
            if tag not in algo.mg3_pnl_by_tag:
                algo.mg3_pnl_by_tag[tag] = []
            algo.mg3_pnl_by_tag[tag].append(pnl)
            algo.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
        else:
            # smart_liquidate returned False: position is STILL held.
            # Do NOT call cleanup_position here — that would wipe entry_prices,
            # highest_prices, and entry_times while the position is open, causing
            # the next exit check to lose PnL context and re-track as an orphan.
            # Instead: increment the failure counter and keep all tracking intact.
            # After MAX_EXIT_RETRIES failures the position is escalated to
            # RECOVERING and a force-market order is used.
            fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
            algo._failed_exit_counts[symbol] = fail_count
            if fail_count >= MG3Config.MAX_EXIT_RETRIES:
                algo._set_position_state(symbol, POSITION_STATE_RECOVERING)
                algo.Debug(
                    f"⚠️ EXIT FAILED ({fail_count}×, RECOVERING): {symbol.Value} "
                    f"| PnL:{pnl:+.2%} | Held:{hours:.1f}h — escalating to force-market"
                )
            else:
                algo._set_position_state(symbol, POSITION_STATE_OPEN)
                algo.Debug(
                    f"⚠️ EXIT FAILED ({fail_count}/{MG3Config.MAX_EXIT_RETRIES}): "
                    f"{symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h — will retry"
                )
