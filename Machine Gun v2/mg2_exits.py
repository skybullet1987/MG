# region imports
from AlgorithmImports import *
from execution import *
from datetime import timedelta
# endregion


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
            algo.entry_prices[symbol] = kvp.Value.AveragePrice
            algo.highest_prices[symbol] = kvp.Value.AveragePrice
            algo.entry_times[symbol] = algo.Time
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

    actual_qty = abs(holding.Quantity)
    rounded_sell = round_quantity(algo, symbol, actual_qty)
    if rounded_sell > actual_qty:
        algo.Debug(f"DUST (rounded sell > actual): {symbol.Value} | actual={actual_qty} rounded={rounded_sell} — cleaning up")
        cleanup_position(algo, symbol)
        algo._failed_exit_counts.pop(symbol, None)
        return
    if symbol not in algo.entry_prices:
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
    entry = algo.entry_prices[symbol]
    highest = algo.highest_prices.get(symbol, entry)
    if price > highest:
        algo.highest_prices[symbol] = price
    pnl = (price - entry) / entry if entry > 0 else 0

    crypto = algo.crypto_data.get(symbol)
    dd = (highest - price) / highest if highest > 0 else 0
    hours = (algo.Time - algo.entry_times.get(symbol, algo.Time)).total_seconds() / 3600
    minutes = hours * 60

    atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None
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
        sl = max(sl - spread_cost, 0.005)

    if tp < sl * 1.5:
        tp = sl * 1.5

    if algo._choppy_regime_entries.get(symbol, False):
        tp = tp * 0.65

    if algo.volatility_regime == "low":
        tp = tp * 0.75

    trailing_activation = algo.trail_activation
    trailing_stop_pct   = algo.trail_stop_pct

    if crypto and crypto['rsi'].IsReady:
        rsi_now = crypto['rsi'].Current.Value
        if rsi_now > 85:
            algo.rsi_peaked_overbought[symbol] = True

    if (not algo._partial_tp_taken.get(symbol, False)
            and pnl >= algo.partial_tp_threshold):
        if partial_smart_sell(algo, symbol, 0.50, "Partial TP"):
            algo._partial_tp_taken[symbol] = True
            algo._breakeven_stops[symbol] = entry * 1.002
            algo.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL→entry+0.2%")
            return

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
            if pnl > 0.05:
                effective_trail_mult = 1.5
            elif pnl > 0.025:
                effective_trail_mult = 2.5
            else:
                effective_trail_mult = algo.atr_trail_mult
            trail_offset = atr * effective_trail_mult
            trail_level = highest - trail_offset
            if crypto:
                crypto['trail_stop'] = trail_level
            if crypto and crypto['trail_stop'] is not None:
                if minutes >= 15 and holding.Quantity > 0 and price <= crypto['trail_stop']:
                    tag = "ATR Trail"
                elif minutes >= 15 and holding.Quantity < 0 and price >= crypto['trail_stop']:
                    tag = "ATR Trail"

        if not tag and hours >= algo.time_stop_hours and pnl < algo.time_stop_pnl_min:
            tag = "Time Stop"

        if not tag and crypto and crypto['rsi'].IsReady:
            if algo.rsi_peaked_overbought.get(symbol, False) and crypto['rsi'].Current.Value < 70:
                tag = "RSI Momentum Exit"

        if not tag and hours >= 2.0 and crypto and len(crypto['volume']) >= 2:
            entry_vol = algo.entry_volumes.get(symbol, 0)
            if entry_vol > 0:
                v1 = crypto['volume'][-1]
                v2 = crypto['volume'][-2]
                if v1 < entry_vol * 0.50 and v2 < entry_vol * 0.50:
                    tag = "Volume Dry-up"

        if not tag and hours >= algo.extended_time_stop_hours and pnl < algo.extended_time_stop_pnl_max:
            tag = "Extended Time Stop"

        if not tag and hours >= algo.stale_position_hours:
            tag = "Stale Position Exit"

    if tag:
        if price * abs(holding.Quantity) < min_notional_usd * 0.9:
            return
        if pnl < 0:
            algo._symbol_loss_cooldowns[symbol] = algo.Time + timedelta(hours=1)
        sold = smart_liquidate(algo, symbol, tag)
        if sold:
            algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)

            algo.rsi_peaked_overbought.pop(symbol, None)
            algo.entry_volumes.pop(symbol, None)
            algo._choppy_regime_entries.pop(symbol, None)
            algo.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
        else:
            fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
            algo._failed_exit_counts[symbol] = fail_count
            algo.Debug(f"⚠️ EXIT FAILED ({tag}) #{fail_count}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
            if fail_count >= 3:
                algo.Debug(f" FATAL EXIT FAILURE: {symbol.Value} — {fail_count} attempts failed, escalating to market order")
                try:
                    holding = algo.Portfolio[symbol]
                    qty = abs(holding.Quantity)
                    if qty > 0:
                        algo.MarketOrder(symbol, -qty, tag=f"Force Exit (fail#{fail_count})")
                except Exception as e:
                    algo.Debug(f"Force market exit error for {symbol.Value}: {e}")
                algo._failed_exit_counts.pop(symbol, None)
                algo.rsi_peaked_overbought.pop(symbol, None)
                algo.entry_volumes.pop(symbol, None)
                algo._choppy_regime_entries.pop(symbol, None)
