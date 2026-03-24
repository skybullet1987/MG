# region imports
from AlgorithmImports import *
from execution import *
from datetime import timedelta
import numpy as np
# endregion


def daily_report_v2(algo):
    daily_report(algo)


def reset_daily_counters(algo):
    algo.daily_trade_count = 0
    algo.last_trade_date = algo.Time.date()
    algo._daily_open_value = algo.Portfolio.TotalPortfolioValue
    for crypto in algo.crypto_data.values():
        crypto['trade_count_today'] = 0
    if len(algo._session_blacklist) > 0:
        algo.Debug(f"Clearing session blacklist ({len(algo._session_blacklist)} items)")
        algo._session_blacklist.clear()
    algo._symbol_entry_cooldowns.clear()
    persist_state(algo)


def on_order_event(algo, event):
    try:
        symbol = event.Symbol
        algo.Debug(f"ORDER: {symbol.Value} {event.Status} {event.Direction} qty={event.FillQuantity or event.Quantity} price={event.FillPrice} id={event.OrderId}")
        if event.Status == OrderStatus.Submitted:
            if symbol not in algo._pending_orders:
                algo._pending_orders[symbol] = 0
            intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
            algo._pending_orders[symbol] += intended_qty
            if symbol not in algo._submitted_orders:
                has_position = symbol in algo.Portfolio and algo.Portfolio[symbol].Invested
                if event.Direction == OrderDirection.Sell and has_position:
                    inferred_intent = 'exit'
                elif event.Direction == OrderDirection.Buy and not has_position:
                    inferred_intent = 'entry'
                else:
                    inferred_intent = 'entry' if event.Direction == OrderDirection.Buy else 'exit'
                algo._submitted_orders[symbol] = {
                    'order_id': event.OrderId,
                    'time': algo.Time,
                    'quantity': event.Quantity,
                    'intent': inferred_intent
                }
            else:
                algo._submitted_orders[symbol]['order_id'] = event.OrderId
        elif event.Status == OrderStatus.PartiallyFilled:
            if symbol in algo._pending_orders:
                algo._pending_orders[symbol] -= abs(event.FillQuantity)
                if algo._pending_orders[symbol] <= 0:
                    algo._pending_orders.pop(symbol, None)
            if event.Direction == OrderDirection.Buy:
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol] = event.FillPrice
                    algo.highest_prices[symbol] = event.FillPrice
                    algo.entry_times[symbol] = algo.Time
            slip_log(algo, symbol, event.Direction, event.FillPrice)
        elif event.Status == OrderStatus.Filled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Buy:
                algo.entry_prices[symbol] = event.FillPrice
                algo.highest_prices[symbol] = event.FillPrice
                algo.entry_times[symbol] = algo.Time
                algo.daily_trade_count += 1
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto['volume']) >= 1:
                    algo.entry_volumes[symbol] = crypto['volume'][-1]
                algo.rsi_peaked_overbought.pop(symbol, None)
            else:
                if symbol in algo._partial_sell_symbols:
                    algo._partial_sell_symbols.discard(symbol)
                else:
                    order = algo.Transactions.GetOrderById(event.OrderId)
                    exit_tag = order.Tag if order and order.Tag else "Unknown"
                    entry = algo.entry_prices.get(symbol, None)
                    if entry is None:
                        entry = event.FillPrice
                        algo.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
                    pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                    algo._rolling_wins.append(1 if pnl > 0 else 0)
                    algo._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                    if pnl > 0:
                        algo._rolling_win_sizes.append(pnl)
                        algo.winning_trades += 1
                        algo.consecutive_losses = 0
                    else:
                        algo._rolling_loss_sizes.append(abs(pnl))
                        algo.losing_trades += 1
                        algo.consecutive_losses += 1
                    algo.total_pnl += pnl
                    if not hasattr(algo, 'pnl_by_tag'):
                        algo.pnl_by_tag = {}
                    if exit_tag not in algo.pnl_by_tag:
                        algo.pnl_by_tag[exit_tag] = []
                    algo.pnl_by_tag[exit_tag].append(pnl)
                    algo.trade_log.append({
                        'time': algo.Time,
                        'symbol': symbol.Value,
                        'pnl_pct': pnl,
                        'exit_reason': exit_tag,
                    })
                    if len(algo._recent_trade_outcomes) >= 12:
                        recent_wr = sum(algo._recent_trade_outcomes) / len(algo._recent_trade_outcomes)
                        if recent_wr < 0.25:
                            algo._cash_mode_until = algo.Time + timedelta(hours=2)
                            algo.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(algo._recent_trade_outcomes)} trades. Pausing 2h.")
                    cleanup_position(algo, symbol)
                    algo._failed_exit_attempts.pop(symbol, None)
                    algo._failed_exit_counts.pop(symbol, None)
            slip_log(algo, symbol, event.Direction, event.FillPrice)
        elif event.Status == OrderStatus.Canceled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Sell and symbol not in algo.entry_prices:
                if is_invested_not_dust(algo, symbol):
                    holding = algo.Portfolio[symbol]
                    algo.entry_prices[symbol] = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol] = algo.Time
                    algo.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
        elif event.Status == OrderStatus.Invalid:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Sell:
                price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
                min_notional = get_min_notional_usd(algo, symbol)
                if price > 0 and symbol in algo.Portfolio and abs(algo.Portfolio[symbol].Quantity) * price < min_notional:
                    algo.Debug(f"DUST CLEANUP on invalid sell: {symbol.Value} — releasing tracking")
                    cleanup_position(algo, symbol)
                    algo._failed_exit_counts.pop(symbol, None)
                else:
                    fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
                    algo._failed_exit_counts[symbol] = fail_count
                    algo.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                    if fail_count >= 3:
                        algo.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} failed exits — releasing tracking")
                        cleanup_position(algo, symbol)
                        algo._failed_exit_counts.pop(symbol, None)
                    elif symbol not in algo.entry_prices:
                        if is_invested_not_dust(algo, symbol):
                            holding = algo.Portfolio[symbol]
                            algo.entry_prices[symbol] = holding.AveragePrice
                            algo.highest_prices[symbol] = holding.AveragePrice
                            algo.entry_times[symbol] = algo.Time
                            algo.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
            algo._session_blacklist.add(symbol.Value)
    except Exception as e:
        algo.Debug(f"OnOrderEvent error: {e}")
    if algo.LiveMode:
        persist_state(algo)


def on_brokerage_message(algo, message):
    try:
        txt = message.Message.lower()
        if "system status:" in txt:
            if "online" in txt:
                algo.kraken_status = "online"
            elif "maintenance" in txt:
                algo.kraken_status = "maintenance"
            elif "cancel_only" in txt:
                algo.kraken_status = "cancel_only"
            elif "post_only" in txt:
                algo.kraken_status = "post_only"
            else:
                algo.kraken_status = "unknown"
            algo.Debug(f"Kraken status update: {algo.kraken_status}")
        if "rate limit" in txt or "too many" in txt:
            algo.Debug(f"⚠️ RATE LIMIT - pausing {algo.rate_limit_cooldown_minutes}min")
            algo._rate_limit_until = algo.Time + timedelta(minutes=algo.rate_limit_cooldown_minutes)
    except Exception as e:
        algo.Debug(f"BrokerageMessage parse error: {e}")


def on_end_of_algorithm(algo):
    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    algo.Debug("=== FINAL ===")
    algo.Debug(f"Trades: {algo.trade_count} | WR: {wr:.1%}")
    algo.Debug(f"Final: ${algo.Portfolio.TotalPortfolioValue:.2f}")
    algo.Debug(f"PnL: {algo.total_pnl:+.2%}")

    if total > 0:
        avg_win = float(np.mean(list(algo._rolling_win_sizes))) if len(algo._rolling_win_sizes) > 0 else 0
        avg_loss = float(np.mean(list(algo._rolling_loss_sizes))) if len(algo._rolling_loss_sizes) > 0 else 0
        algo.Debug("=== REALISM CHECKS ===")
        algo.Debug(f"Win Rate: {wr:.1%} (realistic range: 45-65%)")
        algo.Debug(f"Avg Win: {avg_win:.2%} (should be > round-trip fees 0.65%)")
        algo.Debug(f"Avg Loss: {avg_loss:.2%}")
        if algo.winning_trades > 0 and algo.losing_trades > 0 and avg_loss > 0:
            pf = (avg_win * algo.winning_trades) / (avg_loss * algo.losing_trades)
            algo.Debug(f"Profit Factor: {pf:.2f}")
        if wr > 0.70:
            algo.Debug("⚠️ RED FLAG: Win rate > 70% — likely backtest overfitting")
        if avg_win < 0.005:
            algo.Debug("⚠️ RED FLAG: Avg win < 0.5% — too small to survive live fees")
        try:
            days = max((algo.Time - algo.StartDate).days, 1)
            cagr = (algo.Portfolio.TotalPortfolioValue / 1000) ** (365 / days) - 1
            if cagr > 5.0:
                algo.Debug("⚠️ RED FLAG: CAGR > 500% — backtest is unreliable")
        except Exception:
            pass

    persist_state(algo)
