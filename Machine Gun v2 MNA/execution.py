# region imports
from AlgorithmImports import *
import json
import math
import numpy as np
from collections import deque
from datetime import timedelta
# endregion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_invested(algo, symbol):
    """Returns True if we hold any position (long or short) in the symbol."""
    if symbol not in algo.Portfolio:
        return False
    return algo.Portfolio[symbol].Quantity != 0


def has_open_orders(algo, symbol=None):
    if symbol is None:
        return len(algo.Transactions.GetOpenOrders()) > 0
    return len(algo.Transactions.GetOpenOrders(symbol)) > 0


def get_spread_pct(algo, symbol):
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                return (ask - bid) / mid
    except Exception as e:
        algo.Debug(f"Error getting spread for {symbol.Value}: {e}")
    return None


def cleanup_position(algo, symbol):
    """Remove all position-tracking state for a symbol."""
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.lowest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    algo.position_direction.pop(symbol, None)
    if symbol in algo.future_data:
        algo.future_data[symbol]['trail_stop'] = None
    algo._partial_tp_taken.pop(symbol, None)
    algo._breakeven_stops.pop(symbol, None)


def smart_liquidate(algo, symbol, tag="Liquidate"):
    """
    Place a market order to close the full position in the symbol.
    Handles both long (positive quantity) and short (negative quantity).
    Returns True if an order was placed, False otherwise.
    """
    if not is_invested(algo, symbol):
        return False
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return False

    holding_qty = algo.Portfolio[symbol].Quantity
    if holding_qty == 0:
        return False

    algo.Transactions.CancelOpenOrders(symbol)

    # Close the position: sell if long, buy back if short
    close_qty = -holding_qty  # opposite of current holding
    ticket = algo.MarketOrder(symbol, close_qty, tag=tag)
    if ticket is not None:
        if symbol not in algo._submitted_orders:
            algo._submitted_orders[symbol] = {
                'order_id': ticket.OrderId,
                'time': algo.Time,
                'quantity': close_qty,
                'intent': 'exit',
            }
    return ticket is not None


def partial_smart_sell(algo, symbol, fraction, tag="Partial TP"):
    """
    Close a fraction (0.0–1.0) of the current position.
    Works for both long and short positions.
    Returns True if an order was placed.
    """
    if not is_invested(algo, symbol):
        return False
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False

    holding_qty = algo.Portfolio[symbol].Quantity
    if holding_qty == 0:
        return False

    close_qty = int(abs(holding_qty) * fraction) or 1
    if close_qty >= abs(holding_qty):
        return smart_liquidate(algo, symbol, tag)

    direction_mult = -1 if holding_qty > 0 else 1
    if hasattr(algo, '_partial_sell_symbols'):
        algo._partial_sell_symbols.add(symbol)

    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price > 0:
        ticket = algo.LimitOrder(symbol, close_qty * direction_mult, price, tag=tag)
    else:
        ticket = algo.MarketOrder(symbol, close_qty * direction_mult, tag=tag)

    return ticket is not None


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=30, tag="Entry"):
    """
    Place a limit order near mid-price for the entry.
    For futures, uses mid-price limit to try for a better fill.
    Returns the ticket, or None if placement is skipped.
    """
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice

        if bid > 0 and ask > 0:
            if quantity > 0:
                # Buy: place just above bid (still inside spread)
                limit_price = bid * 1.0005
                limit_price = min(limit_price, ask)
            else:
                # Sell: place just below ask
                limit_price = ask * 0.9995
                limit_price = max(limit_price, bid)
        else:
            limit_price = sec.Price
            if limit_price <= 0:
                algo.Debug(f"Price unavailable for {symbol.Value}, using market order")
                return algo.MarketOrder(symbol, quantity, tag=tag)

        ticket = algo.LimitOrder(symbol, quantity, limit_price, tag=tag)

        if hasattr(algo, '_submitted_orders'):
            algo._submitted_orders[symbol] = {
                'order_id': ticket.OrderId,
                'time': algo.Time,
                'quantity': quantity,
                'is_limit_entry': True,
                'timeout_seconds': timeout_seconds,
                'intent': 'entry',
            }

        if algo.LiveMode:
            algo.Debug(f"LIMIT ENTRY: {symbol.Value} | qty={quantity} | limit=${limit_price:.2f} | timeout={timeout_seconds}s")
        return ticket

    except Exception as e:
        algo.Debug(f"Error placing limit order for {symbol.Value}: {e}, falling back to market")
        return algo.MarketOrder(symbol, quantity, tag=tag)


def normalize_order_time(order_time):
    return order_time.replace(tzinfo=None) if order_time.tzinfo is not None else order_time


def record_exit_pnl(algo, symbol, entry_price, exit_price, direction=1, exit_tag="Unknown"):
    """
    Record trade PnL. Direction: 1 = long exit, -1 = short exit.
    Returns the PnL as a decimal fraction.
    """
    if entry_price <= 0 or exit_price <= 0:
        algo.Debug(f"Cannot record PnL for {symbol.Value}: invalid prices")
        return None

    if direction >= 0:
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price

    algo._rolling_wins.append(1 if pnl > 0 else 0)
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
    algo.pnl_by_tag.setdefault(exit_tag, []).append(pnl)

    return pnl


def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds


def cancel_stale_new_orders(algo):
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        timeout_seconds = effective_stale_timeout(algo)
        for order in open_orders:
            order_time = normalize_order_time(order.Time)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                sym_val = order.Symbol.Value
                algo.Debug(f"Canceling stale: {sym_val} (age: {order_age/60:.1f}m)")
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)
    except Exception as e:
        algo.Debug(f"Error in cancel_stale_new_orders: {e}")


def kelly_fraction(algo):
    """Half-Kelly position sizing based on recent win/loss history."""
    if len(algo._rolling_wins) < 10:
        return 1.0
    win_rate = sum(algo._rolling_wins) / len(algo._rolling_wins)
    if win_rate <= 0 or win_rate >= 1:
        return 1.0
    avg_win  = np.mean(list(algo._rolling_win_sizes))  if len(algo._rolling_win_sizes)  > 0 else 0.01
    avg_loss = np.mean(list(algo._rolling_loss_sizes)) if len(algo._rolling_loss_sizes) > 0 else 0.01
    if avg_loss <= 0:
        return 1.0
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    half_kelly = kelly * 0.5
    return max(0.5, min(1.5, half_kelly / 0.5))


def debug_limited(algo, msg):
    if algo.log_budget > 0:
        algo.Debug(msg)
        algo.log_budget -= 1
    elif algo.LiveMode:
        algo.Debug(msg)


def slip_log(algo, symbol, direction, fill_price):
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid <= 0 or ask <= 0:
            return
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return
        side = 1 if direction == OrderDirection.Buy else -1
        slip = side * (fill_price - mid) / mid
        algo._slip_abs.append(abs(slip))
        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except Exception as e:
        algo.Debug(f"Error in slip_log for {symbol.Value}: {e}")


def sync_existing_positions(algo):
    """Re-establish tracking for any positions held at startup."""
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced = 0
    for symbol in algo.Portfolio.Keys:
        if not is_invested(algo, symbol):
            continue
        if symbol in algo.entry_prices:
            continue
        holding = algo.Portfolio[symbol]
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        direction = 1 if holding.Quantity > 0 else -1
        algo.position_direction[symbol] = direction
        if direction == 1:
            algo.highest_prices[symbol] = holding.AveragePrice
        else:
            algo.lowest_prices[symbol] = holding.AveragePrice
        synced += 1
        algo.Debug(f"SYNCED: {symbol.Value} | qty={holding.Quantity} | entry=${holding.AveragePrice:.2f}")
    algo.Debug(f"Synced {synced} positions. Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug("=" * 50)


def persist_state(algo):
    if not algo.LiveMode:
        return
    try:
        state = {
            "winning_trades":    algo.winning_trades,
            "losing_trades":     algo.losing_trades,
            "total_pnl":         algo.total_pnl,
            "consecutive_losses": algo.consecutive_losses,
            "daily_trade_count": algo.daily_trade_count,
            "trade_count":       algo.trade_count,
            "peak_value":        algo.peak_value if algo.peak_value is not None else 0,
        }
        algo.ObjectStore.Save("live_state_mna", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")


def load_persisted_state(algo):
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("live_state_mna"):
            raw  = algo.ObjectStore.Read("live_state_mna")
            data = json.loads(raw)
            algo.winning_trades    = data.get("winning_trades", 0)
            algo.losing_trades     = data.get("losing_trades", 0)
            algo.total_pnl         = data.get("total_pnl", 0.0)
            algo.consecutive_losses = data.get("consecutive_losses", 0)
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count       = data.get("trade_count", 0)
            peak = data.get("peak_value", 0)
            if peak > 0:
                algo.peak_value = peak
            algo.Debug(f"Loaded persisted state: W:{algo.winning_trades}/L:{algo.losing_trades}")
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")


def verify_order_fills(algo):
    """
    Check submitted orders for timeouts. Cancel and retry once before giving up.
    """
    if algo.IsWarmingUp:
        return

    current_time = algo.Time
    symbols_to_remove = []

    for symbol, order_info in list(algo._submitted_orders.items()):
        order_age = (current_time - order_info['time']).total_seconds()
        order_id  = order_info['order_id']

        if order_info.get('is_limit_entry', False):
            timeout = order_info.get('timeout_seconds', 60)
        elif order_info.get('is_limit_exit', False):
            timeout = 90
        else:
            timeout = algo.order_timeout_seconds

        if order_age > timeout:
            retry_count = algo._order_retries.get(order_id, 0)
            if retry_count == 0:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo._retry_pending[symbol] = current_time
                    algo._order_retries[order_id] = 1
                    symbols_to_remove.append(symbol)
                    algo.Debug(f"ORDER TIMEOUT: {symbol.Value} — cancel requested")
                except Exception as e:
                    algo.Debug(f"Error canceling order for {symbol.Value}: {e}")
                    symbols_to_remove.append(symbol)
            else:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    algo.Debug(f"ORDER TIMEOUT (2nd): {symbol.Value} — abandoned")
                except Exception as e:
                    algo.Debug(f"Error on 2nd cancel for {symbol.Value}: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)

    for symbol in symbols_to_remove:
        algo._submitted_orders.pop(symbol, None)


def health_check(algo):
    if algo.IsWarmingUp:
        return
    issues = []

    for symbol in list(algo.entry_prices.keys()):
        if not is_invested(algo, symbol):
            issues.append(f"Orphan tracking: {symbol.Value}")
            cleanup_position(algo, symbol)

    for kvp in algo.Portfolio:
        if is_invested(algo, kvp.Key) and kvp.Key not in algo.entry_prices:
            issues.append(f"Untracked position: {kvp.Key.Value}")

    open_orders = algo.Transactions.GetOpenOrders()
    if len(open_orders) > 0:
        issues.append(f"Open orders: {len(open_orders)}")

    if issues:
        algo.Debug("=== HEALTH CHECK ===")
        for issue in issues:
            algo.Debug(f"  ⚠️ {issue}")
    else:
        debug_limited(algo, "Health check: OK")


def daily_report(algo):
    if algo.IsWarmingUp:
        return
    total = algo.winning_trades + algo.losing_trades
    wr  = algo.winning_trades / total if total > 0 else 0
    avg = algo.total_pnl / total if total > 0 else 0
    algo.Debug(f"=== DAILY {algo.Time.date()} ===")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug(f"Market: {algo.market_regime} {algo.volatility_regime}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    for kvp in algo.Portfolio:
        if is_invested(algo, kvp.Key):
            h = kvp.Value
            price = algo.Securities[kvp.Key].Price if kvp.Key in algo.Securities else h.Price
            entry = algo.entry_prices.get(kvp.Key, h.AveragePrice)
            direction = algo.position_direction.get(kvp.Key, 1)
            if direction >= 0:
                pnl = (price - entry) / entry if entry > 0 else 0
            else:
                pnl = (entry - price) / entry if entry > 0 else 0
            side = "LONG" if direction >= 0 else "SHORT"
            algo.Debug(f"  [{side}] {kvp.Key.Value}: qty={h.Quantity} | entry=${entry:.2f} | now=${price:.2f} | PnL:{pnl:+.2%}")


def live_safety_checks(algo):
    if not algo.LiveMode:
        return True
    if algo.Portfolio.Cash < 100.0:
        debug_limited(algo, "LIVE SAFETY: Cash below $100, pausing new entries")
        return False
    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < 90:
            return False
    return True
