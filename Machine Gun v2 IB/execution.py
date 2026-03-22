# region imports
from AlgorithmImports import *
import json
import math
import numpy as np
from collections import deque
from datetime import timedelta
# endregion

# Fractional haircut applied before re-rounding a sell quantity that exceeds the actual
# portfolio holding.  The 0.9999 factor (0.01% reduction) is enough to push floating-point
# values below the lot_size boundary while keeping the sell amount virtually identical.
QUANTITY_HAIRCUT_FACTOR = 0.9999

# Tolerance multiplier used when comparing a rounded quantity to the actual holding.
QUANTITY_OVERSHOOT_TOLERANCE = 1.0001


class MNQSlippage:
    """
    MNQ slippage: 1-2 ticks in RTH, 2-4 ticks in extended hours.
    MNQ tick size = $0.25, tick value = $0.50 per contract.
    """
    TICK_SIZE = 0.25
    RTH_TICKS = 1
    ETH_TICKS = 3

    def get_slippage_approximation(self, asset, order):
        price = asset.Price
        if price <= 0:
            return 0
        try:
            hour = asset.LocalTime.hour if hasattr(asset, 'LocalTime') else 12
        except Exception:
            hour = 12
        is_rth = 8 <= hour < 15  # Approximate CT
        ticks = self.RTH_TICKS if is_rth else self.ETH_TICKS
        return ticks * self.TICK_SIZE


def track_exit_order(algo, symbol, ticket, quantity):
    """Helper to track exit order in _submitted_orders for verification."""
    if hasattr(algo, '_submitted_orders') and ticket is not None:
        algo._submitted_orders[symbol] = {
            'order_id': ticket.OrderId,
            'time': algo.Time,
            'quantity': quantity,
            'intent': 'exit'
        }


def smart_liquidate(algo, symbol, tag="Liquidate"):
    """
    Liquidate a futures position. Futures trade in whole contracts — no dust,
    no minimum notional, no fee reserve required.
    """
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    # Futures: just market order to close — no dust, no min qty issues
    ticket = algo.MarketOrder(symbol, -holding_qty, tag=tag)
    track_exit_order(algo, symbol, ticket, -holding_qty)
    return True


def partial_smart_sell(algo, symbol, fraction, tag="Partial TP"):
    """
    Sell a fraction (0.0–1.0) of the current futures position.
    For futures, both halves must be at least 1 contract; falls back to full liquidate.
    Returns True if an order was placed successfully, False otherwise.
    """
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    if holding_qty == 0:
        return False
    sell_contracts = int(abs(holding_qty) * fraction)
    remaining_contracts = int(abs(holding_qty)) - sell_contracts
    # Both halves must be at least 1 contract
    if sell_contracts < 1 or remaining_contracts < 1:
        return smart_liquidate(algo, symbol, tag)
    direction_mult = -1 if holding_qty > 0 else 1
    if hasattr(algo, '_partial_sell_symbols'):
        algo._partial_sell_symbols.add(symbol)
    ticket = algo.MarketOrder(symbol, sell_contracts * direction_mult, tag=tag)
    if ticket is not None:
        algo.Debug(f"PARTIAL SELL: {symbol.Value} | frac={fraction:.0%} contracts={sell_contracts} of {abs(holding_qty)}")
    return ticket is not None


def is_invested_not_dust(algo, symbol):
    """For futures: any non-zero position is valid — no dust concept."""
    if symbol not in algo.Portfolio:
        return False
    h = algo.Portfolio[symbol]
    return h.Invested and h.Quantity != 0


def get_actual_position_count(algo):
    return sum(1 for kvp in algo.Portfolio if is_invested_not_dust(algo, kvp.Key))


def has_open_orders(algo, symbol=None):
    if symbol is None:
        return len(algo.Transactions.GetOpenOrders()) > 0
    return len(algo.Transactions.GetOpenOrders(symbol)) > 0


def has_non_stale_open_orders(algo, symbol):
    """Check if symbol has open orders that are NOT stale (younger than timeout)."""
    try:
        orders = algo.Transactions.GetOpenOrders(symbol)
        if len(orders) == 0:
            return False
        timeout_seconds = effective_stale_timeout(algo)
        for order in orders:
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age <= timeout_seconds:
                return True
        return False
    except Exception:
        return False


def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds


def cancel_stale_new_orders(algo):
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        timeout_seconds = effective_stale_timeout(algo)
        for order in open_orders:
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                algo.Debug(f"Canceling stale: {order.Symbol.Value} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")

                if is_invested_not_dust(algo, order.Symbol):
                    algo.Debug(f"STALE ORDER but position exists: {order.Symbol.Value} — re-tracking")
                    holding = algo.Portfolio[order.Symbol]
                    algo.entry_prices[order.Symbol] = holding.AveragePrice
                    algo.highest_prices[order.Symbol] = holding.AveragePrice
                    algo.entry_times[order.Symbol] = algo.Time
                    algo.Transactions.CancelOrder(order.Id)
                    continue

                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)

                has_position_or_tracked = order.Symbol in algo.entry_prices or (
                    order.Symbol in algo.Portfolio and algo.Portfolio[order.Symbol].Quantity != 0
                )

                if has_position_or_tracked:
                    algo.Debug(f"STALE EXIT: {order.Symbol.Value} - cooldown only, not blacklisted")
                else:
                    algo._symbol_entry_cooldowns[order.Symbol.Value] = algo.Time + timedelta(minutes=15)
                    algo.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {order.Symbol.Value} - entry cooldown 15min")
    except Exception as e:
        algo.Debug(f"Error in cancel_stale_new_orders: {e}")


def cleanup_position(algo, symbol, record_pnl=False, exit_price=None):
    """
    Clean up position tracking for a symbol.
    If record_pnl=True, records the PnL before cleanup using record_exit_pnl helper.
    """
    entry_price = algo.entry_prices.get(symbol, None)
    if record_pnl and entry_price is not None and entry_price > 0:
        if exit_price is None:
            try:
                exit_price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
            except Exception:
                exit_price = 0
        if exit_price > 0:
            record_exit_pnl(algo, symbol, entry_price, exit_price)
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    # Clear trail_stop on mnq_data if this is the active contract
    if hasattr(algo, 'mnq_data') and algo.mnq_data is not None:
        algo.mnq_data['trail_stop'] = None
    if hasattr(algo, '_spike_entries'):
        algo._spike_entries.pop(symbol, None)
    if hasattr(algo, '_partial_tp_taken'):
        algo._partial_tp_taken.pop(symbol, None)
    if hasattr(algo, '_breakeven_stops'):
        algo._breakeven_stops.pop(symbol, None)


def sync_existing_positions(algo):
    """Sync existing futures positions on startup (no AddCrypto needed — futures already added)."""
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    positions_to_close = []
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        holding = algo.Portfolio[symbol]
        ticker = symbol.Value
        if symbol in algo.entry_prices:
            continue
        if symbol not in algo.Securities:
            algo.Debug(f"RESYNC: {ticker} not in Securities — skipping (futures may not be active contract)")
            continue
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        algo.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.2f} | Now: ${current_price:.2f} | PnL: {pnl_pct:+.2%}")
        if current_price > holding.AveragePrice:
            algo.highest_prices[symbol] = current_price
        if pnl_pct >= algo.base_take_profit:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync TP"))
        elif pnl_pct <= -algo.base_stop_loss:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync SL"))
    algo.Debug(f"Synced {synced_count} futures positions")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f}")
    algo.Debug("=" * 50)
    for symbol, ticker, pnl_pct, reason in positions_to_close:
        algo.Debug(f"IMMEDIATE {reason}: {ticker} at {pnl_pct:+.2%}")
        sold = smart_liquidate(algo, symbol, reason)
        if not sold:
            algo.Debug(f"⚠️ IMMEDIATE {reason} FAILED: {ticker} — cleaning up tracking")
            cleanup_position(algo, symbol)


def debug_limited(algo, msg):
    if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
        algo.Debug(msg)
        return
    if algo.log_budget > 0:
        algo.Debug(msg)
        algo.log_budget -= 1
    elif algo.LiveMode:
        algo.Debug(msg)


def slip_log(algo, symbol, direction, fill_price):
    """Enhanced slip_log with live outlier alert and symbol-level slippage tracking."""
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
        abs_slip = abs(slip)
        algo._slip_abs.append(abs_slip)

        if hasattr(algo, '_symbol_slippage_history'):
            ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
            if ticker not in algo._symbol_slippage_history:
                algo._symbol_slippage_history[ticker] = deque(maxlen=30)
            algo._symbol_slippage_history[ticker].append((algo.Time, abs_slip))

        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except Exception as e:
        algo.Debug(f"Error in slip_log for {symbol.Value}: {e}")


def persist_state(algo):
    """Persist key state to object store for live trading recovery."""
    if not algo.LiveMode:
        return
    try:
        state = {
            "winning_trades": algo.winning_trades,
            "losing_trades": algo.losing_trades,
            "total_pnl": algo.total_pnl,
            "consecutive_losses": algo.consecutive_losses,
            "daily_trade_count": algo.daily_trade_count,
            "trade_count": algo.trade_count,
            "peak_value": algo.peak_value if algo.peak_value is not None else 0,
        }
        algo.ObjectStore.Save("live_state", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")


def load_persisted_state(algo):
    """Load persisted state from object store on live trading startup."""
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("live_state"):
            raw = algo.ObjectStore.Read("live_state")
            data = json.loads(raw)
            algo.winning_trades = data.get("winning_trades", 0)
            algo.losing_trades = data.get("losing_trades", 0)
            algo.total_pnl = data.get("total_pnl", 0.0)
            algo.consecutive_losses = data.get("consecutive_losses", 0)
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count = data.get("trade_count", 0)
            peak = data.get("peak_value", 0)
            if peak > 0:
                algo.peak_value = peak
            algo.Debug(f"Loaded persisted state: trades W:{algo.winning_trades}/L:{algo.losing_trades}")
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")


def cleanup_object_store(algo):
    """Clean up stale object store keys."""
    try:
        n = 0
        for i in algo.ObjectStore.GetEnumerator():
            k = i.Key if hasattr(i, 'Key') else str(i)
            if k != "live_state":
                try:
                    algo.ObjectStore.Delete(k)
                    n += 1
                except Exception as e:
                    algo.Debug(f"Error deleting key {k}: {e}")
        if n:
            algo.Debug(f"Cleaned {n} keys")
    except Exception as e:
        algo.Debug(f"Cleanup err: {e}")


def live_safety_checks(algo):
    """Extra safety checks for live trading."""
    if not algo.LiveMode:
        return True

    try:
        cash = algo.Portfolio.Cash
    except Exception:
        cash = 0

    if cash < 500:
        debug_limited(algo, "LIVE SAFETY: Cash below $500, pausing new entries")
        return False

    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < 90:
            return False

    return True


def kelly_fraction(algo):
    """Kelly criterion-based position sizing."""
    if len(algo._rolling_wins) < 10:
        return 1.0
    win_rate = sum(algo._rolling_wins) / len(algo._rolling_wins)
    if win_rate <= 0 or win_rate >= 1:
        return 1.0
    avg_win = np.mean(list(algo._rolling_win_sizes)) if len(algo._rolling_win_sizes) > 0 else 0.02
    avg_loss = np.mean(list(algo._rolling_loss_sizes)) if len(algo._rolling_loss_sizes) > 0 else 0.02
    if avg_loss <= 0:
        return 1.0
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    half_kelly = kelly * 0.5
    return max(0.5, min(1.5, half_kelly / 0.5))


def verify_order_fills(algo):
    """
    Verify submitted orders filled/timed out. Retry once before giving up.
    """
    if algo.IsWarmingUp:
        return

    current_time = algo.Time
    symbols_to_remove = []

    for symbol, order_info in list(algo._submitted_orders.items()):
        order_age_seconds = (current_time - order_info['time']).total_seconds()
        order_id = order_info['order_id']
        timeout = algo.order_timeout_seconds
        intent = order_info.get('intent', 'entry')

        if order_age_seconds > algo.order_fill_check_threshold_seconds:
            try:
                order = algo.Transactions.GetOrderById(order_id)
                if order is not None and order.Status == OrderStatus.Filled:
                    if intent == 'exit':
                        entry = algo.entry_prices.get(symbol, None)
                        if entry:
                            current_price = algo.Securities[symbol].Price if symbol in algo.Securities else None
                            if current_price is not None and current_price > 0:
                                pnl = record_exit_pnl(algo, symbol, entry, current_price)
                                if pnl is not None:
                                    algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} | PnL: {pnl:+.2%}")
                            cleanup_position(algo, symbol)
                        symbols_to_remove.append(symbol)
                        algo._order_retries.pop(order_id, None)
                        continue
                    else:
                        if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                            holding = algo.Portfolio[symbol]
                            entry_price = holding.AveragePrice
                            if symbol not in algo.entry_prices:
                                algo.entry_prices[symbol] = entry_price
                                algo.highest_prices[symbol] = entry_price
                                algo.entry_times[symbol] = order_info['time']
                                algo.daily_trade_count += 1
                                algo.Debug(f"FILL VERIFIED (missed event): {symbol.Value} | Entry: ${entry_price:.2f}")
                        symbols_to_remove.append(symbol)
                        algo._order_retries.pop(order_id, None)
                        continue
            except Exception as e:
                algo.Debug(f"Error checking order status for {symbol.Value}: {e}")

            if intent == 'exit':
                holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
                if holding is None or not holding.Invested or holding.Quantity == 0:
                    entry = algo.entry_prices.get(symbol, None)
                    if entry:
                        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else None
                        if current_price is not None and current_price > 0:
                            pnl = record_exit_pnl(algo, symbol, entry, current_price)
                            if pnl is not None:
                                algo.Debug(f"⚠️ PHANTOM EXIT DETECTED: {symbol.Value} | PnL: {pnl:+.2%}")
                        cleanup_position(algo, symbol)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    continue

        if order_age_seconds > timeout:
            retry_count = algo._order_retries.get(order_id, 0)
            if retry_count == 0:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo._retry_pending[symbol] = current_time
                    algo._order_retries[order_id] = 1
                    symbols_to_remove.append(symbol)
                    algo.Debug(f"ORDER TIMEOUT: {symbol.Value} - cancel requested")
                except Exception as e:
                    algo.Debug(f"Error canceling order for {symbol.Value}: {e}")
                    symbols_to_remove.append(symbol)
            else:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    algo.Debug(f"ORDER TIMEOUT (attempt 2): {symbol.Value}")
                except Exception as e:
                    algo.Debug(f"Error canceling order {order_id} on second timeout: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)

    for symbol in symbols_to_remove:
        algo._submitted_orders.pop(symbol, None)


def health_check(algo):
    """Health check: orphan detection, stale orders, phantom positions."""
    if algo.IsWarmingUp:
        return

    resync_holdings_full(algo)

    issues = []
    if algo.Portfolio.Cash < 500:
        issues.append(f"Low cash: ${algo.Portfolio.Cash:.2f}")

    for symbol in list(algo.entry_prices.keys()):
        open_orders = algo.Transactions.GetOpenOrders(symbol)
        if len(open_orders) > 0:
            all_stale = True
            for o in open_orders:
                order_time = normalize_order_time(o.Time)
                if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                    all_stale = False
                    break
            if not all_stale:
                continue
            for o in open_orders:
                try:
                    algo.Transactions.CancelOrder(o.Id)
                    issues.append(f"Canceled stale order: {symbol.Value}")
                except Exception as e:
                    algo.Debug(f"Error canceling stale order for {symbol.Value}: {e}")

        if not is_invested_not_dust(algo, symbol):
            issues.append(f"Orphan tracking: {symbol.Value}")
            cleanup_position(algo, symbol)

    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key) and kvp.Key not in algo.entry_prices:
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


def resync_holdings_full(algo):
    """
    Live-only safety: backfill tracking for any holdings not registered via OnOrderEvent.
    Adapted for futures: no AddCrypto calls, just track by Symbol directly.
    """
    if algo.IsWarmingUp:
        return
    if not algo.LiveMode:
        return

    if not hasattr(algo, '_last_resync_log') or (algo.Time - algo._last_resync_log).total_seconds() > algo.resync_log_interval_seconds:
        algo.Debug(f"RESYNC CHECK: tracked={len(algo.entry_prices)}")
        algo._last_resync_log = algo.Time

    # Forward resync: find holdings we're not tracking
    missing = []
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        if symbol in algo.entry_prices:
            continue
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        if has_non_stale_open_orders(algo, symbol):
            continue
        missing.append(symbol)

    if missing:
        algo.Debug(f"RESYNC: detected {len(missing)} holdings without tracking; backfilling.")
        for symbol in missing:
            try:
                holding = algo.Portfolio[symbol]
                entry = holding.AveragePrice
                algo.entry_prices[symbol] = entry
                algo.highest_prices[symbol] = entry
                algo.entry_times[symbol] = algo.Time
                current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                pnl_pct = (current_price - entry) / entry if entry > 0 else 0
                algo.Debug(f"RESYNCED: {symbol.Value} | Qty: {holding.Quantity} | Entry: ${entry:.2f} | Now: ${current_price:.2f} | PnL: {pnl_pct:+.2%}")
            except Exception as e:
                algo.Debug(f"Resync error {symbol.Value}: {e}")

    # Reverse resync: detect phantom positions
    phantoms = []
    for symbol in list(algo.entry_prices.keys()):
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
        if holding is None or not holding.Invested or holding.Quantity == 0:
            open_orders = algo.Transactions.GetOpenOrders(symbol)
            if len(open_orders) > 0:
                all_stuck = True
                for o in open_orders:
                    order_time = normalize_order_time(o.Time)
                    if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                        all_stuck = False
                        break
                if not all_stuck:
                    continue
                for o in open_orders:
                    try:
                        algo.Transactions.CancelOrder(o.Id)
                    except Exception as e:
                        algo.Debug(f"Error canceling stuck order for {symbol.Value}: {e}")
            phantoms.append(symbol)

    if phantoms:
        algo.Debug(f"⚠️ REVERSE RESYNC: detected {len(phantoms)} phantom positions")
        for symbol in phantoms:
            algo.Debug(f"⚠️ PHANTOM POSITION: {symbol.Value} — tracked but broker qty=0, cleaning up")
            cleanup_position(algo, symbol, record_pnl=True)
            if hasattr(algo, 'exit_cooldown_hours') and hasattr(algo, '_exit_cooldowns'):
                algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
        persist_state(algo)


def normalize_order_time(order_time):
    """Helper to normalize order time by removing timezone info if present."""
    return order_time.replace(tzinfo=None) if order_time.tzinfo is not None else order_time


def record_exit_pnl(algo, symbol, entry_price, exit_price, exit_tag="Unknown"):
    """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
    if entry_price <= 0 or exit_price <= 0:
        algo.Debug(f"⚠️ Cannot record PnL for {symbol.Value}: invalid prices (entry=${entry_price:.2f}, exit=${exit_price:.2f})")
        return None

    pnl = (exit_price - entry_price) / entry_price
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
    if exit_tag not in algo.pnl_by_tag:
        algo.pnl_by_tag[exit_tag] = []
    algo.pnl_by_tag[exit_tag].append(pnl)
    return pnl


def get_open_buy_orders_value(algo):
    """Calculate total value reserved by open buy orders."""
    total_reserved = 0
    for o in algo.Transactions.GetOpenOrders():
        if o.Direction == OrderDirection.Buy:
            if o.Price > 0:
                order_price = o.Price
            elif o.Symbol in algo.Securities:
                order_price = algo.Securities[o.Symbol].Price
                if order_price <= 0:
                    continue
            else:
                continue
            total_reserved += abs(o.Quantity) * order_price
    return total_reserved


def portfolio_sanity_check(algo):
    """
    Check for portfolio value mismatches. Simplified for futures (single instrument).
    """
    if algo.IsWarmingUp:
        return

    total_qc = algo.Portfolio.TotalPortfolioValue
    cash = algo.Portfolio.Cash

    tracked_value = 0.0
    tracked_positions = {}

    for sym in list(algo.entry_prices.keys()):
        if sym in algo.Securities:
            price = algo.Securities[sym].Price
            if sym in algo.Portfolio:
                qty = algo.Portfolio[sym].Quantity
                value = abs(qty) * price
                tracked_value += value
                tracked_positions[sym.Value] = {'qty': qty, 'price': price, 'value': value}

    expected = cash + tracked_value
    abs_diff = abs(total_qc - expected)

    if total_qc > 1.0:
        pct_diff = abs_diff / total_qc
        should_warn = pct_diff > algo.portfolio_mismatch_threshold and abs_diff > algo.portfolio_mismatch_min_dollars
        if should_warn:
            if algo._last_mismatch_warning is None or (algo.Time - algo._last_mismatch_warning).total_seconds() >= algo.portfolio_mismatch_cooldown_seconds:
                algo.Debug(f"⚠️ PORTFOLIO MISMATCH: QC total=${total_qc:.2f} expected=${expected:.2f} (diff=${abs_diff:.2f}, {pct_diff:.2%})")
                resync_holdings_full(algo)
                algo._last_mismatch_warning = algo.Time


def review_performance(algo):
    """Review recent performance and adjust max_positions accordingly."""
    if algo.IsWarmingUp or len(algo.trade_log) < 10:
        return

    recent_trades = algo.trade_log[-15:] if len(algo.trade_log) >= 15 else algo.trade_log
    if len(recent_trades) == 0:
        return

    recent_win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
    recent_avg_pnl = np.mean([t['pnl_pct'] for t in recent_trades])
    old_max = algo.max_positions

    if recent_win_rate < 0.2 and recent_avg_pnl < -0.05:
        algo.max_positions = 1
        if old_max != 1:
            algo.Debug(f"PERFORMANCE DECAY: max_pos=1 (WR:{recent_win_rate:.0%}, PnL:{recent_avg_pnl:+.2%})")
    elif recent_win_rate > 0.35 and recent_avg_pnl > -0.01:
        algo.max_positions = algo.base_max_positions
        if old_max != algo.base_max_positions:
            algo.Debug(f"PERFORMANCE RECOVERY: max_pos={algo.base_max_positions}")


def daily_report(algo):
    """Generate daily report with portfolio status and position details."""
    if algo.IsWarmingUp:
        return

    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    avg = algo.total_pnl / total if total > 0 else 0
    algo.Debug(f"=== DAILY {algo.Time.date()} ===")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug(f"Pos: {get_actual_position_count(algo)}/{algo.base_max_positions} | {algo.market_regime} {algo.volatility_regime}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    algo.Debug(f"VIX: {getattr(algo, 'vix_value', 20.0):.1f}")
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key):
            s = kvp.Key
            entry = algo.entry_prices.get(s, kvp.Value.AveragePrice)
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            pnl = (cur - entry) / entry if entry > 0 else 0
            qty = kvp.Value.Quantity
            algo.Debug(f"  {s.Value}: qty={qty} ${entry:.2f}→${cur:.2f} ({pnl:+.2%})")
    persist_state(algo)
