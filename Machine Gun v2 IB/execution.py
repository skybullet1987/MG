# region imports
from AlgorithmImports import *
import json
import math
import numpy as np
from collections import deque
from datetime import timedelta
from scoring import ALL_SETUP_TYPES
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
    if not tag:
        tag = "Exit"
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
        # Collect bracket order IDs so we never cancel a live TP or SL order
        bracket_ids = set()
        if hasattr(algo, '_bracket_orders'):
            for b in algo._bracket_orders.values():
                bracket_ids.add(b.get('tp_id'))
                bracket_ids.add(b.get('sl_id'))
        for order in open_orders:
            # Never cancel bracket (TP/SL) legs — they are intentionally long-lived
            if order.Id in bracket_ids:
                continue
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                algo.Debug(f"Canceling stale: {order.Symbol.Value} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)

                has_position = is_invested_not_dust(algo, order.Symbol)
                if has_position:
                    algo.Debug(f"STALE EXIT: {order.Symbol.Value} - cooldown only, not blacklisted")
                else:
                    algo._symbol_entry_cooldowns[order.Symbol.Value] = algo.Time + timedelta(minutes=15)
                    algo.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {order.Symbol.Value} - entry cooldown 15min")
    except Exception as e:
        algo.Debug(f"Error in cancel_stale_new_orders: {e}")


def cleanup_position(algo, symbol, record_pnl=False, exit_price=None):
    """
    Clean up bracket order tracking for a symbol after a position is fully closed.
    """
    # Cancel any lingering bracket legs for this symbol
    bracket = algo._bracket_orders.pop(symbol, None) if hasattr(algo, '_bracket_orders') else None
    if bracket is not None:
        for order_id in (bracket.get('tp_id'), bracket.get('sl_id')):
            if order_id is not None:
                try:
                    algo.Transactions.CancelOrder(order_id)
                except Exception:
                    pass
    # Clear trail_stop on instrument_data for this contract
    if hasattr(algo, 'instrument_data') and symbol in algo.instrument_data:
        algo.instrument_data[symbol]['trail_stop'] = None


def sync_existing_positions(algo):
    """Sync existing futures positions on startup. Any open positions have bracket orders
    submitted immediately so exits are handled natively."""
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        holding = algo.Portfolio[symbol]
        ticker = symbol.Value
        if symbol not in algo.Securities:
            algo.Debug(f"RESYNC: {ticker} not in Securities — skipping (futures may not be active contract)")
            continue
        if symbol in algo._bracket_orders:
            continue
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        is_short = holding.Quantity < 0
        if is_short:
            pnl_pct = (holding.AveragePrice - current_price) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        else:
            pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        algo.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.2f} | Now: ${current_price:.2f} | PnL: {pnl_pct:+.2%}")
        # Re-establish bracket orders for the existing position
        if pnl_pct >= algo.base_take_profit:
            algo.Debug(f"IMMEDIATE TP (Sync): {ticker} at {pnl_pct:+.2%}")
            smart_liquidate(algo, symbol, "Sync TP")
        elif pnl_pct <= -algo.base_stop_loss:
            algo.Debug(f"IMMEDIATE SL (Sync): {ticker} at {pnl_pct:+.2%}")
            smart_liquidate(algo, symbol, "Sync SL")
        elif hasattr(algo, '_submit_bracket_orders'):
            direction = -1 if is_short else 1
            contracts = abs(int(holding.Quantity))
            algo._submit_bracket_orders(symbol, holding.AveragePrice, direction, contracts)
    algo.Debug(f"Synced {synced_count} futures positions")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f}")
    algo.Debug("=" * 50)


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
            "daily_trade_count": algo.daily_trade_count,
            "trade_count": algo.trade_count,
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
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count = data.get("trade_count", 0)
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

        if order_age_seconds > algo.order_fill_check_threshold_seconds:
            try:
                order = algo.Transactions.GetOrderById(order_id)
                if order is not None and order.Status == OrderStatus.Filled:
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    continue
            except Exception as e:
                algo.Debug(f"Error checking order status for {symbol.Value}: {e}")

            # Check if position is already gone
            holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
            if holding is None or not holding.Invested or holding.Quantity == 0:
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

    # Check for any stale bracket orders whose position has been closed
    bracket_orders = getattr(algo, '_bracket_orders', {})
    for symbol, bracket in list(bracket_orders.items()):
        if not is_invested_not_dust(algo, symbol):
            issues.append(f"Orphan bracket: {symbol.Value} — position closed, cleaning brackets")
            cleanup_position(algo, symbol)

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
    Live-only safety: detect any holdings not covered by bracket orders and
    re-establish brackets so they have native exits.
    """
    if algo.IsWarmingUp:
        return
    if not algo.LiveMode:
        return

    if not hasattr(algo, '_last_resync_log') or (algo.Time - algo._last_resync_log).total_seconds() > algo.resync_log_interval_seconds:
        bracket_count = len(getattr(algo, '_bracket_orders', {}))
        algo.Debug(f"RESYNC CHECK: brackets={bracket_count}")
        algo._last_resync_log = algo.Time

    bracket_orders = getattr(algo, '_bracket_orders', {})

    # Forward resync: find holdings without bracket coverage
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        if symbol in bracket_orders:
            continue
        if symbol in algo._submitted_orders:
            continue
        if has_non_stale_open_orders(algo, symbol):
            continue
        try:
            holding = algo.Portfolio[symbol]
            is_short = holding.Quantity < 0
            direction = -1 if is_short else 1
            contracts = abs(int(holding.Quantity))
            current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
            algo.Debug(f"RESYNCED: {symbol.Value} | Qty: {holding.Quantity} | Entry: ${holding.AveragePrice:.2f}")
            if hasattr(algo, '_submit_bracket_orders'):
                algo._submit_bracket_orders(symbol, holding.AveragePrice, direction, contracts)
        except Exception as e:
            algo.Debug(f"Resync error {symbol.Value}: {e}")

    # Reverse resync: detect phantom bracket tracking (bracket exists but no position)
    for symbol in list(bracket_orders.keys()):
        if symbol in algo._submitted_orders:
            continue
        holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
        if holding is None or not holding.Invested or holding.Quantity == 0:
            algo.Debug(f"⚠️ PHANTOM BRACKET: {symbol.Value} — tracked but broker qty=0, cleaning up")
            cleanup_position(algo, symbol)
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
    else:
        algo._rolling_loss_sizes.append(abs(pnl))
        algo.losing_trades += 1
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

    for kvp in algo.Portfolio:
        sym = kvp.Key
        if not is_invested_not_dust(algo, sym):
            continue
        if sym in algo.Securities:
            price = algo.Securities[sym].Price
            qty = kvp.Value.Quantity
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
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            entry = kvp.Value.AveragePrice
            pnl = (cur - entry) / entry if entry > 0 else 0
            qty = kvp.Value.Quantity
            algo.Debug(f"  {s.Value}: qty={qty} ${entry:.2f}→${cur:.2f} ({pnl:+.2%})")
    persist_state(algo)


# =============================================================================
# Diagnostics and trade attribution (DiagnosticsLogger)
# =============================================================================

import numpy as np
from collections import defaultdict


class DiagnosticsLogger:
    """
    Central diagnostics/attribution logger for Machine Gun v2 IB.

    Designed to be stateless with respect to the rest of the algorithm:
    it only stores data, never makes trading decisions.
    """

    def __init__(self, algo):
        self.algo = algo

        # Enhanced trade log: one record per closed trade
        self.trade_log = []

        # Candidate pipeline counters
        self.candidates_generated = 0
        self.candidates_accepted  = 0
        self.candidates_rejected  = 0

        self.rejection_counts    = defaultdict(int)   # reason  → count
        self.generation_by_setup = defaultdict(int)   # setup   → count
        self.accepted_by_setup   = defaultdict(int)   # setup   → count
        self.rejected_by_setup   = defaultdict(int)   # setup   → count

        # In-flight entries: entry_order_id → metadata dict
        self._pending_meta = {}
        # Symbol fallback (in case we need to look up by symbol instead of order id)
        self._symbol_meta  = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def record_candidate_generated(self, candidate):
        """Call whenever a SetupCandidate is produced by a setup evaluator."""
        self.candidates_generated += 1
        self.generation_by_setup[candidate.setup_type] += 1

    def record_candidate_accepted(self, candidate, order_id):
        """Call when a candidate passes all filters and a market order is placed."""
        self.candidates_accepted += 1
        self.accepted_by_setup[candidate.setup_type] += 1

        meta = {
            "setup_type":  candidate.setup_type,
            "direction":   candidate.direction,
            "score":       candidate.score,
            "threshold":   candidate.threshold,
            "long_score":  candidate.long_score,
            "short_score": candidate.short_score,
            "components":  dict(candidate.components),
            "regime":      candidate.regime,
            "session":     candidate.session,
            "vix":         candidate.vix,
            "symbol_name": candidate.symbol_name,
            "entry_time":  self.algo.Time,
            "entry_price": None,   # filled in by note_entry_filled()
        }
        self._pending_meta[order_id]               = meta
        self._symbol_meta[candidate.symbol_name]   = meta

    def record_candidate_rejected(self, candidate, reason):
        """Call when a candidate is generated but not turned into an order."""
        self.candidates_rejected += 1
        self.rejection_counts[reason] += 1
        self.rejected_by_setup[candidate.setup_type] += 1

    def record_rejection(self, reason):
        """Record a global rejection not tied to a specific candidate."""
        self.rejection_counts[reason] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Entry fill notification
    # ─────────────────────────────────────────────────────────────────────────

    def note_entry_filled(self, entry_order_id, symbol, fill_price):
        """
        Called from OnOrderEvent when an entry order fills.
        Stores the actual fill price so hold-time and PnL are accurate.
        """
        meta = self._pending_meta.get(entry_order_id)
        symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        if meta is None:
            meta = self._symbol_meta.get(symbol_name)
        if meta is not None:
            meta["entry_price"] = fill_price
            meta["entry_time"]  = self.algo.Time
            # Also store by symbol so bracket fills can find it
            self._symbol_meta[symbol_name] = meta

    # ─────────────────────────────────────────────────────────────────────────
    # Exit recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_exit(self, symbol, entry_order_id, entry_price, exit_price,
                    exit_reason, entry_time=None):
        """
        Record a completed trade with full attribution.

        Looks up entry metadata by order_id first, then by symbol name as a
        fallback.  Always removes the pending metadata entry to prevent stale data.
        """
        symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)

        # Try to find metadata in priority order
        meta = self._pending_meta.pop(entry_order_id, None)
        if meta is None:
            meta = self._symbol_meta.pop(symbol_name, None)
        if meta is None:
            meta = {}

        direction = meta.get("direction", 1)

        if entry_price > 0 and exit_price > 0:
            pnl_pct = (
                (exit_price - entry_price) / entry_price if direction == 1
                else (entry_price - exit_price) / entry_price
            )
        else:
            pnl_pct = 0.0

        # Hold time
        t_entry = meta.get("entry_time") or entry_time
        hold_hours = None
        if t_entry is not None:
            try:
                hold_secs = (self.algo.Time - t_entry).total_seconds()
                hold_hours = hold_secs / 3600.0
            except Exception:
                pass

        record = {
            "time":        self.algo.Time,
            "symbol":      symbol_name,
            "setup_type":  meta.get("setup_type",  "unknown"),
            "direction":   direction,
            "score":       meta.get("score",       0.0),
            "threshold":   meta.get("threshold",   0.0),
            "long_score":  meta.get("long_score",  0.0),
            "short_score": meta.get("short_score", 0.0),
            "components":  meta.get("components",  {}),
            "regime":      meta.get("regime",      "unknown"),
            "session":     meta.get("session",     "unknown"),
            "vix":         meta.get("vix",         20.0),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "exit_reason": exit_reason,
            "pnl_pct":     pnl_pct,
            "hold_hours":  hold_hours,
            "entry_time":  t_entry,
        }
        self.trade_log.append(record)
        return record

    # ─────────────────────────────────────────────────────────────────────────
    # Summary reports
    # ─────────────────────────────────────────────────────────────────────────

    def print_summary(self):
        """
        Print a comprehensive attribution summary to the algorithm's Debug log.
        Covers the full backtest run.
        """
        algo   = self.algo
        trades = self.trade_log
        total  = len(trades)

        algo.Debug("=" * 64)
        algo.Debug("=== MACHINE GUN v2 IB  — DIAGNOSTICS SUMMARY ===")
        algo.Debug(
            f"Candidates: generated={self.candidates_generated} "
            f"accepted={self.candidates_accepted} "
            f"rejected={self.candidates_rejected}"
        )
        algo.Debug(f"Closed trades: {total}")

        if total == 0:
            algo.Debug("No closed trades — nothing to attribute.")
            algo.Debug("=" * 64)
            return

        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr   = wins / total
        avg  = np.mean([t["pnl_pct"] for t in trades])
        algo.Debug(f"Overall: WR={wr:.1%}  Avg={avg:+.4%}")

        # ── By setup type ─────────────────────────────────────────────────────
        algo.Debug("--- By Setup ---")
        for st in ALL_SETUP_TYPES:
            st_trades = [t for t in trades if t["setup_type"] == st]
            gen = self.generation_by_setup.get(st, 0)
            acc = self.accepted_by_setup.get(st, 0)
            if not st_trades:
                algo.Debug(f"  {st}: 0 closed trades  (gen={gen} acc={acc})")
                continue
            st_wins = sum(1 for t in st_trades if t["pnl_pct"] > 0)
            st_wr   = st_wins / len(st_trades)
            st_avg  = np.mean([t["pnl_pct"] for t in st_trades])
            algo.Debug(
                f"  {st}: {len(st_trades)} trades | WR={st_wr:.1%} | "
                f"Avg={st_avg:+.4%} | gen={gen} acc={acc}"
            )

        # ── By symbol ─────────────────────────────────────────────────────────
        algo.Debug("--- By Symbol ---")
        for sym in sorted(set(t["symbol"] for t in trades)):
            st = [t for t in trades if t["symbol"] == sym]
            sw = sum(1 for t in st if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {sym}: {len(st)} trades | WR={sw/len(st):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in st]):+.4%}"
            )

        # ── By exit reason ────────────────────────────────────────────────────
        algo.Debug("--- By Exit Reason ---")
        for reason in sorted(set(t["exit_reason"] for t in trades)):
            rt = [t for t in trades if t["exit_reason"] == reason]
            rw = sum(1 for t in rt if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {reason}: {len(rt)} trades | WR={rw/len(rt):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in rt]):+.4%}"
            )

        # ── By regime ─────────────────────────────────────────────────────────
        algo.Debug("--- By Regime ---")
        for regime in sorted(set(t["regime"] for t in trades)):
            rt = [t for t in trades if t["regime"] == regime]
            rw = sum(1 for t in rt if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {regime}: {len(rt)} trades | WR={rw/len(rt):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in rt]):+.4%}"
            )

        # ── By session ────────────────────────────────────────────────────────
        algo.Debug("--- By Session ---")
        for session in sorted(set(t["session"] for t in trades)):
            st = [t for t in trades if t["session"] == session]
            sw = sum(1 for t in st if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {session}: {len(st)} trades | WR={sw/len(st):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in st]):+.4%}"
            )

        # ── Hold time distribution ─────────────────────────────────────────────
        hold_hours = [t["hold_hours"] for t in trades if t.get("hold_hours") is not None]
        if hold_hours:
            algo.Debug(
                f"--- Hold Time (h): min={min(hold_hours):.2f} "
                f"avg={np.mean(hold_hours):.2f} "
                f"max={max(hold_hours):.2f} ---"
            )

        # ── Rejection reasons ─────────────────────────────────────────────────
        if self.rejection_counts:
            algo.Debug("--- Rejection / Skip Reasons ---")
            for reason, count in sorted(
                self.rejection_counts.items(), key=lambda x: -x[1]
            ):
                algo.Debug(f"  {reason}: {count}")

        algo.Debug("=" * 64)

    def print_daily_summary(self):
        """
        Print today's attribution summary (called from DailyReport scheduled task).
        """
        algo  = self.algo
        today = algo.Time.date()

        trades_today = [
            t for t in self.trade_log
            if t.get("time") and t["time"].date() == today
        ]
        total = len(trades_today)

        algo.Debug(f"=== DAILY {today} ===")
        algo.Debug(
            f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | "
            f"Cash: ${algo.Portfolio.Cash:.2f}"
        )
        algo.Debug(
            f"VIX: {getattr(algo, 'vix_value', 20.0):.1f} | "
            f"Regime: {getattr(algo, 'market_regime', 'unknown')}"
        )
        algo.Debug(
            f"Candidates today: gen={self.candidates_generated} "
            f"acc={self.candidates_accepted} | Trades closed: {total}"
        )

        if total > 0:
            wins = sum(1 for t in trades_today if t["pnl_pct"] > 0)
            wr   = wins / total
            avg  = np.mean([t["pnl_pct"] for t in trades_today])
            algo.Debug(f"Today: WR={wr:.1%} | Avg={avg:+.4%}")

            for st in ALL_SETUP_TYPES:
                st_t = [t for t in trades_today if t["setup_type"] == st]
                if st_t:
                    sw = sum(1 for t in st_t if t["pnl_pct"] > 0)
                    algo.Debug(
                        f"  {st}: {len(st_t)} trades | WR={sw/len(st_t):.1%} | "
                        f"Avg={np.mean([t['pnl_pct'] for t in st_t]):+.4%}"
                    )
