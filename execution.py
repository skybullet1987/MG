# region imports
from AlgorithmImports import *
import json
import math
import numpy as np
from collections import deque
from datetime import timedelta
# endregion

# Constants from main_qa.py lines 31-68
SYMBOL_BLACKLIST = {
    "BTCUSD",  # Reference symbol only — never trade (too expensive for small capital, always dust)
    "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD",
    "DAIUSD", "TUSDUSD", "WETHUSD", "WBTCUSD", "WAXLUSD",
    "SHIBUSD", "XMRUSD", "ZECUSD", "DASHUSD",
    "XNYUSD",
    "BDXNUSD", "RAIINUSD", "LUNAUSD", "LUNCUSD", "USTCUSD", "ABORDUSD",
    "BONDUSD", "KEEPUSD", "ORNUSD",
    "MUSD", "ICNTUSD",
    "EPTUSD", "LMWRUSD",
    "CPOOLUSD",
    "ARCUSD", "PAXGUSD",
    "PARTIUSD", "RAREUSD", "BANANAS31USD",
    # Forex pairs
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "JPYUSD", "CADUSD", "CHFUSD", "CNYUSD", "HKDUSD", "SGDUSD",
    "SEKUSD", "NOKUSD", "DKKUSD", "KRWUSD", "TRYUSD", "ZARUSD", "MXNUSD", "INRUSD", "BRLUSD",
    "PLNUSD", "THBUSD",
}

# Known fiat currency codes used to filter forex pairs from the crypto universe
KNOWN_FIAT_CURRENCIES = frozenset({
    "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF", "CNY", "HKD", "SGD",
    "SEK", "NOK", "DKK", "KRW", "TRY", "ZAR", "MXN", "INR", "BRL", "PLN", "THB",
})

# Fractional haircut applied before re-rounding a sell quantity that exceeds the actual
# portfolio holding.  The 0.9999 factor (0.01% reduction) is enough to push floating-point
# values below the lot_size boundary while keeping the sell amount virtually identical.
QUANTITY_HAIRCUT_FACTOR = 0.9999

# Tolerance multiplier used when comparing a rounded quantity to the actual holding.
# A tiny IEEE 754 overshoot (e.g. 0.09440399000000001 vs 0.09440399) is far smaller
# than 0.01%, so this threshold distinguishes real overshoots from floating-point noise.
QUANTITY_OVERSHOOT_TOLERANCE = 1.0001

KRAKEN_MIN_QTY_FALLBACK = {
    'AXSUSD': 5.0, 'SANDUSD': 10.0, 'MANAUSD': 10.0, 'ADAUSD': 10.0,
    'MATICUSD': 10.0, 'DOTUSD': 1.0, 'LINKUSD': 0.5, 'AVAXUSD': 0.2,
    'ATOMUSD': 0.5, 'NEARUSD': 1.0, 'SOLUSD': 0.05,
    'ALGOUSD': 10.0, 'XLMUSD': 30.0, 'TRXUSD': 50.0, 'ENJUSD': 10.0,
    'BATUSD': 10.0, 'CRVUSD': 5.0, 'SNXUSD': 3.0, 'COMPUSD': 0.1,
    'AAVEUSD': 0.05, 'MKRUSD': 0.01, 'YFIUSD': 0.001, 'UNIUSD': 1.0,
    'SUSHIUSD': 5.0, '1INCHUSD': 5.0, 'GRTUSD': 10.0, 'FTMUSD': 10.0,
    'IMXUSD': 5.0, 'APEUSD': 2.0, 'GMTUSD': 10.0, 'OPUSD': 5.0,
    'LDOUSD': 5.0, 'ARBUSD': 5.0, 'LPTUSD': 5.0, 'KTAUSD': 10.0,
    'GUNUSD': 50.0, 'BANANAS31USD': 500.0, 'CHILLHOUSEUSD': 500.0,
    'PHAUSD': 50.0, 'MUSD': 50.0, 'ICNTUSD': 50.0,
    'SHIBUSD': 50000.0, 'XRPUSD': 2.0,
}

MIN_NOTIONAL_FALLBACK = {
    'EWTUSD': 2.0, 'SANDUSD': 8.0, 'CTSIUSD': 18.0, 'MKRUSD': 0.01,
    'AUDUSD': 10.0, 'LPTUSD': 0.3, 'OXTUSD': 40.0, 'ENJUSD': 15.0,
    'UNIUSD': 0.5, 'LSKUSD': 3.0, 'BCHUSD': 1.0,
}

# Fee buffer applied when computing sell quantity to prevent overselling due to
# Kraken deducting fees from the base asset after a buy (Cash Modeling discrepancy).
KRAKEN_SELL_FEE_BUFFER = 0.006  # 0.6% (0.4% base fee + 0.2% safety margin)


class RealisticCryptoSlippage:
    """Volume-aware slippage model for crypto."""
    
    def __init__(self):
        self.base_slippage_pct = 0.001
        self.volume_impact_factor = 0.10
        self.max_slippage_pct = 0.02
    
    def get_slippage_approximation(self, asset, order):
        price = asset.Price
        if price <= 0:
            return 0
        
        slippage_pct = self.base_slippage_pct
        
        volume = asset.Volume
        if volume > 0:
            order_value = abs(order.Quantity) * price
            volume_value = volume * price
            if volume_value > 0:
                participation_rate = order_value / volume_value
                volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                slippage_pct += volume_impact
        
        if price < 0.01:
            slippage_pct *= 3.0
        elif price < 0.10:
            slippage_pct *= 2.0
        elif price < 1.0:
            slippage_pct *= 1.5
        
        slippage_pct = min(slippage_pct, self.max_slippage_pct)
        
        return price * slippage_pct


def get_min_quantity(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    try:
        if symbol in algo.Securities:
            sec = algo.Securities[symbol]
            if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                min_size = sec.SymbolProperties.MinimumOrderSize
                if min_size is not None and min_size > 0:
                    return float(min_size)
    except Exception as e:
        algo.Debug(f"Error getting min quantity for {ticker}: {e}")
        pass
    if ticker in KRAKEN_MIN_QTY_FALLBACK:
        return KRAKEN_MIN_QTY_FALLBACK[ticker]
    return estimate_min_qty(algo, symbol)


def estimate_min_qty(algo, symbol):
    try:
        price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    except Exception as e:
        algo.Debug(f"Error getting price for min qty estimate: {e}")
        price = 0
    if price <= 0: return 50.0
    if price < 0.001: return 1000.0
    elif price < 0.01: return 500.0
    elif price < 0.1: return 50.0
    elif price < 1.0: return 10.0
    elif price < 10.0: return 5.0
    elif price < 100.0: return 1.0
    elif price < 1000.0: return 0.1
    else: return 0.01


def get_min_notional_usd(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    fallback = MIN_NOTIONAL_FALLBACK.get(ticker, algo.min_notional)
    try:
        price = algo.Securities[symbol].Price
        min_qty = get_min_quantity(algo, symbol)
        implied = price * min_qty if price > 0 else fallback
        return max(fallback, implied, algo.min_notional)
    except Exception as e:
        algo.Debug(f"Error in get_min_notional_usd for {symbol.Value}: {e}")
        return max(fallback, algo.min_notional)


def round_quantity(algo, symbol, quantity):
    try:
        lot_size = algo.Securities[symbol].SymbolProperties.LotSize
        if lot_size is not None and lot_size > 0:
            rounded = float(math.floor(quantity / lot_size)) * lot_size
            # Clamp: never return more than the input (IEEE 754 safety)
            if rounded > quantity:
                rounded = rounded - lot_size
                if rounded < 0:
                    rounded = 0.0
            return rounded
        return quantity
    except Exception as e:
        algo.Debug(f"Error rounding quantity for {symbol.Value}: {e}")
        return quantity


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
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return False
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price * abs(holding_qty) < min_notional * 0.9:
        return False
    # Verify fee reserve before selling (Kraken cash account requirement)
    if algo.LiveMode and holding_qty > 0:
        estimated_fee = price * abs(holding_qty) * 0.006  # 0.6% fee estimate (0.4% base + 0.2% safety buffer)
        try:
            available_usd = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_usd = algo.Portfolio.Cash
        is_stop_loss = "Stop Loss" in tag or "Stop" in tag
        if available_usd < estimated_fee and not is_stop_loss:
            algo.Debug(f"⚠️ SKIP SELL {symbol.Value}: fee reserve too low "
                       f"(need ${estimated_fee:.4f}, have ${available_usd:.4f})")
            if symbol not in algo.entry_prices:
                algo.entry_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.highest_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.entry_times[symbol] = algo.Time
            return False
    algo.Transactions.CancelOpenOrders(symbol)
    # For exits: use MinimumOrderSize from the security (brokerage-enforced floor).
    # lot_size < MinimumOrderSize for some assets (e.g. ETHUSD: lot=0.001, min=0.002),
    # so using lot_size alone allows placing orders the brokerage will reject.
    try:
        sec = algo.Securities[symbol]
        min_order_size = sec.SymbolProperties.MinimumOrderSize
        if min_order_size is not None and min_order_size > 0:
            exit_min_qty = float(min_order_size)
        else:
            exit_min_qty = min_qty
    except Exception as e:
        algo.Debug(f"Warning: could not get MinimumOrderSize for {symbol.Value}: {e}")
        exit_min_qty = min_qty
    if abs(holding_qty) < exit_min_qty:
        return False
    # The portfolio already reflects the post-fee quantity (Kraken deducts fees from
    # the base asset at buy time), so no fee buffer is needed on the sell side.
    safe_qty = round_quantity(algo, symbol, abs(holding_qty))
    # Ensure we never attempt to sell more than the actual portfolio quantity.
    # round_quantity can round UP due to floating-point, causing "insufficient buying power"
    # errors on Kraken Cash Modeling (e.g. post-fee BTC holdings are slightly less than
    # the ordered quantity).  Apply a haircut before re-rounding to guarantee floor division
    # stays at or below actual_qty.
    actual_qty = abs(algo.Portfolio[symbol].Quantity)
    if safe_qty > actual_qty * QUANTITY_OVERSHOOT_TOLERANCE:  # tolerance for floating-point
        safe_qty = round_quantity(algo, symbol, actual_qty)
    # Final hard cap — never exceed actual holding regardless of floating-point rounding
    safe_qty = min(safe_qty, actual_qty)
    if safe_qty < exit_min_qty:
        # Position rounded down below MinimumOrderSize — treat as dust, cannot sell
        return False
    if safe_qty > 0:
        direction_mult = -1 if holding_qty > 0 else 1
        # Spread-aware exit logic
        is_stop_loss = "Stop Loss" in tag
        if not is_stop_loss:
            spread_pct = get_spread_pct(algo, symbol)
            if spread_pct is not None:
                if spread_pct > 0.03:  # 3% spread - log warning but still exit with market
                    algo.Debug(f"⚠️ WIDE SPREAD EXIT: {symbol.Value} spread={spread_pct:.2%}, using market order")
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
                elif algo.LiveMode and spread_pct > 0.015:  # 1.5% spread - use limit order with fallback (live mode only)
                    try:
                        sec = algo.Securities[symbol]
                        bid = sec.BidPrice
                        ask = sec.AskPrice
                        if bid > 0 and ask > 0:
                            mid = 0.5 * (bid + ask)
                            limit_order = algo.LimitOrder(symbol, safe_qty * direction_mult, mid, tag=tag)
                            # Track for fallback in VerifyOrderFills (90 second timeout handled there)
                            if hasattr(algo, '_submitted_orders'):
                                algo._submitted_orders[symbol] = {
                                    'order_id': limit_order.OrderId,
                                    'time': algo.Time,
                                    'quantity': safe_qty * direction_mult,  # Store signed quantity
                                    'is_limit_exit': True,
                                    'intent': 'exit'
                                }
                            algo.Debug(f"LIMIT EXIT: {symbol.Value} at mid ${mid:.4f} (spread={spread_pct:.2%})")
                            return True
                        else:
                            ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                            track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                            return True
                    except Exception as e:
                        algo.Debug(f"Error placing limit exit for {symbol.Value}: {e}")
                        ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                        track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                        return True
                else:
                    # Spread is acceptable, use market order
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
            else:
                # Spread unknown, use market order
                ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                return True
        else:
            # Stop loss - always use market order
            ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
            track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
            return True
    else:
        algo.Debug(f"Warning: {symbol.Value} holding {holding_qty} rounds to 0")
        return False


def partial_smart_sell(algo, symbol, fraction, tag="Partial TP"):
    """
    Sell a fraction (0.0–1.0) of the current position.
    Both the sell portion and the remaining portion must meet the minimum
    order size; otherwise falls back to a full smart_liquidate.
    Returns True if an order was placed successfully, False otherwise.
    """
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    if holding_qty == 0:
        return False
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price <= 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    sell_qty = round_quantity(algo, symbol, abs(holding_qty) * fraction)
    remaining_qty = round_quantity(algo, symbol, abs(holding_qty) * (1.0 - fraction))
    # Both halves must be tradeable; fall back to full exit if not
    if sell_qty < min_qty or remaining_qty < min_qty:
        return smart_liquidate(algo, symbol, tag)
    if sell_qty * price < min_notional * 0.9:
        return smart_liquidate(algo, symbol, tag)
    # Flag as partial so OnOrderEvent skips position cleanup
    if hasattr(algo, '_partial_sell_symbols'):
        algo._partial_sell_symbols.add(symbol)
    direction_mult = -1 if holding_qty > 0 else 1
    ticket = algo.MarketOrder(symbol, sell_qty * direction_mult, tag=tag)
    if ticket is not None:
        algo.Debug(f"PARTIAL SELL: {symbol.Value} | frac={fraction:.0%} qty={sell_qty:.6f} of {abs(holding_qty):.6f}")
    return ticket is not None



    try:
        open_orders = algo.Transactions.GetOpenOrders()
        if len(open_orders) > 0:
            algo.Debug(f"Found {len(open_orders)} open orders - canceling all...")
            for order in open_orders:
                algo.Transactions.CancelOrder(order.Id)
    except Exception as e:
        algo.Debug(f"Error canceling stale orders: {e}")


def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds


def cancel_stale_new_orders(algo):
    # Allow cancel gate when venue is online or unknown (fallback handled elsewhere)
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        timeout_seconds = effective_stale_timeout(algo)
        for order in open_orders:
            sym_val = order.Symbol.Value
            if sym_val in algo._session_blacklist:
                continue
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                algo.Debug(f"Canceling stale: {sym_val} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")
                
                # Check if this order actually filled (fill event missed)
                if is_invested_not_dust(algo, order.Symbol):
                    # Position exists — order filled, we just missed the event
                    # Re-track instead of blacklisting
                    algo.Debug(f"STALE ORDER but position exists: {sym_val} — re-tracking")
                    holding = algo.Portfolio[order.Symbol]
                    algo.entry_prices[order.Symbol] = holding.AveragePrice
                    algo.highest_prices[order.Symbol] = holding.AveragePrice
                    algo.entry_times[order.Symbol] = algo.Time
                    algo.Transactions.CancelOrder(order.Id)  # Cancel the stale order
                    continue  # Don't blacklist
                
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)
                
                # Only blacklist stale ENTRY orders, not EXIT orders
                # Exit orders that are stale just get cooldown to allow retry
                has_position_or_tracked = order.Symbol in algo.entry_prices or (
                    order.Symbol in algo.Portfolio and algo.Portfolio[order.Symbol].Quantity != 0
                )
                
                if has_position_or_tracked:
                    algo.Debug(f"STALE EXIT: {sym_val} - cooldown only, not blacklisted")
                else:
                    algo._session_blacklist.add(sym_val)
                    algo.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {sym_val} - blacklisted for session")
    except Exception as e:
        algo.Debug(f"Error in _cancel_stale_new_orders: {e}")


def is_invested_not_dust(algo, symbol):
    if symbol not in algo.Portfolio:
        return False
    h = algo.Portfolio[symbol]
    if not h.Invested or h.Quantity == 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else h.Price
    notional_ok = (price > 0) and (abs(h.Quantity) * price >= min_notional * 0.5)
    qty_ok = abs(h.Quantity) >= min_qty * 0.5
    return notional_ok or qty_ok


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
                return True  # At least one order is not stale
        return False  # All orders are stale
    except Exception:
        return False


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
        pass
    return None


def spread_ok(algo, symbol):
    sp = get_spread_pct(algo, symbol)
    if sp is None:
        # In live mode with small accounts (< $100), allow unknown spreads
        # Otherwise in live mode, fail-closed (reject if spread unknown); in backtest, allow
        if algo.LiveMode:
            portfolio_value = algo.Portfolio.TotalPortfolioValue
            if portfolio_value < 100:
                return True
            return False
        return True
    effective_spread_cap = algo.max_spread_pct
    if algo.LiveMode and (algo.volatility_regime == "high" or algo.market_regime == "sideways"):
        effective_spread_cap = min(effective_spread_cap, 0.025)
    if sp > effective_spread_cap:
        return False
    crypto = algo.crypto_data.get(symbol)
    if crypto and len(crypto.get('spreads', [])) >= 4:
        median_sp = np.median(list(crypto['spreads']))
        if median_sp > 0 and sp > algo.spread_widen_mult * median_sp:
            return False
    return True


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
    # Then do existing cleanup
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    if symbol in algo.crypto_data:
        algo.crypto_data[symbol]['trail_stop'] = None
    if hasattr(algo, '_spike_entries'):
        algo._spike_entries.pop(symbol, None)
    if hasattr(algo, '_partial_tp_taken'):
        algo._partial_tp_taken.pop(symbol, None)
    if hasattr(algo, '_breakeven_stops'):
        algo._breakeven_stops.pop(symbol, None)


def sync_existing_positions(algo):
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    positions_to_close = []
    for symbol in algo.Portfolio.Keys:
        holding = algo.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0:
            continue
        ticker = symbol.Value
        if symbol in algo.entry_prices:
            continue
        if symbol not in algo.Securities:
            try:
                algo.AddCrypto(ticker, Resolution.Minute, Market.Kraken)
            except Exception as e:
                algo.Debug(f"Error adding crypto {ticker}: {e}")
                continue
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        algo.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
        if current_price > holding.AveragePrice:
            algo.highest_prices[symbol] = current_price
        if pnl_pct >= algo.base_take_profit:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync TP"))
        elif pnl_pct <= -algo.base_stop_loss:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync SL"))
    algo.Debug(f"Synced {synced_count} positions")
    algo.Debug(f"Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug("=" * 50)
    for symbol, ticker, pnl_pct, reason in positions_to_close:
        algo.Debug(f"IMMEDIATE {reason}: {ticker} at {pnl_pct:+.2%}")
        sold = smart_liquidate(algo, symbol, reason)
        if not sold:
            algo.Debug(f"⚠️ IMMEDIATE {reason} FAILED: {ticker} — position unsellable, cleaning up tracking")
            cleanup_position(algo, symbol)
        # Let OnOrderEvent handle cleanup and PnL tracking on fill


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
        
        # Track slippage per symbol (store (time, abs_slippage) tuples for time-based decay)
        if hasattr(algo, '_symbol_slippage_history'):
            ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
            if ticker not in algo._symbol_slippage_history:
                algo._symbol_slippage_history[ticker] = deque(maxlen=30)
            algo._symbol_slippage_history[ticker].append((algo.Time, abs_slip))
        
        # Live slippage alert for unusually high slippage
        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except Exception as e:
        algo.Debug(f"Error in slip_log for {symbol.Value}: {e}")
        pass


def persist_state(algo):
    """Enhanced persist_state with trade_count and peak_value from main_opus."""
    if not algo.LiveMode:
        return
    try:
        spike_entries = [s.Value for s in getattr(algo, '_spike_entries', {}).keys() if hasattr(s, 'Value')]
        state = {
            "session_blacklist": list(algo._session_blacklist),
            "winning_trades": algo.winning_trades,
            "losing_trades": algo.losing_trades,
            "total_pnl": algo.total_pnl,
            "consecutive_losses": algo.consecutive_losses,
            "daily_trade_count": algo.daily_trade_count,
            "trade_count": algo.trade_count,
            "peak_value": algo.peak_value if algo.peak_value is not None else 0,
            "spike_entries": spike_entries,
        }
        algo.ObjectStore.Save("live_state", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")


def load_persisted_state(algo):
    """Enhanced load_persisted_state with trade_count and peak_value from main_opus."""
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("live_state"):
            raw = algo.ObjectStore.Read("live_state")
            data = json.loads(raw)
            algo._session_blacklist = set(data.get("session_blacklist", []))
            algo.winning_trades = data.get("winning_trades", 0)
            algo.losing_trades = data.get("losing_trades", 0)
            algo.total_pnl = data.get("total_pnl", 0.0)
            algo.consecutive_losses = data.get("consecutive_losses", 0)
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count = data.get("trade_count", 0)
            peak = data.get("peak_value", 0)
            if peak > 0:
                algo.peak_value = peak
            # Restore spike entries: map string values back to Symbol objects
            spike_entry_values = set(data.get("spike_entries", []))
            if spike_entry_values and hasattr(algo, '_spike_entries'):
                for symbol in algo.Securities.Keys:
                    if hasattr(symbol, 'Value') and symbol.Value in spike_entry_values:
                        algo._spike_entries[symbol] = True
            algo.Debug(f"Loaded persisted state: blacklist {len(algo._session_blacklist)}, "
                       f"trades W:{algo.winning_trades}/L:{algo.losing_trades}")
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")


def cleanup_object_store(algo):
    """From main_opus._cleanup_object_store."""
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
                    pass
        if n:
            algo.Debug(f"Cleaned {n} keys")
    except Exception as e:
        algo.Debug(f"Cleanup err: {e}")


def live_safety_checks(algo):
    """Extra safety checks for live trading from main_opus."""
    if not algo.LiveMode:
        return True
    
    # Check if we have minimum viable cash
    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except Exception as e:
        algo.Debug(f"Error getting cash from CashBook, using Portfolio.Cash: {e}")
        cash = algo.Portfolio.Cash
    
    if cash < 2.0:
        debug_limited(algo, "LIVE SAFETY: Cash below $2, pausing new entries")
        return False
    
    # Rate limit: don't trade more than once per 5 minutes in live
    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < 300:
            return False
    
    return True


def kelly_fraction(algo):
    """Kelly criterion-based position sizing from main_opus."""
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


def get_slippage_penalty(algo, symbol):
    """
    Calculate position size multiplier based on historical slippage for a symbol.
    Returns a value between 0.3 and 1.0. Ignores slippage entries older than 48 hours.
    """
    if not hasattr(algo, '_symbol_slippage_history'):
        return 1.0
    
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    if ticker not in algo._symbol_slippage_history:
        return 1.0
    
    slippage_history = algo._symbol_slippage_history[ticker]
    if len(slippage_history) == 0:
        return 1.0
    
    # Filter out entries older than 48 hours
    cutoff = algo.Time - timedelta(hours=48)
    recent_slips = [slip for entry_time, slip in slippage_history if entry_time >= cutoff]
    if len(recent_slips) == 0:
        return 1.0
    
    avg_slippage = sum(recent_slips) / len(recent_slips)
    
    # Apply penalties based on average slippage
    if avg_slippage > 0.010:  # > 1.0%
        return 0.3
    elif avg_slippage > 0.005:  # > 0.5%
        return 0.6
    elif avg_slippage > 0.003:  # > 0.3%
        return 0.8
    else:
        return 1.0


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=30, tag="Entry"):
    """
    Place a limit order at mid-price with fallback to market order after timeout.
    In backtest mode, use market orders directly.
    Returns the ticket from the order placement.
    """
    # In backtest mode, use market orders directly
    if not algo.LiveMode:
        return algo.MarketOrder(symbol, quantity, tag=tag)
    
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        
        # If bid/ask unavailable, use market order immediately
        if bid <= 0 or ask <= 0:
            algo.Debug(f"BID/ASK unavailable for {symbol.Value}, using market order")
            return algo.MarketOrder(symbol, quantity, tag=tag)
        
        # Place limit order at or slightly above bid (Maker order for 0.25% fee)
        # Placing just above bid keeps us in the order book as a maker while
        # improving fill odds slightly vs. the raw bid.
        limit_price = bid * 1.0005  # 0.05% above bid – still below mid, still maker
        limit_price = min(limit_price, ask)  # never cross the spread

        # Place maker limit order
        limit_ticket = algo.LimitOrder(symbol, quantity, limit_price, tag=tag)

        # Track for fallback in VerifyOrderFills
        if hasattr(algo, '_submitted_orders'):
            algo._submitted_orders[symbol] = {
                'order_id': limit_ticket.OrderId,
                'time': algo.Time,
                'quantity': quantity,  # Store signed quantity
                'is_limit_entry': True,
                'timeout_seconds': timeout_seconds,
                'intent': 'entry'
            }

        algo.Debug(f"MAKER LIMIT: {symbol.Value} | qty={quantity} | bid=${bid:.4f} | limit=${limit_price:.4f} | timeout={timeout_seconds}s")
        return limit_ticket
        
    except Exception as e:
        algo.Debug(f"Error placing limit order for {symbol.Value}: {e}, falling back to market")
        return algo.MarketOrder(symbol, quantity, tag=tag)


def normalize_order_time(order_time):
    """Helper to normalize order time by removing timezone info if present."""
    return order_time.replace(tzinfo=None) if order_time.tzinfo is not None else order_time


def record_exit_pnl(algo, symbol, entry_price, exit_price):
    """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
    if entry_price <= 0 or exit_price <= 0:
        algo.Debug(f"⚠️ Cannot record PnL for {symbol.Value}: invalid prices (entry=${entry_price:.4f}, exit=${exit_price:.4f})")
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


def health_check(algo):
    """
    Enhanced health check with improved orphan detection and reverse resync.
    Implements Fix 3: check if ALL orders are stale before skipping orphan check.
    Includes reverse resync (phantom position detection) for live mode.
    """
    if algo.IsWarmingUp:
        return
    
    # Live resync to catch any holdings the event stream might have missed
    resync_holdings_full(algo)
    
    issues = []
    if algo.Portfolio.Cash < 5:
        issues.append(f"Low cash: ${algo.Portfolio.Cash:.2f}")
    
    for symbol in list(algo.entry_prices.keys()):
        open_orders = algo.Transactions.GetOpenOrders(symbol)
        if len(open_orders) > 0:
            # Check if all orders are stuck (older than timeout)
            all_stale = True
            for o in open_orders:
                order_time = normalize_order_time(o.Time)
                if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                    all_stale = False
                    break
            if not all_stale:
                continue  # Has non-stale orders, skip this symbol
            
            # All orders are stale - cancel them
            for o in open_orders:
                try:
                    algo.Transactions.CancelOrder(o.Id)
                    issues.append(f"Canceled stale order: {symbol.Value} (order {o.Id})")
                except Exception as e:
                    algo.Debug(f"Error canceling stale order for {symbol.Value}: {e}")
        
        if not is_invested_not_dust(algo, symbol):
            issues.append(f"Orphan tracking: {symbol.Value}")
            cleanup_position(algo, symbol)
    
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key) and kvp.Key not in algo.entry_prices:
            issues.append(f"Untracked position: {kvp.Key.Value}")
    
    if len(algo._session_blacklist) > 50:
        issues.append(f"Large session blacklist: {len(algo._session_blacklist)}")
    
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
    Live-only safety: backfills tracking for any holdings that exist in the brokerage
    but were not registered via OnOrderEvent (e.g., missed fill events).
    Implements Fix 2: Reverse resync to detect phantom positions.
    """
    if algo.IsWarmingUp:
        return
    if not algo.LiveMode:
        return
    
    # Only log periodically to avoid spam (every 30 minutes)
    if not hasattr(algo, '_last_resync_log') or (algo.Time - algo._last_resync_log).total_seconds() > algo.resync_log_interval_seconds:
        algo.Debug(f"RESYNC CHECK: Portfolio keys={len(list(algo.Portfolio.Keys))}, tracked={len(algo.entry_prices)}")
        algo._last_resync_log = algo.Time
    
    # Forward resync: find holdings we're not tracking
    missing = []
    for symbol in algo.Portfolio.Keys:
        holding = algo.Portfolio[symbol]
        if not holding.Invested or holding.Quantity == 0:
            continue
        if symbol in algo.entry_prices:
            continue
        # Skip symbols being tracked by VerifyOrderFills to avoid conflicts
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        # Check if there are non-stale open orders
        # If all open orders are stale, we should resync anyway
        if has_non_stale_open_orders(algo, symbol):
            continue
        missing.append(symbol)
    
    if missing:
        algo.Debug(f"RESYNC: detected {len(missing)} holdings without tracking; backfilling.")
        for symbol in missing:
            try:
                if symbol not in algo.Securities:
                    algo.AddCrypto(symbol.Value, Resolution.Minute, Market.Kraken)
                holding = algo.Portfolio[symbol]
                entry = holding.AveragePrice
                algo.entry_prices[symbol] = entry
                algo.highest_prices[symbol] = entry
                algo.entry_times[symbol] = algo.Time
                current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                pnl_pct = (current_price - entry) / entry if entry > 0 else 0
                algo.Debug(f"RESYNCED: {symbol.Value} | Qty: {holding.Quantity} | Entry: ${entry:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
            except Exception as e:
                algo.Debug(f"Resync error {symbol.Value}: {e}")
    
    # Reverse resync: detect positions we're tracking but broker no longer holds
    # Fix 2: Phantom position detection
    phantoms = []
    for symbol in list(algo.entry_prices.keys()):
        # Skip if order still pending
        if symbol in algo._submitted_orders:
            continue
        # Skip if in exit cooldown
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        
        holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
        if holding is None or not holding.Invested or holding.Quantity == 0:
            # We're tracking this position but broker says it's gone
            # Check for open orders first
            open_orders = algo.Transactions.GetOpenOrders(symbol)
            if len(open_orders) > 0:
                # Has open orders - check if all are stuck
                all_stuck = True
                for o in open_orders:
                    order_time = normalize_order_time(o.Time)
                    if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                        all_stuck = False
                        break
                
                if not all_stuck:
                    continue  # Has non-stuck orders, wait
                
                # All orders are stuck - cancel them
                for o in open_orders:
                    try:
                        algo.Transactions.CancelOrder(o.Id)
                        algo.Debug(f"PHANTOM: Canceling stuck order {o.Id} for {symbol.Value}")
                    except Exception as e:
                        algo.Debug(f"Error canceling stuck order for {symbol.Value}: {e}")
            
            phantoms.append(symbol)
    
    if phantoms:
        algo.Debug(f"⚠️ REVERSE RESYNC: detected {len(phantoms)} phantom positions")
        for symbol in phantoms:
            algo.Debug(f"⚠️ PHANTOM POSITION DETECTED: {symbol.Value} — tracked but broker qty=0, cleaning up")
            cleanup_position(algo, symbol, record_pnl=True)
            # Fix 3: Set exit cooldown after phantom cleanup
            if hasattr(algo, 'exit_cooldown_hours') and hasattr(algo, '_exit_cooldowns'):
                algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
        # Fix 4: Call persist_state after phantom cleanup
        persist_state(algo)


def verify_order_fills(algo):
    """
    Verify submitted orders filled/timed out. Retry once before blacklisting.
    Implements Fix 1: stuck "New" order detection.
    Implements Fix 4: exit-intent order handling for phantom exits.
    """
    if algo.IsWarmingUp:
        return
    
    current_time = algo.Time
    symbols_to_remove = []
    
    # Process pending retries first
    for symbol in list(algo._retry_pending.keys()):
        cancel_time = algo._retry_pending[symbol]
        if (current_time - cancel_time).total_seconds() >= algo.retry_pending_cooldown_seconds:
            # Check if position exists now (cancel failed or order filled after cancel)
            if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                # Position exists, don't retry - just track it
                holding = algo.Portfolio[symbol]
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol] = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol] = current_time
                    algo.Debug(f"RETRY SKIPPED (position exists): {symbol.Value}")
                del algo._retry_pending[symbol]
            else:
                # Safe to retry now
                algo.Debug(f"RETRY NOW: {symbol.Value} - cancel confirmed")
                del algo._retry_pending[symbol]
    
    for symbol, order_info in list(algo._submitted_orders.items()):
        order_age_seconds = (current_time - order_info['time']).total_seconds()
        order_id = order_info['order_id']
        
        # Determine timeout based on order type
        if order_info.get('is_limit_entry', False):
            timeout = order_info.get('timeout_seconds', 60)
        elif order_info.get('is_limit_exit', False):
            timeout = 90
        else:
            timeout = algo.order_timeout_seconds
        
        # Check for missed fills (order filled but event missed)
        if order_age_seconds > algo.order_fill_check_threshold_seconds:
            intent = order_info.get('intent', 'entry')
            
            # Check order status
            try:
                order = algo.Transactions.GetOrderById(order_id)
                if order is not None and order.Status == OrderStatus.Filled:
                    # Order filled but we missed the event
                    if intent == 'exit':
                        # Exit order filled - position should be gone
                        # Fix 4: Handle exit-intent order properly
                        entry = algo.entry_prices.get(symbol, None)
                        if entry:
                            # Calculate PnL using last known price
                            current_price = algo.Securities[symbol].Price if symbol in algo.Securities else None
                            if current_price is not None and current_price > 0:
                                pnl = record_exit_pnl(algo, symbol, entry, current_price)
                                if pnl is not None:
                                    algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} | PnL: {pnl:+.2%} | Entry: ${entry:.4f} | Exit: ${current_price:.4f}")
                                else:
                                    algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} | Cannot calculate PnL (invalid price), cleaning up anyway")
                            else:
                                algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} | Cannot get current price, cleaning up without PnL")
                            cleanup_position(algo, symbol)
                        symbols_to_remove.append(symbol)
                        algo._order_retries.pop(order_id, None)
                        continue
                    else:
                        # Entry order filled
                        if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                            holding = algo.Portfolio[symbol]
                            entry_price = holding.AveragePrice
                            current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                            
                            # Only set tracking if not already tracked (avoid double increment)
                            if symbol not in algo.entry_prices:
                                algo.entry_prices[symbol] = entry_price
                                algo.highest_prices[symbol] = max(current_price, entry_price)
                                algo.entry_times[symbol] = order_info['time']
                                algo.daily_trade_count += 1
                                algo.Debug(f"FILL VERIFIED (missed event): {symbol.Value} | Entry: ${entry_price:.4f} | Qty: {holding.Quantity}")
                            
                            symbols_to_remove.append(symbol)
                            algo._order_retries.pop(order_id, None)
                            continue
            except Exception as e:
                algo.Debug(f"Error checking order status for {symbol.Value}: {e}")
            
            # Check for exit orders where position is now gone (fill event missed)
            # Fix 1 & Fix 4: Detect phantom exits
            if intent == 'exit':
                holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
                if holding is None or not holding.Invested or holding.Quantity == 0:
                    # Position is gone - the exit filled but we missed the event
                    entry = algo.entry_prices.get(symbol, None)
                    if entry:
                        # Calculate PnL using last known price
                        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else None
                        if current_price is not None and current_price > 0:
                            pnl = record_exit_pnl(algo, symbol, entry, current_price)
                            if pnl is not None:
                                algo.Debug(f"⚠️ PHANTOM EXIT DETECTED (position gone): {symbol.Value} | PnL: {pnl:+.2%} | Entry: ${entry:.4f} | Estimated Exit: ${current_price:.4f}")
                            else:
                                algo.Debug(f"⚠️ PHANTOM EXIT DETECTED (position gone): {symbol.Value} | Cannot calculate PnL (invalid price), cleaning up anyway")
                        else:
                            algo.Debug(f"⚠️ PHANTOM EXIT DETECTED (position gone): {symbol.Value} | Cannot get current price, cleaning up without PnL")
                        cleanup_position(algo, symbol)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    continue
        
        # Handle timeout with retry logic
        if order_age_seconds > timeout:
            retry_count = algo._order_retries.get(order_id, 0)
            
            if retry_count == 0:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo.Debug(f"ORDER TIMEOUT (attempt 1): {symbol.Value} - cancel requested, will retry after cooldown")
                    
                    # Set cooldown flag - do NOT retry immediately
                    algo._retry_pending[symbol] = current_time
                    algo._order_retries[order_id] = 1
                    symbols_to_remove.append(symbol)
                except Exception as e:
                    algo.Debug(f"Error canceling order for {symbol.Value}: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
            else:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo._session_blacklist.add(symbol.Value)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    algo.Debug(f"ORDER TIMEOUT (attempt 2): {symbol.Value} - blacklisted")
                except Exception as e:
                    algo.Debug(f"Error canceling order {order_id} on second timeout: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
    
    # Remove processed orders
    for symbol in symbols_to_remove:
        algo._submitted_orders.pop(symbol, None)


def portfolio_sanity_check(algo):
    """
    Check for portfolio value mismatches between QC and tracked positions.
    Fixed: Use CashBook["USD"].Amount for actual USD cash, not Portfolio.Cash
    (which in Cash account mode includes crypto holdings value).
    Fix 6: Trigger resync_holdings_full on mismatch and log detailed breakdown.
    """
    if algo.IsWarmingUp:
        return
    
    total_qc = algo.Portfolio.TotalPortfolioValue
    
    # Use actual USD cash, NOT Portfolio.Cash (which double-counts in Cash accounts)
    try:
        usd_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        usd_cash = algo.Portfolio.Cash
    
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
    
    expected = usd_cash + tracked_value
    
    # Use both percentage and minimum absolute threshold to avoid spam on small portfolios
    # Also add 1-hour cooldown between warnings
    abs_diff = abs(total_qc - expected)
    if total_qc > 1.0:
        pct_diff = abs_diff / total_qc
        should_warn = pct_diff > algo.portfolio_mismatch_threshold and abs_diff > algo.portfolio_mismatch_min_dollars
        if should_warn:
            if algo._last_mismatch_warning is None or (algo.Time - algo._last_mismatch_warning).total_seconds() >= algo.portfolio_mismatch_cooldown_seconds:
                algo.Debug(f"⚠️ PORTFOLIO MISMATCH: QC total=${total_qc:.2f} but usd_cash+tracked=${expected:.2f} (diff=${abs_diff:.2f}, {pct_diff:.2%})")
                
                # Log detailed breakdown
                algo.Debug("=== PORTFOLIO BREAKDOWN ===")
                algo.Debug(f"USD Cash: ${usd_cash:.2f}")
                
                # Show tracked positions
                if tracked_positions:
                    algo.Debug(f"Tracked Positions ({len(tracked_positions)}):")
                    for ticker, info in tracked_positions.items():
                        algo.Debug(f"  {ticker}: qty={info['qty']:.6f}, price=${info['price']:.4f}, value=${info['value']:.2f}")
                else:
                    algo.Debug("Tracked Positions: None")
                
                # Show untracked portfolio holdings
                untracked = []
                for symbol in algo.Portfolio.Keys:
                    holding = algo.Portfolio[symbol]
                    if holding.Invested and holding.Quantity != 0:
                        if symbol not in algo.entry_prices:
                            price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                            value = abs(holding.Quantity) * price
                            untracked.append({'ticker': symbol.Value, 'qty': holding.Quantity, 'price': price, 'value': value})
                
                if untracked:
                    algo.Debug(f"Untracked Portfolio Holdings ({len(untracked)}):")
                    for info in untracked:
                        algo.Debug(f"  {info['ticker']}: qty={info['qty']:.6f}, price=${info['price']:.4f}, value=${info['value']:.2f}")
                else:
                    algo.Debug("Untracked Portfolio Holdings: None")
                
                # Show crypto CashBook entries (non-USD currencies)
                crypto_cash = []
                try:
                    for currency, cash in algo.Portfolio.CashBook.items():
                        if currency != "USD" and cash.Amount != 0:
                            crypto_cash.append({'currency': currency, 'amount': cash.Amount, 'value': cash.ValueInAccountCurrency})
                except Exception:
                    pass
                
                if crypto_cash:
                    algo.Debug(f"Crypto CashBook Entries ({len(crypto_cash)}):")
                    for info in crypto_cash:
                        algo.Debug(f"  {info['currency']}: amount={info['amount']:.6f}, value=${info['value']:.2f}")
                else:
                    algo.Debug("Crypto CashBook Entries: None")
                
                # Show which side is higher
                if total_qc > expected:
                    algo.Debug(f"QC Portfolio is higher by ${abs_diff:.2f} ({pct_diff:.2%})")
                else:
                    algo.Debug(f"Tracked value is higher by ${abs_diff:.2f} ({pct_diff:.2%})")
                
                algo.Debug("=== END BREAKDOWN ===")
                
                # Trigger resync_holdings_full to attempt auto-fix
                algo.Debug("Triggering resync_holdings_full to attempt auto-fix...")
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
    algo.Debug(f"Pos: {get_actual_position_count(algo)}/{algo.base_max_positions} | {algo.market_regime} {algo.volatility_regime} {algo.market_breadth:.0%}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    if algo._session_blacklist:
        algo.Debug(f"Blacklist: {len(algo._session_blacklist)}")
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key):
            s = kvp.Key
            entry = algo.entry_prices.get(s, kvp.Value.AveragePrice)
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            pnl = (cur - entry) / entry if entry > 0 else 0
            algo.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")
    persist_state(algo)
