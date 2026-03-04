# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, smart_liquidate, normalize_order_time, record_exit_pnl,
    persist_state, health_check, resync_holdings_full, verify_order_fills,
    portfolio_sanity_check, review_performance, SYMBOL_BLACKLIST, KNOWN_FIAT_CURRENCIES,
)
from collections import deque
import config as MG3Config
# endregion

# ---------------------------------------------------------------------------
# Position lifecycle states used by MG3 state-machine hooks.
# Each open position progresses: FLAT -> ENTERING -> OPEN -> EXITING -> FLAT
# These constants are imported by other modules that need them.
# ---------------------------------------------------------------------------
POSITION_STATE_FLAT     = "flat"
POSITION_STATE_ENTERING = "entering"
POSITION_STATE_OPEN     = "open"
POSITION_STATE_EXITING  = "exiting"


class AppMixin:
    """Bootstrap and wiring helpers for SimplifiedCryptoStrategy.

    Contains parameter loading, position state-machine helpers, reconciliation,
    scheduled-task callbacks, universe filtering, and symbol initialisation.
    """

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
            return default
        except Exception as e:
            self.Debug(f"Error getting parameter {name}: {e}")
            return default

    def _get_param_str(self, name, default):
        """Get a string parameter (mode, etc.) with fallback to default."""
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return str(param).strip().lower()
            return default
        except Exception as e:
            self.Debug(f"Error getting string parameter {name}: {e}")
            return default

    # ---------------------------------------------------------------------------
    # MG3: Position lifecycle state-machine helpers
    # ---------------------------------------------------------------------------

    def _set_position_state(self, symbol, state):
        """Transition a symbol to the given lifecycle state and log the change."""
        prev = self._position_states.get(symbol, POSITION_STATE_FLAT)
        if prev != state:
            self._position_states[symbol] = state
            self.Debug(f"POS_STATE {symbol.Value}: {prev} -> {state}")

    def _get_position_state(self, symbol):
        """Return the current lifecycle state for a symbol."""
        return self._position_states.get(symbol, POSITION_STATE_FLAT)

    # ---------------------------------------------------------------------------
    # MG3: Reconciliation hook (local-vs-exchange check, Kraken preparation)
    # ---------------------------------------------------------------------------

    def ReconcilePositions(self):
        """
        Periodic reconciliation check: compare locally tracked positions against
        the portfolio as reported by QuantConnect / the brokerage.

        In backtest mode the portfolio is always consistent, so this acts as a
        lightweight sanity check and logs any discrepancies.  In live mode (Kraken)
        this would additionally trigger a brokerage query to catch missed fills.

        Hook points for future Kraken-native reconciliation:
          - kraken_price_normalize(price, symbol)   → tick/precision rounding
          - kraken_qty_normalize(qty, symbol)        → lot-size rounding
          - kraken_reconcile_with_api()              → REST /Balance + /OpenOrders
        """
        if self.IsWarmingUp:
            return
        now = self.Time
        self._last_reconciliation_time = now

        # 1. Find positions tracked locally but with zero brokerage holding.
        # Also catch stuck ENTERING orders (entry never filled or was silently cancelled).
        phantoms = []
        for sym in list(self._position_states.keys()):
            state = self._position_states[sym]
            if state == POSITION_STATE_FLAT:
                continue
            if sym not in self.Portfolio or not self.Portfolio[sym].Invested:
                # Skip if a pending submitted order is still outstanding
                if sym in self._submitted_orders:
                    continue
                # For ENTERING state: also skip if a non-stale open order exists
                if state == POSITION_STATE_ENTERING:
                    open_orders = self.Transactions.GetOpenOrders(sym)
                    timeout = self.live_stale_order_timeout_seconds if self.LiveMode else self.stale_order_timeout_seconds
                    has_live_order = any(
                        (self.Time - (o.Time.replace(tzinfo=None) if o.Time.tzinfo else o.Time)).total_seconds() <= timeout
                        for o in open_orders
                    )
                    if has_live_order:
                        continue
                phantoms.append(sym)

        for sym in phantoms:
            self.Debug(f"[RECONCILE] phantom state={self._position_states[sym]} for {sym.Value} — resetting to flat")
            self._set_position_state(sym, POSITION_STATE_FLAT)

        # 2. Find brokerage holdings not tracked in our state-machine
        untracked = []
        for kvp in self.Portfolio:
            sym = kvp.Key
            if not is_invested_not_dust(self, sym):
                continue
            if self._get_position_state(sym) == POSITION_STATE_FLAT:
                untracked.append(sym)

        for sym in untracked:
            self.Debug(f"[RECONCILE] untracked holding {sym.Value} — setting state to open")
            self._set_position_state(sym, POSITION_STATE_OPEN)

        # Kraken preparation placeholder:
        # In a future live integration, call self._kraken_api_reconcile() here.
        # self._kraken_api_reconcile()  # TODO: implement for live Kraken deployment

    def _normalize_order_time(self, order_time):
        """Helper to normalize order time by removing timezone info if present."""
        return normalize_order_time(order_time)

    def _record_exit_pnl(self, symbol, entry_price, exit_price):
        """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
        return record_exit_pnl(self, symbol, entry_price, exit_price)

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        # MG3: snapshot portfolio value at day-open for daily loss guard
        self._daily_open_value = self.Portfolio.TotalPortfolioValue
        for crypto in self.crypto_data.values():
            crypto['trade_count_today'] = 0
        if len(self._session_blacklist) > 0:
            self.Debug(f"Clearing session blacklist ({len(self._session_blacklist)} items)")
            self._session_blacklist.clear()
        persist_state(self)

    def HealthCheck(self):
        if self.IsWarmingUp: return
        health_check(self)

    def ResyncHoldings(self):
        if self.IsWarmingUp: return
        if not self.LiveMode: return
        resync_holdings_full(self)

    def VerifyOrderFills(self):
        if self.IsWarmingUp: return
        verify_order_fills(self)

    def PortfolioSanityCheck(self):
        if self.IsWarmingUp: return
        portfolio_sanity_check(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10: return
        review_performance(self)

    def _cancel_stale_orders(self):
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                self.Debug(f"Found {len(open_orders)} open orders - canceling all...")
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug(f"Error canceling stale orders: {e}")

    def UniverseFilter(self, universe):
        selected = []
        for crypto in universe:
            ticker = crypto.Symbol.Value
            if ticker in SYMBOL_BLACKLIST or ticker in self._session_blacklist:
                continue
            if not ticker.endswith("USD"):
                continue
            # Filter out forex pairs by checking that the base currency is not a known fiat
            base = ticker[:-3]  # remove "USD" suffix
            if base in KNOWN_FIAT_CURRENCIES:
                continue
            if crypto.VolumeInUsd is None or crypto.VolumeInUsd == 0:
                continue
            if crypto.VolumeInUsd >= self.min_volume_usd:
                selected.append(crypto)
        selected.sort(key=lambda x: x.VolumeInUsd, reverse=True)
        return [c.Symbol for c in selected[:self.max_universe_size]]

    def _initialize_symbol(self, symbol):
        self.crypto_data[symbol] = {
            'prices': deque(maxlen=self.lookback),
            'returns': deque(maxlen=self.lookback),
            'volume': deque(maxlen=self.lookback),
            'volume_ma': deque(maxlen=self.medium_period),
            'dollar_volume': deque(maxlen=self.lookback),
            'ema_ultra_short': ExponentialMovingAverage(self.ultra_short_period),
            'ema_short': ExponentialMovingAverage(self.short_period),
            'ema_medium': ExponentialMovingAverage(self.medium_period),
            'ema_5': ExponentialMovingAverage(5),
            'atr': AverageTrueRange(14),
            'adx': AverageDirectionalIndex(self.adx_min_period),
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(7),
            'rs_vs_btc': deque(maxlen=self.medium_period),
            'zscore': deque(maxlen=self.short_period),
            'last_price': 0,
            'recent_net_scores': deque(maxlen=3),
            'spreads': deque(maxlen=self.spread_median_window),
            'trail_stop': None,
            'highs': deque(maxlen=self.lookback),
            'lows': deque(maxlen=self.lookback),
            'bb_upper': deque(maxlen=self.short_period),
            'bb_lower': deque(maxlen=self.short_period),
            'bb_width': deque(maxlen=self.medium_period),
            'trade_count_today': 0,
            'last_loss_time': None,
            'bid_size': 0.0,
            'ask_size': 0.0,
            'vwap_pv': deque(maxlen=20),
            'vwap_v': deque(maxlen=20),
            'vwap': 0.0,
            'volume_long': deque(maxlen=1440),
            'vwap_sd': 0.0,
            'vwap_sd2_lower': 0.0,
            'vwap_sd3_lower': 0.0,
            'cvd': deque(maxlen=self.lookback),
            'ker': deque(maxlen=self.short_period),
            'kalman_estimate': 0.0,
            'kalman_error_cov': 1.0,
        }

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data:
                self._initialize_symbol(symbol)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if not self.IsWarmingUp and is_invested_not_dust(self, symbol):
                smart_liquidate(self, symbol, "Removed from universe")
                # Don't cleanup here — let OnOrderEvent handle it on fill
                self.Debug(f"FORCED EXIT: {symbol.Value} - removed from universe")
            # Only delete crypto_data if not invested (otherwise OnOrderEvent needs it)
            if symbol in self.crypto_data and not is_invested_not_dust(self, symbol):
                del self.crypto_data[symbol]
