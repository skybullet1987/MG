# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, smart_liquidate, partial_smart_sell,
    persist_state, load_persisted_state, cleanup_object_store,
    health_check, resync_holdings_full, verify_order_fills,
    portfolio_sanity_check, review_performance,
    SYMBOL_BLACKLIST, KNOWN_FIAT_CURRENCIES,
    get_spread_pct, debug_limited, kelly_fraction, sync_existing_positions,
    spread_ok, cancel_stale_new_orders, has_open_orders, get_actual_position_count,
    get_min_quantity, get_min_notional_usd, round_quantity, place_limit_or_market,
    get_open_buy_orders_value, get_slippage_penalty, live_safety_checks,
    KRAKEN_SELL_FEE_BUFFER, cleanup_position, slip_log, daily_report,
)
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
import config as MG3Config
from datetime import timedelta
# Helper modules — plain functions called with self to keep main.py under QC's
# 64 000-character file-size limit and avoid multiple-inheritance issues.
import mg3_data
import mg3_rebalance
import mg3_exits
import mg3_reporting
from mg3_constants import (
    POSITION_STATE_FLAT, POSITION_STATE_ENTERING, POSITION_STATE_OPEN,
    POSITION_STATE_EXITING, POSITION_STATE_RECOVERING,
)
# endregion


class MakerTakerFeeModel(FeeModel):
    """Custom Fee Model: 0.25% Maker (Limit), 0.40% Taker (Market).
    Note: assumes all Limit orders rest in the book as maker orders."""
    def GetOrderFee(self, parameters):
        order = parameters.Order
        fee_pct = 0.0025 if order.Type == OrderType.Limit else 0.0040
        trade_value = order.AbsoluteQuantity * parameters.Security.Price
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


class SimplifiedCryptoStrategy(QCAlgorithm):
    """
    Machine Gun 3 - Micro-Scalping System v8.1.0
    =============================================
    Single-class QCAlgorithm implementation — QC/PythonNet compatible.
    No mixin inheritance, no dynamic method injection.

    Required files (all must be present in the same QC project directory):
      main.py          - this file; single-class entrypoint (< 64 000 chars)
      config.py        - all tunable parameters
      config_loader.py - config validation helpers
      execution.py     - shared order-execution utilities
      scoring.py       - MicroScalpEngine signal calculations
      mg3_constants.py - position lifecycle state constants (shared)
      mg3_data.py      - per-bar data update helpers
      mg3_rebalance.py - entry-selection and trade-execution helpers
      mg3_exits.py     - exit condition and liquidation helpers
      mg3_reporting.py - order-event, reporting, and metrics helpers
    """


    def Initialize(self):
        self.SetStartDate(2025, 1, 1)
        self.SetCash(2000)  # $2,000 test-account baseline
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        # --- MG3: load mode from config (overridable via QC parameter) ---
        self.mg3_mode = self._get_param_str("mode", MG3Config.MODE_DEFAULT)
        self.mg3_long_enabled  = True   # longs always enabled
        self.mg3_short_enabled = False  # shorts disabled by default

        # --- MG3: backtest realism flags (sourced from config) ---
        self.mg3_strict_limit_fill    = MG3Config.STRICT_LIMIT_FILL
        self.mg3_fee_assumption_rt    = MG3Config.FEE_ASSUMPTION_RT
        self.mg3_slippage_buffer      = MG3Config.SLIPPAGE_BUFFER

        # --- MG3: Kraken normalization hooks (None = use exchange defaults) ---
        self.mg3_kraken_price_precision = MG3Config.KRAKEN_PRICE_PRECISION
        self.mg3_kraken_lot_size        = MG3Config.KRAKEN_LOT_SIZE

        # --- MG3: reconciliation cadence (seconds) ---
        self.mg3_reconciliation_cadence = MG3Config.RECONCILIATION_CADENCE_SECONDS

        # --- MG3: position lifecycle state-machine ---
        # Maps symbol -> one of POSITION_STATE_* constants
        self._position_states = {}

        # --- MG3: backtest metrics counters ---
        self.mg3_cancel_count    = 0   # total orders cancelled
        self.mg3_fill_count      = 0   # total orders filled
        self.mg3_invalid_count   = 0   # total invalid orders received
        self.mg3_peak_positions  = 0   # max simultaneous open positions seen
        self.mg3_recovery_events = 0   # positions escalated to force-market recovery
        # PnL by exit tag: tag -> [pnl_pct, ...]
        self.mg3_pnl_by_tag    = {}
        # Last reconciliation timestamp
        self._last_reconciliation_time = None
        # Daily open portfolio value for daily loss guard
        self._daily_open_value = None

        self.entry_threshold = 0.40
        self.high_conviction_threshold = 0.60

        self.quick_take_profit = self._get_param("quick_take_profit", 0.080)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.025)
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  4.0)
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  2.0)
        self.trail_activation  = self._get_param("trail_activation",  0.010)
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.005)
        self.time_stop_hours   = self._get_param("time_stop_hours",   2.0)
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.003)
        self.extended_time_stop_hours   = self._get_param("extended_time_stop_hours",   4.0)
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.015)
        self.stale_position_hours       = self._get_param("stale_position_hours",       6.0)

        # Keep legacy names used elsewhere
        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 2.0

        self.position_size_pct  = 0.20  # cap for high-vol size boost; base sizing in scoring.py
        self.base_max_positions = MG3Config.MAX_OPEN_POSITIONS
        self.max_positions      = MG3Config.MAX_OPEN_POSITIONS
        self.min_notional       = 5.5
        self.max_position_usd   = self._get_param("max_position_usd", MG3Config.MAX_SYMBOL_EXPOSURE_USD)
        self.min_price_usd      = 0.005
        self.cash_reserve_pct   = 0.0
        self.min_notional_fee_buffer = 1.5

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap", 0.80)
        self.min_asset_vol_floor     = 0.05

        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(60 * 24 * 365)

        self.max_spread_pct         = 0.003
        self.spread_median_window   = 12
        self.spread_widen_mult      = 2.5
        self.min_dollar_volume_usd  = 5000
        self.min_volume_usd         = 10000

        self.skip_hours_utc         = []
        self.max_daily_trades       = 24000
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 1.0
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 3

        self.expected_round_trip_fees = MG3Config.FEE_ASSUMPTION_RT
        self.fee_slippage_buffer      = MG3Config.SLIPPAGE_BUFFER
        self.min_expected_profit_pct  = MG3Config.MIN_EXPECTED_PROFIT_PCT
        self.adx_min_period           = 10

        self.stale_order_timeout_seconds      = MG3Config.ORDER_TIMEOUT_SECONDS
        self.live_stale_order_timeout_seconds = MG3Config.LIVE_ORDER_TIMEOUT_SECONDS
        self.max_concurrent_open_orders       = MG3Config.MAX_CONCURRENT_OPEN_ORDERS
        self.open_orders_cash_threshold       = MG3Config.OPEN_ORDERS_CASH_THRESHOLD
        self.order_fill_check_threshold_seconds = 60
        self.order_timeout_seconds              = MG3Config.ORDER_TIMEOUT_SECONDS
        self.resync_log_interval_seconds        = 1800
        self.portfolio_mismatch_threshold       = MG3Config.PORTFOLIO_MISMATCH_THRESHOLD
        self.portfolio_mismatch_min_dollars     = MG3Config.PORTFOLIO_MISMATCH_MIN_DOLLARS
        self.portfolio_mismatch_cooldown_seconds = MG3Config.PORTFOLIO_MISMATCH_COOLDOWN_SECONDS
        self.retry_pending_cooldown_seconds     = MG3Config.RETRY_BACKOFF_BASE_SECONDS * 2
        self.rate_limit_cooldown_minutes        = 10

        self.max_drawdown_limit    = MG3Config.MAX_DRAWDOWN_LIMIT
        self.cooldown_hours        = MG3Config.DRAWDOWN_COOLDOWN_HOURS
        self.consecutive_losses    = 0
        self.max_consecutive_losses = MG3Config.MAX_CONSECUTIVE_LOSSES
        self._consecutive_loss_halve_remaining = 0
        self.circuit_breaker_expiry = None

        self._positions_synced    = False
        self._session_blacklist   = set()
        self._max_session_blacklist_size = 100
        self._first_post_warmup   = True
        self._submitted_orders    = {}
        self._symbol_slippage_history = {}
        self._order_retries       = {}
        self._retry_pending       = {}
        self._rate_limit_until    = None
        self._last_mismatch_warning = None
        self._failed_exit_attempts = {}
        self._failed_exit_counts   = {}

        self.peak_value       = None
        self.drawdown_cooldown = 0
        self.crypto_data      = {}
        self.entry_prices     = {}
        self.highest_prices   = {}
        self.entry_times      = {}
        self.entry_volumes    = {}   # for volume dry-up exit
        self._partial_tp_taken      = {}
        self._breakeven_stops       = {}
        self._partial_sell_symbols  = set()
        self._choppy_regime_entries = {}
        self.partial_tp_threshold   = 0.025
        self.stagnation_minutes     = 45
        self.stagnation_pnl_threshold = 0.005
        self.rsi_peaked_overbought = {}
        self.trade_count      = 0
        self._pending_orders  = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns  = {}
        self._symbol_loss_cooldowns = {}
        self._cash_mode_until = None
        self._recent_trade_outcomes = deque(maxlen=20)
        self.trailing_grace_hours = 1
        self._slip_abs        = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold  = 0.0015
        self.slip_outlier_threshold = 0.004
        self.slip_alert_duration_hours = 2
        self._bad_symbol_counts = {}
        self._recent_tickets  = deque(maxlen=25)

        self._rolling_wins      = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None

        self.btc_symbol       = None
        self.btc_returns      = deque(maxlen=72)
        self.btc_prices       = deque(maxlen=72)
        self.btc_volatility   = deque(maxlen=72)
        self.btc_ema_24       = ExponentialMovingAverage(24)
        self.market_regime    = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth   = 0.5
        self._regime_hold_count = 0

        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.trade_log      = []
        self.log_budget     = 0
        self.last_log_time  = None

        self.max_universe_size = 60

        self.kraken_status = "unknown"
        self._last_skip_reason = None

        # --- MG3: small-account auto-scaling ---
        # When SetCash() is below SMALL_ACCOUNT_THRESHOLD_USD (or SMALL_ACCOUNT_MODE
        # is forced on in config.py), parameters are automatically scaled down so the
        # strategy is compatible with accounts as small as $120.
        # This override runs *after* all default parameters are set, so it takes
        # precedence over any config.py values.
        initial_cash = self.Portfolio.Cash
        _small_account = (
            MG3Config.SMALL_ACCOUNT_MODE
            or initial_cash < MG3Config.SMALL_ACCOUNT_THRESHOLD_USD
        )
        if _small_account:
            self.base_max_positions = MG3Config.SMALL_ACCOUNT_MAX_POSITIONS
            self.max_positions      = MG3Config.SMALL_ACCOUNT_MAX_POSITIONS
            # Cap per-position size to 30% of cash, not exceeding the config hard limit.
            self.max_position_usd   = min(
                MG3Config.SMALL_ACCOUNT_MAX_EXPOSURE_USD,
                max(initial_cash * 0.30, self.min_notional * 3),
            )
            self.Debug(
                f"[MG3] Small-account mode: ${initial_cash:.0f} → "
                f"max_pos={self.max_positions}  max_usd=${self.max_position_usd:.1f}"
            )

        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Minute, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except Exception as e:
            self.Debug(f"Warning: Could not add BTC - {e}")

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1), self.DailyReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0), self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=6)), self.ReviewPerformance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0), self.HealthCheck)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=5)), self.ResyncHoldings)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=2)), self.VerifyOrderFills)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=15)), self.PortfolioSanityCheck)

        self.SetWarmUp(timedelta(days=4))
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.Settings.FreePortfolioValuePercentage = 0.01
        self.Settings.InsightScore = False

        self._scoring_engine = MicroScalpEngine(self)

        # MG3: schedule periodic local-vs-exchange reconciliation check
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(seconds=self.mg3_reconciliation_cadence)),
            self.ReconcilePositions,
        )

        if self.LiveMode:
            if self.mg3_mode != "live":
                self.Error("MG3: mode is not 'live' but LiveMode=True. Update config.py before deploying.")
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MG3) v8.0.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")
        else:
            self.Debug(f"=== MG3 BACKTEST/PAPER mode={self.mg3_mode} | "
                       f"strict_fill={self.mg3_strict_limit_fill} | "
                       f"fee_rt={self.mg3_fee_assumption_rt:.3%} ===")

    def CustomSecurityInitializer(self, security):
        """Applies volume-aware slippage (RealisticCryptoSlippage) and the custom
        Maker/Taker fee model (0.25% for Limit orders, 0.40% for Market orders)."""
        security.SetSlippageModel(RealisticCryptoSlippage())
        security.SetFeeModel(MakerTakerFeeModel())

    # -----------------------------------------------------------------------
    # Parameter helpers, state-machine, reconciliation, scheduled callbacks,
    # universe filter, and symbol initialization
    # -----------------------------------------------------------------------

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

        # 3. Force-market exit for RECOVERING positions that have no open orders.
        # RECOVERING means the normal limit-exit path has exhausted its retries;
        # we escalate here so the position cannot be silently abandoned.
        for sym in list(self._position_states.keys()):
            if self._position_states[sym] != POSITION_STATE_RECOVERING:
                continue
            if not is_invested_not_dust(self, sym):
                # Position already gone — clean up state
                self._set_position_state(sym, POSITION_STATE_FLAT)
                self._failed_exit_counts.pop(sym, None)
                continue
            if len(self.Transactions.GetOpenOrders(sym)) > 0:
                continue  # exit order already in flight
            holding = self.Portfolio[sym]
            qty = abs(holding.Quantity)
            if qty == 0:
                self._set_position_state(sym, POSITION_STATE_FLAT)
                continue
            direction_mult = -1 if holding.Quantity > 0 else 1
            self.Debug(f"[RECONCILE] FORCE MARKET EXIT for RECOVERING {sym.Value} qty={qty:.6f}")
            try:
                self.MarketOrder(sym, qty * direction_mult, tag="Reconcile Force Exit")
                self._set_position_state(sym, POSITION_STATE_EXITING)
                self._failed_exit_counts.pop(sym, None)
            except Exception as e:
                self.Debug(f"[RECONCILE] force market exit failed for {sym.Value}: {e}")

        # Kraken preparation placeholder:
        # In a future live integration, call self._kraken_api_reconcile() here.
        # self._kraken_api_reconcile()  # TODO: implement for live Kraken deployment

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

    # -----------------------------------------------------------------------
    # OnData — per-bar data updates, market context, and trading dispatch
    # -----------------------------------------------------------------------

    def OnData(self, data):
        # === BTC reference data ===
        if self.btc_symbol and data.Bars.ContainsKey(self.btc_symbol):
            btc_bar = data.Bars[self.btc_symbol]
            btc_price = float(btc_bar.Close)
            if len(self.btc_prices) > 0:
                btc_return = (btc_price - self.btc_prices[-1]) / self.btc_prices[-1]
                self.btc_returns.append(btc_return)
            self.btc_prices.append(btc_price)
            self.btc_ema_24.Update(btc_bar.EndTime, btc_price)
            if len(self.btc_returns) >= 10:
                self.btc_volatility.append(np.std(list(self.btc_returns)[-10:]))
        for symbol in list(self.crypto_data.keys()):
            if not data.Bars.ContainsKey(symbol):
                continue
            try:
                quote_bar = data.QuoteBars[symbol] if data.QuoteBars.ContainsKey(symbol) else None
                self._update_symbol_data(symbol, data.Bars[symbol], quote_bar)
            except Exception as e:
                self.Debug(f"Error updating symbol data for {symbol.Value}: {e}")
                pass
        if self.IsWarmingUp:
            return
        if not self._positions_synced:
            if not self._first_post_warmup:
                self._cancel_stale_orders()
            sync_existing_positions(self)
            self._positions_synced = True
            self._first_post_warmup = False
            # Fallback: if status never set, assume online after warmup
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
                self.Debug("Fallback: kraken_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        self._update_market_context()
        self.Rebalance()
        self.CheckExits()

    def _update_symbol_data(self, symbol, bar, quote_bar=None):
        """Delegate per-bar symbol-data updates to mg3_data helper."""
        mg3_data.update_symbol_data(self, symbol, bar, quote_bar)

    def _update_market_context(self):
        """Delegate market-context update to mg3_data helper."""
        mg3_data.update_market_context(self)

    def _annualized_vol(self, crypto):
        if crypto is None:
            return None
        if len(crypto.get('volatility', [])) == 0:
            return None
        return float(crypto['volatility'][-1]) * self.sqrt_annualization

    def _compute_portfolio_risk_estimate(self):
        total_value = self.Portfolio.TotalPortfolioValue
        if total_value <= 0:
            return 0.0
        risk = 0.0
        for kvp in self.Portfolio:
            symbol, holding = kvp.Key, kvp.Value
            if not is_invested_not_dust(self, symbol):
                continue
            crypto = self.crypto_data.get(symbol)
            asset_vol_ann = self._annualized_vol(crypto)
            if asset_vol_ann is None:
                asset_vol_ann = self.min_asset_vol_floor
            weight = abs(holding.HoldingsValue) / total_value
            risk += weight * asset_vol_ann
        return risk

    def _normalize(self, v, mn, mx):
        if mx - mn <= 0:
            return 0.5
        return max(0, min(1, (v - mn) / (mx - mn)))

    def _calculate_factor_scores(self, symbol, crypto):
        """Evaluate signals and return scores respecting MG3 direction toggles."""
        long_score, long_components = self._scoring_engine.calculate_scalp_score(crypto)
        short_score, short_components = self._scoring_engine.calculate_short_score(crypto)

        # MG3: honour direction toggles from config
        if not self.mg3_long_enabled:
            long_score = 0.0
        if not self.mg3_short_enabled:
            short_score = 0.0

        components = long_components.copy()
        components['_scalp_score'] = long_score
        components['_direction'] = 1
        components['_long_score'] = long_score
        components['_short_score'] = short_score
        return components

    def _calculate_composite_score(self, factors, crypto=None):
        """Return the pre-computed scalp score."""
        return factors.get('_scalp_score', 0.0)

    def _apply_fee_adjustment(self, score):
        """Return score unchanged – signal thresholds already require >1% moves."""
        return score

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        """Delegates to MicroScalpEngine.calculate_position_size.

        Returns a fraction of available reserved cash (0.07–0.15), capped by
        MAX_SYMBOL_EXPOSURE_USD in config.py.  A Kelly multiplier (0.5–1.5) is
        applied on top based on recent win/loss history.
        """
        return self._scoring_engine.calculate_position_size(score, threshold, asset_vol_ann)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    def _get_max_daily_trades(self):
        return self.max_daily_trades

    def _get_threshold(self):
        return self.entry_threshold

    def _check_correlation(self, new_symbol):
        """Reject candidate if it is too correlated with any existing position (item 8)."""
        if not self.entry_prices:
            return True
        new_crypto = self.crypto_data.get(new_symbol)
        if not new_crypto or len(new_crypto['returns']) < 24:
            return True
        new_rets = np.array(list(new_crypto['returns'])[-24:])
        if np.std(new_rets) < 1e-10:
            return True
        for sym in list(self.entry_prices.keys()):
            if sym == new_symbol:
                continue
            existing = self.crypto_data.get(sym)
            if not existing or len(existing['returns']) < 24:
                continue
            exist_rets = np.array(list(existing['returns'])[-24:])
            if np.std(exist_rets) < 1e-10:
                continue
            try:
                corr = np.corrcoef(new_rets, exist_rets)[0, 1]
                if corr > 0.85:
                    return False
            except Exception:
                continue
        return True

    def _log_skip(self, reason):
        if self.LiveMode:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason
        elif reason != self._last_skip_reason:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason

    def _daily_loss_exceeded(self):
        """
        MG3: Returns True if today's portfolio loss exceeds MAX_DAILY_LOSS_PCT.
        Uses the daily opening value stored at ResetDailyCounters time.
        """
        if not hasattr(self, '_daily_open_value') or self._daily_open_value is None:
            return False
        current_value = self.Portfolio.TotalPortfolioValue
        if self._daily_open_value <= 0:
            return False
        daily_loss = (self._daily_open_value - current_value) / self._daily_open_value
        return daily_loss >= MG3Config.MAX_DAILY_LOSS_PCT

    # -----------------------------------------------------------------------
    # Rebalance and _execute_trades
    # Delegated to mg3_rebalance helper module.
    # -----------------------------------------------------------------------

    def Rebalance(self):
        """Delegate to mg3_rebalance.rebalance for entry-selection logic."""
        mg3_rebalance.rebalance(self)

    def _get_open_buy_orders_value(self):
        """Calculate total value reserved by open buy orders."""
        return get_open_buy_orders_value(self)

    def _execute_trades(self, candidates, threshold_now):
        """Delegate to mg3_rebalance.execute_trades for per-candidate order placement."""
        mg3_rebalance.execute_trades(self, candidates, threshold_now)

    # -----------------------------------------------------------------------
    # CheckExits, _check_exit, _force_market_liquidate
    # Delegated to mg3_exits helper module.
    # -----------------------------------------------------------------------

    def _is_ready(self, c):
        return len(c['prices']) >= 10 and c['rsi'].IsReady

    def CheckExits(self):
        """Delegate to mg3_exits.check_exits for all open-position exit logic."""
        mg3_exits.check_exits(self)

    def _force_market_liquidate(self, symbol):
        """Delegate to mg3_exits.force_market_liquidate for RECOVERING positions."""
        mg3_exits.force_market_liquidate(self, symbol)

    def _check_exit(self, symbol, price, holding):
        """Delegate to mg3_exits.check_exit for per-position exit condition checks."""
        mg3_exits.check_exit(self, symbol, price, holding)

    # -----------------------------------------------------------------------
    # OnOrderEvent, OnBrokerageMessage, OnEndOfAlgorithm, metrics, DailyReport
    # Delegated to mg3_reporting helper module.
    # -----------------------------------------------------------------------

    def OnOrderEvent(self, event):
        """Delegate to mg3_reporting.on_order_event."""
        mg3_reporting.on_order_event(self, event)

    def OnBrokerageMessage(self, message):
        """Delegate to mg3_reporting.on_brokerage_message."""
        mg3_reporting.on_brokerage_message(self, message)

    def OnEndOfAlgorithm(self):
        """Delegate to mg3_reporting.on_end_of_algorithm."""
        mg3_reporting.on_end_of_algorithm(self)

    def _log_mg3_metrics(self):
        """Delegate to mg3_reporting.log_mg3_metrics."""
        mg3_reporting.log_mg3_metrics(self)

    def DailyReport(self):
        """Delegate to mg3_reporting.daily_report_wrapper."""
        mg3_reporting.daily_report_wrapper(self)
