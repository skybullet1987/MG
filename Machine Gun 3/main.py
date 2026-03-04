# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
import config as MG3Config
# from QuantConnect.Orders.Slippage import SlippageModel

# Modular mixins – each module owns one concern of SimplifiedCryptoStrategy
from app import AppMixin, POSITION_STATE_FLAT, POSITION_STATE_ENTERING, POSITION_STATE_OPEN, POSITION_STATE_EXITING
from data_layer import DataLayerMixin
from orchestration import OrchestrationMixin
from exit_handler import ExitHandlerMixin
from reporting import ReportingMixin
# endregion


class MakerTakerFeeModel(FeeModel):
    """Custom Fee Model: 0.25% Maker (Limit), 0.40% Taker (Market).
    Note: assumes all Limit orders rest in the book as maker orders. Aggressive
    limit orders that cross the spread will be charged the taker rate by the exchange
    at settlement, but this model applies the maker rate as an approximation for
    backtesting purposes."""
    def GetOrderFee(self, parameters):
        order = parameters.Order
        fee_pct = 0.0025 if order.Type == OrderType.Limit else 0.0040
        trade_value = order.AbsoluteQuantity * parameters.Security.Price
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


class SimplifiedCryptoStrategy(
    AppMixin, DataLayerMixin, OrchestrationMixin, ExitHandlerMixin, ReportingMixin,
    QCAlgorithm,
):
    """
    Machine Gun 3 - Micro-Scalping System v8.0.0
    =============================================
    Built on top of Machine Gun 2 (v7.1.0) with the following additions:

    * Centralized config (config.py) – all parameters in one place.
    * Position lifecycle state-machine (flat/entering/open/exiting).
    * Reconciliation hooks for local-vs-exchange state validation (Kraken prep).
    * Backtest realism knobs: strict limit fill, explicit fee/slippage settings.
    * Backtest metrics: cancel-to-fill ratio, invalid order count, PnL by exit tag.
    * Strategy toggles: long enabled, short disabled by default.
    * Conservative safe defaults – live trading must be explicitly enabled.

    Module layout (post file-size refactor)
    ----------------------------------------
    main.py          – this file; thin entrypoint, Initialize, CustomSecurityInitializer
    app.py           – bootstrap/wiring helpers (AppMixin)
    data_layer.py    – per-bar data ingestion and scoring pipeline (DataLayerMixin)
    orchestration.py – Rebalance and _execute_trades entry logic (OrchestrationMixin)
    exit_handler.py  – CheckExits and _check_exit exit logic (ExitHandlerMixin)
    reporting.py     – OnOrderEvent, OnEndOfAlgorithm, metrics (ReportingMixin)
    config_loader.py – config helpers and mode validation
    run_backtest.py  – backtest flow documentation and helpers
    run_paper.py     – paper/live-sim flow documentation and helpers
    execution.py     – shared order-execution utilities
    scoring.py       – MicroScalpEngine signal calculations
    config.py        – all tunable parameters
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(19)
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
        self.mg3_cancel_count  = 0   # total orders cancelled
        self.mg3_fill_count    = 0   # total orders filled
        self.mg3_invalid_count = 0   # total invalid orders received
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

        self.position_size_pct  = 0.70
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
        self.min_dollar_volume_usd  = 50000
        self.min_volume_usd         = 15000000

        self.skip_hours_utc         = []
        self.max_daily_trades       = 24
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
        self.open_orders_cash_threshold       = 0.5
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
