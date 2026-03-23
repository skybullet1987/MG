# region imports
from AlgorithmImports import *
from execution import (
    MNQSlippage,
    cancel_stale_new_orders,
    cleanup_position,
    daily_report,
    debug_limited,
    get_actual_position_count,
    has_open_orders,
    health_check,
    is_invested_not_dust,
    kelly_fraction,
    live_safety_checks,
    load_persisted_state,
    cleanup_object_store,
    persist_state,
    portfolio_sanity_check,
    resync_holdings_full,
    review_performance,
    slip_log,
    sync_existing_positions,
    verify_order_fills,
    normalize_order_time,
    record_exit_pnl,
)
from config import MGConfig
from candidates import (
    REJECT_ALREADY_INVESTED,
    REJECT_COOLDOWN,
    REJECT_DAILY_LOSS_EXCEEDED,
    REJECT_LOW_EXPECTED_MOVE,
    REJECT_MARGIN_INSUFFICIENT,
    REJECT_OPEN_ORDERS,
    REJECT_PORTFOLIO_CAP,
    REJECT_SCORE_TOO_LOW,
    REJECT_SYMBOL_TRADE_LIMIT,
    REJECT_VIX_EXTREME,
    REJECT_ZERO_CONTRACTS,
)
from setups import evaluate_all_setups
from reporting import DiagnosticsLogger

from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
# endregion


class MNQFeeModel(FeeModel):
    """IBKR micro futures: ~$0.32 commission + $0.30 exchange + $0.01 NFA per side."""
    FEE_PER_SIDE = 0.63

    def GetOrderFee(self, parameters):
        contracts = abs(parameters.Order.AbsoluteQuantity)
        return OrderFee(CashAmount(contracts * self.FEE_PER_SIDE, "USD"))


class MNQStrategy(QCAlgorithm):
    """
    Machine Gun v2 IB -- Research-first futures micro-scalping strategy.

    Architecture overview (v8.0.0)
    --------------------------------
    The strategy uses a setup-driven pipeline:

      1. Every bar: evaluate_all_setups() checks each active contract against
         three explicit setup families (trend_pullback, mean_reversion,
         breakout_compression) and returns structured SetupCandidates.

      2. All candidates across all symbols are collected first, then ranked by
         score descending.  Symbol iteration order does not determine which
         trades are taken.

      3. Per-candidate filters (invested, cooldown, daily cap, ATR quality gate,
         score threshold) are applied before portfolio selection.

      4. Portfolio selection respects the portfolio-wide contract cap and the
         margin floor.

      5. Orders carry setup attribution so every trade is traceable: which setup
         fired, what score, what regime, what session.

      6. DiagnosticsLogger records PnL attribution by setup / symbol / regime /
         session / exit reason, plus candidate generation and rejection counts.

    Configuration
    -------------
    All strategy parameters live in MGConfig (config.py).  The default MODE is
    "research" with fixed 1-contract sizing and Kelly disabled to keep the
    backtest diagnostic path clean.

    Operational safeguards (health checks, state persistence, bracket orders,
    resync, fill verification, portfolio sanity) are preserved from the prior
    version and run on their original schedules.
    """

    def Initialize(self):
        # Config
        self._cfg = MGConfig()
        cfg = self._cfg

        self.SetStartDate(2024, 1, 1)
        self.SetCash(cfg.START_CASH)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        # TP/SL parameters (param-overridable for grid search)
        self.quick_take_profit  = self._get_param("quick_take_profit", cfg.QUICK_TAKE_PROFIT)
        self.tight_stop_loss    = self._get_param("tight_stop_loss",   cfg.TIGHT_STOP_LOSS)
        self.atr_tp_mult        = self._get_param("atr_tp_mult",  cfg.ATR_TP_MULT)
        self.atr_sl_mult        = self._get_param("atr_sl_mult",  cfg.ATR_SL_MULT)
        self.trail_activation   = self._get_param("trail_activation",  0.005)
        self.trail_stop_pct     = self._get_param("trail_stop_pct",    0.002)
        self.time_stop_hours    = self._get_param("time_stop_hours",   cfg.TIME_STOP_HOURS)
        self.stale_position_hours = cfg.STALE_POSITION_HOURS

        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 3.5

        # Position limits
        self.max_contracts           = cfg.MAX_CONTRACTS_PER_SYMBOL
        self.max_portfolio_contracts = cfg.MAX_PORTFOLIO_CONTRACTS
        self.base_max_positions      = 3
        self.max_positions           = 3
        self.position_size_pct       = 0.50
        self.invalid_entry_cooldown_minutes = cfg.INVALID_ENTRY_COOLDOWN_MINUTES

        # Vol-targeting (research mode: disabled)
        self.target_position_ann_vol = cfg.TARGET_POSITION_ANN_VOL
        self.portfolio_vol_cap       = 0.80
        self.min_asset_vol_floor     = 0.05

        # Indicator lookback periods
        self.ultra_short_period = cfg.ULTRA_SHORT_PERIOD
        self.short_period       = cfg.SHORT_PERIOD
        self.medium_period      = cfg.MEDIUM_PERIOD
        self.lookback           = cfg.LOOKBACK
        self.sqrt_annualization = np.sqrt(60 * 24 * 252)
        self.adx_min_period     = 10

        # Fee parameters
        self.expected_round_trip_fees = 1.26
        self.fee_slippage_buffer      = 0.50
        self.min_expected_profit_pct  = cfg.MIN_EXPECTED_PROFIT_PCT

        # Trade limits
        self.skip_hours_utc            = []
        self.max_daily_trades          = cfg.MAX_DAILY_TRADES
        self.daily_trade_count         = 0
        self.last_trade_date           = None
        self.exit_cooldown_hours       = cfg.EXIT_COOLDOWN_HOURS
        self.cancel_cooldown_minutes   = cfg.CANCEL_COOLDOWN_MINUTES
        self.max_symbol_trades_per_day = cfg.MAX_SYMBOL_TRADES_PER_DAY
        self.max_drawdown_limit        = 0.20

        # Timing / order-management parameters
        self.stale_order_timeout_seconds        = 30
        self.live_stale_order_timeout_seconds   = 60
        self.max_concurrent_open_orders         = 2
        self.open_orders_cash_threshold         = 0.90
        self.order_fill_check_threshold_seconds = 60
        self.order_timeout_seconds              = 30
        self.resync_log_interval_seconds        = 1800
        self.portfolio_mismatch_threshold       = 0.10
        self.portfolio_mismatch_min_dollars     = 5.00
        self.portfolio_mismatch_cooldown_seconds = 3600
        self.retry_pending_cooldown_seconds     = 60
        self.rate_limit_cooldown_minutes        = 10

        # Internal state
        self._positions_synced       = False
        self._session_blacklist      = set()
        self._symbol_entry_cooldowns = {}
        self._spread_warning_times   = {}
        self._first_post_warmup      = True
        self._submitted_orders       = {}
        self._symbol_slippage_history = {}
        self._order_retries          = {}
        self._retry_pending          = {}
        self._rate_limit_until       = None
        self._last_mismatch_warning  = None
        self._daily_open_value       = None

        self.trade_count      = 0
        self._pending_orders  = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns  = {}
        self._symbol_loss_cooldowns = {}
        self._slip_abs        = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold   = 0.001
        self.slip_outlier_threshold = 0.002
        self.slip_alert_duration_hours = 2
        self._recent_tickets  = deque(maxlen=25)

        # Bracket order tracking
        # entry_order_id -> {direction, contracts, candidate_ref}
        self._pending_entry_info = {}
        # symbol -> {tp_id, sl_id, direction, entry_price, setup attribution, ...}
        self._bracket_orders     = {}

        # Rolling performance stats (used only in "live" mode for Kelly)
        self._rolling_wins       = deque(maxlen=50)
        self._rolling_win_sizes  = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None

        # Regime state
        self.market_regime     = "unknown"
        self.volatility_regime = "normal"
        self._regime_hold_count = 0

        # Legacy counters (kept for backward compat / daily_report / review_performance)
        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.trade_log      = []
        self.pnl_by_tag     = {}
        self.log_budget     = 0
        self.last_log_time  = None
        self._last_skip_reason = None

        # Futures instruments
        mnq_future = self.AddFuture(Futures.Indices.MicroNASDAQ100EMini, Resolution.Minute)
        mnq_future.SetFilter(timedelta(days=0), timedelta(days=90))
        self.mnq_base_symbol = mnq_future.Symbol

        m2k_future = self.AddFuture(Futures.Indices.MicroRussell2000EMini, Resolution.Minute)
        m2k_future.SetFilter(timedelta(days=0), timedelta(days=90))
        self.m2k_base_symbol = m2k_future.Symbol

        mgc_future = self.AddFuture(Futures.Metals.MicroGold, Resolution.Minute)
        mgc_future.SetFilter(timedelta(days=0), timedelta(days=90))
        self.mgc_base_symbol = mgc_future.Symbol

        self.base_symbols     = [self.mnq_base_symbol, self.m2k_base_symbol, self.mgc_base_symbol]
        self.active_contracts = {}
        self.instrument_data  = {}

        # VIX as regime overlay
        self.vix_symbol = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        self.vix_value  = 20.0

        # Diagnostics logger
        self.diagnostics = DiagnosticsLogger(self)

        # Scheduled tasks
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

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MNQ/M2K/MGC MICRO-SCALP) v8.0.0 ===")
            self.Debug("Capital: ${:.2f} | Mode: {}".format(self.Portfolio.Cash, self._cfg.MODE))

    def CustomSecurityInitializer(self, security):
        security.SetSlippageModel(MNQSlippage())
        security.SetFeeModel(MNQFeeModel())

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
        except Exception:
            pass
        return default

    def _normalize_order_time(self, order_time):
        return normalize_order_time(order_time)

    def _record_exit_pnl(self, symbol, entry_price, exit_price, exit_tag="Unknown"):
        return record_exit_pnl(self, symbol, entry_price, exit_price, exit_tag=exit_tag)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    # -------------------------------------------------------------------------
    # Scheduled task callbacks
    # -------------------------------------------------------------------------

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date   = self.Time.date()
        self._daily_open_value = self.Portfolio.TotalPortfolioValue
        for data in self.instrument_data.values():
            data["trade_count_today"] = 0
        self._symbol_entry_cooldowns.clear()
        persist_state(self)

    def HealthCheck(self):
        if self.IsWarmingUp:
            return
        health_check(self)

    def ResyncHoldings(self):
        if self.IsWarmingUp:
            return
        if not self.LiveMode:
            return
        resync_holdings_full(self)

    def VerifyOrderFills(self):
        if self.IsWarmingUp:
            return
        verify_order_fills(self)

    def PortfolioSanityCheck(self):
        if self.IsWarmingUp:
            return
        portfolio_sanity_check(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10:
            return
        review_performance(self)

    def DailyReport(self):
        if self.IsWarmingUp:
            return
        self.diagnostics.print_daily_summary()
        daily_report(self)

    def _cancel_stale_orders(self):
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                self.Debug("Canceling {} stale orders on startup".format(len(open_orders)))
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug("Error canceling stale orders: {}".format(e))

    # -------------------------------------------------------------------------
    # Instrument data factory
    # -------------------------------------------------------------------------

    def _make_instrument_data(self):
        """Create and return a fresh indicator/data structure for one futures contract."""
        return {
            "prices":           deque(maxlen=self.lookback),
            "returns":          deque(maxlen=self.lookback),
            "volume":           deque(maxlen=self.lookback),
            "volume_ma":        deque(maxlen=self.medium_period),
            "ema_ultra_short":  ExponentialMovingAverage(self.ultra_short_period),
            "ema_short":        ExponentialMovingAverage(self.short_period),
            "ema_medium":       ExponentialMovingAverage(self.medium_period),
            "ema_5":            ExponentialMovingAverage(5),
            "atr":              AverageTrueRange(14),
            "adx":              AverageDirectionalIndex(self.adx_min_period),
            "volatility":       deque(maxlen=self.medium_period),
            "rsi":              RelativeStrengthIndex(7),
            "zscore":           deque(maxlen=self.short_period),
            "last_price":       0,
            "recent_net_scores": deque(maxlen=3),
            "trail_stop":       None,
            "highs":            deque(maxlen=self.lookback),
            "lows":             deque(maxlen=self.lookback),
            "bb_upper":         deque(maxlen=self.short_period),
            "bb_lower":         deque(maxlen=self.short_period),
            "bb_width":         deque(maxlen=self.medium_period),
            "trade_count_today": 0,
            "last_loss_time":   None,
            "vwap_pv":          deque(maxlen=20),
            "vwap_v":           deque(maxlen=20),
            "vwap":             0.0,
            "volume_long":      deque(maxlen=1440),
            "vwap_sd":          0.0,
            "vwap_sd2_lower":   0.0,
            "vwap_sd3_lower":   0.0,
            "cvd":              deque(maxlen=self.lookback),
            "ker":              deque(maxlen=self.short_period),
            "kalman_estimate":  0.0,
            "kalman_error_cov": 1.0,
        }

    # -------------------------------------------------------------------------
    # Session helpers
    # -------------------------------------------------------------------------

    def _is_rth(self):
        """Regular Trading Hours: 8:30 AM - 3:00 PM CT (13:30-20:00 UTC)."""
        utc_min = self.Time.hour * 60 + self.Time.minute
        return 13 * 60 + 30 <= utc_min < 20 * 60

    def _is_extended_hours(self):
        return not self._is_rth()

    # -------------------------------------------------------------------------
    # OnData
    # -------------------------------------------------------------------------

    def OnData(self, data):
        # Contract rollover: maintain front-month symbol per base instrument
        for chain in data.FutureChains:
            base_sym = chain.Key
            if base_sym not in self.base_symbols:
                continue
            contracts = sorted(chain.Value, key=lambda c: c.Expiry)
            if contracts:
                front = contracts[0]
                current_active = self.active_contracts.get(base_sym)
                if current_active != front.Symbol:
                    if current_active and self.Portfolio[current_active].Invested:
                        self.Liquidate(current_active, "Contract rollover")
                    if current_active is not None:
                        self.instrument_data.pop(current_active, None)
                    self.active_contracts[base_sym] = front.Symbol
                    self.instrument_data[front.Symbol] = self._make_instrument_data()
                    self.Debug("Active contract: {} expiry {}".format(front.Symbol.Value, front.Expiry))

        if self.IsWarmingUp or not self.active_contracts:
            return

        for base_sym, contract in self.active_contracts.items():
            if contract not in self.instrument_data:
                self.instrument_data[contract] = self._make_instrument_data()
            if data.Bars.ContainsKey(contract):
                self._update_symbol_data(contract, data.Bars[contract])

        if data.ContainsKey(self.vix_symbol):
            vix = data[self.vix_symbol]
            if vix is not None:
                self.vix_value = vix.Value

        if not self._positions_synced:
            if not self._first_post_warmup:
                self._cancel_stale_orders()
            sync_existing_positions(self)
            self._positions_synced  = True
            self._first_post_warmup = False
            for base_sym, contract in self.active_contracts.items():
                d     = self.instrument_data.get(contract)
                ready = self._is_ready(d) if d else False
                self.Debug("Post-warmup: {} data {}".format(contract.Value, "ready" if ready else "warming"))

        self._update_market_context()
        self.Rebalance()

    # -------------------------------------------------------------------------
    # Indicator update
    # -------------------------------------------------------------------------

    def _update_symbol_data(self, symbol, bar):
        """Update all indicator state for one symbol from the latest bar."""
        mnq = self.instrument_data.get(symbol)
        if mnq is None:
            return

        price  = float(bar.Close)
        high   = float(bar.High)
        low    = float(bar.Low)
        volume = float(bar.Volume)

        mnq["prices"].append(price)
        mnq["highs"].append(high)
        mnq["lows"].append(low)
        if mnq["last_price"] > 0:
            mnq["returns"].append((price - mnq["last_price"]) / mnq["last_price"])
        mnq["last_price"] = price

        mnq["volume"].append(volume)
        if len(mnq["volume"]) >= self.short_period:
            mnq["volume_ma"].append(np.mean(list(mnq["volume"])[-self.short_period:]))

        mnq["ema_ultra_short"].Update(bar.EndTime, price)
        mnq["ema_short"].Update(bar.EndTime, price)
        mnq["ema_medium"].Update(bar.EndTime, price)
        mnq["ema_5"].Update(bar.EndTime, price)
        mnq["atr"].Update(bar)
        mnq["adx"].Update(bar)

        mnq["vwap_pv"].append(price * volume)
        mnq["vwap_v"].append(volume)
        total_v = sum(mnq["vwap_v"])
        if total_v > 0:
            mnq["vwap"] = sum(mnq["vwap_pv"]) / total_v
        mnq["volume_long"].append(volume)

        if len(mnq["vwap_v"]) >= 5 and mnq["vwap"] > 0:
            pv_list    = list(mnq["vwap_pv"])
            v_list     = list(mnq["vwap_v"])
            bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
            if len(bar_prices) >= 5:
                sd = float(np.std(bar_prices))
                mnq["vwap_sd"]        = sd
                mnq["vwap_sd2_lower"] = mnq["vwap"] - 2.0 * sd
                mnq["vwap_sd3_lower"] = mnq["vwap"] - 3.0 * sd

        if len(mnq["returns"]) >= 10:
            mnq["volatility"].append(np.std(list(mnq["returns"])[-10:]))

        mnq["rsi"].Update(bar.EndTime, price)

        if len(mnq["prices"]) >= self.medium_period:
            prices_arr = np.array(list(mnq["prices"])[-self.medium_period:])
            std  = np.std(prices_arr)
            mean = np.mean(prices_arr)
            if std > 0:
                mnq["zscore"].append((price - mean) / std)
                mnq["bb_upper"].append(mean + 2 * std)
                mnq["bb_lower"].append(mean - 2 * std)
                mnq["bb_width"].append(4 * std / mean if mean > 0 else 0)

        high_low  = high - low
        bar_delta = (volume * ((price - low) - (high - price)) / high_low
                     if high_low > 0 else 0.0)
        prev_cvd = mnq["cvd"][-1] if len(mnq["cvd"]) > 0 else 0.0
        mnq["cvd"].append(prev_cvd + bar_delta)

        if len(mnq["prices"]) >= 15:
            price_change   = abs(mnq["prices"][-1] - mnq["prices"][-15])
            volatility_sum = sum(abs(mnq["prices"][i] - mnq["prices"][i - 1]) for i in range(-14, 0))
            mnq["ker"].append(price_change / volatility_sum if volatility_sum > 0 else 0.0)

        Q = 1e-5
        R = 0.01
        if mnq["kalman_estimate"] == 0.0:
            mnq["kalman_estimate"] = price
        pred  = mnq["kalman_estimate"]
        ep    = mnq["kalman_error_cov"] + Q
        gain  = ep / (ep + R)
        mnq["kalman_estimate"]  = pred + gain * (price - pred)
        mnq["kalman_error_cov"] = (1 - gain) * ep

    # -------------------------------------------------------------------------
    # Regime detection
    # -------------------------------------------------------------------------

    def _update_market_context(self):
        """
        VIX-based regime detection with hysteresis.
        Requires 3 consecutive bars at the new regime level before switching,
        preventing rapid flickering near boundary values.
        """
        vix = self.vix_value
        if vix < 15:
            new_regime, new_vol = "bull",    "low"
        elif vix < 25:
            new_regime, new_vol = "sideways", "normal"
        elif vix < 35:
            new_regime, new_vol = "bear",    "high"
        else:
            new_regime, new_vol = "bear",    "extreme"

        if new_regime != self.market_regime:
            self._regime_hold_count += 1
            if self._regime_hold_count >= 3:
                self.market_regime      = new_regime
                self._regime_hold_count = 0
        else:
            self._regime_hold_count = 0

        self.volatility_regime = new_vol

    # -------------------------------------------------------------------------
    # Readiness check
    # -------------------------------------------------------------------------

    def _is_ready(self, mnq):
        if mnq is None:
            return False
        return len(mnq["prices"]) >= 10 and mnq["rsi"].IsReady

    # -------------------------------------------------------------------------
    # Helpers used by execution.py
    # -------------------------------------------------------------------------

    def _annualized_vol(self, mnq):
        if mnq is None or len(mnq.get("volatility", [])) == 0:
            return None
        return float(mnq["volatility"][-1]) * self.sqrt_annualization

    def _normalize(self, v, mn, mx):
        if mx - mn <= 0:
            return 0.5
        return max(0.0, min(1.0, (v - mn) / (mx - mn)))

    def _daily_loss_exceeded(self):
        """True when portfolio has dropped >= DAILY_LOSS_LIMIT_PCT from today open."""
        if self._daily_open_value is None or self._daily_open_value <= 0:
            return False
        current = self.Portfolio.TotalPortfolioValue
        if current <= 0:
            return True
        return (self._daily_open_value - current) / self._daily_open_value >= self._cfg.DAILY_LOSS_LIMIT_PCT

    def _log_skip(self, reason):
        if self.LiveMode:
            debug_limited(self, "Rebalance skip: {}".format(reason))
            self._last_skip_reason = reason
        elif reason != self._last_skip_reason:
            debug_limited(self, "Rebalance skip: {}".format(reason))
            self._last_skip_reason = reason

    # -------------------------------------------------------------------------
    # Rebalance -- setup-driven candidate ranking pipeline
    # -------------------------------------------------------------------------

    def Rebalance(self):
        """
        Entry logic: evaluate -> rank -> select -> order.

        1. Evaluate all active contracts against all enabled setup families.
           All candidates are collected before any selection decision is made.
        2. Apply per-candidate filters (already invested, cooldown, score,
           ATR quality gate, symbol daily cap).
        3. Rank valid candidates by score descending.
           Symbol iteration order has no influence on which trades are taken.
        4. Select best candidates within portfolio / margin constraints.
        5. Place market orders with full setup attribution in the order tag.
        """
        if self.IsWarmingUp or not self.active_contracts:
            return

        if self._cfg.RTH_ONLY and not self._is_rth():
            return

        if self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded")
            self.diagnostics.record_rejection(REJECT_DAILY_LOSS_EXCEEDED)
            return

        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            self._log_skip("rate limited")
            return

        if self.LiveMode and not live_safety_checks(self):
            return

        cancel_stale_new_orders(self)

        if self.daily_trade_count >= self._cfg.MAX_DAILY_TRADES:
            self._log_skip("max daily trades")
            return

        if self.vix_value > self._cfg.VIX_EXTREME_THRESHOLD:
            self._log_skip("VIX extreme ({:.1f}) - blocking all entries".format(self.vix_value))
            self.diagnostics.record_rejection(REJECT_VIX_EXTREME)
            return

        session     = "rth" if self._is_rth() else "eth"
        self.log_budget = 20

        # Step 1: Evaluate all active contracts
        all_candidates = []
        for base_sym, active_contract in list(self.active_contracts.items()):
            mnq = self.instrument_data.get(active_contract)
            if mnq is None or not self._is_ready(mnq):
                continue
            candidates = evaluate_all_setups(
                active_contract, mnq,
                self.market_regime, session, self.vix_value,
                self._cfg,
            )
            for c in candidates:
                self.diagnostics.record_candidate_generated(c)
            all_candidates.extend(candidates)

        if not all_candidates:
            return

        # Step 2: Filter candidates
        valid_candidates = []
        for c in all_candidates:
            reject = self._check_candidate_filters(c)
            if reject:
                self.diagnostics.record_candidate_rejected(c, reject)
            else:
                valid_candidates.append(c)

        if not valid_candidates:
            return

        # Step 3: Rank by score descending
        ranked = sorted(valid_candidates, key=lambda c: c.score, reverse=True)

        # Step 4: Select candidates within constraints
        total_open       = get_actual_position_count(self)
        orders_this_tick = 0
        used_symbols     = set()

        for candidate in ranked:
            if total_open + orders_this_tick >= self._cfg.MAX_PORTFOLIO_CONTRACTS:
                self._log_skip("portfolio cap ({}) reached".format(self._cfg.MAX_PORTFOLIO_CONTRACTS))
                self.diagnostics.record_candidate_rejected(candidate, REJECT_PORTFOLIO_CAP)
                continue

            sym = candidate.symbol
            if sym in used_symbols:
                continue

            if self.Portfolio.MarginRemaining <= 500:
                self._log_skip("insufficient margin")
                self.diagnostics.record_candidate_rejected(candidate, REJECT_MARGIN_INSUFFICIENT)
                break

            contracts = self._size_candidate(candidate)
            if contracts < 1:
                self.diagnostics.record_candidate_rejected(candidate, REJECT_ZERO_CONTRACTS)
                continue

            try:
                dir_str   = "LONG" if candidate.direction == 1 else "SHORT"
                order_qty = contracts * candidate.direction
                price     = self.Securities[sym].Price
                tag       = "MG [{}] {} score={:.3f} thresh={:.3f}".format(
                    candidate.setup_type, dir_str,
                    candidate.score, candidate.threshold)
                ticket = self.MarketOrder(sym, order_qty, tag=tag)
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    self._pending_entry_info[ticket.OrderId] = {
                        "direction":     candidate.direction,
                        "contracts":     contracts,
                        "candidate_ref": candidate,
                    }
                    self.diagnostics.record_candidate_accepted(candidate, ticket.OrderId)
                    orders_this_tick += 1
                    used_symbols.add(sym)
                    mnq = self.instrument_data.get(sym, {})
                    mnq["trade_count_today"] = mnq.get("trade_count_today", 0) + 1
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
                    self.Debug("ENTRY [{}] {}: {} {}c @ ${:.2f} | score={:.3f} | {}".format(
                        candidate.setup_type, dir_str, sym.Value,
                        contracts, price, candidate.score, candidate.components_str()))
            except Exception as e:
                self.Debug("ORDER FAILED: {} - {}".format(sym.Value, e))

        self._last_skip_reason = None

    # -------------------------------------------------------------------------
    # Per-candidate pre-order filters
    # -------------------------------------------------------------------------

    def _check_candidate_filters(self, candidate):
        """
        Apply per-candidate filters before ranking.
        Returns a rejection-reason constant if the candidate should be skipped,
        or None if it passes all filters.
        """
        sym      = candidate.symbol
        sym_name = candidate.symbol_name
        mnq      = self.instrument_data.get(sym, {})

        if is_invested_not_dust(self, sym):
            return REJECT_ALREADY_INVESTED

        if has_open_orders(self, sym):
            return REJECT_OPEN_ORDERS

        if (sym_name in self._symbol_entry_cooldowns
                and self.Time < self._symbol_entry_cooldowns[sym_name]):
            return REJECT_COOLDOWN

        if mnq.get("trade_count_today", 0) >= self._cfg.MAX_SYMBOL_TRADES_PER_DAY:
            return REJECT_SYMBOL_TRADE_LIMIT

        if not candidate.is_valid:
            return REJECT_SCORE_TOO_LOW

        atr_ind = mnq.get("atr")
        if atr_ind is not None and atr_ind.IsReady and sym in self.Securities:
            price = self.Securities[sym].Price
            if price > 0:
                expected_move = (atr_ind.Current.Value * self._cfg.ATR_TP_MULT) / price
                if expected_move < self._cfg.MIN_EXPECTED_PROFIT_PCT:
                    return REJECT_LOW_EXPECTED_MOVE

        return None

    # -------------------------------------------------------------------------
    # Position sizing
    # -------------------------------------------------------------------------

    def _size_candidate(self, candidate):
        """
        Determine number of contracts for a candidate.

        Research mode (default): fixed DEFAULT_CONTRACTS (1), Kelly off,
        vol-targeting off.  Keeps backtest path clean for entry-logic attribution.

        Live mode: optional Kelly and vol-targeting enabled via MGConfig.
        """
        cfg       = self._cfg
        contracts = cfg.DEFAULT_CONTRACTS

        if cfg.MODE != "research":
            if cfg.VOL_TARGETING_ENABLED:
                mnq = self.instrument_data.get(candidate.symbol, {})
                vol = self._annualized_vol(mnq)
                if vol is not None and vol > 0:
                    vol_scalar = min(1.0, cfg.TARGET_POSITION_ANN_VOL / vol)
                    if vol_scalar < cfg.VOL_TARGETING_REDUCTION_THRESHOLD:
                        contracts = max(0, contracts - 1)

            if cfg.KELLY_ENABLED:
                if kelly_fraction(self) < 0.6:
                    contracts = max(0, contracts - 1)

        # Graduated VIX reduction (defaults to 0 in research mode = no effect)
        if self.vix_value > cfg.VIX_HIGH_THRESHOLD:
            contracts = max(0, contracts - cfg.VIX_HIGH_SIZE_REDUCTION)
        elif self.vix_value > cfg.VIX_ELEVATED_THRESHOLD:
            contracts = max(0, contracts - cfg.VIX_ELEVATED_SIZE_REDUCTION)

        available_margin = self.Portfolio.MarginRemaining
        max_by_margin    = int(available_margin / cfg.MARGIN_PER_CONTRACT) if cfg.MARGIN_PER_CONTRACT > 0 else 0
        contracts = min(contracts, max_by_margin, cfg.MAX_CONTRACTS_PER_SYMBOL, cfg.MAX_PORTFOLIO_CONTRACTS)

        return max(0, contracts)

    # -------------------------------------------------------------------------
    # Bracket order management
    # -------------------------------------------------------------------------

    def _submit_bracket_orders(self, symbol, fill_price, direction, contracts,
                               entry_order_id=None, candidate=None):
        """
        Submit TP (LimitOrder) and SL (StopMarketOrder) bracket legs.
        Attribution metadata from the SetupCandidate is stored in the bracket
        record so OnOrderEvent can log it on exit.
        """
        mnq     = self.instrument_data.get(symbol)
        atr_ind = mnq["atr"] if mnq else None
        atr_val = atr_ind.Current.Value if atr_ind is not None and atr_ind.IsReady else None

        if atr_val and fill_price > 0:
            sl_pct = max((atr_val * self.atr_sl_mult) / fill_price, self.tight_stop_loss)
            tp_pct = max((atr_val * self.atr_tp_mult) / fill_price, self.quick_take_profit)
        else:
            sl_pct = self.tight_stop_loss
            tp_pct = self.quick_take_profit

        if tp_pct < sl_pct * self._cfg.MIN_TP_SL_RATIO:
            tp_pct = sl_pct * self._cfg.MIN_TP_SL_RATIO

        exit_qty = -contracts * direction
        if direction == 1:
            tp_price = fill_price * (1 + tp_pct)
            sl_price = fill_price * (1 - sl_pct)
        else:
            tp_price = fill_price * (1 - tp_pct)
            sl_price = fill_price * (1 + sl_pct)

        tp_ticket = self.LimitOrder(symbol, exit_qty, tp_price, tag="Take Profit")
        sl_ticket = self.StopMarketOrder(symbol, exit_qty, sl_price, tag="Stop Loss")

        if tp_ticket is not None and sl_ticket is not None:
            bracket_meta = {
                "tp_id":          tp_ticket.OrderId,
                "sl_id":          sl_ticket.OrderId,
                "direction":      direction,
                "entry_price":    fill_price,
                "entry_time":     self.Time,
                "entry_order_id": entry_order_id,
            }
            if candidate is not None:
                bracket_meta.update({
                    "setup_type":  candidate.setup_type,
                    "score":       candidate.score,
                    "threshold":   candidate.threshold,
                    "long_score":  candidate.long_score,
                    "short_score": candidate.short_score,
                    "components":  dict(candidate.components),
                    "regime":      candidate.regime,
                    "session":     candidate.session,
                    "vix":         candidate.vix,
                    "symbol_name": candidate.symbol_name,
                })
            self._bracket_orders[symbol] = bracket_meta
            dir_str = "LONG" if direction == 1 else "SHORT"
            self.Debug("BRACKET [{}] {} ({}) | TP=${:.2f} SL=${:.2f}".format(
                bracket_meta.get("setup_type", "?"), symbol.Value, dir_str,
                tp_price, sl_price))
        else:
            self.Debug("BRACKET FAILED for {} - tp={} sl={}".format(symbol.Value, tp_ticket, sl_ticket))

    # -------------------------------------------------------------------------
    # OnOrderEvent
    # -------------------------------------------------------------------------

    def OnOrderEvent(self, event):
        try:
            symbol = event.Symbol
            self.Debug("ORDER: {} {} {} qty={} price={} id={}".format(
                symbol.Value, event.Status, event.Direction,
                event.FillQuantity or event.Quantity,
                event.FillPrice, event.OrderId))

            if event.Status == OrderStatus.Submitted:
                if symbol not in self._pending_orders:
                    self._pending_orders[symbol] = 0
                intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
                self._pending_orders[symbol] += intended_qty
                if symbol not in self._submitted_orders:
                    has_position    = symbol in self.Portfolio and self.Portfolio[symbol].Invested
                    inferred_intent = "exit" if has_position else "entry"
                    self._submitted_orders[symbol] = {
                        "order_id": event.OrderId,
                        "time":     self.Time,
                        "quantity": event.Quantity,
                        "intent":   inferred_intent,
                    }
                else:
                    self._submitted_orders[symbol]["order_id"] = event.OrderId

            elif event.Status == OrderStatus.PartiallyFilled:
                if symbol in self._pending_orders:
                    self._pending_orders[symbol] -= abs(event.FillQuantity)
                    if self._pending_orders[symbol] <= 0:
                        self._pending_orders.pop(symbol, None)
                slip_log(self, symbol, event.Direction, event.FillPrice)

            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)

                entry_info = self._pending_entry_info.pop(event.OrderId, None)
                if entry_info is not None:
                    # Entry fill: attach bracket orders
                    self.daily_trade_count += 1
                    candidate = entry_info.get("candidate_ref")
                    self.diagnostics.note_entry_filled(event.OrderId, symbol, event.FillPrice)
                    self._submit_bracket_orders(
                        symbol,
                        event.FillPrice,
                        entry_info["direction"],
                        entry_info["contracts"],
                        entry_order_id=event.OrderId,
                        candidate=candidate,
                    )
                else:
                    # Bracket fill: cancel the other leg and record attributed PnL
                    bracket = self._bracket_orders.get(symbol)
                    if bracket is not None:
                        order    = self.Transactions.GetOrderById(event.OrderId)
                        exit_tag = order.Tag if order and order.Tag else "Unknown"

                        if event.OrderId == bracket["tp_id"]:
                            try:
                                self.Transactions.CancelOrder(bracket["sl_id"])
                            except Exception as ce:
                                self.Debug("Cancel SL error {}: {}".format(symbol.Value, ce))
                        elif event.OrderId == bracket["sl_id"]:
                            try:
                                self.Transactions.CancelOrder(bracket["tp_id"])
                            except Exception as ce:
                                self.Debug("Cancel TP error {}: {}".format(symbol.Value, ce))

                        entry_price    = bracket["entry_price"]
                        direction      = bracket["direction"]
                        entry_order_id = bracket.get("entry_order_id")
                        entry_time     = bracket.get("entry_time")

                        # Rich attribution record
                        self.diagnostics.record_exit(
                            symbol=symbol,
                            entry_order_id=entry_order_id,
                            entry_price=entry_price,
                            exit_price=event.FillPrice,
                            exit_reason=exit_tag,
                            entry_time=entry_time,
                        )

                        # Legacy counters
                        if direction == 1:
                            pnl = (event.FillPrice - entry_price) / entry_price if entry_price > 0 else 0
                        else:
                            pnl = (entry_price - event.FillPrice) / entry_price if entry_price > 0 else 0

                        self._rolling_wins.append(1 if pnl > 0 else 0)
                        if pnl > 0:
                            self._rolling_win_sizes.append(pnl)
                            self.winning_trades += 1
                        else:
                            self._rolling_loss_sizes.append(abs(pnl))
                            self.losing_trades += 1
                        self.total_pnl += pnl
                        if exit_tag not in self.pnl_by_tag:
                            self.pnl_by_tag[exit_tag] = []
                        self.pnl_by_tag[exit_tag].append(pnl)
                        self.trade_log.append({
                            "time":        self.Time,
                            "symbol":      symbol.Value,
                            "pnl_pct":     pnl,
                            "exit_reason": exit_tag,
                        })

                        setup_tag = bracket.get("setup_type", "?")
                        self.Debug("EXIT [{}] ({}): {} | entry=${:.2f} exit=${:.2f} PnL:{:+.2%}".format(
                            setup_tag, exit_tag, symbol.Value,
                            entry_price, event.FillPrice, pnl))
                        self._bracket_orders.pop(symbol, None)
                        cleanup_position(self, symbol)

                slip_log(self, symbol, event.Direction, event.FillPrice)

            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                self._pending_entry_info.pop(event.OrderId, None)

            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                self._pending_entry_info.pop(event.OrderId, None)
                msg = getattr(event, "Message", "") or ""
                self.Debug("INVALID ORDER: {} dir={} qty={} | {}".format(
                    symbol.Value, event.Direction, event.Quantity, msg or "unknown reason"))
                if not is_invested_not_dust(self, symbol):
                    cooldown_until = self.Time + timedelta(minutes=self._cfg.INVALID_ENTRY_COOLDOWN_MINUTES)
                    self._symbol_entry_cooldowns[symbol.Value] = cooldown_until
                    self.Debug("Entry cooldown: {} until {}".format(symbol.Value, cooldown_until))

        except Exception as e:
            self.Debug("OnOrderEvent error: {}".format(e))

        if self.LiveMode:
            persist_state(self)

    # -------------------------------------------------------------------------
    # End of algorithm
    # -------------------------------------------------------------------------

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr    = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL ===")
        self.Debug("Trades: {} | WR: {:.1%}".format(self.trade_count, wr))
        self.Debug("Final: ${:.2f}".format(self.Portfolio.TotalPortfolioValue))
        self.Debug("PnL: {:+.2%}".format(self.total_pnl))
        self.diagnostics.print_summary()
        persist_state(self)
