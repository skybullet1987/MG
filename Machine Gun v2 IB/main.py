# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
# endregion


class MNQFeeModel(FeeModel):
    """IBKR micro futures: ~$0.32 commission + $0.30 exchange + $0.01 NFA per side"""
    FEE_PER_SIDE = 0.63

    def GetOrderFee(self, parameters):
        contracts = abs(parameters.Order.AbsoluteQuantity)
        return OrderFee(CashAmount(contracts * self.FEE_PER_SIDE, "USD"))


class MNQStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(3000)  # Need ~$1,300 margin per MNQ contract
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.entry_threshold = 0.75
        self.high_conviction_threshold = 0.60

        # TP/SL Parameters — recalibrated for MNQ micro-scalp
        self.quick_take_profit = self._get_param("quick_take_profit", 0.0025)  # 0.25% (~50 NQ points)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.002)   # 0.20% (~40 NQ points)
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  2.5)
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  1.5)
        self.trail_activation  = self._get_param("trail_activation",  0.005)   # 0.5%
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.002)   # 0.2%
        self.time_stop_hours   = self._get_param("time_stop_hours",   1.5)     # Shorter for intraday futures
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.001)
        self.extended_time_stop_hours   = self._get_param("extended_time_stop_hours",   3.0)
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.004)
        self.stale_position_hours       = self._get_param("stale_position_hours",       4.0)

        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 3.5

        # Position sizing — contract-based, not percentage-based
        self.max_contracts     = 2       # Max 2 contracts per instrument
        self.position_size_pct = 0.50    # Use 50% of available margin
        self.base_max_positions = 3      # One position per instrument (MNQ, M2K, MGC)
        self.max_positions      = 3
        # Portfolio-wide hard cap: with $3,000 capital, keep total open contracts
        # across ALL symbols to at most 2 to avoid IB margin rejections.
        self.max_portfolio_contracts = 2
        # Minutes to cool down a symbol after an Invalid entry order (margin rejection)
        self.invalid_entry_cooldown_minutes = 30

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap", 0.80)
        self.min_asset_vol_floor     = 0.05

        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(60 * 24 * 252)  # Trading days for futures

        # Fee parameters — flat rate, not percentage
        self.expected_round_trip_fees = 1.26   # $1.26 per RT (not percentage!)
        self.fee_slippage_buffer      = 0.50   # $0.50 buffer
        self.min_expected_profit_pct  = 0.0015  # 0.15% — ensure ATR-based move covers fees/spread
        self.adx_min_period           = 10

        self.skip_hours_utc         = []
        self.max_daily_trades       = 20
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 0.5
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 5

        self.stale_order_timeout_seconds      = 30
        self.live_stale_order_timeout_seconds = 60
        self.max_concurrent_open_orders       = 2
        self.open_orders_cash_threshold       = 0.90
        self.order_fill_check_threshold_seconds = 60
        self.order_timeout_seconds              = 30
        self.resync_log_interval_seconds        = 1800
        self.portfolio_mismatch_threshold       = 0.10
        self.portfolio_mismatch_min_dollars     = 5.00
        self.portfolio_mismatch_cooldown_seconds = 3600
        self.retry_pending_cooldown_seconds     = 60
        self.rate_limit_cooldown_minutes        = 10

        self.max_drawdown_limit    = 0.20

        self._positions_synced    = False
        self._session_blacklist   = set()
        self._symbol_entry_cooldowns = {}
        self._spread_warning_times = {}
        self._first_post_warmup   = True
        self._submitted_orders    = {}
        self._symbol_slippage_history = {}
        self._order_retries       = {}
        self._retry_pending       = {}
        self._rate_limit_until    = None
        self._last_mismatch_warning = None
        self._daily_open_value     = None
        self.pnl_by_tag            = {}

        self.trade_count      = 0
        self._pending_orders  = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns  = {}
        self._symbol_loss_cooldowns = {}
        self._slip_abs        = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold  = 0.001
        self.slip_outlier_threshold = 0.002
        self.slip_alert_duration_hours = 2
        self._recent_tickets  = deque(maxlen=25)

        # Bracket order tracking — replaces manual exit (CheckExits) logic
        self._pending_entry_info = {}  # entry_order_id → {direction, contracts}
        self._bracket_orders     = {}  # symbol → {tp_id, sl_id, direction, entry_price}

        self._rolling_wins      = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None

        self.market_regime    = "unknown"
        self.volatility_regime = "normal"
        self._regime_hold_count = 0

        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.trade_log      = []
        self.log_budget     = 0
        self.last_log_time  = None
        self._last_skip_reason = None

        # Multi-symbol futures setup (MNQ, M2K, MGC)
        mnq_future = self.AddFuture(Futures.Indices.MicroNASDAQ100EMini, Resolution.Minute)
        mnq_future.SetFilter(timedelta(days=0), timedelta(days=90))
        self.mnq_base_symbol = mnq_future.Symbol

        m2k_future = self.AddFuture(Futures.Indices.MicroRussell2000EMini, Resolution.Minute)
        m2k_future.SetFilter(timedelta(days=0), timedelta(days=90))
        self.m2k_base_symbol = m2k_future.Symbol

        mgc_future = self.AddFuture(Futures.Metals.MicroGold, Resolution.Minute)
        mgc_future.SetFilter(timedelta(days=0), timedelta(days=90))
        self.mgc_base_symbol = mgc_future.Symbol

        self.base_symbols = [self.mnq_base_symbol, self.m2k_base_symbol, self.mgc_base_symbol]
        self.active_contracts = {}   # base_symbol -> active front-month contract symbol
        self.instrument_data  = {}   # active contract symbol -> indicator/price data dict

        # VIX as regime overlay (replaces Fear & Greed and BTC reference)
        self.vix_symbol = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        self.vix_value = 20.0  # neutral default

        # Per-symbol instrument data — replaces single self.mnq_data
        # (populated in OnData when active contracts are resolved)

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

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MNQ/M2K/MGC MICRO-SCALP) v7.3.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max contracts: {self.max_contracts} | Symbols: MNQ, M2K, MGC")

    def CustomSecurityInitializer(self, security):
        security.SetSlippageModel(MNQSlippage())
        security.SetFeeModel(MNQFeeModel())

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
            return default
        except Exception as e:
            self.Debug(f"Error getting parameter {name}: {e}")
            return default

    def _normalize_order_time(self, order_time):
        return normalize_order_time(order_time)

    def _record_exit_pnl(self, symbol, entry_price, exit_price, exit_tag="Unknown"):
        return record_exit_pnl(self, symbol, entry_price, exit_price, exit_tag=exit_tag)

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        self._daily_open_value = self.Portfolio.TotalPortfolioValue
        for data in self.instrument_data.values():
            data['trade_count_today'] = 0
        self._symbol_entry_cooldowns.clear()
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

    def _make_instrument_data(self):
        """Create and return a new instrument data structure for one futures contract."""
        return {
            'prices': deque(maxlen=self.lookback),
            'returns': deque(maxlen=self.lookback),
            'volume': deque(maxlen=self.lookback),
            'volume_ma': deque(maxlen=self.medium_period),
            'ema_ultra_short': ExponentialMovingAverage(self.ultra_short_period),
            'ema_short': ExponentialMovingAverage(self.short_period),
            'ema_medium': ExponentialMovingAverage(self.medium_period),
            'ema_5': ExponentialMovingAverage(5),
            'atr': AverageTrueRange(14),
            'adx': AverageDirectionalIndex(self.adx_min_period),
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(7),
            'zscore': deque(maxlen=self.short_period),
            'last_price': 0,
            'recent_net_scores': deque(maxlen=3),
            'trail_stop': None,
            'highs': deque(maxlen=self.lookback),
            'lows': deque(maxlen=self.lookback),
            'bb_upper': deque(maxlen=self.short_period),
            'bb_lower': deque(maxlen=self.short_period),
            'bb_width': deque(maxlen=self.medium_period),
            'trade_count_today': 0,
            'last_loss_time': None,
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

    def _is_rth(self):
        """Regular Trading Hours: 8:30 AM - 3:00 PM CT (13:30-20:00 UTC)"""
        t = self.Time
        hour_utc = t.hour
        minute_utc = t.minute
        utc_minutes = hour_utc * 60 + minute_utc
        rth_start = 13 * 60 + 30  # 13:30 UTC = 8:30 CT
        rth_end = 20 * 60          # 20:00 UTC = 15:00 CT
        return rth_start <= utc_minutes < rth_end

    def _is_extended_hours(self):
        """Overnight/extended session"""
        return not self._is_rth()

    def OnData(self, data):
        # === Contract selection / rollover — handled independently per base symbol ===
        for chain in data.FutureChains:
            base_sym = chain.Key
            # Only process chains for our subscribed instruments
            if base_sym not in self.base_symbols:
                continue
            contracts = sorted(chain.Value, key=lambda c: c.Expiry)
            if contracts:
                front = contracts[0]
                current_active = self.active_contracts.get(base_sym)
                if current_active != front.Symbol:
                    if current_active and self.Portfolio[current_active].Invested:
                        self.Liquidate(current_active, "Contract rollover")
                    # Drop stale data for the old contract
                    if current_active is not None:
                        self.instrument_data.pop(current_active, None)
                    self.active_contracts[base_sym] = front.Symbol
                    self.instrument_data[front.Symbol] = self._make_instrument_data()
                    self.Debug(f"Active contract: {front.Symbol.Value} expiry {front.Expiry}")

        if self.IsWarmingUp or not self.active_contracts:
            return

        # Process bars for each active contract
        for base_sym, contract in self.active_contracts.items():
            if contract not in self.instrument_data:
                self.instrument_data[contract] = self._make_instrument_data()
            if data.Bars.ContainsKey(contract):
                bar = data.Bars[contract]
                self._update_symbol_data(contract, bar)

        # VIX data
        if data.ContainsKey(self.vix_symbol):
            vix = data[self.vix_symbol]
            if vix is not None:
                self.vix_value = vix.Value

        if not self._positions_synced:
            if not self._first_post_warmup:
                self._cancel_stale_orders()
            sync_existing_positions(self)
            self._positions_synced = True
            self._first_post_warmup = False
            for base_sym, contract in self.active_contracts.items():
                d = self.instrument_data.get(contract)
                ready = self._is_ready(d) if d else False
                self.Debug(f"Post-warmup: {contract.Value} data {'ready' if ready else 'warming'}")

        self._update_market_context()
        self.Rebalance()

    def _update_symbol_data(self, symbol, bar):
        """Update instrument data indicators from the latest bar."""
        mnq = self.instrument_data.get(symbol)
        if mnq is None:
            return
        price = float(bar.Close)
        high = float(bar.High)
        low = float(bar.Low)
        volume = float(bar.Volume)
        mnq['prices'].append(price)
        mnq['highs'].append(high)
        mnq['lows'].append(low)
        if mnq['last_price'] > 0:
            ret = (price - mnq['last_price']) / mnq['last_price']
            mnq['returns'].append(ret)
        mnq['last_price'] = price
        mnq['volume'].append(volume)
        if len(mnq['volume']) >= self.short_period:
            mnq['volume_ma'].append(np.mean(list(mnq['volume'])[-self.short_period:]))
        mnq['ema_ultra_short'].Update(bar.EndTime, price)
        mnq['ema_short'].Update(bar.EndTime, price)
        mnq['ema_medium'].Update(bar.EndTime, price)
        mnq['ema_5'].Update(bar.EndTime, price)
        mnq['atr'].Update(bar)
        mnq['adx'].Update(bar)
        mnq['vwap_pv'].append(price * volume)
        mnq['vwap_v'].append(volume)
        total_v = sum(mnq['vwap_v'])
        if total_v > 0:
            mnq['vwap'] = sum(mnq['vwap_pv']) / total_v
        mnq['volume_long'].append(volume)
        if len(mnq['vwap_v']) >= 5 and mnq['vwap'] > 0:
            vwap_val = mnq['vwap']
            pv_list = list(mnq['vwap_pv'])
            v_list = list(mnq['vwap_v'])
            bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
            if len(bar_prices) >= 5:
                sd = float(np.std(bar_prices))
                mnq['vwap_sd'] = sd
                mnq['vwap_sd2_lower'] = vwap_val - 2.0 * sd
                mnq['vwap_sd3_lower'] = vwap_val - 3.0 * sd
        if len(mnq['returns']) >= 10:
            mnq['volatility'].append(np.std(list(mnq['returns'])[-10:]))
        mnq['rsi'].Update(bar.EndTime, price)
        if len(mnq['prices']) >= self.medium_period:
            prices_arr = np.array(list(mnq['prices'])[-self.medium_period:])
            std = np.std(prices_arr)
            mean = np.mean(prices_arr)
            if std > 0:
                mnq['zscore'].append((price - mean) / std)
                mnq['bb_upper'].append(mean + 2 * std)
                mnq['bb_lower'].append(mean - 2 * std)
                mnq['bb_width'].append(4 * std / mean if mean > 0 else 0)
        high_low = high - low
        if high_low > 0:
            bar_delta = volume * ((price - low) - (high - price)) / high_low
        else:
            bar_delta = 0.0
        prev_cvd = mnq['cvd'][-1] if len(mnq['cvd']) > 0 else 0.0
        mnq['cvd'].append(prev_cvd + bar_delta)
        if len(mnq['prices']) >= 15:
            price_change = abs(mnq['prices'][-1] - mnq['prices'][-15])
            volatility_sum = sum(abs(mnq['prices'][i] - mnq['prices'][i-1]) for i in range(-14, 0))
            if volatility_sum > 0:
                mnq['ker'].append(price_change / volatility_sum)
            else:
                mnq['ker'].append(0.0)
        Q = 1e-5
        R = 0.01
        if mnq['kalman_estimate'] == 0.0:
            mnq['kalman_estimate'] = price
        estimate_pred = mnq['kalman_estimate']
        error_cov_pred = mnq['kalman_error_cov'] + Q
        kalman_gain = error_cov_pred / (error_cov_pred + R)
        mnq['kalman_estimate'] = estimate_pred + kalman_gain * (price - estimate_pred)
        mnq['kalman_error_cov'] = (1 - kalman_gain) * error_cov_pred

    def _update_market_context(self):
        """VIX-based regime detection (replaces BTC-based)."""
        vix = self.vix_value
        if vix < 15:
            new_regime = "bull"
            new_vol_regime = "low"
        elif vix < 25:
            new_regime = "sideways"
            new_vol_regime = "normal"
        elif vix < 35:
            new_regime = "bear"
            new_vol_regime = "high"
        else:
            new_regime = "bear"
            new_vol_regime = "extreme"

        # Hysteresis: only change regime if held for 3+ bars
        if new_regime != self.market_regime:
            self._regime_hold_count += 1
            if self._regime_hold_count >= 3:
                self.market_regime = new_regime
                self._regime_hold_count = 0
        else:
            self._regime_hold_count = 0

        self.volatility_regime = new_vol_regime

    def _annualized_vol(self, mnq):
        if mnq is None:
            return None
        if len(mnq.get('volatility', [])) == 0:
            return None
        return float(mnq['volatility'][-1]) * self.sqrt_annualization

    def _normalize(self, v, mn, mx):
        if mx - mn <= 0:
            return 0.5
        return max(0, min(1, (v - mn) / (mx - mn)))

    def _calculate_factor_scores(self, symbol, mnq):
        """Evaluate long and short signals; pick the dominant direction."""
        long_score, long_components = self._scoring_engine.calculate_scalp_score(mnq)
        short_score, short_components = self._scoring_engine.calculate_short_score(mnq)

        # Require a minimum score gap to avoid trading on mixed/choppy signals
        min_gap = 0.15
        if abs(long_score - short_score) < min_gap:
            return None

        if short_score > long_score and short_score >= self._get_threshold():
            components = short_components.copy()
            components['_scalp_score'] = short_score
            components['_direction'] = -1
            components['_long_score'] = long_score
            components['_short_score'] = short_score
        else:
            components = long_components.copy()
            components['_scalp_score'] = long_score
            components['_direction'] = 1
            components['_long_score'] = long_score
            components['_short_score'] = short_score
        return components

    def _calculate_composite_score(self, factors, mnq=None):
        """Return the pre-computed scalp score."""
        return factors.get('_scalp_score', 0.0)

    def _apply_fee_adjustment(self, score):
        """Return score unchanged — MNQ fees are negligible ($1.26 RT)."""
        return score

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        """Contract-count based position sizing for MNQ."""
        return self._scoring_engine.calculate_position_size(score, threshold, asset_vol_ann)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    def _get_max_daily_trades(self):
        return self.max_daily_trades

    def _get_threshold(self):
        return self.entry_threshold

    def _daily_loss_exceeded(self):
        """Returns True if the portfolio has dropped >= 3% from today's open value."""
        if self._daily_open_value is None or self._daily_open_value <= 0:
            return False
        current = self.Portfolio.TotalPortfolioValue
        if current <= 0:
            return True
        drop = (self._daily_open_value - current) / self._daily_open_value
        return drop >= 0.03

    def _log_skip(self, reason):
        if self.LiveMode:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason
        elif reason != self._last_skip_reason:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason

    def Rebalance(self):
        if self.IsWarmingUp or not self.active_contracts:
            return

        if not self._is_rth():
            return  # Only enter new trades during Regular Trading Hours

        # Circuit breaker 1: max 3% daily portfolio loss
        if self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded (3%)")
            return

        self.log_budget = 20

        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            self._log_skip("rate limited")
            return

        if self.LiveMode and not live_safety_checks(self):
            return

        cancel_stale_new_orders(self)
        if self.daily_trade_count >= self._get_max_daily_trades():
            self._log_skip("max daily trades")
            return

        # VIX-based position limit — avoid erratic (> 25) regimes
        if self.vix_value > 25:
            self._log_skip(f"VIX too high ({self.vix_value:.1f}) — market too erratic/dangerous")
            return

        # === Per-symbol trading loop ===
        # Count already-open contracts across all instruments before we begin so
        # we can enforce the portfolio-wide margin cap within a single Rebalance call.
        total_open_contracts = get_actual_position_count(self)
        orders_queued_this_rebalance = 0

        for base_sym, active_contract in list(self.active_contracts.items()):
            # Portfolio-wide cap: stop queuing new entries when the combined count
            # (already open + queued this tick) would exceed our margin limit.
            if total_open_contracts + orders_queued_this_rebalance >= self.max_portfolio_contracts:
                self._log_skip(f"Portfolio max contracts ({self.max_portfolio_contracts}) reached")
                break

            mnq = self.instrument_data.get(active_contract)
            if mnq is None or not self._is_ready(mnq):
                continue

            # Circuit breaker 2: max 5 trades per symbol per day
            if mnq.get('trade_count_today', 0) >= self.max_symbol_trades_per_day:
                self._log_skip(f"{active_contract.Value}: max {self.max_symbol_trades_per_day} trades/day reached")
                continue

            # Score this contract
            factors = self._calculate_factor_scores(active_contract, mnq)
            if not factors:
                continue

            composite_score = self._calculate_composite_score(factors, mnq)
            net_score = self._apply_fee_adjustment(composite_score)
            mnq['recent_net_scores'].append(net_score)

            threshold_now = self._get_threshold()
            debug_limited(self, f"REBALANCE {active_contract.Value}: score={net_score:.2f} thresh={threshold_now:.2f} | VIX={self.vix_value:.1f} | {self.market_regime}")

            if net_score < threshold_now:
                continue

            if is_invested_not_dust(self, active_contract):
                continue

            if has_open_orders(self, active_contract):
                continue

            if active_contract.Value in self._symbol_entry_cooldowns and self.Time < self._symbol_entry_cooldowns[active_contract.Value]:
                continue

            atr_val = mnq['atr'].Current.Value if mnq['atr'].IsReady else None
            if atr_val and self.Securities[active_contract].Price > 0:
                price = self.Securities[active_contract].Price
                expected_move_pct = (atr_val * self.atr_tp_mult) / price
                if expected_move_pct < self.min_expected_profit_pct:
                    continue

            # Margin safety check: ensure enough remaining margin before entry
            if self.Portfolio.MarginRemaining <= 500:
                self._log_skip(f"Insufficient margin: ${self.Portfolio.MarginRemaining:.0f} remaining")
                break

            vol = self._annualized_vol(mnq)
            contracts = self._calculate_position_size(net_score, threshold_now, vol)

            if contracts < 1:
                continue

            try:
                direction = factors.get('_direction', 1)
                price = self.Securities[active_contract].Price
                order_qty = contracts * direction
                ticket = self.MarketOrder(active_contract, order_qty, tag=f"MG Entry score={net_score:.2f}")
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    # Store entry info keyed by order ID so OnOrderEvent can attach bracket orders
                    self._pending_entry_info[ticket.OrderId] = {
                        'direction': direction,
                        'contracts': contracts,
                    }
                    orders_queued_this_rebalance += 1
                    components = factors
                    dir_str = "LONG" if direction == 1 else "SHORT"
                    sig_str = (f"tick_imb={components.get('obi', 0):.2f} "
                               f"vol={components.get('vol_ignition', 0):.2f} "
                               f"trend={components.get('micro_trend', 0):.2f} "
                               f"adx={components.get('adx_trend', 0):.2f} "
                               f"mean_rev={components.get('mean_reversion', 0):.2f} "
                               f"vwap={components.get('vwap_signal', 0):.2f}")
                    self.Debug(f"{active_contract.Value} ENTRY ({dir_str}): {contracts} contract(s) | score={net_score:.2f} | ${price:.2f} | {sig_str}")
                    self.trade_count += 1
                    mnq['trade_count_today'] = mnq.get('trade_count_today', 0) + 1
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {active_contract.Value} - {e}")

        self._last_skip_reason = None

    def _is_ready(self, mnq):
        if mnq is None:
            return False
        return len(mnq['prices']) >= 10 and mnq['rsi'].IsReady

    def _submit_bracket_orders(self, symbol, fill_price, direction, contracts):
        """Submit TP (LimitOrder) and SL (StopMarketOrder) bracket legs after an entry fill."""
        mnq = self.instrument_data.get(symbol)
        atr_val = mnq['atr'].Current.Value if mnq and mnq['atr'].IsReady else None
        if atr_val and fill_price > 0:
            sl_pct = max((atr_val * self.atr_sl_mult) / fill_price, self.tight_stop_loss)
            tp_pct = max((atr_val * self.atr_tp_mult) / fill_price, self.quick_take_profit)
        else:
            sl_pct = self.tight_stop_loss
            tp_pct = self.quick_take_profit

        if tp_pct < sl_pct * 1.5:
            tp_pct = sl_pct * 1.5

        exit_qty = -contracts * direction  # opposite of entry to close position
        if direction == 1:  # Long entry
            tp_price = fill_price * (1 + tp_pct)
            sl_price = fill_price * (1 - sl_pct)
        else:  # Short entry
            tp_price = fill_price * (1 - tp_pct)
            sl_price = fill_price * (1 + sl_pct)

        tp_ticket = self.LimitOrder(symbol, exit_qty, tp_price, tag="Take Profit")
        sl_ticket = self.StopMarketOrder(symbol, exit_qty, sl_price, tag="Stop Loss")

        if tp_ticket is not None and sl_ticket is not None:
            self._bracket_orders[symbol] = {
                'tp_id': tp_ticket.OrderId,
                'sl_id': sl_ticket.OrderId,
                'direction': direction,
                'entry_price': fill_price,
            }
            dir_str = "LONG" if direction == 1 else "SHORT"
            self.Debug(f"BRACKET: {symbol.Value} ({dir_str}) | TP=${tp_price:.2f} SL=${sl_price:.2f}")
        else:
            self.Debug(f"⚠️ BRACKET FAILED for {symbol.Value} — tp={tp_ticket} sl={sl_ticket}")

    def OnOrderEvent(self, event):
        try:
            symbol = event.Symbol
            self.Debug(f"ORDER: {symbol.Value} {event.Status} {event.Direction} qty={event.FillQuantity or event.Quantity} price={event.FillPrice} id={event.OrderId}")
            if event.Status == OrderStatus.Submitted:
                if symbol not in self._pending_orders:
                    self._pending_orders[symbol] = 0
                intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
                self._pending_orders[symbol] += intended_qty
                if symbol not in self._submitted_orders:
                    has_position = symbol in self.Portfolio and self.Portfolio[symbol].Invested
                    inferred_intent = 'exit' if has_position else 'entry'
                    self._submitted_orders[symbol] = {
                        'order_id': event.OrderId,
                        'time': self.Time,
                        'quantity': event.Quantity,
                        'intent': inferred_intent
                    }
                else:
                    self._submitted_orders[symbol]['order_id'] = event.OrderId
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

                # --- Entry fill: attach bracket orders ---
                entry_info = self._pending_entry_info.pop(event.OrderId, None)
                if entry_info is not None:
                    self.daily_trade_count += 1
                    self._submit_bracket_orders(
                        symbol,
                        event.FillPrice,
                        entry_info['direction'],
                        entry_info['contracts'],
                    )
                else:
                    # --- Bracket fill: cancel the other leg and record PnL ---
                    bracket = self._bracket_orders.get(symbol)
                    if bracket is not None:
                        order = self.Transactions.GetOrderById(event.OrderId)
                        exit_tag = order.Tag if order and order.Tag else "Unknown"
                        if event.OrderId == bracket['tp_id']:
                            # TP filled — cancel the SL
                            try:
                                self.Transactions.CancelOrder(bracket['sl_id'])
                            except Exception as ce:
                                self.Debug(f"Cancel SL error {symbol.Value}: {ce}")
                        elif event.OrderId == bracket['sl_id']:
                            # SL filled — cancel the TP
                            try:
                                self.Transactions.CancelOrder(bracket['tp_id'])
                            except Exception as ce:
                                self.Debug(f"Cancel TP error {symbol.Value}: {ce}")
                        # Record PnL
                        entry = bracket['entry_price']
                        direction = bracket['direction']
                        if direction == 1:
                            pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                        else:
                            pnl = (entry - event.FillPrice) / entry if entry > 0 else 0
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
                            'time': self.Time,
                            'symbol': symbol.Value,
                            'pnl_pct': pnl,
                            'exit_reason': exit_tag,
                        })
                        self.Debug(f"EXIT ({exit_tag}): {symbol.Value} | entry=${entry:.2f} exit=${event.FillPrice:.2f} PnL:{pnl:+.2%}")
                        self._bracket_orders.pop(symbol, None)
                        cleanup_position(self, symbol)

                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                # If a pending entry was canceled before filling, remove its info
                self._pending_entry_info.pop(event.OrderId, None)
            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                self._pending_entry_info.pop(event.OrderId, None)
                msg = getattr(event, 'Message', '') or ''
                self.Debug(f"INVALID ORDER: {symbol.Value} dir={event.Direction} qty={event.Quantity} | Reason: {msg or 'unknown'}")
                # If this was a failed entry attempt (not invested), apply a cooldown
                if not is_invested_not_dust(self, symbol):
                    cooldown_until = self.Time + timedelta(minutes=self.invalid_entry_cooldown_minutes)
                    self._symbol_entry_cooldowns[symbol.Value] = cooldown_until
                    self.Debug(f"Entry cooldown set for {symbol.Value} until {cooldown_until} after invalid order")
        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")
        if self.LiveMode:
            persist_state(self)

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%}")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"PnL: {self.total_pnl:+.2%}")
        persist_state(self)

    def DailyReport(self):
        if self.IsWarmingUp: return
        daily_report(self)
