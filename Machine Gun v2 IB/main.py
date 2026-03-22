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

        self.entry_threshold = 0.50
        self.high_conviction_threshold = 0.60

        # TP/SL Parameters — recalibrated for MNQ micro-scalp
        self.quick_take_profit = self._get_param("quick_take_profit", 0.0025)  # 0.25% (~50 NQ points)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.0015)  # 0.15% (~30 NQ points)
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
        self.atr_trail_mult      = 2.0

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
        self.min_expected_profit_pct  = 0.001  # 0.1% — lower threshold for low-ATR periods
        self.adx_min_period           = 10

        self.skip_hours_utc         = []
        self.max_daily_trades       = 20
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 0.5
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 10

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
        self.cooldown_hours        = 4
        self.consecutive_losses    = 0
        self.max_consecutive_losses = 4
        self._consecutive_loss_halve_remaining = 0
        self.circuit_breaker_expiry = None

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
        self._failed_exit_attempts = {}
        self._failed_exit_counts   = {}
        self._daily_open_value     = None
        self.pnl_by_tag            = {}

        self.peak_value       = None
        self.drawdown_cooldown = 0
        self.entry_prices     = {}
        self.highest_prices   = {}
        self.lowest_prices    = {}
        self._entry_directions = {}
        self.entry_times      = {}
        self.entry_volumes    = {}
        self._partial_tp_taken      = {}
        self._breakeven_stops       = {}
        self._partial_sell_symbols  = set()
        self._choppy_regime_entries = {}
        self.partial_tp_threshold   = 0.004  # 0.4% partial TP for MNQ
        self.stagnation_minutes     = 60
        self.stagnation_pnl_threshold = 0.001
        self.rsi_peaked_overbought = {}
        self.trade_count      = 0
        self._pending_orders  = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns  = {}
        self._symbol_loss_cooldowns = {}
        self._cash_mode_until = None
        self._recent_trade_outcomes = deque(maxlen=20)
        self.trailing_grace_hours = 0.5
        self._slip_abs        = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold  = 0.001
        self.slip_outlier_threshold = 0.002
        self.slip_alert_duration_hours = 2
        self._recent_tickets  = deque(maxlen=25)

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
        self.CheckExits()

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

        if self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded")
            return

        if self._cash_mode_until is not None and self.Time < self._cash_mode_until:
            self._log_skip("cash mode - poor recent performance")
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

        val = self.Portfolio.TotalPortfolioValue
        if self.peak_value is None or self.peak_value < 1:
            self.peak_value = val
        if self.drawdown_cooldown > 0:
            self.drawdown_cooldown -= 1
            if self.drawdown_cooldown <= 0:
                self.peak_value = val
                self.consecutive_losses = 0
            else:
                self._log_skip(f"drawdown cooldown {self.drawdown_cooldown}h")
                return

        self.peak_value = max(self.peak_value, val)
        dd = (self.peak_value - val) / self.peak_value if self.peak_value > 0 else 0
        if dd > self.max_drawdown_limit:
            self.drawdown_cooldown = self.cooldown_hours
            self._log_skip(f"drawdown {dd:.1%} > limit")
            return

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.drawdown_cooldown = 0.5
            self._consecutive_loss_halve_remaining = 3
            self.consecutive_losses = 0
            self._log_skip("consecutive loss cooldown")
            return

        if self.consecutive_losses >= 3:
            self.circuit_breaker_expiry = self.Time + timedelta(minutes=15)
            self.consecutive_losses = 0
            self._log_skip("circuit breaker triggered (3 consecutive losses)")
            return

        if self.circuit_breaker_expiry is not None and self.Time < self.circuit_breaker_expiry:
            self._log_skip("circuit breaker active")
            return

        # VIX-based position limit in extreme regimes
        if self.vix_value >= 35:
            self._log_skip(f"VIX extreme ({self.vix_value:.1f}) — halting new entries")
            return

        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
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

            vol = self._annualized_vol(mnq)
            contracts = self._calculate_position_size(net_score, threshold_now, vol)

            if self._consecutive_loss_halve_remaining > 0:
                contracts = max(0, contracts - 1)

            if contracts < 1:
                continue

            try:
                direction = factors.get('_direction', 1)
                price = self.Securities[active_contract].Price
                order_qty = contracts * direction
                self._entry_directions[active_contract] = direction
                ticket = self.MarketOrder(active_contract, order_qty, tag=f"MG Entry score={net_score:.2f}")
                if ticket is not None:
                    self._recent_tickets.append(ticket)
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
                    adx_ind = mnq.get('adx')
                    is_choppy = (adx_ind is not None and adx_ind.IsReady
                                 and adx_ind.Current.Value < 25)
                    self._choppy_regime_entries[active_contract] = is_choppy
                    if self._consecutive_loss_halve_remaining > 0:
                        self._consecutive_loss_halve_remaining -= 1
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {active_contract.Value} - {e}")

        self._last_skip_reason = None

    def _is_ready(self, mnq):
        if mnq is None:
            return False
        return len(mnq['prices']) >= 10 and mnq['rsi'].IsReady

    def CheckExits(self):
        if self.IsWarmingUp:
            return
        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            return

        for kvp in self.Portfolio:
            if not is_invested_not_dust(self, kvp.Key):
                self._failed_exit_attempts.pop(kvp.Key, None)
                self._failed_exit_counts.pop(kvp.Key, None)
                continue

            if self._failed_exit_counts.get(kvp.Key, 0) >= 3:
                continue
            self._check_exit(kvp.Key, self.Securities[kvp.Key].Price, kvp.Value)

        for kvp in self.Portfolio:
            symbol = kvp.Key
            if not is_invested_not_dust(self, symbol):
                continue
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = kvp.Value.AveragePrice
                self.entry_times[symbol] = self.Time
                if kvp.Value.Quantity < 0:
                    self.lowest_prices[symbol] = kvp.Value.AveragePrice
                else:
                    self.highest_prices[symbol] = kvp.Value.AveragePrice
                self.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")

    def _check_exit(self, symbol, price, holding):
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            return

        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time
            if holding.Quantity < 0:
                self.lowest_prices[symbol] = holding.AveragePrice
            else:
                self.highest_prices[symbol] = holding.AveragePrice

        is_short = holding.Quantity < 0
        entry = self.entry_prices[symbol]

        if is_short:
            lowest = self.lowest_prices.get(symbol, entry)
            if price < lowest:
                self.lowest_prices[symbol] = price
                lowest = price
            pnl = (entry - price) / entry if entry > 0 else 0
            dd = (price - lowest) / lowest if lowest > 0 else 0
        else:
            highest = self.highest_prices.get(symbol, entry)
            if price > highest:
                self.highest_prices[symbol] = price
                highest = price
            pnl = (price - entry) / entry if entry > 0 else 0
            dd = (highest - price) / highest if highest > 0 else 0

        mnq = self.instrument_data.get(symbol)
        hours = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        minutes = hours * 60

        atr = mnq['atr'].Current.Value if mnq and mnq['atr'].IsReady else None
        if atr and entry > 0:
            sl = max((atr * self.atr_sl_mult) / entry, self.tight_stop_loss)
            tp = max((atr * self.atr_tp_mult) / entry, self.quick_take_profit)
        else:
            sl = self.tight_stop_loss
            tp = self.quick_take_profit

        if tp < sl * 1.5:
            tp = sl * 1.5

        if self._choppy_regime_entries.get(symbol, False):
            tp = tp * 0.65

        if self.volatility_regime == "low":
            tp = tp * 0.75

        trailing_activation = self.trail_activation
        trailing_stop_pct   = self.trail_stop_pct

        if mnq and mnq['rsi'].IsReady:
            rsi_now = mnq['rsi'].Current.Value
            if rsi_now > 85:
                self.rsi_peaked_overbought[symbol] = True

        if (not self._partial_tp_taken.get(symbol, False)
                and pnl >= self.partial_tp_threshold):
            if partial_smart_sell(self, symbol, 0.50, "Partial TP"):
                self._partial_tp_taken[symbol] = True
                if is_short:
                    self._breakeven_stops[symbol] = entry * 0.999
                else:
                    self._breakeven_stops[symbol] = entry * 1.001
                self.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL→entry±0.1%")
                return

        tag = ""

        if self._partial_tp_taken.get(symbol, False):
            be_price = self._breakeven_stops.get(symbol, entry)
            if is_short:
                if price >= be_price:
                    tag = "Breakeven Stop"
            else:
                if price <= be_price:
                    tag = "Breakeven Stop"
        elif pnl <= -sl:
            tag = "Stop Loss"

        if not tag and minutes > self.stagnation_minutes and pnl < self.stagnation_pnl_threshold:
            tag = "Stagnation Exit"

        elif not tag:
            if not self._partial_tp_taken.get(symbol, False) and pnl >= tp:
                tag = "Take Profit"

            elif pnl > trailing_activation and dd >= trailing_stop_pct:
                tag = "Trailing Stop"

            elif atr and entry > 0 and holding.Quantity != 0:
                trail_offset = atr * self.atr_trail_mult
                if is_short:
                    trail_level = lowest + trail_offset
                else:
                    trail_level = highest - trail_offset
                if mnq:
                    mnq['trail_stop'] = trail_level
                if mnq and mnq['trail_stop'] is not None:
                    if holding.Quantity > 0 and price <= mnq['trail_stop']:
                        tag = "ATR Trail"
                    elif holding.Quantity < 0 and price >= mnq['trail_stop']:
                        tag = "ATR Trail"

            if not tag and hours >= self.time_stop_hours and pnl < self.time_stop_pnl_min:
                tag = "Time Stop"

            if not tag and hours >= self.extended_time_stop_hours and pnl < self.extended_time_stop_pnl_max:
                tag = "Extended Time Stop"

            if not tag and hours >= self.stale_position_hours:
                tag = "Stale Position Exit"

        if tag:
            if pnl < 0:
                self._symbol_loss_cooldowns[symbol] = self.Time + timedelta(hours=1)
            sold = smart_liquidate(self, symbol, tag)
            if sold:
                self._exit_cooldowns[symbol] = self.Time + timedelta(hours=self.exit_cooldown_hours)
                self.rsi_peaked_overbought.pop(symbol, None)
                self.entry_volumes.pop(symbol, None)
                self._choppy_regime_entries.pop(symbol, None)
                self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
            else:
                fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                self._failed_exit_counts[symbol] = fail_count
                self.Debug(f"⚠️ EXIT FAILED ({tag}) #{fail_count}: {symbol.Value} | PnL:{pnl:+.2%}")
                if fail_count >= 3:
                    self.Debug(f"FATAL EXIT FAILURE: {symbol.Value} — escalating to market order")
                    try:
                        qty = abs(holding.Quantity)
                        if qty > 0:
                            self.MarketOrder(symbol, -qty, tag=f"Force Exit (fail#{fail_count})")
                    except Exception as e:
                        self.Debug(f"Force market exit error for {symbol.Value}: {e}")
                    self._failed_exit_counts.pop(symbol, None)
                    self.rsi_peaked_overbought.pop(symbol, None)
                    self.entry_volumes.pop(symbol, None)
                    self._choppy_regime_entries.pop(symbol, None)

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
                    if has_position:
                        inferred_intent = 'exit'
                    else:
                        inferred_intent = 'entry'
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
                intended_dir = self._entry_directions.get(symbol, 1)
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    if intended_dir == -1:
                        self.lowest_prices[symbol] = event.FillPrice
                    else:
                        self.highest_prices[symbol] = event.FillPrice
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                intended_dir = self._entry_directions.get(symbol, 1)
                is_new_entry = symbol not in self.entry_prices
                is_long_entry = is_new_entry and event.Direction == OrderDirection.Buy and intended_dir == 1
                is_short_entry = is_new_entry and event.Direction == OrderDirection.Sell and intended_dir == -1
                if is_long_entry:
                    self.entry_prices[symbol] = event.FillPrice
                    self.highest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1
                    self.rsi_peaked_overbought.pop(symbol, None)
                elif is_short_entry:
                    self.entry_prices[symbol] = event.FillPrice
                    self.lowest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1
                    self.rsi_peaked_overbought.pop(symbol, None)
                else:
                    if symbol in self._partial_sell_symbols:
                        self._partial_sell_symbols.discard(symbol)
                    else:
                        order = self.Transactions.GetOrderById(event.OrderId)
                        exit_tag = order.Tag if order and order.Tag else "Unknown"
                        entry = self.entry_prices.get(symbol, None)
                        if entry is None:
                            entry = event.FillPrice
                            self.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} exit, using fill price")
                        if intended_dir == -1:
                            pnl = (entry - event.FillPrice) / entry if entry > 0 else 0
                        else:
                            pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                        self._rolling_wins.append(1 if pnl > 0 else 0)
                        self._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                        if pnl > 0:
                            self._rolling_win_sizes.append(pnl)
                            self.winning_trades += 1
                            self.consecutive_losses = 0
                        else:
                            self._rolling_loss_sizes.append(abs(pnl))
                            self.losing_trades += 1
                            self.consecutive_losses += 1
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
                        if len(self._recent_trade_outcomes) >= 12:
                            recent_wr = sum(self._recent_trade_outcomes) / len(self._recent_trade_outcomes)
                            if recent_wr < 0.25:
                                self._cash_mode_until = self.Time + timedelta(minutes=30)
                                self.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(self._recent_trade_outcomes)} trades. Pausing 30min.")
                        cleanup_position(self, symbol)
                        self._failed_exit_attempts.pop(symbol, None)
                        self._failed_exit_counts.pop(symbol, None)
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if symbol not in self.entry_prices and is_invested_not_dust(self, symbol):
                    holding = self.Portfolio[symbol]
                    self.entry_prices[symbol] = holding.AveragePrice
                    self.entry_times[symbol] = self.Time
                    if holding.Quantity < 0:
                        self.lowest_prices[symbol] = holding.AveragePrice
                    else:
                        self.highest_prices[symbol] = holding.AveragePrice
                    self.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                # Log the rejection reason so we can diagnose margin/other issues
                msg = getattr(event, 'Message', '') or ''
                self.Debug(f"INVALID ORDER: {symbol.Value} dir={event.Direction} qty={event.Quantity} | Reason: {msg or 'unknown'}")
                if event.Direction == OrderDirection.Sell:
                    fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                    self._failed_exit_counts[symbol] = fail_count
                    self.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                    if fail_count >= 3:
                        self.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} failed exits")
                        cleanup_position(self, symbol)
                        self._failed_exit_counts.pop(symbol, None)
                    elif symbol not in self.entry_prices:
                        if is_invested_not_dust(self, symbol):
                            holding = self.Portfolio[symbol]
                            self.entry_prices[symbol] = holding.AveragePrice
                            self.highest_prices[symbol] = holding.AveragePrice
                            self.entry_times[symbol] = self.Time
                            self.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
                # If this was a failed entry attempt (not invested), apply a cooldown
                # to prevent the bot from spamming the same order every minute bar.
                if not is_invested_not_dust(self, symbol) and symbol not in self.entry_prices:
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
