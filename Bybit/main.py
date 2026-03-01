# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount, SecurityMarginModel
from QuantConnect.Orders.Slippage import SlippageModel
# endregion

class MakerTakerFeeModel(FeeModel):
    """Bybit: 0.1% Maker/Taker fee (spot margin)."""
    def GetOrderFee(self, parameters):
        order = parameters.Order
        fee_pct = 0.001
        trade_value = order.AbsoluteQuantity * parameters.Security.Price
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))

class SimplifiedCryptoStrategy(QCAlgorithm):
    """Micro-Scalping System v7.1.0 (Bybit) - Long+Short."""

    def Initialize(self):
        self.SetStartDate(2025, 1, 1)

        # CLEAR THE DEFAULT 100K USD, ONLY SET USDT
        self.Portfolio.SetCash(1000)  # Temporarily reset default USD to 1000 to avoid 100k bug
        self.SetCash("USDT", 1000)
        self.SetCash("USD", 0)        # Wipe out the USD so ONLY USDT exists

        # Force Margin account so we can use leverage
        self.SetBrokerageModel(BrokerageName.Bybit, AccountType.Margin)

        self.entry_threshold = 0.60
        self.high_conviction_threshold = 0.80

        self.quick_take_profit = self._get_param("quick_take_profit", 0.050)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.030)
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  5.0)
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  4.0)
        self.trail_activation  = self._get_param("trail_activation",  0.020)
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.010)
        self.time_stop_hours   = self._get_param("time_stop_hours",   4.0)
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.003)
        self.extended_time_stop_hours   = self._get_param("extended_time_stop_hours",   8.0)
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.015) # +1.5% ceiling
        self.stale_position_hours       = self._get_param("stale_position_hours",       12.0)

        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 4.0

        self.position_size_pct  = 0.15
        self.base_max_positions = 6
        self.max_positions      = 6
        self.min_notional       = 5.5
        self.max_position_usd   = self._get_param("max_position_usd", 1500.0)
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

        self.max_spread_pct         = 0.005
        self.spread_median_window   = 12
        self.spread_widen_mult      = 2.5
        self.min_dollar_volume_usd  = 50000
        self.min_volume_usd         = 10000000

        self.skip_hours_utc         = []
        self.max_daily_trades       = 24
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 1.0
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 4

        self.expected_round_trip_fees = 0.0020
        self.fee_slippage_buffer      = 0.001
        self.min_expected_profit_pct  = 0.010
        self.adx_min_period           = 14

        self.stale_order_timeout_seconds      = 30
        self.live_stale_order_timeout_seconds = 60
        self.max_concurrent_open_orders       = 2
        self.open_orders_cash_threshold       = 0.5
        self.order_fill_check_threshold_seconds = 60
        self.order_timeout_seconds              = 30
        self.resync_log_interval_seconds        = 1800
        self.portfolio_mismatch_threshold       = 0.10
        self.portfolio_mismatch_min_dollars     = 1.00
        self.portfolio_mismatch_cooldown_seconds = 3600
        self.retry_pending_cooldown_seconds     = 60
        self.rate_limit_cooldown_minutes        = 10

        self.max_drawdown_limit    = 0.25
        self.cooldown_hours        = 6
        self.consecutive_losses    = 0
        self.max_consecutive_losses = 5
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
        self.lowest_prices    = {}
        self.position_directions = {}
        self.entry_times      = {}
        self.entry_volumes    = {}
        self._partial_tp_taken      = {}
        self._breakeven_stops       = {}
        self._partial_sell_symbols  = set()
        self.partial_tp_threshold   = 0.025
        self.stagnation_minutes     = 45
        self.stagnation_pnl_threshold = 0.005
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

        self.bybit_status = "unknown"
        self._last_skip_reason = None

        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(CryptoUniverse.Bybit(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSDT", Resolution.Minute, Market.Bybit)
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

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MICRO-SCALP BYBIT) v7.1.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.CashBook['USDT'].Amount:.2f} | Mode: Long+Short")

    def CustomSecurityInitializer(self, security):
        """Sets slippage, fee model, AND MARGIN CAPABILITY."""
        security.SetSlippageModel(RealisticCryptoSlippage())
        security.SetFeeModel(MakerTakerFeeModel())
        # Add 3x leverage capability
        security.SetBuyingPowerModel(SecurityMarginModel(3.0))

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

    def _record_exit_pnl(self, symbol, entry_price, exit_price):
        return record_exit_pnl(self, symbol, entry_price, exit_price)

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
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
            if not ticker.endswith("USDT"):  # Bybit uses USDT for margin pairs mostly
                continue
            base = ticker[:-4]  # remove "USDT" suffix
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
            'opens': deque(maxlen=self.lookback),
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
                self.Debug(f"FORCED EXIT: {symbol.Value} - removed from universe")
            if symbol in self.crypto_data and not is_invested_not_dust(self, symbol):
                del self.crypto_data[symbol]

    def OnData(self, data):
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
            if self.bybit_status == "unknown":
                self.bybit_status = "online"
                self.Debug("Fallback: bybit_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        self._update_market_context()
        self.Rebalance()
        self.CheckExits()

    def _update_symbol_data(self, symbol, bar, quote_bar=None):
        crypto = self.crypto_data[symbol]
        price = float(bar.Close)
        open_price = float(bar.Open)
        high = float(bar.High)
        low = float(bar.Low)
        volume = float(bar.Volume)
        crypto['prices'].append(price)
        crypto['opens'].append(open_price)
        crypto['highs'].append(high)
        crypto['lows'].append(low)
        if crypto['last_price'] > 0:
            ret = (price - crypto['last_price']) / crypto['last_price']
            crypto['returns'].append(ret)
        crypto['last_price'] = price
        crypto['volume'].append(volume)
        crypto['dollar_volume'].append(price * volume)
        if len(crypto['volume']) >= self.short_period:
            crypto['volume_ma'].append(np.mean(list(crypto['volume'])[-self.short_period:]))
        crypto['ema_ultra_short'].Update(bar.EndTime, price)
        crypto['ema_short'].Update(bar.EndTime, price)
        crypto['ema_medium'].Update(bar.EndTime, price)
        crypto['ema_5'].Update(bar.EndTime, price)
        crypto['atr'].Update(bar)
        crypto['adx'].Update(bar)
        crypto['vwap_pv'].append(price * volume)
        crypto['vwap_v'].append(volume)
        total_v = sum(crypto['vwap_v'])
        if total_v > 0:
            crypto['vwap'] = sum(crypto['vwap_pv']) / total_v
        crypto['volume_long'].append(volume)
        if len(crypto['vwap_v']) >= 5 and crypto['vwap'] > 0:
            vwap_val = crypto['vwap']
            pv_list = list(crypto['vwap_pv'])
            v_list = list(crypto['vwap_v'])
            bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
            if len(bar_prices) >= 5:
                sd = float(np.std(bar_prices))
                crypto['vwap_sd'] = sd
                crypto['vwap_sd2_lower'] = vwap_val - 2.0 * sd
                crypto['vwap_sd3_lower'] = vwap_val - 3.0 * sd
        if len(crypto['returns']) >= 10:
            crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))
        crypto['rsi'].Update(bar.EndTime, price)
        if len(crypto['returns']) >= self.short_period and len(self.btc_returns) >= self.short_period:
            coin_ret = np.sum(list(crypto['returns'])[-self.short_period:])
            btc_ret = np.sum(list(self.btc_returns)[-self.short_period:])
            crypto['rs_vs_btc'].append(coin_ret - btc_ret)
        if len(crypto['prices']) >= self.medium_period:
            prices_arr = np.array(list(crypto['prices'])[-self.medium_period:])
            std = np.std(prices_arr)
            mean = np.mean(prices_arr)
            if std > 0:
                crypto['zscore'].append((price - mean) / std)
                crypto['bb_upper'].append(mean + 2 * std)
                crypto['bb_lower'].append(mean - 2 * std)
                crypto['bb_width'].append(4 * std / mean if mean > 0 else 0)
        high_low = high - low
        if high_low > 0:
            bar_delta = volume * ((price - low) - (high - price)) / high_low
        else:
            bar_delta = 0.0
        prev_cvd = crypto['cvd'][-1] if len(crypto['cvd']) > 0 else 0.0
        crypto['cvd'].append(prev_cvd + bar_delta)
        if len(crypto['prices']) >= 15:
            price_change = abs(crypto['prices'][-1] - crypto['prices'][-15])
            volatility_sum = sum(abs(crypto['prices'][i] - crypto['prices'][i-1]) for i in range(-14, 0))
            if volatility_sum > 0:
                crypto['ker'].append(price_change / volatility_sum)
            else:
                crypto['ker'].append(0.0)
        Q = 1e-5
        R = 0.01
        if crypto['kalman_estimate'] == 0.0:
            crypto['kalman_estimate'] = price
        estimate_pred = crypto['kalman_estimate']
        error_cov_pred = crypto['kalman_error_cov'] + Q
        kalman_gain = error_cov_pred / (error_cov_pred + R)
        crypto['kalman_estimate'] = estimate_pred + kalman_gain * (price - estimate_pred)
        crypto['kalman_error_cov'] = (1 - kalman_gain) * error_cov_pred
        sp = get_spread_pct(self, symbol)
        if sp is not None:
            crypto['spreads'].append(sp)
        if quote_bar is not None:
            try:
                bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
                ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
                if bid_sz > 0 or ask_sz > 0:
                    crypto['bid_size'] = bid_sz
                    crypto['ask_size'] = ask_sz
            except Exception:
                pass

    def _update_market_context(self):
        if len(self.btc_prices) >= 48:
            btc_arr = np.array(list(self.btc_prices))
            current_btc = btc_arr[-1]
            btc_mom_12 = np.mean(list(self.btc_returns)[-12:]) if len(self.btc_returns) >= 12 else 0.0
            btc_sma = np.mean(btc_arr[-48:])
            if current_btc > btc_sma * 1.02:
                new_regime = "bull"
            elif current_btc < btc_sma * 0.98:
                new_regime = "bear"
            else:
                new_regime = "sideways"
            if new_regime == "sideways" and len(self.btc_returns) >= 12:
                if btc_mom_12 > 0.0001:
                    new_regime = "bull"
                elif btc_mom_12 < -0.0001:
                    new_regime = "bear"
            if new_regime != self.market_regime:
                self._regime_hold_count += 1
                if self._regime_hold_count >= 3:
                    self.market_regime = new_regime
                    self._regime_hold_count = 0
            else:
                self._regime_hold_count = 0
        if len(self.btc_volatility) >= 5:
            current_vol = self.btc_volatility[-1]
            avg_vol = np.mean(list(self.btc_volatility))
            if current_vol > avg_vol * 1.5:
                self.volatility_regime = "high"
            elif current_vol < avg_vol * 0.5:
                self.volatility_regime = "low"
            else:
                self.volatility_regime = "normal"
        uptrend_count = 0
        total_ready = 0
        for crypto in self.crypto_data.values():
            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                total_ready += 1
                if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                    uptrend_count += 1
        if total_ready > 5:
            self.market_breadth = uptrend_count / total_ready

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
        """Evaluate both signals and prioritize Longs over Shorts."""
        long_score, long_components = self._scoring_engine.calculate_scalp_score(crypto)
        short_score, short_components = self._scoring_engine.calculate_short_score(crypto)

        threshold = self._get_threshold()

        # MG FIRST: Always prioritize long setups if they meet the threshold
        if long_score >= threshold:
            components = long_components.copy()
            components['_scalp_score'] = long_score
            components['_direction'] = 1
            components['_long_score'] = long_score
            components['_short_score'] = 0.0
            return components

        # FALLBACK: If no valid long, check for a valid short ONLY if regime is bear
        elif short_score >= threshold and self.market_regime == "bear":
            components = short_components.copy()
            components['_scalp_score'] = short_score
            components['_direction'] = -1
            components['_long_score'] = 0.0
            components['_short_score'] = short_score
            return components

        return None

    def _calculate_composite_score(self, factors, crypto=None):
        return factors.get('_scalp_score', 0.0)

    def _apply_fee_adjustment(self, score):
        return score

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        return self._scoring_engine.calculate_position_size(score, threshold, asset_vol_ann)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    def _get_max_daily_trades(self):
        return self.max_daily_trades

    def _get_threshold(self):
        return self.entry_threshold

    def _check_correlation(self, new_symbol):
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

    def Rebalance(self):
        if self.IsWarmingUp:
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
        if self.LiveMode and self.bybit_status in ("maintenance", "cancel_only"):
            self._log_skip("bybit not online")
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
            self.drawdown_cooldown = 3
            self._consecutive_loss_halve_remaining = 3
            self.consecutive_losses = 0
            self._log_skip("consecutive loss cooldown (5 losses)")
            return
        if self.consecutive_losses >= 3:
            self.circuit_breaker_expiry = self.Time + timedelta(hours=12)
            self.consecutive_losses = 0
            self._log_skip("circuit breaker triggered (3 consecutive losses)")
            return
        if self.circuit_breaker_expiry is not None and self.Time < self.circuit_breaker_expiry:
            self._log_skip("circuit breaker active")
            return
        dynamic_max_pos = min(4, max(self.base_max_positions, int(val // self.max_position_usd) + 1))
        pos_count = get_actual_position_count(self)
        if pos_count >= dynamic_max_pos:
            self._log_skip("at max positions")
            return
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
            return

        count_not_blacklisted = 0
        count_no_open_orders = 0
        count_spread_ok = 0
        count_ready = 0
        count_scored = 0
        count_above_thresh = 0

        scores = []
        threshold_now = self._get_threshold()
        for symbol in list(self.crypto_data.keys()):
            if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in self._session_blacklist:
                continue
            count_not_blacklisted += 1

            if has_open_orders(self, symbol):
                continue
            count_no_open_orders += 1

            if not spread_ok(self, symbol):
                continue
            count_spread_ok += 1

            crypto = self.crypto_data[symbol]
            if not self._is_ready(crypto):
                continue
            count_ready += 1

            factor_scores = self._calculate_factor_scores(symbol, crypto)
            if not factor_scores:
                continue
            count_scored += 1

            composite_score = self._calculate_composite_score(factor_scores, crypto)
            net_score = self._apply_fee_adjustment(composite_score)

            crypto['recent_net_scores'].append(net_score)

            if net_score >= threshold_now:
                count_above_thresh += 1
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'net_score': net_score,
                    'direction': factor_scores.get('_direction', 1),
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
                })

        try:
            cash = self.Portfolio.CashBook["USDT"].Amount
        except (KeyError, AttributeError):
            cash = self.Portfolio.TotalPortfolioValue

        debug_limited(self, f"REBALANCE: {count_above_thresh}/{count_scored} above thresh={threshold_now:.2f} | cash=${cash:.2f}")

        if len(scores) == 0:
            self._log_skip("no candidates passed filters")
            return
        scores.sort(key=lambda x: x['net_score'], reverse=True)
        self._last_skip_reason = None
        self._execute_trades(scores, threshold_now, dynamic_max_pos)

    def _get_open_buy_orders_value(self):
        return get_open_buy_orders_value(self)

    def _execute_trades(self, candidates, threshold_now, dynamic_max_pos):
        if not self._positions_synced:
            return
        if self.LiveMode and self.bybit_status in ("maintenance", "cancel_only"):
            return
        cancel_stale_new_orders(self)
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            return

        try:
            available_cash = self.Portfolio.CashBook["USDT"].Amount
        except (KeyError, AttributeError):
            available_cash = self.Portfolio.TotalPortfolioValue

        open_buy_orders_value = self._get_open_buy_orders_value()

        if available_cash <= 0:
            debug_limited(self, f"SKIP TRADES: No cash available (${available_cash:.2f})")
            return
        if open_buy_orders_value > available_cash * self.open_orders_cash_threshold:
            debug_limited(self, f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved (>{self.open_orders_cash_threshold:.0%} of ${available_cash:.2f})")
            return

        reject_pending_orders = 0
        reject_open_orders = 0
        reject_already_invested = 0
        reject_spread = 0
        reject_exit_cooldown = 0
        reject_loss_cooldown = 0
        reject_price_invalid = 0
        reject_price_too_low = 0
        reject_cash_reserve = 0
        reject_min_qty_too_large = 0
        reject_dollar_volume = 0
        reject_notional = 0
        success_count = 0

        for cand in candidates:
            if self.daily_trade_count >= self._get_max_daily_trades():
                break
            if get_actual_position_count(self) >= dynamic_max_pos:
                break
            sym = cand['symbol']
            net_score = cand.get('net_score', 0.5)
            if sym in self._pending_orders and self._pending_orders[sym] > 0:
                reject_pending_orders += 1
                continue
            if has_open_orders(self, sym):
                reject_open_orders += 1
                continue
            if is_invested_not_dust(self, sym):
                reject_already_invested += 1
                continue
            if not spread_ok(self, sym):
                reject_spread += 1
                continue
            if sym in self._exit_cooldowns and self.Time < self._exit_cooldowns[sym]:
                reject_exit_cooldown += 1
                continue
            if sym in self._symbol_loss_cooldowns and self.Time < self._symbol_loss_cooldowns[sym]:
                reject_loss_cooldown += 1
                continue
            sec = self.Securities[sym]
            price = sec.Price
            if price is None or price <= 0:
                reject_price_invalid += 1
                continue
            if price < self.min_price_usd:
                reject_price_too_low += 1
                continue

            try:
                available_cash = self.Portfolio.CashBook["USDT"].Amount
            except (KeyError, AttributeError):
                available_cash = self.Portfolio.TotalPortfolioValue

            available_cash = max(0, available_cash - open_buy_orders_value)
            total_value = self.Portfolio.TotalPortfolioValue
            fee_reserve = max(total_value * 0.01, 0.10)
            reserved_cash = available_cash - fee_reserve
            if reserved_cash <= 0:
                reject_cash_reserve += 1
                continue

            min_qty = get_min_quantity(self, sym)
            min_notional_usd = get_min_notional_usd(self, sym)
            if min_qty * price > reserved_cash * 0.90:
                reject_min_qty_too_large += 1
                continue

            crypto = self.crypto_data.get(sym)
            if not crypto:
                continue

            if crypto.get('trade_count_today', 0) >= self.max_symbol_trades_per_day:
                continue

            atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
            if atr_val and price > 0:
                expected_move_pct = (atr_val * self.atr_tp_mult) / price
                min_profit_gate = self.min_expected_profit_pct
                min_required = self.expected_round_trip_fees + self.fee_slippage_buffer + min_profit_gate
                if expected_move_pct < min_required:
                    continue

            if len(crypto['dollar_volume']) >= 3:
                recent_dv = np.mean(list(crypto['dollar_volume'])[-3:])
                dv_threshold = self.min_dollar_volume_usd
                if recent_dv < dv_threshold:
                    reject_dollar_volume += 1
                    continue

            vol = self._annualized_vol(crypto)
            size = self._calculate_position_size(net_score, threshold_now, vol)

            if self._consecutive_loss_halve_remaining > 0:
                size *= 0.50

            if self.volatility_regime == "high":
                size = min(size * 1.1, self.position_size_pct)

            slippage_penalty = get_slippage_penalty(self, sym)
            size *= slippage_penalty

            val = reserved_cash * size

            val = max(val, self.min_notional)

            if len(crypto['dollar_volume']) >= 3:
                whale_cap = sum(list(crypto['dollar_volume'])[-3:]) * 0.02
                val = min(val, whale_cap)

            val = min(val, self.max_position_usd)

            qty = round_quantity(self, sym, val / price)
            if qty < min_qty:
                qty = round_quantity(self, sym, min_qty)
                val = qty * price
            total_cost_with_fee = val * 1.006
            if total_cost_with_fee > available_cash:
                reject_cash_reserve += 1
                continue
            if val < min_notional_usd * self.min_notional_fee_buffer or val < self.min_notional or val > reserved_cash:
                reject_notional += 1
                continue

            try:
                sec = self.Securities[sym]
                min_order_size = float(sec.SymbolProperties.MinimumOrderSize or 0)
                lot_size = float(sec.SymbolProperties.LotSize or 0)
                actual_min = max(min_order_size, lot_size)
                if actual_min > 0 and qty < actual_min:
                    self.Debug(f"REJECT ENTRY {sym.Value}: qty={qty} < min_order_size={actual_min} (unsellable)")
                    reject_notional += 1
                    continue
                if min_order_size > 0:
                    post_fee_qty = qty * (1.0 - BYBIT_SELL_FEE_BUFFER)
                    if post_fee_qty < min_order_size:
                        required_qty = round_quantity(self, sym, min_order_size / (1.0 - BYBIT_SELL_FEE_BUFFER))
                        if required_qty * price <= available_cash * 0.99:
                            qty = required_qty
                            val = qty * price
                        else:
                            self.Debug(f"REJECT ENTRY {sym.Value}: post-fee qty={post_fee_qty:.6f} < min_order_size={min_order_size}")
                            reject_notional += 1
                            continue
            except Exception as e:
                self.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

            try:
                direction = cand.get('direction', 1)
                dir_label = "LONG" if direction == 1 else "SHORT"
                if self.LiveMode:
                    signed_qty = qty if direction == 1 else -qty
                    ticket = place_limit_or_market(self, sym, signed_qty, timeout_seconds=30, tag=f"Entry {dir_label}")
                else:
                    signed_qty = qty if direction == 1 else -qty
                    ticket = self.MarketOrder(sym, signed_qty, tag=f"Entry {dir_label}")
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    self.position_directions[sym] = direction
                    components = cand.get('factors', {})
                    sig_str = (f"obi={components.get('obi', 0):.2f} "
                               f"vol={components.get('vol_ignition', 0):.2f} "
                               f"trend={components.get('micro_trend', 0):.2f} "
                               f"adx={components.get('adx_filter', 0):.2f} "
                               f"vwap={components.get('vwap_signal', 0):.2f}")
                    self.Debug(f"SCALP {dir_label}: {sym.Value} | score={net_score:.2f} | ${val:.2f} | {sig_str}")
                    success_count += 1
                    self.trade_count += 1
                    crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                    if self._consecutive_loss_halve_remaining > 0:
                        self._consecutive_loss_halve_remaining -= 1
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {sym.Value} - {e}")
                self._session_blacklist.add(sym.Value)
                continue
            if self.LiveMode and success_count >= 1:
                break

        if success_count > 0 or (reject_exit_cooldown + reject_loss_cooldown) > 3:
            debug_limited(self, f"EXECUTE: {success_count}/{len(candidates)} rejects: cd={reject_exit_cooldown} loss={reject_loss_cooldown} dv={reject_dollar_volume}")

    def _is_ready(self, c):
        return len(c['prices']) >= 10 and c['rsi'].IsReady

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
                direction = self.position_directions.get(symbol, 1 if kvp.Value.Quantity > 0 else -1)
                if direction == -1:
                    self.lowest_prices[symbol] = kvp.Value.AveragePrice
                else:
                    self.highest_prices[symbol] = kvp.Value.AveragePrice
                self.entry_times[symbol] = self.Time
                self.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")

    def _check_exit(self, symbol, price, holding):
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            return

        min_notional_usd = get_min_notional_usd(self, symbol)
        if price > 0 and abs(holding.Quantity) * price < min_notional_usd * 0.3:
            try:
                self.Liquidate(symbol)
            except Exception as e:
                self.Debug(f"DUST liquidation failed for {symbol.Value}: {e}")
            cleanup_position(self, symbol)
            self._failed_exit_counts.pop(symbol, None)
            return

        actual_qty = abs(holding.Quantity)
        rounded_sell = round_quantity(self, symbol, actual_qty)
        if rounded_sell > actual_qty:
            self.Debug(f"DUST: {symbol.Value} actual={actual_qty} rounded={rounded_sell}")
            cleanup_position(self, symbol)
            self._failed_exit_counts.pop(symbol, None)
            return

        direction = self.position_directions.get(symbol, 1 if holding.Quantity > 0 else -1)
        is_short = (direction == -1)

        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = holding.AveragePrice
            if is_short:
                self.lowest_prices[symbol] = holding.AveragePrice
            else:
                self.highest_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time

        entry = self.entry_prices[symbol]

        if is_short:
            if symbol not in self.lowest_prices:
                self.lowest_prices[symbol] = entry
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

        crypto = self.crypto_data.get(symbol)
        hours = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        minutes = hours * 60

        atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None
        if atr and entry > 0:
            sl = max((atr * self.atr_sl_mult) / entry, self.tight_stop_loss)
            tp = max((atr * self.atr_tp_mult) / entry, self.quick_take_profit)
        else:
            sl = self.tight_stop_loss
            tp = self.quick_take_profit

        if tp < sl * 1.5:
            tp = sl * 1.5

        trailing_activation = self.trail_activation
        trailing_stop_pct   = self.trail_stop_pct

        if (not is_short
                and not self._partial_tp_taken.get(symbol, False)
                and pnl >= self.partial_tp_threshold):
            if partial_smart_sell(self, symbol, 0.50, "Partial TP"):
                self._partial_tp_taken[symbol] = True
                self._breakeven_stops[symbol] = entry * 1.002
                self.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL→entry+0.2%")
                return  # Don't trigger full exit this bar

        tag = ""

        if not is_short and self._partial_tp_taken.get(symbol, False):
            be_price = self._breakeven_stops.get(symbol, entry)
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
                    lowest_ref = self.lowest_prices.get(symbol, entry)
                    trail_level = lowest_ref + trail_offset
                    if crypto:
                        crypto['trail_stop'] = trail_level
                    if crypto and crypto['trail_stop'] is not None and price >= crypto['trail_stop']:
                        tag = "ATR Trail"
                else:
                    highest_ref = self.highest_prices.get(symbol, entry)
                    trail_level = highest_ref - trail_offset
                    if crypto:
                        crypto['trail_stop'] = trail_level
                    if crypto and crypto['trail_stop'] is not None and price <= crypto['trail_stop']:
                        tag = "ATR Trail"

            if not tag and hours >= 2.0 and crypto and len(crypto['volume']) >= 2:
                entry_vol = self.entry_volumes.get(symbol, 0)
                if entry_vol > 0:
                    v1 = crypto['volume'][-1]
                    v2 = crypto['volume'][-2]
                    if v1 < entry_vol * 0.50 and v2 < entry_vol * 0.50:
                        tag = "Volume Dry-up"

            if not tag and hours >= self.time_stop_hours and pnl < self.time_stop_pnl_min:
                tag = "Time Stop"

            if not tag and hours >= self.extended_time_stop_hours and pnl < self.extended_time_stop_pnl_max:
                tag = "Extended Time Stop"

            if not tag and hours >= self.stale_position_hours:
                tag = "Stale Position Exit"

        if tag:
            if price * abs(holding.Quantity) < min_notional_usd * 0.9:
                return
            if pnl < 0:
                self._symbol_loss_cooldowns[symbol] = self.Time + timedelta(hours=1)
            sold = smart_liquidate(self, symbol, tag)
            if sold:
                self._exit_cooldowns[symbol] = self.Time + timedelta(hours=self.exit_cooldown_hours)
                self.entry_volumes.pop(symbol, None)
                dir_label = "SHORT" if is_short else "LONG"
                self.Debug(f"{tag} ({dir_label}): {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
            else:

                self.Debug(f"⚠️ EXIT FAILED ({tag}): {symbol.Value} | PnL:{pnl:+.2%}")
                cleanup_position(self, symbol)
                self.entry_volumes.pop(symbol, None)

    def _check_cash_mode(self):
        """Pause trading 24h if recent win rate < 25%."""
        oto = self._recent_trade_outcomes
        if len(oto) >= 8:
            wr = sum(oto) / len(oto)
            if wr < 0.25:
                self._cash_mode_until = self.Time + timedelta(hours=24)
                self.Debug(f"CASH MODE: WR={wr:.0%}. Pausing 24h.")

    def OnOrderEvent(self, event):
        try:
            symbol = event.Symbol
            self.Debug(f"ORDER: {symbol.Value} {event.Status} qty={event.FillQuantity or event.Quantity} price={event.FillPrice}")
            if event.Status == OrderStatus.Submitted:
                if symbol not in self._pending_orders:
                    self._pending_orders[symbol] = 0
                intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
                self._pending_orders[symbol] += intended_qty
                if symbol not in self._submitted_orders:
                    has_position = symbol in self.Portfolio and self.Portfolio[symbol].Invested
                    if event.Direction == OrderDirection.Sell and has_position:
                        inferred_intent = 'exit'
                    elif event.Direction == OrderDirection.Buy and not has_position:
                        inferred_intent = 'entry'
                    else:
                        inferred_intent = 'entry' if event.Direction == OrderDirection.Buy else 'exit'

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
                if event.Direction == OrderDirection.Buy:
                    if symbol not in self.entry_prices:
                        self.entry_prices[symbol] = event.FillPrice
                        self.highest_prices[symbol] = event.FillPrice
                        self.entry_times[symbol] = self.Time
                elif event.Direction == OrderDirection.Sell:
                    if symbol not in self.entry_prices and self.position_directions.get(symbol, 1) == -1:
                        self.entry_prices[symbol] = event.FillPrice
                        self.lowest_prices[symbol] = event.FillPrice
                        self.entry_times[symbol] = self.Time
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)

                current_direction = self.position_directions.get(symbol, None)
                portfolio_qty = self.Portfolio[symbol].Quantity if symbol in self.Portfolio else 0

                if event.Direction == OrderDirection.Buy:
                    if current_direction == -1:
                        entry = self.entry_prices.get(symbol, None)
                        if entry is None:
                            entry = event.FillPrice
                            self.Debug(f"⚠️ Missing entry price for {symbol.Value}, using fill price")
                        pnl = (entry - event.FillPrice) / entry if entry > 0 else 0
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
                        self.trade_log.append({
                            'time': self.Time,
                            'symbol': symbol.Value,
                            'pnl_pct': pnl,
                            'direction': 'short',
                            'exit_reason': 'cover_buy',
                        })
                        self._check_cash_mode()
                        cleanup_position(self, symbol)
                        self._failed_exit_attempts.pop(symbol, None)
                        self._failed_exit_counts.pop(symbol, None)
                    else:
                        self.entry_prices[symbol] = event.FillPrice
                        self.highest_prices[symbol] = event.FillPrice
                        self.entry_times[symbol] = self.Time
                        self.position_directions[symbol] = 1
                        self.daily_trade_count += 1
                        crypto = self.crypto_data.get(symbol)
                        if crypto and len(crypto['volume']) >= 1:
                            self.entry_volumes[symbol] = crypto['volume'][-1]
                else:
                    if symbol in self.entry_prices and current_direction != -1:
                        if symbol in self._partial_sell_symbols:
                            self._partial_sell_symbols.discard(symbol)
                        else:
                            entry = self.entry_prices.get(symbol, None)
                            if entry is None:
                                entry = event.FillPrice
                                self.Debug(f"⚠️ Missing entry price for {symbol.Value}, using fill price")
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
                            self.trade_log.append({
                                'time': self.Time,
                                'symbol': symbol.Value,
                                'pnl_pct': pnl,
                                'direction': 'long',
                                'exit_reason': 'filled_sell',
                            })
                            self._check_cash_mode()
                            cleanup_position(self, symbol)
                            self._failed_exit_attempts.pop(symbol, None)
                            self._failed_exit_counts.pop(symbol, None)
                    else:
                        self.entry_prices[symbol] = event.FillPrice
                        self.lowest_prices[symbol] = event.FillPrice
                        self.entry_times[symbol] = self.Time
                        self.position_directions[symbol] = -1
                        self.daily_trade_count += 1
                        crypto = self.crypto_data.get(symbol)
                        if crypto and len(crypto['volume']) >= 1:
                            self.entry_volumes[symbol] = crypto['volume'][-1]
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if symbol not in self.entry_prices:
                    if is_invested_not_dust(self, symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
                        direction = self.position_directions.get(symbol, 1 if holding.Quantity > 0 else -1)
                        if direction == -1:
                            self.lowest_prices[symbol] = holding.AveragePrice
                        else:
                            self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = self.Time
                        self.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if event.Direction == OrderDirection.Sell:
                    price = self.Securities[symbol].Price if symbol in self.Securities else 0
                    min_notional = get_min_notional_usd(self, symbol)

                    if price > 0 and symbol in self.Portfolio and abs(self.Portfolio[symbol].Quantity) * price < min_notional:
                        self.Debug(f"DUST CLEANUP: {symbol.Value}")
                        cleanup_position(self, symbol)
                        self._failed_exit_counts.pop(symbol, None)
                    else:

                        fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                        self._failed_exit_counts[symbol] = fail_count
                        self.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                        if fail_count >= 3:

                            self.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} invalid exits")
                            cleanup_position(self, symbol)
                            self._failed_exit_counts.pop(symbol, None)
                        elif symbol not in self.entry_prices:
                            if is_invested_not_dust(self, symbol):
                                holding = self.Portfolio[symbol]
                                self.entry_prices[symbol] = holding.AveragePrice
                                direction = self.position_directions.get(symbol, 1 if holding.Quantity > 0 else -1)
                                if direction == -1:
                                    self.lowest_prices[symbol] = holding.AveragePrice
                                else:
                                    self.highest_prices[symbol] = holding.AveragePrice
                                self.entry_times[symbol] = self.Time
                                self.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
                self._session_blacklist.add(symbol.Value)
        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")
        if self.LiveMode:
            persist_state(self)

    def OnBrokerageMessage(self, message):
        try:
            txt = message.Message.lower()
            if "system status:" in txt:
                if "online" in txt:
                    self.bybit_status = "online"
                elif "maintenance" in txt:
                    self.bybit_status = "maintenance"
                elif "cancel_only" in txt:
                    self.bybit_status = "cancel_only"
                elif "post_only" in txt:
                    self.bybit_status = "post_only"
                else:
                    self.bybit_status = "unknown"
                self.Debug(f"Bybit status update: {self.bybit_status}")

            if "rate limit" in txt or "too many" in txt:
                self.Debug(f"⚠️ RATE LIMIT - pausing {self.rate_limit_cooldown_minutes}min")
                self._rate_limit_until = self.Time + timedelta(minutes=self.rate_limit_cooldown_minutes)
                self._last_live_trade_time = self.Time
        except Exception as e:
            self.Debug(f"BrokerageMessage parse error: {e}")

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
