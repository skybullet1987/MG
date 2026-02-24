# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
# endregion


class SimplifiedCryptoStrategy(QCAlgorithm):
    """
    Micro-Scalping System - v7.1.0
    5-signal micro-scalp engine, regime-adaptive, bull/sideways/bear-aware.
    """

    def Initialize(self):
        self.SetStartDate(2025, 1, 1)
        self.SetCash(1000)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        # === Entry thresholds (scalp score 0-1) ===
        self.entry_threshold = 0.60   # 3/5 signals firing (raised from 0.40 to improve win rate)
        self.high_conviction_threshold = 0.80  # 4+ signals (raised from 0.60)

        # === Exit parameters (aggressive profit-taking) ===
        self.quick_take_profit = self._get_param("quick_take_profit", 0.020)  # 2.0% base TP
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.012)  # 1.2% base SL
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  2.5)   # ATR × 2.5 for TP
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  1.5)   # ATR × 1.5 for SL
        self.trail_activation  = self._get_param("trail_activation",  0.005)  # activate at +0.5%
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.005)  # trail 0.5% from high
        self.time_stop_hours   = self._get_param("time_stop_hours",   4.0)    # exit after 4h if PnL < +0.3%
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.003)  # +0.3% floor
        self.extended_time_stop_hours   = self._get_param("extended_time_stop_hours",   6.0)   # exit after 6h if not clearly winning
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.015) # +1.5% ceiling
        self.stale_position_hours       = self._get_param("stale_position_hours",       8.0)   # unconditional exit after 8h

        # Keep legacy names used elsewhere
        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 2.0

        # === Position sizing (aggressive compounding) ===
        self.position_size_pct  = 0.90   # minimum base size; scoring engine scales to 99% at full conviction
        self.base_max_positions = 1
        self.max_positions      = 1
        self.min_notional       = 5.0
        self.max_position_usd   = self._get_param("max_position_usd", 1500.0)
        self.min_price_usd      = 0.005
        self.cash_reserve_pct   = 0.0    # no dead-money reserve at $20
        # 50% buffer ensures post-fee qty stays above MinimumOrderSize.
        self.min_notional_fee_buffer = 1.5

        # === Volatility / risk targets (kept for ATR sizing) ===
        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap", 0.80)
        self.min_asset_vol_floor     = 0.05

        # === Indicator periods (optimised for 1-minute scalping) ===
        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12   # was 24
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(60 * 24 * 365)  # minute-resolution annualisation

        # === Liquidity filters ===
        self.max_spread_pct         = 0.005   # 0.5% – tight spread required
        self.spread_median_window   = 12
        self.spread_widen_mult      = 2.5
        self.min_dollar_volume_usd  = 50000   # $50k/hour minimum (checked via 3h avg in execute)
        self.min_volume_usd         = 10000000  # $10M minimum VolumeInUsd for universe filter

        # === Trade frequency & timing ===
        self.skip_hours_utc         = []      # 24/7 trading – no skip hours
        self.max_daily_trades       = 24      # increased to allow more opportunities
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 1.0     # 1-hour exit cooldown (raised from 15-min)
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 4    # increased from 2 to allow more per-symbol trades

        # === Fees & slippage ===
        self.expected_round_trip_fees = 0.0050   # 0.50% min (maker+maker)
        self.fee_slippage_buffer      = 0.001
        self.min_expected_profit_pct  = 0.015    # 1.5% minimum net profit above fees+slippage
        self.adx_min_period           = 14       # ADX indicator period

        # === Order management ===
        self.stale_order_timeout_seconds      = 30    # 30s limit-entry timeout
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

        # === Risk management ===
        self.max_drawdown_limit    = 0.25   # 25% – pause 6h
        self.cooldown_hours        = 6
        self.consecutive_losses    = 0
        self.max_consecutive_losses = 5    # pause + halve size for next 5 trades after 5 losses
        self._consecutive_loss_halve_remaining = 0
        self.circuit_breaker_expiry = None  # halt new entries for 12h after 3 consecutive losses

        # === State ===
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
        self._failed_exit_attempts = {}  # tracks consecutive Invalid sell orders per symbol
        self._failed_exit_counts   = {}  # tracks failed exit attempts for dust-loop prevention

        self.peak_value       = None
        self.drawdown_cooldown = 0
        self.crypto_data      = {}
        self.entry_prices     = {}
        self.highest_prices   = {}
        self.entry_times      = {}
        self.entry_volumes    = {}   # for volume dry-up exit
        self._partial_tp_taken      = {}       # True after 50% sold at +2.5%
        self._breakeven_stops       = {}       # per-symbol breakeven SL price after partial TP
        self._partial_sell_symbols  = set()    # symbols with in-flight partial TP sell orders
        self.partial_tp_threshold   = 0.025    # +2.5% triggers partial scale-out
        self.stagnation_minutes     = 45       # stagnation exit: trade must be > 45 min old
        self.stagnation_pnl_threshold = 0.005  # stagnation exit: pnl must be >= 0.5%
        self.rsi_peaked_above_50 = {}  # for RSI momentum exit
        self.trade_count      = 0
        self._pending_orders  = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns  = {}
        self._symbol_loss_cooldowns = {}
        self._cash_mode_until = None
        self._recent_trade_outcomes = deque(maxlen=20)
        self.trailing_grace_hours = 1  # reduced – allow trailing after 1h
        self._slip_abs        = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold  = 0.0015
        self.slip_outlier_threshold = 0.004
        self.slip_alert_duration_hours = 2
        self._bad_symbol_counts = {}
        self._recent_tickets  = deque(maxlen=25)

        # Rolling performance tracking (for Kelly)
        self._rolling_wins      = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None

        # Market context
        self.btc_symbol       = None
        self.btc_returns      = deque(maxlen=72)
        self.btc_prices       = deque(maxlen=72)
        self.btc_volatility   = deque(maxlen=72)
        self.btc_ema_24       = ExponentialMovingAverage(24)
        self.market_regime    = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth   = 0.5
        self._regime_hold_count = 0

        # Performance tracking
        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.trade_log      = []
        self.log_budget     = 0
        self.last_log_time  = None

        # Universe
        self.max_universe_size = 20

        # Kraken status gate
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
        self.SetSecurityInitializer(lambda security: security.SetSlippageModel(RealisticCryptoSlippage()))
        self.Settings.FreePortfolioValuePercentage = 0.01
        self.Settings.InsightScore = False

        self._scoring_engine = MicroScalpEngine(self)

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MICRO-SCALP) v7.0.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")

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
        """Helper to normalize order time by removing timezone info if present."""
        return normalize_order_time(order_time)
    
    def _record_exit_pnl(self, symbol, entry_price, exit_price):
        """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
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
            'adx': AverageDirectionalIndex(self.adx_min_period),  # ADX regime filter
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(7),   # RSI(7) for faster signals
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
            # Rolling VWAP: accumulate price*volume and volume over 20 bars
            'vwap_pv': deque(maxlen=20),  # price × volume
            'vwap_v': deque(maxlen=20),   # volume
            'vwap': 0.0,                  # current 20-bar rolling VWAP
            'volume_long': deque(maxlen=1440),  # ~24h volume for adaptive thresholds
            'vwap_sd': 0.0,                     # VWAP rolling standard deviation
            'vwap_sd2_lower': 0.0,              # VWAP - 2*SD lower band
            'vwap_sd3_lower': 0.0,              # VWAP - 3*SD lower band
            'cvd': deque(maxlen=self.lookback),  # Cumulative Volume Delta
            'ker': deque(maxlen=self.short_period),  # Kaufman Efficiency Ratio
            'kalman_estimate': 0.0,             # Kalman filter price estimate
            'kalman_error_cov': 1.0,            # Kalman filter error covariance
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
        crypto = self.crypto_data[symbol]
        price = float(bar.Close)
        high = float(bar.High)
        low = float(bar.Low)
        volume = float(bar.Volume)
        crypto['prices'].append(price)
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
        # Rolling 20-bar VWAP
        crypto['vwap_pv'].append(price * volume)
        crypto['vwap_v'].append(volume)
        total_v = sum(crypto['vwap_v'])
        if total_v > 0:
            crypto['vwap'] = sum(crypto['vwap_pv']) / total_v
        # Long-term volume baseline for adaptive scoring thresholds (~24h window)
        crypto['volume_long'].append(volume)
        # VWAP SD bands: compute std of bar prices within the rolling VWAP window
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
        # CVD: Tick Delta approximation
        high_low = high - low
        if high_low > 0:
            bar_delta = volume * ((price - low) - (high - price)) / high_low
        else:
            bar_delta = 0.0
        prev_cvd = crypto['cvd'][-1] if len(crypto['cvd']) > 0 else 0.0
        crypto['cvd'].append(prev_cvd + bar_delta)
        # KER: Kaufman Efficiency Ratio (15-period)
        if len(crypto['prices']) >= 15:
            price_change = abs(crypto['prices'][-1] - crypto['prices'][-15])
            volatility_sum = sum(abs(crypto['prices'][i] - crypto['prices'][i-1]) for i in range(-14, 0))
            if volatility_sum > 0:
                crypto['ker'].append(price_change / volatility_sum)
            else:
                crypto['ker'].append(0.0)
        # 1D Kalman Filter for price
        Q = 1e-5  # Process noise variance
        R = 0.01  # Measurement noise variance
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
        # Update bid/ask sizes from QuoteBar for Order Book Imbalance calculation
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
            # Keep momentum confirmation but make it more sensitive
            if new_regime == "sideways" and len(self.btc_returns) >= 12:
                if btc_mom_12 > 0.0001:
                    new_regime = "bull"
                elif btc_mom_12 < -0.0001:
                    new_regime = "bear"
            # Hysteresis: only change if held for 3+ bars
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
        """Delegate to MicroScalpEngine.  Returns a dict with scalp score components."""
        score, components = self._scoring_engine.calculate_scalp_score(crypto)
        components['_scalp_score'] = score
        return components

    def _calculate_composite_score(self, factors, crypto=None):
        """Return the pre-computed scalp score."""
        return factors.get('_scalp_score', 0.0)

    def _apply_fee_adjustment(self, score):
        """Return score unchanged – signal thresholds already require >1% moves."""
        return score

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        """Aggressive 70% base size, Kelly-adjusted, bear-halved."""
        return self._scoring_engine.calculate_position_size(score, threshold, asset_vol_ann)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    def _get_max_daily_trades(self):
        return self.max_daily_trades

    def _get_threshold(self):
        # In ranging/sideways markets, lower the entry threshold so that
        # mean-reversion setups (which score slightly below 0.60) can enter.
        if self.market_regime == "sideways":
            return max(0.50, self.entry_threshold - 0.10)
        elif self.market_regime == "bull":
            return max(0.45, self.entry_threshold - 0.15) # Relax threshold significantly in bull runs
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

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        
        # Cash mode — pause trading when recent performance is poor
        if self._cash_mode_until is not None and self.Time < self._cash_mode_until:
            self._log_skip("cash mode - poor recent performance")
            return
        
        # Reset log budget at each rebalance call for consistent logging
        self.log_budget = 20
        
        # Check rate limit hard block
        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            self._log_skip("rate limited")
            return
        
        # Live safety checks
        if self.LiveMode and not live_safety_checks(self):
            return
        # Only block on explicit bad states; unknown is allowed (and will have fallback after warmup)
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            self._log_skip("kraken not online")
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
            # Pause 3h and halve size for next 5 trades
            self.drawdown_cooldown = 3
            self._consecutive_loss_halve_remaining = 3
            self.consecutive_losses = 0
            self._log_skip("consecutive loss cooldown (5 losses)")
            return
        # Circuit breaker: halt new entries for 12h after 3 consecutive losses
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

        # Diagnostic counters for filter funnel
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

            # Store for diagnostic purposes
            crypto['recent_net_scores'].append(net_score)

            if net_score >= threshold_now:
                count_above_thresh += 1
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'net_score': net_score,
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
                })

        # Log diagnostic summary
        try:
            cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            cash = self.Portfolio.Cash

        debug_limited(self, f"REBALANCE: {count_above_thresh}/{count_scored} above thresh={threshold_now:.2f} | cash=${cash:.2f}")

        if len(scores) == 0:
            self._log_skip("no candidates passed filters")
            return
        scores.sort(key=lambda x: x['net_score'], reverse=True)
        self._last_skip_reason = None
        self._execute_trades(scores, threshold_now, dynamic_max_pos)

    def _get_open_buy_orders_value(self):
        """Calculate total value reserved by open buy orders."""
        return get_open_buy_orders_value(self)

    def _execute_trades(self, candidates, threshold_now, dynamic_max_pos):
        if not self._positions_synced:
            return
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            return
        cancel_stale_new_orders(self)
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            return
        
        try:
            available_cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = self.Portfolio.Cash
        
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
                available_cash = self.Portfolio.CashBook["USD"].Amount
            except (KeyError, AttributeError):
                available_cash = self.Portfolio.Cash

            available_cash = max(0, available_cash - open_buy_orders_value)
            total_value = self.Portfolio.TotalPortfolioValue
            # Minimal fee reserve only
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

            # Per-symbol daily trade limit
            if crypto.get('trade_count_today', 0) >= self.max_symbol_trades_per_day:
                continue

            # Fee-adjusted profit gate: ATR-projected move must cover fees + slippage.
            atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
            if atr_val and price > 0:
                expected_move_pct = (atr_val * self.atr_tp_mult) / price
                min_profit_gate = 0.008 if self.market_regime == "sideways" else self.min_expected_profit_pct
                min_required = self.expected_round_trip_fees + self.fee_slippage_buffer + min_profit_gate
                if expected_move_pct < min_required:
                    continue

            # Dollar-volume liquidity gate (relaxed in sideways/low-activity periods)
            if len(crypto['dollar_volume']) >= 3:
                recent_dv = np.mean(list(crypto['dollar_volume'])[-3:])
                dv_threshold = 5000 if self.market_regime == "sideways" else self.min_dollar_volume_usd
                if recent_dv < dv_threshold:
                    reject_dollar_volume += 1
                    continue

            # Relative strength vs BTC: altcoin must outperform BTC momentum
            if self.btc_symbol is not None and len(crypto.get('rs_vs_btc', [])) >= 1:
                rs_latest = crypto['rs_vs_btc'][-1]
                if rs_latest < 0:
                    continue

            # Position sizing: 70% base, Kelly-adjusted
            vol = self._annualized_vol(crypto)
            size = self._calculate_position_size(net_score, threshold_now, vol)

            # Halve size if in consecutive-loss recovery mode
            if self._consecutive_loss_halve_remaining > 0:
                size *= 0.50

            # High-volatility regime: widen sizing slightly (vol = opportunity)
            if self.volatility_regime == "high":
                size = min(size * 1.1, self.position_size_pct)

            slippage_penalty = get_slippage_penalty(self, sym)
            size *= slippage_penalty

            val = reserved_cash * size

            # Volume-Pegged Position Sizing ("Whale Rule"): cap at 2% of recent 3-min dollar volume
            if len(crypto['dollar_volume']) >= 3:
                whale_cap = sum(list(crypto['dollar_volume'])[-3:]) * 0.02
                val = min(val, whale_cap)

            # Absolute Hard Cap on position size
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

            # Exit feasibility check: qty must be >= MinimumOrderSize so the position can be sold.
            try:
                sec = self.Securities[sym]
                min_order_size = float(sec.SymbolProperties.MinimumOrderSize or 0)
                lot_size = float(sec.SymbolProperties.LotSize or 0)
                actual_min = max(min_order_size, lot_size)
                if actual_min > 0 and qty < actual_min:
                    self.Debug(f"REJECT ENTRY {sym.Value}: qty={qty} < min_order_size={actual_min} (unsellable)")
                    reject_notional += 1
                    continue
                # Ensure post-fee qty >= MinimumOrderSize so position is sellable.
                if min_order_size > 0:
                    post_fee_qty = qty * (1.0 - KRAKEN_SELL_FEE_BUFFER)
                    if post_fee_qty < min_order_size:
                        required_qty = round_quantity(self, sym, min_order_size / (1.0 - KRAKEN_SELL_FEE_BUFFER))
                        if required_qty * price <= available_cash * 0.99:  # 1% cash safety margin
                            qty = required_qty
                            val = qty * price
                        else:
                            self.Debug(f"REJECT ENTRY {sym.Value}: post-fee qty={post_fee_qty:.6f} < min_order_size={min_order_size} and can't upsize")
                            reject_notional += 1
                            continue
            except Exception as e:
                self.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

            try:
                if self.LiveMode:
                    ticket = place_limit_or_market(self, sym, qty, timeout_seconds=30, tag="Entry")
                else:
                    ticket = self.MarketOrder(sym, qty, tag="Entry")
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    components = cand.get('factors', {})
                    sig_str = (f"obi={components.get('obi', 0):.2f} "
                               f"vol={components.get('vol_ignition', 0):.2f} "
                               f"trend={components.get('micro_trend', 0):.2f} "
                               f"adx={components.get('adx_filter', 0):.2f} "
                               f"vwap={components.get('vwap_signal', 0):.2f}")
                    self.Debug(f"SCALP ENTRY: {sym.Value} | score={net_score:.2f} | ${val:.2f} | {sig_str}")
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
            debug_limited(self, f"EXECUTE: {success_count}/{len(candidates)} | rejects: cd={reject_exit_cooldown} loss={reject_loss_cooldown} dv={reject_dollar_volume}")

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
            self.Debug(f"DUST (rounded sell > actual): {symbol.Value} | actual={actual_qty} rounded={rounded_sell} — cleaning up")
            cleanup_position(self, symbol)
            self._failed_exit_counts.pop(symbol, None)
            return
        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = holding.AveragePrice
            self.highest_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time
        entry = self.entry_prices[symbol]
        highest = self.highest_prices.get(symbol, entry)
        if price > highest:
            self.highest_prices[symbol] = price
        pnl = (price - entry) / entry if entry > 0 else 0

        crypto = self.crypto_data.get(symbol)
        dd = (highest - price) / highest if highest > 0 else 0
        hours = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        minutes = hours * 60


        atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None
        if atr and entry > 0:
            sl = max((atr * self.atr_sl_mult) / entry, self.tight_stop_loss)
            tp = max((atr * self.atr_tp_mult) / entry, self.quick_take_profit)
        else:
            sl = self.tight_stop_loss   # 1.0-2.0% stop floor
            tp = self.quick_take_profit  # 1.5-3.0% take-profit floor

        if self.volatility_regime == "high":
            sl *= 1.2
            tp *= 1.2
        elif self.market_regime == "bear":
            tp = min(tp, 0.015)
        elif self.market_regime == "sideways":
            tp = min(tp, 0.030)
            sl = min(sl, 0.010)
        if self.market_regime == "bull":
            tp = max(tp * 3.0, 0.25)
            sl = max(sl * 2.0, 0.05)
        if tp < sl * 1.5:
            tp = sl * 1.5

        trailing_activation = self.trail_activation
        trailing_stop_pct   = self.trail_stop_pct


        if crypto and crypto['rsi'].IsReady:
            rsi_now = crypto['rsi'].Current.Value
            if rsi_now > 50:
                self.rsi_peaked_above_50[symbol] = True


        if (not self._partial_tp_taken.get(symbol, False)
                and pnl >= self.partial_tp_threshold):
            if partial_smart_sell(self, symbol, 0.50, "Partial TP"):
                self._partial_tp_taken[symbol] = True
                if self.market_regime == "bull":
                    self._breakeven_stops[symbol] = entry * 0.95
                    self.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL→entry-5% (bull)")
                else:
                    self._breakeven_stops[symbol] = entry * 1.002
                    self.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | SL→entry+0.2%")
                return  # Don't trigger full exit this bar

        tag = ""

        if self._partial_tp_taken.get(symbol, False):
            be_price = self._breakeven_stops.get(symbol, entry)
            if price <= be_price:
                tag = "Breakeven Stop"
        elif pnl <= -sl:
            tag = "Stop Loss"


        if not tag and self.market_regime != "bull" and minutes > self.stagnation_minutes and pnl < self.stagnation_pnl_threshold:
            tag = "Stagnation Exit"


        elif not tag:

            if not self._partial_tp_taken.get(symbol, False) and pnl >= tp:
                tag = "Take Profit"


            elif pnl > trailing_activation and dd >= trailing_stop_pct:
                tag = "Trailing Stop"


            elif atr and entry > 0 and holding.Quantity != 0:
                trail_offset = atr * self.atr_trail_mult
                trail_level = highest - trail_offset  # anchor to highest price since entry
                if crypto:
                    crypto['trail_stop'] = trail_level
                if crypto and crypto['trail_stop'] is not None:
                    if holding.Quantity > 0 and price <= crypto['trail_stop']:
                        tag = "ATR Trail"
                    elif holding.Quantity < 0 and price >= crypto['trail_stop']:
                        tag = "ATR Trail"


            if not tag and crypto and crypto['rsi'].IsReady:
                rsi_now = crypto['rsi'].Current.Value
                if self.rsi_peaked_above_50.get(symbol, False) and rsi_now < 50:
                    if self.market_regime == "bull" and pnl < 0.005:
                        pass  # Do not exit at a loss during bull regime
                    else:
                        tag = "RSI Momentum Exit"


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

                self.rsi_peaked_above_50.pop(symbol, None)
                self.entry_volumes.pop(symbol, None)
                self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
            else:

                self.Debug(f"⚠️ EXIT FAILED ({tag}): {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h -- position unsellable, cleaning up")
                cleanup_position(self, symbol)
                self.rsi_peaked_above_50.pop(symbol, None)
                self.entry_volumes.pop(symbol, None)

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
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)  # Remove from verification tracking
                self._order_retries.pop(event.OrderId, None)  # Clean up retry tracking
                if event.Direction == OrderDirection.Buy:
                    self.entry_prices[symbol] = event.FillPrice
                    self.highest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1

                    crypto = self.crypto_data.get(symbol)
                    if crypto and len(crypto['volume']) >= 1:
                        self.entry_volumes[symbol] = crypto['volume'][-1]
                    self.rsi_peaked_above_50.pop(symbol, None)
                else:

                    if symbol in self._partial_sell_symbols:
                        self._partial_sell_symbols.discard(symbol)
                    else:
                        entry = self.entry_prices.get(symbol, None)
                        if entry is None:
                            entry = event.FillPrice
                            self.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
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
                            'exit_reason': 'filled_sell',
                        })

                        if len(self._recent_trade_outcomes) >= 8:
                            recent_wr = sum(self._recent_trade_outcomes) / len(self._recent_trade_outcomes)
                            if recent_wr < 0.25:
                                self._cash_mode_until = self.Time + timedelta(hours=24)
                                self.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(self._recent_trade_outcomes)} trades. Pausing 24h.")
                        cleanup_position(self, symbol)
                        self._failed_exit_attempts.pop(symbol, None)
                        self._failed_exit_counts.pop(symbol, None)
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if event.Direction == OrderDirection.Sell and symbol not in self.entry_prices:
                    if is_invested_not_dust(self, symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
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
                        self.Debug(f"DUST CLEANUP on invalid sell: {symbol.Value} — releasing tracking")
                        cleanup_position(self, symbol)
                        self._failed_exit_counts.pop(symbol, None)
                    else:

                        fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                        self._failed_exit_counts[symbol] = fail_count
                        self.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                        if fail_count >= 3:

                            self.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} failed exits — releasing tracking")
                            cleanup_position(self, symbol)
                            self._failed_exit_counts.pop(symbol, None)
                        elif symbol not in self.entry_prices:
                            if is_invested_not_dust(self, symbol):
                                holding = self.Portfolio[symbol]
                                self.entry_prices[symbol] = holding.AveragePrice
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
                    self.kraken_status = "online"
                elif "maintenance" in txt:
                    self.kraken_status = "maintenance"
                elif "cancel_only" in txt:
                    self.kraken_status = "cancel_only"
                elif "post_only" in txt:
                    self.kraken_status = "post_only"
                else:
                    self.kraken_status = "unknown"
                self.Debug(f"Kraken status update: {self.kraken_status}")
            
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
