# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
# endregion


class MNQFuturesStrategy(QCAlgorithm):
    """
    Machine Gun v2 MNA — CME Micro Nasdaq 100 E-mini Futures (MNQ)

    Core strategy adapted from Machine Gun v2 (crypto scalper) for
    Tradovate / CME futures:
      - Single instrument: MNQ continuous front-month future
      - Both Long and Short allowed (no borrow restrictions on futures)
      - Integer contract sizing (1 or 2 contracts)
      - Regular Trading Hours filter: entries only 09:30–15:45 ET
      - Flat-at-Close: all positions liquidated at 15:55 ET
      - Tradovate brokerage + Margin account (default fee/slippage models)
      - Alpha Upgrades retained: Graduated Gate, Signal 4 Split, Spread Haircut
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(10000)

        # Eastern Time for session filtering
        self.SetTimeZone(TimeZones.NewYork)

        # Tradovate + Margin (no custom crypto fee/slippage — use defaults)
        self.SetBrokerageModel(BrokerageName.Tradovate, AccountType.Margin)

        # ------------------------------------------------------------------
        # MNQ Continuous Future (front month)
        # ------------------------------------------------------------------
        self.mnq = self.AddFuture(Futures.Indices.MicroNASDAQ100EMini, Resolution.Minute)
        self.mnq.SetFilter(0, 120)   # front month only
        self.symbol = self.mnq.Symbol

        # ------------------------------------------------------------------
        # Strategy parameters
        # ------------------------------------------------------------------
        self.entry_threshold          = 0.50
        self.high_conviction_threshold = 0.60
        self.max_contracts            = self._get_param("max_contracts", 2)   # safety cap
        self.max_positions            = 1   # one position at a time (long OR short)

        # Exit parameters calibrated for futures intraday moves
        self.quick_take_profit    = self._get_param("quick_take_profit",    0.015)  # 1.5%
        self.tight_stop_loss      = self._get_param("tight_stop_loss",      0.005)  # 0.5%
        self.atr_tp_mult          = self._get_param("atr_tp_mult",          4.0)
        self.atr_sl_mult          = self._get_param("atr_sl_mult",          2.0)
        self.trail_activation     = self._get_param("trail_activation",     0.008)  # 0.8%
        self.trail_stop_pct       = self._get_param("trail_stop_pct",       0.004)  # 0.4%
        self.time_stop_hours      = self._get_param("time_stop_hours",      3.0)
        self.time_stop_pnl_min    = self._get_param("time_stop_pnl_min",    0.001)
        self.extended_time_stop_hours   = self._get_param("extended_time_stop_hours",   4.0)
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.005)
        self.stale_position_hours = self._get_param("stale_position_hours", 5.0)

        self.trailing_activation   = self.trail_activation
        self.trailing_stop_pct    = self.trail_stop_pct
        self.base_stop_loss       = self.tight_stop_loss
        self.base_take_profit     = self.quick_take_profit
        self.atr_trail_mult       = 2.0

        self.partial_tp_threshold    = 0.010   # 1.0% — take half off at this profit
        self.stagnation_minutes      = 60
        self.stagnation_pnl_threshold = 0.002

        # ------------------------------------------------------------------
        # Indicator periods
        # ------------------------------------------------------------------
        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(60 * 24 * 252)   # annualization for 1-min bars

        # ------------------------------------------------------------------
        # Risk management
        # ------------------------------------------------------------------
        self.max_drawdown_limit             = 0.15   # 15% drawdown → cooldown
        self.cooldown_hours                 = 6
        self.consecutive_losses             = 0
        self.max_consecutive_losses         = 5
        self._consecutive_loss_halve_remaining = 0
        self.circuit_breaker_expiry         = None

        self.target_position_ann_vol = 0.35
        self.portfolio_vol_cap       = 0.80
        self.min_asset_vol_floor     = 0.05

        self.exit_cooldown_hours         = 0.5    # 30-min post-exit cool-down
        self.cancel_cooldown_minutes     = 1
        self.adx_min_period              = 10
        self.spread_median_window        = 12

        # ------------------------------------------------------------------
        # Order management
        # ------------------------------------------------------------------
        self.stale_order_timeout_seconds      = 60
        self.live_stale_order_timeout_seconds = 90
        self.max_concurrent_open_orders       = 2
        self.order_fill_check_threshold_seconds = 60
        self.order_timeout_seconds              = 30

        # ------------------------------------------------------------------
        # Trade-count & daily counters
        # ------------------------------------------------------------------
        self.max_daily_trades            = 500
        self.daily_trade_count           = 0
        self.last_trade_date             = None
        self.max_symbol_trades_per_day   = 20

        # ------------------------------------------------------------------
        # Performance tracking
        # ------------------------------------------------------------------
        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.trade_count    = 0
        self.trade_log      = []
        self.pnl_by_tag     = {}
        self.log_budget     = 0

        self._rolling_wins       = deque(maxlen=50)
        self._rolling_win_sizes  = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._recent_trade_outcomes = deque(maxlen=20)
        self._last_live_trade_time  = None

        # ------------------------------------------------------------------
        # Position tracking
        # ------------------------------------------------------------------
        self.future_data       = {}   # indicator / signal data (keyed by symbol)
        self.entry_prices      = {}
        self.highest_prices    = {}   # for long trailing stops
        self.lowest_prices     = {}   # for short trailing stops
        self.entry_times       = {}
        self.position_direction = {}  # 1 = long, -1 = short
        self._partial_tp_taken  = {}
        self._breakeven_stops   = {}
        self._partial_sell_symbols = set()
        self._choppy_regime_entries = {}

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self._positions_synced    = False
        self._first_post_warmup   = True
        self._submitted_orders    = {}
        self._pending_orders      = {}
        self._cancel_cooldowns    = {}
        self._exit_cooldowns      = {}
        self._order_retries       = {}
        self._retry_pending       = {}
        self._failed_exit_counts  = {}
        self._failed_exit_attempts = {}
        self._cash_mode_until     = None
        self._last_skip_reason    = None
        self._slip_abs            = deque(maxlen=50)
        self.slip_outlier_threshold = 0.0015
        self._symbol_loss_cooldowns = {}
        self._recent_tickets        = deque(maxlen=25)
        self._daily_open_value      = None

        # Portfolio peak / drawdown tracking
        self.peak_value        = None
        self.drawdown_cooldown = 0

        # Market regime (derived from MNQ itself)
        self.market_regime    = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth   = 0.5
        self._regime_hold_count = 0

        # ------------------------------------------------------------------
        # Session filter boundaries (Eastern Time — algorithm is in ET)
        # ------------------------------------------------------------------
        self._session_start_min = 9 * 60 + 30    # 09:30 ET in minutes
        self._session_end_min   = 15 * 60 + 45   # 15:45 ET in minutes

        # ------------------------------------------------------------------
        # Scheduled tasks
        # ------------------------------------------------------------------
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0), self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1), self.DailyReport)
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=6)),
            self.ReviewPerformance,
        )
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0), self.HealthCheck)
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(minutes=2)),
            self.VerifyOrderFills,
        )
        # Flat-at-Close: liquidate all positions at 15:55 ET
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(15, 55), self.FlatAtClose)

        # ------------------------------------------------------------------
        # Warm-up & initialisation
        # ------------------------------------------------------------------
        self.SetWarmUp(timedelta(days=4))
        self.Settings.FreePortfolioValuePercentage = 0.01

        self._scoring_engine = MicroScalpEngine(self)

        if self.LiveMode:
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING: MNQ Futures (MNA) v1.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max contracts: {self.max_contracts}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
            return default
        except Exception:
            return default

    def _in_session(self):
        """Returns True if within Regular Trading Hours: 09:30–15:45 ET."""
        current_min = self.Time.hour * 60 + self.Time.minute
        return self._session_start_min <= current_min <= self._session_end_min

    def _normalize_order_time(self, order_time):
        return normalize_order_time(order_time)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    # ------------------------------------------------------------------
    # Scheduled tasks
    # ------------------------------------------------------------------

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date   = self.Time.date()
        self._daily_open_value = self.Portfolio.TotalPortfolioValue
        future = self.future_data.get(self.symbol)
        if future is not None:
            future['trade_count_today'] = 0
        persist_state(self)

    def HealthCheck(self):
        if self.IsWarmingUp:
            return
        health_check(self)

    def VerifyOrderFills(self):
        if self.IsWarmingUp:
            return
        verify_order_fills(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10:
            return
        recent = self.trade_log[-15:] if len(self.trade_log) >= 15 else self.trade_log
        recent_wr  = sum(1 for t in recent if t['pnl_pct'] > 0) / len(recent)
        recent_pnl = np.mean([t['pnl_pct'] for t in recent])
        if recent_wr < 0.20 and recent_pnl < -0.02:
            self.max_contracts = 1
            self.Debug(f"PERFORMANCE DECAY: max_contracts=1 (WR:{recent_wr:.0%}, PnL:{recent_pnl:+.2%})")
        elif recent_wr > 0.35 and recent_pnl > 0:
            self.max_contracts = int(self._get_param("max_contracts", 2))

    def FlatAtClose(self):
        """
        Flat-at-Close: market-liquidate all positions and cancel all pending
        orders at 15:55 ET to avoid the 16:00–17:00 ET settlement window.
        """
        if self.IsWarmingUp:
            return
        try:
            self.Transactions.CancelOpenOrders()
        except Exception as e:
            self.Debug(f"FlatAtClose cancel error: {e}")

        if is_invested(self, self.symbol):
            qty = self.Portfolio[self.symbol].Quantity
            try:
                self.MarketOrder(self.symbol, -qty, tag="Flat at Close")
                cleanup_position(self, self.symbol)
                self.Debug(f"FLAT AT CLOSE: {self.symbol.Value} qty={qty} @ {self.Time}")
            except Exception as e:
                self.Debug(f"FlatAtClose order error: {e}")

    def DailyReport(self):
        if self.IsWarmingUp:
            return
        daily_report(self)

    # ------------------------------------------------------------------
    # Initialise indicator data for a symbol
    # ------------------------------------------------------------------

    def _initialize_symbol(self, symbol):
        self.future_data[symbol] = {
            'prices':      deque(maxlen=self.lookback),
            'returns':     deque(maxlen=self.lookback),
            'volume':      deque(maxlen=self.lookback),
            'volume_ma':   deque(maxlen=self.medium_period),
            'dollar_volume': deque(maxlen=self.lookback),
            'ema_ultra_short': ExponentialMovingAverage(self.ultra_short_period),
            'ema_short':       ExponentialMovingAverage(self.short_period),
            'ema_medium':      ExponentialMovingAverage(self.medium_period),
            'ema_5':           ExponentialMovingAverage(5),
            'atr':  AverageTrueRange(14),
            'adx':  AverageDirectionalIndex(self.adx_min_period),
            'rsi':  RelativeStrengthIndex(7),
            'volatility':  deque(maxlen=self.medium_period),
            'zscore':      deque(maxlen=self.short_period),
            'last_price':  0,
            'spreads':     deque(maxlen=self.spread_median_window),
            'highs':  deque(maxlen=self.lookback),
            'lows':   deque(maxlen=self.lookback),
            'bb_upper': deque(maxlen=self.short_period),
            'bb_lower': deque(maxlen=self.short_period),
            'bb_width': deque(maxlen=self.medium_period),
            'trade_count_today': 0,
            'bid_size': 0.0,
            'ask_size': 0.0,
            'vwap_pv':       deque(maxlen=20),
            'vwap_v':        deque(maxlen=20),
            'vwap':          0.0,
            'vwap_sd':       0.0,
            'vwap_sd2_lower': 0.0,
            'vwap_sd3_lower': 0.0,
            'vwap_sd2_upper': 0.0,
            'vwap_sd3_upper': 0.0,
            'volume_long': deque(maxlen=1440),
            'cvd':  deque(maxlen=self.lookback),
            'ker':  deque(maxlen=self.short_period),
            'kalman_estimate': 0.0,
            'kalman_error_cov': 1.0,
            'trail_stop': None,
            'recent_net_scores': deque(maxlen=3),
        }

    # ------------------------------------------------------------------
    # Securities changes
    # ------------------------------------------------------------------

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            sym = security.Symbol
            # Track data for the continuous contract and any mapped contracts
            if sym not in self.future_data:
                self._initialize_symbol(sym)
        for security in changes.RemovedSecurities:
            sym = security.Symbol
            # If we're holding a position in the removed contract, liquidate
            if not self.IsWarmingUp and is_invested(self, sym):
                smart_liquidate(self, sym, "Contract Roll — Removed")
                self.Debug(f"CONTRACT ROLL: {sym.Value} removed, liquidating")

    # ------------------------------------------------------------------
    # OnData
    # ------------------------------------------------------------------

    def OnData(self, data):
        # Update indicator data for the continuous symbol
        if data.Bars.ContainsKey(self.symbol):
            if self.symbol not in self.future_data:
                self._initialize_symbol(self.symbol)
            try:
                quote_bar = (data.QuoteBars[self.symbol]
                             if data.QuoteBars.ContainsKey(self.symbol) else None)
                self._update_symbol_data(self.symbol, data.Bars[self.symbol], quote_bar)
            except Exception as e:
                self.Debug(f"Error updating symbol data: {e}")

        if self.IsWarmingUp:
            return

        if not self._positions_synced:
            if not self._first_post_warmup:
                self.Transactions.CancelOpenOrders()
            sync_existing_positions(self)
            self._positions_synced = True
            self._first_post_warmup = False

        self._update_market_context()
        self.Rebalance()
        self.CheckExits()

    # ------------------------------------------------------------------
    # Indicator update
    # ------------------------------------------------------------------

    def _update_symbol_data(self, symbol, bar, quote_bar=None):
        future = self.future_data[symbol]
        price  = float(bar.Close)
        high   = float(bar.High)
        low    = float(bar.Low)
        volume = float(bar.Volume)

        future['prices'].append(price)
        future['highs'].append(high)
        future['lows'].append(low)

        if future['last_price'] > 0:
            ret = (price - future['last_price']) / future['last_price']
            future['returns'].append(ret)
        future['last_price'] = price

        future['volume'].append(volume)
        future['dollar_volume'].append(price * volume)
        future['volume_long'].append(volume)

        if len(future['volume']) >= self.short_period:
            future['volume_ma'].append(np.mean(list(future['volume'])[-self.short_period:]))

        future['ema_ultra_short'].Update(bar.EndTime, price)
        future['ema_short'].Update(bar.EndTime, price)
        future['ema_medium'].Update(bar.EndTime, price)
        future['ema_5'].Update(bar.EndTime, price)
        future['atr'].Update(bar)
        future['adx'].Update(bar)
        future['rsi'].Update(bar.EndTime, price)

        # VWAP + standard deviation bands
        future['vwap_pv'].append(price * volume)
        future['vwap_v'].append(volume)
        total_v = sum(future['vwap_v'])
        if total_v > 0:
            future['vwap'] = sum(future['vwap_pv']) / total_v

        if len(future['vwap_v']) >= 5 and future['vwap'] > 0:
            vwap_val = future['vwap']
            pv_list  = list(future['vwap_pv'])
            v_list   = list(future['vwap_v'])
            bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
            if len(bar_prices) >= 5:
                sd = float(np.std(bar_prices))
                future['vwap_sd']       = sd
                future['vwap_sd2_lower'] = vwap_val - 2.0 * sd
                future['vwap_sd3_lower'] = vwap_val - 3.0 * sd
                future['vwap_sd2_upper'] = vwap_val + 2.0 * sd
                future['vwap_sd3_upper'] = vwap_val + 3.0 * sd

        # Bollinger Bands
        if len(future['prices']) >= self.medium_period:
            prices_arr = np.array(list(future['prices'])[-self.medium_period:])
            std  = np.std(prices_arr)
            mean = np.mean(prices_arr)
            if std > 0:
                future['zscore'].append((price - mean) / std)
                future['bb_upper'].append(mean + 2 * std)
                future['bb_lower'].append(mean - 2 * std)
                future['bb_width'].append(4 * std / mean if mean > 0 else 0)

        # Volatility
        if len(future['returns']) >= 10:
            future['volatility'].append(np.std(list(future['returns'])[-10:]))

        # CVD (Cumulative Volume Delta)
        high_low = high - low
        if high_low > 0:
            bar_delta = volume * ((price - low) - (high - price)) / high_low
        else:
            bar_delta = 0.0
        prev_cvd = future['cvd'][-1] if len(future['cvd']) > 0 else 0.0
        future['cvd'].append(prev_cvd + bar_delta)

        # Kaufman Efficiency Ratio (KER)
        if len(future['prices']) >= 15:
            price_change   = abs(future['prices'][-1] - future['prices'][-15])
            volatility_sum = sum(abs(future['prices'][i] - future['prices'][i - 1])
                                 for i in range(-14, 0))
            if volatility_sum > 0:
                future['ker'].append(price_change / volatility_sum)
            else:
                future['ker'].append(0.0)

        # Kalman Filter
        Q = 1e-5
        R = 0.01
        if future['kalman_estimate'] == 0.0:
            future['kalman_estimate'] = price
        est_pred   = future['kalman_estimate']
        err_pred   = future['kalman_error_cov'] + Q
        kg         = err_pred / (err_pred + R)
        future['kalman_estimate']   = est_pred + kg * (price - est_pred)
        future['kalman_error_cov']  = (1 - kg) * err_pred

        # Order Book Imbalance from quote bar
        if quote_bar is not None:
            try:
                bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
                ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
                if bid_sz > 0 or ask_sz > 0:
                    future['bid_size'] = bid_sz
                    future['ask_size'] = ask_sz
            except Exception:
                pass

        # Spread tracking
        sp = get_spread_pct(self, symbol)
        if sp is not None:
            future['spreads'].append(sp)

    # ------------------------------------------------------------------
    # Market context (regime derived from MNQ itself)
    # ------------------------------------------------------------------

    def _update_market_context(self):
        future = self.future_data.get(self.symbol)
        if future is None or len(future['prices']) < 48:
            return

        arr      = np.array(list(future['prices']))
        current  = arr[-1]
        sma      = np.mean(arr[-48:])
        returns  = list(future['returns'])
        mom_12   = np.mean(returns[-12:]) if len(returns) >= 12 else 0.0

        if current > sma * 1.005:
            new_regime = "bull"
        elif current < sma * 0.995:
            new_regime = "bear"
        else:
            new_regime = "sideways"

        if new_regime == "sideways":
            if mom_12 > 0.0001:
                new_regime = "bull"
            elif mom_12 < -0.0001:
                new_regime = "bear"

        if new_regime != self.market_regime:
            self._regime_hold_count += 1
            if self._regime_hold_count >= 3:
                self.market_regime = new_regime
                self._regime_hold_count = 0
        else:
            self._regime_hold_count = 0

        vols = list(future.get('volatility', []))
        if len(vols) >= 5:
            current_vol = vols[-1]
            avg_vol     = np.mean(vols)
            if current_vol > avg_vol * 1.5:
                self.volatility_regime = "high"
            elif current_vol < avg_vol * 0.5:
                self.volatility_regime = "low"
            else:
                self.volatility_regime = "normal"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _annualized_vol(self, future):
        if future is None:
            return None
        if len(future.get('volatility', [])) == 0:
            return None
        return float(future['volatility'][-1]) * self.sqrt_annualization

    def _compute_portfolio_risk_estimate(self):
        total_value = self.Portfolio.TotalPortfolioValue
        if total_value <= 0:
            return 0.0
        risk = 0.0
        for kvp in self.Portfolio:
            if not is_invested(self, kvp.Key):
                continue
            future = self.future_data.get(kvp.Key)
            asset_vol_ann = self._annualized_vol(future)
            if asset_vol_ann is None:
                asset_vol_ann = self.min_asset_vol_floor
            weight = abs(kvp.Value.HoldingsValue) / total_value
            risk  += weight * asset_vol_ann
        return risk

    def _calculate_factor_scores(self, symbol, future):
        """
        Evaluate both long and short signals. Returns the components dict
        for the higher-scoring direction (if it meets the entry threshold),
        or the long components with a low score if neither qualifies.
        The '_direction' key indicates: 1 = long, -1 = short.
        The '_scalp_score' key holds the winning score.
        """
        long_score,  long_components  = self._scoring_engine.calculate_long_score(future)
        short_score, short_components = self._scoring_engine.calculate_short_score(future)

        # Spread haircut (MNQ spreads are tight but still apply the penalty)
        sp = get_spread_pct(self, symbol)
        if sp is not None and sp > 0:
            spread_penalty = min((sp / 0.005) * 0.15, 0.15)
            long_score  *= (1.0 - spread_penalty)
            short_score *= (1.0 - spread_penalty)

        # Pick the stronger direction
        if short_score > long_score:
            components = short_components.copy()
            components['_scalp_score'] = short_score
            components['_direction']   = -1
            components['_long_score']  = long_score
            components['_short_score'] = short_score
        else:
            components = long_components.copy()
            components['_scalp_score'] = long_score
            components['_direction']   = 1
            components['_long_score']  = long_score
            components['_short_score'] = short_score

        return components

    def _calculate_composite_score(self, factors, future=None):
        return factors.get('_scalp_score', 0.0)

    def _is_ready(self, f):
        return len(f['prices']) >= 10 and f['rsi'].IsReady

    def _daily_loss_exceeded(self):
        """Returns True if today's portfolio has dropped >= 3% from open."""
        if not hasattr(self, '_daily_open_value') or self._daily_open_value is None or self._daily_open_value <= 0:
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

    # ------------------------------------------------------------------
    # Rebalance
    # ------------------------------------------------------------------

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        # --- Session filter: only new entries during RTH ---
        if not self._in_session():
            self._log_skip("outside RTH (09:30–15:45 ET)")
            return

        if self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded (3%)")
            return

        if self._cash_mode_until is not None and self.Time < self._cash_mode_until:
            self._log_skip("cash mode — poor recent performance")
            return

        if self.LiveMode and not live_safety_checks(self):
            return

        cancel_stale_new_orders(self)

        if self.daily_trade_count >= self.max_daily_trades:
            self._log_skip("max daily trades")
            return

        self.log_budget = 10

        val = self.Portfolio.TotalPortfolioValue
        if self.peak_value is None or self.peak_value < 1:
            self.peak_value = val
        if self.drawdown_cooldown > 0:
            self.drawdown_cooldown -= 1
            if self.drawdown_cooldown <= 0:
                self.peak_value = val
                self.consecutive_losses = 0
            else:
                self._log_skip(f"drawdown cooldown {self.drawdown_cooldown}")
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
            self._log_skip("consecutive loss cooldown")
            return

        if self.circuit_breaker_expiry is not None and self.Time < self.circuit_breaker_expiry:
            self._log_skip("circuit breaker active")
            return

        if self.consecutive_losses >= 4:
            self.circuit_breaker_expiry = self.Time + timedelta(hours=1)
            self.consecutive_losses = 0
            self._log_skip("circuit breaker triggered (4 losses)")
            return

        # If already in a position, don't open another
        if is_invested(self, self.symbol):
            self._log_skip("already in position")
            return

        if has_open_orders(self, self.symbol):
            self._log_skip("open orders pending")
            return

        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
            return

        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            self._log_skip("portfolio risk cap exceeded")
            return

        future = self.future_data.get(self.symbol)
        if future is None or not self._is_ready(future):
            self._log_skip("indicator warmup incomplete")
            return

        if future.get('trade_count_today', 0) >= self.max_symbol_trades_per_day:
            self._log_skip("max daily trades for symbol")
            return

        # Score the opportunity
        factor_scores = self._calculate_factor_scores(self.symbol, future)
        net_score     = self._calculate_composite_score(factor_scores, future)

        future['recent_net_scores'].append(net_score)

        threshold = self.entry_threshold
        if net_score < threshold:
            self._log_skip(f"score {net_score:.2f} below threshold {threshold:.2f}")
            return

        direction = factor_scores.get('_direction', 1)

        debug_limited(self, (
            f"REBALANCE: score={net_score:.2f} dir={'LONG' if direction > 0 else 'SHORT'} | "
            f"regime={self.market_regime} vol={self.volatility_regime}"
        ))

        self._last_skip_reason = None
        self._execute_trade(factor_scores, threshold)

    # ------------------------------------------------------------------
    # Execute trade
    # ------------------------------------------------------------------

    def _execute_trade(self, factor_scores, threshold_now):
        if not self._positions_synced:
            return

        cancel_stale_new_orders(self)

        net_score = factor_scores.get('_scalp_score', 0.0)
        direction = factor_scores.get('_direction', 1)  # 1 = long, -1 = short

        # Integer contract sizing
        contracts = self._scoring_engine.calculate_position_size(
            net_score, threshold_now, self.max_contracts
        )

        if self._consecutive_loss_halve_remaining > 0:
            contracts = 1

        order_qty = contracts * direction  # positive = long, negative = short

        try:
            ticket = place_limit_or_market(
                self, self.symbol, order_qty, timeout_seconds=30, tag="Entry"
            )
            if ticket is not None:
                self._recent_tickets.append(ticket)

                components = factor_scores
                side = "LONG" if direction > 0 else "SHORT"
                self.Debug(
                    f"SCALP ENTRY [{side}]: {self.symbol.Value} | "
                    f"score={net_score:.2f} | contracts={contracts} | "
                    f"obi={components.get('obi', 0):.2f} "
                    f"vol={components.get('vol_ignition', 0):.2f} "
                    f"trend={components.get('micro_trend', 0):.2f} "
                    f"adx={components.get('adx_trend', 0):.2f} "
                    f"vwap={components.get('vwap_signal', 0):.2f}"
                )
                self.trade_count += 1
                future = self.future_data.get(self.symbol)
                if future is not None:
                    future['trade_count_today'] = future.get('trade_count_today', 0) + 1
                    adx_ind  = future.get('adx')
                    is_choppy = (adx_ind is not None and adx_ind.IsReady
                                 and adx_ind.Current.Value < 25)
                    self._choppy_regime_entries[self.symbol] = is_choppy

                if self._consecutive_loss_halve_remaining > 0:
                    self._consecutive_loss_halve_remaining -= 1
                if self.LiveMode:
                    self._last_live_trade_time = self.Time

        except Exception as e:
            self.Debug(f"ORDER FAILED: {self.symbol.Value} - {e}")

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def CheckExits(self):
        if self.IsWarmingUp:
            return
        for kvp in self.Portfolio:
            symbol = kvp.Key
            if not is_invested(self, symbol):
                self._failed_exit_counts.pop(symbol, None)
                continue
            if self._failed_exit_counts.get(symbol, 0) >= 3:
                continue
            self._check_exit(symbol, self.Securities[symbol].Price, kvp.Value)

        # Orphan recovery
        for kvp in self.Portfolio:
            symbol = kvp.Key
            if not is_invested(self, symbol):
                continue
            if symbol not in self.entry_prices:
                self.entry_prices[symbol]   = kvp.Value.AveragePrice
                direction = 1 if kvp.Value.Quantity > 0 else -1
                self.position_direction[symbol] = direction
                if direction == 1:
                    self.highest_prices[symbol] = kvp.Value.AveragePrice
                else:
                    self.lowest_prices[symbol]  = kvp.Value.AveragePrice
                self.entry_times[symbol] = self.Time
                self.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")

    def _check_exit(self, symbol, price, holding):
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            return

        is_short = holding.Quantity < 0
        entry     = self.entry_prices.get(symbol, holding.AveragePrice)

        # Track high/low water marks
        if is_short:
            lowest = self.lowest_prices.get(symbol, entry)
            if price < lowest:
                self.lowest_prices[symbol] = lowest = price
            pnl = (entry - price) / entry if entry > 0 else 0
            dd  = (price - lowest) / lowest if lowest > 0 else 0  # drawdown from best (short)
        else:
            highest = self.highest_prices.get(symbol, entry)
            if price > highest:
                self.highest_prices[symbol] = highest = price
            pnl = (price - entry) / entry if entry > 0 else 0
            dd  = (highest - price) / highest if highest > 0 else 0  # drawdown from peak

        future = self.future_data.get(symbol)
        hours   = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        minutes = hours * 60

        atr = future['atr'].Current.Value if future and future['atr'].IsReady else None
        if atr and entry > 0:
            sl = max((atr * self.atr_sl_mult) / entry, self.tight_stop_loss)
            tp = max((atr * self.atr_tp_mult) / entry, self.quick_take_profit)
        else:
            sl = self.tight_stop_loss
            tp = self.quick_take_profit

        if tp < sl * 1.5:
            tp = sl * 1.5

        if self._choppy_regime_entries.get(symbol, False):
            tp *= 0.65

        # Partial TP
        if (not self._partial_tp_taken.get(symbol, False)
                and pnl >= self.partial_tp_threshold):
            if partial_smart_sell(self, symbol, 0.50, "Partial TP"):
                self._partial_tp_taken[symbol] = True
                self._breakeven_stops[symbol]  = entry
                side = "SHORT" if is_short else "LONG"
                self.Debug(f"PARTIAL TP [{side}]: {symbol.Value} | PnL:{pnl:+.2%}")
                return

        tag = ""

        # Breakeven stop (after partial TP)
        if self._partial_tp_taken.get(symbol, False):
            be_price = self._breakeven_stops.get(symbol, entry)
            if is_short and price >= be_price:
                tag = "Breakeven Stop"
            elif not is_short and price <= be_price:
                tag = "Breakeven Stop"

        # Stop loss
        if not tag and pnl <= -sl:
            tag = "Stop Loss"

        # Stagnation exit
        if not tag and minutes > self.stagnation_minutes and pnl < self.stagnation_pnl_threshold:
            tag = "Stagnation Exit"

        if not tag:
            # Take profit
            if not self._partial_tp_taken.get(symbol, False) and pnl >= tp:
                tag = "Take Profit"

            # Percentage trailing stop
            elif pnl > self.trail_activation and dd >= self.trail_stop_pct:
                tag = "Trailing Stop"

            # ATR trailing stop
            elif atr and entry > 0 and holding.Quantity != 0:
                trail_offset = atr * self.atr_trail_mult
                if is_short:
                    # For short: trail is anchor_lowest + offset
                    lowest = self.lowest_prices.get(symbol, entry)
                    trail_level = lowest + trail_offset
                    if future:
                        future['trail_stop'] = trail_level
                    if future and future['trail_stop'] is not None and price >= future['trail_stop']:
                        tag = "ATR Trail"
                else:
                    # For long: trail is anchor_highest - offset
                    highest = self.highest_prices.get(symbol, entry)
                    trail_level = highest - trail_offset
                    if future:
                        future['trail_stop'] = trail_level
                    if future and future['trail_stop'] is not None and price <= future['trail_stop']:
                        tag = "ATR Trail"

            # Time stops
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
                self._choppy_regime_entries.pop(symbol, None)
                side = "SHORT" if is_short else "LONG"
                self.Debug(f"{tag} [{side}]: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
            else:
                fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                self._failed_exit_counts[symbol] = fail_count
                self.Debug(f"EXIT FAILED ({tag}) #{fail_count}: {symbol.Value} | PnL:{pnl:+.2%}")
                if fail_count >= 3:
                    self.Debug(f"FORCE EXIT: {symbol.Value} after {fail_count} failed attempts")
                    try:
                        qty = abs(holding.Quantity)
                        if qty > 0:
                            close_qty = qty if is_short else -qty
                            self.MarketOrder(symbol, close_qty, tag=f"Force Exit (fail#{fail_count})")
                    except Exception as e:
                        self.Debug(f"Force market exit error for {symbol.Value}: {e}")
                    self._failed_exit_counts.pop(symbol, None)
                    self._choppy_regime_entries.pop(symbol, None)

    # ------------------------------------------------------------------
    # Order events
    # ------------------------------------------------------------------

    def OnOrderEvent(self, event):
        try:
            symbol = event.Symbol
            self.Debug(
                f"ORDER: {symbol.Value} {event.Status} {event.Direction} "
                f"qty={event.FillQuantity or event.Quantity} price={event.FillPrice} "
                f"id={event.OrderId}"
            )

            if event.Status == OrderStatus.Submitted:
                if symbol not in self._pending_orders:
                    self._pending_orders[symbol] = 0
                intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
                self._pending_orders[symbol] += intended_qty
                # Infer intent if not already set from place_limit_or_market / smart_liquidate
                if symbol not in self._submitted_orders:
                    has_position = symbol in self.entry_prices
                    if event.Direction == OrderDirection.Sell and has_position and self.position_direction.get(symbol, 1) == 1:
                        inferred_intent = 'exit'
                    elif event.Direction == OrderDirection.Buy and has_position and self.position_direction.get(symbol, -1) == -1:
                        inferred_intent = 'exit'
                    else:
                        inferred_intent = 'entry'
                    self._submitted_orders[symbol] = {
                        'order_id': event.OrderId,
                        'time':     self.Time,
                        'quantity': event.Quantity,
                        'intent':   inferred_intent,
                    }

            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                order_info = self._submitted_orders.pop(symbol, {})
                intent     = order_info.get('intent', 'unknown')
                self._order_retries.pop(event.OrderId, None)

                # Use intent to determine entry vs exit
                is_entry = (
                    intent == 'entry'
                    or (intent == 'unknown' and symbol not in self.entry_prices)
                )

                if event.Direction == OrderDirection.Buy:
                    if is_entry:
                        # Long entry
                        self.entry_prices[symbol]       = event.FillPrice
                        self.highest_prices[symbol]     = event.FillPrice
                        self.lowest_prices.pop(symbol, None)
                        self.entry_times[symbol]        = self.Time
                        self.position_direction[symbol] = 1
                        self.daily_trade_count += 1
                        self.Debug(f"LONG ENTRY FILL: {symbol.Value} @ ${event.FillPrice:.2f}")
                    else:
                        # Closing a short position
                        if symbol in self._partial_sell_symbols:
                            self._partial_sell_symbols.discard(symbol)
                        else:
                            self._record_trade_exit(symbol, event.FillPrice)

                else:  # Sell
                    if is_entry:
                        # Short entry
                        self.entry_prices[symbol]       = event.FillPrice
                        self.lowest_prices[symbol]      = event.FillPrice
                        self.highest_prices.pop(symbol, None)
                        self.entry_times[symbol]        = self.Time
                        self.position_direction[symbol] = -1
                        self.daily_trade_count += 1
                        self.Debug(f"SHORT ENTRY FILL: {symbol.Value} @ ${event.FillPrice:.2f}")
                    else:
                        # Closing a long position
                        if symbol in self._partial_sell_symbols:
                            self._partial_sell_symbols.discard(symbol)
                        else:
                            self._record_trade_exit(symbol, event.FillPrice)

                slip_log(self, symbol, event.Direction, event.FillPrice)

            elif event.Status == OrderStatus.PartiallyFilled:
                if symbol in self._pending_orders:
                    self._pending_orders[symbol] -= abs(event.FillQuantity)
                    if self._pending_orders[symbol] <= 0:
                        self._pending_orders.pop(symbol, None)
                slip_log(self, symbol, event.Direction, event.FillPrice)

            elif event.Status in (OrderStatus.Canceled, OrderStatus.Invalid):
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)

        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")

        if self.LiveMode:
            persist_state(self)

    def _record_trade_exit(self, symbol, exit_price):
        """Record PnL and clean up tracking for a filled exit order."""
        entry     = self.entry_prices.get(symbol, exit_price)
        direction = self.position_direction.get(symbol, 1)

        # Identify the exit tag from the most recent order for this symbol
        exit_tag = "Unknown"
        try:
            orders = [o for o in self.Transactions.GetOrders(lambda o: o.Symbol == symbol)
                      if o.Status == OrderStatus.Filled]
            if orders:
                latest = max(orders, key=lambda o: o.Time)
                exit_tag = latest.Tag if latest.Tag else "Unknown"
        except Exception:
            pass

        if direction >= 0:
            pnl = (exit_price - entry) / entry if entry > 0 else 0
        else:
            pnl = (entry - exit_price) / entry if entry > 0 else 0

        self._rolling_wins.append(1 if pnl > 0 else 0)
        self._recent_trade_outcomes.append(1 if pnl > 0 else 0)

        if pnl > 0:
            self._rolling_win_sizes.append(pnl)
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self._rolling_loss_sizes.append(abs(pnl))
            self.losing_trades  += 1
            self.consecutive_losses += 1

        self.total_pnl += pnl
        self.pnl_by_tag.setdefault(exit_tag, []).append(pnl)
        self.trade_log.append({
            'time':        self.Time,
            'symbol':      symbol.Value,
            'pnl_pct':     pnl,
            'direction':   'LONG' if direction >= 0 else 'SHORT',
            'exit_reason': exit_tag,
        })

        if len(self._recent_trade_outcomes) >= 12:
            recent_wr = sum(self._recent_trade_outcomes) / len(self._recent_trade_outcomes)
            if recent_wr < 0.25:
                self._cash_mode_until = self.Time + timedelta(hours=2)
                self.Debug(f"CASH MODE: WR={recent_wr:.0%} over {len(self._recent_trade_outcomes)} trades. Pausing 2h.")

        side = "LONG" if direction >= 0 else "SHORT"
        self.Debug(f"EXIT FILL [{side}]: {symbol.Value} | entry=${entry:.2f} | exit=${exit_price:.2f} | PnL:{pnl:+.2%}")
        cleanup_position(self, symbol)
        self._failed_exit_counts.pop(symbol, None)
        self._failed_exit_attempts.pop(symbol, None)

    # ------------------------------------------------------------------
    # End of algorithm
    # ------------------------------------------------------------------

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr    = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%} | Avg PnL: {self.total_pnl / total:+.2%}" if total > 0 else "Trades: 0")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        pnl_tags = {tag: (sum(p > 0 for p in ps), len(ps)) for tag, ps in self.pnl_by_tag.items()}
        for tag, (wins, cnt) in sorted(pnl_tags.items(), key=lambda x: -x[1][1]):
            self.Debug(f"  {tag}: {wins}/{cnt} wins")
        persist_state(self)
