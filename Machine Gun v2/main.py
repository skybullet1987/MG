# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
from datetime import timedelta
from mg2_data import (
    initialize_symbol, update_symbol_data, update_market_context,
    annualized_vol, compute_portfolio_risk_estimate, universe_filter, is_ready,
)
from mg2_entries import rebalance, execute_trades
from mg2_exits import check_exits
from mg2_reporting import daily_report_v2, reset_daily_counters
# endregion


class MakerTakerFeeModel(FeeModel):
    """Blended crypto fee model: 75% taker fills for limit orders."""

    LIMIT_TAKER_RATIO = 0.75

    def GetOrderFee(self, parameters):
        order = parameters.Order
        if order.Type == OrderType.Limit:
            fee_pct = (1 - self.LIMIT_TAKER_RATIO) * 0.0025 + self.LIMIT_TAKER_RATIO * 0.0040
        else:
            fee_pct = 0.0040
        trade_value = order.AbsoluteQuantity * parameters.Security.Price
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


class SimplifiedCryptoStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetCash(1000)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        self.entry_threshold = 0.60
        self.high_conviction_threshold = 0.75

        self.quick_take_profit = self._get_param("quick_take_profit", 0.150)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.035)
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  6.0)
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  2.5)
        self.trail_activation  = self._get_param("trail_activation",  0.040)
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.025)
        self.time_stop_hours   = self._get_param("time_stop_hours",   3.0)
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.003)
        self.extended_time_stop_hours   = self._get_param("extended_time_stop_hours",   4.0)
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.015)
        self.stale_position_hours       = self._get_param("stale_position_hours",       6.0)

        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 3.5

        self.position_size_pct  = 0.50
        self.base_max_positions = 3
        self.max_positions      = 3
        self.min_notional       = 5.5
        self.max_position_usd   = self._get_param("max_position_usd", 500.0)
        self.min_price_usd      = 0.01
        self.cash_reserve_pct   = 0.00
        self.min_notional_fee_buffer = 1.5

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap", 0.80)
        self.min_asset_vol_floor     = 0.05

        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(60 * 24 * 365)

        self.max_spread_pct         = 0.004
        self.spread_median_window   = 12
        self.spread_widen_mult      = 2.5
        self.min_dollar_volume_usd  = 100000
        self.min_volume_usd         = 100000

        self.skip_hours_utc         = []
        self.max_daily_trades       = 24000
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 2.0
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 3

        self.expected_round_trip_fees = 0.0025
        self.fee_slippage_buffer      = 0.001
        self.min_expected_profit_pct  = 0.015
        self.adx_min_period           = 10

        self.stale_order_timeout_seconds      = 30
        self.live_stale_order_timeout_seconds = 60
        self.max_concurrent_open_orders       = 5
        self.open_orders_cash_threshold       = 0.90
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
        self.crypto_data      = {}
        self.entry_prices     = {}
        self.highest_prices   = {}
        self.entry_times      = {}
        self.entry_volumes    = {}
        self._partial_tp_taken      = {}
        self._breakeven_stops       = {}
        self._partial_sell_symbols  = set()
        self._choppy_regime_entries = {}
        self.partial_tp_threshold   = 0.025
        self.stagnation_minutes     = 120
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

        self.max_universe_size = 75

        self.kraken_status = "unknown"
        self._last_skip_reason = None

        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Minute, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except Exception as e:
            self.Debug(f"Warning: Could not add BTC - {e}")

        try:
            from alt_data import FearGreedData
            self.fear_greed_symbol = self.AddData(FearGreedData, "FNG", Resolution.Daily).Symbol
            self.fear_greed_value = 50
        except Exception as e:
            self.Debug(f"Warning: Could not add Fear & Greed data - {e}")
            self.fear_greed_symbol = None
            self.fear_greed_value = 50

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
            self.Debug("=== LIVE TRADING (MICRO-SCALP) v7.3.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")

    def CustomSecurityInitializer(self, security):
        security.SetSlippageModel(RealisticCryptoSlippage())
        security.SetFeeModel(MakerTakerFeeModel())
        security.SetFillModel(RealisticLimitFillModel())

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

    def UniverseFilter(self, universe):
        return universe_filter(self, universe)

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data:
                initialize_symbol(self, symbol)
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
        if hasattr(self, 'fear_greed_symbol') and self.fear_greed_symbol and data.ContainsKey(self.fear_greed_symbol):
            fg = data[self.fear_greed_symbol]
            if fg is not None:
                self.fear_greed_value = fg.Value
        for symbol in list(self.crypto_data.keys()):
            if not data.Bars.ContainsKey(symbol):
                continue
            try:
                quote_bar = data.QuoteBars[symbol] if data.QuoteBars.ContainsKey(symbol) else None
                update_symbol_data(self, symbol, data.Bars[symbol], quote_bar)
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
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
                self.Debug("Fallback: kraken_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        update_market_context(self)
        rebalance(self)
        check_exits(self)

    def _cancel_stale_orders(self):
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                self.Debug(f"Found {len(open_orders)} open orders - canceling all...")
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug(f"Error canceling stale orders: {e}")

    def Rebalance(self):
        rebalance(self)

    def CheckExits(self):
        check_exits(self)

    def DailyReport(self):
        if self.IsWarmingUp: return
        daily_report_v2(self)

    def ResetDailyCounters(self):
        reset_daily_counters(self)

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
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                if event.Direction == OrderDirection.Buy:
                    self.entry_prices[symbol] = event.FillPrice
                    self.highest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1
                    crypto = self.crypto_data.get(symbol)
                    if crypto and len(crypto['volume']) >= 1:
                        self.entry_volumes[symbol] = crypto['volume'][-1]
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
                        if not hasattr(self, 'pnl_by_tag'):
                            self.pnl_by_tag = {}
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
                                self._cash_mode_until = self.Time + timedelta(hours=2)
                                self.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(self._recent_trade_outcomes)} trades. Pausing 2h.")
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
        except Exception as e:
            self.Debug(f"BrokerageMessage parse error: {e}")

    def OnEndOfAlgorithm(self):
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%}")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"PnL: {self.total_pnl:+.2%}")

        if total > 0:
            avg_win = float(np.mean(list(self._rolling_win_sizes))) if len(self._rolling_win_sizes) > 0 else 0
            avg_loss = float(np.mean(list(self._rolling_loss_sizes))) if len(self._rolling_loss_sizes) > 0 else 0
            self.Debug("=== REALISM CHECKS ===")
            self.Debug(f"Win Rate: {wr:.1%} (realistic range: 45-65%)")
            self.Debug(f"Avg Win: {avg_win:.2%} (should be > round-trip fees 0.65%)")
            self.Debug(f"Avg Loss: {avg_loss:.2%}")
            if self.winning_trades > 0 and self.losing_trades > 0 and avg_loss > 0:
                pf = (avg_win * self.winning_trades) / (avg_loss * self.losing_trades)
                self.Debug(f"Profit Factor: {pf:.2f}")
            if wr > 0.70:
                self.Debug("⚠️ RED FLAG: Win rate > 70% — likely backtest overfitting")
            if avg_win < 0.005:
                self.Debug("⚠️ RED FLAG: Avg win < 0.5% — too small to survive live fees")
            try:
                days = max((self.Time - self.StartDate).days, 1)
                cagr = (self.Portfolio.TotalPortfolioValue / 1000) ** (365 / days) - 1
                if cagr > 5.0:
                    self.Debug("⚠️ RED FLAG: CAGR > 500% — backtest is unreliable")
            except Exception:
                pass

        persist_state(self)
