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
from mg2_entries import rebalance, execute_trades, DiagnosticsEngine
from mg2_exits import check_exits
# endregion


class MakerTakerFeeModel(FeeModel):
    """
    Conservative crypto fee model for Kraken.

    Assumes a high taker ratio (85%) to reflect realistic fill behavior:
    - Fast candles and breakout entries are rarely filled as pure maker.
    - Exit fees (stop-loss escalation) are taker rate.
    """

    LIMIT_TAKER_RATIO = 0.85  # 85% of limit fills treated as taker (conservative)

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

        # ── Entry thresholds ──────────────────────────────────────────────────
        # Now represents minimum setup confidence (not additive score)
        self.entry_threshold           = 0.55
        self.high_conviction_threshold = 0.72

        # ── Exit parameters ───────────────────────────────────────────────────
        self.quick_take_profit = self._get_param("quick_take_profit", 0.150)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.035)
        self.atr_tp_mult       = self._get_param("atr_tp_mult",       6.0)
        self.atr_sl_mult       = self._get_param("atr_sl_mult",       3.0)   # raised from 2.5 — more breathing room
        # Trailing: wider activation allows runners to develop
        self.trail_activation  = self._get_param("trail_activation",  0.060)  # 6% (was 4%)
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.040)  # 4% (was 2.5%)
        # Time exits
        self.time_stop_hours          = self._get_param("time_stop_hours",          4.0)
        self.time_stop_pnl_min        = self._get_param("time_stop_pnl_min",        0.005)
        self.extended_time_stop_hours = self._get_param("extended_time_stop_hours", 6.0)
        self.extended_time_stop_pnl_max = self._get_param("extended_time_stop_pnl_max", 0.020)
        self.stale_position_hours     = self._get_param("stale_position_hours",     8.0)
        self.stagnation_minutes       = self._get_param("stagnation_minutes",       90.0)

        self.trailing_activation = self.trail_activation
        self.trailing_stop_pct   = self.trail_stop_pct
        self.base_stop_loss      = self.tight_stop_loss
        self.base_take_profit    = self.quick_take_profit
        self.atr_trail_mult      = 3.5

        # ── Position sizing ───────────────────────────────────────────────────
        # Kelly disabled by default — ensures we measure raw entry edge
        self.use_kelly                   = False
        self.position_size_pct           = 0.35   # base size for qualifying setups
        self.position_size_high_conviction = 0.45  # high-conviction size
        self.max_position_pct            = 0.50   # hard cap per position
        self.base_max_positions          = 3
        self.max_positions               = 3
        self.min_notional                = 5.5
        self.max_position_usd            = self._get_param("max_position_usd", 500.0)
        self.min_price_usd               = 0.01
        self.cash_reserve_pct            = 0.00
        self.min_notional_fee_buffer     = 1.5

        # ── Volatility targeting ──────────────────────────────────────────────
        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.40)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap",       0.80)
        self.min_asset_vol_floor     = 0.05

        # ── Indicator periods ─────────────────────────────────────────────────
        self.ultra_short_period  = 3
        self.short_period        = 6
        self.medium_period       = 12
        self.lookback            = 48
        self.sqrt_annualization  = np.sqrt(60 * 24 * 365)

        # ── Spread / liquidity filters ────────────────────────────────────────
        self.max_spread_pct        = 0.004
        self.spread_median_window  = 12
        self.spread_widen_mult     = 2.5
        self.min_dollar_volume_usd = 100000
        self.min_volume_usd        = 100000

        # ── Trade-count limits ────────────────────────────────────────────────
        self.skip_hours_utc          = []
        self.max_daily_trades        = 24000
        self.daily_trade_count       = 0
        self.last_trade_date         = None
        self.exit_cooldown_hours     = 1.0   # reduced from 2.0 — faster re-entry into momentum runners
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 5   # raised from 3 — allow momentum re-entries

        # ── Fee / profit floor ────────────────────────────────────────────────
        self.expected_round_trip_fees = 0.008   # conservative: 0.8% round-trip (taker)
        self.fee_slippage_buffer      = 0.002   # 0.2% slippage buffer
        self.min_expected_profit_pct  = 0.015   # require at least 1.5% expected move
        self.adx_min_period           = 10

        # ── Order management ──────────────────────────────────────────────────
        self.stale_order_timeout_seconds         = 30
        self.live_stale_order_timeout_seconds    = 60
        self.max_concurrent_open_orders          = 5
        self.open_orders_cash_threshold          = 0.90
        self.order_fill_check_threshold_seconds  = 60
        self.order_timeout_seconds               = 30
        self.resync_log_interval_seconds         = 1800
        self.portfolio_mismatch_threshold        = 0.10
        self.portfolio_mismatch_min_dollars      = 1.00
        self.portfolio_mismatch_cooldown_seconds = 3600
        self.retry_pending_cooldown_seconds      = 60
        self.rate_limit_cooldown_minutes         = 10

        # ── Drawdown / risk controls ──────────────────────────────────────────
        self.max_drawdown_limit              = 0.25
        self.cooldown_hours                  = 6
        self.consecutive_losses              = 0
        self.max_consecutive_losses          = 5
        self._consecutive_loss_halve_remaining = 0
        self.circuit_breaker_expiry          = None

        # ── State ─────────────────────────────────────────────────────────────
        self._positions_synced      = False
        self._session_blacklist     = set()
        self._max_session_blacklist_size = 100
        self._symbol_entry_cooldowns = {}
        self._spread_warning_times  = {}
        self._first_post_warmup     = True
        self._submitted_orders      = {}
        self._symbol_slippage_history = {}
        self._order_retries         = {}
        self._retry_pending         = {}
        self._rate_limit_until      = None
        self._last_mismatch_warning = None
        self._failed_exit_attempts  = {}
        self._failed_exit_counts    = {}
        self._daily_open_value      = None
        self.pnl_by_tag             = {}

        self.peak_value             = None
        self.drawdown_cooldown      = 0
        self.crypto_data            = {}
        self.entry_prices           = {}
        self.highest_prices         = {}
        self.entry_times            = {}
        self.entry_volumes          = {}
        self._partial_tp_taken      = {}
        self._breakeven_stops       = {}
        self._partial_sell_symbols  = set()
        self._choppy_regime_entries = {}
        self.partial_tp_threshold   = 0.080   # updated: 8% (was 2.5%)
        self.stagnation_pnl_threshold = 0.006
        self.rsi_peaked_overbought  = {}
        self.trade_count            = 0
        self._pending_orders        = {}
        self._cancel_cooldowns      = {}
        self._exit_cooldowns        = {}
        self._symbol_loss_cooldowns = {}
        self._cash_mode_until       = None
        self._recent_trade_outcomes = deque(maxlen=20)
        self.trailing_grace_hours   = 1
        self._slip_abs              = deque(maxlen=50)
        self._slippage_alert_until  = None
        self.slip_alert_threshold   = 0.0015
        self.slip_outlier_threshold = 0.004
        self.slip_alert_duration_hours = 2
        self._bad_symbol_counts     = {}
        self._recent_tickets        = deque(maxlen=25)

        self._rolling_wins          = deque(maxlen=50)
        self._rolling_win_sizes     = deque(maxlen=50)
        self._rolling_loss_sizes    = deque(maxlen=50)
        self._last_live_trade_time  = None

        # ── BTC / market context ──────────────────────────────────────────────
        self.btc_symbol      = None
        self.btc_returns     = deque(maxlen=72)
        self.btc_prices      = deque(maxlen=72)
        self.btc_volatility  = deque(maxlen=72)
        self.btc_ema_24      = ExponentialMovingAverage(24)
        self.market_regime   = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth  = 0.5
        self._regime_hold_count = 0

        # ── Performance tracking ──────────────────────────────────────────────
        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.trade_log      = []
        self.log_budget     = 0
        self.last_log_time  = None

        self.max_universe_size = 75
        self.kraken_status     = "unknown"
        self._last_skip_reason = None

        # ── LEAN setup ────────────────────────────────────────────────────────
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
            self.fear_greed_value  = 50
        except Exception as e:
            self.Debug(f"Warning: Could not add Fear & Greed data - {e}")
            self.fear_greed_symbol = None
            self.fear_greed_value  = 50

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1),   self.DailyReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0),   self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=6)), self.ReviewPerformance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0),  self.HealthCheck)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=5)),  self.ResyncHoldings)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=2)),  self.VerifyOrderFills)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=15)), self.PortfolioSanityCheck)

        self.SetWarmUp(timedelta(days=4))
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.Settings.FreePortfolioValuePercentage = 0.01
        self.Settings.InsightScore = False

        self._scoring_engine = MicroScalpEngine(self)

        # ── Diagnostics engine ────────────────────────────────────────────────
        self._diagnostics = DiagnosticsEngine(self)

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MOMENTUM BREAKOUT RUNNER) v8.0.0 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")
            self.Debug(f"Kelly: {'ON' if self.use_kelly else 'OFF (flat sizing)'}")
            self.Debug("Setup families: IgnitionBreakout | CompressionExpansion | MomentumContinuation")

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

    def _kelly_fraction(self):
        return kelly_fraction(self)

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
        _daily_report(self)

    def ResetDailyCounters(self):
        _reset_daily_counters(self)

    def HealthCheck(self):
        if self.IsWarmingUp: return
        _health_check(self)

    def ResyncHoldings(self):
        if self.IsWarmingUp: return
        if not self.LiveMode: return
        resync_holdings_full(self)

    def VerifyOrderFills(self):
        if self.IsWarmingUp: return
        verify_order_fills(self)

    def PortfolioSanityCheck(self):
        if self.IsWarmingUp: return
        _portfolio_sanity_check(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10: return
        _review_performance(self)

    def OnOrderEvent(self, event):
        _on_order_event(self, event)

    def OnBrokerageMessage(self, message):
        _on_brokerage_message(self, message)

    def OnEndOfAlgorithm(self):
        _on_end_of_algorithm(self)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers (merged from mg2_reporting.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_setup_type_for_exit(algo, symbol):
    """Retrieve the setup type from open diagnostics record if available."""
    diag = getattr(algo, '_diagnostics', None)
    if diag is None:
        return None
    rec = diag._open_records.get(symbol)
    if rec is not None:
        return rec.setup_type
    return None


def _daily_report(algo):
    """Generate daily report with portfolio status and position details."""
    if algo.IsWarmingUp:
        return
    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    avg = algo.total_pnl / total if total > 0 else 0
    algo.Debug(f"=== DAILY {algo.Time.date()} ===")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug(f"Pos: {get_actual_position_count(algo)}/{algo.base_max_positions} | {algo.market_regime} {algo.volatility_regime} {algo.market_breadth:.0%}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    if algo._session_blacklist:
        algo.Debug(f"Blacklist: {len(algo._session_blacklist)}")
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key):
            s = kvp.Key
            entry = algo.entry_prices.get(s, kvp.Value.AveragePrice)
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            pnl = (cur - entry) / entry if entry > 0 else 0
            algo.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")
    persist_state(algo)


def _reset_daily_counters(algo):
    algo.daily_trade_count = 0
    algo.last_trade_date = algo.Time.date()
    algo._daily_open_value = algo.Portfolio.TotalPortfolioValue
    for crypto in algo.crypto_data.values():
        crypto['trade_count_today'] = 0
    if len(algo._session_blacklist) > 0:
        algo.Debug(f"Clearing session blacklist ({len(algo._session_blacklist)} items)")
        algo._session_blacklist.clear()
    algo._symbol_entry_cooldowns.clear()
    persist_state(algo)


def _on_order_event(algo, event):
    try:
        symbol = event.Symbol
        algo.Debug(f"ORDER: {symbol.Value} {event.Status} {event.Direction} qty={event.FillQuantity or event.Quantity} price={event.FillPrice} id={event.OrderId}")
        if event.Status == OrderStatus.Submitted:
            if symbol not in algo._pending_orders:
                algo._pending_orders[symbol] = 0
            intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
            algo._pending_orders[symbol] += intended_qty
            if symbol not in algo._submitted_orders:
                has_position = symbol in algo.Portfolio and algo.Portfolio[symbol].Invested
                if event.Direction == OrderDirection.Sell and has_position:
                    inferred_intent = 'exit'
                elif event.Direction == OrderDirection.Buy and not has_position:
                    inferred_intent = 'entry'
                else:
                    inferred_intent = 'entry' if event.Direction == OrderDirection.Buy else 'exit'
                algo._submitted_orders[symbol] = {
                    'order_id': event.OrderId,
                    'time': algo.Time,
                    'quantity': event.Quantity,
                    'intent': inferred_intent
                }
            else:
                algo._submitted_orders[symbol]['order_id'] = event.OrderId
        elif event.Status == OrderStatus.PartiallyFilled:
            if symbol in algo._pending_orders:
                algo._pending_orders[symbol] -= abs(event.FillQuantity)
                if algo._pending_orders[symbol] <= 0:
                    algo._pending_orders.pop(symbol, None)
            if event.Direction == OrderDirection.Buy:
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol] = event.FillPrice
                    algo.highest_prices[symbol] = event.FillPrice
                    algo.entry_times[symbol] = algo.Time
            slip_log(algo, symbol, event.Direction, event.FillPrice)
        elif event.Status == OrderStatus.Filled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Buy:
                algo.entry_prices[symbol] = event.FillPrice
                algo.highest_prices[symbol] = event.FillPrice
                algo.entry_times[symbol] = algo.Time
                algo.daily_trade_count += 1
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto['volume']) >= 1:
                    algo.entry_volumes[symbol] = crypto['volume'][-1]
                algo.rsi_peaked_overbought.pop(symbol, None)
            else:
                if symbol in algo._partial_sell_symbols:
                    algo._partial_sell_symbols.discard(symbol)
                else:
                    order = algo.Transactions.GetOrderById(event.OrderId)
                    exit_tag = order.Tag if order and order.Tag else "Unknown"
                    entry = algo.entry_prices.get(symbol, None)
                    if entry is None:
                        entry = event.FillPrice
                        algo.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
                    pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                    algo._rolling_wins.append(1 if pnl > 0 else 0)
                    algo._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                    if pnl > 0:
                        algo._rolling_win_sizes.append(pnl)
                        algo.winning_trades += 1
                        algo.consecutive_losses = 0
                    else:
                        algo._rolling_loss_sizes.append(abs(pnl))
                        algo.losing_trades += 1
                        algo.consecutive_losses += 1
                    algo.total_pnl += pnl
                    if not hasattr(algo, 'pnl_by_tag'):
                        algo.pnl_by_tag = {}
                    if exit_tag not in algo.pnl_by_tag:
                        algo.pnl_by_tag[exit_tag] = []
                    algo.pnl_by_tag[exit_tag].append(pnl)
                    algo.trade_log.append({
                        'time':         algo.Time,
                        'symbol':       symbol.Value,
                        'pnl_pct':      pnl,
                        'exit_reason':  exit_tag,
                        'setup_type':   _get_setup_type_for_exit(algo, symbol),
                        'market_regime': algo.market_regime,
                        'vol_regime':   algo.volatility_regime,
                    })
                    if len(algo._recent_trade_outcomes) >= 12:
                        recent_wr = sum(algo._recent_trade_outcomes) / len(algo._recent_trade_outcomes)
                        if recent_wr < 0.25:
                            algo._cash_mode_until = algo.Time + timedelta(hours=2)
                            algo.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(algo._recent_trade_outcomes)} trades. Pausing 2h.")
                    cleanup_position(algo, symbol)
                    algo._failed_exit_attempts.pop(symbol, None)
                    algo._failed_exit_counts.pop(symbol, None)
            slip_log(algo, symbol, event.Direction, event.FillPrice)
        elif event.Status == OrderStatus.Canceled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Sell and symbol not in algo.entry_prices:
                if is_invested_not_dust(algo, symbol):
                    holding = algo.Portfolio[symbol]
                    algo.entry_prices[symbol] = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol] = algo.Time
                    algo.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
        elif event.Status == OrderStatus.Invalid:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Sell:
                price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
                min_notional = get_min_notional_usd(algo, symbol)
                if price > 0 and symbol in algo.Portfolio and abs(algo.Portfolio[symbol].Quantity) * price < min_notional:
                    algo.Debug(f"DUST CLEANUP on invalid sell: {symbol.Value} — releasing tracking")
                    cleanup_position(algo, symbol)
                    algo._failed_exit_counts.pop(symbol, None)
                else:
                    fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
                    algo._failed_exit_counts[symbol] = fail_count
                    algo.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                    if fail_count >= 3:
                        algo.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} failed exits — releasing tracking")
                        cleanup_position(algo, symbol)
                        algo._failed_exit_counts.pop(symbol, None)
                    elif symbol not in algo.entry_prices:
                        if is_invested_not_dust(algo, symbol):
                            holding = algo.Portfolio[symbol]
                            algo.entry_prices[symbol] = holding.AveragePrice
                            algo.highest_prices[symbol] = holding.AveragePrice
                            algo.entry_times[symbol] = algo.Time
                            algo.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
            algo._session_blacklist.add(symbol.Value)
    except Exception as e:
        algo.Debug(f"OnOrderEvent error: {e}")
    if algo.LiveMode:
        persist_state(algo)


def _on_brokerage_message(algo, message):
    try:
        txt = message.Message.lower()
        if "system status:" in txt:
            if "online" in txt:
                algo.kraken_status = "online"
            elif "maintenance" in txt:
                algo.kraken_status = "maintenance"
            elif "cancel_only" in txt:
                algo.kraken_status = "cancel_only"
            elif "post_only" in txt:
                algo.kraken_status = "post_only"
            else:
                algo.kraken_status = "unknown"
            algo.Debug(f"Kraken status update: {algo.kraken_status}")
        if "rate limit" in txt or "too many" in txt:
            algo.Debug(f"⚠️ RATE LIMIT - pausing {algo.rate_limit_cooldown_minutes}min")
            algo._rate_limit_until = algo.Time + timedelta(minutes=algo.rate_limit_cooldown_minutes)
    except Exception as e:
        algo.Debug(f"BrokerageMessage parse error: {e}")


def _on_end_of_algorithm(algo):
    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    algo.Debug("=== FINAL REPORT — Machine Gun v2 Momentum Breakout Runner v8.0.0 ===")
    algo.Debug(f"Trades: {algo.trade_count} | WR: {wr:.1%}")
    algo.Debug(f"Final: ${algo.Portfolio.TotalPortfolioValue:.2f}")
    algo.Debug(f"PnL: {algo.total_pnl:+.2%}")

    if total > 0:
        avg_win  = float(np.mean(list(algo._rolling_win_sizes)))  if len(algo._rolling_win_sizes)  > 0 else 0
        avg_loss = float(np.mean(list(algo._rolling_loss_sizes))) if len(algo._rolling_loss_sizes) > 0 else 0
        algo.Debug("=== REALISM CHECKS ===")
        algo.Debug(f"Win Rate: {wr:.1%} (target range: 45-65% for this strategy)")
        algo.Debug(f"Avg Win: {avg_win:.2%} (should be > round-trip fees 0.80%)")
        algo.Debug(f"Avg Loss: {avg_loss:.2%}")
        if algo.winning_trades > 0 and algo.losing_trades > 0 and avg_loss > 0:
            pf = (avg_win * algo.winning_trades) / (avg_loss * algo.losing_trades)
            algo.Debug(f"Profit Factor: {pf:.2f}")
        if wr > 0.70:
            algo.Debug("WARNING: Win rate > 70% — possible backtest overfitting")
        if avg_win < 0.008:
            algo.Debug("WARNING: Avg win < 0.8% — too small to survive live fees")
        try:
            days = max((algo.Time - algo.StartDate).days, 1)
            cagr = (algo.Portfolio.TotalPortfolioValue / 1000) ** (365 / days) - 1
            if cagr > 5.0:
                algo.Debug("WARNING: CAGR > 500% — backtest is likely unreliable")
        except Exception:
            pass

    if hasattr(algo, 'pnl_by_tag') and algo.pnl_by_tag:
        algo.Debug("=== PNL BY EXIT TAG ===")
        for tag, pnls in sorted(algo.pnl_by_tag.items()):
            n = len(pnls)
            avg = float(np.mean(pnls)) if pnls else 0
            wins = sum(1 for p in pnls if p > 0)
            algo.Debug(f"  {tag}: n={n} wr={wins/n:.1%} avg={avg:+.2%}")

    diag = getattr(algo, '_diagnostics', None)
    if diag is not None:
        diag.print_summary()

    persist_state(algo)


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring helpers (merged from execution.py)
# ─────────────────────────────────────────────────────────────────────────────

def _health_check(algo):
    """
    Enhanced health check with improved orphan detection and reverse resync.
    """
    if algo.IsWarmingUp:
        return

    resync_holdings_full(algo)

    issues = []
    if algo.Portfolio.Cash < 5:
        issues.append(f"Low cash: ${algo.Portfolio.Cash:.2f}")

    for symbol in list(algo.entry_prices.keys()):
        open_orders = algo.Transactions.GetOpenOrders(symbol)
        if len(open_orders) > 0:
            all_stale = True
            for o in open_orders:
                order_time = normalize_order_time(o.Time)
                if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                    all_stale = False
                    break
            if not all_stale:
                continue
            for o in open_orders:
                try:
                    algo.Transactions.CancelOrder(o.Id)
                    issues.append(f"Canceled stale order: {symbol.Value} (order {o.Id})")
                except Exception as e:
                    algo.Debug(f"Error canceling stale order for {symbol.Value}: {e}")

        if not is_invested_not_dust(algo, symbol):
            issues.append(f"Orphan tracking: {symbol.Value}")
            cleanup_position(algo, symbol)

    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key) and kvp.Key not in algo.entry_prices:
            issues.append(f"Untracked position: {kvp.Key.Value}")

    if len(algo._session_blacklist) > 50:
        issues.append(f"Large session blacklist: {len(algo._session_blacklist)}")

    open_orders = algo.Transactions.GetOpenOrders()
    if len(open_orders) > 0:
        issues.append(f"Open orders: {len(open_orders)}")

    if issues:
        algo.Debug("=== HEALTH CHECK ===")
        for issue in issues:
            algo.Debug(f"  ⚠️ {issue}")
    else:
        debug_limited(algo, "Health check: OK")


def _portfolio_sanity_check(algo):
    """
    Check for portfolio value mismatches between QC and tracked positions.
    """
    if algo.IsWarmingUp:
        return

    total_qc = algo.Portfolio.TotalPortfolioValue

    try:
        usd_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        usd_cash = algo.Portfolio.Cash

    tracked_value = 0.0
    tracked_positions = {}

    for sym in list(algo.entry_prices.keys()):
        if sym in algo.Securities:
            price = algo.Securities[sym].Price
            if sym in algo.Portfolio:
                qty = algo.Portfolio[sym].Quantity
                value = abs(qty) * price
                tracked_value += value
                tracked_positions[sym.Value] = {'qty': qty, 'price': price, 'value': value}

    expected = usd_cash + tracked_value
    abs_diff = abs(total_qc - expected)
    if total_qc > 1.0:
        pct_diff = abs_diff / total_qc
        should_warn = pct_diff > algo.portfolio_mismatch_threshold and abs_diff > algo.portfolio_mismatch_min_dollars
        if should_warn:
            if algo._last_mismatch_warning is None or (algo.Time - algo._last_mismatch_warning).total_seconds() >= algo.portfolio_mismatch_cooldown_seconds:
                algo.Debug(f"⚠️ PORTFOLIO MISMATCH: QC total=${total_qc:.2f} but usd_cash+tracked=${expected:.2f} (diff=${abs_diff:.2f}, {pct_diff:.2%})")
                algo.Debug("=== PORTFOLIO BREAKDOWN ===")
                algo.Debug(f"USD Cash: ${usd_cash:.2f}")
                if tracked_positions:
                    algo.Debug(f"Tracked Positions ({len(tracked_positions)}):")
                    for ticker, info in tracked_positions.items():
                        algo.Debug(f"  {ticker}: qty={info['qty']:.6f}, price=${info['price']:.4f}, value=${info['value']:.2f}")
                else:
                    algo.Debug("Tracked Positions: None")
                untracked = []
                for symbol in algo.Portfolio.Keys:
                    holding = algo.Portfolio[symbol]
                    if holding.Invested and holding.Quantity != 0:
                        if symbol not in algo.entry_prices:
                            price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                            value = abs(holding.Quantity) * price
                            untracked.append({'ticker': symbol.Value, 'qty': holding.Quantity, 'price': price, 'value': value})
                if untracked:
                    algo.Debug(f"Untracked Portfolio Holdings ({len(untracked)}):")
                    for info in untracked:
                        algo.Debug(f"  {info['ticker']}: qty={info['qty']:.6f}, price=${info['price']:.4f}, value=${info['value']:.2f}")
                else:
                    algo.Debug("Untracked Portfolio Holdings: None")
                crypto_cash = []
                try:
                    for currency, cash in algo.Portfolio.CashBook.items():
                        if currency != "USD" and cash.Amount != 0:
                            crypto_cash.append({'currency': currency, 'amount': cash.Amount, 'value': cash.ValueInAccountCurrency})
                except Exception:
                    pass
                if crypto_cash:
                    algo.Debug(f"Crypto CashBook Entries ({len(crypto_cash)}):")
                    for info in crypto_cash:
                        algo.Debug(f"  {info['currency']}: amount={info['amount']:.6f}, value=${info['value']:.2f}")
                else:
                    algo.Debug("Crypto CashBook Entries: None")
                if total_qc > expected:
                    algo.Debug(f"QC Portfolio is higher by ${abs_diff:.2f} ({pct_diff:.2%})")
                else:
                    algo.Debug(f"Tracked value is higher by ${abs_diff:.2f} ({pct_diff:.2%})")
                algo.Debug("=== END BREAKDOWN ===")
                algo.Debug("Triggering resync_holdings_full to attempt auto-fix...")
                resync_holdings_full(algo)
                algo._last_mismatch_warning = algo.Time


def _review_performance(algo):
    """Review recent performance and adjust max_positions accordingly."""
    if algo.IsWarmingUp or len(algo.trade_log) < 10:
        return

    recent_trades = algo.trade_log[-15:] if len(algo.trade_log) >= 15 else algo.trade_log
    if len(recent_trades) == 0:
        return

    recent_win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
    recent_avg_pnl = np.mean([t['pnl_pct'] for t in recent_trades])
    old_max = algo.max_positions

    if recent_win_rate < 0.2 and recent_avg_pnl < -0.05:
        algo.max_positions = 1
        if old_max != 1:
            algo.Debug(f"PERFORMANCE DECAY: max_pos=1 (WR:{recent_win_rate:.0%}, PnL:{recent_avg_pnl:+.2%})")
    elif recent_win_rate > 0.35 and recent_avg_pnl > -0.01:
        algo.max_positions = algo.base_max_positions
        if old_max != algo.base_max_positions:
            algo.Debug(f"PERFORMANCE RECOVERY: max_pos={algo.base_max_positions}")