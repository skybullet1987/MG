# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, get_min_notional_usd, cleanup_position,
    persist_state, slip_log, daily_report, get_actual_position_count,
)
from datetime import timedelta
import config as MG3Config
from app import POSITION_STATE_FLAT, POSITION_STATE_OPEN, POSITION_STATE_EXITING
# endregion


class ReportingMixin:
    """Order-event handling, brokerage-message parsing, and reporting for SimplifiedCryptoStrategy.

    Contains OnOrderEvent, OnBrokerageMessage, OnEndOfAlgorithm, backtest-metrics
    logging, and the daily performance report.
    """

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
                    # MG3: partial fill → position is now at least partially OPEN
                    self._set_position_state(symbol, POSITION_STATE_OPEN)
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Filled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)  # Remove from verification tracking
                self._order_retries.pop(event.OrderId, None)  # Clean up retry tracking
                # MG3: increment fill counter
                self.mg3_fill_count += 1
                if event.Direction == OrderDirection.Buy:
                    self.entry_prices[symbol] = event.FillPrice
                    self.highest_prices[symbol] = event.FillPrice
                    self.entry_times[symbol] = self.Time
                    self.daily_trade_count += 1
                    # MG3: position fully entered → OPEN
                    self._set_position_state(symbol, POSITION_STATE_OPEN)
                    # MG3: track peak simultaneous open positions
                    current_pos = get_actual_position_count(self)
                    if current_pos > self.mg3_peak_positions:
                        self.mg3_peak_positions = current_pos

                    crypto = self.crypto_data.get(symbol)
                    if crypto and len(crypto['volume']) >= 1:
                        self.entry_volumes[symbol] = crypto['volume'][-1]
                    self.rsi_peaked_overbought.pop(symbol, None)
                else:
                    if symbol in self._partial_sell_symbols:
                        self._partial_sell_symbols.discard(symbol)
                    else:
                        entry = self.entry_prices.get(symbol, None)
                        if entry is None:
                            entry = event.FillPrice
                            self.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
                        pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                        # MG3: record hold duration for avg-hold-time metric
                        hold_hours = (
                            (self.Time - self.entry_times[symbol]).total_seconds() / 3600
                            if symbol in self.entry_times else 0.0
                        )
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
                            'hold_hours': hold_hours,
                            'exit_reason': 'filled_sell',
                        })

                        if len(self._recent_trade_outcomes) >= 8:
                            recent_wr = sum(self._recent_trade_outcomes) / len(self._recent_trade_outcomes)
                            if recent_wr < 0.25:
                                self._cash_mode_until = self.Time + timedelta(hours=24)
                                self.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(self._recent_trade_outcomes)} trades. Pausing 24h.")
                        cleanup_position(self, symbol)
                        # MG3: position exited → FLAT
                        self._set_position_state(symbol, POSITION_STATE_FLAT)
                        self._failed_exit_attempts.pop(symbol, None)
                        self._failed_exit_counts.pop(symbol, None)
                slip_log(self, symbol, event.Direction, event.FillPrice)
            elif event.Status == OrderStatus.Canceled:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                # MG3: increment cancel counter
                self.mg3_cancel_count += 1
                if event.Direction == OrderDirection.Sell and symbol not in self.entry_prices:
                    if is_invested_not_dust(self, symbol):
                        holding = self.Portfolio[symbol]
                        self.entry_prices[symbol] = holding.AveragePrice
                        self.highest_prices[symbol] = holding.AveragePrice
                        self.entry_times[symbol] = self.Time
                        self.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
                    # MG3: if cancel of an entry order, revert to FLAT
                    if event.Direction == OrderDirection.Buy:
                        self._set_position_state(symbol, POSITION_STATE_FLAT)
            elif event.Status == OrderStatus.Invalid:
                self._pending_orders.pop(symbol, None)
                self._submitted_orders.pop(symbol, None)
                self._order_retries.pop(event.OrderId, None)
                # MG3: increment invalid order counter
                self.mg3_invalid_count += 1
                if event.Direction == OrderDirection.Sell:
                    price = self.Securities[symbol].Price if symbol in self.Securities else 0
                    min_notional = get_min_notional_usd(self, symbol)

                    if price > 0 and symbol in self.Portfolio and abs(self.Portfolio[symbol].Quantity) * price < min_notional:
                        self.Debug(f"DUST CLEANUP on invalid sell: {symbol.Value} — releasing tracking")
                        cleanup_position(self, symbol)
                        self._set_position_state(symbol, POSITION_STATE_FLAT)
                        self._failed_exit_counts.pop(symbol, None)
                    else:
                        fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                        self._failed_exit_counts[symbol] = fail_count
                        self.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                        if fail_count >= 3:
                            self.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} failed exits — releasing tracking")
                            cleanup_position(self, symbol)
                            self._set_position_state(symbol, POSITION_STATE_FLAT)
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

        # MG3: emit backtest validation metrics
        self._log_mg3_metrics()

        persist_state(self)

    def _log_mg3_metrics(self):
        """Log MG3-specific backtest validation and forensic metrics.

        These numbers are the primary signal-quality gate before moving from
        backtest to paper or live (see README Phased Rollout – Phase 1).

        Acceptance thresholds:
          - Cancel-to-fill ratio  < 2
          - Invalid orders        < 5 % of fills
          - Stop Loss avg PnL     more positive than -tight_stop_loss (slippage check)
          - Recovery events       ideally 0; > 5 indicates systemic exit failure
        """
        self.Debug("=== MG3 BACKTEST METRICS ===")

        # --- Order quality ---
        if self.mg3_fill_count > 0:
            ctf = self.mg3_cancel_count / self.mg3_fill_count
            invalid_pct = self.mg3_invalid_count / self.mg3_fill_count
            self.Debug(
                f"Cancel-to-fill ratio : {ctf:.2f}  "
                f"({self.mg3_cancel_count} cancels / {self.mg3_fill_count} fills)"
            )
            self.Debug(
                f"Invalid orders       : {self.mg3_invalid_count}  "
                f"({invalid_pct:.1%} of fills)"
            )
        else:
            self.Debug(f"Cancel-to-fill ratio : N/A  (fills=0, cancels={self.mg3_cancel_count})")
            self.Debug(f"Invalid orders       : {self.mg3_invalid_count}")

        # --- Recovery events (failed exits escalated to force-market) ---
        self.Debug(f"Recovery events      : {self.mg3_recovery_events}")

        # --- Peak concurrent positions ---
        self.Debug(f"Peak open positions  : {self.mg3_peak_positions}")

        # --- Hold-time statistics (from trade_log) ---
        hold_times = [t.get('hold_hours', 0) for t in self.trade_log if t.get('hold_hours', 0) > 0]
        if hold_times:
            avg_hold = sum(hold_times) / len(hold_times)
            max_hold = max(hold_times)
            self.Debug(f"Avg hold time        : {avg_hold:.2f}h  (max {max_hold:.1f}h)")
        else:
            self.Debug("Avg hold time        : N/A")

        # --- Estimated round-trip fees paid ---
        # Approximation: fills / 2 round trips × FEE_ASSUMPTION_RT × avg position size.
        # Exact fee data is in the QC trade blotter; this is a sanity-check estimate.
        rt_count = self.mg3_fill_count // 2
        est_fee_pct = MG3Config.FEE_ASSUMPTION_RT + MG3Config.SLIPPAGE_BUFFER
        self.Debug(
            f"Est. cost per RT     : {est_fee_pct:.2%}  "
            f"(fee {MG3Config.FEE_ASSUMPTION_RT:.2%} + slip {MG3Config.SLIPPAGE_BUFFER:.2%})  "
            f"× ~{rt_count} round trips"
        )

        # --- PnL by exit tag ---
        if self.mg3_pnl_by_tag:
            self.Debug("PnL by exit tag:")
            for tag, pnls in sorted(self.mg3_pnl_by_tag.items()):
                n = len(pnls)
                avg = sum(pnls) / n if n > 0 else 0.0
                wins = sum(1 for p in pnls if p > 0)
                wr = wins / n if n > 0 else 0.0
                self.Debug(f"  {tag:<30} n={n:>4}  avg={avg:+.3%}  wr={wr:.0%}")
        else:
            self.Debug("PnL by exit tag      : (no tagged exits recorded)")

        self.Debug("=== END MG3 METRICS ===")

    def DailyReport(self):
        if self.IsWarmingUp: return
        daily_report(self)
