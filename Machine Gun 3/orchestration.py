# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, spread_ok, cancel_stale_new_orders, has_open_orders,
    get_actual_position_count, get_min_quantity, get_min_notional_usd, round_quantity,
    place_limit_or_market, get_open_buy_orders_value, debug_limited,
    get_slippage_penalty, live_safety_checks, KRAKEN_SELL_FEE_BUFFER, SYMBOL_BLACKLIST,
)
import numpy as np
import config as MG3Config
from datetime import timedelta
from app import POSITION_STATE_ENTERING
# endregion


class OrchestrationMixin:
    """Core entry-side trading loop for SimplifiedCryptoStrategy.

    Contains the Rebalance gate checks and _execute_trades position-entry logic.
    """

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

        # MG3: max daily loss guard
        if not self.IsWarmingUp and self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded")
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
        pos_count = get_actual_position_count(self)
        if pos_count >= self.max_positions:
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
        self._execute_trades(scores, threshold_now)

    def _get_open_buy_orders_value(self):
        """Calculate total value reserved by open buy orders."""
        return get_open_buy_orders_value(self)

    def _execute_trades(self, candidates, threshold_now):
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
            if get_actual_position_count(self) >= self.max_positions:
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
                min_profit_gate = self.min_expected_profit_pct
                min_required = self.expected_round_trip_fees + self.fee_slippage_buffer + min_profit_gate
                if expected_move_pct < min_required:
                    continue

            # Dollar-volume liquidity gate: use 12-bar average for more stable assessment
            if len(crypto['dollar_volume']) >= 3:
                dv_window = min(len(crypto['dollar_volume']), 12)
                recent_dv = np.mean(list(crypto['dollar_volume'])[-dv_window:])
                dv_threshold = self.min_dollar_volume_usd
                if recent_dv < dv_threshold:
                    reject_dollar_volume += 1
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

            # Floor at min_notional to support small accounts
            val = max(val, self.min_notional)

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
                ticket = place_limit_or_market(self, sym, qty, timeout_seconds=30, tag="Entry")
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
                    # MG3: transition to ENTERING state
                    self._set_position_state(sym, POSITION_STATE_ENTERING)
                    # Record ADX regime at entry for tighter TP in choppy markets
                    adx_ind = crypto.get('adx')
                    is_choppy = (adx_ind is not None and adx_ind.IsReady
                                 and adx_ind.Current.Value < 25)
                    self._choppy_regime_entries[sym] = is_choppy
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
