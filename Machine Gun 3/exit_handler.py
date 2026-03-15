# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, get_min_notional_usd, round_quantity, smart_liquidate,
    partial_smart_sell, cleanup_position,
)
from datetime import timedelta
import config as MG3Config
from app import (
    POSITION_STATE_FLAT, POSITION_STATE_OPEN, POSITION_STATE_EXITING,
    POSITION_STATE_RECOVERING,
)
# endregion


class ExitHandlerMixin:
    """Exit-side trading logic for SimplifiedCryptoStrategy.

    Contains the _is_ready guard, CheckExits loop, and the per-position _check_exit
    state machine that decides when and how to close positions.
    """

    def _is_ready(self, c):
        return len(c['prices']) >= 10 and c['rsi'].IsReady

    def CheckExits(self):
        if self.IsWarmingUp:
            return

        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            return
        for kvp in self.Portfolio:
            symbol = kvp.Key
            if not is_invested_not_dust(self, symbol):
                self._failed_exit_attempts.pop(symbol, None)
                self._failed_exit_counts.pop(symbol, None)
                continue

            state = self._get_position_state(symbol)

            # RECOVERING: exit retries exhausted — use a direct market order.
            if state == POSITION_STATE_RECOVERING:
                self._force_market_liquidate(symbol)
                continue

            # Belt-and-suspenders: if failure count reached threshold but state
            # was not updated (e.g. failures counted via OnOrderEvent Invalid),
            # escalate now.
            if self._failed_exit_counts.get(symbol, 0) >= MG3Config.MAX_EXIT_RETRIES:
                self._set_position_state(symbol, POSITION_STATE_RECOVERING)
                self._force_market_liquidate(symbol)
                continue

            self._check_exit(symbol, self.Securities[symbol].Price, kvp.Value)

        for kvp in self.Portfolio:
            symbol = kvp.Key
            if not is_invested_not_dust(self, symbol):
                continue
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = kvp.Value.AveragePrice
                self.highest_prices[symbol] = kvp.Value.AveragePrice
                self.entry_times[symbol] = self.Time
                self.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")

    def _force_market_liquidate(self, symbol):
        """Force market liquidation for a position stuck in RECOVERING state.

        Called when the normal limit-exit path has failed MAX_EXIT_RETRIES times.
        Uses a market order so the position is closed regardless of spread.
        Does NOT call cleanup_position – OnOrderEvent handles cleanup on fill.
        """
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return  # exit already in flight
        if symbol not in self.Portfolio or self.Portfolio[symbol].Quantity == 0:
            self._set_position_state(symbol, POSITION_STATE_FLAT)
            self._failed_exit_counts.pop(symbol, None)
            return
        holding = self.Portfolio[symbol]
        qty = abs(holding.Quantity)
        if qty == 0:
            self._set_position_state(symbol, POSITION_STATE_FLAT)
            return
        direction_mult = -1 if holding.Quantity > 0 else 1
        entry = self.entry_prices.get(symbol, holding.AveragePrice)
        price = self.Securities[symbol].Price if symbol in self.Securities else 0
        pnl = (price - entry) / entry if entry > 0 and price > 0 else 0
        self.Debug(f"⚠️ FORCE MARKET LIQUIDATE (recovering): {symbol.Value} qty={qty:.6f} PnL:{pnl:+.2%}")
        try:
            self.MarketOrder(symbol, qty * direction_mult, tag="Force Recovery Exit")
            self._set_position_state(symbol, POSITION_STATE_EXITING)
            self._failed_exit_counts.pop(symbol, None)
            self.mg3_recovery_events += 1
        except Exception as e:
            self.Debug(f"FORCE MARKET LIQUIDATE failed for {symbol.Value}: {e}")

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

        if tp < sl * 1.5:
            tp = sl * 1.5

        # Tighten take-profit when this position was entered in a choppy (ADX < 25) regime
        if self._choppy_regime_entries.get(symbol, False):
            tp = tp * 0.65   # 35% tighter TP – trend continuation unlikely in choppy market

        # Tighten take-profit in low-volatility regime to secure profits faster
        if self.volatility_regime == "low":
            tp = tp * 0.75   # 25% tighter TP – low-vol moves mean-revert quickly

        trailing_activation = self.trail_activation
        trailing_stop_pct   = self.trail_stop_pct

        if crypto and crypto['rsi'].IsReady:
            rsi_now = crypto['rsi'].Current.Value
            if rsi_now > 85:
                self.rsi_peaked_overbought[symbol] = True

        if (not self._partial_tp_taken.get(symbol, False)
                and pnl >= self.partial_tp_threshold):
            if partial_smart_sell(self, symbol, 0.50, "Partial TP"):
                self._partial_tp_taken[symbol] = True
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

        if not tag and minutes > self.stagnation_minutes and pnl < self.stagnation_pnl_threshold:
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
                if self.rsi_peaked_overbought.get(symbol, False) and rsi_now < 75:
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
            # MG3: transition to EXITING state before placing sell order
            self._set_position_state(symbol, POSITION_STATE_EXITING)
            sold = smart_liquidate(self, symbol, tag)
            if sold:
                self._exit_cooldowns[symbol] = self.Time + timedelta(hours=self.exit_cooldown_hours)
                self.rsi_peaked_overbought.pop(symbol, None)
                self.entry_volumes.pop(symbol, None)
                self._choppy_regime_entries.pop(symbol, None)
                # MG3: record PnL by exit tag for post-backtest analysis
                if tag not in self.mg3_pnl_by_tag:
                    self.mg3_pnl_by_tag[tag] = []
                self.mg3_pnl_by_tag[tag].append(pnl)
                self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
            else:
                # smart_liquidate returned False: the position is STILL held.
                # Do NOT call cleanup_position here — that would wipe entry_prices,
                # highest_prices, and entry_times while the position is open, which
                # causes the next exit check to lose PnL context and re-track the
                # position as an orphan with a wrong entry price.
                # Instead: increment the failure counter and keep all tracking intact.
                # After MAX_EXIT_RETRIES failures the position is escalated to
                # RECOVERING and a force market order is used.
                fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                self._failed_exit_counts[symbol] = fail_count
                if fail_count >= MG3Config.MAX_EXIT_RETRIES:
                    self._set_position_state(symbol, POSITION_STATE_RECOVERING)
                    self.Debug(
                        f"⚠️ EXIT FAILED ({fail_count}×, RECOVERING): {symbol.Value} "
                        f"| PnL:{pnl:+.2%} | Held:{hours:.1f}h — escalating to force-market"
                    )
                else:
                    self._set_position_state(symbol, POSITION_STATE_OPEN)
                    self.Debug(
                        f"⚠️ EXIT FAILED ({fail_count}/{MG3Config.MAX_EXIT_RETRIES}): "
                        f"{symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h — will retry"
                    )
