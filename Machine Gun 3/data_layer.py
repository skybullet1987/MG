# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, get_spread_pct, debug_limited,
    kelly_fraction, sync_existing_positions,
)
import numpy as np
import config as MG3Config
# endregion


class DataLayerMixin:
    """Data update and scoring helper methods for SimplifiedCryptoStrategy.

    Handles per-bar market-data ingestion, market-context classification, and
    the factor/composite-score pipeline that gates entry decisions.
    """

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
        """Evaluate signals and return scores respecting MG3 direction toggles."""
        long_score, long_components = self._scoring_engine.calculate_scalp_score(crypto)
        short_score, short_components = self._scoring_engine.calculate_short_score(crypto)

        # MG3: honour direction toggles from config
        if not self.mg3_long_enabled:
            long_score = 0.0
        if not self.mg3_short_enabled:
            short_score = 0.0

        components = long_components.copy()
        components['_scalp_score'] = long_score
        components['_direction'] = 1
        components['_long_score'] = long_score
        components['_short_score'] = short_score
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

    def _daily_loss_exceeded(self):
        """
        MG3: Returns True if today's portfolio loss exceeds MAX_DAILY_LOSS_PCT.
        Uses the daily opening value stored at ResetDailyCounters time.
        """
        if not hasattr(self, '_daily_open_value') or self._daily_open_value is None:
            return False
        current_value = self.Portfolio.TotalPortfolioValue
        if self._daily_open_value <= 0:
            return False
        daily_loss = (self._daily_open_value - current_value) / self._daily_open_value
        return daily_loss >= MG3Config.MAX_DAILY_LOSS_PCT
