# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v7.3.0 (MNQ/Futures)

    High-frequency market microstructure scalping system.
    Uses cutting-edge microstructure signals tuned for 1-minute bars on Micro NASDAQ 100 (MNQ).
    Adapted from crypto version for CME futures via Interactive Brokers.

    Score: 0.0 – 1.0 across five equal signals (0.20 each).
      >= 0.60 → entry (3/5 signals firing; 0.50 in sideways regime)
      >= 0.80 → high-conviction entry (4+ signals) → maximum position size

    Signals
    -------
    1. Tick Imbalance (replaces OBI): up-tick vs down-tick pressure over last N bars
    2. Volume Ignition: 4× volume surge (tightened from 3×)
    3. MTF Trend Alignment: EMA5 > EMA20 (short-term trend aligned with medium)
    4a. ADX Trend: ADX > 18 with bullish DI bias (max 0.15)
    4b. Mean Reversion: RSI oversold + price near lower BB when ADX is low (max 0.15)
    5. VWAP Reclaim: price above rolling 20-bar VWAP (institutional reference level)
    6. CVD Divergence: cumulative volume delta absorption at support
    7. Kalman Mean Reversion: over-extension below Kalman estimate (tightened for MNQ)
    """

    # Tunable signal thresholds (easy to adjust for backtesting)
    TICK_IMBALANCE_STRONG   = 0.60   # 60% up-ticks = strong buy pressure
    TICK_IMBALANCE_PARTIAL  = 0.40   # 40% up-ticks = partial buy pressure
    VOL_SURGE_STRONG        = 4.0    # 4× average volume = strong ignition
    VOL_SURGE_PARTIAL       = 2.5    # 2.5× volume = moderate spike
    ADX_STRONG_THRESHOLD    = 18     # strong directional trend
    ADX_MODERATE_THRESHOLD  = 13     # moderate directional trend
    VWAP_BUFFER             = 1.0005  # 0.05% above VWAP for confirmed reclaim
    # Ranging-market mean reversion thresholds (used when ADX < ADX_MODERATE_THRESHOLD)
    RSI_OVERSOLD_THRESHOLD        = 45   # RSI < 45 → oversold, mean reversion buy signal
    RSI_MILDLY_OVERSOLD_THRESHOLD = 50   # RSI < 50 → mildly oversold, partial credit
    BB_NEAR_LOWER_PCT             = 0.03  # within 3% of lower Bollinger Band = near support

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (score, components_dict)
    # ------------------------------------------------------------------
    def calculate_scalp_score(self, mnq):
        """
        Calculate the aggregate scalp score using seven microstructure signals.

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components maps each
        signal name to its individual contribution (0.20 max each).
        """
        components = {
            'obi':            0.0,
            'vol_ignition':   0.0,
            'micro_trend':    0.0,
            'adx_trend':      0.0,
            'mean_reversion': 0.0,
            'vwap_signal':    0.0,
        }

        try:
            # ----------------------------------------------------------
            # Signal 1: Tick Imbalance (replaces OBI for futures)
            # Count up-ticks vs down-ticks over last 10 bars.
            # MNQ doesn't have reliable bid/ask sizes in TradeBar data,
            # so we approximate order flow imbalance via return signs.
            # ----------------------------------------------------------
            returns = mnq.get('returns', [])
            if len(returns) >= 10:
                recent_returns = list(returns)[-10:]
                up_ticks = sum(1 for r in recent_returns if r > 0)
                down_ticks = sum(1 for r in recent_returns if r <= 0)
                total_ticks = up_ticks + down_ticks
                if total_ticks > 0:
                    tick_imbalance = (up_ticks - down_ticks) / total_ticks
                    if tick_imbalance > self.TICK_IMBALANCE_STRONG:
                        components['obi'] = 0.20
                    elif tick_imbalance > self.TICK_IMBALANCE_PARTIAL:
                        # Partial credit for meaningful buy-side imbalance
                        components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition
            # Current volume surge vs adaptive rolling baseline.
            # Uses 24h long-term average when available instead of a fixed
            # 20-bar (20-minute) window, so thresholds stay relevant during
            # low-volatility periods.
            # ----------------------------------------------------------
            if len(mnq['volume']) >= 20:
                volumes = list(mnq['volume'])
                current_vol = volumes[-1]
                # Adaptive baseline: prefer long-term rolling average (up to 24h)
                vol_long = list(mnq.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                # ADX regime filter: lower thresholds in choppy markets (ADX < 25)
                adx_indicator = mnq.get('adx')
                is_choppy = (adx_indicator is not None and adx_indicator.IsReady
                             and adx_indicator.Current.Value < 25)
                vol_strong  = 1.8 if is_choppy else self.VOL_SURGE_STRONG
                vol_partial = 1.2 if is_choppy else self.VOL_SURGE_PARTIAL
                if vol_baseline > 0:
                    ratio = current_vol / vol_baseline
                    if ratio >= vol_strong:
                        components['vol_ignition'] = 0.20
                    elif ratio >= vol_partial:
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: MTF Trend Alignment
            # Price > EMA5 AND EMA5 > EMA20 → short-term and medium-term
            # trends are aligned.
            # ----------------------------------------------------------
            if (mnq['ema_5'].IsReady and mnq.get('ema_medium') is not None
                    and mnq['ema_medium'].IsReady and len(mnq['prices']) >= 1):
                price = mnq['prices'][-1]
                ema5 = mnq['ema_5'].Current.Value
                ema20 = mnq['ema_medium'].Current.Value
                if price > ema5 and ema5 > ema20:
                    components['micro_trend'] = 0.20
                elif price > ema5:
                    components['micro_trend'] = 0.10

            # ----------------------------------------------------------
            # Signal 3b: Steady Grind (Bull Market Only)
            # ----------------------------------------------------------
            if self.algo.market_regime == "bull":
                if (mnq['ema_ultra_short'].IsReady and mnq['ema_short'].IsReady
                        and mnq.get('ema_medium') is not None and mnq['ema_medium'].IsReady
                        and len(mnq['prices']) >= 1):
                    price = mnq['prices'][-1]
                    ema_ultra = mnq['ema_ultra_short'].Current.Value
                    ema_short = mnq['ema_short'].Current.Value
                    ema_medium = mnq['ema_medium'].Current.Value
                    if ema_ultra > ema_short and ema_short > ema_medium:
                        if price <= ema_ultra * 1.002 and price > ema_short:
                            components['steady_grind'] = 0.25
                            components['micro_trend'] = 0

            # ----------------------------------------------------------
            # Signal 4a: ADX Trend — scores only when ADX is HIGH
            # Signal 4b: Mean Reversion — scores only when ADX is LOW
            # ----------------------------------------------------------
            adx_indicator = mnq.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_plus > di_minus:
                    components['adx_trend'] = 0.15
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_plus > di_minus:
                    components['adx_trend'] = 0.10
                if adx_val <= self.ADX_STRONG_THRESHOLD:
                    rsi_ind = mnq.get('rsi')
                    bb_lower_data = mnq.get('bb_lower', [])
                    if (rsi_ind is not None and rsi_ind.IsReady
                            and len(bb_lower_data) >= 1 and len(mnq['prices']) >= 1):
                        rsi_val = rsi_ind.Current.Value
                        price = mnq['prices'][-1]
                        bb_lower = bb_lower_data[-1]
                        is_mild_oversold_ranging = (adx_val <= self.ADX_MODERATE_THRESHOLD
                                                    and rsi_val < self.RSI_MILDLY_OVERSOLD_THRESHOLD)
                        if (self.algo.market_regime == 'sideways'
                                and bb_lower > 0 and price <= bb_lower * 1.005 and rsi_val < 35):
                            components['mean_reversion'] = 0.15
                        elif (adx_val <= self.ADX_MODERATE_THRESHOLD
                                and rsi_val < self.RSI_OVERSOLD_THRESHOLD
                                and bb_lower > 0
                                and price <= bb_lower * (1 + self.BB_NEAR_LOWER_PCT)):
                            components['mean_reversion'] = 0.15
                        elif is_mild_oversold_ranging:
                            components['mean_reversion'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: VWAP Reclaim / SD Band Bounce
            # VWAP is THE key institutional level on NQ futures.
            # ----------------------------------------------------------
            vwap = mnq.get('vwap', 0.0)
            vwap_sd = mnq.get('vwap_sd', 0.0)
            vwap_sd2_lower = mnq.get('vwap_sd2_lower', 0.0)
            vwap_sd3_lower = mnq.get('vwap_sd3_lower', 0.0)
            if vwap > 0 and len(mnq['prices']) >= 1:
                price = mnq['prices'][-1]
                if price > vwap * self.VWAP_BUFFER:
                    components['vwap_signal'] = 0.20
                elif price > vwap:
                    components['vwap_signal'] = 0.10
                elif (vwap_sd > 0 and vwap_sd3_lower > 0
                      and price >= vwap_sd3_lower * 1.005
                      and price < vwap_sd2_lower):
                    components['vwap_signal'] = 0.20
                elif (vwap_sd > 0 and vwap_sd2_lower > 0
                      and price >= vwap_sd2_lower * 1.003):
                    components['vwap_signal'] = 0.15

            # ----------------------------------------------------------
            # Signal 6: CVD Divergence (Absorption)
            # Approximated from TradeBar: bar_delta = volume * ((close-low) - (high-close)) / (high-low)
            # This is a valid proxy for futures bars.
            # ----------------------------------------------------------
            cvd = mnq.get('cvd')
            if (vwap_sd2_lower > 0 and len(mnq['prices']) >= 1
                    and cvd is not None and len(cvd) >= 5):
                price = mnq['prices'][-1]
                if price <= vwap_sd2_lower and cvd[-1] > cvd[-5]:
                    components['cvd_absorption'] = 0.25

            # ----------------------------------------------------------
            # Signal 7: Kalman Mean Reversion
            # Tightened threshold for MNQ: 0.15% extension (was 0.4% for crypto).
            # MNQ intraday ranges are much tighter than crypto.
            # ----------------------------------------------------------
            ker = mnq.get('ker')
            kalman_estimate = mnq.get('kalman_estimate', 0.0)
            if (ker is not None and len(ker) > 0 and ker[-1] < 0.3
                    and kalman_estimate > 0 and len(mnq['prices']) >= 1):
                price = mnq['prices'][-1]
                if price < kalman_estimate * 0.9985:  # 0.15% extension (was 0.4%)
                    components['kalman_reversion'] = 0.20

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())

        # Graduated microstructure gate: smoothly raises the score ceiling
        # based on real order-flow presence (tick imbalance + vol_ignition strength).
        microstructure_strength = components.get('obi', 0) + components.get('vol_ignition', 0)
        gate_cap = 0.50 + min(microstructure_strength / 0.20, 1.0) * 0.50
        score = min(score, gate_cap)

        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Short scoring — inverse of long signals
    # ------------------------------------------------------------------
    def calculate_short_score(self, mnq):
        """
        Calculate the aggregate short scalp score (inverse of long signals).

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components maps each
        signal name to its individual contribution (0.20 max each).
        """
        components = {
            'obi':            0.0,
            'vol_ignition':   0.0,
            'micro_trend':    0.0,
            'adx_trend':      0.0,
            'mean_reversion': 0.0,
            'vwap_signal':    0.0,
        }

        try:
            # ----------------------------------------------------------
            # Signal 1: Tick Imbalance (short = down-tick pressure)
            # ----------------------------------------------------------
            returns = mnq.get('returns', [])
            if len(returns) >= 10:
                recent_returns = list(returns)[-10:]
                up_ticks = sum(1 for r in recent_returns if r > 0)
                down_ticks = sum(1 for r in recent_returns if r <= 0)
                total_ticks = up_ticks + down_ticks
                if total_ticks > 0:
                    tick_imbalance = (down_ticks - up_ticks) / total_ticks
                    if tick_imbalance > self.TICK_IMBALANCE_STRONG:
                        components['obi'] = 0.20
                    elif tick_imbalance > self.TICK_IMBALANCE_PARTIAL:
                        components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition on down move
            # ----------------------------------------------------------
            if len(mnq['volume']) >= 20:
                volumes = list(mnq['volume'])
                current_vol = volumes[-1]
                vol_long = list(mnq.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                adx_indicator = mnq.get('adx')
                is_choppy = (adx_indicator is not None and adx_indicator.IsReady
                             and adx_indicator.Current.Value < 25)
                vol_strong  = 1.8 if is_choppy else self.VOL_SURGE_STRONG
                vol_partial = 1.2 if is_choppy else self.VOL_SURGE_PARTIAL
                last_return = list(returns)[-1] if len(returns) >= 1 else 0
                if vol_baseline > 0 and last_return < 0:
                    ratio = current_vol / vol_baseline
                    if ratio >= vol_strong:
                        components['vol_ignition'] = 0.20
                    elif ratio >= vol_partial:
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: MTF Trend Alignment (short = price < EMA5 < EMA20)
            # ----------------------------------------------------------
            if (mnq['ema_5'].IsReady and mnq.get('ema_medium') is not None
                    and mnq['ema_medium'].IsReady and len(mnq['prices']) >= 1):
                price = mnq['prices'][-1]
                ema5 = mnq['ema_5'].Current.Value
                ema20 = mnq['ema_medium'].Current.Value
                if price < ema5 and ema5 < ema20:
                    components['micro_trend'] = 0.20
                elif price < ema5:
                    components['micro_trend'] = 0.10

            # ----------------------------------------------------------
            # Signal 4a: ADX Trend (short = ADX high + DI- > DI+)
            # Signal 4b: Mean Reversion (short = RSI overbought + near upper BB)
            # ----------------------------------------------------------
            adx_indicator = mnq.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_minus > di_plus:
                    components['adx_trend'] = 0.15
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_minus > di_plus:
                    components['adx_trend'] = 0.10
                if adx_val <= self.ADX_STRONG_THRESHOLD:
                    rsi_ind = mnq.get('rsi')
                    bb_upper_data = mnq.get('bb_upper', [])
                    if (rsi_ind is not None and rsi_ind.IsReady
                            and len(bb_upper_data) >= 1 and len(mnq['prices']) >= 1):
                        rsi_val = rsi_ind.Current.Value
                        price = mnq['prices'][-1]
                        bb_upper = bb_upper_data[-1]
                        RSI_OVERBOUGHT = 55
                        RSI_MILDLY_OVERBOUGHT = 50
                        is_mild_overbought = (adx_val <= self.ADX_MODERATE_THRESHOLD
                                              and rsi_val > RSI_MILDLY_OVERBOUGHT)
                        if (self.algo.market_regime == 'sideways'
                                and bb_upper > 0 and price >= bb_upper * 0.995 and rsi_val > 65):
                            components['mean_reversion'] = 0.15
                        elif (adx_val <= self.ADX_MODERATE_THRESHOLD
                                and rsi_val > RSI_OVERBOUGHT
                                and bb_upper > 0
                                and price >= bb_upper * (1 - self.BB_NEAR_LOWER_PCT)):
                            components['mean_reversion'] = 0.15
                        elif is_mild_overbought:
                            components['mean_reversion'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: VWAP Signal (short = price below VWAP)
            # ----------------------------------------------------------
            vwap = mnq.get('vwap', 0.0)
            if vwap > 0 and len(mnq['prices']) >= 1:
                price = mnq['prices'][-1]
                if price < vwap / self.VWAP_BUFFER:
                    components['vwap_signal'] = 0.20
                elif price < vwap:
                    components['vwap_signal'] = 0.10

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_short_score error: {e}")

        score = sum(components.values())

        microstructure_strength = components.get('obi', 0) + components.get('vol_ignition', 0)
        gate_cap = 0.50 + min(microstructure_strength / 0.20, 1.0) * 0.50
        score = min(score, gate_cap)

        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing — contract-count based for MNQ futures
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Position sizing for MNQ futures — returns number of contracts.

        MNQ initial margin ~$1,300 per contract. With $3,000 capital, max 2 contracts.
        Fees are flat ~$1.26 RT (negligible vs crypto's 0.65%), so sizing focuses
        on margin utilization and vol-targeting rather than fee survival.
        """
        available_margin = self.algo.Portfolio.MarginRemaining
        margin_per_contract = 1400  # MNQ initial margin ~$1,300 + buffer

        max_by_margin = int(available_margin / margin_per_contract) if margin_per_contract > 0 else 0
        max_contracts = getattr(self.algo, 'max_contracts', 2)

        if score >= 0.80:
            contracts = min(2, max_by_margin, max_contracts)
        elif score >= self.algo.high_conviction_threshold:
            contracts = min(1, max_by_margin, max_contracts)
        elif score >= threshold:
            contracts = min(1, max_by_margin, max_contracts)
        else:
            contracts = 0

        # Vol-targeting: reduce in high-vol environments
        if asset_vol_ann is not None and asset_vol_ann > 0:
            target_vol = getattr(self.algo, 'target_position_ann_vol', 0.35)
            vol_scalar = min(1.0, target_vol / asset_vol_ann)
            if vol_scalar < 0.5:
                contracts = max(0, contracts - 1)

        # Kelly adjustment
        kelly = self.algo._kelly_fraction() if hasattr(self.algo, '_kelly_fraction') else 1.0
        if kelly < 0.6:
            contracts = max(0, contracts - 1)

        return max(0, contracts)
