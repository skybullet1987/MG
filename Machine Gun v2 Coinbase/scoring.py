# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - Coinbase Crypto Edition v1.0

    Adapted from the MNA Futures version for Coinbase spot crypto trading.
    Supports both Long and Short directions via separate scoring methods.

    Long Score  → buy signal  (price rising, bid dominance, VWAP above)
    Short Score → sell signal (price falling, ask dominance, VWAP below)

    Score: 0.0 – 1.0 (same gate logic as futures version)
      >= 0.50 → entry threshold
      >= 0.60 → high-conviction entry → full position size
    """

    # Tunable signal thresholds
    OBI_STRONG_THRESHOLD    = 0.35   # strong bid/ask imbalance
    OBI_PARTIAL_THRESHOLD   = 0.20   # partial imbalance
    VOL_SURGE_STRONG        = 4.0    # 4× average volume = strong ignition
    VOL_SURGE_PARTIAL       = 2.5    # 2.5× volume = moderate spike
    ADX_STRONG_THRESHOLD    = 18     # strong directional trend
    ADX_MODERATE_THRESHOLD  = 13     # moderate directional trend
    VWAP_BUFFER             = 1.0005  # 0.05% buffer for confirmed reclaim/rejection
    RSI_OVERSOLD_THRESHOLD        = 45
    RSI_MILDLY_OVERSOLD_THRESHOLD = 50
    RSI_OVERBOUGHT_THRESHOLD      = 55
    RSI_MILDLY_OVERBOUGHT_THRESHOLD = 50
    BB_NEAR_LOWER_PCT = 0.03  # within 3% of lower BB = near support
    BB_NEAR_UPPER_PCT = 0.03  # within 3% of upper BB = near resistance

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Long scoring
    # ------------------------------------------------------------------
    def calculate_long_score(self, future):
        """
        Detect long (buy) setup using microstructure signals.

        Returns
        -------
        (score, components) where score ∈ [0, 1].
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
            # Signal 1: Order Book Imbalance — bid pressure (bullish)
            # ----------------------------------------------------------
            bid_size = future.get('bid_size', 0.0)
            ask_size = future.get('ask_size', 0.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                obi = (bid_size - ask_size) / total_size
                if obi > self.OBI_STRONG_THRESHOLD:
                    components['obi'] = 0.20
                elif obi > self.OBI_PARTIAL_THRESHOLD:
                    components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition (directional volume surge)
            # ----------------------------------------------------------
            if len(future['volume']) >= 20:
                volumes = list(future['volume'])
                current_vol = volumes[-1]
                vol_long = list(future.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                adx_indicator = future.get('adx')
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
            # Signal 3: MTF Trend Alignment (bullish: price > EMA5 > EMA20)
            # ----------------------------------------------------------
            if (future['ema_5'].IsReady and future.get('ema_medium') is not None
                    and future['ema_medium'].IsReady and len(future['prices']) >= 1):
                price = future['prices'][-1]
                ema5 = future['ema_5'].Current.Value
                ema20 = future['ema_medium'].Current.Value
                if price > ema5 and ema5 > ema20:
                    components['micro_trend'] = 0.20
                elif price > ema5:
                    components['micro_trend'] = 0.10

            # Signal 3b: Steady Grind (bull regime — EMA stack rising)
            if self.algo.market_regime == "bull":
                if (future['ema_ultra_short'].IsReady and future['ema_short'].IsReady
                        and future.get('ema_medium') is not None and future['ema_medium'].IsReady
                        and len(future['prices']) >= 1):
                    price = future['prices'][-1]
                    ema_ultra = future['ema_ultra_short'].Current.Value
                    ema_short = future['ema_short'].Current.Value
                    ema_medium = future['ema_medium'].Current.Value
                    if ema_ultra > ema_short and ema_short > ema_medium:
                        if price <= ema_ultra * 1.002 and price > ema_short:
                            components['steady_grind'] = 0.25
                            components['micro_trend'] = 0

            # ----------------------------------------------------------
            # Signal 4a: ADX Trend — fires when trend is strong & bullish
            # Signal 4b: Mean Reversion — fires when choppy & oversold
            # ----------------------------------------------------------
            adx_indicator = future.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus  = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_plus > di_minus:
                    components['adx_trend'] = 0.15
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_plus > di_minus:
                    components['adx_trend'] = 0.10
                if adx_val <= self.ADX_STRONG_THRESHOLD:
                    rsi_ind = future.get('rsi')
                    bb_lower_data = future.get('bb_lower', [])
                    if (rsi_ind is not None and rsi_ind.IsReady
                            and len(bb_lower_data) >= 1 and len(future['prices']) >= 1):
                        rsi_val  = rsi_ind.Current.Value
                        price    = future['prices'][-1]
                        bb_lower = bb_lower_data[-1]
                        is_mild = (adx_val <= self.ADX_MODERATE_THRESHOLD
                                   and rsi_val < self.RSI_MILDLY_OVERSOLD_THRESHOLD)
                        if (self.algo.market_regime == 'sideways'
                                and bb_lower > 0 and price <= bb_lower * 1.005 and rsi_val < 35):
                            components['mean_reversion'] = 0.15
                        elif (adx_val <= self.ADX_MODERATE_THRESHOLD
                                and rsi_val < self.RSI_OVERSOLD_THRESHOLD
                                and bb_lower > 0
                                and price <= bb_lower * (1 + self.BB_NEAR_LOWER_PCT)):
                            components['mean_reversion'] = 0.15
                        elif is_mild:
                            components['mean_reversion'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: VWAP Reclaim / -2SD / -3SD Band Bounce
            # ----------------------------------------------------------
            vwap          = future.get('vwap', 0.0)
            vwap_sd       = future.get('vwap_sd', 0.0)
            vwap_sd2_lower = future.get('vwap_sd2_lower', 0.0)
            vwap_sd3_lower = future.get('vwap_sd3_lower', 0.0)
            if vwap > 0 and len(future['prices']) >= 1:
                price = future['prices'][-1]
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
            # Signal 6: CVD Absorption (buyers absorbing at -2SD support)
            # ----------------------------------------------------------
            cvd = future.get('cvd')
            if (vwap_sd2_lower > 0 and len(future['prices']) >= 1
                    and cvd is not None and len(cvd) >= 5):
                price = future['prices'][-1]
                if price <= vwap_sd2_lower and cvd[-1] > cvd[-5]:
                    components['cvd_absorption'] = 0.25

            # ----------------------------------------------------------
            # Signal 7: Kalman Mean Reversion (price extended below estimate)
            # ----------------------------------------------------------
            ker             = future.get('ker')
            kalman_estimate = future.get('kalman_estimate', 0.0)
            if (ker is not None and len(ker) > 0 and ker[-1] < 0.3
                    and kalman_estimate > 0 and len(future['prices']) >= 1):
                price = future['prices'][-1]
                if price < kalman_estimate * 0.996:
                    components['kalman_reversion'] = 0.20

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_long_score error: {e}")

        score = sum(components.values())
        microstructure_strength = components.get('obi', 0) + components.get('vol_ignition', 0)
        gate_cap = 0.50 + min(microstructure_strength / 0.20, 1.0) * 0.50
        score = min(score, gate_cap)
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Short scoring — exact inverse of the long logic
    # ------------------------------------------------------------------
    def calculate_short_score(self, future):
        """
        Detect short (sell) setup.

        Inverts all long signals:
          - OBI: ask-side pressure dominates (supply wall)
          - MTF Trend: bearish alignment (price < EMA5 < EMA20)
          - ADX Trend: di_minus > di_plus
          - Mean Reversion: RSI overbought + price near upper BB
          - VWAP Rejection: price below VWAP
          - CVD Distribution: CVD trending down at +2SD upper band
          - Kalman: price extended above estimate

        Returns
        -------
        (score, components) where score ∈ [0, 1].
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
            # Signal 1: OBI — ask pressure (bearish supply dominance)
            # ----------------------------------------------------------
            bid_size = future.get('bid_size', 0.0)
            ask_size = future.get('ask_size', 0.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                obi = (ask_size - bid_size) / total_size  # inverted: ask side
                if obi > self.OBI_STRONG_THRESHOLD:
                    components['obi'] = 0.20
                elif obi > self.OBI_PARTIAL_THRESHOLD:
                    components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition (same — high volume is directional)
            # ----------------------------------------------------------
            if len(future['volume']) >= 20:
                volumes = list(future['volume'])
                current_vol = volumes[-1]
                vol_long = list(future.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                adx_indicator = future.get('adx')
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
            # Signal 3: MTF Trend (bearish: price < EMA5 < EMA20)
            # ----------------------------------------------------------
            if (future['ema_5'].IsReady and future.get('ema_medium') is not None
                    and future['ema_medium'].IsReady and len(future['prices']) >= 1):
                price = future['prices'][-1]
                ema5 = future['ema_5'].Current.Value
                ema20 = future['ema_medium'].Current.Value
                if price < ema5 and ema5 < ema20:
                    components['micro_trend'] = 0.20
                elif price < ema5:
                    components['micro_trend'] = 0.10

            # Signal 3b: Steady Grind (bear regime — EMA stack falling)
            if self.algo.market_regime == "bear":
                if (future['ema_ultra_short'].IsReady and future['ema_short'].IsReady
                        and future.get('ema_medium') is not None and future['ema_medium'].IsReady
                        and len(future['prices']) >= 1):
                    price = future['prices'][-1]
                    ema_ultra  = future['ema_ultra_short'].Current.Value
                    ema_short  = future['ema_short'].Current.Value
                    ema_medium = future['ema_medium'].Current.Value
                    if ema_ultra < ema_short and ema_short < ema_medium:
                        if price >= ema_ultra * 0.998 and price < ema_short:
                            components['steady_grind'] = 0.25
                            components['micro_trend'] = 0

            # ----------------------------------------------------------
            # Signal 4a: ADX Trend (bearish: di_minus > di_plus)
            # Signal 4b: Mean Reversion (overbought near upper BB)
            # ----------------------------------------------------------
            adx_indicator = future.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val  = adx_indicator.Current.Value
                di_plus  = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_minus > di_plus:
                    components['adx_trend'] = 0.15
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_minus > di_plus:
                    components['adx_trend'] = 0.10
                if adx_val <= self.ADX_STRONG_THRESHOLD:
                    rsi_ind = future.get('rsi')
                    bb_upper_data = future.get('bb_upper', [])
                    if (rsi_ind is not None and rsi_ind.IsReady
                            and len(bb_upper_data) >= 1 and len(future['prices']) >= 1):
                        rsi_val  = rsi_ind.Current.Value
                        price    = future['prices'][-1]
                        bb_upper = bb_upper_data[-1]
                        is_mild = (adx_val <= self.ADX_MODERATE_THRESHOLD
                                   and rsi_val > self.RSI_MILDLY_OVERBOUGHT_THRESHOLD)
                        if (self.algo.market_regime == 'sideways'
                                and bb_upper > 0 and price >= bb_upper * 0.995 and rsi_val > 65):
                            components['mean_reversion'] = 0.15
                        elif (adx_val <= self.ADX_MODERATE_THRESHOLD
                                and rsi_val > self.RSI_OVERBOUGHT_THRESHOLD
                                and bb_upper > 0
                                and price >= bb_upper * (1 - self.BB_NEAR_UPPER_PCT)):
                            components['mean_reversion'] = 0.15
                        elif is_mild:
                            components['mean_reversion'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: VWAP Rejection / +2SD / +3SD Band Rejection
            # ----------------------------------------------------------
            vwap          = future.get('vwap', 0.0)
            vwap_sd       = future.get('vwap_sd', 0.0)
            vwap_sd2_upper = future.get('vwap_sd2_upper', 0.0)
            vwap_sd3_upper = future.get('vwap_sd3_upper', 0.0)
            if vwap > 0 and len(future['prices']) >= 1:
                price = future['prices'][-1]
                if price < vwap / self.VWAP_BUFFER:
                    components['vwap_signal'] = 0.20
                elif price < vwap:
                    components['vwap_signal'] = 0.10
                elif (vwap_sd > 0 and vwap_sd3_upper > 0
                      and price <= vwap_sd3_upper * 0.995
                      and price > vwap_sd2_upper):
                    components['vwap_signal'] = 0.20
                elif (vwap_sd > 0 and vwap_sd2_upper > 0
                      and price <= vwap_sd2_upper * 0.997):
                    components['vwap_signal'] = 0.15

            # ----------------------------------------------------------
            # Signal 6: CVD Distribution (sellers distributing at +2SD)
            # ----------------------------------------------------------
            cvd = future.get('cvd')
            if (vwap_sd2_upper > 0 and len(future['prices']) >= 1
                    and cvd is not None and len(cvd) >= 5):
                price = future['prices'][-1]
                if price >= vwap_sd2_upper and cvd[-1] < cvd[-5]:
                    components['cvd_absorption'] = 0.25

            # ----------------------------------------------------------
            # Signal 7: Kalman Mean Reversion Short
            # Price extended above Kalman estimate → likely to revert down
            # ----------------------------------------------------------
            ker             = future.get('ker')
            kalman_estimate = future.get('kalman_estimate', 0.0)
            if (ker is not None and len(ker) > 0 and ker[-1] < 0.3
                    and kalman_estimate > 0 and len(future['prices']) >= 1):
                price = future['prices'][-1]
                if price > kalman_estimate * 1.004:
                    components['kalman_reversion'] = 0.20

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_short_score error: {e}")

        score = sum(components.values())
        microstructure_strength = components.get('obi', 0) + components.get('vol_ignition', 0)
        gate_cap = 0.50 + min(microstructure_strength / 0.20, 1.0) * 0.50
        score = min(score, gate_cap)
        return min(score, 1.0), components

    # Keep for backward compatibility
    def calculate_scalp_score(self, future):
        return self.calculate_long_score(future)

    # ------------------------------------------------------------------
    # Position sizing — fractional quantities for crypto
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, max_size_fraction=1.0):
        """
        Returns a sizing multiplier (0.0 – 1.0) for crypto fractional orders.

        Crypto trading uses fractional quantities, so this returns a fraction
        of the target allocation rather than an integer contract count.

        High-conviction (score >= 0.80)           → max_size_fraction (full allocation)
        High-conviction (score >= high_conviction) → 75% of allocation
        Standard entry  (score >= threshold)       → 50% of allocation
        Kelly scaling: if Kelly < 0.70, reduce by an additional 50%.
        """
        if score >= 0.80:
            size_fraction = max_size_fraction
        elif score >= self.algo.high_conviction_threshold:
            size_fraction = max_size_fraction * 0.75
        else:
            size_fraction = max_size_fraction * 0.50

        kelly = self.algo._kelly_fraction()
        if kelly < 0.70:
            size_fraction *= 0.50

        return max(0.0, min(max_size_fraction, size_fraction))
