# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v7.3.0

    High-frequency market microstructure scalping system.
    Uses cutting-edge microstructure signals tuned for 1-minute bars on Kraken.
    Adapts to both trending (Jan–Mar) and ranging/sideways (Apr–Oct) regimes.

    Score: 0.0 – 1.0 across five equal signals (0.20 each).
      >= 0.60 → entry (3/5 signals firing; 0.50 in sideways regime)
      >= 0.80 → high-conviction entry (4+ signals) → maximum position size

    Signals
    -------
    1. Order Book Imbalance (OBI): bid/ask pressure (tightened threshold)
    2. Volume Ignition: 4× volume surge (tightened from 3×)
    3. MTF Trend Alignment: EMA5 > EMA20 (short-term trend aligned with medium)
    4a. ADX Trend: ADX > 18 with bullish DI bias (max 0.15)
    4b. Mean Reversion: RSI oversold + price near lower BB when ADX is low (max 0.15)
    5. VWAP Reclaim: price above rolling 20-bar VWAP (institutional reference level)
    """

    # Tunable signal thresholds (easy to adjust for backtesting)
    OBI_STRONG_THRESHOLD    = 0.35   # strong bid-side imbalance
    OBI_PARTIAL_THRESHOLD   = 0.20   # partial bid-side imbalance
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
    def calculate_scalp_score(self, crypto):
        """
        Calculate the aggregate scalp score using five microstructure signals.

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
            # Signal 1: Order Book Imbalance (OBI)
            # OBI = (bid_size - ask_size) / (bid_size + ask_size)
            # Strong buy pressure when OBI > 0.6 (bid wall dominates).
            # Tightened from 0.5 → 0.6 to reduce false signals.
            # ----------------------------------------------------------
            bid_size = crypto.get('bid_size', 0.0)
            ask_size = crypto.get('ask_size', 0.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                obi = (bid_size - ask_size) / total_size
                if obi > self.OBI_STRONG_THRESHOLD:
                    components['obi'] = 0.20
                elif obi > self.OBI_PARTIAL_THRESHOLD:
                    # Partial credit for meaningful buy-side imbalance
                    components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition
            # Current volume surge vs adaptive rolling baseline (Apr-Oct fix).
            # Uses 24h long-term average when available instead of a fixed
            # 20-bar (20-minute) window, so thresholds stay relevant during
            # low-volatility summer periods.
            # ----------------------------------------------------------
            if len(crypto['volume']) >= 20:
                volumes = list(crypto['volume'])
                current_vol = volumes[-1]
                # Adaptive baseline: prefer long-term rolling average (up to 24h)
                vol_long = list(crypto.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                # ADX regime filter: lower thresholds in choppy markets (ADX < 25)
                # to fire more trades when trend continuation is unlikely.
                adx_indicator = crypto.get('adx')
                is_choppy = (adx_indicator is not None and adx_indicator.IsReady
                             and adx_indicator.Current.Value < 25)
                vol_strong  = 1.8 if is_choppy else self.VOL_SURGE_STRONG
                vol_partial = 1.2 if is_choppy else self.VOL_SURGE_PARTIAL
                if vol_baseline > 0:
                    ratio = current_vol / vol_baseline
                    if ratio >= vol_strong:
                        components['vol_ignition'] = 0.20
                    elif ratio >= vol_partial:
                        # Partial credit for a meaningful volume spike
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: MTF Trend Alignment
            # Price > EMA5 AND EMA5 > EMA20 → short-term and medium-term
            # trends are aligned (simulates 5m/20m multi-timeframe check).
            # ----------------------------------------------------------
            if (crypto['ema_5'].IsReady and crypto.get('ema_medium') is not None
                    and crypto['ema_medium'].IsReady and len(crypto['prices']) >= 1):
                price = crypto['prices'][-1]
                ema5 = crypto['ema_5'].Current.Value
                ema20 = crypto['ema_medium'].Current.Value
                if price > ema5 and ema5 > ema20:
                    # Full credit: short-term and medium-term trends aligned
                    components['micro_trend'] = 0.20
                elif price > ema5:
                    # Partial credit: price above immediate EMA only
                    components['micro_trend'] = 0.10

            # ----------------------------------------------------------
            # Signal 3b: Steady Grind (Bull Market Only)
            # Rewards a slow, steady EMA stack even when ADX is weak.
            # Price pulled back to the ultra-short EMA but trend is intact.
            # ----------------------------------------------------------
            if self.algo.market_regime == "bull":
                if (crypto['ema_ultra_short'].IsReady and crypto['ema_short'].IsReady
                        and crypto.get('ema_medium') is not None and crypto['ema_medium'].IsReady
                        and len(crypto['prices']) >= 1):
                    price = crypto['prices'][-1]
                    ema_ultra = crypto['ema_ultra_short'].Current.Value
                    ema_short = crypto['ema_short'].Current.Value
                    ema_medium = crypto['ema_medium'].Current.Value
                    if ema_ultra > ema_short and ema_short > ema_medium:
                        if price <= ema_ultra * 1.002 and price > ema_short:
                            components['steady_grind'] = 0.25
                            # steady_grind acts as an upgrade/replacement for micro_trend:
                            # it is a stronger, more specific bull signal, so we zero out
                            # micro_trend to prevent the two signals from stacking additively.
                            components['micro_trend'] = 0

            # ----------------------------------------------------------
            # Signal 4a: ADX Trend — scores only when ADX is HIGH
            # Signal 4b: Mean Reversion — scores only when ADX is LOW
            # Decoupled so they never overwrite each other and can
            # add up independently into the final score.
            # ----------------------------------------------------------
            adx_indicator = crypto.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                # adx_trend: only fires when trend is strong and bullish
                if adx_val > self.ADX_STRONG_THRESHOLD and di_plus > di_minus:
                    components['adx_trend'] = 0.15
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_plus > di_minus:
                    components['adx_trend'] = 0.10
                # mean_reversion: only fires when ADX is low (ranging/choppy)
                if adx_val <= self.ADX_STRONG_THRESHOLD:
                    rsi_ind = crypto.get('rsi')
                    bb_lower_data = crypto.get('bb_lower', [])
                    if (rsi_ind is not None and rsi_ind.IsReady
                            and len(bb_lower_data) >= 1 and len(crypto['prices']) >= 1):
                        rsi_val = rsi_ind.Current.Value
                        price = crypto['prices'][-1]
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
            # Price > rolling 20-bar VWAP → institutional buying support.
            # Price bouncing aggressively off -2 or -3 SD lower band →
            # extreme mean-reversion entry signal (higher score than bare
            # VWAP reclaim in some cases).
            # ----------------------------------------------------------
            vwap = crypto.get('vwap', 0.0)
            vwap_sd = crypto.get('vwap_sd', 0.0)
            vwap_sd2_lower = crypto.get('vwap_sd2_lower', 0.0)
            vwap_sd3_lower = crypto.get('vwap_sd3_lower', 0.0)
            if vwap > 0 and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                if price > vwap * self.VWAP_BUFFER:
                    # Price clearly above VWAP (0.1% buffer)
                    components['vwap_signal'] = 0.20
                elif price > vwap:
                    # Price marginally above VWAP
                    components['vwap_signal'] = 0.10
                elif (vwap_sd > 0 and vwap_sd3_lower > 0
                      and price >= vwap_sd3_lower * 1.005
                      and price < vwap_sd2_lower):
                    # Aggressive bounce off -3 SD lower band (extreme support)
                    components['vwap_signal'] = 0.20
                elif (vwap_sd > 0 and vwap_sd2_lower > 0
                      and price >= vwap_sd2_lower * 1.003):
                    # Bounce off -2 SD lower band (strong support)
                    components['vwap_signal'] = 0.15

            # ----------------------------------------------------------
            # Signal 6: CVD Divergence (Absorption)
            # Price at or below VWAP -2SD lower band AND CVD trending up
            # over last 5 bars → limit buyers absorbing sellers at support.
            # ----------------------------------------------------------
            cvd = crypto.get('cvd')
            if (vwap_sd2_lower > 0 and len(crypto['prices']) >= 1
                    and cvd is not None and len(cvd) >= 5):
                price = crypto['prices'][-1]
                if price <= vwap_sd2_lower and cvd[-1] > cvd[-5]:
                    components['cvd_absorption'] = 0.25

            # ----------------------------------------------------------
            # Signal 7: Kalman Mean Reversion
            # In choppy markets (KER < 0.3), price extending >0.4% below
            # Kalman estimate → over-extension likely to revert.
            # ----------------------------------------------------------
            ker = crypto.get('ker')
            kalman_estimate = crypto.get('kalman_estimate', 0.0)
            if (ker is not None and len(ker) > 0 and ker[-1] < 0.3
                    and kalman_estimate > 0 and len(crypto['prices']) >= 1):
                price = crypto['prices'][-1]
                if price < kalman_estimate * 0.996:
                    components['kalman_reversion'] = 0.20

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())

        # Graduated microstructure gate: smoothly raises the score ceiling
        # based on real order-flow presence (OBI + vol_ignition strength).
        # No microstructure → cap at 0.50; full microstructure → uncapped.
        microstructure_strength = components.get('obi', 0) + components.get('vol_ignition', 0)
        gate_cap = 0.50 + min(microstructure_strength / 0.20, 1.0) * 0.50
        score = min(score, gate_cap)

        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Position sizing calibrated for fee survival with vol-targeting.

        At 0.65% round-trip fees, positions must be large enough for the
        TP target to cover fees, but small enough that stop losses don't
        cascade into drawdown -> circuit breaker -> passivity.

        Target: max 2% account risk per trade.
        Returns 25-50% of available capital depending on conviction,
        scaled by asset volatility.
        """
        if score >= 0.80:
            # 4+ signals firing – high conviction
            size = 0.50
        elif score >= self.algo.high_conviction_threshold:
            # 3+ signals: good conviction
            size = 0.40
        elif score >= threshold:
            # Entry threshold met: moderate sizing
            size = 0.35
        else:
            size = 0.25

        # Vol-targeting: scale down for volatile assets, keep size for calmer ones
        if asset_vol_ann is not None and asset_vol_ann > 0:
            target_vol = self.algo.target_position_ann_vol  # 0.35
            vol_scalar = min(target_vol / asset_vol_ann, 1.0)
            size *= max(vol_scalar, 0.5)  # Never reduce below 50% of base size

        kelly = self.algo._kelly_fraction()
        return size * kelly
