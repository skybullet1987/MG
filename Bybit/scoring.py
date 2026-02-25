# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v7.1.0 (Bybit)

    High-frequency market microstructure scalping system.
    Supports both Long and Short signals tuned for 1-minute bars on Bybit.
    Adapts to both trending (Jan–Mar) and ranging/sideways (Apr–Oct) regimes.

    Long Score: 0.0 – 1.0 across five equal signals (0.20 each).
      >= 0.60 → entry (3/5 signals firing; 0.50 in sideways regime)
      >= 0.80 → high-conviction entry (4+ signals) → maximum position size

    Short Score: mirrors Long but looks for bearish microstructure signals.

    Long Signals
    ------------
    1. Order Book Imbalance (OBI): bid-side pressure
    2. Volume Ignition: 3× volume surge on up candles
    3. MTF Trend Alignment: EMA5 > EMA20 (uptrend)
    4. ADX / Mean Reversion: ADX > 20 bullish; or RSI oversold near lower BB
    5. VWAP Reclaim: price above rolling 20-bar VWAP

    Short Signals
    -------------
    1. OBI: ask-side pressure (obi < -threshold)
    2. Volume Ignition: 3× volume surge on down candles
    3. MTF Trend: EMA5 < EMA20 (downtrend)
    4. ADX: ADX > 20 bearish (DI- > DI+); or RSI overbought near upper BB
    5. VWAP: price below rolling 20-bar VWAP
    """

    # Tunable signal thresholds (easy to adjust for backtesting)
    OBI_STRONG_THRESHOLD    = 0.50   # strong bid-side imbalance
    OBI_PARTIAL_THRESHOLD   = 0.25   # partial bid-side imbalance
    VOL_SURGE_STRONG        = 3.0    # 3× average volume = strong ignition
    VOL_SURGE_PARTIAL       = 2.0    # 2× volume = moderate spike
    ADX_STRONG_THRESHOLD    = 20     # strong directional trend
    ADX_MODERATE_THRESHOLD  = 15     # moderate directional trend
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
            'adx_filter':     0.0,
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
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"OBI Signal: obi={obi:.3f} bid={bid_size:.2f} ask={ask_size:.2f}")
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
                if vol_baseline > 0:
                    ratio = current_vol / vol_baseline
                    if ratio >= self.VOL_SURGE_STRONG:
                        components['vol_ignition'] = 0.20
                        if not self.algo.IsWarmingUp:
                            self.algo.Debug(
                                f"Volume Ignition: vol={current_vol:.2f} baseline={vol_baseline:.2f} ratio={ratio:.1f}x")
                    elif ratio >= self.VOL_SURGE_PARTIAL:
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

            # ----------------------------------------------------------
            # Signal 4: ADX Regime Filter OR Mean Reversion
            # Trending market (ADX > 20): ADX directional bias confirms trend.
            # Ranging market (ADX ≤ 20): Mean reversion setup (RSI oversold +
            # price near lower Bollinger Band) serves as the entry signal.
            # This adaptation allows the engine to trade profitably in both
            # trending (Jan–Mar) and sideways/consolidating (Apr–Oct) regimes.
            # ----------------------------------------------------------
            adx_indicator = crypto.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_plus > di_minus:
                    # Strong trend with bullish bias
                    components['adx_filter'] = 0.20
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_plus > di_minus:
                    # Moderate trend with bullish bias
                    components['adx_filter'] = 0.10
                elif adx_val < self.ADX_STRONG_THRESHOLD and self.algo.market_regime == 'sideways':
                    # Deep mean-reversion signal for sideways regime
                    rsi_indicator = crypto.get('rsi')
                    bb_lower_data = crypto.get('bb_lower', [])
                    if (rsi_indicator is not None and rsi_indicator.IsReady
                            and len(bb_lower_data) >= 1 and len(crypto['prices']) >= 1):
                        rsi_val = rsi_indicator.Current.Value
                        price = crypto['prices'][-1]
                        lower_bb = bb_lower_data[-1]
                        if lower_bb > 0 and price <= lower_bb * 1.005 and rsi_val < 35:
                            components['adx_filter'] = 0.25
                            if not self.algo.IsWarmingUp:
                                self.algo.Debug(
                                    f"Deep Mean Reversion (Sideways): rsi={rsi_val:.1f} "
                                    f"price={price:.4f} bb_lower={lower_bb:.4f}")
                elif adx_val <= self.ADX_MODERATE_THRESHOLD:
                    # Ranging market: use mean reversion signal instead of ADX
                    if (crypto['rsi'].IsReady and len(crypto['bb_lower']) >= 1
                            and len(crypto['prices']) >= 1):
                        rsi_val = crypto['rsi'].Current.Value
                        price = crypto['prices'][-1]
                        bb_lower = crypto['bb_lower'][-1]
                        if (rsi_val < self.RSI_OVERSOLD_THRESHOLD and bb_lower > 0
                                and price <= bb_lower * (1 + self.BB_NEAR_LOWER_PCT)):
                            # Oversold near lower band → strong mean reversion signal
                            components['adx_filter'] = 0.20
                            if not self.algo.IsWarmingUp:
                                self.algo.Debug(
                                    f"Mean Reversion Signal: rsi={rsi_val:.1f} "
                                    f"price={price:.4f} bb_lower={bb_lower:.4f}")
                        elif rsi_val < self.RSI_MILDLY_OVERSOLD_THRESHOLD:
                            # Mildly oversold in ranging market → partial credit
                            components['adx_filter'] = 0.10

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
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"VWAP -3SD Bounce: price={price:.4f} sd3_lower={vwap_sd3_lower:.4f}")
                elif (vwap_sd > 0 and vwap_sd2_lower > 0
                      and price >= vwap_sd2_lower * 1.003):
                    # Bounce off -2 SD lower band (strong support)
                    components['vwap_signal'] = 0.15
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"VWAP -2SD Bounce: price={price:.4f} sd2_lower={vwap_sd2_lower:.4f}")

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
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Short scoring: bearish microstructure signals (mirror of long)
    # ------------------------------------------------------------------
    # Short-specific thresholds
    OBI_SHORT_STRONG_THRESHOLD   = 0.75    # ask_size / total_size > 0.75 → strong ask pressure
    OBI_SHORT_PARTIAL_THRESHOLD  = 0.625   # ask_size / total_size > 0.625 → partial ask pressure
    RSI_OVERBOUGHT_THRESHOLD     = 55      # RSI > 55 → overbought, mean reversion sell signal
    RSI_MILDLY_OVERBOUGHT_THRESHOLD = 50  # RSI > 50 → mildly overbought, partial credit
    BB_NEAR_UPPER_PCT            = 0.03   # within 3% of upper Bollinger Band = near resistance

    def calculate_short_score(self, crypto):
        """
        Calculate the aggregate short scalp score using five bearish microstructure signals.
        Mirrors calculate_scalp_score but looks for the *opposite* conditions.

        Returns
        -------
        (score, components) where score ∈ [0, 1] and components maps each
        signal name to its individual contribution (0.20 max each).
        """
        components = {
            'obi':            0.0,
            'vol_ignition':   0.0,
            'micro_trend':    0.0,
            'adx_filter':     0.0,
            'vwap_signal':    0.0,
        }

        try:
            # ----------------------------------------------------------
            # Signal 1: Order Book Imbalance (OBI) – Ask-side pressure
            # OBI = ask_size / total_size (heavy ask pressure → bearish)
            # Strong sell pressure when ask_size / total_size > 0.75.
            # ----------------------------------------------------------
            bid_size = crypto.get('bid_size', 0.0)
            ask_size = crypto.get('ask_size', 0.0)
            total_size = bid_size + ask_size
            if total_size > 0:
                obi = ask_size / total_size
                if obi > self.OBI_SHORT_STRONG_THRESHOLD:
                    components['obi'] = 0.20
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"Short OBI Signal: ask_ratio={obi:.3f} bid={bid_size:.2f} ask={ask_size:.2f}")
                elif obi > self.OBI_SHORT_PARTIAL_THRESHOLD:
                    components['obi'] = 0.10

            # ----------------------------------------------------------
            # Signal 2: Volume Ignition on down candles
            # Current volume surge vs adaptive rolling baseline on a bar
            # where price closed below its open (Red/Down candle: close < open).
            # ----------------------------------------------------------
            if len(crypto['volume']) >= 20 and len(crypto['prices']) >= 1 and len(crypto.get('opens', [])) >= 1:
                volumes = list(crypto['volume'])
                current_vol = volumes[-1]
                prices = list(crypto['prices'])
                opens = list(crypto['opens'])
                # Only count ignition on a red/down candle (close < open)
                is_down_bar = prices[-1] < opens[-1]
                vol_long = list(crypto.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                if vol_baseline > 0 and is_down_bar:
                    ratio = current_vol / vol_baseline
                    if ratio >= self.VOL_SURGE_STRONG:
                        components['vol_ignition'] = 0.20
                        if not self.algo.IsWarmingUp:
                            self.algo.Debug(
                                f"Short Volume Ignition: vol={current_vol:.2f} baseline={vol_baseline:.2f} ratio={ratio:.1f}x")
                    elif ratio >= self.VOL_SURGE_PARTIAL:
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # Signal 3: MTF Trend Alignment (bearish)
            # EMA5 < EMA20 → short-term EMA below medium-term EMA (downtrend).
            # ----------------------------------------------------------
            if (crypto['ema_5'].IsReady and crypto.get('ema_medium') is not None
                    and crypto['ema_medium'].IsReady and len(crypto['prices']) >= 1):
                price = crypto['prices'][-1]
                ema5 = crypto['ema_5'].Current.Value
                ema20 = crypto['ema_medium'].Current.Value
                if ema5 < ema20:
                    components['micro_trend'] = 0.20
                elif price < ema5:
                    components['micro_trend'] = 0.10

            # ----------------------------------------------------------
            # Signal 4: ADX Regime Filter OR Mean Reversion (bearish)
            # Trending: ADX > 20 with DI- > DI+ (bearish momentum).
            # Ranging: RSI overbought + price near upper Bollinger Band.
            # ----------------------------------------------------------
            adx_indicator = crypto.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > self.ADX_STRONG_THRESHOLD and di_minus > di_plus:
                    # Strong bearish trend
                    components['adx_filter'] = 0.20
                elif adx_val > self.ADX_MODERATE_THRESHOLD and di_minus > di_plus:
                    components['adx_filter'] = 0.10
                elif adx_val < self.ADX_STRONG_THRESHOLD and self.algo.market_regime == 'sideways':
                    # Deep mean-reversion (overbought) for sideways regime
                    rsi_indicator = crypto.get('rsi')
                    bb_upper_data = crypto.get('bb_upper', [])
                    if (rsi_indicator is not None and rsi_indicator.IsReady
                            and len(bb_upper_data) >= 1 and len(crypto['prices']) >= 1):
                        rsi_val = rsi_indicator.Current.Value
                        price = crypto['prices'][-1]
                        upper_bb = bb_upper_data[-1]
                        if upper_bb > 0 and price >= upper_bb * 0.995 and rsi_val > 65:
                            components['adx_filter'] = 0.25
                            if not self.algo.IsWarmingUp:
                                self.algo.Debug(
                                    f"Deep Mean Reversion Short (Sideways): rsi={rsi_val:.1f} "
                                    f"price={price:.4f} bb_upper={upper_bb:.4f}")
                elif adx_val <= self.ADX_MODERATE_THRESHOLD:
                    # Ranging market: RSI overbought near upper band
                    if (crypto['rsi'].IsReady and len(crypto['bb_upper']) >= 1
                            and len(crypto['prices']) >= 1):
                        rsi_val = crypto['rsi'].Current.Value
                        price = crypto['prices'][-1]
                        bb_upper = crypto['bb_upper'][-1]
                        if (rsi_val > self.RSI_OVERBOUGHT_THRESHOLD and bb_upper > 0
                                and price >= bb_upper * (1 - self.BB_NEAR_UPPER_PCT)):
                            components['adx_filter'] = 0.20
                            if not self.algo.IsWarmingUp:
                                self.algo.Debug(
                                    f"Short Mean Reversion Signal: rsi={rsi_val:.1f} "
                                    f"price={price:.4f} bb_upper={bb_upper:.4f}")
                        elif rsi_val > self.RSI_MILDLY_OVERBOUGHT_THRESHOLD:
                            components['adx_filter'] = 0.10

            # ----------------------------------------------------------
            # Signal 5: VWAP Below / SD Band Rejection
            # Price < rolling 20-bar VWAP → institutional selling pressure.
            # Price rejected aggressively off +2 or +3 SD upper band →
            # extreme mean-reversion short signal.
            # ----------------------------------------------------------
            vwap = crypto.get('vwap', 0.0)
            vwap_sd = crypto.get('vwap_sd', 0.0)
            vwap_sd2_upper = vwap + 2.0 * vwap_sd if vwap_sd > 0 else 0.0
            vwap_sd3_upper = vwap + 3.0 * vwap_sd if vwap_sd > 0 else 0.0
            if vwap > 0 and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                if price < vwap / self.VWAP_BUFFER:
                    # Price clearly below VWAP
                    components['vwap_signal'] = 0.20
                elif price < vwap:
                    components['vwap_signal'] = 0.10
                elif (vwap_sd > 0 and vwap_sd3_upper > 0
                      and price <= vwap_sd3_upper * 0.995
                      and price > vwap_sd2_upper):
                    # Rejection off +3 SD upper band (extreme resistance)
                    components['vwap_signal'] = 0.20
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"VWAP +3SD Rejection: price={price:.4f} sd3_upper={vwap_sd3_upper:.4f}")
                elif (vwap_sd > 0 and vwap_sd2_upper > 0
                      and price <= vwap_sd2_upper * 0.997):
                    # Rejection off +2 SD upper band
                    components['vwap_signal'] = 0.15
                    if not self.algo.IsWarmingUp:
                        self.algo.Debug(
                            f"VWAP +2SD Rejection: price={price:.4f} sd2_upper={vwap_sd2_upper:.4f}")

            # ----------------------------------------------------------
            # Signal 6: CVD Divergence (Distribution)
            # Price at or above VWAP +2SD upper band AND CVD trending down
            # over last 5 bars → limit sellers absorbing buyers at resistance.
            # ----------------------------------------------------------
            cvd = crypto.get('cvd')
            if (vwap_sd > 0 and vwap_sd2_upper > 0 and len(crypto['prices']) >= 1
                    and cvd is not None and len(cvd) >= 5):
                price = crypto['prices'][-1]
                if price >= vwap_sd2_upper and cvd[-1] < cvd[-5]:
                    components['cvd_absorption'] = 0.25

            # ----------------------------------------------------------
            # Signal 7: Kalman Mean Reversion (overbought)
            # In choppy markets (KER < 0.3), price extending >0.4% above
            # Kalman estimate → over-extension likely to revert downward.
            # ----------------------------------------------------------
            ker = crypto.get('ker')
            kalman_estimate = crypto.get('kalman_estimate', 0.0)
            if (ker is not None and len(ker) > 0 and ker[-1] < 0.3
                    and kalman_estimate > 0 and len(crypto['prices']) >= 1):
                price = crypto['prices'][-1]
                if price > kalman_estimate * 1.004:
                    components['kalman_reversion'] = 0.20

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_short_score error: {e}")

        score = sum(components.values())
        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Conservative position sizing prioritising capital preservation.

        Returns 70–90% of available capital depending on conviction.
        Bear regime: base reduced by 30%.
        """
        if score >= 0.80:
            # 4+ signals firing – high conviction
            size = 0.90
        elif score >= self.algo.high_conviction_threshold:
            # 3+ signals: good conviction
            size = 0.80
        elif score >= threshold:
            # Entry threshold met: moderate sizing
            size = 0.70
        else:
            size = 0.50

        if self.algo.market_regime == "bear":
            size *= 0.50  # Bear: 50% of standard allocation
        elif self.algo.market_regime == "sideways":
            size *= 1.00  # Sideways: 100% of standard allocation
        # Bull market: 100% of standard allocation (no adjustment needed)

        kelly = self.algo._kelly_fraction()
        return size * kelly
