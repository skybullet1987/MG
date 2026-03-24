# region imports
from AlgorithmImports import *
import numpy as np
# endregion

"""
scoring.py — Setup-Based Score Engine for Machine Gun v2
=========================================================
Owns setup definitions, confidence/ranking logic, and position sizing.
The MicroScalpEngine class is preserved for compatibility with main.py and
mg2_entries.py.  All three setup families (IgnitionBreakout,
CompressionExpansion, MomentumContinuation) are defined in this file.

Design principles
-----------------
• One setup fires per symbol per bar — no additive blending between families.
• A trade only fires when a *specific, named* setup qualifies.
• Kalman is used only for trend confirmation and anti-chase, not mean reversion.
• Mean-reversion entries are not included in this strategy version.
• Kelly is disabled by default; flat fractional sizing with vol-scaling instead.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Minimum confidence threshold for a setup to generate an entry signal.
# This replaces the old ``entry_threshold`` additive score gate.
SETUP_MIN_CONFIDENCE = 0.65

# High-conviction threshold — triggers maximum position size
SETUP_HIGH_CONVICTION = 0.78

# ─────────────────────────────────────────────────────────────────────────────
# Setup-specific minimum confidence gates
# ─────────────────────────────────────────────────────────────────────────────
IGNITION_BREAKOUT_MIN_CONFIDENCE     = 0.65
COMPRESSION_EXPANSION_MIN_CONFIDENCE = 0.65
MOMENTUM_CONTINUATION_MIN_CONFIDENCE = 0.68   # strictest — most prone to over-qualifying

# Order-book imbalance thresholds
OBI_STRONG  = 0.30
OBI_PARTIAL = 0.20   # raised from 0.15 — require clearer imbalance signal

# Volume surge thresholds (ratio vs rolling baseline)
VOL_SURGE_STRONG  = 3.5
VOL_SURGE_PARTIAL = 2.5   # raised from 2.0 — require more meaningful surge

# Compression: BB width must be below rolling-median by at least this fraction
COMPRESSION_BB_WIDTH_FACTOR = 0.80   # tighter: width < 80% of 30-bar median (was 0.85)
COMPRESSION_MIN_BARS        = 6      # need at least 6 compressed bars (was 4)

# Expansion: BB width must expand by at least this factor from the compression min
EXPANSION_MIN_FACTOR        = 1.50   # tighter: width > 150% of compression min (was 1.35)

# Breakout freshness: entries within this many bars of the high break are "fresh"
# Stale breakouts (beyond this window) are a hard reject — no partial credit.
BREAKOUT_FRESH_BARS         = 2      # tightened from 3 — only the freshest ignitions

# Anti-chase: price must not be more than this % above Kalman estimate
ANTICHASE_MAX_PCT           = 0.020  # 2% max distance (was 3%) — tighter anti-chase

# Kalman slope: minimum positive slope (as % per bar) to confirm trend
KALMAN_SLOPE_MIN            = 0.0002  # raised from 0.0001 — require more visible trend

# ADX thresholds — raised to require a more established trend
ADX_STRONG    = 25   # was 20
ADX_MODERATE  = 18   # was 14

# RS vs BTC: prefer symbols outperforming BTC (positive = outperforming)
RS_BTC_POSITIVE = 0.002   # raised from 0.001 — require clearer outperformance

# Maximum spread fraction allowed for valid setups
MAX_SPREAD_FOR_SETUP = 0.006  # 0.6% (tightened from 0.8%)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities (shared by all setup classes)
# ─────────────────────────────────────────────────────────────────────────────

def _obi_score(crypto):
    """Return (raw_obi_value, obi_contribution 0-1) from rolling OBI history."""
    history = list(crypto.get('obi_history', []))
    if len(history) >= 3:
        obi = float(np.mean(history[-3:]))
        min_obi = min(history[-3:])
    elif len(history) >= 1:
        obi = float(history[-1])
        min_obi = obi
    else:
        bid = crypto.get('bid_size', 0.0)
        ask = crypto.get('ask_size', 0.0)
        total = bid + ask
        obi = (bid - ask) / total if total > 0 else 0.0
        min_obi = obi

    if obi > OBI_STRONG and min_obi > 0.08:
        return obi, 1.0
    elif obi > OBI_PARTIAL and min_obi > 0.03:
        return obi, 0.5
    return obi, 0.0


def _vol_score(crypto):
    """Return volume ratio and contribution (0-1) vs rolling baseline."""
    volumes = list(crypto.get('volume', []))
    if len(volumes) < 5:
        return 0.0, 0.0
    current = volumes[-1]
    long_vols = list(crypto.get('volume_long', []))
    baseline = float(np.mean(long_vols[-120:])) if len(long_vols) >= 120 \
               else float(np.mean(volumes[-20:]))
    if baseline <= 0:
        return 0.0, 0.0
    ratio = current / baseline
    if ratio >= VOL_SURGE_STRONG:
        return ratio, 1.0
    elif ratio >= VOL_SURGE_PARTIAL:
        return ratio, 0.5
    return ratio, 0.0


def _ema_bullish(crypto):
    """True if EMA5 > EMA20 and price > EMA5 (full bullish EMA stack)."""
    if not (crypto['ema_5'].IsReady
            and crypto.get('ema_medium') is not None
            and crypto['ema_medium'].IsReady):
        return False
    prices = list(crypto['prices'])
    if not prices:
        return False
    price = prices[-1]
    ema5  = crypto['ema_5'].Current.Value
    ema20 = crypto['ema_medium'].Current.Value
    return price > ema5 > ema20


def _anti_chase_ok(crypto, max_pct=ANTICHASE_MAX_PCT):
    """
    Return True if price is not too far above Kalman estimate (anti-chase).
    Also check VWAP distance as a secondary reference.
    """
    prices = list(crypto['prices'])
    if not prices:
        return True
    price = prices[-1]

    kalman = crypto.get('kalman_estimate', 0.0)
    if kalman > 0:
        dist_kalman = (price - kalman) / kalman
        if dist_kalman > max_pct:
            return False

    vwap = crypto.get('vwap', 0.0)
    if vwap > 0:
        dist_vwap = (price - vwap) / vwap
        if dist_vwap > max_pct * 1.5:  # slightly looser for VWAP
            return False

    return True


def _rs_vs_btc(crypto):
    """Return relative-strength vs BTC over recent window (float)."""
    rs_hist = list(crypto.get('rs_vs_btc', []))
    if not rs_hist:
        return 0.0
    return float(rs_hist[-1])


def _kalman_slope(crypto):
    """
    Return Kalman estimate slope (per bar) using last 2 kalman snapshots.
    If only one estimate is available, return 0.
    """
    hist = list(crypto.get('kalman_history', []))
    if len(hist) < 2:
        return 0.0
    slope = (hist[-1] - hist[-2]) / max(hist[-2], 1e-10)
    return float(slope)


def _cvd_uptick(crypto, lookback=5):
    """Return True if CVD has been rising over the last `lookback` bars."""
    cvd = list(crypto.get('cvd', []))
    if len(cvd) < lookback + 1:
        return False
    return cvd[-1] > cvd[-lookback - 1]


# ─────────────────────────────────────────────────────────────────────────────
# Setup 1: Ignition Breakout
# ─────────────────────────────────────────────────────────────────────────────

class IgnitionBreakoutSetup:
    """
    Price surges above the rolling 20-bar high with strong volume and OBI.

    Scoring breakdown (all components sum to 1.0):
      - range_break   (0.30): price > 20-bar high
      - vol_ignition  (0.25): volume ratio vs baseline
      - obi_pressure  (0.20): order-book imbalance
      - ema_alignment (0.15): EMA5 > EMA20, price > EMA5
      - anti_chase    (0.10): price not extended above Kalman/VWAP
    """

    MIN_CONFIDENCE = IGNITION_BREAKOUT_MIN_CONFIDENCE

    @classmethod
    def evaluate(cls, crypto, algo):
        components = {
            'setup_type':    'IgnitionBreakout',
            'range_break':   0.0,
            'vol_ignition':  0.0,
            'obi_pressure':  0.0,
            'ema_alignment': 0.0,
            'anti_chase':    0.0,
            'rs_vs_btc':     0.0,
            'kalman_slope':  0.0,
        }

        prices = list(crypto['prices'])
        highs  = list(crypto.get('highs', []))
        if len(prices) < 21 or len(highs) < 21:
            return False, 0.0, components

        price = prices[-1]

        # ── Component 1: Range break (30% weight) ───────────────────────────
        range_high_20 = float(np.max(highs[-21:-1]))  # exclude current bar
        freshness = crypto.get('breakout_freshness', 999)
        if price > range_high_20 and freshness <= BREAKOUT_FRESH_BARS:
            components['range_break'] = 0.30  # fresh breakout only — stale = hard reject
        else:
            return False, 0.0, components  # Hard gate: must be a FRESH range break

        # ── Component 2: Volume ignition (25% weight) ────────────────────────
        _, v_contrib = _vol_score(crypto)
        components['vol_ignition'] = v_contrib * 0.25

        # ── Component 3: OBI pressure (20% weight) ───────────────────────────
        _, o_contrib = _obi_score(crypto)
        components['obi_pressure'] = o_contrib * 0.20

        # ── Component 4: EMA alignment (15% weight) ──────────────────────────
        if _ema_bullish(crypto):
            components['ema_alignment'] = 0.15
        elif (crypto['ema_5'].IsReady and crypto.get('ema_medium') is not None
              and crypto['ema_medium'].IsReady):
            ema5  = crypto['ema_5'].Current.Value
            ema20 = crypto['ema_medium'].Current.Value
            if ema5 > ema20:  # at least EMA5 > EMA20
                components['ema_alignment'] = 0.08

        # ── Component 5: Anti-chase (10% weight) ─────────────────────────────
        if _anti_chase_ok(crypto):
            components['anti_chase'] = 0.10

        # ── Bonus: RS vs BTC ─────────────────────────────────────────────────
        rs = _rs_vs_btc(crypto)
        if rs > RS_BTC_POSITIVE:
            components['rs_vs_btc'] = min(rs * 5, 0.05)  # up to +5% bonus

        # ── Bonus: Kalman slope confirming uptrend ────────────────────────────
        slope = _kalman_slope(crypto)
        components['kalman_slope_raw'] = slope
        if slope >= KALMAN_SLOPE_MIN:
            components['kalman_slope'] = 0.05

        # ── Bonus: Volume persistence (multi-bar buying pressure) ─────────────
        # Real ignition breakouts are preceded by sustained above-average volume,
        # not just a single spike.  2+ consecutive above-baseline bars = genuine.
        vol_persist = crypto.get('vol_persistence', 0)
        if vol_persist >= 5:
            components['vol_persistence'] = 0.08
        elif vol_persist >= 2:
            components['vol_persistence'] = 0.05
        else:
            components['vol_persistence'] = 0.0

        confidence = (components['range_break']
                      + components['vol_ignition']
                      + components['obi_pressure']
                      + components['ema_alignment']
                      + components['anti_chase']
                      + components['rs_vs_btc']
                      + components['kalman_slope']
                      + components['vol_persistence'])

        # Must have both volume AND OBI — don't let one comp carry the whole score
        if components['vol_ignition'] == 0.0 and components['obi_pressure'] == 0.0:
            return False, 0.0, components

        qualified = confidence >= cls.MIN_CONFIDENCE
        return qualified, round(min(confidence, 1.0), 4), components


# ─────────────────────────────────────────────────────────────────────────────
# Setup 2: Compression → Expansion Breakout
# ─────────────────────────────────────────────────────────────────────────────

class CompressionExpansionSetup:
    """
    Bollinger Band squeeze releases upward.

    Detection:
      - Compression: BB width < 85% of 30-bar width median for COMPRESSION_MIN_BARS+ bars
      - Expansion trigger: current width > 135% of recent compression minimum
      - Direction: price above upper band or at least bullish EMA

    Scoring breakdown:
      - compression_quality (0.30): how long / tight the squeeze was
      - expansion_strength  (0.25): how aggressively width is expanding
      - breakout_direction  (0.20): price above BB upper band
      - vol_ignition        (0.15): volume confirms
      - anti_chase          (0.10): not extended
    """

    MIN_CONFIDENCE = COMPRESSION_EXPANSION_MIN_CONFIDENCE

    @classmethod
    def evaluate(cls, crypto, algo):
        components = {
            'setup_type':           'CompressionExpansion',
            'compression_quality':  0.0,
            'expansion_strength':   0.0,
            'breakout_direction':   0.0,
            'vol_ignition':         0.0,
            'anti_chase':           0.0,
            'rs_vs_btc':            0.0,
            'obi_pressure':         0.0,
        }

        prices    = list(crypto['prices'])
        bb_widths = list(crypto.get('bb_width', []))
        bb_uppers = list(crypto.get('bb_upper', []))

        if len(prices) < 10 or len(bb_widths) < 15:
            return False, 0.0, components

        price  = prices[-1]
        width  = bb_widths[-1]
        w_arr  = np.array(bb_widths[-30:] if len(bb_widths) >= 30 else bb_widths)
        w_med  = float(np.median(w_arr))

        # ── Check compression history ─────────────────────────────────────────
        compressed_bars = crypto.get('compression_bars', 0)
        compression_min = crypto.get('compression_min_width', width)

        if compressed_bars < COMPRESSION_MIN_BARS:
            return False, 0.0, components  # Hard gate: must have had a squeeze

        # ── Component 1: Compression quality (30%) ───────────────────────────
        squeeze_ratio = compression_min / max(w_med, 1e-10)
        bar_bonus = min((compressed_bars - COMPRESSION_MIN_BARS) / 8.0, 1.0)
        tightness = max(0.0, 1.0 - squeeze_ratio / COMPRESSION_BB_WIDTH_FACTOR)
        comp_quality = min((0.5 + 0.5 * tightness) * (0.7 + 0.3 * bar_bonus), 1.0)
        components['compression_quality'] = comp_quality * 0.30

        # ── Component 2: Expansion strength (25%) ────────────────────────────
        if compression_min > 0:
            expansion_ratio = width / compression_min
        else:
            expansion_ratio = 1.0
        if expansion_ratio >= EXPANSION_MIN_FACTOR:
            exp_score = min((expansion_ratio - 1.0) / 0.5, 1.0)
            components['expansion_strength'] = exp_score * 0.25
        else:
            return False, 0.0, components  # Hard gate: must be expanding now

        # ── Component 3: Breakout direction (20%) ────────────────────────────
        if bb_uppers and price > bb_uppers[-1]:
            components['breakout_direction'] = 0.20
        elif _ema_bullish(crypto):
            components['breakout_direction'] = 0.12

        # ── Component 4: Volume confirmation (15%) ───────────────────────────
        _, v_contrib = _vol_score(crypto)
        components['vol_ignition'] = v_contrib * 0.15

        # ── Component 5: Anti-chase (10%) ────────────────────────────────────
        if _anti_chase_ok(crypto, max_pct=ANTICHASE_MAX_PCT * 1.5):
            components['anti_chase'] = 0.10

        # ── Bonus: OBI pressure ───────────────────────────────────────────────
        _, o_contrib = _obi_score(crypto)
        components['obi_pressure'] = o_contrib * 0.05

        # ── Bonus: RS vs BTC ─────────────────────────────────────────────────
        rs = _rs_vs_btc(crypto)
        if rs > RS_BTC_POSITIVE:
            components['rs_vs_btc'] = min(rs * 5, 0.05)

        # ── Bonus: Volume persistence (sustained buying into the expansion) ───
        vol_persist = crypto.get('vol_persistence', 0)
        if vol_persist >= 2:
            components['vol_persistence'] = 0.04
        else:
            components['vol_persistence'] = 0.0

        confidence = (components['compression_quality']
                      + components['expansion_strength']
                      + components['breakout_direction']
                      + components['vol_ignition']
                      + components['anti_chase']
                      + components['obi_pressure']
                      + components['rs_vs_btc']
                      + components['vol_persistence'])

        qualified = confidence >= cls.MIN_CONFIDENCE
        return qualified, round(min(confidence, 1.0), 4), components


# ─────────────────────────────────────────────────────────────────────────────
# Setup 3: Momentum Continuation / Reclaim
# ─────────────────────────────────────────────────────────────────────────────

class MomentumContinuationSetup:
    """
    Established trend pulls back to a key level (VWAP / EMA20 / Kalman estimate)
    and reclaims it — fresh momentum continuation opportunity.

    Requirements:
      - Strong trend already established (ADX > ADX_STRONG, EMA stack bullish)
      - Price pulled back toward (but not through) key level
      - Reclaim confirmed: price now back above the level with CVD uptick + OBI
      - Not over-extended

    Scoring breakdown:
      - trend_strength   (0.25): ADX + EMA stack
      - level_reclaim    (0.25): which level was reclaimed and how cleanly
      - obi_freshness    (0.20): OBI positive + CVD uptick
      - vol_confirmation (0.15): volume above baseline
      - anti_chase       (0.15): not extended above all levels
    """

    MIN_CONFIDENCE = MOMENTUM_CONTINUATION_MIN_CONFIDENCE

    @classmethod
    def evaluate(cls, crypto, algo):
        components = {
            'setup_type':      'MomentumContinuation',
            'trend_strength':  0.0,
            'level_reclaim':   0.0,
            'obi_freshness':   0.0,
            'vol_confirmation':0.0,
            'anti_chase':      0.0,
            'rs_vs_btc':       0.0,
            'kalman_slope':    0.0,
        }

        prices = list(crypto['prices'])
        if len(prices) < 12:
            return False, 0.0, components

        price = prices[-1]

        # ── Hard gate: ADX trend must be established ──────────────────────────
        adx_ind = crypto.get('adx')
        if adx_ind is None or not adx_ind.IsReady:
            return False, 0.0, components

        adx_val  = adx_ind.Current.Value
        di_plus  = adx_ind.PositiveDirectionalIndex.Current.Value
        di_minus = adx_ind.NegativeDirectionalIndex.Current.Value

        if adx_val < ADX_MODERATE or di_plus <= di_minus:
            return False, 0.0, components  # no established uptrend

        # ── Component 1: Trend strength (25%) ────────────────────────────────
        if adx_val >= ADX_STRONG and di_plus > di_minus * 1.2:
            components['trend_strength'] = 0.25
        elif adx_val >= ADX_MODERATE:
            components['trend_strength'] = 0.15

        # ── Component 2: Level reclaim (25%) ─────────────────────────────────
        vwap          = crypto.get('vwap', 0.0)
        ema20_val     = (crypto['ema_medium'].Current.Value
                         if crypto.get('ema_medium') and crypto['ema_medium'].IsReady else 0.0)
        kalman_est    = crypto.get('kalman_estimate', 0.0)
        vwap_sd2_lower  = crypto.get('vwap_sd2_lower', 0.0)   # VWAP -2σ band reclaim

        best_reclaim = 0.0

        # Tighter reclaim window: was_near checks 3 bars back, is_above is stricter.
        # Price must have actually pulled back TO the level, not just been near it.
        for level, weight in [(vwap, 0.25), (ema20_val, 0.22), (kalman_est, 0.20), (vwap_sd2_lower, 0.18)]:
            if level <= 0:
                continue
            dist_below = (level - prices[-3]) / level if len(prices) >= 3 else 1.0
            dist_current = (price - level) / level

            # Tightened: was within 0.5% below to 1% above (was -0.8% to 1.5%)
            # and currently 0% to 1% above (was 0% to 2%)
            was_near = -0.005 <= dist_below <= 0.010
            is_above = 0.000 <= dist_current <= 0.010
            if was_near and is_above:
                best_reclaim = max(best_reclaim, weight)

        if best_reclaim == 0.0:
            return False, 0.0, components  # no valid reclaim

        components['level_reclaim'] = best_reclaim

        # ── Component 3: OBI freshness + CVD uptick (20%) ────────────────────
        _, o_contrib = _obi_score(crypto)
        cvd_up = _cvd_uptick(crypto, lookback=3)
        # Require BOTH OBI and CVD to avoid false signals from one noisy indicator
        if o_contrib > 0 and cvd_up:
            components['obi_freshness'] = 0.20
        else:
            return False, 0.0, components  # Hard gate: need both OBI + CVD confirmation

        # ── Component 4: Volume confirmation (15%) ───────────────────────────
        _, v_contrib = _vol_score(crypto)
        components['vol_confirmation'] = v_contrib * 0.15

        # ── Component 5: Anti-chase (15%) ────────────────────────────────────
        if _anti_chase_ok(crypto, max_pct=ANTICHASE_MAX_PCT * 0.8):
            components['anti_chase'] = 0.15

        # ── Bonus: RS vs BTC ─────────────────────────────────────────────────
        rs = _rs_vs_btc(crypto)
        if rs > RS_BTC_POSITIVE:
            components['rs_vs_btc'] = min(rs * 5, 0.05)

        # ── Bonus: Kalman slope ───────────────────────────────────────────────
        slope = _kalman_slope(crypto)
        components['kalman_slope_raw'] = slope
        if slope >= KALMAN_SLOPE_MIN:
            components['kalman_slope'] = 0.05

        confidence = (components['trend_strength']
                      + components['level_reclaim']
                      + components['obi_freshness']
                      + components['vol_confirmation']
                      + components['anti_chase']
                      + components['rs_vs_btc']
                      + components['kalman_slope'])

        qualified = confidence >= cls.MIN_CONFIDENCE
        return qualified, round(min(confidence, 1.0), 4), components


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluator: run all setups, return the best-qualified one
# ─────────────────────────────────────────────────────────────────────────────

_SETUP_CLASSES = [
    IgnitionBreakoutSetup,
    CompressionExpansionSetup,
    MomentumContinuationSetup,
]


def evaluate_all_setups(crypto, algo):
    """
    Evaluate all setup families against a single symbol's crypto data.

    Returns a list of qualified setups, sorted by confidence descending.
    Each element is (setup_type: str, confidence: float, components: dict).
    An empty list means no setup qualifies.
    """
    results = []
    for cls in _SETUP_CLASSES:
        try:
            qualified, confidence, components = cls.evaluate(crypto, algo)
            if qualified:
                results.append((components['setup_type'], confidence, components))
        except Exception as e:
            if hasattr(algo, 'Debug'):
                algo.Debug(f"Setup eval error ({cls.__name__}): {e}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def best_setup(crypto, algo):
    """
    Return the single highest-confidence qualifying setup as
    (setup_type, confidence, components) or (None, 0.0, {}) if none qualify.
    """
    setups = evaluate_all_setups(crypto, algo)
    if setups:
        return setups[0]
    return None, 0.0, {}


class MicroScalpEngine:
    """
    Momentum-Breakout Setup Engine — v8.0.0

    Entry logic is now driven exclusively by three explicit setup families
    (IgnitionBreakout, CompressionExpansion, MomentumContinuation) defined
    in this file.

    The additive multi-signal score has been removed.  Each bar, the engine
    asks: "Does this symbol match a known, high-quality breakout setup?"
    If yes, it returns the setup type, confidence, and components.
    If no setup qualifies, no trade is taken.

    Kalman repurposed
    -----------------
    • Kalman slope → trend direction confirmation
    • Kalman distance → anti-chase filter
    • Kalman estimate → dynamic trailing reference (in exits)

    Mean reversion removed
    ----------------------
    Mean-reversion bounce entries are no longer a valid entry path.
    RSI oversold and BB band bounce signals have been removed from entries.
    """

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (setup_type, confidence, components)
    # or (None, 0.0, {}) when no setup qualifies.
    # ------------------------------------------------------------------
    def evaluate_setup(self, crypto):
        """
        Evaluate all momentum-breakout setup families for a symbol.

        Returns
        -------
        (setup_type: str | None, confidence: float, components: dict)
        """
        return best_setup(crypto, self.algo)

    # Legacy compatibility shim used by mg2_entries.py
    def calculate_scalp_score(self, crypto):
        """
        Backward-compatible wrapper.

        Returns (score: float, components: dict) where score = confidence and
        components includes 'setup_type' for attribution.
        """
        setup_type, confidence, components = self.evaluate_setup(crypto)
        if setup_type is None:
            return 0.0, {'setup_type': None}
        return confidence, components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Flat-fractional sizing with volatility scaling.

        Kelly is disabled for this version — it obscures whether the entry
        architecture has real edge and can amplify noise on small accounts.
        Use the ``use_kelly`` parameter (default False) in main.py to re-enable.

        High-conviction setups (score >= SETUP_HIGH_CONVICTION) get larger
        base size; otherwise a conservative base is used.

        Volatility scaling reduces size for hyper-volatile assets but never
        drops below 50% of base size.

        Returns a fraction of available capital (0.0 – 1.0).
        """
        # Base size by conviction
        if score >= SETUP_HIGH_CONVICTION:
            base_size = getattr(self.algo, 'position_size_high_conviction', 0.45)
        elif score >= threshold:
            base_size = getattr(self.algo, 'position_size_pct', 0.35)
        else:
            base_size = 0.25

        # Volatility scaling: target annual vol of position ≈ target_position_ann_vol
        if asset_vol_ann is not None and asset_vol_ann > 0:
            target_vol = getattr(self.algo, 'target_position_ann_vol', 0.40)
            vol_scalar = min(target_vol / asset_vol_ann, 1.0)
            base_size *= max(vol_scalar, 0.50)

        # Kelly gate: only apply if explicitly enabled in config
        if getattr(self.algo, 'use_kelly', False):
            kelly = self.algo._kelly_fraction()
            base_size *= kelly

        # Hard cap: never exceed max_position_pct of portfolio per trade
        max_pct = getattr(self.algo, 'max_position_pct', 0.50)
        base_size = min(base_size, max_pct)

        return base_size
