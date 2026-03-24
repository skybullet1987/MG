# region imports
from AlgorithmImports import *
import numpy as np
# endregion

"""
mg2_setups.py — Setup-Driven Entry Logic for Machine Gun v2
============================================================
Replaces the additive "score soup" with three explicit, non-overlapping
setup families that are coherent with a max-profit momentum-breakout thesis.

Setup families
--------------
1. IgnitionBreakout
   Price breaks above the rolling N-bar high with a surge in volume and
   strong order-book imbalance.  Anti-chase filter via Kalman/VWAP distance.

2. CompressionExpansion
   Bollinger Band squeeze (width below rolling median for K+ bars) releases
   upward: price clears the upper band on rising volume with bullish EMA stack.

3. MomentumContinuation
   An established trend (ADX strong, EMA stack bullish) pulls back to a key
   level (VWAP / EMA20 / Kalman) and reclaims it with fresh OBI and CVD uptick.

Each evaluator returns
    (qualified: bool, confidence: float 0.0–1.0, components: dict)

The dict always includes a ``setup_type`` string for attribution.
A setup is only "qualified" if confidence >= its own minimum confidence gate.
No additive mixing between setups — the single best-qualified setup wins.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants / thresholds
# ─────────────────────────────────────────────────────────────────────────────
IGNITION_BREAKOUT_MIN_CONFIDENCE     = 0.55
COMPRESSION_EXPANSION_MIN_CONFIDENCE = 0.55
MOMENTUM_CONTINUATION_MIN_CONFIDENCE = 0.55

# Order-book imbalance thresholds
OBI_STRONG  = 0.30
OBI_PARTIAL = 0.15

# Volume surge thresholds (ratio vs rolling baseline)
VOL_SURGE_STRONG  = 3.5
VOL_SURGE_PARTIAL = 2.0

# Compression: BB width must be below rolling-median by at least this fraction
COMPRESSION_BB_WIDTH_FACTOR = 0.85   # width < 85% of 30-bar median = compressed
COMPRESSION_MIN_BARS        = 4      # need at least 4 compressed bars

# Expansion: BB width must expand by at least this factor from the compression min
EXPANSION_MIN_FACTOR        = 1.35   # width > 135% of compression min = expanding

# Breakout freshness: entries within this many bars of the high break are "fresh"
BREAKOUT_FRESH_BARS         = 3

# Anti-chase: price must not be more than this % above Kalman estimate
ANTICHASE_MAX_PCT           = 0.030  # 3% max distance above Kalman / VWAP

# Kalman slope: minimum positive slope (as % per bar) to confirm trend
KALMAN_SLOPE_MIN            = 0.0001  # 0.01% per bar

# ADX thresholds
ADX_STRONG    = 20
ADX_MODERATE  = 14

# RS vs BTC: prefer symbols outperforming BTC (positive = outperforming)
RS_BTC_POSITIVE = 0.001   # at least 0.1% outperformance to get RS bonus

# Maximum spread fraction allowed for valid setups
MAX_SPREAD_FOR_SETUP = 0.008  # 0.8% (wider = no entry)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
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
        if price > range_high_20:
            if freshness <= BREAKOUT_FRESH_BARS:
                components['range_break'] = 0.30  # fresh breakout
            else:
                components['range_break'] = 0.15  # stale breakout (partial)
        else:
            return False, 0.0, components  # Hard gate: must have a range break

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

        confidence = (components['range_break']
                      + components['vol_ignition']
                      + components['obi_pressure']
                      + components['ema_alignment']
                      + components['anti_chase']
                      + components['rs_vs_btc']
                      + components['kalman_slope'])

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
        # Count how many of the recent bars (excluding current) were compressed
        compressed_bars = crypto.get('compression_bars', 0)
        compression_min = crypto.get('compression_min_width', width)

        if compressed_bars < COMPRESSION_MIN_BARS:
            return False, 0.0, components  # Hard gate: must have had a squeeze

        # ── Component 1: Compression quality (30%) ───────────────────────────
        squeeze_ratio = compression_min / max(w_med, 1e-10)
        # Tighter squeeze + longer = better quality
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
        if _anti_chase_ok(crypto, max_pct=ANTICHASE_MAX_PCT * 1.5):  # slightly looser for BB breakout
            components['anti_chase'] = 0.10

        # ── Bonus: OBI pressure ───────────────────────────────────────────────
        _, o_contrib = _obi_score(crypto)
        components['obi_pressure'] = o_contrib * 0.05

        # ── Bonus: RS vs BTC ─────────────────────────────────────────────────
        rs = _rs_vs_btc(crypto)
        if rs > RS_BTC_POSITIVE:
            components['rs_vs_btc'] = min(rs * 5, 0.05)

        confidence = (components['compression_quality']
                      + components['expansion_strength']
                      + components['breakout_direction']
                      + components['vol_ignition']
                      + components['anti_chase']
                      + components['obi_pressure']
                      + components['rs_vs_btc'])

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
        # Check which key level price is reclaiming
        vwap        = crypto.get('vwap', 0.0)
        ema20_val   = (crypto['ema_medium'].Current.Value
                       if crypto.get('ema_medium') and crypto['ema_medium'].IsReady else 0.0)
        kalman_est  = crypto.get('kalman_estimate', 0.0)

        # Price should be close to but above at least one of these levels
        best_reclaim = 0.0

        for level, weight in [(vwap, 0.25), (ema20_val, 0.22), (kalman_est, 0.20)]:
            if level <= 0:
                continue
            dist_below = (level - prices[-3]) / level if len(prices) >= 3 else 1.0
            dist_current = (price - level) / level

            # Reclaim: price was at/below level recently, now above it
            was_near = -0.008 <= dist_below <= 0.015   # was near or just below
            is_above = 0.000 <= dist_current <= 0.020  # now just above, not extended
            if was_near and is_above:
                best_reclaim = max(best_reclaim, weight)

        if best_reclaim == 0.0:
            return False, 0.0, components  # no valid reclaim

        components['level_reclaim'] = best_reclaim

        # ── Component 3: OBI freshness + CVD uptick (20%) ────────────────────
        _, o_contrib = _obi_score(crypto)
        cvd_up = _cvd_uptick(crypto, lookback=3)
        if o_contrib > 0 and cvd_up:
            components['obi_freshness'] = 0.20
        elif o_contrib > 0 or cvd_up:
            components['obi_freshness'] = 0.10

        # ── Component 4: Volume confirmation (15%) ───────────────────────────
        _, v_contrib = _vol_score(crypto)
        components['vol_confirmation'] = v_contrib * 0.15

        # ── Component 5: Anti-chase (15%) ────────────────────────────────────
        if _anti_chase_ok(crypto, max_pct=ANTICHASE_MAX_PCT * 0.8):  # tighter for continuation
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
