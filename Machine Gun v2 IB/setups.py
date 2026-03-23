"""
setups.py — Machine Gun v2 IB: Setup evaluation logic

Three explicit setup families replace the legacy blended composite score:

  1. trend_pullback
       Enter in the direction of the established trend after a short pullback
       to EMA5 / VWAP support.  Requires clear trend (EMA alignment + ADX).

  2. mean_reversion
       Enter counter-trend when price is extended from equilibrium in a
       ranging/low-trend environment.  Requires ADX < threshold (choppy).

  3. breakout_compression
       Enter a breakout after a period of Bollinger-Band compression with
       volume surge confirmation.

Design principles
-----------------
* Each evaluator is fully self-contained — signals are clear and purposeful.
* No blended microstructure gate obscures individual signal contributions.
* Consistent threshold semantics: score >= threshold ⟹ valid entry.
* Directional selection: a side is ONLY chosen if its score is meaningfully
  higher than the other side (min_gap enforced), never by elimination.
* Easy to add/remove/disable signals without side effects.
"""

import numpy as np
from candidates import (
    SetupCandidate,
    SETUP_TREND_PULLBACK,
    SETUP_MEAN_REVERSION,
    SETUP_BREAKOUT_COMPRESSION,
)


# =============================================================================
# Internal helpers
# =============================================================================

def _safe_last(seq, default=0.0):
    """Return the last element of seq, or default if empty/None."""
    return float(seq[-1]) if seq and len(seq) > 0 else default


def _ema_val(mnq, key):
    ind = mnq.get(key)
    if ind is None or not ind.IsReady:
        return None
    return float(ind.Current.Value)


def _adx_triplet(mnq):
    """Return (adx, di_plus, di_minus) or (None, None, None) if not ready."""
    ind = mnq.get("adx")
    if ind is None or not ind.IsReady:
        return None, None, None
    return (
        float(ind.Current.Value),
        float(ind.PositiveDirectionalIndex.Current.Value),
        float(ind.NegativeDirectionalIndex.Current.Value),
    )


def _rsi_val(mnq):
    ind = mnq.get("rsi")
    if ind is None or not ind.IsReady:
        return None
    return float(ind.Current.Value)


def _vol_ratio(mnq):
    """Current-bar volume / rolling baseline.  Returns None if insufficient data."""
    vol_list = list(mnq.get("volume", []))
    if len(vol_list) < 20:
        return None
    current = vol_list[-1]
    vol_long = list(mnq.get("volume_long", []))
    baseline = float(np.mean(vol_long)) if len(vol_long) >= 60 else float(np.mean(vol_list[-20:]))
    return (current / baseline) if baseline > 0 else None


def _bb_state(mnq):
    """
    Return (price, bb_upper, bb_lower, bb_width, bb_width_ma).
    Any unavailable value is returned as None.
    """
    prices = mnq.get("prices", [])
    price = _safe_last(prices)
    if price <= 0:
        return None, None, None, None, None

    bb_upper_data = mnq.get("bb_upper", [])
    bb_lower_data = mnq.get("bb_lower", [])
    bb_width_data = mnq.get("bb_width", [])

    bb_upper   = float(bb_upper_data[-1])  if len(bb_upper_data)  >= 1 else None
    bb_lower   = float(bb_lower_data[-1])  if len(bb_lower_data)  >= 1 else None
    bb_width   = float(bb_width_data[-1])  if len(bb_width_data)  >= 1 else None
    bb_width_ma = float(np.mean(list(bb_width_data))) if len(bb_width_data) >= 5 else None

    return price, bb_upper, bb_lower, bb_width, bb_width_ma


# =============================================================================
# Direction-selection helper
# =============================================================================

def _pick_direction(long_score, long_comps, long_notes,
                    short_score, short_comps, short_notes,
                    min_gap):
    """
    Choose direction from long/short scores.

    Rules
    -----
    1. |long_score - short_score| < min_gap  →  ambiguous signal, return None.
    2. Winning side must also have score > 0.10 to avoid choosing by default.
    3. Long wins when long_score > short_score; short wins otherwise.

    A side is NEVER chosen simply because the other side failed its threshold.
    """
    gap = abs(long_score - short_score)
    if gap < min_gap:
        return None, 0.0, {}, [f"ambiguous: gap={gap:.3f} < min_gap={min_gap}"]

    if long_score > short_score:
        if long_score < 0.10:
            return None, 0.0, {}, ["long score too low to act on"]
        return 1, long_score, long_comps, long_notes
    else:
        if short_score < 0.10:
            return None, 0.0, {}, ["short score too low to act on"]
        return -1, short_score, short_comps, short_notes


# =============================================================================
# Setup 1: Trend Pullback
# =============================================================================
#
# Rationale: enter in the direction of the established trend after a short
# pullback to the fast EMA (EMA5) or VWAP support/resistance.
#
# Long conditions:
#   - Medium trend up: EMA5 > EMA20
#   - Trend strength: ADX > 13 with DI+ > DI-
#   - Pullback: price within ±0.2% of EMA5
#   - VWAP: price >= VWAP (still above institutional reference)
#   - RSI not overbought (< 70)
#
# Short conditions (mirror image):
#   - EMA5 < EMA20
#   - ADX > 13 with DI- > DI+
#   - Pullback: price within ±0.2% of EMA5
#   - Price <= VWAP
#   - RSI not oversold (> 30)
#
# Scoring components (max weights):
#   trend_ema  : 0.35   EMA stack alignment
#   trend_adx  : 0.30   ADX strength + directional index
#   pullback   : 0.20   proximity to EMA5
#   vwap       : 0.15   VWAP alignment
# Max total: 1.00
# =============================================================================

_TP_PULLBACK_PCT  = 0.002   # price within ±0.2% of EMA5 = pullback zone
_TP_ADX_STRONG    = 20
_TP_ADX_MODERATE  = 13


def _score_trend_pullback_long(mnq, regime):
    components = {}
    notes = []

    price = _safe_last(mnq.get("prices", []))
    if price <= 0:
        return 0.0, components, ["no price data"]

    ema5  = _ema_val(mnq, "ema_5")
    ema20 = _ema_val(mnq, "ema_medium")
    adx_val, di_plus, di_minus = _adx_triplet(mnq)
    rsi  = _rsi_val(mnq)
    vwap = float(mnq.get("vwap", 0.0))

    # ── Trend alignment (EMA stack) ──────────────────────────────────────────
    if ema5 is None or ema20 is None:
        return 0.0, {}, ["EMAs not ready"]
    if ema5 > ema20:
        components["trend_ema"] = 0.35
        notes.append("EMA5>EMA20 uptrend")
    else:
        # No uptrend → this setup doesn't apply for longs
        return 0.0, {"trend_ema": 0.0}, ["EMA5<=EMA20 (no uptrend)"]

    # ── Trend strength (ADX) ─────────────────────────────────────────────────
    if adx_val is not None and di_plus is not None and di_minus is not None:
        if di_plus > di_minus:
            if adx_val >= _TP_ADX_STRONG:
                components["trend_adx"] = 0.30
                notes.append(f"ADX={adx_val:.1f} DI+>DI- strong")
            elif adx_val >= _TP_ADX_MODERATE:
                components["trend_adx"] = 0.20
                notes.append(f"ADX={adx_val:.1f} DI+>DI- moderate")
            else:
                components["trend_adx"] = 0.10
                notes.append(f"ADX={adx_val:.1f} DI+>DI- weak")
        else:
            components["trend_adx"] = 0.0
            notes.append(f"DI->=DI+ ({di_minus:.1f} vs {di_plus:.1f}) bearish bias")

    # ── Pullback quality (proximity to EMA5) ─────────────────────────────────
    if ema5 > 0:
        dist = (price - ema5) / ema5   # negative = price pulled back below EMA5
        if -_TP_PULLBACK_PCT <= dist <= _TP_PULLBACK_PCT:
            components["pullback"] = 0.20
            notes.append(f"price {dist:+.4f} EMA5 (pullback zone)")
        elif -2 * _TP_PULLBACK_PCT <= dist <= 2 * _TP_PULLBACK_PCT:
            components["pullback"] = 0.10
            notes.append(f"price {dist:+.4f} EMA5 (near pullback)")
        else:
            components["pullback"] = 0.0
            notes.append(f"price {dist:+.4f} EMA5 (not in pullback zone)")

    # ── VWAP alignment (long: price >= VWAP) ─────────────────────────────────
    if vwap > 0:
        if price >= vwap:
            components["vwap"] = 0.15
            notes.append("price>=VWAP")
        else:
            components["vwap"] = 0.0
            notes.append("price<VWAP (below institutional reference)")

    # ── Quality gate: RSI must not be overbought ──────────────────────────────
    if rsi is not None and rsi >= 70:
        notes.append(f"RSI={rsi:.1f} overbought: reducing pullback score")
        components["pullback"] = max(0.0, components.get("pullback", 0.0) - 0.10)

    score = sum(components.values())
    return min(score, 1.0), components, notes


def _score_trend_pullback_short(mnq, regime):
    components = {}
    notes = []

    price = _safe_last(mnq.get("prices", []))
    if price <= 0:
        return 0.0, components, ["no price data"]

    ema5  = _ema_val(mnq, "ema_5")
    ema20 = _ema_val(mnq, "ema_medium")
    adx_val, di_plus, di_minus = _adx_triplet(mnq)
    rsi  = _rsi_val(mnq)
    vwap = float(mnq.get("vwap", 0.0))

    # ── Trend alignment ───────────────────────────────────────────────────────
    if ema5 is None or ema20 is None:
        return 0.0, {}, ["EMAs not ready"]
    if ema5 < ema20:
        components["trend_ema"] = 0.35
        notes.append("EMA5<EMA20 downtrend")
    else:
        return 0.0, {"trend_ema": 0.0}, ["EMA5>=EMA20 (no downtrend)"]

    # ── Trend strength ────────────────────────────────────────────────────────
    if adx_val is not None and di_plus is not None and di_minus is not None:
        if di_minus > di_plus:
            if adx_val >= _TP_ADX_STRONG:
                components["trend_adx"] = 0.30
                notes.append(f"ADX={adx_val:.1f} DI->DI+ strong")
            elif adx_val >= _TP_ADX_MODERATE:
                components["trend_adx"] = 0.20
                notes.append(f"ADX={adx_val:.1f} DI->DI+ moderate")
            else:
                components["trend_adx"] = 0.10
                notes.append(f"ADX={adx_val:.1f} DI->DI+ weak")
        else:
            components["trend_adx"] = 0.0
            notes.append(f"DI+>=DI- ({di_plus:.1f} vs {di_minus:.1f}) bullish bias")

    # ── Pullback quality ──────────────────────────────────────────────────────
    if ema5 > 0:
        dist = (price - ema5) / ema5
        if -_TP_PULLBACK_PCT <= dist <= _TP_PULLBACK_PCT:
            components["pullback"] = 0.20
            notes.append(f"price {dist:+.4f} EMA5 (pullback zone)")
        elif -2 * _TP_PULLBACK_PCT <= dist <= 2 * _TP_PULLBACK_PCT:
            components["pullback"] = 0.10
            notes.append(f"price {dist:+.4f} EMA5 (near pullback)")
        else:
            components["pullback"] = 0.0
            notes.append(f"price {dist:+.4f} EMA5 (not in pullback zone)")

    # ── VWAP alignment (short: price <= VWAP) ─────────────────────────────────
    if vwap > 0:
        if price <= vwap:
            components["vwap"] = 0.15
            notes.append("price<=VWAP")
        else:
            components["vwap"] = 0.0
            notes.append("price>VWAP (above institutional reference)")

    # ── Quality gate: RSI must not be oversold ────────────────────────────────
    if rsi is not None and rsi <= 30:
        notes.append(f"RSI={rsi:.1f} oversold: reducing pullback score")
        components["pullback"] = max(0.0, components.get("pullback", 0.0) - 0.10)

    score = sum(components.values())
    return min(score, 1.0), components, notes


def evaluate_trend_pullback(symbol, mnq, regime, session, vix, config):
    """
    Evaluate both long and short trend-pullback setups for one symbol.
    Returns the best-direction SetupCandidate, or None if neither qualifies.
    """
    threshold = config.TREND_PULLBACK_THRESHOLD
    long_score,  long_comps,  long_notes  = _score_trend_pullback_long(mnq, regime)
    short_score, short_comps, short_notes = _score_trend_pullback_short(mnq, regime)

    direction, score, comps, notes = _pick_direction(
        long_score, long_comps, long_notes,
        short_score, short_comps, short_notes,
        config.MIN_DIRECTION_GAP,
    )
    if direction is None:
        return None

    symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    return SetupCandidate(
        symbol=symbol,
        symbol_name=symbol_name,
        direction=direction,
        setup_type=SETUP_TREND_PULLBACK,
        score=score,
        threshold=threshold,
        long_score=long_score,
        short_score=short_score,
        components=comps,
        regime=regime,
        session=session,
        vix=vix,
        notes=notes,
    )


# =============================================================================
# Setup 2: Mean Reversion
# =============================================================================
#
# Rationale: enter counter-trend when price is extended from equilibrium
# in a ranging/low-trend environment.
#
# Hard prerequisite: ADX < 20 (market must be ranging, not trending).
#
# Long conditions:
#   - RSI < 40 (oversold), or < 50 for partial credit
#   - Price near/below lower Bollinger Band
#   - KER < 0.30 (choppy price path, not directional)
#   - Price below VWAP (extended)
#
# Short conditions (mirror image):
#   - RSI > 60 (overbought), or > 50 for partial credit
#   - Price near/above upper BB
#   - KER < 0.30
#   - Price above VWAP
#
# Scoring components (max weights):
#   rsi          : 0.35
#   bb_extension : 0.30
#   choppiness   : 0.20
#   vwap_dist    : 0.15
# Max total: 1.00
# =============================================================================

_MR_ADX_MAX           = 20
_MR_RSI_OVERSOLD      = 40
_MR_RSI_MILD_OVERSOLD = 50
_MR_RSI_OVERBOUGHT    = 60
_MR_RSI_MILD_OB       = 50
_MR_BB_NEAR_PCT       = 0.02    # within 2% of BB = "near"
_MR_KER_CHOPPY        = 0.30    # KER < 0.30 = choppy


def _score_mean_reversion_long(mnq, regime):
    components = {}
    notes = []

    price = _safe_last(mnq.get("prices", []))
    if price <= 0:
        return 0.0, components, ["no price data"]

    adx_val, _, _ = _adx_triplet(mnq)
    rsi = _rsi_val(mnq)
    _, _, bb_lower, _, _ = _bb_state(mnq)
    ker_data = mnq.get("ker", [])
    ker  = float(ker_data[-1]) if ker_data and len(ker_data) > 0 else None
    vwap = float(mnq.get("vwap", 0.0))

    # Hard requirement: ADX must indicate a ranging market
    if adx_val is not None and adx_val >= _MR_ADX_MAX:
        return 0.0, {}, [f"ADX={adx_val:.1f}>={_MR_ADX_MAX} (trending, skip mean-rev)"]

    # ── RSI extension (primary signal) ───────────────────────────────────────
    if rsi is not None:
        if rsi < _MR_RSI_OVERSOLD:
            components["rsi"] = 0.35
            notes.append(f"RSI={rsi:.1f} oversold")
        elif rsi < _MR_RSI_MILD_OVERSOLD:
            components["rsi"] = 0.20
            notes.append(f"RSI={rsi:.1f} mildly oversold")
        else:
            # Require at least mild oversold for a mean-reversion long
            return 0.0, {}, [f"RSI={rsi:.1f} not oversold (need <{_MR_RSI_MILD_OVERSOLD})"]
    else:
        return 0.0, {}, ["RSI not ready"]

    # ── BB extension ─────────────────────────────────────────────────────────
    if bb_lower is not None and bb_lower > 0:
        dist = (price - bb_lower) / bb_lower   # negative = price below lower BB
        if dist <= 0:
            components["bb_extension"] = 0.30
            notes.append(f"price {dist:.4f} below lower BB")
        elif dist <= _MR_BB_NEAR_PCT:
            components["bb_extension"] = 0.20
            notes.append(f"price {dist:.4f} near lower BB")
        elif dist <= 2 * _MR_BB_NEAR_PCT:
            components["bb_extension"] = 0.10
            notes.append(f"price {dist:.4f} approaching lower BB")
        else:
            notes.append(f"price {dist:.4f} far from lower BB")
    else:
        notes.append("BB lower not available")

    # ── Choppiness (KER) ─────────────────────────────────────────────────────
    if ker is not None:
        if ker < _MR_KER_CHOPPY:
            components["choppiness"] = 0.20
            notes.append(f"KER={ker:.3f} (choppy/ranging)")
        elif ker < 0.5:
            components["choppiness"] = 0.10
            notes.append(f"KER={ker:.3f} (moderate choppiness)")
        else:
            notes.append(f"KER={ker:.3f} (trending, low confidence for mean-rev)")

    # ── VWAP distance: long below VWAP is more extended ──────────────────────
    if vwap > 0:
        if price < vwap:
            vwap_dist = (vwap - price) / vwap
            if vwap_dist >= 0.002:
                components["vwap_dist"] = 0.15
                notes.append(f"price {vwap_dist:.4f} below VWAP (extended)")
            else:
                components["vwap_dist"] = 0.07
                notes.append("price slightly below VWAP")
        else:
            notes.append("price above VWAP (less extended for long mean-rev)")

    score = sum(components.values())
    return min(score, 1.0), components, notes


def _score_mean_reversion_short(mnq, regime):
    components = {}
    notes = []

    price = _safe_last(mnq.get("prices", []))
    if price <= 0:
        return 0.0, components, ["no price data"]

    adx_val, _, _ = _adx_triplet(mnq)
    rsi = _rsi_val(mnq)
    _, bb_upper, _, _, _ = _bb_state(mnq)
    ker_data = mnq.get("ker", [])
    ker  = float(ker_data[-1]) if ker_data and len(ker_data) > 0 else None
    vwap = float(mnq.get("vwap", 0.0))

    if adx_val is not None and adx_val >= _MR_ADX_MAX:
        return 0.0, {}, [f"ADX={adx_val:.1f}>={_MR_ADX_MAX} (trending, skip mean-rev)"]

    # ── RSI extension ─────────────────────────────────────────────────────────
    if rsi is not None:
        if rsi > _MR_RSI_OVERBOUGHT:
            components["rsi"] = 0.35
            notes.append(f"RSI={rsi:.1f} overbought")
        elif rsi > _MR_RSI_MILD_OB:
            components["rsi"] = 0.20
            notes.append(f"RSI={rsi:.1f} mildly overbought")
        else:
            return 0.0, {}, [f"RSI={rsi:.1f} not overbought (need >{_MR_RSI_MILD_OB})"]
    else:
        return 0.0, {}, ["RSI not ready"]

    # ── BB extension ─────────────────────────────────────────────────────────
    if bb_upper is not None and bb_upper > 0:
        dist = (bb_upper - price) / bb_upper   # negative = price above upper BB
        if dist <= 0:
            components["bb_extension"] = 0.30
            notes.append(f"price {-dist:.4f} above upper BB")
        elif dist <= _MR_BB_NEAR_PCT:
            components["bb_extension"] = 0.20
            notes.append(f"price {dist:.4f} near upper BB")
        elif dist <= 2 * _MR_BB_NEAR_PCT:
            components["bb_extension"] = 0.10
            notes.append(f"price approaching upper BB")
        else:
            notes.append("price far from upper BB")
    else:
        notes.append("BB upper not available")

    # ── Choppiness (KER) ─────────────────────────────────────────────────────
    if ker is not None:
        if ker < _MR_KER_CHOPPY:
            components["choppiness"] = 0.20
        elif ker < 0.5:
            components["choppiness"] = 0.10

    # ── VWAP distance ─────────────────────────────────────────────────────────
    if vwap > 0:
        if price > vwap:
            vwap_dist = (price - vwap) / vwap
            if vwap_dist >= 0.002:
                components["vwap_dist"] = 0.15
                notes.append(f"price {vwap_dist:.4f} above VWAP (extended)")
            else:
                components["vwap_dist"] = 0.07
        else:
            notes.append("price below VWAP (less extended for short mean-rev)")

    score = sum(components.values())
    return min(score, 1.0), components, notes


def evaluate_mean_reversion(symbol, mnq, regime, session, vix, config):
    """
    Evaluate both long and short mean-reversion setups for one symbol.
    Returns the best-direction SetupCandidate, or None if neither qualifies.
    """
    threshold = config.MEAN_REVERSION_THRESHOLD
    long_score,  long_comps,  long_notes  = _score_mean_reversion_long(mnq, regime)
    short_score, short_comps, short_notes = _score_mean_reversion_short(mnq, regime)

    direction, score, comps, notes = _pick_direction(
        long_score, long_comps, long_notes,
        short_score, short_comps, short_notes,
        config.MIN_DIRECTION_GAP,
    )
    if direction is None:
        return None

    symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    return SetupCandidate(
        symbol=symbol,
        symbol_name=symbol_name,
        direction=direction,
        setup_type=SETUP_MEAN_REVERSION,
        score=score,
        threshold=threshold,
        long_score=long_score,
        short_score=short_score,
        components=comps,
        regime=regime,
        session=session,
        vix=vix,
        notes=notes,
    )


# =============================================================================
# Setup 3: Breakout / Compression
# =============================================================================
#
# Rationale: enter when price breaks out of a compressed range (narrow BB width)
# with volume surge confirmation.
#
# Required:
#   - BB width below its recent average (compression present)
#   - Price breaks above upper BB (long) or below lower BB (short)
#   - Volume surge: current > 1.5× baseline (partial) or > 2.5× (strong)
#
# Scoring components (max weights):
#   compression : 0.35   how compressed the range was
#   breakout    : 0.40   how far price has broken out
#   volume      : 0.25   volume confirmation strength
# Max total: 1.00
# =============================================================================

_BC_VOL_STRONG  = 2.5
_BC_VOL_PARTIAL = 1.5


def _score_breakout_compression_long(mnq, regime):
    components = {}
    notes = []

    _, bb_upper, bb_lower, bb_width, bb_width_ma = _bb_state(mnq)
    price = _safe_last(mnq.get("prices", []))
    if price <= 0:
        return 0.0, {}, ["no price data"]

    vol_ratio = _vol_ratio(mnq)

    # ── Compression quality ───────────────────────────────────────────────────
    if bb_width is None or bb_width_ma is None or bb_width_ma <= 0:
        return 0.0, {}, ["BB width data not ready"]

    compression_ratio = bb_width / bb_width_ma
    if compression_ratio < 0.70:
        components["compression"] = 0.35
        notes.append(f"BB compressed {compression_ratio:.2f}x mean (strong)")
    elif compression_ratio < 0.85:
        components["compression"] = 0.20
        notes.append(f"BB compressed {compression_ratio:.2f}x mean (moderate)")
    elif compression_ratio < 1.0:
        components["compression"] = 0.10
        notes.append(f"BB slightly compressed {compression_ratio:.2f}x mean")
    else:
        # No compression → breakout/compression setup doesn't apply
        return 0.0, {}, [f"BB not compressed ({compression_ratio:.2f}x mean)"]

    # ── Breakout strength (price above upper BB) ──────────────────────────────
    if bb_upper is None or bb_upper <= 0:
        return 0.0, components, ["BB upper not available"]

    if price > bb_upper:
        overshoot = (price - bb_upper) / bb_upper
        if overshoot > 0.001:
            components["breakout"] = 0.40
            notes.append(f"price {overshoot:.4f} above upper BB (strong breakout)")
        else:
            components["breakout"] = 0.25
            notes.append("price just above upper BB")
    elif price >= bb_upper * 0.999:
        components["breakout"] = 0.15
        notes.append("price at upper BB (approaching breakout)")
    else:
        # Not breaking out upward → long breakout setup doesn't apply
        return 0.0, components, [f"price below upper BB (no long breakout)"]

    # ── Volume confirmation ───────────────────────────────────────────────────
    if vol_ratio is not None:
        if vol_ratio >= _BC_VOL_STRONG:
            components["volume"] = 0.25
            notes.append(f"volume {vol_ratio:.1f}x baseline (strong)")
        elif vol_ratio >= _BC_VOL_PARTIAL:
            components["volume"] = 0.15
            notes.append(f"volume {vol_ratio:.1f}x baseline (moderate)")
        else:
            notes.append(f"volume {vol_ratio:.1f}x baseline (weak)")
    else:
        notes.append("volume data not available")

    score = sum(components.values())
    return min(score, 1.0), components, notes


def _score_breakout_compression_short(mnq, regime):
    components = {}
    notes = []

    _, bb_upper, bb_lower, bb_width, bb_width_ma = _bb_state(mnq)
    price = _safe_last(mnq.get("prices", []))
    if price <= 0:
        return 0.0, {}, ["no price data"]

    vol_ratio = _vol_ratio(mnq)

    # ── Compression quality ───────────────────────────────────────────────────
    if bb_width is None or bb_width_ma is None or bb_width_ma <= 0:
        return 0.0, {}, ["BB width data not ready"]

    compression_ratio = bb_width / bb_width_ma
    if compression_ratio < 0.70:
        components["compression"] = 0.35
        notes.append(f"BB compressed {compression_ratio:.2f}x mean (strong)")
    elif compression_ratio < 0.85:
        components["compression"] = 0.20
        notes.append(f"BB compressed {compression_ratio:.2f}x mean (moderate)")
    elif compression_ratio < 1.0:
        components["compression"] = 0.10
        notes.append(f"BB slightly compressed {compression_ratio:.2f}x mean")
    else:
        return 0.0, {}, [f"BB not compressed ({compression_ratio:.2f}x mean)"]

    # ── Breakout strength (price below lower BB) ──────────────────────────────
    if bb_lower is None or bb_lower <= 0:
        return 0.0, components, ["BB lower not available"]

    if price < bb_lower:
        undershoot = (bb_lower - price) / bb_lower
        if undershoot > 0.001:
            components["breakout"] = 0.40
            notes.append(f"price {undershoot:.4f} below lower BB (strong breakout)")
        else:
            components["breakout"] = 0.25
            notes.append("price just below lower BB")
    elif price <= bb_lower * 1.001:
        components["breakout"] = 0.15
        notes.append("price at lower BB (approaching breakout)")
    else:
        return 0.0, components, ["price above lower BB (no short breakout)"]

    # ── Volume confirmation ───────────────────────────────────────────────────
    if vol_ratio is not None:
        if vol_ratio >= _BC_VOL_STRONG:
            components["volume"] = 0.25
            notes.append(f"volume {vol_ratio:.1f}x baseline (strong)")
        elif vol_ratio >= _BC_VOL_PARTIAL:
            components["volume"] = 0.15
            notes.append(f"volume {vol_ratio:.1f}x baseline (moderate)")
        else:
            notes.append(f"volume {vol_ratio:.1f}x baseline (weak)")
    else:
        notes.append("volume data not available")

    score = sum(components.values())
    return min(score, 1.0), components, notes


def evaluate_breakout_compression(symbol, mnq, regime, session, vix, config):
    """
    Evaluate both long and short breakout/compression setups for one symbol.
    Returns the best-direction SetupCandidate, or None if neither qualifies.
    """
    threshold = config.BREAKOUT_COMPRESSION_THRESHOLD
    long_score,  long_comps,  long_notes  = _score_breakout_compression_long(mnq, regime)
    short_score, short_comps, short_notes = _score_breakout_compression_short(mnq, regime)

    direction, score, comps, notes = _pick_direction(
        long_score, long_comps, long_notes,
        short_score, short_comps, short_notes,
        config.MIN_DIRECTION_GAP,
    )
    if direction is None:
        return None

    symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    return SetupCandidate(
        symbol=symbol,
        symbol_name=symbol_name,
        direction=direction,
        setup_type=SETUP_BREAKOUT_COMPRESSION,
        score=score,
        threshold=threshold,
        long_score=long_score,
        short_score=short_score,
        components=comps,
        regime=regime,
        session=session,
        vix=vix,
        notes=notes,
    )


# =============================================================================
# Top-level: evaluate all enabled setups for one symbol
# =============================================================================

def evaluate_all_setups(symbol, mnq, regime, session, vix, config):
    """
    Evaluate all enabled setup families for a single active contract.

    Returns a list of SetupCandidates (may be empty if no setup fires).
    Each candidate is independent — the best one (by score) will be chosen
    by the portfolio selector, not here.
    """
    candidates = []

    try:
        if config.ENABLE_TREND_PULLBACK:
            c = evaluate_trend_pullback(symbol, mnq, regime, session, vix, config)
            if c is not None:
                candidates.append(c)
    except Exception as e:
        pass   # individual setup errors must not stop other setups

    try:
        if config.ENABLE_MEAN_REVERSION:
            c = evaluate_mean_reversion(symbol, mnq, regime, session, vix, config)
            if c is not None:
                candidates.append(c)
    except Exception as e:
        pass

    try:
        if config.ENABLE_BREAKOUT_COMPRESSION:
            c = evaluate_breakout_compression(symbol, mnq, regime, session, vix, config)
            if c is not None:
                candidates.append(c)
    except Exception as e:
        pass

    return candidates
