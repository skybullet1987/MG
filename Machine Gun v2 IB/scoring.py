"""
scoring.py — Machine Gun v2 IB: Setup candidates, constants, and evaluation logic

This module is the consolidated signal layer for the setup-driven pipeline.
It replaces the legacy blended composite-score workflow with three explicit,
separately-attributable setup families.

Contents (in order)
-------------------
  1. Setup family identifiers and rejection-reason constants
  2. SetupCandidate — structured data object produced by each setup evaluator
  3. Setup evaluation logic:
       - trend_pullback:        enter with the trend after a pullback
       - mean_reversion:        enter counter-trend in choppy / ranging markets
       - breakout_compression:  enter after BB squeeze with volume surge
  4. evaluate_all_setups()     top-level entry point used by Rebalance()
  5. MicroScalpEngine (legacy) — preserved for reference / comparison only

Design principles
-----------------
* Each evaluator produces a SetupCandidate with full attribution metadata.
* Directional selection uses a hard minimum gap — never by elimination.
* Thresholds are per-setup and explicitly named; no semantic ambiguity.
* Kelly sizing and complex adaptive logic live in execution.py / config.py.
"""

# region imports
from AlgorithmImports import *
import numpy as np
from collections import defaultdict
# endregion

# ─────────────────────────────────────────────────────────────────────────────
# Setup family identifiers
# ─────────────────────────────────────────────────────────────────────────────
SETUP_TREND_PULLBACK        = "trend_pullback"
SETUP_MEAN_REVERSION        = "mean_reversion"
SETUP_BREAKOUT_COMPRESSION  = "breakout_compression"

ALL_SETUP_TYPES = [
    SETUP_TREND_PULLBACK,
    SETUP_MEAN_REVERSION,
    SETUP_BREAKOUT_COMPRESSION,
]

# ─────────────────────────────────────────────────────────────────────────────
# Rejection reason constants  (used in DiagnosticsLogger.rejection_counts)
# ─────────────────────────────────────────────────────────────────────────────
REJECT_SCORE_TOO_LOW          = "score_below_threshold"
REJECT_DIRECTION_GAP_TOO_LOW  = "direction_gap_too_small"
REJECT_ALREADY_INVESTED       = "already_invested"
REJECT_OPEN_ORDERS            = "has_open_orders"
REJECT_COOLDOWN               = "in_cooldown"
REJECT_PORTFOLIO_CAP          = "portfolio_cap_reached"
REJECT_DAILY_TRADE_LIMIT      = "daily_trade_limit"
REJECT_SYMBOL_TRADE_LIMIT     = "symbol_trade_limit"
REJECT_LOW_EXPECTED_MOVE      = "expected_move_too_small"
REJECT_MARGIN_INSUFFICIENT    = "margin_insufficient"
REJECT_VIX_EXTREME            = "vix_extreme"
REJECT_NOT_RTH                = "not_rth"
REJECT_DAILY_LOSS_EXCEEDED    = "daily_loss_exceeded"
REJECT_DATA_NOT_READY         = "data_not_ready"
REJECT_ZERO_CONTRACTS         = "zero_contracts_sized"
REJECT_RATE_LIMITED           = "rate_limited"


class SetupCandidate:
    """
    Structured candidate produced by a setup evaluator.

    All fields are set at construction time.  The portfolio selector reads them
    to decide which candidates to act on.  The DiagnosticsLogger stores them for
    trade attribution and PnL breakdown.

    Key fields
    ----------
    symbol        : QC Symbol object for the active futures contract
    symbol_name   : Human-readable ticker string (e.g. "MNQ XYZW")
    direction     : 1 = long, -1 = short
    setup_type    : one of SETUP_* constants above
    score         : aggregate signal score, 0.0–1.0
    threshold     : minimum score required for a valid entry
    long_score    : raw long-side score (for debugging direction choice)
    short_score   : raw short-side score
    components    : dict mapping signal name → individual contribution
    regime        : VIX-derived regime: "bull" | "sideways" | "bear"
    session       : "rth" | "eth"
    vix           : VIX value at evaluation time
    stop_pct      : suggested stop distance as fraction of price (optional)
    target_pct    : suggested take-profit distance (optional)
    notes         : list of debug notes from the evaluator
    eval_time     : datetime when the candidate was evaluated
    """

    __slots__ = (
        "symbol", "symbol_name", "direction", "setup_type",
        "score", "threshold",
        "long_score", "short_score", "components",
        "regime", "session", "vix",
        "stop_pct", "target_pct", "hold_hours",
        "notes", "eval_time",
    )

    def __init__(
        self,
        symbol,
        symbol_name,
        direction,
        setup_type,
        score,
        threshold,
        long_score=0.0,
        short_score=0.0,
        components=None,
        regime="unknown",
        session="rth",
        vix=20.0,
        stop_pct=None,
        target_pct=None,
        hold_hours=None,
        notes=None,
        eval_time=None,
    ):
        self.symbol       = symbol
        self.symbol_name  = symbol_name
        self.direction    = direction
        self.setup_type   = setup_type
        self.score        = score
        self.threshold    = threshold
        self.long_score   = long_score
        self.short_score  = short_score
        self.components   = components if components is not None else {}
        self.regime       = regime
        self.session      = session
        self.vix          = vix
        self.stop_pct     = stop_pct
        self.target_pct   = target_pct
        self.hold_hours   = hold_hours
        self.notes        = notes if notes is not None else []
        self.eval_time    = eval_time

    @property
    def is_valid(self):
        """True when score >= threshold — candidate qualifies for entry."""
        return self.score >= self.threshold

    def summary(self):
        """One-line human-readable summary for Debug logging."""
        dir_str = "LONG" if self.direction == 1 else "SHORT"
        return (
            f"{self.symbol_name} [{self.setup_type}] {dir_str} "
            f"score={self.score:.3f} thresh={self.threshold:.3f} "
            f"long={self.long_score:.3f} short={self.short_score:.3f} "
            f"regime={self.regime} session={self.session} vix={self.vix:.1f}"
        )

    def components_str(self):
        """Compact component breakdown string for Debug logging."""
        return " ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(self.components.items())
            if v != 0.0
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

def evaluate_all_setups(symbol, mnq, regime, session, vix, config, log_fn=None):
    """
    Evaluate all enabled setup families for a single active contract.

    Returns a list of SetupCandidates (may be empty if no setup fires).
    Each candidate is independent — the best one (by score) will be chosen
    by the portfolio selector, not here.

    Parameters
    ----------
    log_fn : callable, optional
        A function accepting a single string argument (e.g. algo.Debug).
        When provided, any exception raised inside an individual setup evaluator
        is logged rather than silently discarded.  Individual setup errors never
        block the evaluation of the other setups.
    """
    candidates = []
    sym_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)

    try:
        if config.ENABLE_TREND_PULLBACK:
            c = evaluate_trend_pullback(symbol, mnq, regime, session, vix, config)
            if c is not None:
                candidates.append(c)
    except Exception as e:
        # Log but continue — one setup error must not block the others
        if log_fn is not None:
            log_fn("setup error [trend_pullback] {}: {}: {}".format(sym_name, type(e).__name__, e))

    try:
        if config.ENABLE_MEAN_REVERSION:
            c = evaluate_mean_reversion(symbol, mnq, regime, session, vix, config)
            if c is not None:
                candidates.append(c)
    except Exception as e:
        if log_fn is not None:
            log_fn("setup error [mean_reversion] {}: {}: {}".format(sym_name, type(e).__name__, e))

    try:
        if config.ENABLE_BREAKOUT_COMPRESSION:
            c = evaluate_breakout_compression(symbol, mnq, regime, session, vix, config)
            if c is not None:
                candidates.append(c)
    except Exception as e:
        if log_fn is not None:
            log_fn("setup error [breakout_compression] {}: {}: {}".format(sym_name, type(e).__name__, e))

    return candidates


# =============================================================================
# LEGACY: MicroScalpEngine (reference only — not used in the trading pipeline)
# =============================================================================

class MicroScalpEngine:
    """
    LEGACY: Micro-Scalping Signal Engine - v7.3.0 (MNQ/Futures)

    This class is no longer used in the main trading pipeline.  It is kept
    for reference and backward compatibility.  See the setup evaluators in
    this file (evaluate_trend_pullback, etc.) for the current architecture.

    Original docstring below:
    -------------------------

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

        The portfolio-wide cap (max_portfolio_contracts) is enforced in Rebalance()
        before this function is called; here we additionally guard against violating
        the per-order margin limit relative to actual remaining margin.
        """
        available_margin = self.algo.Portfolio.MarginRemaining
        # Use a conservative per-contract margin estimate that covers any of our
        # three instruments (MNQ ~$1,300, MGC ~$1,000, M2K ~$700) with a buffer.
        margin_per_contract = 1400  # conservative buffer above highest (MNQ ~$1,300)

        max_by_margin = int(available_margin / margin_per_contract) if margin_per_contract > 0 else 0
        max_contracts = getattr(self.algo, 'max_contracts', 2)
        # Also respect the portfolio-wide cap so sizing never exceeds overall limit
        portfolio_cap = getattr(self.algo, 'max_portfolio_contracts', max_contracts)
        max_contracts = min(max_contracts, portfolio_cap)

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

