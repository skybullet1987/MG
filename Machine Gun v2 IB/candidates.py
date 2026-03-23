"""
candidates.py — Machine Gun v2 IB: Setup candidate types and constants

SetupCandidate is the central data object produced by each setup evaluator
and consumed by the portfolio selection / order-placement stage.

Every trade originates from a SetupCandidate, so trade attribution is always
available: what setup fired, why, what happened.

Rejection reason constants are shared by Rebalance() and DiagnosticsLogger
so rejection counts are consistent across the codebase.
"""

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
