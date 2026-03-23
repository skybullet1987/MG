"""
config.py — Machine Gun v2 IB: research-first configuration

Default mode is "research" with:
  - fixed 1-contract sizing (Kelly disabled)
  - configurable regime thresholds (not hard veto, except extreme VIX)
  - all three setup families enabled
  - diagnostics/attribution enabled

Switch MODE to "live" to enable adaptive sizing and the full risk-gate path.
"""


class MGConfig:
    # ─────────────────────────────────────────────────────────────────────────
    # Strategy mode
    # ─────────────────────────────────────────────────────────────────────────
    # "research" — fixed-size, Kelly off, looser regime gates.
    #              Use this for backtesting to keep causality clear.
    # "live"     — adaptive sizing, Kelly on, tighter safety gates.
    MODE = "research"

    # ─────────────────────────────────────────────────────────────────────────
    # Capital & broker
    # ─────────────────────────────────────────────────────────────────────────
    START_CASH = 3_000
    # Conservative per-contract margin estimate covering MNQ (~$1,300),
    # MGC (~$1,000), and M2K (~$700) with a buffer.
    MARGIN_PER_CONTRACT = 1_400

    # ─────────────────────────────────────────────────────────────────────────
    # Setup families — toggle individually for ablation studies
    # ─────────────────────────────────────────────────────────────────────────
    ENABLE_TREND_PULLBACK        = True
    ENABLE_MEAN_REVERSION        = True
    ENABLE_BREAKOUT_COMPRESSION  = True

    # ─────────────────────────────────────────────────────────────────────────
    # Entry thresholds — per-setup, explicit semantics:
    #   score >= THRESHOLD  ⟹  candidate is valid for entry
    #
    # These thresholds are deliberately separate so each setup can be tuned
    # independently without cross-contamination from the other setups.
    # ─────────────────────────────────────────────────────────────────────────
    TREND_PULLBACK_THRESHOLD       = 0.55
    MEAN_REVERSION_THRESHOLD       = 0.50
    BREAKOUT_COMPRESSION_THRESHOLD = 0.55

    # Minimum absolute gap between long_score and short_score required before
    # we commit to a direction.  Prevents entering on ambiguous mixed signals.
    MIN_DIRECTION_GAP = 0.10

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing  (research-first: fixed 1 contract by default)
    # ─────────────────────────────────────────────────────────────────────────
    DEFAULT_CONTRACTS        = 1   # fixed size in research mode
    MAX_CONTRACTS_PER_SYMBOL = 2   # hard cap per instrument
    MAX_PORTFOLIO_CONTRACTS  = 2   # portfolio-wide hard cap

    # Kelly criterion — DISABLED by default in research mode.
    # Enabling Kelly on small sample windows amplifies noise rather than edge.
    # Set True only in MODE="live" after validating positive expectancy.
    KELLY_ENABLED = False

    # Volatility targeting — DISABLED by default in research mode.
    # Adaptive vol-sizing obscures whether the entry logic itself has real edge.
    VOL_TARGETING_ENABLED             = False
    TARGET_POSITION_ANN_VOL           = 0.35
    VOL_TARGETING_REDUCTION_THRESHOLD = 0.5   # reduce contracts when vol_scalar < this

    # ─────────────────────────────────────────────────────────────────────────
    # Take-profit / stop-loss
    # ─────────────────────────────────────────────────────────────────────────
    QUICK_TAKE_PROFIT = 0.0025   # 0.25%  (~50 NQ points)
    TIGHT_STOP_LOSS   = 0.0020   # 0.20%  (~40 NQ points)
    ATR_TP_MULT       = 2.5      # ATR multiplier for TP distance
    ATR_SL_MULT       = 1.5      # ATR multiplier for SL distance
    MIN_TP_SL_RATIO   = 1.5      # TP must be >= 1.5 × SL distance

    # ─────────────────────────────────────────────────────────────────────────
    # Time stops
    # ─────────────────────────────────────────────────────────────────────────
    TIME_STOP_HOURS      = 1.5
    STALE_POSITION_HOURS = 4.0

    # ─────────────────────────────────────────────────────────────────────────
    # Regime handling
    #
    # Instead of a hard VIX veto at one threshold, we use a graduated approach:
    #   - VIX_EXTREME_THRESHOLD: hard block (market is genuinely dislocated)
    #   - VIX_HIGH_THRESHOLD: optional size reduction (configurable)
    #   - VIX_ELEVATED_THRESHOLD: optional mild size reduction
    #
    # In research mode both size-reduction values default to 0 so VIX only
    # triggers the extreme block.  In live mode set them to 1 if desired.
    # ─────────────────────────────────────────────────────────────────────────
    VIX_ELEVATED_THRESHOLD     = 20.0   # above this → consider mild reduction
    VIX_HIGH_THRESHOLD         = 25.0   # above this → consider larger reduction
    VIX_EXTREME_THRESHOLD      = 35.0   # above this → block all new entries

    VIX_ELEVATED_SIZE_REDUCTION = 0     # contracts to subtract at elevated VIX
    VIX_HIGH_SIZE_REDUCTION     = 0     # contracts to subtract at high VIX

    # ─────────────────────────────────────────────────────────────────────────
    # Daily / symbol trade limits
    # ─────────────────────────────────────────────────────────────────────────
    MAX_DAILY_TRADES          = 20
    MAX_SYMBOL_TRADES_PER_DAY = 5
    DAILY_LOSS_LIMIT_PCT      = 0.03   # 3% daily portfolio drawdown → halt

    # ─────────────────────────────────────────────────────────────────────────
    # Session filter
    # ─────────────────────────────────────────────────────────────────────────
    RTH_ONLY = True   # only enter new trades during Regular Trading Hours

    # ─────────────────────────────────────────────────────────────────────────
    # Minimum expected profit (entry quality gate)
    # The ATR-projected move must exceed this to ensure the trade has room
    # to breathe and cover fees+slippage.
    # ─────────────────────────────────────────────────────────────────────────
    MIN_EXPECTED_PROFIT_PCT = 0.0015   # 0.15%

    # ─────────────────────────────────────────────────────────────────────────
    # Cooldowns
    # ─────────────────────────────────────────────────────────────────────────
    INVALID_ENTRY_COOLDOWN_MINUTES = 30
    EXIT_COOLDOWN_HOURS            = 0.5
    CANCEL_COOLDOWN_MINUTES        = 1

    # ─────────────────────────────────────────────────────────────────────────
    # Indicator lookback periods
    # ─────────────────────────────────────────────────────────────────────────
    ULTRA_SHORT_PERIOD = 3
    SHORT_PERIOD       = 6
    MEDIUM_PERIOD      = 12
    LOOKBACK           = 48
