# region imports
# endregion
"""
Machine Gun 3 - Centralized Configuration
==========================================
All tunable parameters in one place. Safe, conservative defaults suitable for
backtesting and paper trading. Live trading must be explicitly enabled via
the ``mode`` parameter.

Quick-start:
  - Backtest: set ``mode = "backtest"`` (default) and run via QuantConnect.
  - Paper:    set ``mode = "paper"``.
  - Live:     set ``mode = "live"`` **and** review every section below before deploying.
"""


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
# "backtest" : paper-safe defaults, stricter limit-fill, detailed logging.
# "paper"    : same safety as backtest but warms up against real feed.
# "live"     : enables live-order paths; review all risk limits carefully.
MODE_DEFAULT = "backtest"

# ---------------------------------------------------------------------------
# Risk controls
# ---------------------------------------------------------------------------
# Maximum fraction of portfolio that can be lost in a single calendar day
# before all new entries are paused.
MAX_DAILY_LOSS_PCT = 0.05          # 5 %

# Hard cap on simultaneously open positions.
MAX_OPEN_POSITIONS = 6

# Maximum USD exposure per single symbol (applies to entry sizing).
MAX_SYMBOL_EXPOSURE_USD = 1500.0

# Maximum total portfolio drawdown before triggering a cooldown pause.
MAX_DRAWDOWN_LIMIT = 0.25          # 25 %

# Hours to pause new entries after a drawdown event fires.
DRAWDOWN_COOLDOWN_HOURS = 6

# Consecutive losing trades before a short pause + size halving.
MAX_CONSECUTIVE_LOSSES = 5

# ---------------------------------------------------------------------------
# Execution controls
# ---------------------------------------------------------------------------
# Seconds before a stale open order is cancelled in backtest/paper mode.
ORDER_TIMEOUT_SECONDS = 30

# Seconds before a stale open order is cancelled in live mode.
LIVE_ORDER_TIMEOUT_SECONDS = 60

# Minimum seconds between consecutive replace/re-submit attempts for the
# same symbol (throttle to avoid order spam).
REPLACE_THROTTLE_SECONDS = 60

# Maximum retry attempts for a failed exit order before giving up and
# cleaning up local tracking.
MAX_EXIT_RETRIES = 3

# Exponential back-off base (seconds) between retry attempts.
RETRY_BACKOFF_BASE_SECONDS = 30

# Max concurrent open orders outstanding at any time.
MAX_CONCURRENT_OPEN_ORDERS = 2

# ---------------------------------------------------------------------------
# Strategy direction toggles
# ---------------------------------------------------------------------------
# Enable long (buy) entries.
LONG_ENABLED = True

# Enable short (sell) entries.  Disabled by default until the short-scoring
# pipeline has been validated against historical data.
SHORT_ENABLED = False

# ---------------------------------------------------------------------------
# Backtest realism knobs
# ---------------------------------------------------------------------------
# Assumed round-trip fee (fraction).  Kraken maker/taker: ~0.25 % / 0.40 %.
FEE_ASSUMPTION_RT = 0.0050         # 0.50 % round trip (conservative)

# Additional slippage buffer applied on top of fee assumption.
SLIPPAGE_BUFFER = 0.001            # 0.10 %

# When True, limit orders in backtest are only filled if the market price
# trades through the limit price (strict fill semantics).  Set False for
# looser backtest optimism (not recommended for validation).
STRICT_LIMIT_FILL = True

# Minimum expected net profit (after fees + slippage) required to enter.
MIN_EXPECTED_PROFIT_PCT = 0.010    # 1.0 %

# ---------------------------------------------------------------------------
# Kraken-specific preparation
# ---------------------------------------------------------------------------
# Tick / price precision normalization: number of decimal places for prices.
# Set to None to use exchange-reported precision.
KRAKEN_PRICE_PRECISION = None

# Lot-size normalization: minimum lot increment to round order quantities to.
# Set to None to use exchange-reported lot size.
KRAKEN_LOT_SIZE = None

# How often (seconds) to run a full local-vs-exchange reconciliation check.
# Lower values catch drift sooner but add API load in live mode.
RECONCILIATION_CADENCE_SECONDS = 300   # 5 min

# Number of reconciliation cycles to skip in backtest (0 = always run).
# In backtest the reconciliation path is a no-op for the exchange call, but
# the local-state check is still useful.
RECONCILIATION_BACKTEST_SKIP = 0

# Maximum tolerated fractional mismatch between local and exchange holdings
# before a reconciliation alert is raised.
PORTFOLIO_MISMATCH_THRESHOLD = 0.10

# Minimum USD mismatch size before the alert fires (avoids dust noise).
PORTFOLIO_MISMATCH_MIN_DOLLARS = 1.00

# Cooldown (seconds) between consecutive mismatch warnings for the same symbol.
PORTFOLIO_MISMATCH_COOLDOWN_SECONDS = 3600
