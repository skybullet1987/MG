# region imports
# endregion
"""
Machine Gun 3 - Centralized Configuration
==========================================
All tunable parameters in one place.  Defaults are calibrated for a **$2,000
test account** on Kraken – conservative enough for paper/backtest validation,
realistic enough to reflect live trading costs.

Quick-start:
  - Backtest: ``MODE_DEFAULT = "backtest"`` (default) – just click Backtest.
  - Paper:    set ``MODE_DEFAULT = "paper"``.
  - Live:     set ``MODE_DEFAULT = "live"`` **and** review every section below.

$2,000 account design targets
-------------------------------
* Max 3 concurrent positions × $250 cap  → ≤ $750 deployed (37.5 % max exposure).
* 3 % daily loss cap                     → ≤ $60 max intraday drawdown.
* 15 % portfolio drawdown cap            → ≤ $300 from peak before cooling off.
* Minimum entry profit gate accounts for realistic fees + slippage (≥ 1 %).
"""


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
# "backtest" : strict limit-fill, detailed logging, no live-order paths.
# "paper"    : same safety as backtest but warms up against real feed.
# "live"     : enables live-order paths; review ALL risk limits before use.
MODE_DEFAULT = "backtest"

# ---------------------------------------------------------------------------
# Risk controls  (tuned for a ~$2,000 test account)
# ---------------------------------------------------------------------------
# Maximum fraction of portfolio that can be lost in a single calendar day
# before all new entries are paused.  3 % of $2,000 ≈ $60.
MAX_DAILY_LOSS_PCT = 0.03          # 3 %  (was 5 %)

# Hard cap on simultaneously open positions.
# 3 positions × $250 = $750 max deployed capital on a $2,000 account.
MAX_OPEN_POSITIONS = 3             # (was 6)

# Maximum USD exposure per single symbol (entry sizing hard cap).
# ~12.5 % of a $2,000 account per position.
MAX_SYMBOL_EXPOSURE_USD = 250.0    # (was 1500)

# Maximum total portfolio drawdown before triggering a cooldown pause.
# 15 % of $2,000 ≈ $300.
MAX_DRAWDOWN_LIMIT = 0.15          # 15 %  (was 25 %)

# Hours to pause new entries after a drawdown event fires.
DRAWDOWN_COOLDOWN_HOURS = 6

# Consecutive losing trades before a short pause + size halving.
MAX_CONSECUTIVE_LOSSES = 4         # (was 5)

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

# Maximum retry attempts for a failed exit order before escalating to a
# force-market recovery liquidation.
MAX_EXIT_RETRIES = 3

# Exponential back-off base (seconds) between retry attempts.
RETRY_BACKOFF_BASE_SECONDS = 30

# Max concurrent open orders outstanding at any time.
MAX_CONCURRENT_OPEN_ORDERS = 2

# Fraction of available cash that may be reserved by open buy orders before
# new entry candidates are skipped.  0.9 = allow up to 90 % reserved,
# preventing only over-committed situations (tuned for small accounts,
# ref: PR #46 optimisation for $120 live account).
OPEN_ORDERS_CASH_THRESHOLD = 0.90  # (was 0.5)

# ---------------------------------------------------------------------------
# Strategy direction toggles
# ---------------------------------------------------------------------------
# Enable long (buy) entries.
LONG_ENABLED = True

# Enable short (sell) entries.  Disabled by default until the short-scoring
# pipeline has been validated against historical data.
SHORT_ENABLED = False

# ---------------------------------------------------------------------------
# Backtest realism knobs  (key levers for closing the backtest-vs-live gap)
# ---------------------------------------------------------------------------
# Assumed round-trip fee (fraction).
# Kraken blended rate: ~0.25 % maker + ~0.40 % taker → 0.50 % RT is realistic.
FEE_ASSUMPTION_RT = 0.0050         # 0.50 % round trip

# Additional slippage buffer on top of fee assumption.  Raised to 0.20 % to
# better capture real spread costs and market-impact on small orders.
SLIPPAGE_BUFFER = 0.002            # 0.20 %  (was 0.10 %)

# When True, limit orders in backtest are only filled if the market price
# trades through the limit price (strict fill semantics).
STRICT_LIMIT_FILL = True

# Minimum expected net profit (after fees + slippage) required to enter.
# 1.0 % gate on a $250 position ≈ $2.50 minimum gross profit required.
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
