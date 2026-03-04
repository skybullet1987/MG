# Machine Gun 3 (MG3)

> **Status:** Backtest / Paper-safe.  Live trading is explicitly disabled by default.

---

## Purpose and Relation to MG2

Machine Gun 3 is a direct successor to **Machine Gun 2 (v7.1.0)**.  MG2 is left
untouched — MG3 was created by copying the MG2 folder and layering the following
improvements on top:

| Area | What changed in MG3 |
|---|---|
| Configuration | Single `config.py` – all parameters in one place with conservative defaults |
| Safety | `mode` toggle (`backtest` / `paper` / `live`); live mode must be explicitly set |
| Risk | `MAX_DAILY_LOSS_PCT` guard added; positions, drawdown and loss limits sourced from config |
| State machine | Position lifecycle states: `flat → entering → open → exiting → flat` |
| Reconciliation | `ReconcilePositions()` hook compares local state vs brokerage holdings |
| Backtest metrics | Cancel-to-fill ratio, invalid order count, PnL by exit tag – all logged at end of run |
| Kraken prep | Price/lot normalisation hooks, reconciliation cadence, retry back-off wired to config |
| Direction toggles | Longs enabled, **shorts disabled by default** until short pipeline is validated |

---

## How to Run Backtests

### 1 – QuantConnect Cloud (recommended)

1. Create a new **Algorithm** project.
2. Upload all five files: `main.py`, `execution.py`, `scoring.py`, `alt_data.py`,
   `config.py`.
3. Set the back-test date range and starting capital in `main.py`
   (`SetStartDate` / `SetCash`).
4. Click **Backtest**.

The algorithm starts in `backtest` mode by default (set in `config.py` →
`MODE_DEFAULT = "backtest"`).

### 2 – QuantConnect CLI (lean CLI)

```bash
# Install Lean CLI if not already installed
pip install lean

# Pull the project
lean project-create mg3

# Copy files into the project folder, then run
lean backtest mg3
```

Full Lean CLI documentation: https://www.lean.io/docs/v2/lean-cli

### 3 – Reviewing backtest output

At the end of every backtest the algorithm prints an **MG3 Backtest Metrics** block:

```
=== MG3 BACKTEST METRICS ===
Cancel-to-fill ratio: 0.42 (21 cancels / 50 fills)
Invalid orders: 3
PnL by exit tag:
  Stop Loss                      n=  18  avg=-0.026  wr= 0%
  Take Profit                    n=  12  avg=+0.082  wr=100%
  Trailing Stop                  n=   8  avg=+0.031  wr=88%
  Time Stop                      n=   5  avg=-0.004  wr=20%
  Stagnation Exit                n=   4  avg=-0.001  wr=25%
  ...
=== END MG3 METRICS ===
```

Use these numbers to validate signal quality before moving to paper or live:

- **Cancel-to-fill ratio** > 3 → limit prices too aggressive; widen entry tolerance.
- **Invalid orders** > 5 % of fills → lot-size or notional issues; check `KRAKEN_LOT_SIZE`.
- **"Stop Loss" avg pnl** should be less negative than `tight_stop_loss`; if not, slippage
  is eating into the stop.

---

## Key Config Parameters and Safe Defaults

All parameters live in `config.py`.  Override values by editing that file **or** by passing
QuantConnect algorithm parameters (the strategy reads them via `GetParameter()`).

| Parameter | Default | Purpose |
|---|---|---|
| `MODE_DEFAULT` | `"backtest"` | Execution mode; must be `"live"` for live trading |
| `MAX_DAILY_LOSS_PCT` | `0.05` | Pause entries when daily drawdown ≥ 5 % |
| `MAX_OPEN_POSITIONS` | `6` | Hard cap on simultaneous open positions |
| `MAX_SYMBOL_EXPOSURE_USD` | `1500` | Max USD exposure per single symbol |
| `MAX_DRAWDOWN_LIMIT` | `0.25` | Portfolio drawdown limit before cooldown |
| `ORDER_TIMEOUT_SECONDS` | `30` | Stale order timeout in backtest/paper |
| `LIVE_ORDER_TIMEOUT_SECONDS` | `60` | Stale order timeout in live mode |
| `REPLACE_THROTTLE_SECONDS` | `60` | Min seconds between replace/re-submit for same symbol |
| `MAX_EXIT_RETRIES` | `3` | Max retries for a failed exit before cleanup |
| `LONG_ENABLED` | `True` | Enable long entries |
| `SHORT_ENABLED` | `False` | Shorts disabled – enable only after short-pipeline validation |
| `FEE_ASSUMPTION_RT` | `0.0050` | Round-trip fee assumption (0.50 %) |
| `SLIPPAGE_BUFFER` | `0.001` | Extra slippage buffer on top of fee assumption |
| `STRICT_LIMIT_FILL` | `True` | Limit orders only fill when market trades through the price |
| `MIN_EXPECTED_PROFIT_PCT` | `0.010` | Min net profit (after fees + slippage) required to enter |
| `KRAKEN_PRICE_PRECISION` | `None` | Price decimal precision; None = exchange default |
| `KRAKEN_LOT_SIZE` | `None` | Lot-size increment; None = exchange default |
| `RECONCILIATION_CADENCE_SECONDS` | `300` | How often to run local-vs-exchange state check |

---

## Phased Rollout

### Phase 1 – Backtest validation (before any real money)

1. Set `MODE_DEFAULT = "backtest"` in `config.py`.
2. Run a minimum 3-month backtest on Kraken data.
3. Review **MG3 Backtest Metrics** at the end of the run:
   - Cancel-to-fill ratio < 2
   - Invalid orders < 5 % of fills
   - Positive expectancy per exit tag in the direction you expect
4. Cross-check equity curve, max drawdown, and Sharpe ratio.

### Phase 2 – Paper / live-sim

1. Change `MODE_DEFAULT = "paper"` (or keep `"backtest"` and use QC Paper Trading).
2. Let the strategy run for at least one week.
3. Verify order logs match expected fill/cancel rates from Phase 1.
4. Monitor reconciliation logs (`[RECONCILE]` prefix) – phantom positions or missed states
   indicate an integration issue that must be fixed before live.

### Phase 3 – Micro live

1. Set `MODE_DEFAULT = "live"` in `config.py`.
2. Start with minimum capital (e.g. $25–$50) and `MAX_OPEN_POSITIONS = 2`.
3. Watch for Kraken-specific issues:
   - `KRAKEN_PRICE_PRECISION` / `KRAKEN_LOT_SIZE` may need tuning for illiquid pairs.
   - Monitor rate-limit messages in the log; adjust `rate_limit_cooldown_minutes` if needed.
4. Only scale capital after at least 20 live trades with results consistent with Phase 1.

### Phase 4 – Full deployment

- Gradually raise `MAX_OPEN_POSITIONS` and `MAX_SYMBOL_EXPOSURE_USD` as confidence grows.
- Keep `MAX_DAILY_LOSS_PCT = 0.05` (or tighten) as a permanent safety rail.

---

## Safety Posture

- **Live trading is off by default.**  You must set `mode = "live"` explicitly.
- If the mode is not `"live"` but `LiveMode=True` in QuantConnect, the algorithm will
  call `self.Error(...)` and refuse to trade.
- All risk limits are sourced from `config.py` so changes are auditable and reversible.
- Short positions are disabled until short-signal validation is completed.
