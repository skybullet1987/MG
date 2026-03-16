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
| Risk | `MAX_DAILY_LOSS_PCT` guard; positions, drawdown and loss limits from config |
| State machine | Lifecycle states: `flat → entering → open → exiting → recovering → flat` |
| Failed-exit fix | Failed exits no longer wipe tracking data; retries use the original entry price |
| Recovery path | After `MAX_EXIT_RETRIES` failures a position escalates to `recovering` and is force-market liquidated |
| Reconciliation | `ReconcilePositions()` compares local state vs brokerage; force-exits stuck recovering positions |
| Backtest metrics | Cancel-to-fill ratio, invalid orders, avg hold time, peak positions, PnL by exit tag |
| Realism | Slippage base raised to 0.25 %, daily order cap removed, volume thresholds tuned for small accounts |
| $2 k defaults | Max 3 positions × $250 cap; 3 % daily loss guard; 15 % drawdown limit |
| Direction toggles | Longs enabled, **shorts disabled by default** until short pipeline is validated |
| Small account | Auto-scaling for accounts < $500 (e.g. $120); see Small-account guide below |
| QC architecture | Single `QCAlgorithm` subclass — no mixin inheritance, no dynamic injection |

---

## Architecture Notes

### Single-class design (v8.1.0+)

QuantConnect uses **PythonNet** to bridge Python and C#. PythonNet has two
hard restrictions that previous MG3 drafts hit:

| Error | Root cause |
|---|---|
| `cannot use multiple inheritance with managed classes` | Class inherited from both Python mixins and `QCAlgorithm` |
| `attribute is read-only` | `setattr()` was used to graft mixin methods onto the class at load time |

The fix is simple: one concrete class, one base class.

```python
class SimplifiedCryptoStrategy(QCAlgorithm):
    # All logic lives here directly — no mixin inheritance, no setattr injection
```

All logic previously split across `app.py`, `data_layer.py`, `orchestration.py`,
`exit_handler.py`, and `reporting.py` is now inlined as methods of
`SimplifiedCryptoStrategy`.  The stateless helper modules (`execution.py`,
`scoring.py`, `config.py`, `config_loader.py`) remain as separate files.

### Required files (5 files total)

| File | Role |
|---|---|
| `main.py` | **Single-class entrypoint** — `Initialize`, all trading logic, all event handlers |
| `config.py` | All tunable parameters |
| `config_loader.py` | Config validation helpers |
| `execution.py` | Shared order-execution utilities (stateless functions) |
| `scoring.py` | `MicroScalpEngine` signal calculations (stateless) |

**Only these 5 files are needed.**  The old mixin files (`app.py`, `data_layer.py`,
`orchestration.py`, `exit_handler.py`, `reporting.py`) have been deleted — do not
upload them to your QC project.

---

## How to Run Backtests

### 1 – QuantConnect Cloud (recommended)

1. Create a new **Algorithm** project in the QC Cloud IDE.
2. Upload **all five module files** into that project.

   | File | Role |
   |---|---|
   | `main.py` | Single-class entrypoint – all logic |
   | `config.py` | All tunable parameters |
   | `config_loader.py` | Config validation helpers |
   | `execution.py` | Shared order-execution utilities |
   | `scoring.py` | `MicroScalpEngine` signal calculations |

3. Set the back-test date range and starting capital in `main.py`
   (`SetStartDate` / `SetCash`).  Default: `2025-01-01`, `$2,000`.
4. Click **Backtest**.

The algorithm starts in `backtest` mode by default (set in `config.py` →
`MODE_DEFAULT = "backtest"`).

### 2 – QuantConnect CLI (lean CLI)

```bash
# Install Lean CLI if not already installed
pip install lean

# Create a new project and copy all .py files into it
lean project-create mg3
cp "Machine Gun 3"/*.py mg3/

# Run a backtest
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
| `MAX_DAILY_LOSS_PCT` | `0.03` | Pause entries when daily loss ≥ 3 % (~$60 on $2 k) |
| `MAX_OPEN_POSITIONS` | `3` | Hard cap on simultaneous open positions |
| `MAX_SYMBOL_EXPOSURE_USD` | `250` | Max USD per single symbol (~12.5 % of $2 k) |
| `MAX_DRAWDOWN_LIMIT` | `0.15` | Portfolio drawdown limit before cooldown (was 25 %) |
| `OPEN_ORDERS_CASH_THRESHOLD` | `0.90` | Max fraction of cash reserved by open buy orders |
| `ORDER_TIMEOUT_SECONDS` | `30` | Stale order timeout in backtest/paper |
| `LIVE_ORDER_TIMEOUT_SECONDS` | `60` | Stale order timeout in live mode |
| `REPLACE_THROTTLE_SECONDS` | `60` | Min seconds between replace/re-submit for same symbol |
| `MAX_EXIT_RETRIES` | `3` | Failed-exit retries before force-market recovery |
| `LONG_ENABLED` | `True` | Enable long entries |
| `SHORT_ENABLED` | `False` | Shorts disabled – enable only after short-pipeline validation |
| `FEE_ASSUMPTION_RT` | `0.0050` | Round-trip fee assumption (0.50 %) |
| `SLIPPAGE_BUFFER` | `0.002` | Slippage buffer on top of fee (0.20 %, raised for realism) |
| `STRICT_LIMIT_FILL` | `True` | Limit orders only fill when market trades through the price |
| `MIN_EXPECTED_PROFIT_PCT` | `0.010` | Min net profit (after fees + slippage) required to enter |
| `KRAKEN_PRICE_PRECISION` | `None` | Price decimal precision; None = exchange default |
| `KRAKEN_LOT_SIZE` | `None` | Lot-size increment; None = exchange default |
| `RECONCILIATION_CADENCE_SECONDS` | `300` | How often to run local-vs-exchange state check |
| `SMALL_ACCOUNT_MODE` | `False` | Force small-account scaling (auto-detect when cash < 500) |
| `SMALL_ACCOUNT_THRESHOLD_USD` | `500` | Cash threshold that triggers small-account mode |
| `SMALL_ACCOUNT_MAX_POSITIONS` | `2` | Max positions in small-account mode |
| `SMALL_ACCOUNT_MAX_EXPOSURE_USD` | `40` | Max USD per position in small-account mode |

### $2,000 test-account quick reference

```python
# main.py defaults
SetCash(2000)
SetStartDate(2025, 1, 1)
position_size_pct = 0.20          # high-vol size cap; effective trade ≈ $150–$250

# config.py defaults (do not change for initial validation)
MAX_OPEN_POSITIONS      = 3       # 3 × $250 = $750 max deployed
MAX_SYMBOL_EXPOSURE_USD = 250.0   # hard cap per position
MAX_DAILY_LOSS_PCT      = 0.03    # $60 daily stop
MAX_DRAWDOWN_LIMIT      = 0.15    # $300 drawdown limit

# Volume thresholds (tuned for small accounts – PR #47/#49)
min_dollar_volume_usd   = 5000    # bar-level liquidity filter
min_volume_usd          = 10000   # universe inclusion filter
max_daily_trades        = 24000   # effectively unlimited (PR #49)
```

### $120 small-account quick reference

For accounts between ~$100–$500, small-account mode is activated automatically
(no code changes needed – just set `SetCash(120)` in `main.py`).

```python
# main.py – change only this line
SetCash(120)          # triggers auto-scaling since 120 < SMALL_ACCOUNT_THRESHOLD_USD (500)
SetStartDate(2025, 1, 1)

# Auto-applied overrides (from config.py SMALL_ACCOUNT_* constants)
# SMALL_ACCOUNT_MAX_POSITIONS    = 2      → max 2 simultaneous positions
# SMALL_ACCOUNT_MAX_EXPOSURE_USD = 40.0   → $40 cap per position
# Effective max deployed: 2 × $40 = $80 (~67% of $120)

# Unchanged constraints that still apply:
MIN_EXPECTED_PROFIT_PCT = 0.010   # 1% min net profit gate – $0.40 on a $40 position
MAX_DAILY_LOSS_PCT      = 0.03    # 3% = ~$3.60 daily stop on $120
MAX_DRAWDOWN_LIMIT      = 0.15    # 15% = $18 max portfolio drawdown before cooldown
```

> **Tip:** With $120 the minimum order notional (`min_notional = $5.50`) is still
> met by the auto-scaled position cap.  If entries are consistently rejected as
> "too small", raise `SMALL_ACCOUNT_MAX_EXPOSURE_USD` slightly (e.g. to `50`).

---

## Phased Rollout

### Phase 1 – Backtest validation (before any real money)

1. Set `MODE_DEFAULT = "backtest"` in `config.py` (it is the default).
2. Run a minimum 3-month backtest on Kraken data with `SetCash(2000)`.
3. Review **MG3 Backtest Metrics** at the end of the run:
   - Cancel-to-fill ratio < 2
   - Invalid orders < 5 % of fills
   - Recovery events ≈ 0 (> 5 indicates a systemic exit-failure problem)
   - "Stop Loss" avg PnL less negative than `-tight_stop_loss` (check for slippage overshoot)
   - Positive expectancy per exit tag in the expected direction
4. Cross-check equity curve, max drawdown, and Sharpe ratio.

### Phase 2 – Paper / live-sim

1. Change `MODE_DEFAULT = "paper"` (or keep `"backtest"` and use QC Paper Trading).
2. Let the strategy run for at least one week.
3. Verify order logs match expected fill/cancel rates from Phase 1.
4. Monitor reconciliation logs (`[RECONCILE]` prefix) – phantom positions or missed states
   indicate an integration issue that must be fixed before live.
5. Confirm `Recovery events = 0` in live paper; any non-zero count needs investigation.

### Phase 3 – Micro live

1. Set `MODE_DEFAULT = "live"` in `config.py`.
2. Start with minimum capital (~$120–$250) and `MAX_OPEN_POSITIONS = 2`.
3. Watch for Kraken-specific issues:
   - `KRAKEN_PRICE_PRECISION` / `KRAKEN_LOT_SIZE` may need tuning for illiquid pairs.
   - Monitor rate-limit messages in the log; adjust `rate_limit_cooldown_minutes` if needed.
4. Only scale capital after at least 20 live trades with results consistent with Phase 1.

### Phase 4 – Full $2,000 deployment

- Set `MAX_OPEN_POSITIONS = 3` and `MAX_SYMBOL_EXPOSURE_USD = 250` (already the defaults).
- Keep `MAX_DAILY_LOSS_PCT = 0.03` as a permanent safety rail.
- Gradually raise `MAX_SYMBOL_EXPOSURE_USD` (up to $500) only after confirming
  backtest-vs-live performance gap is < 30 %.

---

## Safety Posture

- **Live trading is off by default.**  You must set `mode = "live"` explicitly.
- If the mode is not `"live"` but `LiveMode=True` in QuantConnect, the algorithm will
  call `self.Error(...)` and refuse to trade.
- All risk limits are sourced from `config.py` so changes are auditable and reversible.
- Short positions are disabled until short-signal validation is completed.
- Failed exits are **never silently abandoned**: after `MAX_EXIT_RETRIES` the position
  escalates to `recovering` and is force-market liquidated via `ReconcilePositions`.
