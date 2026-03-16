# Machine Gun Algo

This repository contains three strategy implementations:

| Folder | Description |
|---|---|
| `Machine Gun v2/` | Original MG v7.1.0 micro-scalper (reference copy, untouched) |
| `Machine Gun 3/` | **MG v8.1.0** — hardened successor; recommended starting point |
| `Bybit/` | Experimental Bybit variant |

See [`Machine Gun 3/README.md`](Machine%20Gun%203/README.md) for full documentation,
parameter guidance, and the phased rollout checklist.

## Machine Gun 3 – Quick Start

1. Upload **all five** `Machine Gun 3/*.py` files to the same QuantConnect project.
2. Set `SetCash(2000)` (or `SetCash(120)` for a small-account test) in `main.py`.
3. Click **Backtest** — the algorithm defaults to `backtest` mode with live trading off.

> MG3 v8.1.0 uses a single `QCAlgorithm` subclass with **no mixin inheritance and
> no dynamic method injection** — both of which caused QC/PythonNet runtime errors
> in earlier versions.  Only **five** `.py` files are required (`main.py`,
> `config.py`, `config_loader.py`, `execution.py`, `scoring.py`).  The old mixin
> files (`app.py`, `data_layer.py`, `orchestration.py`, `exit_handler.py`,
> `reporting.py`) have been deleted — do not upload them.
