# Machine Gun Algo

This repository contains three strategy implementations:

| Folder | Description |
|---|---|
| `Machine Gun v2/` | Original MG v7.1.0 micro-scalper (reference copy, untouched) |
| `Machine Gun 3/` | **MG v8.0.0** — hardened successor; recommended starting point |
| `Bybit/` | Experimental Bybit variant |

See [`Machine Gun 3/README.md`](Machine%20Gun%203/README.md) for full documentation,
parameter guidance, and the phased rollout checklist.

## Machine Gun 3 – Quick Start

1. Upload **all ten** `Machine Gun 3/*.py` files to the same QuantConnect project.
2. Set `SetCash(2000)` (or `SetCash(120)` for a small-account test) in `main.py`.
3. Click **Backtest** — the algorithm defaults to `backtest` mode with live trading off.

> MG3 uses a single `QCAlgorithm` subclass with post-definition method injection,
> which resolves the PythonNet multiple-inheritance incompatibility that affected
> earlier refactoring attempts.  All ten `.py` files are required; a missing file
> causes `No module named '<file>'` at startup.
