"""
Machine Gun 3 – Backtest Flow
==============================
Thin module that documents and provides helpers for the MG3 backtest workflow.

Running a backtest
------------------
MG3 uses QuantConnect's built-in backtest engine.  There is no standalone
runner script; the entry point is ``main.py`` (the ``SimplifiedCryptoStrategy``
class).

**QuantConnect Cloud (recommended)**

1. Create a new Algorithm project in the QC web IDE.
2. Upload all module files::

       main.py  app.py  data_layer.py  orchestration.py
       exit_handler.py  reporting.py  config_loader.py
       execution.py  scoring.py  alt_data.py  config.py

3. Confirm ``MODE_DEFAULT = "backtest"`` in ``config.py`` (it is the default).
4. Set start/end dates and starting cash in ``main.py`` (``SetStartDate`` /
   ``SetCash``).
5. Click **Backtest**.

**Lean CLI**::

    pip install lean
    lean project-create mg3
    # copy all .py files into the mg3/ project folder
    lean backtest mg3

Acceptance criteria (Phase 1 – see README)
-------------------------------------------
* Cancel-to-fill ratio < 2
* Invalid orders < 5 % of fills
* Positive expectancy per exit tag in the expected direction
"""

import config as MG3Config


def backtest_summary_header(algo) -> None:
    """Log a backtest-start banner with key configuration values."""
    algo.Debug(
        f"=== MG3 BACKTEST START | mode={MG3Config.MODE_DEFAULT} "
        f"strict_fill={MG3Config.STRICT_LIMIT_FILL} "
        f"fee_rt={MG3Config.FEE_ASSUMPTION_RT:.3%} ==="
    )


def check_backtest_readiness(algo) -> bool:
    """Return True if the algorithm state is suitable for backtest entry decisions.

    Checks that warm-up has finished and at least one symbol is ready.
    """
    if algo.IsWarmingUp:
        return False
    ready = sum(
        1 for c in algo.crypto_data.values()
        if len(c.get('prices', [])) >= 10
    )
    return ready > 0
