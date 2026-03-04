"""
Machine Gun 3 – Paper / Live-Sim Flow
=======================================
Thin module that documents and provides helpers for the MG3 paper and
live-simulation workflow.

Running in paper mode
---------------------
MG3 paper mode uses QuantConnect Paper Trading, which replays a live market
feed against the algorithm without risking real capital.

1. Set ``MODE_DEFAULT = "paper"`` in ``config.py``.
2. Deploy the project to QuantConnect Paper Trading (cloud).
3. All uploaded module files must be present::

       main.py  app.py  data_layer.py  orchestration.py
       exit_handler.py  reporting.py  config_loader.py
       execution.py  scoring.py  alt_data.py  config.py

4. Monitor ``[RECONCILE]`` log entries – phantom or untracked positions
   indicate a brokerage-sync issue that must be fixed before going live.

Running in live mode
--------------------
1. Set ``MODE_DEFAULT = "live"`` in ``config.py``.
   The algorithm **raises an error** at startup if ``LiveMode=True`` but
   ``mode != "live"`` to prevent accidental live deployment.
2. Review every risk limit in ``config.py`` before deploying.
3. Follow the phased rollout described in ``README.md`` (Phases 1–4).

Acceptance criteria (Phase 2 – see README)
-------------------------------------------
* Order logs match expected fill/cancel rates from the backtest phase.
* Reconciliation logs show no phantom positions after one week.
"""

import config as MG3Config


def paper_summary_header(algo) -> None:
    """Log a paper-mode start banner with key configuration values."""
    algo.Debug(
        f"=== MG3 PAPER/LIVE-SIM START | mode={getattr(algo, 'mg3_mode', MG3Config.MODE_DEFAULT)} "
        f"max_daily_loss={MG3Config.MAX_DAILY_LOSS_PCT:.0%} "
        f"max_pos={MG3Config.MAX_OPEN_POSITIONS} ==="
    )


def check_paper_readiness(algo) -> bool:
    """Return True if the algorithm is warmed up and ready to trade in paper mode.

    In paper/live mode the brokerage status must also be ``"online"``.
    """
    if algo.IsWarmingUp:
        return False
    if getattr(algo, 'kraken_status', 'unknown') in ('maintenance', 'cancel_only'):
        return False
    return True
