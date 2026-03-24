# region imports
from AlgorithmImports import *
from execution import *
# endregion


def daily_report_v2(algo):
    daily_report(algo)


def reset_daily_counters(algo):
    algo.daily_trade_count = 0
    algo.last_trade_date = algo.Time.date()
    algo._daily_open_value = algo.Portfolio.TotalPortfolioValue
    for crypto in algo.crypto_data.values():
        crypto['trade_count_today'] = 0
    if len(algo._session_blacklist) > 0:
        algo.Debug(f"Clearing session blacklist ({len(algo._session_blacklist)} items)")
        algo._session_blacklist.clear()
    algo._symbol_entry_cooldowns.clear()
    persist_state(algo)
