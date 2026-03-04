# Bybit main.py

# Updated Configuration for MG v2.
entry_threshold = 0.40
high_conviction_threshold = 0.60
quick_take_profit = 0.080
tight_stop_loss = 0.025
trail_activation = 0.010
trail_stop_pct = 0.005
time_stop_hours = 2.0
time_stop_pnl_min = 0.003
position_size_pct = 0.70

# Regime Filtering for Rebalance
# Require minimum score of 0.85 for counter-trend entries.

# Modified CheckExits function
# Separated Long and Short PnL calculations for accurate trailing stops.
