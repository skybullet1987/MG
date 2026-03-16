# Machine Gun 3 — Position lifecycle state constants
# Defined here (not in main.py) so that helper modules can import them
# without creating circular-import dependencies.

POSITION_STATE_FLAT       = "flat"
POSITION_STATE_ENTERING   = "entering"
POSITION_STATE_OPEN       = "open"
POSITION_STATE_EXITING    = "exiting"
POSITION_STATE_RECOVERING = "recovering"  # failed exits escalated to force-market
