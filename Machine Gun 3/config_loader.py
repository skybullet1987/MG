"""
Machine Gun 3 – Config Loader
==============================
Helpers for loading and validating MG3 configuration values.

All canonical parameter definitions live in ``config.py``.  This module
provides utilities for safe parameter access and mode validation so that
``main.py`` stays thin.

Usage (within a QCAlgorithm subclass)::

    from config_loader import validate_mode, get_float_param, get_str_param

    mode = get_str_param(self, "mode", MG3Config.MODE_DEFAULT)
    validate_mode(mode)  # raises if mode is unrecognised
"""

import config as MG3Config

_VALID_MODES = {"backtest", "paper", "live"}


def validate_mode(mode: str) -> str:
    """Return *mode* unchanged if valid, otherwise raise ``ValueError``.

    Args:
        mode: One of ``"backtest"``, ``"paper"``, or ``"live"``.

    Returns:
        The validated mode string.

    Raises:
        ValueError: If *mode* is not a recognised trading mode.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"MG3: unrecognised mode '{mode}'. "
            f"Valid options: {sorted(_VALID_MODES)}"
        )
    return mode


def get_float_param(algo, name: str, default: float) -> float:
    """Safely read a float algorithm parameter with a fallback default.

    Args:
        algo: The QCAlgorithm instance (provides ``GetParameter``).
        name: Parameter key to look up.
        default: Value returned when the parameter is missing or blank.

    Returns:
        The float value of the parameter, or *default*.
    """
    try:
        param = algo.GetParameter(name)
        if param is not None and param != "":
            return float(param)
        return default
    except Exception as e:
        algo.Debug(f"[config_loader] Error reading float param '{name}': {e}")
        return default


def get_str_param(algo, name: str, default: str) -> str:
    """Safely read a string algorithm parameter with a fallback default.

    Args:
        algo: The QCAlgorithm instance (provides ``GetParameter``).
        name: Parameter key to look up.
        default: Value returned when the parameter is missing or blank.

    Returns:
        The lowercased, stripped string value, or *default*.
    """
    try:
        param = algo.GetParameter(name)
        if param is not None and param != "":
            return str(param).strip().lower()
        return default
    except Exception as e:
        algo.Debug(f"[config_loader] Error reading str param '{name}': {e}")
        return default


def summarise_config(algo) -> None:
    """Log a one-line config summary to the algorithm output.

    Useful at algorithm startup to confirm which parameters are active.
    """
    algo.Debug(
        f"[MG3 CONFIG] mode={getattr(algo, 'mg3_mode', '?')} "
        f"strict_fill={MG3Config.STRICT_LIMIT_FILL} "
        f"fee_rt={MG3Config.FEE_ASSUMPTION_RT:.3%} "
        f"max_pos={MG3Config.MAX_OPEN_POSITIONS}"
    )
