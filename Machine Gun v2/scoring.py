# region imports
from AlgorithmImports import *
import numpy as np
# endregion

"""
scoring.py — Setup-Based Score Engine for Machine Gun v2
=========================================================
Replaces the additive "score soup" with a clean setup-dispatch layer.
The MicroScalpEngine class is preserved for compatibility with main.py and
mg2_entries.py, but its internals now delegate entirely to the explicit
setup evaluators in mg2_setups.py.

Design principles
-----------------
• One setup fires per symbol per bar — no additive blending between families.
• A trade only fires when a *specific, named* setup qualifies.
• Kalman is used only for trend confirmation and anti-chase, not mean reversion.
• Mean-reversion entries are not included in this strategy version.
• Kelly is disabled by default; flat fractional sizing with vol-scaling instead.
"""

# Guard: import setups lazily to allow this file to be imported standalone
try:
    from mg2_setups import best_setup, evaluate_all_setups
    _SETUPS_AVAILABLE = True
except ImportError:
    _SETUPS_AVAILABLE = False


# Minimum confidence threshold for a setup to generate an entry signal.
# This replaces the old ``entry_threshold`` additive score gate.
SETUP_MIN_CONFIDENCE = 0.55

# High-conviction threshold — triggers maximum position size
SETUP_HIGH_CONVICTION = 0.72


class MicroScalpEngine:
    """
    Momentum-Breakout Setup Engine — v8.0.0

    Entry logic is now driven exclusively by three explicit setup families
    (IgnitionBreakout, CompressionExpansion, MomentumContinuation).

    The additive multi-signal score has been removed.  Each bar, the engine
    asks: "Does this symbol match a known, high-quality breakout setup?"
    If yes, it returns the setup type, confidence, and components.
    If no setup qualifies, no trade is taken.

    Kalman repurposed
    -----------------
    • Kalman slope → trend direction confirmation
    • Kalman distance → anti-chase filter
    • Kalman estimate → dynamic trailing reference (in exits)

    Mean reversion removed
    ----------------------
    Mean-reversion bounce entries are no longer a valid entry path.
    RSI oversold and BB band bounce signals have been removed from entries.
    """

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (setup_type, confidence, components)
    # or (None, 0.0, {}) when no setup qualifies.
    # ------------------------------------------------------------------
    def evaluate_setup(self, crypto):
        """
        Evaluate all momentum-breakout setup families for a symbol.

        Returns
        -------
        (setup_type: str | None, confidence: float, components: dict)
        """
        if not _SETUPS_AVAILABLE:
            return None, 0.0, {}
        return best_setup(crypto, self.algo)

    # Legacy compatibility shim used by mg2_entries.py
    def calculate_scalp_score(self, crypto):
        """
        Backward-compatible wrapper.

        Returns (score: float, components: dict) where score = confidence and
        components includes 'setup_type' for attribution.
        """
        setup_type, confidence, components = self.evaluate_setup(crypto)
        if setup_type is None:
            return 0.0, {'setup_type': None}
        return confidence, components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Flat-fractional sizing with volatility scaling.

        Kelly is disabled for this version — it obscures whether the entry
        architecture has real edge and can amplify noise on small accounts.
        Use the ``use_kelly`` parameter (default False) in main.py to re-enable.

        High-conviction setups (score >= SETUP_HIGH_CONVICTION) get larger
        base size; otherwise a conservative base is used.

        Volatility scaling reduces size for hyper-volatile assets but never
        drops below 50% of base size.

        Returns a fraction of available capital (0.0 – 1.0).
        """
        # Base size by conviction
        if score >= SETUP_HIGH_CONVICTION:
            base_size = getattr(self.algo, 'position_size_high_conviction', 0.45)
        elif score >= threshold:
            base_size = getattr(self.algo, 'position_size_pct', 0.35)
        else:
            base_size = 0.25

        # Volatility scaling: target annual vol of position ≈ target_position_ann_vol
        if asset_vol_ann is not None and asset_vol_ann > 0:
            target_vol = getattr(self.algo, 'target_position_ann_vol', 0.40)
            vol_scalar = min(target_vol / asset_vol_ann, 1.0)
            base_size *= max(vol_scalar, 0.50)

        # Kelly gate: only apply if explicitly enabled in config
        if getattr(self.algo, 'use_kelly', False):
            kelly = self.algo._kelly_fraction()
            base_size *= kelly

        # Hard cap: never exceed max_position_pct of portfolio per trade
        max_pct = getattr(self.algo, 'max_position_pct', 0.50)
        base_size = min(base_size, max_pct)

        return base_size
