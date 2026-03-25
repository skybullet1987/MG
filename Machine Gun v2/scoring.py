# region imports
from AlgorithmImports import *
import numpy as np
# endregion

"""
scoring.py — Leader Breakout v0 | Machine Gun v2
=================================================
One setup only: IgnitionBreakout.

A valid entry requires ALL of the following hard gates.  If any fails, the
symbol is skipped with no partial credit or fallback blending:

  1. Strong short-term RS vs BTC  (5-bar cumulative relative return)
  2. Strong medium-term RS vs BTC (20-bar cumulative relative return)
  3. Breakout above recent 20-bar high, freshness ≤ BREAKOUT_FRESH_BARS
  4. Meaningful volume surge (≥ VOL_SURGE_MIN × rolling 60-bar baseline)
  5. Price above VWAP
  6. EMA fast > EMA medium (bullish trend structure)
  7. Not too extended above Kalman estimate (anti-chase)
  8. Spread acceptable (≤ MAX_SPREAD_SETUP)
  9. Expected move estimate clears fees + slippage + edge buffer
 10. Volume ignition fresh enough (≤ VOL_IGNITION_FRESH_MAX bars)
 11. VWAP reclaim fresh enough (≤ VWAP_RECLAIM_FRESH_MAX bars)
 12. False-breakout fingerprint filter (rule-based rejection of likely-fail patterns)

The confidence score (0.55–1.0) is computed only when all gates pass and
is used solely for candidate ranking — not as an additional entry gate.
Cross-sectional RS rank is incorporated into confidence scoring.

No CompressionExpansion.  No MomentumContinuation.  No setup blending.
"""

# ── Hard gate thresholds ──────────────────────────────────────────────────────
RS_SHORT_MIN         = 0.003   # min 5-bar cumulative RS vs BTC
RS_MEDIUM_MIN        = 0.006   # min 20-bar cumulative RS vs BTC
BREAKOUT_FRESH_BARS  = 2       # max bars since 20-bar high was broken
VOL_SURGE_MIN        = 2.5     # minimum volume ratio vs 60-bar baseline
VOL_SURGE_STRONG     = 4.0     # "strong surge" threshold (extra ranking weight)
ANTICHASE_MAX_PCT    = 0.025   # max extension above Kalman estimate (2.5%)
MAX_SPREAD_SETUP     = 0.010   # 1.0% hard spread cap for setup qualification

# Freshness staleness hard limits (bars)
VOL_IGNITION_FRESH_MAX  = 4    # reject if volume ignition was > 4 bars ago
VWAP_RECLAIM_FRESH_MAX  = 6    # reject if VWAP reclaim was > 6 bars ago

# Expected-move gate: ATR × TP_MULT / price must clear this net threshold
ATR_TP_MULT          = 6.0
ROUND_TRIP_COST_EST  = 0.012   # 2 × (0.4% fee + 0.2% slippage)
EDGE_BUFFER          = 0.005   # minimum edge above costs

# Confidence tiers (for position sizing)
SETUP_MIN_CONFIDENCE   = 0.55
SETUP_HIGH_CONVICTION  = 0.75

# ── Arming state constants ─────────────────────────────────────────────────────
ARM_DORMANT     = 'DORMANT'
ARM_ARMING      = 'ARMING'
ARM_READY       = 'READY'
ARM_TRIGGERED   = 'TRIGGERED'
ARM_COOLDOWN    = 'COOLDOWN'
ARM_INVALIDATED = 'INVALIDATED'


# ── False-breakout fingerprint filter ─────────────────────────────────────────
class FalseBreakoutFilter:
    """
    Rule-based filter that rejects likely-fail breakout patterns before
    an entry is placed.  Each rule returns a string reason if the filter
    triggers, or None if the pattern is acceptable.

    Rejection patterns:
      1. Too extended above VWAP already (likely exhaustion, not ignition).
      2. BB width contracting right after breakout bar (expansion failed).
      3. Declining volume persistence after initial ignition (fake follow-through).
      4. Negative OBI (order book imbalance) on the breakout bar.
      5. Combined: weak breadth AND already extended (double risk flag).
    """

    # Hard extension limit: beyond this % above VWAP we consider the move exhausted
    MAX_VWAP_EXTENSION_PCT   = 0.05   # 5%
    # If ATR is available, also gate on ATR-normalised extension
    MAX_ATR_EXTENSION_MULT   = 3.5    # price more than 3.5 ATRs above VWAP = too extended
    # OBI threshold: negative imbalance on breakout bar is a bearish signal
    OBI_NEGATIVE_THRESHOLD   = -0.15
    # Vol persistence drop: if vol_persistence fell by this much vs recent high, warn
    VOL_PERSIST_DROP_MIN     = 3
    # Breadth below this level combined with extension is a double red flag
    WEAK_BREADTH_THRESHOLD   = 0.40

    def check(self, crypto, algo):
        """
        Returns (reject, reason) where reject is True if the setup should be
        skipped.  Runs all rules and returns on the first match.
        """
        price  = list(crypto.get('prices', []))[-1] if crypto.get('prices') else 0.0
        vwap   = float(crypto.get('vwap', 0.0))

        # Rule 1: too extended above VWAP
        if vwap > 0 and price > 0:
            ext_pct = (price - vwap) / vwap
            if ext_pct > self.MAX_VWAP_EXTENSION_PCT:
                return True, 'fb_too_extended_vwap'

            # Rule 1b: ATR-normalised extension check
            atr = crypto.get('atr')
            if (atr and hasattr(atr, 'IsReady') and atr.IsReady
                    and atr.Current.Value > 0 and vwap > 0):
                atr_val = float(atr.Current.Value)
                atr_ext = (price - vwap) / atr_val
                if atr_ext > self.MAX_ATR_EXTENSION_MULT:
                    return True, 'fb_too_extended_atr'

        # Rule 2: BB width contracting after breakout (expansion failed)
        bb_widths = list(crypto.get('bb_width', []))
        if len(bb_widths) >= 4:
            # The breakout bar should see expansion; if the last 2 bars narrow, suspicious
            recent_w   = float(np.mean(bb_widths[-2:]))
            breakout_w = float(bb_widths[-3]) if len(bb_widths) >= 3 else recent_w
            if recent_w < breakout_w * 0.90:
                return True, 'fb_bb_contracting'

        # Rule 3: declining volume persistence (fake follow-through)
        vol_pers = int(crypto.get('vol_persistence', 0))
        if vol_pers <= 0:
            return True, 'fb_vol_persistence_zero'

        # Rule 4: negative OBI on breakout bar
        obi_hist = list(crypto.get('obi_history', []))
        if len(obi_hist) >= 2:
            latest_obi = float(obi_hist[-1])
            if latest_obi < self.OBI_NEGATIVE_THRESHOLD:
                return True, 'fb_obi_negative'

        # Rule 5: weak breadth + extended (double flag)
        breadth = float(getattr(algo, 'market_breadth', 0.5))
        if vwap > 0 and price > 0:
            ext_pct = (price - vwap) / vwap
            if breadth < self.WEAK_BREADTH_THRESHOLD and ext_pct > 0.02:
                return True, 'fb_weak_breadth_extended'

        return False, None


# ── Setup implementation ───────────────────────────────────────────────────────
class IgnitionBreakoutSetup:
    """
    Leader breakout setup: price surges above rolling 20-bar high with
    strong relative strength vs BTC, meaningful volume expansion, and
    clean EMA/VWAP/Kalman structure.

    All conditions are hard gates.  If any fails, returns (None, 0.0, {}).
    Confidence score is computed only when all gates pass and is used for
    candidate ranking — not as an additional entry gate.
    """

    NAME = "IgnitionBreakout"

    _false_breakout_filter = FalseBreakoutFilter()

    def evaluate(self, crypto, algo):
        """
        Returns (setup_name, confidence, components) if all gates pass.
        Returns (None, 0.0, {'_reject': reason}) on first failing gate.
        """
        prices = list(crypto.get('prices', []))
        if len(prices) < 20:
            return None, 0.0, {}
        price = prices[-1]
        if price <= 0:
            return None, 0.0, {}

        # ── Gate 1: Short-term RS vs BTC ──────────────────────────────────────
        rs_short = self._rs_short(crypto)
        if rs_short <= RS_SHORT_MIN:
            return None, 0.0, {'_reject': 'rs_short_weak'}

        # ── Gate 2: Medium-term RS vs BTC ─────────────────────────────────────
        rs_medium = float(crypto.get('rs_vs_btc_medium', 0.0))
        if rs_medium <= RS_MEDIUM_MIN:
            return None, 0.0, {'_reject': 'rs_medium_weak'}

        # ── Gate 3: Breakout freshness ─────────────────────────────────────────
        freshness = crypto.get('breakout_freshness', 999)
        if freshness > BREAKOUT_FRESH_BARS:
            return None, 0.0, {'_reject': 'stale_breakout'}

        # ── Gate 4: Volume surge ───────────────────────────────────────────────
        vol_ratio = self._vol_ratio(crypto)
        if vol_ratio < VOL_SURGE_MIN:
            return None, 0.0, {'_reject': 'volume_low'}

        # ── Gate 5: Price above VWAP ───────────────────────────────────────────
        vwap = float(crypto.get('vwap', 0.0))
        if vwap > 0 and price <= vwap:
            return None, 0.0, {'_reject': 'below_vwap'}

        # ── Gate 6: EMA bullish structure ──────────────────────────────────────
        if not self._ema_bullish(crypto, price):
            return None, 0.0, {'_reject': 'ema_not_bullish'}

        # ── Gate 7: Anti-chase (Kalman extension) ─────────────────────────────
        kalman = float(crypto.get('kalman_estimate', 0.0))
        if not self._antichase_ok(price, kalman):
            return None, 0.0, {'_reject': 'too_extended'}

        # ── Gate 8: Spread ─────────────────────────────────────────────────────
        spreads = list(crypto.get('spreads', []))
        if spreads:
            recent_spread = float(spreads[-1])
            if recent_spread > MAX_SPREAD_SETUP:
                return None, 0.0, {'_reject': 'spread_too_wide'}

        # ── Gate 9: Expected move clears fees + slippage + edge buffer ──────────
        atr = crypto.get('atr')
        atr_val = (float(atr.Current.Value)
                   if (atr and hasattr(atr, 'IsReady') and atr.IsReady
                       and atr.Current.Value > 0)
                   else None)
        if not self._expected_move_ok(price, atr_val):
            return None, 0.0, {'_reject': 'expected_move_insufficient'}

        # ── Gate 10: Volume ignition freshness ────────────────────────────────
        bars_vol_ign = int(crypto.get('bars_since_vol_ignition', 999))
        if bars_vol_ign > VOL_IGNITION_FRESH_MAX:
            return None, 0.0, {'_reject': 'stale_vol_ignition'}

        # ── Gate 11: VWAP reclaim freshness ───────────────────────────────────
        bars_vwap_rec = int(crypto.get('bars_since_vwap_reclaim', 999))
        if bars_vwap_rec > VWAP_RECLAIM_FRESH_MAX:
            return None, 0.0, {'_reject': 'stale_vwap_reclaim'}

        # ── Gate 12: False-breakout fingerprint filter ─────────────────────────
        fb_reject, fb_reason = self._false_breakout_filter.check(crypto, algo)
        if fb_reject:
            return None, 0.0, {'_reject': fb_reason}

        # ── All gates passed — compute confidence for ranking ──────────────────
        rs_rank = float(crypto.get('rs_rank', 0.5))
        confidence = self._compute_confidence(
            rs_short, rs_medium, vol_ratio, freshness, price, vwap, rs_rank
        )
        components = {
            'setup_type':          self.NAME,
            'rs_short':            rs_short,
            'rs_medium':           rs_medium,
            'vol_ratio':           vol_ratio,
            'freshness':           freshness,
            'vwap':                vwap,
            'atr':                 atr_val,
            'kalman':              kalman,
            'rs_rank':             rs_rank,
            'bars_vol_ignition':   bars_vol_ign,
            'bars_vwap_reclaim':   bars_vwap_rec,
        }
        return self.NAME, confidence, components

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rs_short(self, crypto):
        """Rolling sum of most recent rs_vs_btc entries (5-bar proxy)."""
        rs_hist = list(crypto.get('rs_vs_btc', []))
        if len(rs_hist) >= 3:
            return float(np.sum(rs_hist[-3:]))
        elif rs_hist:
            return float(rs_hist[-1])
        return 0.0

    def _vol_ratio(self, crypto):
        """Current volume / rolling 60-bar baseline."""
        volumes = list(crypto.get('volume', []))
        if len(volumes) < 5:
            return 0.0
        current   = volumes[-1]
        long_vols = list(crypto.get('volume_long', []))
        if len(long_vols) >= 60:
            baseline = float(np.mean(long_vols[-60:]))
        elif len(long_vols) >= 10:
            baseline = float(np.mean(long_vols))
        else:
            baseline = float(np.mean(volumes[-min(len(volumes), 10):]))
        return (current / baseline) if baseline > 0 else 0.0

    def _ema_bullish(self, crypto, price):
        """True if price > EMA_fast > EMA_medium."""
        ema5    = crypto.get('ema_5')
        ema_med = crypto.get('ema_medium')
        if not (ema5    and hasattr(ema5,    'IsReady') and ema5.IsReady    and
                ema_med and hasattr(ema_med, 'IsReady') and ema_med.IsReady):
            return False
        return price > ema5.Current.Value > ema_med.Current.Value

    def _antichase_ok(self, price, kalman):
        """True if price is not more than ANTICHASE_MAX_PCT above Kalman."""
        if kalman > 0:
            dist = (price - kalman) / kalman
            if dist > ANTICHASE_MAX_PCT:
                return False
        return True

    def _expected_move_ok(self, price, atr_val):
        """True if ATR × TP_MULT / price clears costs + edge buffer."""
        if atr_val and price > 0:
            expected_pct = atr_val * ATR_TP_MULT / price
            min_required = ROUND_TRIP_COST_EST + EDGE_BUFFER
            return expected_pct >= min_required
        return True   # no ATR yet — pass (benefit of doubt during warmup)

    def _compute_confidence(self, rs_short, rs_medium, vol_ratio, freshness,
                             price, vwap, rs_rank=0.5):
        """
        Compute ranking confidence (0.55–1.0).

        Weights:
          RS strength (short + medium)     30%
          Cross-sectional RS rank           15%
          Volume surge magnitude            25%
          Breakout freshness                15%
          VWAP extension (less = better)   15%
        """
        # RS component (absolute strength)
        rs_combined = rs_short + rs_medium
        if rs_combined >= 0.05:
            rs_score = 1.0
        elif rs_combined >= 0.025:
            rs_score = 0.75
        elif rs_combined >= 0.012:
            rs_score = 0.55
        else:
            rs_score = 0.35

        # Cross-sectional rank component (percentile 0→1; top-quartile gets bonus)
        rank_score = float(rs_rank)   # already 0–1; 1.0 = strongest in universe

        # Volume component
        if vol_ratio >= VOL_SURGE_STRONG:
            vol_score = 1.0
        elif vol_ratio >= VOL_SURGE_MIN * 1.4:
            vol_score = 0.75
        else:
            vol_score = 0.50

        # Freshness component (0 bars = freshest = best)
        if freshness == 0:
            fresh_score = 1.0
        elif freshness == 1:
            fresh_score = 0.70
        else:
            fresh_score = 0.40   # freshness == 2

        # Extension component (tighter to VWAP = better entry)
        ext_score = 0.65
        if vwap > 0 and price > 0:
            ext_pct = (price - vwap) / vwap
            if ext_pct < 0.005:
                ext_score = 1.0
            elif ext_pct < 0.015:
                ext_score = 0.80
            elif ext_pct < 0.025:
                ext_score = 0.55
            else:
                ext_score = 0.30

        confidence = (
            rs_score    * 0.30 +
            rank_score  * 0.15 +
            vol_score   * 0.25 +
            fresh_score * 0.15 +
            ext_score   * 0.15
        )
        return min(1.0, max(SETUP_MIN_CONFIDENCE, confidence))


# ── Engine ─────────────────────────────────────────────────────────────────────
class LeaderBreakoutEngine:
    """
    Leader Breakout v0 engine.

    Wraps the single IgnitionBreakout setup and provides the position-sizing
    interface used by mg2_entries.execute_trades().
    """

    def __init__(self, algo):
        self.algo   = algo
        self._setup = IgnitionBreakoutSetup()

    def evaluate_setup(self, crypto):
        """
        Evaluate IgnitionBreakout for a single symbol.
        Returns (setup_name, confidence, components) or (None, 0.0, {}).
        """
        return self._setup.evaluate(crypto, self.algo)

    def calculate_position_size(self, confidence, threshold, ann_vol):
        """
        Two-tier concentrated sizing:
          - High conviction (confidence >= SETUP_HIGH_CONVICTION): position_size_high_conviction
          - Standard: position_size_pct
        Volatility scaling applied for extreme-vol assets (never below 50% of base).
        Hard cap at max_position_pct.
        """
        algo = self.algo
        if confidence >= SETUP_HIGH_CONVICTION:
            base = getattr(algo, 'position_size_high_conviction', 0.50)
        else:
            base = getattr(algo, 'position_size_pct', 0.40)

        # Volatility scaling: shrink proportionally if asset vol >> target
        if ann_vol and ann_vol > 0:
            target_vol = getattr(algo, 'target_position_ann_vol', 0.50)
            if ann_vol > target_vol:
                scale = target_vol / ann_vol
                base  = base * max(scale, 0.50)

        return min(base, getattr(algo, 'max_position_pct', 0.60))


# Backward-compatible alias — main.py imports MicroScalpEngine
MicroScalpEngine = LeaderBreakoutEngine
