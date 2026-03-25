# region imports
from AlgorithmImports import *
from execution import *
import numpy as np
from datetime import timedelta
from mg2_data import is_ready, annualized_vol, compute_portfolio_risk_estimate
# endregion

"""
mg2_entries.py — Concentrated Entry Logic for Leader Breakout v0
================================================================
Evaluates the focused universe with the single IgnitionBreakout setup,
selects at most the top 1–2 candidates ranked by leadership quality,
and executes entries with concentrated two-tier sizing.

Also owns trade attribution (DiagnosticsEngine / TradeRecord) and the
per-symbol arming state machine so that lifecycle tracking lives here.
"""

# During a bear market regime, require higher setup confidence
BEAR_REGIME_MIN_CONFIDENCE = 0.70

# Max candidates to execute per rebalance cycle (the "concentrated" part)
MAX_NEW_POSITIONS_PER_CYCLE = 2

# Minimum dollar volume required for a symbol to be a valid candidate
MIN_CANDIDATE_DOLLAR_VOLUME = 80_000   # 80k USD per bar (on top of universe filter)

# Breadth gate: skip new entries when < this fraction of universe is in uptrend
BREADTH_MIN_ENTRY = 0.30

# Configurable cooldown (hours) applied to a symbol after a FailedBreakout exit
FAILED_BREAKOUT_COOLDOWN_HOURS = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# Arming State Machine
# ─────────────────────────────────────────────────────────────────────────────

class ArmingStateMachine:
    """
    Per-symbol lifecycle tracker that prevents entries directly from raw
    one-bar conditions.

    States
    ------
    DORMANT     : symbol shows no relevant leadership signals
    ARMING      : symbol has started showing leadership (RS positive, vol rising)
                  but has not yet held it for the required minimum bars
    READY       : symbol has satisfied arming criteria for ARM_MIN_BARS
                  consecutive bars and may be evaluated for a breakout trigger
    TRIGGERED   : a trade entry has been placed for this symbol
    COOLDOWN    : symbol is in post-exit cooldown (general or failed-breakout)
    INVALIDATED : symbol violated its arming criteria and reverted to DORMANT
    """

    ARM_MIN_BARS     = 2     # bars of continuous leadership to reach READY
    ARM_RS_THRESHOLD = 0.001 # minimum instantaneous RS vs BTC bar to stay ARMING
    ARM_VOL_MULT     = 1.2   # volume must be at least 1.2× baseline to stay ARMING

    def update(self, symbol, crypto, algo):
        """
        Advance the arming state for one symbol using the current bar's data.
        Modifies crypto['arm_state'] and crypto['arm_state_bars'] in-place.
        Returns the new state string.
        """
        state      = crypto.get('arm_state', 'DORMANT')
        state_bars = int(crypto.get('arm_state_bars', 0))

        # ── TRIGGERED / COOLDOWN: managed externally; don't auto-advance ────
        if state in ('TRIGGERED', 'COOLDOWN'):
            crypto['arm_state_bars'] = state_bars + 1
            return state

        # ── Check if arming criteria are met ──────────────────────────────────
        arming_ok = self._arming_criteria_met(crypto, algo)

        if state == 'DORMANT':
            if arming_ok:
                crypto['arm_state']      = 'ARMING'
                crypto['arm_state_bars'] = 1
            return crypto['arm_state']

        if state == 'ARMING':
            if arming_ok:
                state_bars += 1
                crypto['arm_state_bars'] = state_bars
                if state_bars >= self.ARM_MIN_BARS:
                    crypto['arm_state'] = 'READY'
            else:
                # Lost leadership — invalidate
                crypto['arm_state']      = 'INVALIDATED'
                crypto['arm_state_bars'] = 0
            return crypto['arm_state']

        if state == 'READY':
            if not arming_ok:
                crypto['arm_state']      = 'INVALIDATED'
                crypto['arm_state_bars'] = 0
            else:
                crypto['arm_state_bars'] = state_bars + 1
            return crypto['arm_state']

        if state == 'INVALIDATED':
            # After invalidation, give the symbol one bar to recover to DORMANT
            crypto['arm_state']      = 'DORMANT'
            crypto['arm_state_bars'] = 0
            return 'DORMANT'

        return state

    def _arming_criteria_met(self, crypto, algo):
        """Return True if current bar shows minimum leadership signals."""
        rs_hist = list(crypto.get('rs_vs_btc', []))
        if not rs_hist or float(rs_hist[-1]) < self.ARM_RS_THRESHOLD:
            return False

        vols = list(crypto.get('volume', []))
        long_vols = list(crypto.get('volume_long', []))
        if len(vols) < 5:
            return True   # not enough data yet — give benefit of doubt
        baseline = (float(np.mean(long_vols[-120:])) if len(long_vols) >= 120
                    else float(np.mean(vols[-20:])))
        if baseline > 0:
            if float(vols[-1]) < baseline * self.ARM_VOL_MULT:
                return False

        return True

    def mark_triggered(self, crypto):
        crypto['arm_state']      = 'TRIGGERED'
        crypto['arm_state_bars'] = 0

    def mark_cooldown(self, crypto):
        crypto['arm_state']      = 'COOLDOWN'
        crypto['arm_state_bars'] = 0

    def mark_dormant(self, crypto):
        crypto['arm_state']      = 'DORMANT'
        crypto['arm_state_bars'] = 0


# Singleton — shared across rebalance and exit calls
_arm_sm = ArmingStateMachine()


def get_arming_state_machine():
    """Return the shared arming state machine instance."""
    return _arm_sm


# ─────────────────────────────────────────────────────────────────────────────
# Trade Attribution — DiagnosticsEngine
# ─────────────────────────────────────────────────────────────────────────────

class TradeRecord:
    """Full metadata record for one completed round-trip trade."""

    __slots__ = [
        'symbol', 'entry_time', 'exit_time',
        'setup_type', 'confidence',
        'entry_components',
        'market_regime', 'volatility_regime',
        'btc_5bar_return', 'rs_vs_btc', 'rs_medium',
        'spread_at_entry', 'spread_at_exit',
        'estimated_slippage_entry', 'realized_slippage_exit',
        'entry_price', 'exit_price',
        'gross_pnl_pct', 'net_pnl_pct',
        'exit_tag', 'hold_minutes',
        'breakout_freshness', 'vol_ratio', 'atr_at_entry',
        # MFE / MAE analytics
        'mfe_pct', 'mae_pct',
        'time_to_mfe_minutes', 'time_to_mae_minutes',
        # Outcome bucket for time-to-outcome analytics
        'outcome_bucket',   # 'instant_fail' | 'dead_trade' | 'delayed_fail' | 'runner' | 'normal'
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        # default outcome bucket
        self.outcome_bucket = 'normal'

    def net_pnl_estimate(self, fee_pct=0.008):
        if self.gross_pnl_pct is not None:
            slip = (self.estimated_slippage_entry or 0.0) + (self.realized_slippage_exit or 0.0)
            return self.gross_pnl_pct - fee_pct - slip
        return None


class DiagnosticsEngine:
    """
    Lightweight trade attribution engine.
    Records per-trade metadata and provides summary statistics.
    """

    ROUND_TRIP_FEE_PCT = 0.008  # 0.4% × 2 taker fills = 0.8%

    def __init__(self, algo):
        self.algo = algo
        self._open_records = {}   # symbol → TradeRecord (in-progress)
        self.completed     = []   # list of completed TradeRecord

    def record_entry(self, symbol, setup_type, confidence, components,
                     entry_price, spread_at_entry=None, estimated_slippage=None):
        algo = self.algo
        rec = TradeRecord()
        rec.symbol             = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
        rec.entry_time         = algo.Time
        rec.setup_type         = setup_type or 'Unknown'
        rec.confidence         = confidence
        rec.entry_components   = {k: v for k, v in (components or {}).items()
                                  if not k.startswith('_') and k != 'setup_type'}
        rec.market_regime      = algo.market_regime
        rec.volatility_regime  = algo.volatility_regime
        rec.spread_at_entry    = spread_at_entry
        rec.estimated_slippage_entry = estimated_slippage
        rec.entry_price        = entry_price
        rec.vol_ratio          = components.get('vol_ratio') if components else None
        rec.breakout_freshness = components.get('freshness') if components else None

        btc_rets = list(algo.btc_returns)
        rec.btc_5bar_return = float(sum(btc_rets[-5:])) if len(btc_rets) >= 5 else None

        crypto = algo.crypto_data.get(symbol)
        if crypto:
            rs_hist = list(crypto.get('rs_vs_btc', []))
            rec.rs_vs_btc  = float(rs_hist[-1]) if rs_hist else None
            rec.rs_medium  = float(crypto.get('rs_vs_btc_medium', 0.0))
            rec.atr_at_entry = (float(crypto['atr'].Current.Value)
                                if crypto.get('atr') and crypto['atr'].IsReady else None)

        self._open_records[symbol] = rec

    def record_exit(self, symbol, exit_price, exit_tag,
                    realized_slippage=None, spread_at_exit=None,
                    mfe_pct=None, mae_pct=None,
                    time_to_mfe_minutes=None, time_to_mae_minutes=None):
        algo = self.algo
        rec  = self._open_records.pop(symbol, None)
        if rec is None:
            return

        rec.exit_time              = algo.Time
        rec.exit_price             = exit_price
        rec.exit_tag               = exit_tag
        rec.spread_at_exit         = spread_at_exit
        rec.realized_slippage_exit = realized_slippage
        rec.mfe_pct                = mfe_pct
        rec.mae_pct                = mae_pct
        rec.time_to_mfe_minutes    = time_to_mfe_minutes
        rec.time_to_mae_minutes    = time_to_mae_minutes

        if rec.entry_price and rec.entry_price > 0:
            rec.gross_pnl_pct = (exit_price - rec.entry_price) / rec.entry_price
            rec.net_pnl_pct   = rec.net_pnl_estimate(self.ROUND_TRIP_FEE_PCT)

        if rec.entry_time and rec.exit_time:
            rec.hold_minutes = (rec.exit_time - rec.entry_time).total_seconds() / 60.0

        # Classify into outcome bucket
        rec.outcome_bucket = _classify_outcome_bucket(rec)

        self.completed.append(rec)

    def stats_by_exit(self):
        by_exit = {}
        for rec in self.completed:
            tag = rec.exit_tag or 'Unknown'
            by_exit.setdefault(tag, []).append(rec)
        result = {}
        for tag, records in by_exit.items():
            pnls = [r.net_pnl_pct for r in records if r.net_pnl_pct is not None]
            result[tag] = {
                'count':         len(records),
                'avg_net_pnl':   float(np.mean(pnls)) if pnls else 0.0,
                'total_net_pnl': float(sum(pnls)) if pnls else 0.0,
                'win_rate':      (sum(1 for p in pnls if p > 0) / len(pnls)
                                  if pnls else 0.0),
            }
        return result

    def stats_by_regime(self):
        by_regime = {}
        for rec in self.completed:
            regime = rec.market_regime or 'unknown'
            by_regime.setdefault(regime, []).append(rec)
        result = {}
        for regime, records in by_regime.items():
            pnls = [r.net_pnl_pct for r in records if r.net_pnl_pct is not None]
            result[regime] = {
                'count':       len(records),
                'avg_net_pnl': float(np.mean(pnls)) if pnls else 0.0,
                'win_rate':    (sum(1 for p in pnls if p > 0) / len(pnls)
                                if pnls else 0.0),
            }
        return result

    def print_summary(self):
        algo = self.algo
        n    = len(self.completed)
        if n == 0:
            algo.Debug("DIAGNOSTICS: No completed trades to report.")
            return

        all_pnls = [r.net_pnl_pct for r in self.completed if r.net_pnl_pct is not None]
        wins     = [p for p in all_pnls if p > 0]
        losses   = [p for p in all_pnls if p <= 0]

        algo.Debug("=" * 60)
        algo.Debug(f"LEADER BREAKOUT v0 SUMMARY — {n} completed trades")
        if all_pnls:
            algo.Debug(f"  Win rate:     {len(wins)/len(all_pnls):.1%}")
            algo.Debug(f"  Avg net PnL:  {float(np.mean(all_pnls)):.2%}")
        if wins:
            algo.Debug(f"  Avg win:      {float(np.mean(wins)):.2%}")
        if losses:
            algo.Debug(f"  Avg loss:     {float(np.mean(losses)):.2%}")
        if wins and losses and sum(losses) != 0:
            algo.Debug(f"  Profit factor:{sum(wins)/abs(sum(losses)):.2f}")

        # MFE / MAE summary
        mfes = [r.mfe_pct for r in self.completed if r.mfe_pct is not None]
        maes = [r.mae_pct for r in self.completed if r.mae_pct is not None]
        if mfes:
            algo.Debug(f"  Avg MFE:      {float(np.mean(mfes)):+.2%}  "
                       f"Max MFE: {float(np.max(mfes)):+.2%}")
        if maes:
            algo.Debug(f"  Avg MAE:      {float(np.mean(maes)):+.2%}  "
                       f"Worst MAE: {float(np.min(maes)):+.2%}")

        algo.Debug("─── By Exit Tag ───")
        for tag, stats in self.stats_by_exit().items():
            algo.Debug(
                f"  {tag}: n={stats['count']} wr={stats['win_rate']:.1%} "
                f"avg={stats['avg_net_pnl']:+.2%} total={stats['total_net_pnl']:+.2%}"
            )

        algo.Debug("─── By Market Regime ───")
        for regime, stats in self.stats_by_regime().items():
            algo.Debug(
                f"  {regime}: n={stats['count']} wr={stats['win_rate']:.1%} "
                f"avg={stats['avg_net_pnl']:+.2%}"
            )

        algo.Debug("─── By Outcome Bucket ───")
        for bucket, stats in self.stats_by_outcome_bucket().items():
            algo.Debug(
                f"  {bucket}: n={stats['count']} avg={stats['avg_net_pnl']:+.2%} "
                f"hold={stats['avg_hold_min']:.0f}m "
                f"MFE={stats['avg_mfe']:+.2%} MAE={stats['avg_mae']:+.2%}"
            )

        algo.Debug("=" * 60)

    def stats_by_outcome_bucket(self):
        by_bucket = {}
        for rec in self.completed:
            bucket = rec.outcome_bucket or 'normal'
            by_bucket.setdefault(bucket, []).append(rec)
        result = {}
        for bucket, records in by_bucket.items():
            pnls  = [r.net_pnl_pct for r in records if r.net_pnl_pct is not None]
            holds = [r.hold_minutes for r in records if r.hold_minutes is not None]
            mfes  = [r.mfe_pct for r in records if r.mfe_pct is not None]
            maes  = [r.mae_pct for r in records if r.mae_pct is not None]
            result[bucket] = {
                'count':        len(records),
                'avg_net_pnl':  float(np.mean(pnls)) if pnls else 0.0,
                'avg_hold_min': float(np.mean(holds)) if holds else 0.0,
                'avg_mfe':      float(np.mean(mfes)) if mfes else 0.0,
                'avg_mae':      float(np.mean(maes)) if maes else 0.0,
            }
        return result


def _classify_outcome_bucket(rec):
    """
    Classify a completed trade into a time-to-outcome bucket.

    Buckets:
      instant_fail  : exit in < 15 minutes with a loss (failed quickly)
      dead_trade    : held 15–90 minutes but exited near breakeven / slight loss
      delayed_fail  : held > 90 minutes but exited with a loss
      runner        : exited with net PnL > 5% (meaningful winner)
      normal        : everything else (small win, or not enough data)
    """
    hold = rec.hold_minutes
    pnl  = rec.net_pnl_pct

    if hold is None or pnl is None:
        return 'normal'
    if pnl > 0.05:
        return 'runner'
    if hold < 15 and pnl < -0.01:
        return 'instant_fail'
    if 15 <= hold <= 90 and pnl < 0.005:
        return 'dead_trade'
    if hold > 90 and pnl < 0:
        return 'delayed_fail'
    return 'normal'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_skip(algo, reason):
    if algo.LiveMode:
        debug_limited(algo, f"Rebalance skip: {reason}")
    elif reason != getattr(algo, '_last_skip_reason', None):
        debug_limited(algo, f"Rebalance skip: {reason}")
    algo._last_skip_reason = reason


def daily_loss_exceeded(algo):
    if algo._daily_open_value is None or algo._daily_open_value <= 0:
        return False
    current = algo.Portfolio.TotalPortfolioValue
    if current <= 0:
        return True
    drop = (algo._daily_open_value - current) / algo._daily_open_value
    return drop >= 0.04


def check_correlation(algo, new_symbol):
    """Reject candidate if it is too correlated with an existing position."""
    if not algo.entry_prices:
        return True
    new_crypto = algo.crypto_data.get(new_symbol)
    if not new_crypto or len(new_crypto['returns']) < 24:
        return True
    new_rets = np.array(list(new_crypto['returns'])[-24:])
    if np.std(new_rets) < 1e-10:
        return True
    for sym in list(algo.entry_prices.keys()):
        if sym == new_symbol:
            continue
        existing = algo.crypto_data.get(sym)
        if not existing or len(existing['returns']) < 24:
            continue
        exist_rets = np.array(list(existing['returns'])[-24:])
        if np.std(exist_rets) < 1e-10:
            continue
        try:
            corr = np.corrcoef(new_rets, exist_rets)[0, 1]
            if corr > 0.85:
                return False
        except Exception:
            continue
    return True


def _leadership_rank(cand):
    """
    Composite leadership rank for sorting candidates.
    Higher = stronger leader.  Used to pick the top 1–2.
    Incorporates cross-sectional RS rank from scoring.
    """
    f = cand.get('factors', {})
    rs_short  = float(f.get('rs_short',  0.0))
    rs_medium = float(f.get('rs_medium', 0.0))
    vol_ratio = float(f.get('vol_ratio', 0.0))
    freshness = int(f.get('freshness',   5))
    rs_rank   = float(f.get('rs_rank',   0.5))
    conf      = float(cand.get('net_score', 0.0))

    fresh_score = 1.0 if freshness == 0 else (0.7 if freshness == 1 else 0.4)

    return (
        (rs_short + rs_medium) * 0.30 +
        rs_rank                * 0.15 +
        min(vol_ratio / 10.0, 1.0) * 0.25 +
        fresh_score            * 0.15 +
        conf                   * 0.15
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point: rebalance
# ─────────────────────────────────────────────────────────────────────────────

def rebalance(algo):
    if algo.IsWarmingUp:
        return

    # ── Update arming states for all symbols ─────────────────────────────────
    arm_sm = get_arming_state_machine()
    for sym, crypto in algo.crypto_data.items():
        try:
            arm_sm.update(sym, crypto, algo)
        except Exception:
            pass

    if daily_loss_exceeded(algo):
        log_skip(algo, "max daily loss exceeded")
        return

    # BTC dump filter
    if len(algo.btc_returns) >= 5 and sum(list(algo.btc_returns)[-5:]) < -0.015:
        log_skip(algo, "BTC dumping")
        return

    if algo._cash_mode_until is not None and algo.Time < algo._cash_mode_until:
        log_skip(algo, "cash mode")
        return

    algo.log_budget = 20

    if algo._rate_limit_until is not None and algo.Time < algo._rate_limit_until:
        log_skip(algo, "rate limited")
        return

    if algo.LiveMode and not live_safety_checks(algo):
        return
    if algo.LiveMode and getattr(algo, 'kraken_status', 'unknown') in ("maintenance", "cancel_only"):
        log_skip(algo, "kraken not online")
        return

    cancel_stale_new_orders(algo)

    if algo.daily_trade_count >= algo.max_daily_trades:
        log_skip(algo, "max daily trades")
        return

    val = algo.Portfolio.TotalPortfolioValue
    if algo.peak_value is None or algo.peak_value < 1:
        algo.peak_value = val
    if algo.drawdown_cooldown > 0:
        algo.drawdown_cooldown -= 1
        if algo.drawdown_cooldown <= 0:
            algo.peak_value         = val
            algo.consecutive_losses = 0
        else:
            log_skip(algo, f"drawdown cooldown {algo.drawdown_cooldown}h")
            return
    algo.peak_value = max(algo.peak_value, val)
    dd = (algo.peak_value - val) / algo.peak_value if algo.peak_value > 0 else 0
    if dd > algo.max_drawdown_limit:
        algo.drawdown_cooldown = algo.cooldown_hours
        log_skip(algo, f"drawdown {dd:.1%} > limit")
        return

    if algo.consecutive_losses >= algo.max_consecutive_losses:
        algo.drawdown_cooldown                  = algo.cooldown_hours
        algo._consecutive_loss_halve_remaining  = 3
        algo.consecutive_losses                 = 0
        log_skip(algo, "max consecutive losses — cooldown")
        return

    if algo.circuit_breaker_expiry is not None and algo.Time < algo.circuit_breaker_expiry:
        log_skip(algo, "circuit breaker active")
        return

    pos_count = get_actual_position_count(algo)
    if pos_count >= algo.max_positions:
        return

    if len(algo.Transactions.GetOpenOrders()) >= algo.max_concurrent_open_orders:
        log_skip(algo, "too many open orders")
        return

    # ── Breadth gate: skip new entries in very weak participation environments ──
    breadth = float(getattr(algo, 'market_breadth', 0.5))
    if breadth < BREADTH_MIN_ENTRY and algo.market_regime != 'bull':
        log_skip(algo, f"weak breadth {breadth:.0%}")
        return

    # ── Evaluate universe: collect IgnitionBreakout candidates ───────────────
    candidates        = []
    count_evaluated   = 0
    reject_reasons    = {}

    for symbol in list(algo.crypto_data.keys()):
        if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in algo._session_blacklist:
            continue
        if (symbol.Value in algo._symbol_entry_cooldowns
                and algo.Time < algo._symbol_entry_cooldowns[symbol.Value]):
            continue
        if has_open_orders(algo, symbol):
            continue
        if is_invested_not_dust(algo, symbol):
            continue
        if not spread_ok(algo, symbol):
            reject_reasons['spread'] = reject_reasons.get('spread', 0) + 1
            continue

        crypto = algo.crypto_data[symbol]
        if not is_ready(crypto):
            continue

        # Arming-state gate: symbol must be in READY (or ARMING as fallback)
        # to be eligible for a trigger.  TRIGGERED/COOLDOWN skip silently.
        arm_state = crypto.get('arm_state', 'DORMANT')
        if arm_state in ('DORMANT', 'INVALIDATED', 'COOLDOWN'):
            reject_reasons['not_armed'] = reject_reasons.get('not_armed', 0) + 1
            continue
        if arm_state == 'TRIGGERED' and not is_invested_not_dust(algo, symbol):
            # Trade may have been closed; reset to DORMANT so it can re-arm
            crypto['arm_state']      = 'DORMANT'
            crypto['arm_state_bars'] = 0
            continue

        count_evaluated += 1

        # Dollar-volume filter (pre-screen before setup evaluation)
        dv_list = list(crypto.get('dollar_volume', []))
        if len(dv_list) >= 6:
            recent_dv = float(np.mean(dv_list[-6:]))
            if recent_dv < MIN_CANDIDATE_DOLLAR_VOLUME:
                reject_reasons['dollar_volume'] = reject_reasons.get('dollar_volume', 0) + 1
                continue

        setup_type, confidence, components = algo._scoring_engine.evaluate_setup(crypto)
        if setup_type is None:
            reason = components.get('_reject', 'no_setup')
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue

        # Bear-regime gate: tighter confidence requirement
        if algo.market_regime == "bear" and confidence < BEAR_REGIME_MIN_CONFIDENCE:
            reject_reasons['bear_regime'] = reject_reasons.get('bear_regime', 0) + 1
            continue

        spread_at_entry = get_spread_pct(algo, symbol)

        candidates.append({
            'symbol':          symbol,
            'setup_type':      setup_type,
            'confidence':      confidence,
            'net_score':       confidence,
            'factors':         components,
            'volatility':      (float(crypto['volatility'][-1])
                                if len(crypto.get('volatility', [])) > 0 else 0.05),
            'dollar_volume':   dv_list[-6:] if len(dv_list) >= 6 else dv_list,
            'spread_at_entry': spread_at_entry,
        })

    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        cash = algo.Portfolio.Cash

    # ── No candidates: log periodically, not every minute ────────────────────
    _now       = algo.Time
    _last_log  = getattr(algo, '_last_rebalance_log_time', None)
    _log_every = 10   # minutes between summary logs when nothing qualifies
    _should_log = (
        _last_log is None
        or (_now - _last_log).total_seconds() >= _log_every * 60
        or len(candidates) > 0
    )

    if not candidates:
        if _should_log:
            top_rejects = sorted(reject_reasons.items(), key=lambda x: -x[1])[:3]
            reject_str  = " | ".join(f"{k}:{v}" for k, v in top_rejects)
            algo.Debug(
                f"SCAN: {count_evaluated} evaluated, 0 qualified "
                f"({reject_str}) | {algo.market_regime} | ${cash:.0f}"
            )
            algo._last_rebalance_log_time = _now
        algo._last_skip_reason = "no_candidates"
        return

    # ── Rank by leadership quality and take top candidates ────────────────────
    candidates.sort(key=_leadership_rank, reverse=True)

    # Log top candidates (always when we have any)
    algo.Debug(
        f"CANDIDATES: {len(candidates)} qualified | "
        f"taking top {min(MAX_NEW_POSITIONS_PER_CYCLE, len(candidates))} | "
        f"{algo.market_regime}/{algo.volatility_regime} | ${cash:.0f}"
    )
    for cand in candidates[:3]:
        f = cand['factors']
        algo.Debug(
            f"  {cand['symbol'].Value} conf={cand['confidence']:.2f} "
            f"rs={f.get('rs_short', 0):.4f}+{f.get('rs_medium', 0):.4f} "
            f"vol={f.get('vol_ratio', 0):.1f}x fresh={f.get('freshness', 99)}"
        )

    algo._last_rebalance_log_time = _now
    algo._last_skip_reason        = None

    execute_trades(algo, candidates[:MAX_NEW_POSITIONS_PER_CYCLE])


# ─────────────────────────────────────────────────────────────────────────────
# execute_trades — place orders for selected candidates
# ─────────────────────────────────────────────────────────────────────────────

def execute_trades(algo, candidates):
    if not algo._positions_synced:
        return
    if algo.LiveMode and algo.kraken_status in ("maintenance", "cancel_only"):
        return
    cancel_stale_new_orders(algo)
    if len(algo.Transactions.GetOpenOrders()) >= algo.max_concurrent_open_orders:
        return
    if compute_portfolio_risk_estimate(algo) > algo.portfolio_vol_cap:
        return

    try:
        available_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        available_cash = algo.Portfolio.Cash

    open_buy_orders_value = get_open_buy_orders_value(algo)

    if available_cash <= 0:
        debug_limited(algo, f"SKIP TRADES: No cash (${available_cash:.2f})")
        return
    if open_buy_orders_value > available_cash * algo.open_orders_cash_threshold:
        debug_limited(algo, f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved")
        return

    success_count = 0

    for cand in candidates:
        if algo.daily_trade_count >= algo.max_daily_trades:
            break
        if get_actual_position_count(algo) >= algo.max_positions:
            break
        if success_count >= MAX_NEW_POSITIONS_PER_CYCLE:
            break

        sym        = cand['symbol']
        confidence = cand.get('confidence', 0.55)
        setup_type = cand.get('setup_type', 'IgnitionBreakout')

        # Final per-symbol checks
        if sym in algo._pending_orders and algo._pending_orders[sym] > 0:
            continue
        if has_open_orders(algo, sym):
            continue
        if is_invested_not_dust(algo, sym):
            continue
        if not spread_ok(algo, sym):
            continue
        if sym in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[sym]:
            continue
        if (sym.Value in algo._symbol_entry_cooldowns
                and algo.Time < algo._symbol_entry_cooldowns[sym.Value]):
            continue
        if sym in algo._symbol_loss_cooldowns and algo.Time < algo._symbol_loss_cooldowns[sym]:
            continue
        if not check_correlation(algo, sym):
            algo.Debug(f"SKIP {sym.Value}: correlated with existing position")
            continue

        sec   = algo.Securities[sym]
        price = sec.Price
        if price is None or price <= 0:
            continue
        if price < algo.min_price_usd:
            continue

        try:
            available_cash = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = algo.Portfolio.Cash

        # Reserve for pending exit fees
        pending_exit_fees = 0
        for _exit_sym in list(algo.entry_prices.keys()):
            if is_invested_not_dust(algo, _exit_sym):
                _hval = abs(algo.Portfolio[_exit_sym].Quantity) * algo.Securities[_exit_sym].Price
                pending_exit_fees += _hval * 0.004
        available_cash = max(0, available_cash - open_buy_orders_value - pending_exit_fees)
        total_value    = algo.Portfolio.TotalPortfolioValue
        fee_reserve    = max(total_value * algo.cash_reserve_pct, 0.50)
        reserved_cash  = available_cash - fee_reserve
        if reserved_cash <= 0:
            continue

        min_qty          = get_min_quantity(algo, sym)
        min_notional_usd = get_min_notional_usd(algo, sym)
        if min_qty * price > reserved_cash * 0.90:
            continue

        crypto = algo.crypto_data.get(sym)
        if not crypto:
            continue
        if crypto.get('trade_count_today', 0) >= algo.max_symbol_trades_per_day:
            continue

        # Expected-move check (uses algo's ATR parameters)
        atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
        if atr_val and price > 0:
            expected_move_pct = (atr_val * algo.atr_tp_mult) / price
            spread_cost       = cand.get('spread_at_entry') or 0.004
            min_required      = (algo.expected_round_trip_fees
                                 + algo.fee_slippage_buffer
                                 + algo.min_expected_profit_pct
                                 + spread_cost)
            if expected_move_pct < min_required:
                algo.Debug(f"SKIP {sym.Value}: expected move {expected_move_pct:.2%} < {min_required:.2%}")
                continue

        # ── Concentrated position sizing (two tiers) ──────────────────────────
        vol  = annualized_vol(algo, crypto)
        size = algo._scoring_engine.calculate_position_size(confidence, algo.entry_threshold, vol)

        # Post-loss halving
        if algo._consecutive_loss_halve_remaining > 0:
            size *= 0.50

        # Slippage penalty
        slippage_penalty = get_slippage_penalty(algo, sym)
        size *= slippage_penalty

        val = reserved_cash * size
        val = max(val, algo.min_notional)
        val = min(val, algo.max_position_usd)

        qty = round_quantity(algo, sym, val / price)
        if qty < min_qty:
            qty = round_quantity(algo, sym, min_qty)
            val = qty * price

        total_cost_with_fee = val * 1.006
        if total_cost_with_fee > available_cash:
            continue
        if (val < min_notional_usd * algo.min_notional_fee_buffer
                or val < algo.min_notional
                or val > reserved_cash):
            continue

        # Min order size compliance
        try:
            sec_props      = algo.Securities[sym].SymbolProperties
            min_order_size = float(sec_props.MinimumOrderSize or 0)
            lot_size       = float(sec_props.LotSize or 0)
            actual_min     = max(min_order_size, lot_size)
            if actual_min > 0 and qty < actual_min:
                algo.Debug(f"REJECT {sym.Value}: qty={qty} < min_order_size={actual_min}")
                continue
            if min_order_size > 0:
                post_fee_qty = qty * (1.0 - KRAKEN_SELL_FEE_BUFFER)
                if post_fee_qty < min_order_size:
                    required_qty = round_quantity(
                        algo, sym, min_order_size / (1.0 - KRAKEN_SELL_FEE_BUFFER)
                    )
                    if required_qty * price <= available_cash * 0.99:
                        qty = required_qty
                        val = qty * price
                    else:
                        continue
        except Exception as e:
            algo.Debug(f"Warning: min_order_size check failed for {sym.Value}: {e}")

        # ── Place order ────────────────────────────────────────────────────────
        try:
            ticket = place_limit_or_market(algo, sym, qty, timeout_seconds=30,
                                           tag=f"Entry:{setup_type}")
            if ticket is not None:
                algo._recent_tickets.append(ticket)
                components = cand.get('factors', {})
                f = components
                algo.Debug(
                    f"ENTRY [{setup_type}] {sym.Value} "
                    f"conf={confidence:.2f} ${val:.0f} "
                    f"rs={f.get('rs_short', 0):.4f}+{f.get('rs_medium', 0):.4f} "
                    f"vol={f.get('vol_ratio', 0):.1f}x fresh={f.get('freshness', 99)} "
                    f"spread={cand.get('spread_at_entry') or 0:.3%} "
                    f"regime={algo.market_regime}/{algo.volatility_regime}"
                )

                # Record in diagnostics
                diag = getattr(algo, '_diagnostics', None)
                if diag is not None:
                    sp       = cand.get('spread_at_entry')
                    est_slip = getattr(algo, '_last_estimated_slippage', None)
                    diag.record_entry(sym, setup_type, confidence, components,
                                      price, spread_at_entry=sp,
                                      estimated_slippage=est_slip)

                # Mark arming state as TRIGGERED
                if crypto:
                    get_arming_state_machine().mark_triggered(crypto)

                success_count += 1
                algo.trade_count += 1
                crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1

                if algo._consecutive_loss_halve_remaining > 0:
                    algo._consecutive_loss_halve_remaining -= 1
                if algo.LiveMode:
                    algo._last_live_trade_time = algo.Time
        except Exception as e:
            algo.Debug(f"ORDER FAILED: {sym.Value} — {e}")
            algo._session_blacklist.add(sym.Value)
            continue

        if algo.LiveMode and success_count >= 2:
            break

    if success_count > 0:
        debug_limited(algo, f"EXECUTED: {success_count}/{len(candidates)} entries")
