# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
import numpy as np
from datetime import timedelta
from mg2_data import is_ready, annualized_vol, compute_portfolio_risk_estimate
# endregion

"""
mg2_entries.py — Setup-Driven Entry Logic for Machine Gun v2

Also owns trade attribution (DiagnosticsEngine / TradeRecord) so that
per-trade metadata is kept alongside the entry logic that generates it.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Trade Attribution — DiagnosticsEngine
# ─────────────────────────────────────────────────────────────────────────────

class TradeRecord:
    """Full metadata record for one completed round-trip trade."""

    __slots__ = [
        # Identity
        'symbol', 'entry_time', 'exit_time',
        # Setup attribution
        'setup_type', 'confidence',
        # Entry signal components (dict: name → value)
        'entry_components',
        # Market context at entry
        'market_regime', 'volatility_regime',
        'btc_5bar_return', 'rs_vs_btc',
        # Execution quality
        'spread_at_entry', 'spread_at_exit',
        'estimated_slippage_entry', 'realized_slippage_exit',
        # Performance
        'entry_price', 'exit_price',
        'gross_pnl_pct',  # (exit - entry) / entry
        'net_pnl_pct',    # gross minus round-trip fee estimate
        # Exit
        'exit_tag', 'hold_minutes',
        # Misc
        'breakout_freshness', 'atr_at_entry',
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def to_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def net_pnl_estimate(self, fee_pct=0.006):
        """Estimate net PnL after round-trip fees if not already set."""
        if self.gross_pnl_pct is not None:
            slip = (self.estimated_slippage_entry or 0.0) + (self.realized_slippage_exit or 0.0)
            return self.gross_pnl_pct - fee_pct - slip
        return None


class DiagnosticsEngine:
    """
    Lightweight trade attribution engine.
    Stores completed trade records and provides summary statistics.
    """

    # Fee estimate for net PnL calculation (round-trip: entry + exit, taker)
    ROUND_TRIP_FEE_PCT = 0.008  # 0.4% × 2 taker fills = 0.8% (conservative)

    def __init__(self, algo):
        self.algo = algo
        self._open_records = {}   # symbol → TradeRecord (in-progress)
        self.completed = []       # list of completed TradeRecord

    # ── Entry recording ───────────────────────────────────────────────────────

    def record_entry(self, symbol, setup_type, confidence, components,
                     entry_price, spread_at_entry=None, estimated_slippage=None):
        algo = self.algo
        rec = TradeRecord()
        rec.symbol        = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
        rec.entry_time    = algo.Time
        rec.setup_type    = setup_type or 'Unknown'
        rec.confidence    = confidence
        rec.entry_components = {k: v for k, v in components.items()
                                if not k.startswith('_') and k != 'setup_type'}
        rec.market_regime     = algo.market_regime
        rec.volatility_regime = algo.volatility_regime
        rec.spread_at_entry   = spread_at_entry
        rec.estimated_slippage_entry = estimated_slippage
        rec.entry_price       = entry_price

        btc_rets = list(algo.btc_returns)
        if len(btc_rets) >= 5:
            rec.btc_5bar_return = float(sum(btc_rets[-5:]))
        else:
            rec.btc_5bar_return = None

        crypto = algo.crypto_data.get(symbol)
        if crypto:
            rs_hist = list(crypto.get('rs_vs_btc', []))
            rec.rs_vs_btc = float(rs_hist[-1]) if rs_hist else None
            rec.breakout_freshness = crypto.get('breakout_freshness', None)
            rec.atr_at_entry = (float(crypto['atr'].Current.Value)
                                if crypto.get('atr') and crypto['atr'].IsReady else None)

        self._open_records[symbol] = rec

    # ── Exit recording ────────────────────────────────────────────────────────

    def record_exit(self, symbol, exit_price, exit_tag,
                    realized_slippage=None, spread_at_exit=None):
        algo = self.algo
        rec = self._open_records.pop(symbol, None)
        if rec is None:
            return

        rec.exit_time  = algo.Time
        rec.exit_price = exit_price
        rec.exit_tag   = exit_tag
        rec.spread_at_exit = spread_at_exit
        rec.realized_slippage_exit = realized_slippage

        if rec.entry_price and rec.entry_price > 0:
            rec.gross_pnl_pct = (exit_price - rec.entry_price) / rec.entry_price
            rec.net_pnl_pct   = rec.net_pnl_estimate(self.ROUND_TRIP_FEE_PCT)

        if rec.entry_time and rec.exit_time:
            rec.hold_minutes = (rec.exit_time - rec.entry_time).total_seconds() / 60.0

        self.completed.append(rec)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats_by_setup(self):
        by_setup = {}
        for rec in self.completed:
            st = rec.setup_type or 'Unknown'
            if st not in by_setup:
                by_setup[st] = []
            by_setup[st].append(rec)

        result = {}
        for st, records in by_setup.items():
            pnls_net = [r.net_pnl_pct for r in records if r.net_pnl_pct is not None]
            wins = [p for p in pnls_net if p > 0]
            losses = [p for p in pnls_net if p <= 0]
            result[st] = {
                'count':       len(records),
                'win_rate':    len(wins) / len(pnls_net) if pnls_net else 0.0,
                'avg_win':     float(np.mean(wins))   if wins   else 0.0,
                'avg_loss':    float(np.mean(losses)) if losses else 0.0,
                'avg_net_pnl': float(np.mean(pnls_net)) if pnls_net else 0.0,
                'total_net_pnl': float(sum(pnls_net)) if pnls_net else 0.0,
                'profit_factor': (sum(wins) / abs(sum(losses))
                                  if losses and sum(losses) != 0 else float('inf')),
                'avg_hold_min': float(np.mean([r.hold_minutes for r in records
                                               if r.hold_minutes is not None])),
            }
        return result

    def stats_by_exit(self):
        by_exit = {}
        for rec in self.completed:
            tag = rec.exit_tag or 'Unknown'
            if tag not in by_exit:
                by_exit[tag] = []
            by_exit[tag].append(rec)

        result = {}
        for tag, records in by_exit.items():
            pnls = [r.net_pnl_pct for r in records if r.net_pnl_pct is not None]
            result[tag] = {
                'count':       len(records),
                'avg_net_pnl': float(np.mean(pnls)) if pnls else 0.0,
                'total_net_pnl': float(sum(pnls)) if pnls else 0.0,
                'win_rate':    (sum(1 for p in pnls if p > 0) / len(pnls)
                                if pnls else 0.0),
            }
        return result

    def stats_by_regime(self):
        by_regime = {}
        for rec in self.completed:
            regime = rec.market_regime or 'unknown'
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(rec)
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
        n = len(self.completed)
        if n == 0:
            algo.Debug("DIAGNOSTICS: No completed trades to report.")
            return

        all_pnls = [r.net_pnl_pct for r in self.completed if r.net_pnl_pct is not None]
        wins  = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]

        algo.Debug("=" * 60)
        algo.Debug(f"DIAGNOSTICS SUMMARY — {n} completed trades")
        if all_pnls:
            algo.Debug(f"  Win rate:       {len(wins)/len(all_pnls):.1%}")
            algo.Debug(f"  Avg net PnL:    {float(np.mean(all_pnls)):.2%}")
        if wins:
            algo.Debug(f"  Avg win:        {float(np.mean(wins)):.2%}")
        if losses:
            algo.Debug(f"  Avg loss:       {float(np.mean(losses)):.2%}")
        if wins and losses:
            pf = sum(wins) / abs(sum(losses))
            algo.Debug(f"  Profit factor:  {pf:.2f}")

        algo.Debug("─── By Setup Type ───")
        for st, stats in self.stats_by_setup().items():
            algo.Debug(
                f"  {st}: n={stats['count']} wr={stats['win_rate']:.1%} "
                f"avg={stats['avg_net_pnl']:+.2%} pf={stats['profit_factor']:.2f} "
                f"hold={stats['avg_hold_min']:.0f}min"
            )

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

        algo.Debug("=" * 60)

    def print_open_positions(self):
        """Log currently open (un-exited) trade records."""
        algo = self.algo
        if not self._open_records:
            return
        algo.Debug(f"DIAGNOSTICS: {len(self._open_records)} open positions tracked:")
        for sym, rec in self._open_records.items():
            sym_name = sym.Value if hasattr(sym, 'Value') else str(sym)
            price = algo.Securities[sym].Price if sym in algo.Securities else 0
            if rec.entry_price and rec.entry_price > 0 and price > 0:
                unreal_pnl = (price - rec.entry_price) / rec.entry_price
                algo.Debug(f"  {sym_name}: setup={rec.setup_type} conf={rec.confidence:.2f} "
                           f"entry=${rec.entry_price:.4f} now=${price:.4f} "
                           f"pnl={unreal_pnl:+.2%} hold={rec.hold_minutes or '?'}min")


mg2_entries.py — Setup-Driven Entry Logic for Machine Gun v2
============================================================
Entry logic is now driven by explicit named setups (IgnitionBreakout,
CompressionExpansion, MomentumContinuation) instead of an additive
"score soup."  Mean-reversion entries have been removed.

Changes from previous version
------------------------------
• ``calculate_factor_scores()`` now calls ``engine.evaluate_setup()`` which
  returns (setup_type, confidence, components) — a named setup or nothing.
• A candidate is only accepted if a named setup qualifies (not a partial sum).
• Diagnostic metadata (setup_type, spread, estimated_slippage, regime) is
  attached to every candidate and forwarded to DiagnosticsEngine on fill.
• Score persistence check is preserved: 2 of last 3 bars must pass threshold.
• Bear-market entry gating: no entries during bear regime unless confidence
  is very high (>= BEAR_REGIME_MIN_CONFIDENCE).
"""

# During a bear market regime, only trade very high-confidence setups
BEAR_REGIME_MIN_CONFIDENCE = 0.70


def _normalize(v, mn, mx):
    if mx - mn <= 0:
        return 0.5
    return max(0, min(1, (v - mn) / (mx - mn)))


def log_skip(algo, reason):
    if algo.LiveMode:
        debug_limited(algo, f"Rebalance skip: {reason}")
        algo._last_skip_reason = reason
    elif reason != algo._last_skip_reason:
        debug_limited(algo, f"Rebalance skip: {reason}")
        algo._last_skip_reason = reason


def calculate_factor_scores(algo, symbol, crypto):
    """
    Evaluate momentum-breakout setups for a symbol.

    Returns a components dict that includes:
    - '_scalp_score'  : float confidence (0-1)
    - '_direction'    : 1 (long-only)
    - '_long_score'   : same as confidence
    - 'setup_type'    : str name of the winning setup or None
    - all setup-specific component keys

    Returns an empty dict if no setup qualifies or data is insufficient.
    """
    setup_type, confidence, components = algo._scoring_engine.evaluate_setup(crypto)

    if setup_type is None or confidence == 0.0:
        return {}   # No valid setup — do not enter

    # Spread penalty: widen spread → reduce effective confidence
    sp = get_spread_pct(algo, symbol)
    if sp is not None and sp > 0:
        spread_penalty = min((sp / 0.006) * 0.12, 0.15)
        confidence = max(0.0, confidence * (1.0 - spread_penalty))

    # Bear-regime gate: higher bar during downtrend
    if algo.market_regime == "bear" and confidence < BEAR_REGIME_MIN_CONFIDENCE:
        return {}

    result = dict(components)
    result['_scalp_score']  = confidence
    result['_direction']    = 1
    result['_long_score']   = confidence
    result['_spread_at_entry'] = sp
    return result


def calculate_composite_score(algo, factors, crypto=None):
    """Return the pre-computed confidence from the setup evaluator."""
    return factors.get('_scalp_score', 0.0)


def apply_fee_adjustment(algo, score):
    """Return score unchanged – entry thresholds already require meaningful moves."""
    return score


def check_correlation(algo, new_symbol):
    """Reject candidate if it is too correlated with any existing position."""
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


def daily_loss_exceeded(algo):
    """Returns True if the portfolio has dropped >= 4% from today's open value."""
    if algo._daily_open_value is None or algo._daily_open_value <= 0:
        return False
    current = algo.Portfolio.TotalPortfolioValue
    if current <= 0:
        return True
    drop = (algo._daily_open_value - current) / algo._daily_open_value
    return drop >= 0.04   # raised from 3% to give the algo more room to run


def rebalance(algo):
    if algo.IsWarmingUp:
        return

    if daily_loss_exceeded(algo):
        log_skip(algo, "max daily loss exceeded")
        return

    # BTC dump filter: don't enter during rapid BTC sell-off
    if len(algo.btc_returns) >= 5 and sum(list(algo.btc_returns)[-5:]) < -0.012:
        log_skip(algo, "BTC dumping")
        return

    if algo._cash_mode_until is not None and algo.Time < algo._cash_mode_until:
        log_skip(algo, "cash mode - poor recent performance")
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
            algo.peak_value = val
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
        algo.drawdown_cooldown = 3
        algo._consecutive_loss_halve_remaining = 3
        algo.consecutive_losses = 0
        log_skip(algo, "consecutive loss cooldown (5 losses)")
        return
    if algo.consecutive_losses >= 4:
        algo.circuit_breaker_expiry = algo.Time + timedelta(hours=1)
        algo.consecutive_losses = 0
        log_skip(algo, "circuit breaker triggered (4 consecutive losses)")
        return
    if algo.circuit_breaker_expiry is not None and algo.Time < algo.circuit_breaker_expiry:
        log_skip(algo, "circuit breaker active")
        return

    pos_count = get_actual_position_count(algo)
    if pos_count >= algo.max_positions:
        log_skip(algo, "at max positions")
        return

    fg_value = getattr(algo, 'fear_greed_value', 50)
    if fg_value >= 85:
        effective_max_pos = max(1, algo.max_positions // 2)
        if pos_count >= effective_max_pos:
            log_skip(algo, f"Fear&Greed extreme greed ({fg_value}) — reduced max positions")
            return

    if len(algo.Transactions.GetOpenOrders()) >= algo.max_concurrent_open_orders:
        log_skip(algo, "too many open orders")
        return

    # ── Score every symbol that has a valid setup ──────────────────────────
    count_scored        = 0
    count_above_thresh  = 0
    scores              = []
    threshold_now       = algo.entry_threshold  # now interpreted as min confidence

    for symbol in list(algo.crypto_data.keys()):
        if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in algo._session_blacklist:
            continue
        if (symbol.Value in algo._symbol_entry_cooldowns
                and algo.Time < algo._symbol_entry_cooldowns[symbol.Value]):
            continue
        if has_open_orders(algo, symbol):
            continue
        if not spread_ok(algo, symbol):
            continue

        crypto = algo.crypto_data[symbol]
        if not is_ready(crypto):
            continue

        factor_scores = calculate_factor_scores(algo, symbol, crypto)
        if not factor_scores:
            continue
        count_scored += 1

        composite_score = calculate_composite_score(algo, factor_scores, crypto)
        net_score       = apply_fee_adjustment(algo, composite_score)

        crypto['recent_net_scores'].append(net_score)

        if net_score >= threshold_now:
            # Require setup to persist for at least 2 of the last 3 bars
            if len(crypto['recent_net_scores']) >= 3:
                above_count = sum(1 for s in list(crypto['recent_net_scores'])[-3:]
                                  if s >= threshold_now)
                if above_count < 2:
                    continue
            count_above_thresh += 1
            scores.append({
                'symbol':          symbol,
                'setup_type':      factor_scores.get('setup_type'),
                'composite_score': composite_score,
                'net_score':       net_score,
                'factors':         factor_scores,
                'volatility':      (crypto['volatility'][-1]
                                    if len(crypto['volatility']) > 0 else 0.05),
                'dollar_volume':   (list(crypto['dollar_volume'])[-6:]
                                    if len(crypto['dollar_volume']) >= 6 else []),
                'spread_at_entry': factor_scores.get('_spread_at_entry'),
            })

    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        cash = algo.Portfolio.Cash

    debug_limited(algo, (f"REBALANCE: {count_above_thresh}/{count_scored} setup-qualified "
                         f"thresh={threshold_now:.2f} | cash=${cash:.2f}"))

    if not scores:
        log_skip(algo, "no setup candidates qualified")
        return

    # Sort by confidence descending (highest-conviction first)
    scores.sort(key=lambda x: x['net_score'], reverse=True)
    algo._last_skip_reason = None
    execute_trades(algo, scores, threshold_now)


def execute_trades(algo, candidates, threshold_now):
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
        debug_limited(algo, f"SKIP TRADES: No cash available (${available_cash:.2f})")
        return
    if open_buy_orders_value > available_cash * algo.open_orders_cash_threshold:
        debug_limited(algo, (f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved "
                             f"(>{algo.open_orders_cash_threshold:.0%} of ${available_cash:.2f})"))
        return

    reject_pending_orders    = 0
    reject_open_orders       = 0
    reject_already_invested  = 0
    reject_spread            = 0
    reject_exit_cooldown     = 0
    reject_loss_cooldown     = 0
    reject_correlation       = 0
    reject_price_invalid     = 0
    reject_price_too_low     = 0
    reject_cash_reserve      = 0
    reject_min_qty_too_large = 0
    reject_dollar_volume     = 0
    reject_notional          = 0
    success_count            = 0

    for cand in candidates:
        if algo.daily_trade_count >= algo.max_daily_trades:
            break
        if get_actual_position_count(algo) >= algo.max_positions:
            break

        sym        = cand['symbol']
        net_score  = cand.get('net_score', 0.5)
        setup_type = cand.get('setup_type', 'Unknown')

        if sym in algo._pending_orders and algo._pending_orders[sym] > 0:
            reject_pending_orders += 1
            continue
        if has_open_orders(algo, sym):
            reject_open_orders += 1
            continue
        if is_invested_not_dust(algo, sym):
            reject_already_invested += 1
            continue
        if not spread_ok(algo, sym):
            reject_spread += 1
            continue

        # Live: depth check
        if algo.LiveMode:
            _crypto_depth = algo.crypto_data.get(sym)
            if _crypto_depth:
                bid_size = _crypto_depth.get('bid_size', 0)
                if bid_size > 0:
                    _sec_depth   = algo.Securities[sym]
                    _price_depth = _sec_depth.Price if _sec_depth.Price > 0 else 1
                    _estimated_val = algo.Portfolio.TotalPortfolioValue * 0.35
                    _our_qty = _estimated_val / _price_depth
                    if _our_qty > bid_size * 0.20:
                        continue

        if sym in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[sym]:
            reject_exit_cooldown += 1
            continue
        if (sym.Value in algo._symbol_entry_cooldowns
                and algo.Time < algo._symbol_entry_cooldowns[sym.Value]):
            reject_loss_cooldown += 1
            continue
        if sym in algo._symbol_loss_cooldowns and algo.Time < algo._symbol_loss_cooldowns[sym]:
            reject_loss_cooldown += 1
            continue
        if not check_correlation(algo, sym):
            reject_correlation += 1
            continue

        sec   = algo.Securities[sym]
        price = sec.Price
        if price is None or price <= 0:
            reject_price_invalid += 1
            continue
        if price < algo.min_price_usd:
            reject_price_too_low += 1
            continue

        try:
            available_cash = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = algo.Portfolio.Cash

        # Reserve for pending exit fees
        pending_exit_fees = 0
        for _exit_sym in list(algo.entry_prices.keys()):
            if is_invested_not_dust(algo, _exit_sym):
                _holding_val = abs(algo.Portfolio[_exit_sym].Quantity) * algo.Securities[_exit_sym].Price
                pending_exit_fees += _holding_val * 0.004
        available_cash = max(0, available_cash - open_buy_orders_value - pending_exit_fees)
        total_value    = algo.Portfolio.TotalPortfolioValue
        fee_reserve    = max(total_value * algo.cash_reserve_pct, 0.50)
        reserved_cash  = available_cash - fee_reserve
        if reserved_cash <= 0:
            reject_cash_reserve += 1
            continue

        min_qty         = get_min_quantity(algo, sym)
        min_notional_usd = get_min_notional_usd(algo, sym)
        if min_qty * price > reserved_cash * 0.90:
            reject_min_qty_too_large += 1
            continue

        crypto = algo.crypto_data.get(sym)
        if not crypto:
            continue
        if crypto.get('trade_count_today', 0) >= algo.max_symbol_trades_per_day:
            continue

        # Expected move filter
        atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
        if atr_val and price > 0:
            expected_move_pct = (atr_val * algo.atr_tp_mult) / price
            spread = get_spread_pct(algo, sym)
            spread_cost = spread if spread is not None else 0.004
            min_required = (algo.expected_round_trip_fees
                            + algo.fee_slippage_buffer
                            + algo.min_expected_profit_pct
                            + spread_cost)
            if expected_move_pct < min_required:
                continue

        # Dollar-volume filter
        if len(crypto['dollar_volume']) >= 3:
            dv_window = min(len(crypto['dollar_volume']), 12)
            recent_dv = np.mean(list(crypto['dollar_volume'])[-dv_window:])
            if recent_dv < algo.min_dollar_volume_usd:
                reject_dollar_volume += 1
                continue

        # ── Position sizing ────────────────────────────────────────────────
        vol  = annualized_vol(algo, crypto)
        size = algo._scoring_engine.calculate_position_size(net_score, threshold_now, vol)

        # Post-loss halving
        if algo._consecutive_loss_halve_remaining > 0:
            size *= 0.50

        # Slippage penalty
        slippage_penalty = get_slippage_penalty(algo, sym)
        size *= slippage_penalty

        # Correlation-based sizing reduction with existing positions
        existing_count = get_actual_position_count(algo)
        if existing_count >= 2:
            max_corr = 0
            if crypto and len(crypto['returns']) >= 12:
                new_rets = list(crypto['returns'])[-12:]
                for existing_sym in list(algo.entry_prices.keys()):
                    if existing_sym == sym:
                        continue
                    existing_crypto = algo.crypto_data.get(existing_sym)
                    if existing_crypto and len(existing_crypto['returns']) >= 12:
                        exist_rets = list(existing_crypto['returns'])[-12:]
                        try:
                            corr = abs(np.corrcoef(new_rets, exist_rets)[0, 1])
                            max_corr = max(max_corr, corr)
                        except Exception:
                            pass
            if max_corr > 0.5:
                size *= (1.0 - max_corr)

        val = reserved_cash * size
        val = max(val, algo.min_notional)
        val = min(val, algo.max_position_usd)

        qty = round_quantity(algo, sym, val / price)
        if qty < min_qty:
            qty = round_quantity(algo, sym, min_qty)
            val = qty * price

        total_cost_with_fee = val * 1.006
        if total_cost_with_fee > available_cash:
            reject_cash_reserve += 1
            continue
        if (val < min_notional_usd * algo.min_notional_fee_buffer
                or val < algo.min_notional
                or val > reserved_cash):
            reject_notional += 1
            continue

        # Min order size compliance
        try:
            sec_props      = algo.Securities[sym].SymbolProperties
            min_order_size = float(sec_props.MinimumOrderSize or 0)
            lot_size       = float(sec_props.LotSize or 0)
            actual_min     = max(min_order_size, lot_size)
            if actual_min > 0 and qty < actual_min:
                algo.Debug(f"REJECT ENTRY {sym.Value}: qty={qty} < min_order_size={actual_min}")
                reject_notional += 1
                continue
            if min_order_size > 0:
                post_fee_qty = qty * (1.0 - KRAKEN_SELL_FEE_BUFFER)
                if post_fee_qty < min_order_size:
                    required_qty = round_quantity(algo, sym, min_order_size / (1.0 - KRAKEN_SELL_FEE_BUFFER))
                    if required_qty * price <= available_cash * 0.99:
                        qty = required_qty
                        val = qty * price
                    else:
                        algo.Debug(f"REJECT ENTRY {sym.Value}: post-fee qty too small and can't upsize")
                        reject_notional += 1
                        continue
        except Exception as e:
            algo.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

        # ── Place order ────────────────────────────────────────────────────
        try:
            ticket = place_limit_or_market(algo, sym, qty, timeout_seconds=30,
                                           tag=f"Entry:{setup_type}")
            if ticket is not None:
                algo._recent_tickets.append(ticket)
                components = cand.get('factors', {})

                # Attribution log
                algo.Debug(
                    f"ENTRY [{setup_type}] {sym.Value} | conf={net_score:.2f} "
                    f"${val:.0f} | spread={cand.get('spread_at_entry', 0) or 0:.3%} "
                    f"regime={algo.market_regime}/{algo.volatility_regime}"
                )

                # Record in diagnostics engine if available
                diag = getattr(algo, '_diagnostics', None)
                if diag is not None:
                    sp = cand.get('spread_at_entry')
                    est_slip = getattr(algo, '_last_estimated_slippage', None)
                    diag.record_entry(sym, setup_type, net_score, components,
                                      price, spread_at_entry=sp,
                                      estimated_slippage=est_slip)

                success_count += 1
                algo.trade_count += 1
                crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1

                # Track whether entry was in a choppy regime (for exit tuning)
                adx_ind = crypto.get('adx')
                is_choppy = (adx_ind is not None and adx_ind.IsReady
                             and adx_ind.Current.Value < 20)
                algo._choppy_regime_entries[sym] = is_choppy

                if algo._consecutive_loss_halve_remaining > 0:
                    algo._consecutive_loss_halve_remaining -= 1
                if algo.LiveMode:
                    algo._last_live_trade_time = algo.Time
        except Exception as e:
            algo.Debug(f"ORDER FAILED: {sym.Value} - {e}")
            algo._session_blacklist.add(sym.Value)
            continue

        if algo.LiveMode and success_count >= 3:
            break

    if success_count > 0 or (reject_exit_cooldown + reject_loss_cooldown) > 3:
        debug_limited(algo, (f"EXECUTE: {success_count}/{len(candidates)} | rejects: "
                             f"cd={reject_exit_cooldown} loss={reject_loss_cooldown} "
                             f"corr={reject_correlation} dv={reject_dollar_volume}"))
