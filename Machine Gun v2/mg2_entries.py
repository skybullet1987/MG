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

Also owns trade attribution (DiagnosticsEngine / TradeRecord) so that
per-trade metadata lives alongside the entry logic that generates it.
"""

# During a bear market regime, require higher setup confidence
BEAR_REGIME_MIN_CONFIDENCE = 0.70

# Max candidates to execute per rebalance cycle (the "concentrated" part)
MAX_NEW_POSITIONS_PER_CYCLE = 2

# Minimum dollar volume required for a symbol to be a valid candidate
MIN_CANDIDATE_DOLLAR_VOLUME = 80_000   # 80k USD per bar (on top of universe filter)

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
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

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
                    realized_slippage=None, spread_at_exit=None):
        algo = self.algo
        rec  = self._open_records.pop(symbol, None)
        if rec is None:
            return

        rec.exit_time              = algo.Time
        rec.exit_price             = exit_price
        rec.exit_tag               = exit_tag
        rec.spread_at_exit         = spread_at_exit
        rec.realized_slippage_exit = realized_slippage

        if rec.entry_price and rec.entry_price > 0:
            rec.gross_pnl_pct = (exit_price - rec.entry_price) / rec.entry_price
            rec.net_pnl_pct   = rec.net_pnl_estimate(self.ROUND_TRIP_FEE_PCT)

        if rec.entry_time and rec.exit_time:
            rec.hold_minutes = (rec.exit_time - rec.entry_time).total_seconds() / 60.0

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
    """
    f = cand.get('factors', {})
    rs_short  = float(f.get('rs_short',  0.0))
    rs_medium = float(f.get('rs_medium', 0.0))
    vol_ratio = float(f.get('vol_ratio', 0.0))
    freshness = int(f.get('freshness',   5))
    conf      = float(cand.get('net_score', 0.0))

    fresh_score = 1.0 if freshness == 0 else (0.7 if freshness == 1 else 0.4)

    return (
        (rs_short + rs_medium) * 0.35 +
        min(vol_ratio / 10.0, 1.0)    * 0.25 +
        fresh_score                    * 0.20 +
        conf                           * 0.20
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point: rebalance
# ─────────────────────────────────────────────────────────────────────────────

def rebalance(algo):
    if algo.IsWarmingUp:
        return

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
