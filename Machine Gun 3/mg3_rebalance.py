"""
Machine Gun 3 — Rebalancing & trade-execution helper functions
==============================================================
Plain functions that implement the main entry-selection loop (``rebalance``)
and the per-candidate order-placement logic (``execute_trades``).

All functions receive the algorithm instance as ``algo`` so they can read and
write algorithm state without inheriting from QCAlgorithm.
"""
import numpy as np
from datetime import timedelta

import config as MG3Config
from execution import (
    cancel_stale_new_orders, live_safety_checks, get_actual_position_count,
    has_open_orders, spread_ok, debug_limited, SYMBOL_BLACKLIST,
    get_min_quantity, get_min_notional_usd, round_quantity,
    place_limit_or_market, get_slippage_penalty, KRAKEN_SELL_FEE_BUFFER,
    get_open_buy_orders_value, is_invested_not_dust,
)
from mg3_constants import POSITION_STATE_ENTERING


# ---------------------------------------------------------------------------
# Rebalance — main entry-selection loop
# ---------------------------------------------------------------------------

def rebalance(algo):
    if algo.IsWarmingUp:
        return

    # Cash mode — pause trading when recent performance is poor
    if algo._cash_mode_until is not None and algo.Time < algo._cash_mode_until:
        algo._log_skip("cash mode - poor recent performance")
        return

    # Reset log budget at each rebalance call for consistent logging
    algo.log_budget = 20

    # Rate-limit hard block
    if algo._rate_limit_until is not None and algo.Time < algo._rate_limit_until:
        algo._log_skip("rate limited")
        return

    # MG3: max daily loss guard
    if not algo.IsWarmingUp and algo._daily_loss_exceeded():
        algo._log_skip("max daily loss exceeded")
        return

    # Live safety checks
    if algo.LiveMode and not live_safety_checks(algo):
        return
    # Only block on explicit bad states; unknown is allowed (fallback after warmup)
    if algo.LiveMode and algo.kraken_status in ("maintenance", "cancel_only"):
        algo._log_skip("kraken not online")
        return

    cancel_stale_new_orders(algo)
    if algo.daily_trade_count >= algo._get_max_daily_trades():
        algo._log_skip("max daily trades")
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
            algo._log_skip(f"drawdown cooldown {algo.drawdown_cooldown}h")
            return

    algo.peak_value = max(algo.peak_value, val)
    dd = (algo.peak_value - val) / algo.peak_value if algo.peak_value > 0 else 0
    if dd > algo.max_drawdown_limit:
        algo.drawdown_cooldown = algo.cooldown_hours
        algo._log_skip(f"drawdown {dd:.1%} > limit")
        return

    if algo.consecutive_losses >= algo.max_consecutive_losses:
        # Pause 3h and halve size for next 5 trades
        algo.drawdown_cooldown = 3
        algo._consecutive_loss_halve_remaining = 3
        algo.consecutive_losses = 0
        algo._log_skip("consecutive loss cooldown (5 losses)")
        return

    # Circuit breaker: halt new entries for 12h after 3 consecutive losses
    if algo.consecutive_losses >= 3:
        algo.circuit_breaker_expiry = algo.Time + timedelta(hours=12)
        algo.consecutive_losses = 0
        algo._log_skip("circuit breaker triggered (3 consecutive losses)")
        return
    if algo.circuit_breaker_expiry is not None and algo.Time < algo.circuit_breaker_expiry:
        algo._log_skip("circuit breaker active")
        return

    pos_count = get_actual_position_count(algo)
    if pos_count >= algo.max_positions:
        algo._log_skip("at max positions")
        return
    if len(algo.Transactions.GetOpenOrders()) >= algo.max_concurrent_open_orders:
        algo._log_skip("too many open orders")
        return

    # Diagnostic counters for filter funnel
    count_not_blacklisted = 0
    count_no_open_orders  = 0
    count_spread_ok       = 0
    count_ready           = 0
    count_scored          = 0
    count_above_thresh    = 0

    scores        = []
    threshold_now = algo._get_threshold()

    for symbol in list(algo.crypto_data.keys()):
        if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in algo._session_blacklist:
            continue
        count_not_blacklisted += 1

        if has_open_orders(algo, symbol):
            continue
        count_no_open_orders += 1

        if not spread_ok(algo, symbol):
            continue
        count_spread_ok += 1

        crypto = algo.crypto_data[symbol]
        if not algo._is_ready(crypto):
            continue
        count_ready += 1

        factor_scores = algo._calculate_factor_scores(symbol, crypto)
        if not factor_scores:
            continue
        count_scored += 1

        composite_score = algo._calculate_composite_score(factor_scores, crypto)
        net_score       = algo._apply_fee_adjustment(composite_score)

        # Store for diagnostic purposes
        crypto['recent_net_scores'].append(net_score)

        if net_score >= threshold_now:
            count_above_thresh += 1
            scores.append({
                'symbol':          symbol,
                'composite_score': composite_score,
                'net_score':       net_score,
                'factors':         factor_scores,
                'volatility':      crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                'dollar_volume':   list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
            })

    # Log diagnostic summary
    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        cash = algo.Portfolio.Cash

    debug_limited(
        algo,
        f"REBALANCE: {count_above_thresh}/{count_scored} above "
        f"thresh={threshold_now:.2f} | cash=${cash:.2f}"
    )

    if len(scores) == 0:
        algo._log_skip("no candidates passed filters")
        return

    scores.sort(key=lambda x: x['net_score'], reverse=True)
    algo._last_skip_reason = None
    execute_trades(algo, scores, threshold_now)


# ---------------------------------------------------------------------------
# Execute trades — per-candidate order placement
# ---------------------------------------------------------------------------

def execute_trades(algo, candidates, threshold_now):
    if not algo._positions_synced:
        return
    if algo.LiveMode and algo.kraken_status in ("maintenance", "cancel_only"):
        return

    cancel_stale_new_orders(algo)
    if len(algo.Transactions.GetOpenOrders()) >= algo.max_concurrent_open_orders:
        return
    if algo._compute_portfolio_risk_estimate() > algo.portfolio_vol_cap:
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
        debug_limited(
            algo,
            f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved "
            f"(>{algo.open_orders_cash_threshold:.0%} of ${available_cash:.2f})"
        )
        return

    reject_pending_orders   = 0
    reject_open_orders      = 0
    reject_already_invested = 0
    reject_spread           = 0
    reject_exit_cooldown    = 0
    reject_loss_cooldown    = 0
    reject_price_invalid    = 0
    reject_price_too_low    = 0
    reject_cash_reserve     = 0
    reject_min_qty_too_large = 0
    reject_dollar_volume    = 0
    reject_notional         = 0
    success_count           = 0

    for cand in candidates:
        if algo.daily_trade_count >= algo._get_max_daily_trades():
            break
        if get_actual_position_count(algo) >= algo.max_positions:
            break

        sym       = cand['symbol']
        net_score = cand.get('net_score', 0.5)

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
        if sym in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[sym]:
            reject_exit_cooldown += 1
            continue
        if sym in algo._symbol_loss_cooldowns and algo.Time < algo._symbol_loss_cooldowns[sym]:
            reject_loss_cooldown += 1
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

        available_cash = max(0, available_cash - open_buy_orders_value)
        total_value    = algo.Portfolio.TotalPortfolioValue
        # Minimal fee reserve only
        fee_reserve   = max(total_value * 0.01, 0.10)
        reserved_cash = available_cash - fee_reserve
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

        # Per-symbol daily trade limit
        if crypto.get('trade_count_today', 0) >= algo.max_symbol_trades_per_day:
            continue

        # Fee-adjusted profit gate: ATR-projected move must cover fees + slippage
        atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
        if atr_val and price > 0:
            expected_move_pct = (atr_val * algo.atr_tp_mult) / price
            min_required = (
                algo.expected_round_trip_fees
                + algo.fee_slippage_buffer
                + algo.min_expected_profit_pct
            )
            if expected_move_pct < min_required:
                continue

        # Dollar-volume liquidity gate: use 12-bar average for more stable assessment
        if len(crypto['dollar_volume']) >= 3:
            dv_window  = min(len(crypto['dollar_volume']), 12)
            recent_dv  = np.mean(list(crypto['dollar_volume'])[-dv_window:])
            if recent_dv < algo.min_dollar_volume_usd:
                reject_dollar_volume += 1
                continue

        # Position sizing: base fraction, Kelly-adjusted
        vol  = algo._annualized_vol(crypto)
        size = algo._calculate_position_size(net_score, threshold_now, vol)

        # Halve size if in consecutive-loss recovery mode
        if algo._consecutive_loss_halve_remaining > 0:
            size *= 0.50

        # High-volatility regime: widen sizing slightly (vol = opportunity)
        if algo.volatility_regime == "high":
            size = min(size * 1.1, algo.position_size_pct)

        slippage_penalty = get_slippage_penalty(algo, sym)
        size *= slippage_penalty

        val = reserved_cash * size

        # Floor at min_notional to support small accounts
        val = max(val, algo.min_notional)

        # Absolute hard cap on position size
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

        # Exit feasibility check: qty must be >= MinimumOrderSize so it can be sold
        try:
            sec            = algo.Securities[sym]
            min_order_size = float(sec.SymbolProperties.MinimumOrderSize or 0)
            lot_size       = float(sec.SymbolProperties.LotSize or 0)
            actual_min     = max(min_order_size, lot_size)
            if actual_min > 0 and qty < actual_min:
                algo.Debug(
                    f"REJECT ENTRY {sym.Value}: qty={qty} < "
                    f"min_order_size={actual_min} (unsellable)"
                )
                reject_notional += 1
                continue
            # Ensure post-fee qty >= MinimumOrderSize so position is sellable
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
                        algo.Debug(
                            f"REJECT ENTRY {sym.Value}: post-fee qty="
                            f"{post_fee_qty:.6f} < min_order_size={min_order_size} "
                            f"and can't upsize"
                        )
                        reject_notional += 1
                        continue
        except Exception as e:
            algo.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

        try:
            ticket = place_limit_or_market(algo, sym, qty, timeout_seconds=30, tag="Entry")
            if ticket is not None:
                algo._recent_tickets.append(ticket)
                components = cand.get('factors', {})
                sig_str = (
                    f"obi={components.get('obi', 0):.2f} "
                    f"vol={components.get('vol_ignition', 0):.2f} "
                    f"trend={components.get('micro_trend', 0):.2f} "
                    f"adx={components.get('adx_filter', 0):.2f} "
                    f"vwap={components.get('vwap_signal', 0):.2f}"
                )
                algo.Debug(
                    f"SCALP ENTRY: {sym.Value} | score={net_score:.2f} | "
                    f"${val:.2f} | {sig_str}"
                )
                success_count += 1
                algo.trade_count += 1
                crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                # MG3: transition to ENTERING state
                algo._set_position_state(sym, POSITION_STATE_ENTERING)
                # Record ADX regime at entry for tighter TP in choppy markets
                adx_ind   = crypto.get('adx')
                is_choppy = (
                    adx_ind is not None
                    and adx_ind.IsReady
                    and adx_ind.Current.Value < 25
                )
                algo._choppy_regime_entries[sym] = is_choppy
                if algo._consecutive_loss_halve_remaining > 0:
                    algo._consecutive_loss_halve_remaining -= 1
                if algo.LiveMode:
                    algo._last_live_trade_time = algo.Time
        except Exception as e:
            algo.Debug(f"ORDER FAILED: {sym.Value} - {e}")
            algo._session_blacklist.add(sym.Value)
            continue

        if algo.LiveMode and success_count >= 1:
            break

    if success_count > 0 or (reject_exit_cooldown + reject_loss_cooldown) > 3:
        debug_limited(
            algo,
            f"EXECUTE: {success_count}/{len(candidates)} | rejects: "
            f"cd={reject_exit_cooldown} loss={reject_loss_cooldown} "
            f"dv={reject_dollar_volume}"
        )
