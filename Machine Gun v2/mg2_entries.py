# region imports
from AlgorithmImports import *
from execution import *
from scoring import MicroScalpEngine
import numpy as np
from datetime import timedelta
from mg2_data import is_ready, annualized_vol, compute_portfolio_risk_estimate
# endregion


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
    """Evaluate long signals only. Short scoring disabled (Cash account)."""
    long_score, long_components = algo._scoring_engine.calculate_scalp_score(crypto)

    sp = get_spread_pct(algo, symbol)
    if sp is not None and sp > 0:
        spread_penalty = min((sp / 0.005) * 0.15, 0.15)
        long_score *= (1.0 - spread_penalty)

    components = long_components.copy()
    components['_scalp_score'] = long_score
    components['_direction'] = 1
    components['_long_score'] = long_score
    return components


def calculate_composite_score(algo, factors, crypto=None):
    """Return the pre-computed scalp score."""
    return factors.get('_scalp_score', 0.0)


def apply_fee_adjustment(algo, score):
    """Return score unchanged – signal thresholds already require >1% moves."""
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
    """Returns True if the portfolio has dropped >= 3% from today's open value."""
    if algo._daily_open_value is None or algo._daily_open_value <= 0:
        return False
    current = algo.Portfolio.TotalPortfolioValue
    if current <= 0:
        return True
    drop = (algo._daily_open_value - current) / algo._daily_open_value
    return drop >= 0.03


def rebalance(algo):
    if algo.IsWarmingUp:
        return

    if daily_loss_exceeded(algo):
        log_skip(algo, "max daily loss exceeded")
        return

    if len(algo.btc_returns) >= 5 and sum(list(algo.btc_returns)[-5:]) < -0.01:
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

    count_scored = 0
    count_above_thresh = 0
    scores = []
    threshold_now = algo.entry_threshold
    for symbol in list(algo.crypto_data.keys()):
        if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in algo._session_blacklist:
            continue
        if symbol.Value in algo._symbol_entry_cooldowns and algo.Time < algo._symbol_entry_cooldowns[symbol.Value]:
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
        net_score = apply_fee_adjustment(algo, composite_score)

        crypto['recent_net_scores'].append(net_score)

        if net_score >= threshold_now:
            if len(crypto['recent_net_scores']) >= 3:
                above_count = sum(1 for s in list(crypto['recent_net_scores'])[-3:] if s >= threshold_now)
                if above_count < 2:
                    continue
            count_above_thresh += 1
            scores.append({
                'symbol': symbol,
                'composite_score': composite_score,
                'net_score': net_score,
                'factors': factor_scores,
                'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
            })

    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        cash = algo.Portfolio.Cash

    debug_limited(algo, f"REBALANCE: {count_above_thresh}/{count_scored} above thresh={threshold_now:.2f} | cash=${cash:.2f}")

    if len(scores) == 0:
        log_skip(algo, "no candidates passed filters")
        return
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
        debug_limited(algo, f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved (>{algo.open_orders_cash_threshold:.0%} of ${available_cash:.2f})")
        return

    reject_pending_orders = 0
    reject_open_orders = 0
    reject_already_invested = 0
    reject_spread = 0
    reject_exit_cooldown = 0
    reject_loss_cooldown = 0
    reject_correlation = 0
    reject_price_invalid = 0
    reject_price_too_low = 0
    reject_cash_reserve = 0
    reject_min_qty_too_large = 0
    reject_dollar_volume = 0
    reject_notional = 0
    success_count = 0

    for cand in candidates:
        if algo.daily_trade_count >= algo.max_daily_trades:
            break
        if get_actual_position_count(algo) >= algo.max_positions:
            break
        sym = cand['symbol']
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
        if algo.LiveMode:
            _crypto_depth = algo.crypto_data.get(sym)
            if _crypto_depth:
                bid_size = _crypto_depth.get('bid_size', 0)
                if bid_size > 0:
                    _sec_depth = algo.Securities[sym]
                    _price_depth = _sec_depth.Price if _sec_depth.Price > 0 else 1
                    _estimated_val = algo.Portfolio.TotalPortfolioValue * 0.35
                    _our_qty = _estimated_val / _price_depth
                    if _our_qty > bid_size * 0.20:
                        continue
        if sym in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[sym]:
            reject_exit_cooldown += 1
            continue
        if sym.Value in algo._symbol_entry_cooldowns and algo.Time < algo._symbol_entry_cooldowns[sym.Value]:
            reject_loss_cooldown += 1
            continue
        if sym in algo._symbol_loss_cooldowns and algo.Time < algo._symbol_loss_cooldowns[sym]:
            reject_loss_cooldown += 1
            continue
        if not check_correlation(algo, sym):
            reject_correlation += 1
            continue
        sec = algo.Securities[sym]
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

        pending_exit_fees = 0
        for _exit_sym in list(algo.entry_prices.keys()):
            if is_invested_not_dust(algo, _exit_sym):
                _holding_val = abs(algo.Portfolio[_exit_sym].Quantity) * algo.Securities[_exit_sym].Price
                pending_exit_fees += _holding_val * 0.004
        available_cash = max(0, available_cash - open_buy_orders_value - pending_exit_fees)
        total_value = algo.Portfolio.TotalPortfolioValue
        fee_reserve = max(total_value * algo.cash_reserve_pct, 0.50)
        reserved_cash = available_cash - fee_reserve
        if reserved_cash <= 0:
            reject_cash_reserve += 1
            continue

        min_qty = get_min_quantity(algo, sym)
        min_notional_usd = get_min_notional_usd(algo, sym)
        if min_qty * price > reserved_cash * 0.90:
            reject_min_qty_too_large += 1
            continue

        crypto = algo.crypto_data.get(sym)
        if not crypto:
            continue

        if crypto.get('trade_count_today', 0) >= algo.max_symbol_trades_per_day:
            continue

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

        if len(crypto['dollar_volume']) >= 3:
            dv_window = min(len(crypto['dollar_volume']), 12)
            recent_dv = np.mean(list(crypto['dollar_volume'])[-dv_window:])
            dv_threshold = algo.min_dollar_volume_usd
            if recent_dv < dv_threshold:
                reject_dollar_volume += 1
                continue

        vol = annualized_vol(algo, crypto)
        size = algo._scoring_engine.calculate_position_size(net_score, threshold_now, vol)

        if algo._consecutive_loss_halve_remaining > 0:
            size *= 0.50

        if algo.volatility_regime == "high":
            size = min(size * 1.1, algo.position_size_pct)

        slippage_penalty = get_slippage_penalty(algo, sym)
        size *= slippage_penalty

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
        if val < min_notional_usd * algo.min_notional_fee_buffer or val < algo.min_notional or val > reserved_cash:
            reject_notional += 1
            continue

        try:
            sec = algo.Securities[sym]
            min_order_size = float(sec.SymbolProperties.MinimumOrderSize or 0)
            lot_size = float(sec.SymbolProperties.LotSize or 0)
            actual_min = max(min_order_size, lot_size)
            if actual_min > 0 and qty < actual_min:
                algo.Debug(f"REJECT ENTRY {sym.Value}: qty={qty} < min_order_size={actual_min} (unsellable)")
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
                        algo.Debug(f"REJECT ENTRY {sym.Value}: post-fee qty={post_fee_qty:.6f} < min_order_size={min_order_size} and can't upsize")
                        reject_notional += 1
                        continue
        except Exception as e:
            algo.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

        try:
            ticket = place_limit_or_market(algo, sym, qty, timeout_seconds=30, tag="Entry")
            if ticket is not None:
                algo._recent_tickets.append(ticket)
                components = cand.get('factors', {})
                sig_str = (f"obi={components.get('obi', 0):.2f} "
                           f"vol={components.get('vol_ignition', 0):.2f} "
                           f"trend={components.get('micro_trend', 0):.2f} "
                           f"adx={components.get('adx_trend', 0):.2f} "
                           f"mean_rev={components.get('mean_reversion', 0):.2f} "
                           f"vwap={components.get('vwap_signal', 0):.2f}")
                algo.Debug(f"SCALP ENTRY: {sym.Value} | score={net_score:.2f} | ${val:.2f} | {sig_str}")
                success_count += 1
                algo.trade_count += 1
                crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                adx_ind = crypto.get('adx')
                is_choppy = (adx_ind is not None and adx_ind.IsReady
                             and adx_ind.Current.Value < 25)
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
        debug_limited(algo, f"EXECUTE: {success_count}/{len(candidates)} | rejects: cd={reject_exit_cooldown} loss={reject_loss_cooldown} corr={reject_correlation} dv={reject_dollar_volume}")
