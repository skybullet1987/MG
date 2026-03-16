"""
Machine Gun 3 — Reporting & order-event helper functions
=========================================================
Plain functions that handle order events, brokerage messages, end-of-algorithm
reporting, and the daily summary.

All functions receive the algorithm instance as ``algo`` so they can read and
write algorithm state without inheriting from QCAlgorithm.

Note: ``from AlgorithmImports import *`` is required here so that
``OrderStatus`` and ``OrderDirection`` (managed .NET enums) are available.
"""
from AlgorithmImports import *  # noqa: F401,F403 – required for QC enums
from datetime import timedelta

import config as MG3Config
from execution import (
    cleanup_position, slip_log, persist_state,
    get_min_notional_usd, is_invested_not_dust,
    get_actual_position_count, daily_report,
)
from mg3_constants import (
    POSITION_STATE_FLAT, POSITION_STATE_OPEN, POSITION_STATE_ENTERING,
)


# ---------------------------------------------------------------------------
# OnOrderEvent handler
# ---------------------------------------------------------------------------

def on_order_event(algo, event):
    try:
        symbol = event.Symbol
        algo.Debug(
            f"ORDER: {symbol.Value} {event.Status} {event.Direction} "
            f"qty={event.FillQuantity or event.Quantity} "
            f"price={event.FillPrice} id={event.OrderId}"
        )

        if event.Status == OrderStatus.Submitted:
            if symbol not in algo._pending_orders:
                algo._pending_orders[symbol] = 0
            intended_qty = (
                abs(event.Quantity) if event.Quantity != 0
                else abs(event.FillQuantity)
            )
            algo._pending_orders[symbol] += intended_qty

            if symbol not in algo._submitted_orders:
                has_position = (
                    symbol in algo.Portfolio and algo.Portfolio[symbol].Invested
                )
                if event.Direction == OrderDirection.Sell and has_position:
                    inferred_intent = 'exit'
                elif event.Direction == OrderDirection.Buy and not has_position:
                    inferred_intent = 'entry'
                else:
                    inferred_intent = (
                        'entry' if event.Direction == OrderDirection.Buy else 'exit'
                    )
                algo._submitted_orders[symbol] = {
                    'order_id': event.OrderId,
                    'time':     algo.Time,
                    'quantity': event.Quantity,
                    'intent':   inferred_intent,
                }
            else:
                algo._submitted_orders[symbol]['order_id'] = event.OrderId

        elif event.Status == OrderStatus.PartiallyFilled:
            if symbol in algo._pending_orders:
                algo._pending_orders[symbol] -= abs(event.FillQuantity)
                if algo._pending_orders[symbol] <= 0:
                    algo._pending_orders.pop(symbol, None)
            if event.Direction == OrderDirection.Buy:
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol]   = event.FillPrice
                    algo.highest_prices[symbol] = event.FillPrice
                    algo.entry_times[symbol]    = algo.Time
                # MG3: partial fill → position is now at least partially OPEN
                algo._set_position_state(symbol, POSITION_STATE_OPEN)
            slip_log(algo, symbol, event.Direction, event.FillPrice)

        elif event.Status == OrderStatus.Filled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            # MG3: increment fill counter
            algo.mg3_fill_count += 1

            if event.Direction == OrderDirection.Buy:
                algo.entry_prices[symbol]   = event.FillPrice
                algo.highest_prices[symbol] = event.FillPrice
                algo.entry_times[symbol]    = algo.Time
                algo.daily_trade_count      += 1
                # MG3: position fully entered → OPEN
                algo._set_position_state(symbol, POSITION_STATE_OPEN)
                # MG3: track peak simultaneous open positions
                current_pos = get_actual_position_count(algo)
                if current_pos > algo.mg3_peak_positions:
                    algo.mg3_peak_positions = current_pos
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto['volume']) >= 1:
                    algo.entry_volumes[symbol] = crypto['volume'][-1]
                algo.rsi_peaked_overbought.pop(symbol, None)

            else:  # Sell fill
                if symbol in algo._partial_sell_symbols:
                    algo._partial_sell_symbols.discard(symbol)
                else:
                    entry = algo.entry_prices.get(symbol, None)
                    if entry is None:
                        entry = event.FillPrice
                        algo.Debug(
                            f"⚠️ WARNING: Missing entry price for "
                            f"{symbol.Value} sell, using fill price"
                        )
                    pnl = (event.FillPrice - entry) / entry if entry > 0 else 0
                    # MG3: record hold duration for avg-hold-time metric
                    hold_hours = (
                        (algo.Time - algo.entry_times[symbol]).total_seconds() / 3600
                        if symbol in algo.entry_times else 0.0
                    )
                    algo._rolling_wins.append(1 if pnl > 0 else 0)
                    algo._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                    if pnl > 0:
                        algo._rolling_win_sizes.append(pnl)
                        algo.winning_trades     += 1
                        algo.consecutive_losses  = 0
                    else:
                        algo._rolling_loss_sizes.append(abs(pnl))
                        algo.losing_trades      += 1
                        algo.consecutive_losses += 1
                    algo.total_pnl += pnl
                    algo.trade_log.append({
                        'time':        algo.Time,
                        'symbol':      symbol.Value,
                        'pnl_pct':     pnl,
                        'hold_hours':  hold_hours,
                        'exit_reason': 'filled_sell',
                    })

                    if len(algo._recent_trade_outcomes) >= 8:
                        recent_wr = (
                            sum(algo._recent_trade_outcomes)
                            / len(algo._recent_trade_outcomes)
                        )
                        if recent_wr < 0.25:
                            algo._cash_mode_until = algo.Time + timedelta(hours=24)
                            algo.Debug(
                                f"⚠️ CASH MODE: WR={recent_wr:.0%} over "
                                f"{len(algo._recent_trade_outcomes)} trades. Pausing 24h."
                            )

                    cleanup_position(algo, symbol)
                    # MG3: position exited → FLAT
                    algo._set_position_state(symbol, POSITION_STATE_FLAT)
                    algo._failed_exit_attempts.pop(symbol, None)
                    algo._failed_exit_counts.pop(symbol, None)

            slip_log(algo, symbol, event.Direction, event.FillPrice)

        elif event.Status == OrderStatus.Canceled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            # MG3: increment cancel counter
            algo.mg3_cancel_count += 1
            if (event.Direction == OrderDirection.Sell
                    and symbol not in algo.entry_prices):
                if is_invested_not_dust(algo, symbol):
                    holding = algo.Portfolio[symbol]
                    algo.entry_prices[symbol]   = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol]    = algo.Time
                    algo.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
            # MG3: if cancel of a buy entry order, revert to FLAT
            if event.Direction == OrderDirection.Buy:
                algo._set_position_state(symbol, POSITION_STATE_FLAT)

        elif event.Status == OrderStatus.Invalid:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            # MG3: increment invalid order counter
            algo.mg3_invalid_count += 1
            if event.Direction == OrderDirection.Sell:
                price       = (
                    algo.Securities[symbol].Price
                    if symbol in algo.Securities else 0
                )
                min_notional = get_min_notional_usd(algo, symbol)
                if (price > 0
                        and symbol in algo.Portfolio
                        and abs(algo.Portfolio[symbol].Quantity) * price < min_notional):
                    algo.Debug(
                        f"DUST CLEANUP on invalid sell: {symbol.Value} "
                        f"— releasing tracking"
                    )
                    cleanup_position(algo, symbol)
                    algo._set_position_state(symbol, POSITION_STATE_FLAT)
                    algo._failed_exit_counts.pop(symbol, None)
                else:
                    fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
                    algo._failed_exit_counts[symbol] = fail_count
                    algo.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                    if fail_count >= 3:
                        algo.Debug(
                            f"FORCE CLEANUP: {symbol.Value} after {fail_count} "
                            f"failed exits — releasing tracking"
                        )
                        cleanup_position(algo, symbol)
                        algo._set_position_state(symbol, POSITION_STATE_FLAT)
                        algo._failed_exit_counts.pop(symbol, None)
                    elif symbol not in algo.entry_prices:
                        if is_invested_not_dust(algo, symbol):
                            holding = algo.Portfolio[symbol]
                            algo.entry_prices[symbol]   = holding.AveragePrice
                            algo.highest_prices[symbol] = holding.AveragePrice
                            algo.entry_times[symbol]    = algo.Time
                            algo.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
            algo._session_blacklist.add(symbol.Value)

    except Exception as e:
        algo.Debug(f"OnOrderEvent error: {e}")

    if algo.LiveMode:
        persist_state(algo)


# ---------------------------------------------------------------------------
# OnBrokerageMessage handler
# ---------------------------------------------------------------------------

def on_brokerage_message(algo, message):
    try:
        txt = message.Message.lower()
        if "system status:" in txt:
            if "online" in txt:
                algo.kraken_status = "online"
            elif "maintenance" in txt:
                algo.kraken_status = "maintenance"
            elif "cancel_only" in txt:
                algo.kraken_status = "cancel_only"
            elif "post_only" in txt:
                algo.kraken_status = "post_only"
            else:
                algo.kraken_status = "unknown"
            algo.Debug(f"Kraken status update: {algo.kraken_status}")

        if "rate limit" in txt or "too many" in txt:
            algo.Debug(
                f"⚠️ RATE LIMIT - pausing {algo.rate_limit_cooldown_minutes}min"
            )
            algo._rate_limit_until     = (
                algo.Time + timedelta(minutes=algo.rate_limit_cooldown_minutes)
            )
            algo._last_live_trade_time = algo.Time
    except Exception as e:
        algo.Debug(f"BrokerageMessage parse error: {e}")


# ---------------------------------------------------------------------------
# OnEndOfAlgorithm handler
# ---------------------------------------------------------------------------

def on_end_of_algorithm(algo):
    total = algo.winning_trades + algo.losing_trades
    wr    = algo.winning_trades / total if total > 0 else 0
    algo.Debug("=== FINAL ===")
    algo.Debug(f"Trades: {algo.trade_count} | WR: {wr:.1%}")
    algo.Debug(f"Final: ${algo.Portfolio.TotalPortfolioValue:.2f}")
    algo.Debug(f"PnL: {algo.total_pnl:+.2%}")
    # MG3: emit backtest validation metrics
    log_mg3_metrics(algo)
    persist_state(algo)


# ---------------------------------------------------------------------------
# MG3 backtest metrics log
# ---------------------------------------------------------------------------

def log_mg3_metrics(algo):
    """Log MG3-specific backtest validation and forensic metrics.

    Acceptance thresholds (see README "Phased Rollout – Phase 1"):
      - Cancel-to-fill ratio  < 2
      - Invalid orders        < 5 % of fills
      - Stop Loss avg PnL     more positive than -tight_stop_loss (slippage check)
      - Recovery events       ideally 0; > 5 indicates systemic exit failure
    """
    algo.Debug("=== MG3 BACKTEST METRICS ===")

    # Order quality
    if algo.mg3_fill_count > 0:
        ctf         = algo.mg3_cancel_count / algo.mg3_fill_count
        invalid_pct = algo.mg3_invalid_count / algo.mg3_fill_count
        algo.Debug(
            f"Cancel-to-fill ratio : {ctf:.2f}  "
            f"({algo.mg3_cancel_count} cancels / {algo.mg3_fill_count} fills)"
        )
        algo.Debug(
            f"Invalid orders       : {algo.mg3_invalid_count}  "
            f"({invalid_pct:.1%} of fills)"
        )
    else:
        algo.Debug(
            f"Cancel-to-fill ratio : N/A  "
            f"(fills=0, cancels={algo.mg3_cancel_count})"
        )
        algo.Debug(f"Invalid orders       : {algo.mg3_invalid_count}")

    # Recovery events (failed exits escalated to force-market)
    algo.Debug(f"Recovery events      : {algo.mg3_recovery_events}")

    # Peak concurrent positions
    algo.Debug(f"Peak open positions  : {algo.mg3_peak_positions}")

    # Hold-time statistics (from trade_log)
    hold_times = [
        t.get('hold_hours', 0) for t in algo.trade_log
        if t.get('hold_hours', 0) > 0
    ]
    if hold_times:
        avg_hold = sum(hold_times) / len(hold_times)
        max_hold = max(hold_times)
        algo.Debug(f"Avg hold time        : {avg_hold:.2f}h  (max {max_hold:.1f}h)")
    else:
        algo.Debug("Avg hold time        : N/A")

    # Estimated round-trip fees paid (sanity-check approximation)
    rt_count    = algo.mg3_fill_count // 2
    est_fee_pct = MG3Config.FEE_ASSUMPTION_RT + MG3Config.SLIPPAGE_BUFFER
    algo.Debug(
        f"Est. cost per RT     : {est_fee_pct:.2%}  "
        f"(fee {MG3Config.FEE_ASSUMPTION_RT:.2%} + "
        f"slip {MG3Config.SLIPPAGE_BUFFER:.2%})  "
        f"× ~{rt_count} round trips"
    )

    # PnL by exit tag
    if algo.mg3_pnl_by_tag:
        algo.Debug("PnL by exit tag:")
        for tag, pnls in sorted(algo.mg3_pnl_by_tag.items()):
            n   = len(pnls)
            avg = sum(pnls) / n if n > 0 else 0.0
            wins = sum(1 for p in pnls if p > 0)
            wr  = wins / n if n > 0 else 0.0
            algo.Debug(f"  {tag:<30} n={n:>4}  avg={avg:+.3%}  wr={wr:.0%}")
    else:
        algo.Debug("PnL by exit tag      : (no tagged exits recorded)")

    algo.Debug("=== END MG3 METRICS ===")


# ---------------------------------------------------------------------------
# DailyReport wrapper
# ---------------------------------------------------------------------------

def daily_report_wrapper(algo):
    if algo.IsWarmingUp:
        return
    daily_report(algo)
