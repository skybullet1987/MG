# region imports
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
# endregion

"""
mg2_diagnostics.py — Trade Attribution and Metadata Logging
============================================================
Records rich per-trade metadata so strategy quality can be evaluated
honestly.  Every entry is stamped with the setup type, major signal
components, market context, and execution quality metrics.  Every exit
is stamped with the exit reason, hold time, and gross/net performance.

Usage
-----
    engine = DiagnosticsEngine(algo)

    # On entry fill:
    engine.record_entry(symbol, setup_type, confidence, components,
                        entry_price, spread_at_entry, estimated_slippage)

    # On exit fill:
    engine.record_exit(symbol, exit_price, exit_tag, realized_slippage)

    # In on_end_of_algorithm:
    engine.print_summary()
"""


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

    # ─────────────────────────────────────────────────────────────────────────
    # Entry recording
    # ─────────────────────────────────────────────────────────────────────────

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

        # BTC 5-bar return at entry
        btc_rets = list(algo.btc_returns)
        if len(btc_rets) >= 5:
            rec.btc_5bar_return = float(sum(btc_rets[-5:]))
        else:
            rec.btc_5bar_return = None

        # RS vs BTC
        crypto = algo.crypto_data.get(symbol)
        if crypto:
            rs_hist = list(crypto.get('rs_vs_btc', []))
            rec.rs_vs_btc = float(rs_hist[-1]) if rs_hist else None
            rec.breakout_freshness = crypto.get('breakout_freshness', None)
            rec.atr_at_entry = (float(crypto['atr'].Current.Value)
                                if crypto.get('atr') and crypto['atr'].IsReady else None)

        self._open_records[symbol] = rec

    # ─────────────────────────────────────────────────────────────────────────
    # Exit recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_exit(self, symbol, exit_price, exit_tag,
                    realized_slippage=None, spread_at_exit=None):
        algo = self.algo
        rec = self._open_records.pop(symbol, None)
        if rec is None:
            return  # no matching open record

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

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def stats_by_setup(self):
        """Return per-setup statistics as a dict."""
        by_setup = {}
        for rec in self.completed:
            st = rec.setup_type or 'Unknown'
            if st not in by_setup:
                by_setup[st] = []
            by_setup[st].append(rec)

        result = {}
        for st, records in by_setup.items():
            pnls_net = [r.net_pnl_pct for r in records if r.net_pnl_pct is not None]
            pnls_gross = [r.gross_pnl_pct for r in records if r.gross_pnl_pct is not None]
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
        """Return per-exit-tag statistics."""
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
        """Return performance split by market regime at entry."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

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
        algo.Debug(f"  Win rate:       {len(wins)/len(all_pnls):.1%}" if all_pnls else "  No PnL data")
        algo.Debug(f"  Avg net PnL:    {float(np.mean(all_pnls)):.2%}" if all_pnls else "")
        algo.Debug(f"  Avg win:        {float(np.mean(wins)):.2%}" if wins else "  Avg win: N/A")
        algo.Debug(f"  Avg loss:       {float(np.mean(losses)):.2%}" if losses else "  Avg loss: N/A")
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
