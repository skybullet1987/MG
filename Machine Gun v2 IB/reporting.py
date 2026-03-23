"""
reporting.py — Machine Gun v2 IB: Diagnostics and trade attribution

DiagnosticsLogger provides first-class analytics so the backtest can explain
itself.  It is the single place where all candidate and trade data is collected.

Recorded per-trade
------------------
  - setup type
  - score and threshold at entry
  - long vs short score comparison
  - regime / session bucket
  - exit reason
  - hold time
  - PnL

Summary reports
---------------
  - PnL attribution by setup / symbol / regime / session / exit reason
  - Skipped/rejected candidate counts by reason
  - Candidate generation counts vs accepted trades

Usage
-----
  algo.diagnostics = DiagnosticsLogger(algo)

  # Each bar — when candidates are generated / filtered:
  algo.diagnostics.record_candidate_generated(candidate)
  algo.diagnostics.record_candidate_accepted(candidate, order_id)
  algo.diagnostics.record_candidate_rejected(candidate, reason)

  # After an entry fills (from OnOrderEvent):
  algo.diagnostics.note_entry_filled(entry_order_id, symbol, fill_price)

  # After a bracket leg fills (exit):
  algo.diagnostics.record_exit(symbol, exit_price, exit_reason)

  # Scheduled daily report:
  algo.diagnostics.print_daily_summary()

  # End of algorithm:
  algo.diagnostics.print_summary()
"""

import numpy as np
from collections import defaultdict
from candidates import ALL_SETUP_TYPES


class DiagnosticsLogger:
    """
    Central diagnostics/attribution logger for Machine Gun v2 IB.

    Designed to be stateless with respect to the rest of the algorithm:
    it only stores data, never makes trading decisions.
    """

    def __init__(self, algo):
        self.algo = algo

        # Enhanced trade log: one record per closed trade
        self.trade_log = []

        # Candidate pipeline counters
        self.candidates_generated = 0
        self.candidates_accepted  = 0
        self.candidates_rejected  = 0

        self.rejection_counts    = defaultdict(int)   # reason  → count
        self.generation_by_setup = defaultdict(int)   # setup   → count
        self.accepted_by_setup   = defaultdict(int)   # setup   → count
        self.rejected_by_setup   = defaultdict(int)   # setup   → count

        # In-flight entries: entry_order_id → metadata dict
        self._pending_meta = {}
        # Symbol fallback (in case we need to look up by symbol instead of order id)
        self._symbol_meta  = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def record_candidate_generated(self, candidate):
        """Call whenever a SetupCandidate is produced by a setup evaluator."""
        self.candidates_generated += 1
        self.generation_by_setup[candidate.setup_type] += 1

    def record_candidate_accepted(self, candidate, order_id):
        """Call when a candidate passes all filters and a market order is placed."""
        self.candidates_accepted += 1
        self.accepted_by_setup[candidate.setup_type] += 1

        meta = {
            "setup_type":  candidate.setup_type,
            "direction":   candidate.direction,
            "score":       candidate.score,
            "threshold":   candidate.threshold,
            "long_score":  candidate.long_score,
            "short_score": candidate.short_score,
            "components":  dict(candidate.components),
            "regime":      candidate.regime,
            "session":     candidate.session,
            "vix":         candidate.vix,
            "symbol_name": candidate.symbol_name,
            "entry_time":  self.algo.Time,
            "entry_price": None,   # filled in by note_entry_filled()
        }
        self._pending_meta[order_id]               = meta
        self._symbol_meta[candidate.symbol_name]   = meta

    def record_candidate_rejected(self, candidate, reason):
        """Call when a candidate is generated but not turned into an order."""
        self.candidates_rejected += 1
        self.rejection_counts[reason] += 1
        self.rejected_by_setup[candidate.setup_type] += 1

    def record_rejection(self, reason):
        """Record a global rejection not tied to a specific candidate."""
        self.rejection_counts[reason] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Entry fill notification
    # ─────────────────────────────────────────────────────────────────────────

    def note_entry_filled(self, entry_order_id, symbol, fill_price):
        """
        Called from OnOrderEvent when an entry order fills.
        Stores the actual fill price so hold-time and PnL are accurate.
        """
        meta = self._pending_meta.get(entry_order_id)
        symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        if meta is None:
            meta = self._symbol_meta.get(symbol_name)
        if meta is not None:
            meta["entry_price"] = fill_price
            meta["entry_time"]  = self.algo.Time
            # Also store by symbol so bracket fills can find it
            self._symbol_meta[symbol_name] = meta

    # ─────────────────────────────────────────────────────────────────────────
    # Exit recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_exit(self, symbol, entry_order_id, entry_price, exit_price,
                    exit_reason, entry_time=None):
        """
        Record a completed trade with full attribution.

        Looks up entry metadata by order_id first, then by symbol name as a
        fallback.  Always removes the pending metadata entry to prevent stale data.
        """
        symbol_name = symbol.Value if hasattr(symbol, "Value") else str(symbol)

        # Try to find metadata in priority order
        meta = self._pending_meta.pop(entry_order_id, None)
        if meta is None:
            meta = self._symbol_meta.pop(symbol_name, None)
        if meta is None:
            meta = {}

        direction = meta.get("direction", 1)

        if entry_price > 0 and exit_price > 0:
            pnl_pct = (
                (exit_price - entry_price) / entry_price if direction == 1
                else (entry_price - exit_price) / entry_price
            )
        else:
            pnl_pct = 0.0

        # Hold time
        t_entry = meta.get("entry_time") or entry_time
        hold_hours = None
        if t_entry is not None:
            try:
                hold_secs = (self.algo.Time - t_entry).total_seconds()
                hold_hours = hold_secs / 3600.0
            except Exception:
                pass

        record = {
            "time":        self.algo.Time,
            "symbol":      symbol_name,
            "setup_type":  meta.get("setup_type",  "unknown"),
            "direction":   direction,
            "score":       meta.get("score",       0.0),
            "threshold":   meta.get("threshold",   0.0),
            "long_score":  meta.get("long_score",  0.0),
            "short_score": meta.get("short_score", 0.0),
            "components":  meta.get("components",  {}),
            "regime":      meta.get("regime",      "unknown"),
            "session":     meta.get("session",     "unknown"),
            "vix":         meta.get("vix",         20.0),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "exit_reason": exit_reason,
            "pnl_pct":     pnl_pct,
            "hold_hours":  hold_hours,
            "entry_time":  t_entry,
        }
        self.trade_log.append(record)
        return record

    # ─────────────────────────────────────────────────────────────────────────
    # Summary reports
    # ─────────────────────────────────────────────────────────────────────────

    def print_summary(self):
        """
        Print a comprehensive attribution summary to the algorithm's Debug log.
        Covers the full backtest run.
        """
        algo   = self.algo
        trades = self.trade_log
        total  = len(trades)

        algo.Debug("=" * 64)
        algo.Debug("=== MACHINE GUN v2 IB  — DIAGNOSTICS SUMMARY ===")
        algo.Debug(
            f"Candidates: generated={self.candidates_generated} "
            f"accepted={self.candidates_accepted} "
            f"rejected={self.candidates_rejected}"
        )
        algo.Debug(f"Closed trades: {total}")

        if total == 0:
            algo.Debug("No closed trades — nothing to attribute.")
            algo.Debug("=" * 64)
            return

        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr   = wins / total
        avg  = np.mean([t["pnl_pct"] for t in trades])
        algo.Debug(f"Overall: WR={wr:.1%}  Avg={avg:+.4%}")

        # ── By setup type ─────────────────────────────────────────────────────
        algo.Debug("--- By Setup ---")
        for st in ALL_SETUP_TYPES:
            st_trades = [t for t in trades if t["setup_type"] == st]
            gen = self.generation_by_setup.get(st, 0)
            acc = self.accepted_by_setup.get(st, 0)
            if not st_trades:
                algo.Debug(f"  {st}: 0 closed trades  (gen={gen} acc={acc})")
                continue
            st_wins = sum(1 for t in st_trades if t["pnl_pct"] > 0)
            st_wr   = st_wins / len(st_trades)
            st_avg  = np.mean([t["pnl_pct"] for t in st_trades])
            algo.Debug(
                f"  {st}: {len(st_trades)} trades | WR={st_wr:.1%} | "
                f"Avg={st_avg:+.4%} | gen={gen} acc={acc}"
            )

        # ── By symbol ─────────────────────────────────────────────────────────
        algo.Debug("--- By Symbol ---")
        for sym in sorted(set(t["symbol"] for t in trades)):
            st = [t for t in trades if t["symbol"] == sym]
            sw = sum(1 for t in st if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {sym}: {len(st)} trades | WR={sw/len(st):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in st]):+.4%}"
            )

        # ── By exit reason ────────────────────────────────────────────────────
        algo.Debug("--- By Exit Reason ---")
        for reason in sorted(set(t["exit_reason"] for t in trades)):
            rt = [t for t in trades if t["exit_reason"] == reason]
            rw = sum(1 for t in rt if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {reason}: {len(rt)} trades | WR={rw/len(rt):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in rt]):+.4%}"
            )

        # ── By regime ─────────────────────────────────────────────────────────
        algo.Debug("--- By Regime ---")
        for regime in sorted(set(t["regime"] for t in trades)):
            rt = [t for t in trades if t["regime"] == regime]
            rw = sum(1 for t in rt if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {regime}: {len(rt)} trades | WR={rw/len(rt):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in rt]):+.4%}"
            )

        # ── By session ────────────────────────────────────────────────────────
        algo.Debug("--- By Session ---")
        for session in sorted(set(t["session"] for t in trades)):
            st = [t for t in trades if t["session"] == session]
            sw = sum(1 for t in st if t["pnl_pct"] > 0)
            algo.Debug(
                f"  {session}: {len(st)} trades | WR={sw/len(st):.1%} | "
                f"Avg={np.mean([t['pnl_pct'] for t in st]):+.4%}"
            )

        # ── Hold time distribution ─────────────────────────────────────────────
        hold_hours = [t["hold_hours"] for t in trades if t.get("hold_hours") is not None]
        if hold_hours:
            algo.Debug(
                f"--- Hold Time (h): min={min(hold_hours):.2f} "
                f"avg={np.mean(hold_hours):.2f} "
                f"max={max(hold_hours):.2f} ---"
            )

        # ── Rejection reasons ─────────────────────────────────────────────────
        if self.rejection_counts:
            algo.Debug("--- Rejection / Skip Reasons ---")
            for reason, count in sorted(
                self.rejection_counts.items(), key=lambda x: -x[1]
            ):
                algo.Debug(f"  {reason}: {count}")

        algo.Debug("=" * 64)

    def print_daily_summary(self):
        """
        Print today's attribution summary (called from DailyReport scheduled task).
        """
        algo  = self.algo
        today = algo.Time.date()

        trades_today = [
            t for t in self.trade_log
            if t.get("time") and t["time"].date() == today
        ]
        total = len(trades_today)

        algo.Debug(f"=== DAILY {today} ===")
        algo.Debug(
            f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | "
            f"Cash: ${algo.Portfolio.Cash:.2f}"
        )
        algo.Debug(
            f"VIX: {getattr(algo, 'vix_value', 20.0):.1f} | "
            f"Regime: {getattr(algo, 'market_regime', 'unknown')}"
        )
        algo.Debug(
            f"Candidates today: gen={self.candidates_generated} "
            f"acc={self.candidates_accepted} | Trades closed: {total}"
        )

        if total > 0:
            wins = sum(1 for t in trades_today if t["pnl_pct"] > 0)
            wr   = wins / total
            avg  = np.mean([t["pnl_pct"] for t in trades_today])
            algo.Debug(f"Today: WR={wr:.1%} | Avg={avg:+.4%}")

            for st in ALL_SETUP_TYPES:
                st_t = [t for t in trades_today if t["setup_type"] == st]
                if st_t:
                    sw = sum(1 for t in st_t if t["pnl_pct"] > 0)
                    algo.Debug(
                        f"  {st}: {len(st_t)} trades | WR={sw/len(st_t):.1%} | "
                        f"Avg={np.mean([t['pnl_pct'] for t in st_t]):+.4%}"
                    )
