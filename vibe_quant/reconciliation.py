"""Paper-vs-validation trade reconciliation.

Reconstructs round-trip trades from ``POSITION_OPEN``/``POSITION_CLOSE``
event pairs in a run's ``logs/events/{run_id}.jsonl`` and diffs two runs
against each other. Used to answer "did paper trading behave like the
validation backtest on the same window?" before promoting to live.

Both paper and validation emit the same event schema
(``vibe_quant.logging.events``) so we can compare them without
touching paper persistence.

Typical usage::

    from vibe_quant.reconciliation import reconcile, load_trades

    paper = load_trades("paper_42")            # trader_id as run_id
    validation = load_trades("42")             # backtest_runs.id
    report = reconcile(paper, validation, tolerance_seconds=120)
    print(report.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_BASE_PATH = _PROJECT_ROOT / "logs" / "events"


@dataclass(frozen=True, slots=True)
class Trade:
    """A reconstructed round-trip trade (position open → close).

    Attributes:
        position_id: Position identifier as emitted by the event log.
        symbol: Instrument symbol.
        side: ``LONG`` or ``SHORT``.
        entry_time: UTC timestamp of POSITION_OPEN.
        exit_time: UTC timestamp of POSITION_CLOSE.
        entry_price: Reported entry price.
        exit_price: Reported exit price.
        quantity: Position size as reported on open.
        net_pnl: Net P&L as reported on close.
        gross_pnl: Gross P&L as reported on close.
        exit_reason: Why the position closed (stop/take_profit/signal/...).
    """

    position_id: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    net_pnl: float
    gross_pnl: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class TradeMatch:
    """Pair of trades matched across two runs for the same position."""

    paper: Trade
    validation: Trade

    @property
    def entry_slippage(self) -> float:
        """Paper entry − validation entry. Positive = paper paid more."""
        return self.paper.entry_price - self.validation.entry_price

    @property
    def pnl_delta(self) -> float:
        """Paper net PnL − validation net PnL."""
        return self.paper.net_pnl - self.validation.net_pnl

    @property
    def side_agrees(self) -> bool:
        return self.paper.side == self.validation.side


@dataclass
class ReconciliationReport:
    """Summary of one paper-vs-validation comparison.

    Attributes:
        matches: Trades present in both runs, paired by entry time.
        paper_only: Trades in paper but not validation (false positives).
        validation_only: Trades in validation but not paper (missed signals).
        paper_run: Run label passed to ``load_trades``.
        validation_run: Run label passed to ``load_trades``.
        tolerance_seconds: Matching window used (seconds).
    """

    matches: list[TradeMatch] = field(default_factory=list)
    paper_only: list[Trade] = field(default_factory=list)
    validation_only: list[Trade] = field(default_factory=list)
    paper_run: str = ""
    validation_run: str = ""
    tolerance_seconds: int = 0

    def parity_rate(self) -> float:
        """Fraction of trades that matched (0..1). 1.0 means perfect parity."""
        total = len(self.matches) + len(self.paper_only) + len(self.validation_only)
        if total == 0:
            return 1.0
        # Each unmatched trade contributes to one side only; matched pairs
        # count once.
        return len(self.matches) / (len(self.matches) + len(self.paper_only) + len(self.validation_only))

    def mean_pnl_delta(self) -> float:
        """Mean P&L delta across matched trades (paper − validation)."""
        if not self.matches:
            return 0.0
        return sum(m.pnl_delta for m in self.matches) / len(self.matches)

    def mean_entry_slippage(self) -> float:
        """Mean entry slippage across matched trades (paper − validation)."""
        if not self.matches:
            return 0.0
        return sum(m.entry_slippage for m in self.matches) / len(self.matches)

    def side_disagreements(self) -> int:
        """Matched trades where paper and validation disagreed on direction."""
        return sum(1 for m in self.matches if not m.side_agrees)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Reconciliation: paper={self.paper_run} vs validation={self.validation_run}\n"
            f"  Matched:              {len(self.matches)}\n"
            f"  Paper-only (phantom): {len(self.paper_only)}\n"
            f"  Validation-only (missed): {len(self.validation_only)}\n"
            f"  Parity rate:          {self.parity_rate():.1%}\n"
            f"  Side disagreements:   {self.side_disagreements()}\n"
            f"  Mean entry slippage:  {self.mean_entry_slippage():+.6f}\n"
            f"  Mean PnL delta:       {self.mean_pnl_delta():+.4f}\n"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation for downstream dashboards."""
        return {
            "paper_run": self.paper_run,
            "validation_run": self.validation_run,
            "tolerance_seconds": self.tolerance_seconds,
            "counts": {
                "matched": len(self.matches),
                "paper_only": len(self.paper_only),
                "validation_only": len(self.validation_only),
            },
            "metrics": {
                "parity_rate": self.parity_rate(),
                "side_disagreements": self.side_disagreements(),
                "mean_entry_slippage": self.mean_entry_slippage(),
                "mean_pnl_delta": self.mean_pnl_delta(),
            },
            "paper_only_ids": [t.position_id for t in self.paper_only],
            "validation_only_ids": [t.position_id for t in self.validation_only],
        }


def load_trades(
    run_id: str,
    base_path: Path | str | None = None,
) -> list[Trade]:
    """Reconstruct round-trip trades from a run's event log.

    Reads ``logs/events/{run_id}.jsonl`` line-by-line. Orphan opens
    (position opened but never closed within the log) are dropped with
    a warning — they represent trades still live at log-end and aren't
    comparable.

    Args:
        run_id: Event log run identifier (trader_id for paper, str(run_id)
            for validation).
        base_path: Override the default ``logs/events/`` directory.

    Returns:
        List of Trade records sorted by entry_time.

    Raises:
        FileNotFoundError: If no log file exists for the run.
    """
    resolved = Path(base_path) if base_path is not None else _DEFAULT_BASE_PATH
    log_path = resolved / f"{run_id}.jsonl"
    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    opens: dict[str, dict[str, object]] = {}  # position_id -> open event dict
    trades: list[Trade] = []

    with log_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines silently — log rotation artifact
            event = record.get("event")
            data = record.get("data") or {}
            if event == "POSITION_OPEN":
                pid = str(data.get("position_id", ""))
                if pid:
                    opens[pid] = {"record": record, "data": data}
            elif event == "POSITION_CLOSE":
                pid = str(data.get("position_id", ""))
                open_rec = opens.pop(pid, None)
                if open_rec is None:
                    continue  # close without a matching open in this log — orphan
                trade = _build_trade(open_rec["record"], open_rec["data"], record, data)
                if trade is not None:
                    trades.append(trade)

    trades.sort(key=lambda t: t.entry_time)
    return trades


def _build_trade(
    open_record: dict[str, object],
    open_data: dict[str, object],
    close_record: dict[str, object],
    close_data: dict[str, object],
) -> Trade | None:
    """Pair an open/close event into a Trade. Returns None on malformed data."""
    entry_ts = _parse_ts(open_record.get("ts"))
    exit_ts = _parse_ts(close_record.get("ts"))
    if entry_ts is None or exit_ts is None:
        return None
    try:
        return Trade(
            position_id=str(open_data.get("position_id", "")),
            symbol=str(open_data.get("symbol", close_data.get("symbol", ""))),
            side=str(open_data.get("side", "")),
            entry_time=entry_ts,
            exit_time=exit_ts,
            entry_price=float(open_data.get("entry_price", 0.0)),
            exit_price=float(close_data.get("exit_price", 0.0)),
            quantity=float(open_data.get("quantity", 0.0)),
            net_pnl=float(close_data.get("net_pnl", 0.0)),
            gross_pnl=float(close_data.get("gross_pnl", 0.0)),
            exit_reason=str(close_data.get("exit_reason", "")),
        )
    except (TypeError, ValueError):
        return None


def _parse_ts(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def reconcile(
    paper: Iterable[Trade],
    validation: Iterable[Trade],
    tolerance_seconds: int = 120,
    paper_run: str = "",
    validation_run: str = "",
) -> ReconciliationReport:
    """Match trades across two runs by symbol + entry_time proximity.

    Greedy matcher:
        - for each paper trade, find the earliest unmatched validation
          trade within ±tolerance_seconds that agrees on symbol;
        - pair them;
        - unmatched trades become paper_only / validation_only.

    Greedy is fine here because paper and validation should have
    near-identical entry timing on the same strategy. If they're far
    apart, that itself is a parity failure worth flagging via the
    unmatched lists rather than heroically matching.

    Args:
        paper: Trades from the paper run.
        validation: Trades from the validation run.
        tolerance_seconds: Max entry-time gap (each side) to consider a match.
        paper_run: Label for the report (paper run id).
        validation_run: Label for the report (validation run id).

    Returns:
        ReconciliationReport with matches, paper_only, validation_only.
    """
    paper_list = sorted(paper, key=lambda t: t.entry_time)
    validation_list = sorted(validation, key=lambda t: t.entry_time)

    matches: list[TradeMatch] = []
    consumed_validation: set[int] = set()

    for p in paper_list:
        best_idx: int | None = None
        best_delta: float | None = None
        for i, v in enumerate(validation_list):
            if i in consumed_validation:
                continue
            if v.symbol != p.symbol:
                continue
            delta = abs((v.entry_time - p.entry_time).total_seconds())
            if delta > tolerance_seconds:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = i
        if best_idx is not None:
            consumed_validation.add(best_idx)
            matches.append(TradeMatch(paper=p, validation=validation_list[best_idx]))

    matched_paper_ids = {m.paper.position_id for m in matches}
    paper_only = [p for p in paper_list if p.position_id not in matched_paper_ids]
    validation_only = [
        v for i, v in enumerate(validation_list) if i not in consumed_validation
    ]

    return ReconciliationReport(
        matches=matches,
        paper_only=paper_only,
        validation_only=validation_only,
        paper_run=paper_run,
        validation_run=validation_run,
        tolerance_seconds=tolerance_seconds,
    )


__all__ = [
    "ReconciliationReport",
    "Trade",
    "TradeMatch",
    "load_trades",
    "reconcile",
]
