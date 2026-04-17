#!/usr/bin/env python3
"""Reconcile a paper trading run against its validation backtest.

Usage:
    # Compare paper trader "paper_42" against validation run 58
    .venv/bin/python scripts/paper_reconciliation.py \\
        --paper-run paper_42 --validation-run 58

    # Override tolerance (default 120s) and emit JSON
    .venv/bin/python scripts/paper_reconciliation.py \\
        --paper-run paper_42 --validation-run 58 \\
        --tolerance-seconds 60 --json-out report.json

Both runs are expected to write to ``logs/events/{run_id}.jsonl``. Paper
uses the configured trader_id; validation uses str(backtest_runs.id).

Exits non-zero when parity rate is below ``--min-parity`` (default 0.80)
so the script can gate CI / runbooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vibe_quant.reconciliation import load_trades, reconcile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-run", required=True, help="Paper run id (trader_id)")
    parser.add_argument(
        "--validation-run", required=True, help="Validation run id (str(backtest_runs.id))"
    )
    parser.add_argument(
        "--logs-path",
        type=Path,
        default=None,
        help="Directory holding event logs. Defaults to <repo>/logs/events.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=int,
        default=120,
        help="Entry-time tolerance for matching (default: 120s).",
    )
    parser.add_argument(
        "--min-parity",
        type=float,
        default=0.80,
        help="Minimum parity rate (0..1). Exit non-zero if below (default 0.80).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON report to this path (in addition to stdout).",
    )
    args = parser.parse_args()

    try:
        paper_trades = load_trades(args.paper_run, base_path=args.logs_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        validation_trades = load_trades(args.validation_run, base_path=args.logs_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = reconcile(
        paper_trades,
        validation_trades,
        tolerance_seconds=args.tolerance_seconds,
        paper_run=args.paper_run,
        validation_run=args.validation_run,
    )

    print(report.summary())

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"report written to {args.json_out}")

    parity = report.parity_rate()
    if parity < args.min_parity:
        print(
            f"FAIL: parity {parity:.1%} below threshold {args.min_parity:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
