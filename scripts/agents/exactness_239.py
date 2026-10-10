"""Fixed-strategy exactness proof (CLAUDE.md § Verification Rules).

Runs one NTScreeningRunner eval of strategy 239 on BTCUSDT 2024-01-01..2026-03-17
and compares against the recorded baseline. Exit 0 = bit-identical, 1 = drift.

With ``--validation-multi`` it additionally runs the two-symbol (BTCUSDT+ETHUSDT)
validation case through ValidationRunner on a throwaway ``sqlite3`` backup of the
state DB and asserts its five metrics. The default run stays screening-only.

Run from the tree under test (in a worktree: PYTHONPATH=$PWD .venv/bin/python ...):
    .venv/bin/python scripts/agents/exactness_239.py
    .venv/bin/python scripts/agents/exactness_239.py --validation-multi
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.screening.pipeline import _dsl_to_dict
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vibe_quant.metrics import PerformanceMetrics

# Baseline since the 2026-10-10 fill-timing semantics break (per-instrument outbox release,
# end-of-run mark-to-market, Binance specs/catalog rebuild); was 1.3169239785208688/68.
# Update together with CLAUDE.md.
EXPECTED_SHARPE = 0.7974716442710239
EXPECTED_TRADES = 67
EXPECTED_TOTAL_RETURN = 0.159784359286041
EXPECTED_PROFIT_FACTOR = 1.6359612484989312
EXPECTED_MAX_DRAWDOWN = 0.05716872229767501

# Two-symbol validation baseline (--validation-multi), same break as above.
VALIDATION_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
VALIDATION_START = "2024-01-01"
VALIDATION_END = "2026-03-17"
EXPECTED_VALIDATION_MULTI = {
    "sharpe": 0.4956474374166028,
    "trades": 164,
    "return": 0.13753127107369148,
    "profit_factor": 1.1990127778814956,
    "max_drawdown": 0.11707174708945581,
}

# Data paths are resolved from this script's repo root so the proof runs from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data/catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data/archive/raw_data.db"


def _find_state_db() -> Path:
    # Worktrees have no data/state (single-writer DB): use the main checkout's, read-only.
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True,
        cwd=REPO_ROOT,
    ).stdout.strip()
    return Path(common).parent / "data/state/vibe_quant.db"


def _metrics_checks(
    m: PerformanceMetrics, expected: Mapping[str, float | int]
) -> dict[str, tuple[float | int, float | int]]:
    return {
        name: (getattr(m, attr), want)
        for name, (attr, want) in {
            "sharpe": ("sharpe_ratio", expected["sharpe"]),
            "trades": ("total_trades", expected["trades"]),
            "return": ("total_return", expected["return"]),
            "profit_factor": ("profit_factor", expected["profit_factor"]),
            "max_drawdown": ("max_drawdown", expected["max_drawdown"]),
        }.items()
    }


def _report(
    label: str, m: PerformanceMetrics, checks: dict[str, tuple[float | int, float | int]]
) -> bool:
    ok = all(actual == expected for actual, expected in checks.values())
    print(
        f"{label}: sharpe={m.sharpe_ratio!r} trades={m.total_trades} "
        f"return={m.total_return!r} profit_factor={m.profit_factor!r} "
        f"max_drawdown={m.max_drawdown!r} -> {'IDENTICAL' if ok else 'DRIFT'}"
    )
    for name, (actual, expected) in checks.items():
        if actual != expected:
            print(f"  {name}: got {actual!r} expected {expected!r}")
    return ok


def _screening_checks(dsl: object) -> bool:
    runner = NTScreeningRunner(
        _dsl_to_dict(dsl),
        ["BTCUSDT"],
        "2024-01-01",
        "2026-03-17",
        catalog_path=str(CATALOG_PATH),
        funding_archive_path=str(ARCHIVE_DB_PATH),
    )
    m = runner({})
    expected = {
        "sharpe": EXPECTED_SHARPE,
        "trades": EXPECTED_TRADES,
        "return": EXPECTED_TOTAL_RETURN,
        "profit_factor": EXPECTED_PROFIT_FACTOR,
        "max_drawdown": EXPECTED_MAX_DRAWDOWN,
    }
    return _report("screening", m, _metrics_checks(m, expected))


def _validation_multi_checks(source_db: Path, timeframe: str) -> bool:
    tmpdir = Path(tempfile.mkdtemp(prefix="exactness-239-"))
    try:
        db = tmpdir / "v.db"
        src = sqlite3.connect(str(source_db))
        dst = sqlite3.connect(str(db))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        conn = sqlite3.connect(str(db))
        run_id = conn.execute(
            "INSERT INTO backtest_runs(strategy_id, run_mode, symbols, timeframe, "
            "start_date, end_date, parameters) VALUES (?,?,?,?,?,?,?)",
            (239, "validation", json.dumps(VALIDATION_SYMBOLS), timeframe,
             VALIDATION_START, VALIDATION_END, "{}"),
        ).lastrowid
        conn.commit()
        conn.close()
        # ValidationRunner resolves catalog/archive paths relative to cwd.
        previous = Path.cwd()
        os.chdir(REPO_ROOT)
        runner = ValidationRunner(db_path=db, logs_path=tmpdir / "logs")
        try:
            m = runner.run(run_id)
        finally:
            runner.close()
            os.chdir(previous)
        return _report("validation-multi", m, _metrics_checks(m, EXPECTED_VALIDATION_MULTI))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-multi",
        action="store_true",
        help="also run the BTCUSDT+ETHUSDT validation case (slow) and assert its metrics",
    )
    args = parser.parse_args(argv)

    db = _find_state_db()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = conn.execute("SELECT dsl_config FROM strategies WHERE id = ?", (239,)).fetchone()
    conn.close()
    if row is None:
        print("strategy 239 not found in data/state/vibe_quant.db")
        return 1
    dsl = validate_strategy_dict(json.loads(row[0]))

    ok = _screening_checks(dsl)
    if args.validation_multi:
        ok = _validation_multi_checks(db, dsl.timeframe) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
