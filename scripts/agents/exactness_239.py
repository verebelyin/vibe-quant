"""Fixed-strategy exactness proof (CLAUDE.md § Verification Rules).

Runs one NTScreeningRunner eval of strategy 239 on BTCUSDT 2024-01-01..2026-03-17
and compares against the recorded baseline. Exit 0 = bit-identical, 1 = drift.

Run from the tree under test (in a worktree: PYTHONPATH=$PWD .venv/bin/python ...):
    .venv/bin/python scripts/agents/exactness_239.py
"""

from __future__ import annotations

import json
import sqlite3
import sys

from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.screening.pipeline import _dsl_to_dict

# Baseline since the 2026-10-03 audit fixes; update together with CLAUDE.md.
EXPECTED_SHARPE = 1.3165049716553048
EXPECTED_TRADES = 68


def main() -> int:
    conn = sqlite3.connect("file:data/state/vibe_quant.db?mode=ro", uri=True)
    row = conn.execute("SELECT dsl_config FROM strategies WHERE id = ?", (239,)).fetchone()
    conn.close()
    if row is None:
        print("strategy 239 not found in data/state/vibe_quant.db")
        return 1
    dsl = validate_strategy_dict(json.loads(row[0]))
    runner = NTScreeningRunner(_dsl_to_dict(dsl), ["BTCUSDT"], "2024-01-01", "2026-03-17")
    m = runner({})
    ok = m.sharpe_ratio == EXPECTED_SHARPE and m.total_trades == EXPECTED_TRADES
    print(f"sharpe={m.sharpe_ratio!r} trades={m.total_trades} -> {'IDENTICAL' if ok else 'DRIFT'}")
    print(f"expected sharpe={EXPECTED_SHARPE!r} trades={EXPECTED_TRADES}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
