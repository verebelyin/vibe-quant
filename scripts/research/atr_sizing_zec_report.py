"""ATR sizing vs all-in on the ZEC RSI+ADX 1m genome (vibe-quant-9sost AC7; journal 2026-11-05).

Full-fidelity validation (ValidationRunner, 1m fills, full costs) of the blind-test #2 genome on
2026-08-01..10-19, isolated liquidation, leverage 5, cap 5x balance, margin limit 100%: all-in
(risk 0.5, cap binds) and ATR(14) x 2 sizing at risk 0.25 / 0.5 / 1%. Each run uses a private
copy of the state DB. A report, not a gate.

    PYTHONPATH=$PWD .venv/bin/python scripts/research/atr_sizing_zec_report.py OUT.json
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "data" / "swarm" / "burst-20261103-blind-rsi-1m" / "ledger.jsonl"
KEY = "RSI+ADX|both|ZECUSDT|2026-05-01|1x|s1|to2026-08-01|ew3|fast|allin|mt300"
BLIND = ("2026-08-01", "2026-10-20")
BASE = {
    "liquidation_mode": "isolated",
    "leverage": 5,
    "max_position_pct": 5.0,
    "max_margin_pct": 1.0,
}
ARMS: dict[str, dict[str, Any]] = {
    "allin-5x": {**BASE, "risk_per_trade": 0.5},
    "atr-0.25%": {**BASE, "risk_per_trade": 0.0025, "sizing_mode": "atr"},
    "atr-0.5%": {**BASE, "risk_per_trade": 0.005, "sizing_mode": "atr"},
    "atr-1%": {**BASE, "risk_per_trade": 0.01, "sizing_mode": "atr"},
}


def _private_db(
    tmp: Path, dsl: dict[str, Any], name: str, params: dict[str, Any]
) -> tuple[Path, int]:
    db = tmp / "v.db"
    src = sqlite3.connect(f"file:{REPO / 'data/state/vibe_quant.db'}?mode=ro", uri=True)
    dst = sqlite3.connect(db)
    src.backup(dst)
    dst.close()
    src.close()
    con = sqlite3.connect(db)
    sid = con.execute(
        "INSERT INTO strategies(name, dsl_config) VALUES (?, ?)",
        (f"atr_report_{name}", json.dumps(dsl)),
    ).lastrowid
    rid = con.execute(
        "INSERT INTO backtest_runs(strategy_id, run_mode, symbols, timeframe, start_date, end_date,"
        " parameters) VALUES (?,?,?,?,?,?,?)",
        (sid, "validation", json.dumps(["ZECUSDT"]), "1m", *BLIND, json.dumps(params)),
    ).lastrowid
    con.commit()
    con.close()
    return db, int(rid or 0)


def job(args: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    name, params, chrom = args
    os.chdir(REPO)
    from vibe_quant.discovery.genome import chromosome_to_dsl, serializable_to_chromosome
    from vibe_quant.validation.runner import ValidationRunner

    dsl = chromosome_to_dsl(serializable_to_chromosome(chrom))
    dsl["timeframe"] = "1m"
    dsl["name"] = "atr_report"
    tmp = Path(tempfile.mkdtemp(prefix="atr-report-"))
    try:
        db, rid = _private_db(tmp, dsl, name, params)
        runner = ValidationRunner(db_path=db, logs_path=tmp / "logs")
        try:
            m = runner.run(rid)
        finally:
            runner.close()
        notional = [t.quantity * t.entry_price for t in m.trades]
        return name, {
            "ret": m.total_return,
            "trades": m.total_trades,
            "sharpe": m.sharpe_ratio,
            "dd": m.max_drawdown,
            "notional_min": min(notional, default=None),
            "notional_max": max(notional, default=None),
            "error": None,
        }
    except Exception as exc:  # one failed arm is reported, not fatal
        return name, {"error": f"{type(exc).__name__}: {exc}"[:400]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    rows = {json.loads(line)["key"]: json.loads(line) for line in LEDGER.read_text().splitlines()}
    chrom = rows[KEY]["best"]["chromosome"]
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(job, [(n, p, chrom) for n, p in ARMS.items()]))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for name, m in res.items():
        print(name, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
