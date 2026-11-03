"""Stage 2 (pre-registered): full-fidelity validation of each full-window PASS, Aug-Oct
(must keep return > 0), plus May-Jul for screening-vs-validation agreement."""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path("/Users/verebelyin/projects/vibe-quant")
PASS = [
    ("data/swarm/burst-20261103-blind-rsi", "RSI+KAMA|long|NEARUSDT|2026-05-01|{L}x|s1|to2026-08-01|ew3|1h|allin", (3, 5)),
    ("data/swarm/burst-20261103-blind-rsi", "RSI+ROC|both|SUIUSDT|2026-05-01|{L}x|s1|to2026-08-01|ew3|15m|allin", (1, 3)),
    ("data/swarm/burst-20261103-blind-rsi-1m", "RSI+ADX|both|ZECUSDT|2026-05-01|{L}x|s1|to2026-08-01|ew3|fast|allin|mt300", (1, 3, 5)),
]
WINDOWS = (("2026-05-01", "2026-08-01"), ("2026-08-01", "2026-10-20"))

def job(args):
    key, cell, chrom, start, end = args
    os.chdir(REPO)
    from vibe_quant.discovery.genome import chromosome_to_dsl, serializable_to_chromosome
    from vibe_quant.validation.runner import ValidationRunner
    dsl = chromosome_to_dsl(serializable_to_chromosome(chrom))
    dsl["timeframe"] = cell["timeframe"]
    dsl["name"] = "blind_stage2"
    lev = int(cell["leverage"])
    params = {"liquidation_mode": "isolated", "leverage": lev, "max_position_pct": float(lev),
              "risk_per_trade": 0.5, "max_margin_pct": 1.0}
    tmp = Path(tempfile.mkdtemp(prefix="stage2-"))
    try:
        db = tmp / "v.db"
        src = sqlite3.connect("file:data/state/vibe_quant.db?mode=ro", uri=True)
        dst = sqlite3.connect(db)
        src.backup(dst)
        dst.close()
        src.close()
        con = sqlite3.connect(db)
        sid = con.execute("INSERT INTO strategies(name, dsl_config) VALUES (?, ?)",
                          (f"blind_stage2_{abs(hash((key, start)))}", json.dumps(dsl))).lastrowid
        rid = con.execute("INSERT INTO backtest_runs(strategy_id, run_mode, symbols, timeframe, start_date, end_date, parameters) VALUES (?,?,?,?,?,?,?)",
                          (sid, "validation", json.dumps([cell["symbol"]]), cell["timeframe"], start, end, json.dumps(params))).lastrowid
        con.commit()
        con.close()
        r = ValidationRunner(db_path=db, logs_path=tmp / "logs")
        try:
            m = r.run(rid)
        finally:
            r.close()
        return key, start, {"ret": m.total_return, "trades": m.total_trades, "sharpe": m.sharpe_ratio, "dd": m.max_drawdown, "error": None}
    except Exception as exc:
        return key, start, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    jobs = []
    for out, tmpl, levs in PASS:
        rows = {json.loads(line)["key"]: json.loads(line) for line in Path(REPO / out / "ledger.jsonl").read_text().splitlines()}
        for L in levs:
            row = rows[tmpl.format(L=L)]
            for s, e in WINDOWS:
                jobs.append((row["key"], row["cell"], row["best"]["chromosome"], s, e))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, jobs))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for key, start, m in res:
        print(start, key.split("|s1|")[0], m)

if __name__ == "__main__":
    main()
