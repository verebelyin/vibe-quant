"""Re-score every pre-registered pick on the FULL blind window (eval_windows=1).
Picks and pass thresholds unchanged
only the blind measurement (transfer used worst-of-3)."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "scripts/agents")
from pathlib import Path

from scripts.research import blind_2026 as t1

BLIND = ("2026-08-01", "2026-10-20")

def load(out, train):
    rows = [json.loads(line) for line in Path(f"{out}/ledger.jsonl").read_text().splitlines()]
    rows = [r for r in rows if (r["cell"]["start"], r["cell"]["end"]) == train]
    return [r for r in rows if not r.get("failed") and r.get("best")]

def full(row):
    import burst

    from vibe_quant.discovery.genome import serializable_to_chromosome
    cell = {**row["cell"], "eval_windows": 1}
    r = burst._transfer_backtest(cell, cell["symbol"], *BLIND)(serializable_to_chromosome(row["best"]["chromosome"]))
    return row["key"], {"ret": float(r.get("total_return", 0)), "trades": int(r.get("total_trades", 0)),
                        "sharpe": float(r.get("sharpe_ratio", 0)), "dd": float(r.get("max_drawdown", 0)), "error": r.get("error")}

def main():
    arms = [("t1", "data/swarm/burst-20261103-blind-2026", ("2026-01-01", "2026-08-01"), 2),
            ("t2", "data/swarm/burst-20261103-blind-rsi", ("2026-05-01", "2026-08-01"), 3),
            ("t2-1m", "data/swarm/burst-20261103-blind-rsi-1m", ("2026-05-01", "2026-08-01"), 3)]
    jobs, meta = [], {}
    for name, out, train, k in arms:
        t1.PER_COIN = k
        picks = t1.select(load(out, train))
        meta[name] = picks
        jobs += picks
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(full, jobs))
    from vibe_quant.discovery.buyhold import load_daily_closes
    closes = load_daily_closes(t1.SYMBOLS if hasattr(t1, "SYMBOLS") else ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","NEARUSDT","SUIUSDT","ZECUSDT","HYPEUSDT"], Path("data/catalog"))
    for name, picks in meta.items():
        n = 0
        print(f"## {name}")
        for r in picks:
            m = res[r["key"]]
            bh = t1._bh(closes, r["cell"]["symbol"])
            if name == "t1":
                ok = not m["error"] and m["trades"] >= 15 and m["ret"] > max(0.0, bh)
            else:
                ok = not m["error"] and m["trades"] >= 15 and m["ret"] > 0 and m["sharpe"] >= 1.0
            n += ok
            print(f"{'PASS' if ok else 'FAIL'} {r['key']} | full blind ret={m['ret']:+.4f} trades={m['trades']} sharpe={m['sharpe']:.3f} dd={m['dd']:.3f} bh={bh:+.4f} err={m['error']}")
        print(f"{name} full-window pass {n}/{len(picks)}")
    Path("/private/tmp/claude-501/-Users-verebelyin-projects-vibe-quant/f1aa73b3-c4c9-4817-921d-b147afa5c8df/scratchpad/rescore_full.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
