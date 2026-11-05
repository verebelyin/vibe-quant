"""Blind 2026 RSI test #2: search May-Jul 2026, test once on Aug-Oct 2026, all-in leverage.

PRE-REGISTERED 2026-11-03 (chief; user asked: "searching in may-jun-july and backtest on
aug-sept-october ... a lot of rsi strategies with leverage"), committed BEFORE the search.
Scratch burst (NOT in DSR N); engine = main after the q3eq5.2 costed-sizing break.

Search (burst.py run --allin, out dir data/swarm/burst-20261103-blind-rsi):
    window 2026-05-01..2026-08-01 (end exclusive); worst of 3 sub-windows (~1 month each);
    12 RSI families: RSI, ADAPTIVE_RSI, RSI+KAMA, RSI+OWN_TREND, RSI+BTC_TREND, RSI+ADX,
    RSI+DONCHIAN, RSI+PRICE_POSITION, RSI+TREND_ENSEMBLE, RSI+RAMS, RSI+FUNDING, RSI+ROC;
    directions long, short, both; coins BTC ETH SOL BNB NEAR SUI ZEC HYPE (the 8 in the catalog);
    timeframes 15m and 1h (4h dropped before the run: a 3-month 4h window cannot reach the
    50-trade winner minimum, smoke 0/18); leverage L in {1, 3, 5} ALL-IN: every entry is
    L x balance at venue leverage L, isolated liquidation on, margin limit 100%;
    NT tier, pop 12 x 3 gens, seed 1 -> 1728 cells.
    A cell's winner = its best genome with >= 50 trades and return > 0 on May-Jul.

Blind window 2026-08-01..2026-10-20 (end exclusive; last full day 2026-10-19): burst.py
transfer's `next_full` evaluation (whole window) of the SAME genome, coin, timeframe and
leverage. Never selects. CORRECTED 2026-11-03: the first read used `next`, the worst of 3
sub-windows; it is still printed beside the verdict (vibe-quant-iqj2h). A row without
`next_full` has no blind verdict: re-score with blind_2026_rescore_full.py.

Stage 1 (this script, run once after the transfer):
    PRIMARY = per coin, the 3 winners with the highest May-Jul best Sharpe (ties: higher
    return, then key) -> at most 24, every one reported.
    PASS iff on Aug-Oct: trades >= 15 AND return > 0 AND Sharpe >= 1.0.
    (Return alone would pass several of 24 by luck; Sharpe >= 1 is the luck guard.) Beating the
    coin's buy-and-hold is reported per pick, not required (user: a profitable strategy is
    acceptable).
    REFERENCE = the same PASS rule over ALL winners: a family-level edge needs the primary pass
    share above the reference share.
Stage 2 (pre-registered, only for stage-1 PASSes): one full-fidelity validation run
(ValidationRunner, 1m fills, full costs) of the same strategy on Aug-Oct; it must keep
return > 0. Survivors go to the overfit-auditor; paper trading only with the user's approval.
A FAIL is final: no re-tuning on Aug-Oct.

    PYTHONPATH=$PWD .venv/bin/python scripts/research/blind_2026_rsi.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import blind_2026 as base  # noqa: E402

OUT = REPO_ROOT / "data" / "swarm" / "burst-20261103-blind-rsi"
TRAIN = ("2026-05-01", "2026-08-01")
PER_COIN = 3
MIN_BLIND_TRADES = 15
MIN_BLIND_SHARPE = 1.0


def passes(nxt: dict | None) -> bool:
    if not nxt or nxt.get("error"):
        return False
    return (
        nxt["trades"] >= MIN_BLIND_TRADES
        and nxt["ret"] > 0
        and nxt["sharpe"] is not None
        and nxt["sharpe"] >= MIN_BLIND_SHARPE
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    ledger = [
        r for r in base._rows(args.out / "ledger.jsonl")
        if (r["cell"]["start"], r["cell"]["end"]) == TRAIN
    ]  # fmt: skip
    winners = [r for r in ledger if not r.get("failed") and r.get("best")]
    nxt, worst = base.blind_evals(args.out)
    from vibe_quant.discovery.buyhold import load_daily_closes

    coins = sorted({r["cell"]["symbol"] for r in ledger})
    closes = load_daily_closes(coins, REPO_ROOT / "data" / "catalog")
    bh = {c: base._bh(closes, c) for c in coins}
    failed = sum(1 for r in ledger if r.get("failed"))
    missing = [r["key"] for r in winners if not nxt.get(r["key"])]
    print(f"cells={len(ledger)} failed={failed} winners={len(winners)} without_blind_eval={len(missing)}")
    print("blind B&H " + " ".join(f"{c}={bh[c]:+.4f}" if bh[c] is not None else f"{c}=n/a" for c in coins))
    base.PER_COIN = PER_COIN
    picked = base.select(winners)
    print(f"\n## PRIMARY: top {PER_COIN} per coin by May-Jul best Sharpe, tested once on Aug-Oct")
    n_pass = 0
    for r in picked:
        b, n = r["best"], nxt.get(r["key"])
        ok = passes(n)
        n_pass += ok
        coin_bh = bh[r["cell"]["symbol"]]
        blind = base.fmt_blind(n)
        if n and not n.get("error"):
            blind += " (beats B&H)" if coin_bh is not None and n["ret"] > coin_bh else " (below B&H)"
        print(
            f"{'PASS' if ok else 'FAIL'} {r['key']} | train sharpe={b['sharpe']:.3f} "
            f"ret={b['ret']:+.4f} trades={b['trades']} | blind {blind} | "
            f"worst-of-N {base.fmt_blind(worst.get(r['key']))}"
        )
    ref = sum(passes(nxt.get(r["key"])) for r in winners)
    pos = sum(1 for r in winners if (nxt.get(r["key"]) or {}).get("ret", 0) > 0)
    print(f"\nPRIMARY pass {n_pass}/{len(picked)}")
    print(f"REFERENCE all winners: pass {ref}/{len(winners)}, blind return > 0: {pos}/{len(winners)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
