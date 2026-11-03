"""Blind 2026 test: search Jan-Jul 2026 per coin, test once on Aug-Oct 2026.

PRE-REGISTERED 2026-11-03 (chief, user-approved option A), committed BEFORE the search ran.
Scratch burst (NOT in DSR N); code = main after the q3eq5.2 costed-sizing break.

Search (burst.py run, out dir data/swarm/burst-20261103-blind-2026):
    window 2026-01-01..2026-08-01 (end exclusive) only; worst of 3 sub-windows (--eval-windows 3);
    families: the 8 burst PAIRS + RSI alone (9); directions long, short, both;
    coins BTC ETH SOL BNB NEAR SUI ZEC HYPE (USDT perps); timeframes 1h and 4h; 3x leverage;
    NT tier, pop 12 x 3 gens, seed 1 -> 432 cells (corrected from '9 PAIRS / 480' at launch,
    before any result: burst.PAIRS has 8; the grid run is unchanged).
    A cell's winner = its best genome with >= 50 trades and return > 0 on Jan-Jul (burst rule).

Blind window: 2026-08-01..2026-10-20 (end exclusive; 2026-10-19 = last full catalog day), via
burst.py transfer's `next` evaluation of the SAME genome on the SAME coin. Never used to select.

Selection (fixed before any Aug-Oct number is read):
    PRIMARY = per coin, the 2 winners with the highest Jan-Jul best Sharpe (ties: higher
    Jan-Jul return, then key). At most 16 strategies. Every one is reported.
    A selected strategy PASSES iff, on Aug-Oct: trades >= 15 AND return > max(0, the coin's
    buy-and-hold return over 2026-08-01..2026-10-19 daily closes).
    REFERENCE = the same pass rule over ALL winners (shows whether the Jan-Jul ranking adds
    anything over a random winner).
Reading: a PASS is a paper-trading candidate, not proof (up to 16 tests; ~3 months of data).
A FAIL is final for that strategy: no re-tuning on Aug-Oct.

    PYTHONPATH=$PWD .venv/bin/python scripts/research/blind_2026.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "data" / "swarm" / "burst-20261103-blind-2026"
TRAIN = ("2026-01-01", "2026-08-01")
BLIND = ("2026-08-01", "2026-10-20")
PER_COIN = 2
MIN_BLIND_TRADES = 15


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bh(closes: dict[str, dict[int, float]], symbol: str) -> float | None:
    from datetime import date

    def day(s: str) -> int:
        return (date.fromisoformat(s) - date(1970, 1, 1)).days

    series = closes.get(symbol, {})
    first, last = series.get(day(BLIND[0]) - 1), series.get(day(BLIND[1]) - 1)
    return None if first is None or last is None else last / first - 1.0


def passes(nxt: dict | None, bh: float | None) -> bool:
    if not nxt or nxt.get("error") or bh is None:
        return False
    return nxt["trades"] >= MIN_BLIND_TRADES and nxt["ret"] > max(0.0, bh)


def select(winners: list[dict]) -> list[dict]:
    by_coin: dict[str, list[dict]] = {}
    for row in winners:
        by_coin.setdefault(row["cell"]["symbol"], []).append(row)
    picked: list[dict] = []
    for coin in sorted(by_coin):
        ranked = sorted(
            by_coin[coin], key=lambda r: (-r["best"]["sharpe"], -r["best"]["ret"], r["key"])
        )
        picked.extend(ranked[:PER_COIN])
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    ledger = [
        r for r in _rows(args.out / "ledger.jsonl")
        if (r["cell"]["start"], r["cell"]["end"]) == TRAIN
    ]  # fmt: skip
    winners = [r for r in ledger if not r.get("failed") and r.get("best")]
    nxt = {r["key"]: r.get("next") for r in _rows(args.out / "transfer.jsonl")}
    from vibe_quant.discovery.buyhold import load_daily_closes

    coins = sorted({r["cell"]["symbol"] for r in ledger})
    closes = load_daily_closes(coins, REPO_ROOT / "data" / "catalog")
    bh = {c: _bh(closes, c) for c in coins}
    missing = [r["key"] for r in winners if r["key"] not in nxt]
    print(f"cells={len(ledger)} winners={len(winners)} winners_without_blind_eval={len(missing)}")
    print("blind B&H " + " ".join(f"{c}={bh[c]:+.4f}" if bh[c] is not None else f"{c}=n/a" for c in coins))
    print("\n## PRIMARY: top 2 per coin by Jan-Jul best Sharpe, tested once on Aug-Oct")
    picked = select(winners)
    n_pass = 0
    for r in picked:
        b, n = r["best"], nxt.get(r["key"])
        ok = passes(n, bh[r["cell"]["symbol"]])
        n_pass += ok
        blind = (
            f"ret={n['ret']:+.4f} trades={n['trades']} sharpe={n['sharpe']}" if n and not n.get("error")
            else f"no blind eval ({(n or {}).get('error')})"
        )  # fmt: skip
        print(
            f"{'PASS' if ok else 'FAIL'} {r['key']} | train sharpe={b['sharpe']:.3f} "
            f"ret={b['ret']:+.4f} trades={b['trades']} | blind {blind} | bh={bh[r['cell']['symbol']]}"
        )
    ref = sum(passes(nxt.get(r["key"]), bh[r["cell"]["symbol"]]) for r in winners)
    pos = sum(1 for r in winners if (nxt.get(r["key"]) or {}).get("ret", 0) > 0)
    print(f"\nPRIMARY pass {n_pass}/{len(picked)}")
    print(f"REFERENCE all winners: pass {ref}/{len(winners)}, blind return > 0: {pos}/{len(winners)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
