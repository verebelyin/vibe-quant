"""Blind 2026 RSI test #2, 1m low-frequency arm: search May-Jul 2026, test once on Aug-Oct 2026.

PRE-REGISTERED 2026-11-03 (chief; user: "lets try 1m but try to make it with parameters with
fewer trades"), committed BEFORE this arm ran. Separate arm of blind_2026_rsi.py: its own pick
list, so these cells never change the 15m/1h picks. Scratch burst, NOT in DSR N.

Search (burst.py run --tier fast --allin --max-trades 300, out data/swarm/burst-20261103-blind-rsi-1m):
    window 2026-05-01..2026-08-01, worst of 3 sub-windows; timeframe 1m (fast tier, NT fallback
    for genomes it cannot replay); 9 RSI families: RSI, ADAPTIVE_RSI, RSI+KAMA, RSI+ADX,
    RSI+DONCHIAN, RSI+PRICE_POSITION, RSI+TREND_ENSEMBLE, RSI+RAMS, RSI+ROC (the FUNDING /
    BTC_TREND / OWN_TREND families are left out: the fast tier cannot replay them, so 1m cells
    would run on NT); long/short/both; the 8 catalog coins; 1/3/5x all-in; pop 12 x 3; 648 cells.
    FEWER TRADES: a winner needs 50..300 trades over May-Jul (<= ~3.3/day); a genome over 300
    trades is never a winner (burst --max-trades 300; smoke: median 1m genome 470-670 trades).

Blind window, stage 1 rule (top 3 per coin by May-Jul best Sharpe; PASS = >= 15 trades, return
> 0, Sharpe >= 1.0; reference = all winners) and stage 2 (full-fidelity validation of each PASS
on Aug-Oct) are exactly blind_2026_rsi.py's; this file only points it at the 1m ledger.

    PYTHONPATH=$PWD .venv/bin/python scripts/research/blind_2026_rsi1m.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import blind_2026_rsi  # noqa: E402

OUT = REPO_ROOT / "data" / "swarm" / "burst-20261103-blind-rsi-1m"

if __name__ == "__main__":
    sys.exit(blind_2026_rsi.main(["--out", str(OUT), *sys.argv[1:]]))
