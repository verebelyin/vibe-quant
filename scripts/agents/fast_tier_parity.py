"""Fast tier vs NT screening: seeded random genomes, fill-by-fill and metric-by-metric.

Runs ``--n`` genomes from ``initialize_population`` (``random.seed(--seed)``, the full
non-context pool or ``--pool``) through ``NTBacktestFn`` and ``FastBacktestFn`` on one
symbol and window, and compares every fill (ts, side, type, qty, price, commission) and the
metrics ``total_trades, total_return, sharpe_ratio, profit_factor, max_drawdown`` EXACTLY.
Exit 0 only when every genome matches (an unsupported genome is reported, not counted as a
match or a failure). Scratch DB under /tmp; the real state DB is never opened.

    PYTHONPATH=$PWD .venv/bin/python scripts/agents/fast_tier_parity.py \\
        --symbol BTCUSDT --start 2026-03-02 --end 2026-03-09 --n 12 --seed 1 [--liq isolated --leverage 10]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("total_trades", "total_return", "sharpe_ratio", "profit_factor", "max_drawdown")


def _fills(positions: list[Any]) -> list[tuple[Any, ...]]:
    out = []
    for pos in sorted(positions, key=lambda p: int(p.ts_opened)):
        for f in pos.events:
            out.append(
                (int(f.ts_event), f.order_side.name, f.order_type.name, str(f.last_qty),
                 str(f.last_px), str(f.commission))
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pool", default="", help="comma list; default = full non-context pool")
    ap.add_argument("--liq", choices=("off", "isolated", "cross"), default="off")
    ap.add_argument("--leverage", type=int, default=10)
    ap.add_argument("--only", type=int, default=-1, help="run only genome index N")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="fast-parity-", dir="/tmp"))
    os.environ["VIBE_QUANT_DB"] = str(tmp / "state.db")
    os.environ["VIBE_QUANT_LOG_DIR"] = str(tmp / "logs")
    os.chdir(REPO_ROOT)

    from vibe_quant.discovery import operators
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.dsl.indicators import indicator_registry
    from vibe_quant.fast_tier import FastTierUnsupported
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation.extraction import all_positions

    operators._ensure_pool()
    allowed = (
        {s.strip() for s in args.pool.split(",") if s.strip()}
        if args.pool
        else {n for n in operators.INDICATOR_POOL if not getattr(indicator_registry.get(n), "needs_context", False)}
    )
    for name in [k for k in operators.INDICATOR_POOL if k not in allowed]:
        del operators.INDICATOR_POOL[name]
    operators._INDICATOR_NAMES[:] = list(operators.INDICATOR_POOL)
    if args.pool:
        from vibe_quant.discovery.genome import MA_POOL

        for name in [k for k in MA_POOL if k not in allowed]:
            del MA_POOL[name]
        operators._MA_NAMES[:] = list(MA_POOL)
    random.seed(args.seed)
    genomes = operators.initialize_population(size=args.n)

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    nt_runner.NTScreeningRunner._extract_metrics = grab  # type: ignore[method-assign]
    kw: dict[str, Any] = {}
    if args.liq != "off":
        kw["liquidation"] = LiquidationSettings(mode=args.liq, leverage=args.leverage)
    nt_fn = NTBacktestFn([args.symbol], "1m", args.start, args.end, **kw)
    fast_fn = FastBacktestFn([args.symbol], "1m", args.start, args.end, **kw)

    ok = bad = unsupported = 0
    t_nt = t_fast = 0.0
    for idx, ch in enumerate(genomes):
        if args.only >= 0 and idx != args.only:
            continue
        label = f"#{idx} {ch.direction.name} " + "+".join(sorted({g.indicator_type for g in ch.entry_genes + ch.exit_genes}))
        t = time.perf_counter()
        try:
            fast = fast_fn._run_single(ch, args.start, args.end)
        except FastTierUnsupported as exc:
            unsupported += 1
            print(f"{label}: UNSUPPORTED {exc}")
            continue
        t_fast += time.perf_counter() - t
        fast_fills = _fills(fast_fn.last_positions)
        t = time.perf_counter()
        captured.clear()
        nt = nt_fn._run_single(ch, args.start, args.end)
        t_nt += time.perf_counter() - t
        nt_fills = _fills(captured.get("positions", []))
        diffs = [m for m in METRICS if nt[m] != fast[m]]
        first = next((i for i, (a, b) in enumerate(zip(nt_fills, fast_fills, strict=False)) if a != b), None)
        if not diffs and nt_fills == fast_fills:
            ok += 1
            print(f"{label}: MATCH trades={nt['total_trades']} return={nt['total_return']}")
            continue
        bad += 1
        print(f"{label}: MISMATCH fills nt={len(nt_fills)} fast={len(fast_fills)} first_diff={first}")
        for m in diffs:
            print(f"    {m}: nt={nt[m]!r} fast={fast[m]!r}")
        if first is not None:
            for j in range(max(0, first - 1), min(first + 2, max(len(nt_fills), len(fast_fills)))):
                print(f"    fill {j}: nt={nt_fills[j] if j < len(nt_fills) else None}")
                print(f"    fill {j}: fa={fast_fills[j] if j < len(fast_fills) else None}")
        elif len(nt_fills) != len(fast_fills):
            j = min(len(nt_fills), len(fast_fills))
            print(f"    extra: nt={nt_fills[j:j + 2]} fast={fast_fills[j:j + 2]}")
    print(
        f"SUMMARY match={ok} mismatch={bad} unsupported={unsupported} "
        f"nt_s={t_nt:.1f} fast_s={t_fast:.1f}"
    )
    return 0 if bad == 0 and ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
