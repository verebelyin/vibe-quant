"""Measure how often SCREENING under-fills market orders that VALIDATION fills fully.

Generated entries and reduce-only signal exits are market IOC. On the screening FillModel an
order above the bar's synthetic L1 size (bar_volume/4) fills once at that size and the rest is
canceled; validation (VolumeSlippageFillModel) fills all of it. This script runs a fixed sample
of strategies through NTScreeningRunner at initial_balance 1000 and 100000 (plus strategy 239
through ValidationRunner), reads the engine's order cache after each run and prints one table.

Measurement only: no engine change; the state DB is opened read-only (validation runs on a
throwaway backup of it); data/catalog and data/archive are only read.

Run from the tree under test:
    PYTHONPATH=$PWD .venv/bin/python scripts/research/underfill/measure.py [--only 239,245]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from vibe_quant.balance_settings import parse_initial_balance
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.fees import FeeSchedule
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.screening.pipeline import _dsl_to_dict
from vibe_quant.sizing_settings import SizingSettings
from vibe_quant.validation.runner import ValidationRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "data/catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data/archive/raw_data.db"
BALANCES = (1000, 100000)
WINDOW_CAP_DAYS = 182  # "last 6 months" of the stored run's window

# strategy 239 on the exactness window, then the most recent distinct stored strategies that
# cover 1m/5m/1h/4h and BTC/ETH/SOL/BNB (picked from backtest_runs, see the README).
FIXED_239 = (239, ["BTCUSDT"], "2024-01-01", "2026-03-17")
SAMPLE_IDS = (245, 244, 243, 229, 16, 223, 219, 2)


def _find_state_db() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    ).stdout.strip()
    return Path(common).parent / "data/state/vibe_quant.db"


def _capped_window(start: str, end: str) -> tuple[str, str]:
    lo = max(date.fromisoformat(start), date.fromisoformat(end) - timedelta(days=WINDOW_CAP_DAYS))
    return lo.isoformat(), end


def _load_sample(db: Path, only: set[int] | None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: list[dict[str, Any]] = []
    try:
        for sid in (FIXED_239[0], *SAMPLE_IDS):
            if only is not None and sid not in only:
                continue
            row = conn.execute("SELECT dsl_config FROM strategies WHERE id = ?", (sid,)).fetchone()
            dsl = validate_strategy_dict(json.loads(row[0]))
            if sid == FIXED_239[0]:
                _, symbols, start, end = FIXED_239
                params: dict[str, Any] = {}
            else:
                r = conn.execute(
                    "SELECT symbols, start_date, end_date, parameters FROM backtest_runs "
                    "WHERE strategy_id = ? AND run_mode IN ('screening','validation') "
                    "ORDER BY id DESC LIMIT 1",
                    (sid,),
                ).fetchone()
                symbols = json.loads(r[0])
                start, end = _capped_window(r[1], r[2])
                params = json.loads(r[3] or "{}")
            out.append({"sid": sid, "dsl": dsl, "symbols": symbols, "start": start,
                        "end": end, "params": params})
    finally:
        conn.close()
    return out


def _collect(engine: Any) -> dict[str, Any]:
    """Per-order and per-position facts from the engine cache (after the run)."""
    from nautilus_trader.model.enums import OrderType

    from vibe_quant.validation.extraction import all_positions

    cache = engine.cache
    pos_of_order: dict[str, Any] = {}
    # Netting reuses position ids: closed incarnations are snapshots, so key by index.
    positions: list[Any] = all_positions(engine)
    pos_key = {id(p): f"{p.id}#{i}" for i, p in enumerate(positions)}
    for pos in positions:
        for ev in pos.events:
            pos_of_order[str(ev.client_order_id)] = pos
    orders = []
    for o in cache.orders():
        if o.order_type != OrderType.MARKET:
            continue
        pos = pos_of_order.get(str(o.client_order_id))
        orders.append({
            "id": str(o.client_order_id),
            "symbol": str(o.instrument_id.symbol),
            "exit": bool(o.is_reduce_only),
            "qty": float(o.quantity),
            "filled": float(o.filled_qty),
            "status": o.status_string(),
            "pos": None if pos is None else pos_key[id(pos)],
            "pos_open": None if pos is None else bool(pos.is_open),
        })
    pos_pnl = {
        pos_key[id(p)]: (str(p.instrument_id.symbol), float(p.realized_pnl.as_double()))
        for p in positions
    }
    return {"orders": orders, "positions": pos_pnl}


def _med(xs: list[float]) -> str:
    return f"{statistics.median(xs):.4f}" if xs else "-"


def _min(xs: list[float]) -> str:
    return f"{min(xs):.4f}" if xs else "-"


def _summarise(label: str, balance: int, tier: str, cap: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    symbols = sorted({o["symbol"] for o in cap["orders"]} | {s for s, _ in cap["positions"].values()})
    for sym in symbols:
        orders = [o for o in cap["orders"] if o["symbol"] == sym]
        ents = [o for o in orders if not o["exit"]]
        exits = [o for o in orders if o["exit"]]
        under = [o for o in orders if o["filled"] < o["qty"]]
        under_ent = [o for o in ents if o["filled"] < o["qty"]]
        under_exit = [o for o in exits if o["filled"] < o["qty"]]
        bad_pos = {o["pos"] for o in under if o["pos"] is not None}
        pnl = {pid: v for pid, (s, v) in cap["positions"].items() if s == sym}
        total = sum(abs(v) for v in pnl.values())
        carried = sum(abs(v) for pid, v in pnl.items() if pid in bad_pos)
        ratios = [o["filled"] / o["qty"] for o in orders]
        rows.append({
            "strategy": label, "symbol": sym, "tier": tier, "balance": balance,
            "entries": len(ents), "entries_under": len(under_ent),
            "exits": len(exits), "exits_under": len(under_exit),
            "ratio_median_all": _med(ratios), "ratio_min_all": _min(ratios),
            "ratio_median_under": _med([o["filled"] / o["qty"] for o in under]),
            "abs_pnl_total": round(total, 4), "abs_pnl_underfilled": round(carried, 4),
            "pnl_share": f"{carried / total:.4f}" if total else "-",
            "residual_exits_open_at_end": sum(1 for o in under_exit if o["pos_open"]),
            "residual_qty": round(sum(o["qty"] - o["filled"] for o in under_exit), 8),
        })
    return rows


def _screen(item: dict[str, Any], balance: int) -> dict[str, Any]:
    params = {**item["params"], "initial_balance": balance}
    liq = LiquidationSettings.from_params(params)
    sizing = SizingSettings.from_params(params)
    runner = NTScreeningRunner(
        _dsl_to_dict(item["dsl"]), item["symbols"], item["start"], item["end"],
        catalog_path=str(CATALOG_PATH), funding_archive_path=str(ARCHIVE_DB_PATH),
        launch=LaunchSettings(
            fee_schedule=FeeSchedule.from_params(params),
            liquidation=liq, sizing=sizing,
            starting_balance=parse_initial_balance(params),
        ),
    )
    captured: list[dict[str, Any]] = []
    orig = runner._extract_metrics

    def hook(p: Any, bt_result: Any, engine: Any, *a: Any, **kw: Any) -> Any:
        captured.append(_collect(engine))
        return orig(p, bt_result, engine, *a, **kw)

    runner._extract_metrics = hook  # type: ignore[method-assign,assignment]
    runner({})
    return captured[0]


def _validate(db: Path, item: dict[str, Any], balance: int) -> dict[str, Any]:
    tmpdir = Path(tempfile.mkdtemp(prefix="underfill-"))
    try:
        copy = tmpdir / "v.db"
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        dst = sqlite3.connect(str(copy))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        conn = sqlite3.connect(str(copy))
        run_id = conn.execute(
            "INSERT INTO backtest_runs(strategy_id, run_mode, symbols, timeframe, "
            "start_date, end_date, parameters) VALUES (?,?,?,?,?,?,?)",
            (item["sid"], "validation", json.dumps(item["symbols"]), item["dsl"].timeframe,
             item["start"], item["end"], json.dumps({"initial_balance": balance})),
        ).lastrowid
        conn.commit()
        conn.close()
        previous = Path.cwd()
        os.chdir(REPO_ROOT)  # ValidationRunner resolves catalog/archive relative to cwd
        runner = ValidationRunner(db_path=copy, logs_path=tmpdir / "logs")
        captured: list[dict[str, Any]] = []
        orig = runner._extract_results

        def hook(*a: Any, **kw: Any) -> Any:
            captured.append(_collect(kw["engine"] if "engine" in kw else a[3]))
            return orig(*a, **kw)

        runner._extract_results = hook  # type: ignore[method-assign]
        try:
            assert run_id is not None
            runner.run(run_id)
        finally:
            runner.close()
            os.chdir(previous)
        return captured[0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


COLS = [
    ("strategy", "strat"), ("symbol", "symbol"), ("tier", "tier"), ("balance", "bal"),
    ("entries", "ent"), ("entries_under", "ent_u"), ("exits", "exit"), ("exits_under", "exit_u"),
    ("ratio_median_all", "ratio_med"), ("ratio_min_all", "ratio_min"),
    ("ratio_median_under", "ratio_med_u"), ("abs_pnl_total", "|pnl|"),
    ("abs_pnl_underfilled", "|pnl|_u"), ("pnl_share", "share"),
    ("residual_exits_open_at_end", "res_open"), ("residual_qty", "res_qty"),
]


def _table(rows: list[dict[str, Any]]) -> str:
    cells = [[h for _, h in COLS]] + [[str(r[k]) for k, _ in COLS] for r in rows]
    widths = [max(len(c[i]) for c in cells) for i in range(len(COLS))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)) for line in cells)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated strategy ids")
    ap.add_argument("--out", default=None, help="JSON path (default: job dir in the main checkout)")
    args = ap.parse_args(argv)
    only = {int(x) for x in args.only.split(",")} if args.only else None
    db = _find_state_db()
    out = Path(args.out) if args.out else (
        db.parents[2] / "data/swarm/20261017-easy3/underfill.json"
    )
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for item in _load_sample(db, only):
        label = f"{item['sid']}@{item['dsl'].timeframe}"
        for balance in BALANCES:
            t0 = time.time()
            cap = _screen(item, balance)
            raw[f"{label}/screening/{balance}"] = cap
            rows += _summarise(label, balance, "screening", cap)
            print(f"[{time.time() - t0:5.1f}s] {label} {item['symbols']} {item['start']}..{item['end']} "
                  f"bal={balance} screening", file=sys.stderr)
            if item["sid"] == FIXED_239[0]:
                t0 = time.time()
                cap = _validate(db, item, balance)
                raw[f"{label}/validation/{balance}"] = cap
                rows += _summarise(label, balance, "validation", cap)
                print(f"[{time.time() - t0:5.1f}s] {label} bal={balance} validation", file=sys.stderr)
    table = _table(rows)
    print(table)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "table": table, "orders": raw}, indent=1) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
