"""NC1: the trend-null calibration (arch-c9-2 §5; chief rulings R-c9t5-1, R-c9t5-2).

Before .10 may use the rotation null's ``null_percentile``, the null must not flag
strategies that cannot have an edge. The null set is the pre-registered .10 arm
(long_flat, no filter, vol target 0.40, vol lookback 30, gross 1.0, band 0.10, spread
0.0001, E0 100,000; BTC/ETH/SOL/BNB 1m, 2023-06-01..2026-09-28) with its signal closes
lagged by K days, K in ``research.NC1_LAGS_DAYS`` (365, 372, ..., 708): 50 strategies
whose signal cannot see the last year. Each is one ``research.evaluate`` with
``signal_lag_days=K`` (draws 100, seed 0): a calibration run that never writes the
trial ledger. X = #{null_percentile >= 0.95}; ``research.nc1_verdict`` rules FAIL iff
X >= 6 (an undefined percentile counts as flagged). No retuning of K, draws or window
after the result.

Output (gitignored ``data/research/portfolio/nc1/``): ``family.json`` (the arm as a
one-arm family; DSR is irrelevant here), ``lag_<K>.json`` per lag (a rerun reuses a
file whose spec hash, draws and seed match, so an interrupted run resumes) and
``nc1.json`` with every percentile and the verdict. The closes for every lag come from
one extra load starting 708 days earlier (``research.lag_spec`` + ``extend_closes``).

    PYTHONPATH=$PWD .venv/bin/python scripts/research/trend_null_calibration.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe_quant.portfolio import research
from vibe_quant.portfolio.spec import CarrySpec
from vibe_quant.portfolio.ticks import load_carry_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.portfolio.ticks import CarryInputs

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data/catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data/archive/raw_data.db"
OUT_DIR = REPO_ROOT / "data/research/portfolio/nc1"

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
START = "2023-06-01"
END = "2026-09-28"
DRAWS = 100
SEED = 0
MAX_WORKERS = 4  # another research job shares the CPU

_INPUTS: CarryInputs | None = None  # the loaded run, one per process


def nc1_spec() -> CarrySpec:
    """The pre-registered .10 arm the lags are applied to."""
    return CarrySpec.from_dates(
        SYMBOLS,
        START,
        END,
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        vol_target=0.40,
        vol_lookback_days=30,
        gross=1.0,
        band=0.10,
        spread=0.0001,
        initial_balance=100_000.0,
    )


def load_nc1_inputs(catalog: Path, archive: Path) -> CarryInputs:
    """The NC1 run with trend closes reaching back to the longest lag (C9-17)."""
    spec = nc1_spec()
    inputs = load_carry_inputs(spec, catalog, archive)
    longer = load_carry_inputs(
        research.lag_spec(spec, max(research.NC1_LAGS_DAYS)), catalog, archive
    )
    closes = research.extend_closes(inputs.trend_closes or {}, longer.trend_closes or {})
    return replace(inputs, trend_closes=closes)


def _init(catalog: Path, archive: Path) -> None:
    global _INPUTS
    _INPUTS = load_nc1_inputs(catalog, archive)


def _key(spec: CarrySpec, lag: int) -> dict[str, Any]:
    return {"lag_days": lag, "spec_hash": spec.spec_hash(), "draws": DRAWS, "seed": SEED}


def _done(out_dir: Path, spec: CarrySpec, lag: int) -> dict[str, Any] | None:
    """The saved row of ``lag`` when its key matches this run, else None."""
    path = out_dir / f"lag_{lag}.json"
    if not path.is_file():
        return None
    row: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    key = _key(spec, lag)
    return row if all(row.get(name) == value for name, value in key.items()) else None


def _evaluate_lag(lag: int, out_dir: Path) -> dict[str, Any]:
    """One lagged evaluate; its row is written before it is returned."""
    assert _INPUTS is not None, "_init was not run in this process"
    started = time.perf_counter()
    result = research.evaluate(
        _INPUTS,
        draws=DRAWS,
        seed=SEED,
        trials_path=out_dir / "trials.jsonl",
        family=research.load_family(out_dir / "family.json"),
        criteria=None,
        signal_lag_days=lag,
    )
    row = {
        **_key(_INPUTS.spec, lag),
        "null_percentile": result["null_percentile"],
        "sharpe": result["sharpe"],
        "lag_mask_mismatches": result["lag_mask_mismatches"],
        "wall_s": time.perf_counter() - started,
        "result": result,
    }
    (out_dir / f"lag_{lag}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return row


def _workers(value: str) -> int:
    workers = int(value)
    if not 1 <= workers <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"--workers must be in [1, {MAX_WORKERS}]")
    return workers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--workers", type=_workers, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = nc1_spec()
    family = {"n_prior": 0, "arms": [spec.arm_hash()]}
    (out_dir / "family.json").write_text(json.dumps(family) + "\n", encoding="utf-8")

    lags = research.NC1_LAGS_DAYS
    todo = [lag for lag in lags if _done(out_dir, spec, lag) is None]
    started = time.perf_counter()
    if todo and args.workers == 1:
        _init(CATALOG_PATH, ARCHIVE_DB_PATH)
        for lag in todo:
            _evaluate_lag(lag, out_dir)
    elif todo:
        with ProcessPoolExecutor(
            args.workers, initializer=_init, initargs=(CATALOG_PATH, ARCHIVE_DB_PATH)
        ) as pool:
            list(pool.map(_evaluate_lag, todo, [out_dir] * len(todo)))
    wall = time.perf_counter() - started

    rows = [_done(out_dir, spec, lag) for lag in lags]
    detail = [
        {
            name: row[name]
            for name in ("lag_days", "null_percentile", "sharpe", "lag_mask_mismatches")
        }
        for row in rows
        if row is not None
    ]
    verdict = research.nc1_verdict([row["null_percentile"] for row in detail])
    summary = {
        "label": "calibration",
        "spec": spec.to_json(),
        "spec_hash": spec.spec_hash(),
        "draws": DRAWS,
        "seed": SEED,
        "percentile_bar": research.NC1_PERCENTILE,
        **verdict,
        "lags_detail": detail,
        "evaluated_now": len(todo),
        "workers": args.workers,
        "wall_s": wall,
    }
    (out_dir / "nc1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in detail:
        print(
            f"lag {row['lag_days']}: null_percentile={row['null_percentile']!r} "
            f"sharpe={row['sharpe']!r} mask_mismatches={row['lag_mask_mismatches']}"
        )
    print(
        f"NC1: X={verdict['flagged']}/{verdict['lags']} fail_at={verdict['fail_at']} "
        f"verdict={verdict['verdict']} (calibration) evaluated={len(todo)} "
        f"workers={args.workers} wall={wall:.1f}s out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
