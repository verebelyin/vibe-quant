"""Null check for the shadow tracker evidence rule (design arch-c7-2 §6): per-row null, family F1, power.

Runs the production rule (``vibe_quant.shadow.estat.log_e_path`` over 7-day block sums, min 26 blocks) on
simulated return paths with no edge and prints the §6 tables as markdown.

    PYTHONPATH=$PWD .venv/bin/python scripts/research/shadow_null_check.py [--quick]

Full: per-row 16,000 paths x 1,825 days (seed 7), family 8,000 paths (seed 5), power 2,000 paths (seed 11);
MEASURED 5.1 s + 24.0 s + 0.8 s idle (rev-c7t1-1), ~8 s + 52 s + 3 s under swarm load.
``--quick``: 2,000 / 1,000 / 500 paths (indicative only).
Falsifier (pre-registered): any per-row rate @100 > 0.01 or any family rate > 0.05 -> exit 1 (full mode).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.shadow.estat import (
    E_PASS,
    FUTILITY_MIN_DAYS,
    FUTILITY_SR,
    SLOTS,
    Z95,
    block_sums,
    log_e_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable

HORIZON = 1825
CHUNK = 2000
THRESHOLDS = (20.0, 100.0, 200.0)
PER_ROW_BAR = 1.0 / E_PASS  # 0.01 per slot
FAMILY_BAR = 0.05
NULLS = ("normal", "t3", "garch", "flat70", "ar0.3", "ar0.5", "art3_0.5")
NULL_LABEL = {
    "normal": "normal",
    "t3": "t3",
    "garch": "GARCH(1,1)-t5",
    "flat70": "70 %-flat t3",
    "ar0.3": "AR(1) 0.3",
    "ar0.5": "AR(1) 0.5",
    "art3_0.5": "AR(1) 0.5, t3",
}
# Family F1: slot k enrols on ENROL[k]; slot index 3 re-enrols slot 0's strategy after slot 0 retires on day 400.
ENROL = (0, 120, 240, 400, 520)
RETIRE = {0: 400}
REENROL_OF_0 = 3
FAMILY_VOL = 0.02
FAMILY_DD_LIMIT = 0.5


def _ar(x: np.ndarray, rho: float) -> np.ndarray:
    for d in range(1, x.shape[1]):
        x[:, d] = rho * x[:, d - 1] + math.sqrt(1 - rho * rho) * x[:, d]
    return x


def _ar_normal(rng: np.random.Generator, paths: int, rho: float) -> np.ndarray:
    x = rng.standard_normal((paths, HORIZON))
    return _ar(x, rho) if rho else x


def gen(kind: str, paths: int, days: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance null paths (same draws as the design probes)."""
    if kind == "normal":
        return rng.standard_normal((paths, days))
    if kind == "t3":
        return rng.standard_t(3, (paths, days)) / math.sqrt(3)
    if kind == "flat70":
        return rng.standard_t(3, (paths, days)) / math.sqrt(3) * (rng.random((paths, days)) < 0.3)
    if kind == "garch":
        x = np.empty((paths, days))
        h = np.ones(paths)
        e = np.zeros(paths)
        for i in range(days):
            h = 0.05 + 0.1 * e * e + 0.85 * h
            e = np.sqrt(0.6 * h) * rng.standard_t(5, paths)
            x[:, i] = e
        return x
    if kind.startswith("art3_"):
        return _ar(rng.standard_t(3, (paths, days)) / math.sqrt(3), float(kind[5:]))
    if kind.startswith("ar"):
        return _ar(rng.standard_normal((paths, days)), float(kind[2:]))
    raise ValueError(kind)


def _log_e(x: np.ndarray) -> np.ndarray:
    return log_e_path(block_sums(x))


def per_row(paths: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for kind in NULLS:
        rng = np.random.default_rng(7)
        hits = np.zeros(len(THRESHOLDS))
        done = 0
        while done < paths:
            n = min(CHUNK, paths - done)
            le = _log_e(gen(kind, n, HORIZON, rng))
            hits += [(le >= math.log(t)).any(axis=1).sum() for t in THRESHOLDS]
            done += n
        out[kind] = list(hits / paths)
    return out


def _stop_day(x: np.ndarray) -> np.ndarray:
    """First day (row clock, 0-based) a futility (Sharpe bound) or drawdown stop fires, else len."""
    n = x.shape[1]
    k = np.arange(1, n + 1)
    cs = np.cumsum(x, 1)
    csq = np.cumsum(x * x, 1)
    sd = np.sqrt(np.maximum((csq - cs * cs / k) / np.maximum(k - 1, 1), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(sd > 0, cs / k / sd, np.nan)
    ub = mu * math.sqrt(365) + Z95 * np.sqrt(365 * (1 + mu * mu / 2) / k)
    eq = np.cumprod(1 + FAMILY_VOL * x, 1)
    dd = 1 - eq / np.maximum.accumulate(np.maximum(eq, 1.0), 1)
    stop = ((k >= FUTILITY_MIN_DAYS) & (ub < FUTILITY_SR)) | (dd > FAMILY_DD_LIMIT)
    return np.where(stop.any(1), stop.argmax(1), n)


def family(paths: int, rho: float, common: float, stops: bool) -> float:
    """P(any of the SLOTS slots passes at E >= E_PASS) with staggered enrolment and a re-enrolment."""
    rng = np.random.default_rng(5)
    base = _ar_normal(rng, paths, rho)
    strat0 = _ar_normal(rng, paths, rho)  # slot 0 and its re-enrolment trade the same strategy
    hit = np.zeros(paths, bool)
    for k, start in enumerate(ENROL[:SLOTS]):
        idio = strat0 if k in (0, REENROL_OF_0) else _ar_normal(rng, paths, rho)
        x = (math.sqrt(common) * base + math.sqrt(1 - common) * idio)[
            :, start : RETIRE.get(k, HORIZON)
        ]
        crossed = _log_e(x) >= math.log(E_PASS)
        any_cross = crossed.any(1)
        pass_day = (crossed.argmax(1) + 1) * 7 - 1  # last day of the first crossing block
        limit = _stop_day(x) if stops else x.shape[1]
        hit |= any_cross & (pass_day < limit)
    return float(hit.mean())


def power(paths: int, kind: str, sr: float, days: int) -> float:
    rng = np.random.default_rng(11)
    x = gen(kind, paths, days, rng) + sr / math.sqrt(365)  # population shift fixed before drawing
    return float((_log_e(x) >= math.log(E_PASS)).any(axis=1).mean())


def _timed(label: str, fn: Callable[[], None]) -> None:
    t0 = time.perf_counter()
    fn()
    print(f"\n({label}: {time.perf_counter() - t0:.1f} s)\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--quick", action="store_true", help="2,000 / 1,000 / 500 paths (indicative only)"
    )
    args = ap.parse_args(argv)
    n_row, n_fam, n_pow = (2000, 1000, 500) if args.quick else (16000, 8000, 2000)
    fails: list[str] = []

    def table_row() -> None:
        print(
            f"## Per-row null: P(E crosses threshold within {HORIZON} days), {n_row:,} paths, seed 7\n"
        )
        print("| null | @20 | @100 | @200 |\n|---|---|---|---|")
        for kind, rates in per_row(n_row).items():
            print(
                f"| {NULL_LABEL[kind]} | " + " | ".join(f"{r:.5f}" for r in rates) + " |",
                flush=True,
            )
            if rates[1] > PER_ROW_BAR:
                fails.append(f"per-row {NULL_LABEL[kind]} @100 {rates[1]:.5f} > {PER_ROW_BAR}")

    def table_family() -> None:
        print(
            f"## Family F1: P(any false pass), {SLOTS} slots, E >= {E_PASS:g}, {n_fam:,} paths, seed 5\n"
        )
        print("| AR(1) | common factor | stops | P(any) |\n|---|---|---|---|")
        for rho in (0.0, 0.3, 0.5):
            for common in (0.0, 0.5):
                for stops in (False, True):
                    p = family(n_fam, rho, common, stops)
                    print(
                        f"| {rho} | {common} | {'yes' if stops else 'no'} | {p:.4f} |", flush=True
                    )
                    if p > FAMILY_BAR:
                        fails.append(
                            f"family rho={rho} common={common} stops={stops} {p:.4f} > {FAMILY_BAR}"
                        )

    def table_power() -> None:
        print(f"## Power: P(pass @{E_PASS:g}) by true Sharpe, {n_pow:,} paths, seed 11\n")
        print("| null | Sharpe | 1 y | 2 y | 3 y |\n|---|---|---|---|---|")
        for kind in ("t3", "ar0.3"):
            for sr in (1.0, 1.5, 2.0, 3.0):
                cells = " | ".join(f"{power(n_pow, kind, sr, d):.3f}" for d in (365, 730, 1095))
                print(f"| {NULL_LABEL[kind]} | {sr} | {cells} |", flush=True)

    _timed("per-row", table_row)
    _timed("family", table_family)
    _timed("power", table_power)
    if fails:
        print("FALSIFIER HIT:\n" + "\n".join(f"- {f}" for f in fails))
    else:
        print(
            f"FALSIFIER: none (every per-row @100 <= {PER_ROW_BAR}, every family <= {FAMILY_BAR})"
        )
    if args.quick:
        print("(quick mode: indicative only; the journaled RESULT needs the full run)")
        return 0
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
