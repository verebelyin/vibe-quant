"""Third exactness proof: the funding-carry engine == the no-NT reference (T6b).

Runs the pinned 4-coin carry book (BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT,
2022-06-01..2026-09-28, E0 10,000, divisor 3, gross 1.0, band 0.10) on the real
catalog and funding archive, then

1. compares the NT engine run, the reference run and both metric reports field by
   field with ``parity.compare_runs`` -- exact equality, no tolerance (R30); and
2. asserts the engine's recorded headline numbers (the T6b pins).

Exit 0 = IDENTICAL, 1 = DRIFT. Data paths resolve from this script's repo root,
so the proof runs from any cwd.

``--variants`` (design x0axf.14 T7) instead proves the three opt-in variants on the
same 4 coins and dates, each against a matched control that differs only in the
toggled field: engine == reference (``compare_runs == []``) under the variant, and
the variant really acts:

* inverse_vol vs equal, BOTH at divisor 2 (k = 2 a side; at divisor 3 a side has one
  leg and inverse vol is a no-op): >= 1 decision with two legs a side keeps the
  control's selected sets and changes an absolute weight within a side, with a vol
  for every selected leg;
* ``min_quote_volume=MIN_QUOTE_VOLUME`` vs off: >= 1 decision where a priced symbol
  the control selects is ``illiquid``, signal None, target 0, in both executors;
* ``liquidation_mode=cross`` vs off: both executors' summaries carry equal cross
  notes (a bust raises and fails the proof).

One line per variant (``mismatches``, ``feature``); exit 1 on any failure. It
writes no trial line and no carry output.

``--signal trend`` (arch-c9-2 R67/R68) proves the trend book on the same 4 coins and
dates: 1m, long_short, vol target 0.40, vol lookback 30, gross 1, band 0.10, spread
0.0001, E0 100,000. NT == reference (``compare_runs == []``), the recorded pins
(``TREND_RECORDED``), and a run ending 2025-06-01 reproduces every decision, fill and
payment before that date (truncation). The 1h tier runs the same parity only when the
archive has 1h klines for all four coins; otherwise it prints
``SKIP 1h: no archive 1h rows`` (exit unaffected).

Run from the tree under test (in a worktree: PYTHONPATH=$PWD .venv/bin/python ...):
    .venv/bin/python scripts/agents/exactness_carry.py [--variants | --signal trend]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.portfolio.metrics import carry_metrics
from vibe_quant.portfolio.parity import compare_runs
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.runner import run_carry_engine
from vibe_quant.portfolio.spec import CarrySpec
from vibe_quant.portfolio.ticks import load_carry_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.metrics import CarryReport
    from vibe_quant.portfolio.sizing import Decision

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
START = "2022-06-01"
END = "2026-09-28"

# Date pins (R20, MEASURED): 2022-06-01 00:00 UTC .. 2026-09-28 00:00 UTC.
EXPECTED_START_NS = 1654041600000000000
EXPECTED_END_NS = 1790553600000000000
EXPECTED_DECISIONS = 225

# Headline numbers of the engine run, recorded verbatim from the first passing
# run (2026-10-15; MEASURED runtime 4.76 s real, and engine == reference with 0
# mismatches over 413 fills, 9,487 payments, 1,580 daily and 225 weekly returns).
RECORDED: dict[str, object] = {
    "weekly_sharpe": 0.74651476705443,
    "total_return": 1.2884394452088221,
    "max_drawdown_daily": 0.3171670714370357,
    "n_fills": 413,
    "n_decisions": EXPECTED_DECISIONS,
    "funding_paid_total": -5670.5484016181945,
    "fees_total": 1743.22191953,
    "funding_fallback_settlements": 0,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data/catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data/archive/raw_data.db"

# --signal trend (R67): the pinned long_short book and its truncation date.
TREND_KW: dict[str, object] = {
    "signal": "trend",
    "schedule": "daily",
    "weighting": "time_series",
    "direction": "long_short",
    "vol_target": 0.40,
    "vol_lookback_days": 30,
    "gross": 1.0,
    "band": 0.10,
    "spread": 0.0001,
    "initial_balance": 100_000.0,
}
TREND_PREFIX_END = "2025-06-01"
EXPECTED_TREND_DECISIONS = 1580

# Headline numbers of the trend engine run, recorded verbatim from the first passing
# run (2026-11-23, T6; MEASURED 15.82 s real, engine == reference with 0 mismatches
# over 4,547 fills, 19,034 payments and 1,580 daily returns; prefix to 2025-06-01 equal).
TREND_RECORDED: dict[str, object] = {
    "daily_sharpe_365": 0.7445681367535776,
    "total_return": 1.545948380014671,
    "max_drawdown_daily": 0.3916003681109057,
    "n_fills": 4547,
    "n_decisions": EXPECTED_TREND_DECISIONS,
    "fees_total": 34570.4587464,
    "funding_paid_total": 17507.981677670963,
    "spread_total": 3457.045874462,
    "funding_fallback_settlements": 3,
    "band_skips": 2330,
    "days_gross_above_cap": 245,
}

# Proof (b)'s threshold, USDT/day. Likely removes BNB (median daily quote volume
# 4.079e+08, design x0axf.14 §1) on decisions where the control selects it.
MIN_QUOTE_VOLUME = 5e8


def _exact() -> int:
    spec = CarrySpec.from_dates(SYMBOLS, START, END)
    inputs = load_carry_inputs(spec, CATALOG_PATH, ARCHIVE_DB_PATH)
    engine = run_carry_engine(inputs)
    reference = simulate_reference(inputs)
    report = carry_metrics(engine, inputs)
    mismatches = compare_runs(engine, reference, report, carry_metrics(reference, inputs))

    print(
        f"parity: engine==reference, {len(mismatches)} mismatches over "
        f"{len(engine.fills)} fills, {len(engine.payments)} payments, "
        f"{len(report.daily)} daily and {len(report.weekly)} weekly returns"
    )
    for message in mismatches[:10]:
        print(f"  {message}")

    checks: dict[str, tuple[object, object]] = {
        "spec.start_ns": (spec.start_ns, EXPECTED_START_NS),
        "spec.end_ns": (spec.end_ns, EXPECTED_END_NS),
        "decisions": (len(inputs.decisions), EXPECTED_DECISIONS),
        "compare_runs": (mismatches, []),
        **{
            f"summary.{name}": (report.summary[name], expected)
            for name, expected in RECORDED.items()
        },
    }
    ok = all(actual == expected for actual, expected in checks.values())
    print(f"carry exactness: {'IDENTICAL' if ok else 'DRIFT'}")
    for name, (actual, expected) in checks.items():
        note = "" if actual == expected else f"  (expected {expected!r})"
        print(f"  {name} = {actual!r}{note}")
    return 0 if ok else 1


def _parity(spec: CarrySpec) -> tuple[list[str], CarryRun, CarryRun, CarryReport, CarryReport]:
    """Both executors and both reports for ``spec``, plus their ``compare_runs`` result."""
    inputs = load_carry_inputs(spec, CATALOG_PATH, ARCHIVE_DB_PATH)
    engine = run_carry_engine(inputs)
    reference = simulate_reference(inputs)
    engine_report = carry_metrics(engine, inputs)
    reference_report = carry_metrics(reference, inputs)
    mismatches = compare_runs(engine, reference, engine_report, reference_report)
    return mismatches, engine, reference, engine_report, reference_report


def _control(spec: CarrySpec) -> tuple[CarryRun, CarryReport]:
    """The matched control: the reference run and report of ``spec``."""
    inputs = load_carry_inputs(spec, CATALOG_PATH, ARCHIVE_DB_PATH)
    run = simulate_reference(inputs)
    return run, carry_metrics(run, inputs)


def _sides(decision: Decision) -> tuple[frozenset[str], frozenset[str]]:
    longs = frozenset(s for s, w in decision.weights.items() if w > 0.0)
    shorts = frozenset(s for s, w in decision.weights.items() if w < 0.0)
    return longs, shorts


def reweighted(variant: CarryRun, control: CarryRun) -> list[int]:
    """Decisions (by ts) with 2 legs a side, the control's selected sets, a changed
    absolute weight within a side, and a vol for every selected leg."""
    by_ts = {decision.ts: decision for decision in control.decisions}
    hits: list[int] = []
    for decision in variant.decisions:
        other = by_ts.get(decision.ts)
        longs, shorts = _sides(decision)
        if other is None or len(longs) != 2 or len(shorts) != 2 or _sides(other) != (longs, shorts):
            continue
        changed = any(
            len({abs(decision.weights[s]) for s in side}) > 1
            and any(abs(decision.weights[s]) != abs(other.weights[s]) for s in side)
            for side in (longs, shorts)
        )
        if changed and decision.vols is not None and (longs | shorts) <= set(decision.vols):
            hits.append(decision.ts)
    return hits


def illiquid(runs: Sequence[CarryRun], control: CarryRun) -> list[int]:
    """Decisions (by ts) where a symbol the control selects is priced, ``illiquid``,
    signal None and target 0 in every one of ``runs``."""
    by_ts = {decision.ts: decision for decision in control.decisions}
    hits: list[int] = []
    for decisions in zip(*(run.decisions for run in runs), strict=True):
        ts = decisions[0].ts
        other = by_ts.get(ts)
        if other is None or any(decision.ts != ts for decision in decisions):
            continue
        if any(
            all(
                symbol in decision.prices
                and decision.excluded.get(symbol) == "illiquid"
                and decision.signals.get(symbol) is None
                and decision.target_steps.get(symbol) == 0
                for decision in decisions
            )
            for symbol, weight in other.weights.items()
            if weight != 0.0
        ):
            hits.append(ts)
    return hits


def _variants() -> int:
    """Proofs (a)-(c): one line per variant; 0 only when all three hold."""
    base = CarrySpec.from_dates(SYMBOLS, START, END)
    control, control_report = _control(base)
    results: list[tuple[str, int, bool, str]] = []

    mismatches, _engine, reference, _er, _rr = _parity(
        CarrySpec.from_dates(SYMBOLS, START, END, legs_divisor=2, weighting="inverse_vol")
    )
    equal, _ = _control(CarrySpec.from_dates(SYMBOLS, START, END, legs_divisor=2))
    hits = reweighted(reference, equal)
    results.append(("inverse_vol", len(mismatches), bool(hits), f"reweighted={len(hits)}"))

    mismatches, engine, reference, _er, _rr = _parity(
        CarrySpec.from_dates(SYMBOLS, START, END, min_quote_volume=MIN_QUOTE_VOLUME)
    )
    hits = illiquid((engine, reference), control)
    results.append(("liquidity", len(mismatches), bool(hits), f"illiquid={len(hits)}"))

    mismatches, _engine, _reference, engine_report, reference_report = _parity(
        CarrySpec.from_dates(SYMBOLS, START, END, liquidation_mode="cross")
    )
    notes = engine_report.summary["liquidation"]
    ok = (
        isinstance(notes, dict)
        and notes.get("mode") == "cross"
        and notes == reference_report.summary["liquidation"]
        and control_report.summary["liquidation"] == "not_modelled"
    )
    count = notes.get("count") if isinstance(notes, dict) else notes
    results.append(("cross", len(mismatches), ok, f"liquidations={count}"))

    for name, count_mismatches, feature, detail in results:
        print(
            f"variant {name}: mismatches={count_mismatches} "
            f"feature={'ok' if feature else 'FAIL'} {detail}"
        )
    passed = all(count_mismatches == 0 and feature for _, count_mismatches, feature, _ in results)
    print(f"carry variants: {'OK' if passed else 'FAIL'}")
    return 0 if passed else 1


def trend_spec(end: str = END, fill_granularity: str = "1m") -> CarrySpec:
    """The R67 trend book ending ``end`` on ``fill_granularity``."""
    return CarrySpec.from_dates(SYMBOLS, START, end, fill_granularity=fill_granularity, **TREND_KW)


def prefix_mismatches(short: CarryRun, long: CarryRun, cutoff_ns: int) -> list[str]:
    """How the run ending at ``cutoff_ns`` differs from ``long`` before it (R67 truncation).

    Its decisions are ``long``'s first ones and all ``<= cutoff_ns``; its fills and
    payments are exactly ``long``'s with ``ts < cutoff_ns``.
    """
    problems: list[str] = []
    count = len(short.decisions)
    if any(d.ts > cutoff_ns for d in short.decisions):
        problems.append("decision after the cutoff")
    if short.decisions != long.decisions[:count]:
        problems.append("decisions are not long's prefix")
    before = sum(1 for d in long.decisions if d.ts <= cutoff_ns)
    if count != before:
        problems.append(f"decisions: {count} != long's {before} up to the cutoff")
    for name in ("fills", "payments"):
        mine = tuple(getattr(short, name))
        theirs = tuple(e for e in getattr(long, name) if e.ts < cutoff_ns)
        if mine != theirs:
            problems.append(f"{name}: {len(mine)} != long's {len(theirs)} before the cutoff")
    return problems


def has_hour_rows(archive: Path, symbols: Sequence[str]) -> bool:
    """True when the archive holds 1h klines for every symbol (R68)."""
    marks = ",".join("?" * len(symbols))
    with sqlite3.connect(f"file:{archive}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT symbol FROM raw_klines WHERE interval = '1h' AND symbol IN ({marks})",
            tuple(symbols),
        ).fetchall()
    conn.close()
    return {row[0] for row in rows} == set(symbols)


def _trend() -> int:
    """R67 at 1m (parity, pins, truncation prefix) and R68 at 1h (parity or SKIP)."""
    spec = trend_spec()
    mismatches, engine, _reference, report, _rr = _parity(spec)
    print(
        f"trend parity: engine==reference, {len(mismatches)} mismatches over "
        f"{len(engine.fills)} fills, {len(engine.payments)} payments, "
        f"{len(report.daily)} daily returns"
    )
    for message in mismatches[:10]:
        print(f"  {message}")
    short_spec = trend_spec(TREND_PREFIX_END)
    short = run_carry_engine(load_carry_inputs(short_spec, CATALOG_PATH, ARCHIVE_DB_PATH))
    prefix = prefix_mismatches(short, engine, short_spec.end_ns)
    print(
        f"trend prefix (end {TREND_PREFIX_END}): {len(short.decisions)} decisions, "
        f"{prefix or 'equal'}"
    )

    checks: dict[str, tuple[object, object]] = {
        "spec.start_ns": (spec.start_ns, EXPECTED_START_NS),
        "spec.end_ns": (spec.end_ns, EXPECTED_END_NS),
        "decisions": (len(engine.decisions), EXPECTED_TREND_DECISIONS),
        "compare_runs": (mismatches, []),
        "prefix": (prefix, []),
        **{
            f"summary.{name}": (report.summary.get(name), expected)
            for name, expected in TREND_RECORDED.items()
        },
    }
    ok = all(actual == expected for actual, expected in checks.values())
    print(f"carry trend exactness: {'IDENTICAL' if ok else 'DRIFT'}")
    for name, (actual, expected) in checks.items():
        note = "" if actual == expected else f"  (expected {expected!r})"
        print(f"  {name} = {actual!r}{note}")

    if not has_hour_rows(ARCHIVE_DB_PATH, SYMBOLS):
        print("SKIP 1h: no archive 1h rows")
        return 0 if ok else 1
    hour, hour_engine, _hr, _her, _hrr = _parity(trend_spec(fill_granularity="1h"))
    print(
        f"trend 1h parity: {len(hour)} mismatches over {len(hour_engine.fills)} fills, "
        f"{len(hour_engine.payments)} payments (research only)"
    )
    for message in hour[:10]:
        print(f"  {message}")
    return 0 if ok and not hour else 1


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description="Carry exactness proof (engine == reference)")
    parser.add_argument(
        "--variants",
        action="store_true",
        help="prove the inverse_vol / liquidity / cross variants against matched controls",
    )
    parser.add_argument(
        "--signal",
        choices=("funding", "trend"),
        default="funding",
        help="trend: the R67 1m proof + truncation prefix, and R68 at 1h when archived",
    )
    args = parser.parse_args(list(argv))
    if args.signal == "trend":
        if args.variants:
            parser.error("--variants is funding only")
        return _trend()
    return _variants() if args.variants else _exact()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
