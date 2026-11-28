"""The bankroll report coordinator (design 2gyk1.3 T3c: C10a report half, C17 serial /
parallel, C20-C25).

Fixture-variant tests drive :func:`run_report` in-process on a temp state DB with a
scripted ``run_variant`` and a thread pool; the slow tests run the real CLI in a fresh
process on clones of the real BTCUSDT catalog bars.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.bankroll import report as rep
from vibe_quant.bankroll.report import (
    REPLAY_LABEL,
    CellRun,
    dsl_hasher,
    headline_mismatch,
    run_report,
)
from vibe_quant.bankroll.snapshot import db_scratch, report_root
from vibe_quant.db import run_lifecycle
from vibe_quant.db.connection import get_connection
from vibe_quant.db.state_manager import BankrollAdmissionError, StateManager
from vibe_quant.jobs.manager import process_start_time
from vibe_quant.metrics import DailyReturns
from vibe_quant.validation import variant as variant_mod
from vibe_quant.validation.results import MtmProbe, ValidationResult
from vibe_quant.validation.variant import VariantInput, VariantInputsMismatch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

REPO = Path(__file__).resolve().parents[2]

NAME = "bankroll_probe_rsi"
RSI_DSL: dict[str, object] = {
    "name": NAME,
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"], "short": ["rsi > 60"]},
    "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 40"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}
PARENT_PARAMS: dict[str, object] = {
    "risk_per_trade": 0.02,
    "max_position_pct": 0.5,
    "liquidation_mode": "isolated",
    "leverage": 10,
}
HEADLINE: dict[str, Any] = {
    "total_return": 0.25,
    "max_drawdown": 0.2,
    "sharpe_ratio": 1.5,
    "total_trades": 40,
    "total_fees": 12.5,
    "total_funding": -3.25,
}
START_NS = 1_740_787_200 * 10**9  # 2025-03-01T00:00Z
DAY_NS = 86_400 * 10**9
VENUE = json.dumps({"starting_balance_usdt": 1000})
SWEEP: dict[str, object] = {"margin_sweep": True}
GRID: list[tuple[str, float, float]] = [
    (m, lev, p) for m in ("isolated", "cross") for lev in (3, 5, 10) for p in (0.5, 1.0)
]
# 80 days alternating +1.1% / -0.9%: mean 0.001, f_half ~ 5 -> the user cap (1.0) binds.
DAILY = tuple(0.011 if i % 2 == 0 else -0.009 for i in range(80))


def alive(pid: int, identity: str | None) -> bool:
    return True


def never_signal(job_id: int) -> None:
    raise AssertionError(f"job {job_id} signalled")


# --- scripted engine ------------------------------------------------------------------


def result_of(
    *,
    total_return: float = HEADLINE["total_return"],
    max_drawdown: float = HEADLINE["max_drawdown"],
    sharpe: float = HEADLINE["sharpe_ratio"],
    trades: int = HEADLINE["total_trades"],
    fees: float = HEADLINE["total_fees"],
    funding: float = HEADLINE["total_funding"],
    dd: float | None = 0.2,
    daily: tuple[float, ...] | None = DAILY,
    force_stopped: bool = False,
    liquidation: dict[str, object] | None = None,
    bust_ts: int | None = None,
) -> ValidationResult:
    res = ValidationResult(
        total_return=total_return, max_drawdown=max_drawdown, sharpe_ratio=sharpe,
        total_trades=trades, total_fees=fees, total_funding=funding,
    )  # fmt: skip
    res.engine_force_stopped = force_stopped
    res.mtm_probe = MtmProbe(
        paths="ok",
        daily=None if daily is None else DailyReturns(START_NS // DAY_NS, array("d", daily)),
        dd=dd,
        bust_ts=bust_ts,
        path_divergence=False,
        ledger="ordinary",
    )
    res.notes["liquidation"] = liquidation or {"mode": "isolated", "count": 0, "events": []}
    return res


def k_of(overrides: Mapping[str, object]) -> float:
    """The grid multiple a cell's overrides encode (baseline ``{}`` is 1.0)."""
    if not overrides:
        return 1.0
    return float(overrides["max_position_pct"]) / 0.5  # type: ignore[arg-type]


def default_cell(k: float, overrides: Mapping[str, object]) -> ValidationResult:
    """k=0.5: DD 0.1; baseline/k=1: the parent; k=2: DD 0.35 (over the 0.30 limit)."""
    if k == 1.0:
        return result_of()
    return result_of(
        total_return=0.12 * k, max_drawdown=0.175 * k, fees=6.0 * k * k,
        funding=-0.7 * k * k, dd=0.1 if k < 1 else 0.35,
    )  # fmt: skip


class Engine:
    """``run_variant`` stand-in: records each call, answers through ``script``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: Callable[[float, Mapping[str, object]], ValidationResult] = default_cell

    def __call__(
        self, inp: VariantInput, overrides: Mapping[str, object], *, catalog_path: Path,
        archive_path: Path, mtm_probe: bool,
    ) -> ValidationResult:  # fmt: skip
        k = k_of(overrides)
        self.calls.append({"k": k, "inp": inp, "overrides": dict(overrides),
                           "catalog": catalog_path, "archive": archive_path,
                           "mtm_probe": mtm_probe,
                           "catalog_files": sorted(p.name for p in catalog_path.rglob("*"))})  # fmt: skip
        return self.script(k, overrides)

    @property
    def ks(self) -> list[float]:
        return [c["k"] for c in self.calls]


# --- world ------------------------------------------------------------------------------


class World:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp
        self.db = tmp / "state.db"
        self.catalog = tmp / "catalog"
        (self.catalog / "data" / "bar").mkdir(parents=True)
        (self.catalog / "data" / "bar" / "part-0.parquet").write_bytes(b"bars")
        self.archive = tmp / "raw_data.db"
        conn = sqlite3.connect(self.archive)
        conn.execute(
            "CREATE TABLE raw_funding_rates (id INTEGER PRIMARY KEY, symbol TEXT, "
            "funding_time INTEGER, funding_rate REAL)"
        )
        conn.execute("INSERT INTO raw_funding_rates VALUES (1, 'BTCUSDT', 0, 0.0001)")
        conn.commit()
        conn.close()
        self.scratch = tmp / "scratch"
        self.sm = StateManager(self.db)
        self.hasher = dsl_hasher(NAME)
        self.engine = Engine()
        monkeypatch.setattr(variant_mod, "run_variant", self.engine)
        self.workers_seen: list[int] = []
        self.strategy_id: int | None = None

        def thread_pool(workers: int) -> ThreadPoolExecutor:
            self.workers_seen.append(workers)
            return ThreadPoolExecutor(max_workers=workers)

        monkeypatch.setattr(rep, "_executor", thread_pool)

    def parent(
        self,
        *,
        params: Mapping[str, object] = PARENT_PARAMS,
        identity: bool = True,
        headline: Mapping[str, Any] = HEADLINE,
        dsl: Mapping[str, object] = RSI_DSL,
        venue: str | None = VENUE,
    ) -> int:
        if self.strategy_id is None:
            self.strategy_id = self.sm.create_strategy(NAME, dict(dsl))
        sid = self.strategy_id
        run_id = self.sm.create_backtest_run(
            sid, "validation", ["BTCUSDT"], "4h", "2025-03-01", "2025-06-01", dict(params)
        )
        notes: dict[str, object] = {}
        if identity:
            recorded: dict[str, object] = {
                "v": 1, "strategy_name": NAME, "dsl_sha256": self.hasher(dict(dsl)),
                "symbols": ["BTCUSDT"], "timeframe": "4h", "start_date": "2025-03-01",
                "end_date": "2025-06-01",
            }  # fmt: skip
            if venue is not None:
                recorded["venue"] = venue
            notes["validation_input"] = recorded
        self.sm.save_backtest_result(run_id, {**headline, "notes": json.dumps(notes)})
        self.sm.update_backtest_run_status(run_id, "completed")
        return run_id

    def admit(self, run_id: int, request: Mapping[str, object] | None = None) -> tuple[int, int]:
        req = {"multiples": [0.5, 1.0, 2.0], "workers": 1, **(request or {})}
        report_id, job_id = self.sm.admit_bankroll_report(
            run_id, req, dsl_hash=self.hasher, job_alive=alive, terminate=never_signal
        )
        conn = get_connection(self.db)
        try:
            run_lifecycle.open_attempt(conn, job_id, "bankroll_report", None, exists=False)
            conn.commit()
        finally:
            conn.close()
        return report_id, job_id

    def run(self, report_id: int, job_id: int) -> int:
        return run_report(
            report_id, job_id, self.db, catalog=self.catalog, archive=self.archive,
            scratch=self.scratch,
        )  # fmt: skip

    def launch(
        self, request: Mapping[str, object] | None = None, **parent: Any
    ) -> tuple[int, int, int, int]:
        """Parent + admission + run; returns (run_id, report_id, job_id, exit code)."""
        run_id = self.parent(**parent)
        report_id, job_id = self.admit(run_id, request)
        return run_id, report_id, job_id, self.run(report_id, job_id)

    def root(self, report_id: int) -> Path:
        """Where this DB's report keeps its scratch (namespaced by the state DB)."""
        return report_root(db_scratch(self.scratch, self.db), report_id)

    def report(self, report_id: int) -> dict[str, Any]:
        row = self.sm.get_bankroll_report(report_id)
        assert row is not None
        return row

    def job(self, job_id: int) -> sqlite3.Row:
        conn = get_connection(self.db)
        try:
            row: sqlite3.Row = conn.execute(
                "SELECT * FROM background_jobs WHERE run_id = ?", (job_id,)
            ).fetchone()
            return row
        finally:
            conn.close()

    def write(self, sql: str, args: tuple[object, ...]) -> None:
        conn = get_connection(self.db)
        try:
            conn.execute(sql, args)
            conn.commit()
        finally:
            conn.close()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    handler = signal.getsignal(signal.SIGTERM)
    w = World(tmp_path, monkeypatch)
    yield w
    w.sm.close()
    assert signal.getsignal(signal.SIGTERM) == handler


def rows_by_k(report: Mapping[str, Any]) -> dict[float, dict[str, Any]]:
    return {r["k"]: r for r in report["rows"]}


def is_margin(o: Mapping[str, object], mode: str, lev: float, pct: float) -> bool:
    """``o`` is the margin-sweep cell (mode, leverage, margin limit)."""
    return "leverage" in o and (o["liquidation_mode"], o["leverage"], o["max_margin_pct"]) == (
        mode, lev, pct,
    )  # fmt: skip


def margin_cell(report: Mapping[str, Any], mode: str, lev: float, pct: float) -> dict[str, Any]:
    (cell,) = [c for c in report["margin_sweep"]["cells"]
               if (c["mode"], c["leverage"], c["max_margin_pct"]) == (mode, lev, pct)]  # fmt: skip
    return cell  # type: ignore[no-any-return]


# --- C21: identity, then headline (red-first) --------------------------------------------


def test_each_engine_run_gets_a_fresh_process() -> None:
    """NT's force-stop flag and the aux-data configuration are process-global: the cell
    pool never runs a second variant in the same worker process."""
    with rep._executor(1) as pool:
        pids = [pool.submit(os.getpid).result() for _ in range(3)]
    assert len(set(pids)) == 3 and os.getpid() not in pids


def test_stale_parent(world: World) -> None:
    """The baseline's recomputed identity differs from the recorded one: the report ends
    ``stale_parent`` naming the keys, before any stress cell and with no recommendation."""

    def mismatch(k: float, overrides: Mapping[str, object]) -> ValidationResult:
        raise VariantInputsMismatch(("dsl_sha256", "end_date"))

    world.engine.script = mismatch
    _run_id, report_id, job_id, code = world.launch()
    row = world.report(report_id)
    assert code == 1
    assert row["status"] == "stale_parent"
    assert row["error_message"] == (
        "stale_parent: inputs_mismatch: dsl_sha256, end_date: re-validate first"
    )
    assert row["report"] is None
    assert world.engine.ks == [1.0]  # the baseline only
    assert world.engine.calls[0]["overrides"] == {}
    assert world.engine.calls[0]["mtm_probe"] is True
    assert world.job(job_id)["status"] == "failed"
    assert not world.root(report_id).exists()


@pytest.mark.parametrize(
    ("field", "replay", "stored_text"),
    [
        (
            "total_return",
            math.nextafter(0.25, 1.0),
            "total_return stored=0.25 replay=0.25000000000000006",
        ),
        ("max_drawdown", 0.2000001, "max_drawdown stored=0.2 replay=0.2000001"),
        ("sharpe", 1.4, "sharpe_ratio stored=1.5 replay=1.4"),
        ("trades", 41, "total_trades stored=40 replay=41"),
        ("fees", 12.0, "total_fees stored=12.5 replay=12.0"),
        ("funding", -3.0, "total_funding stored=-3.25 replay=-3.0"),
    ],
)
def test_stale_parent_headline(world: World, field: str, replay: object, stored_text: str) -> None:
    """Identity matches but one stored headline differs by any amount (one ulp included):
    ``stale_parent`` with both values; no stress cell runs."""
    world.engine.script = lambda k, o: result_of(**{field: replay})
    _run_id, report_id, _job_id, code = world.launch()
    row = world.report(report_id)
    assert (code, row["status"], row["report"]) == (1, "stale_parent", None)
    assert row["error_message"] == (
        f"stale_parent: headline_mismatch: {stored_text}: re-validate first"
    )
    assert world.engine.ks == [1.0]


def test_headline_equality_edges() -> None:
    stored = dict(HEADLINE, id=1)
    same = CellRun(status="ok", **{k: v for k, v in HEADLINE.items()})
    assert headline_mismatch(stored, same) == []
    # SQLite stores NaN as NULL: NULL vs NaN is equal, NULL vs a number is not.
    assert headline_mismatch({**stored, "sharpe_ratio": None},
                             CellRun(status="ok", **{**HEADLINE, "sharpe_ratio": math.nan})) == []  # fmt: skip
    assert headline_mismatch({**stored, "sharpe_ratio": None}, same) == [
        "sharpe_ratio stored=None replay=1.5"
    ]
    assert headline_mismatch(
        stored, CellRun(status="ok", **{**HEADLINE, "sharpe_ratio": math.nan})
    ) == ["sharpe_ratio stored=1.5 replay=nan"]


# --- happy path, C25 funding from each run, C17 scratch ------------------------------


def test_completed_report(world: World) -> None:
    abandoned = world.root(999)
    (abandoned / "w0").mkdir(parents=True)
    _run_id, report_id, job_id, code = world.launch()
    assert code == 0
    row = world.report(report_id)
    assert row["status"] == "completed" and row["error_message"] is None
    report = row["report"]
    json.dumps(report, allow_nan=False)
    assert report["label"] == REPLAY_LABEL
    assert report["mode"] == "isolated"
    assert world.engine.ks == [1.0, 0.5, 2.0]  # baseline, then cells; k=1 is the baseline
    rows = rows_by_k(report)
    assert [r["k"] for r in report["rows"]] == [0.5, 1.0, 2.0]
    assert rows[1.0]["role"] == "baseline" and rows[0.5]["role"] == "cell"
    assert rows[0.5]["effective_cap"] == 0.25 and rows[2.0]["effective_cap"] == 1.0
    assert rows[2.0]["effective_risk"] == 0.04 and rows[2.0]["risk_held_at_limit"] is False
    assert rows[0.5]["eligible"] is True and rows[1.0]["eligible"] is True
    assert rows[2.0]["ineligible_reason"] == "dd_over_limit"
    assert report["kelly"]["status"] == "ok" and report["kelly"]["actionable_reason"] is None
    assert report["kelly"]["n"] == 80
    assert report["recommendation"] == {
        "k": 1.0, "ceiling": 1.0, "reason": None, "binding": "user_cap",
        "text": "Your size cap binds",
    }  # fmt: skip
    assert report["baseline"]["total_return"] == 0.25
    assert report["baseline"]["liquidation_count"] == 0
    assert world.job(job_id)["status"] == "completed"
    assert not world.root(report_id).exists()
    assert not abandoned.exists()


def test_funding_from_run(world: World) -> None:
    """C25: each row's fees and funding are its own run's (here k^2-shaped, so any
    k-scaling of the baseline would differ); the bankroll code never multiplies them."""
    _run_id, report_id, _job_id, code = world.launch({"multiples": [0.5, 1.0, 2.0, 3.0]})
    assert code == 0
    rows = rows_by_k(world.report(report_id)["report"])
    assert rows[2.0]["total_funding"] == -2.8 and rows[2.0]["total_fees"] == 24.0
    assert rows[3.0]["total_funding"] == -6.299999999999999 and rows[3.0]["total_fees"] == 54.0
    assert rows[0.5]["total_funding"] == -0.175
    assert rows[1.0]["total_funding"] == -3.25  # the baseline's own
    assert abs(rows[2.0]["total_funding"]) > abs(rows[0.5]["total_funding"])
    hits = [
        f"{p.name}:{n}"
        for p in sorted((REPO / "vibe_quant" / "bankroll").glob("*.py"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"(fees|funding)[^=]*\*", line)
    ]
    assert hits == []


def test_engines_get_private_clones_of_one_seed(world: World) -> None:
    _run_id, report_id, _job_id, code = world.launch()
    assert code == 0
    root = world.root(report_id)
    catalogs = [c["catalog"] for c in world.engine.calls]
    assert catalogs == [root / "baseline" / "catalog", root / "w0" / "catalog",
                        root / "w1" / "catalog"]  # fmt: skip
    assert {c["archive"] for c in world.engine.calls} == {root / "archive.db"}
    assert all(c["catalog_files"] == ["bar", "data", "part-0.parquet"]
               for c in world.engine.calls)  # fmt: skip


def test_serial_parallel_equal(world: World) -> None:
    """C17 (coordinator): one worker and three workers publish identical rows."""
    reports = []
    for workers in (1, 3):
        _r, report_id, _j, code = world.launch(
            {"workers": workers, "multiples": [0.25, 0.5, 1.0, 2.0, 3.0]}
        )
        assert code == 0
        reports.append(world.report(report_id)["report"])
    assert world.workers_seen == [1, 3]
    assert reports[0]["rows"] == reports[1]["rows"]
    assert reports[0]["recommendation"] == reports[1]["recommendation"]


# --- C10a / C23: force stop and failures ---------------------------------------------


def test_force_stopped_baseline(world: World) -> None:
    world.engine.script = lambda k, o: result_of(force_stopped=True)
    _run_id, report_id, job_id, code = world.launch()
    row = world.report(report_id)
    assert (code, row["status"], row["error_message"]) == (1, "failed", "engine_force_stopped")
    assert row["report"] is None
    assert world.engine.ks == [1.0]
    assert world.job(job_id)["error_message"] == "engine_force_stopped"


def test_force_stopped_cell_is_a_failed_row(world: World) -> None:
    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        return result_of(dd=0.05, force_stopped=k == 0.5)

    world.engine.script = script
    _run_id, report_id, _job_id, code = world.launch({"multiples": [0.5, 1.0]})
    assert code == 0
    row = rows_by_k(world.report(report_id)["report"])[0.5]
    assert row["status"] == "failed" and row["error"] == "engine_force_stopped"
    assert row["engine_force_stopped"] is True
    assert row["eligible"] is False and row["ineligible_reason"] == "engine_force_stopped"


def test_cell_failure(world: World) -> None:
    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        if k == 2.0:
            raise RuntimeError("engine exploded")
        return default_cell(k, o)

    world.engine.script = script
    _run_id, report_id, _job_id, code = world.launch()
    row = world.report(report_id)
    assert (code, row["status"]) == (0, "completed")
    cell = rows_by_k(row["report"])[2.0]
    assert cell["status"] == "failed" and cell["error"] == "RuntimeError: engine exploded"
    assert cell["eligible"] is False and "total_return" not in cell
    assert row["report"]["recommendation"]["k"] == 1.0


def test_baseline_failure_fails_the_report(world: World) -> None:
    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        raise ValueError("no bars")

    world.engine.script = script
    _run_id, report_id, job_id, code = world.launch()
    row = world.report(report_id)
    assert (code, row["status"]) == (1, "failed")
    assert row["error_message"] == "baseline failed: ValueError: no bars"
    assert world.job(job_id)["status"] == "failed"
    assert not world.root(report_id).exists()


def test_not_runnable_cell_is_reported_and_others_run(world: World) -> None:
    """Parent cap 4.0 at L5: k=2 asks cap 8 > leverage, refused on its own."""
    params = {**PARENT_PARAMS, "max_position_pct": 4.0, "leverage": 5}

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        return result_of(dd=0.1)

    world.engine.script = script
    _run_id, report_id, _job_id, code = world.launch(params=params)
    assert code == 0
    rows = rows_by_k(world.report(report_id)["report"])
    assert rows[2.0]["status"] == "not_runnable"
    assert rows[2.0]["error"] == "max_position_pct (8.0) must not exceed leverage (5.0)"
    caps = [c["overrides"].get("max_position_pct") for c in world.engine.calls]
    assert caps == [None, 2.0]  # the baseline, then k=0.5 only


# --- C24: liquidation witness and the k=1 control --------------------------------------


def test_liquidation_witness(world: World) -> None:
    """Isolated parent with hits: the k=1 row IS the baseline (ineligible ``liquidated``,
    Kelly not actionable). In cross mode k=1 is a labelled control with its own count."""
    hits = {"mode": "isolated", "count": 2,
            "events": [{"ts": START_NS + 3 * DAY_NS}, {"ts": START_NS + 9 * DAY_NS}]}  # fmt: skip

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        if o.get("liquidation_mode") == "cross":
            return result_of(total_return=0.2, dd=0.25,
                             liquidation={"mode": "cross", "count": 0, "events": []})  # fmt: skip
        return result_of(liquidation=hits)

    world.engine.script = script
    _run_id, report_id, _job_id, code = world.launch({"multiples": [1.0]})
    assert code == 0
    report = world.report(report_id)["report"]
    k1 = rows_by_k(report)[1.0]
    assert k1["role"] == "baseline" and k1["liquidation_count"] == 2
    assert k1["ineligible_reason"] == "liquidated"
    assert k1["time_to_first_liquidation_s"] == 3 * 86_400.0 and k1["time_to_ruin_s"] is None
    assert report["kelly"]["actionable_reason"] == "liquidated"
    assert report["recommendation"]["k"] is None
    assert world.engine.ks == [1.0]

    world.engine.calls.clear()
    _run_id, report_id, _job_id, code = world.launch(
        {"multiples": [1.0], "liquidation_mode": "cross"}
    )
    assert code == 0
    report = world.report(report_id)["report"]
    control = rows_by_k(report)[1.0]
    assert control["role"] == "control" and control["mode"] == "cross"
    assert control["liquidation_count"] == 0 and control["total_return"] == 0.2
    assert report["baseline"]["liquidation_count"] == 2
    assert report["baseline"]["total_return"] == 0.25
    assert "not compared to the parent" in report["disclosures"]["k1_control"]
    assert world.engine.ks == [1.0, 1.0]
    assert world.engine.calls[1]["overrides"]["liquidation_mode"] == "cross"


def test_off_parent_runs_cross_control(world: World) -> None:
    params = {k: v for k, v in PARENT_PARAMS.items() if k != "liquidation_mode"}
    world.engine.script = lambda k, o: result_of(dd=0.1)
    _run_id, report_id, _job_id, code = world.launch({"multiples": [1.0]}, params=params)
    assert code == 0
    report = world.report(report_id)["report"]
    assert report["mode"] == "cross" and report["parent"]["mode"] == "off"
    assert rows_by_k(report)[1.0]["role"] == "control"
    assert world.engine.calls[1]["overrides"]["liquidation_mode"] == "cross"


# --- C20: the job reads only the admission snapshot ------------------------------------


def test_snapshot_only(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_strategy_reads(*a: object, **k: object) -> None:
        raise AssertionError("the job read the strategies table")

    for name in ("get_strategy", "get_strategy_by_name", "list_strategies"):
        if hasattr(StateManager, name):
            monkeypatch.setattr(StateManager, name, no_strategy_reads)
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)
    inputs = world.report(report_id)["inputs"]
    assert world.run(report_id, job_id) == 0
    inp = world.engine.calls[0]["inp"]
    assert inp.dsl_json == inputs["dsl_json"]
    assert inp.parent_identity == inputs["parent_identity"]
    run_config = json.loads(inp.run_config_json)
    assert run_config["id"] == run_id
    assert run_config["symbols"] == ["BTCUSDT"] and run_config["parameters"] == PARENT_PARAMS


def test_admission_refuses_unverified_and_edited_parents(world: World) -> None:
    unverified = world.parent(identity=False)
    with pytest.raises(BankrollAdmissionError) as err:
        world.admit(unverified)
    assert err.value.code == "inputs_unverified" and "re-validate first" in str(err.value)

    # Score-equivalent edit (an indicator neither entry nor exit uses) is still stale.
    run_id = world.parent()
    sid = world.sm.get_backtest_run(run_id)["strategy_id"]  # type: ignore[index]
    edited = {**RSI_DSL, "indicators": {"rsi": {"type": "RSI", "period": 14},
                                        "unused": {"type": "RSI", "period": 7}}}  # fmt: skip
    world.write("UPDATE strategies SET dsl_config = ? WHERE id = ?", (json.dumps(edited), sid))
    with pytest.raises(BankrollAdmissionError) as err:
        world.admit(run_id)
    assert err.value.code == "stale_parent" and "re-validate first" in str(err.value)


def test_dsl_hasher() -> None:
    a, b = dsl_hasher(NAME), dsl_hasher("other")
    h1 = a(dict(RSI_DSL))
    h2 = a({**RSI_DSL, "indicators": {"rsi": {"type": "RSI", "period": 21}}})
    assert isinstance(h1, str) and len(h1) == 64 and h1 != h2
    assert a(dict(RSI_DSL)) == h1 and b(dict(RSI_DSL)) == h1  # native DSL: name unused
    assert a({"timeframe": "4h"}) is None and a("not a dict") is None


TAKER_NATIVE: dict[str, object] = {
    **RSI_DSL,
    "indicators": {"tk": {"type": "taker_imbalance", "period": 10}},
    "entry_conditions": {"long": ["tk > 0.1"]},
    "exit_conditions": {"long": ["tk < 0"]},
}
TAKER_FRONTEND: dict[str, object] = {
    "general": {"timeframe": "4h"},
    "indicators": [{"type": "taker_imbalance", "params": {"period": 10}}],
    "conditions": {"long_entry": ["taker_imbalance_10 > 0.1"],
                   "long_exit": ["taker_imbalance_10 < 0"]},
}  # fmt: skip


@pytest.mark.parametrize(
    ("dsl", "symbols", "expected"),
    [
        ({**TAKER_NATIVE, "indicators": {"tk": {"type": "TAKER_IMBALANCE", "period": 10}}},
         ["BTCUSDT", "ETHUSDT"], ("BTCUSDT", "ETHUSDT")),
        # The schema upper-cases a stored lowercase type; the API stores the raw body.
        (TAKER_NATIVE, ["SOLUSDT"], ("SOLUSDT",)),
        (TAKER_FRONTEND, ["BTCUSDT", "BNBUSDT"], ("BTCUSDT", "BNBUSDT")),
        (RSI_DSL, ["BTCUSDT"], ()),
    ],
)  # fmt: skip
def test_taker_symbols(
    dsl: Mapping[str, object], symbols: list[str], expected: tuple[str, ...]
) -> None:
    """The kline symbols come from the VALIDATED DSL (the one the engine runs)."""
    inputs = {"dsl_json": json.dumps(dsl), "strategy_name": "tk"}
    assert rep._taker_symbols(inputs, symbols) == expected


OI_NATIVE: dict[str, object] = {
    **RSI_DSL,
    "indicators": {"oi": {"type": "oi_change", "period": 12}},
    "entry_conditions": {"long": ["oi < -0.002"]},
    "exit_conditions": {"long": ["oi > 0"]},
}
POSITIONING_FRONTEND: dict[str, object] = {
    "general": {"timeframe": "4h"},
    "indicators": [{"type": "dvol_z", "params": {"period": 168}},
                   {"type": "ls_ratio_z", "params": {"period": 288}},
                   {"type": "rsi", "params": {"period": 14}}],
    "conditions": {"long_entry": ["dvol_z_168 > 1", "ls_ratio_z_288 < -1"],
                   "long_exit": ["rsi_14 > 60"]},
}  # fmt: skip


@pytest.mark.parametrize(
    ("dsl", "names"),
    [(OI_NATIVE, "OI_CHANGE"), (POSITIONING_FRONTEND, "DVOL_Z, LS_RATIO_Z")],
)
def test_positioning_strategy_refused_before_snapshot(
    world: World, monkeypatch: pytest.MonkeyPatch, dsl: Mapping[str, object], names: str
) -> None:
    """R36: the snapshot has no metrics/DVOL rows, so the report refuses before copying
    anything or running a cell (the RSI parent is covered by ``test_completed_report``)."""
    captured: list[object] = []
    monkeypatch.setattr(rep, "capture_market", lambda *a, **k: captured.append(a))
    _run_id, report_id, job_id, code = world.launch(SWEEP, dsl=dsl)
    row = world.report(report_id)
    assert (code, row["status"]) == (1, "failed")
    assert row["error_message"] == (
        f"bankroll report does not support positioning indicators ({names}): "
        "its market snapshot has no metrics or DVOL rows"
    )
    assert captured == [] and world.engine.calls == []
    assert world.job(job_id)["status"] == "failed"


# --- C22: freshness at publish ---------------------------------------------------------


def _edit_dsl(world: World, run_id: int) -> None:
    sid = world.sm.get_backtest_run(run_id)["strategy_id"]  # type: ignore[index]
    edited = {**RSI_DSL, "stop_loss": {"type": "fixed_pct", "percent": 2.5}}
    world.write("UPDATE strategies SET dsl_config = ? WHERE id = ?", (json.dumps(edited), sid))


def _replace_result(world: World, run_id: int) -> None:
    stored = world.sm.get_backtest_result(run_id)
    assert stored is not None
    world.sm.save_backtest_result(run_id, {**HEADLINE, "notes": stored["notes"]})


def _rerun_in_flight(world: World, run_id: int) -> None:
    world.write("UPDATE backtest_runs SET status = 'running' WHERE id = ?", (run_id,))


def _rerun_failed(world: World, run_id: int) -> None:
    world.write("UPDATE backtest_runs SET status = 'failed' WHERE id = ?", (run_id,))


def _cli_detail_rerun(world: World, run_id: int) -> None:
    """A CLI-only ``--detail-timeframe`` re-run: same row, a new result (new id)."""
    _replace_result(world, run_id)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (_edit_dsl, "dsl_changed"),
        (_replace_result, "result_replaced"),
        (_rerun_in_flight, "parent_not_completed (running)"),
        (_rerun_failed, "parent_not_completed (failed)"),
        (_cli_detail_rerun, "result_replaced"),
    ],
)
@pytest.mark.parametrize("stage", ["size", "margin"])
def test_freshness(
    world: World, change: Callable[[World, int], None], reason: str, stage: str
) -> None:
    """The parent changes after the baseline passed (during a size cell, or during the
    margin cell cross 10x / 100%): the publish sees it inside its transaction ->
    ``stale_parent``; GET says stale and masks the recommendation."""
    run_id = world.parent()

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        hit = is_margin(o, "cross", 10, 1.0) if stage == "margin" else k == 2.0
        if hit:
            change(world, run_id)
        return default_cell(k, o)

    world.engine.script = script
    report_id, job_id = world.admit(run_id, SWEEP if stage == "margin" else None)
    assert world.run(report_id, job_id) == 1
    row = world.report(report_id)
    assert row["status"] == "stale_parent" and row["report"] is None
    assert reason in row["error_message"] and "re-validate first" in row["error_message"]
    latest = world.sm.get_latest_bankroll_report(run_id, dsl_hash=world.hasher)
    assert latest is not None
    assert latest["freshness"] == "stale" and reason in latest["stale_reasons"]
    assert world.engine.ks == [1.0, 0.5, 2.0] + ([1.0] * 12 if stage == "margin" else [])


def test_result_replaced_before_the_run_stops_after_the_baseline(world: World) -> None:
    """A same-headline re-run replaced the parent result after admission: the report
    stops at the baseline (``result_replaced``) and runs no stress cell."""
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)
    _replace_result(world, run_id)
    assert world.run(report_id, job_id) == 1
    row = world.report(report_id)
    assert (row["status"], row["error_message"]) == (
        "stale_parent", "stale_parent: result_replaced: re-validate first",
    )  # fmt: skip
    assert world.engine.ks == [1.0]


@pytest.mark.parametrize("sweep", [False, True])
def test_get_masks_a_report_gone_stale_after_publish(world: World, sweep: bool) -> None:
    """A stale report shows neither the recommended size nor the safest setting; its rows
    and margin cells stay (R25)."""
    run_id, report_id, _job_id, code = world.launch(SWEEP if sweep else None)
    assert code == 0
    fresh = world.sm.get_latest_bankroll_report(run_id, dsl_hash=world.hasher)
    assert fresh is not None and fresh["freshness"] == "fresh"
    assert fresh["report"]["recommendation"]["k"] == 1.0
    safest = {"mode": "cross", "leverage": 3, "max_margin_pct": 0.5,
              "move": {"status": "ok", "value": 1.9880478087649402, "side": "short",
                       "legs": [["BTCUSDT", 500.0]], "reason": None,
                       "path_dependent": False}}  # fmt: skip
    if sweep:
        assert fresh["report"]["margin_sweep"]["safest"] == safest
    else:
        assert fresh["report"]["margin_sweep"] is None
    _edit_dsl(world, run_id)
    stale = world.sm.get_latest_bankroll_report(run_id, dsl_hash=world.hasher)
    assert stale is not None and stale["id"] == report_id
    assert stale["freshness"] == "stale" and stale["report"]["recommendation"] is None
    if sweep:
        assert stale["report"]["margin_sweep"]["safest"] is None
        assert len(stale["report"]["margin_sweep"]["cells"]) == 12
    else:
        assert stale["report"]["margin_sweep"] is None


# --- margin sweep at the recommended size (design arch-bankroll-1 T3: R20-R25) ----------

ISO_L3_MOVE: dict[str, object] = {
    "status": "ok", "value": 0.32802124833997337, "side": "short",
    "legs": [["BTCUSDT", 500.0]], "reason": None, "path_dependent": False,
}  # fmt: skip
CROSS_MOVE: dict[str, object] = {**ISO_L3_MOVE, "value": 1.9880478087649402}


def margin_calls(world: World) -> list[tuple[object, object, object]]:
    return [(c["overrides"]["liquidation_mode"], c["overrides"]["leverage"],
             c["overrides"]["max_margin_pct"])
            for c in world.engine.calls if "leverage" in c["overrides"]]  # fmt: skip


def test_margin_sweep_runs_at_recommended_k(world: World) -> None:
    """R20: 12 margin cells at the recommended k=1 (sizing of the k-row, each with its own
    mode/leverage/limit), each cell's liquidating move at the 1000 USDT starting balance,
    and the safest setting cross 3x / 50% (one coin: cross is not path dependent)."""
    _run_id, report_id, _job_id, code = world.launch(SWEEP)
    assert code == 0
    report = world.report(report_id)["report"]
    json.dumps(report, allow_nan=False)
    assert world.engine.ks == [1.0, 0.5, 2.0] + [1.0] * 12
    assert margin_calls(world) == GRID
    for call, (mode, lev, pct) in zip(world.engine.calls[3:], GRID, strict=True):
        assert call["overrides"] == {
            "max_position_pct": 0.5, "risk_per_trade": 0.02, "liquidation_mode": mode,
            "leverage": lev, "max_margin_pct": pct,
        }  # fmt: skip
    assert report["recommendation"]["k"] == 1.0
    sweep = report["margin_sweep"]
    assert (sweep["status"], sweep["reason"], sweep["recommendation_reason"], sweep["k"]) == (
        "completed", None, None, 1.0,
    )  # fmt: skip
    assert sweep["scenario"] == {
        "starting_balance": 1000.0, "notional": 500.0, "sides": ["long", "short"],
        "symbols": ["BTCUSDT"], "label": rep.SCENARIO_LABEL,
    }  # fmt: skip
    assert [(c["mode"], c["leverage"], c["max_margin_pct"]) for c in sweep["cells"]] == GRID
    iso = margin_cell(report, "isolated", 3, 0.5)
    assert iso["move"] == ISO_L3_MOVE
    assert (iso["positions"], iso["remainder"], iso["reused"]) == (1, 0.0, False)
    assert (iso["k"], iso["role"], iso["status"], iso["error"]) == (1.0, "cell", "ok", None)
    assert (iso["eligible"], iso["ineligible_reason"]) == (True, None)
    assert (iso["total_return"], iso["mtm_dd"], iso["liquidation_count"]) == (0.25, 0.2, 0)
    assert iso["effective_cap"] == 0.5 and iso["effective_risk"] == 0.02
    for mode, lev, pct in GRID[6:]:
        cross = margin_cell(report, mode, lev, pct)
        assert cross["move"] == CROSS_MOVE and cross["mode"] == "cross"
    assert sweep["safest"] == {"mode": "cross", "leverage": 3, "max_margin_pct": 0.5,
                               "move": CROSS_MOVE}  # fmt: skip
    assert sweep["safest_reason"] is None


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        ({"dd": 0.35}, "no_eligible_row"),
        ({"liquidation": {"mode": "isolated", "count": 1, "events": []}}, "liquidated"),
    ],
)
def test_margin_sweep_skipped(world: World, result: dict[str, Any], reason: str) -> None:
    """R21 (user Q1=A): no recommended size -> no margin engine call; the sweep says
    skipped with the recommendation's own reason."""
    world.engine.script = lambda k, o: result_of(**result)
    _run_id, report_id, _job_id, code = world.launch(SWEEP)
    assert code == 0
    report = world.report(report_id)["report"]
    assert report["recommendation"]["k"] is None
    assert report["recommendation"]["reason"] == reason
    assert world.engine.ks == [1.0, 0.5, 2.0]
    assert report["margin_sweep"] == {
        "status": "skipped", "reason": "no_recommended_size", "recommendation_reason": reason,
        "k": None, "scenario": None, "cells": [], "safest": None, "safest_reason": None,
    }  # fmt: skip


def test_legacy_request_no_sweep(world: World) -> None:
    """R22: a stored request without ``margin_sweep`` (and one with it false) replays as
    v1: no margin cell runs and the payload says ``margin_sweep: None``. Explicit null
    sweep lists read as the defaults."""
    nulls = {"sweep_modes": None, "sweep_leverages": None, "sweep_margin_limits": None}
    for req in (None, {"margin_sweep": False}, {"margin_sweep": False, **nulls}):
        world.engine.calls.clear()
        _run_id, report_id, _job_id, code = world.launch(req)
        assert code == 0
        report = world.report(report_id)["report"]
        assert report["margin_sweep"] is None
        assert report["request"]["margin_sweep"] is False
        assert world.engine.ks == [1.0, 0.5, 2.0]


def test_margin_cell_failure(world: World) -> None:
    """R23: a margin cell that raises is that cell's failed row; the report completes and
    the safest setting is picked from the other cells."""

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        if is_margin(o, "cross", 5, 1.0):
            raise RuntimeError("boom")
        return default_cell(k, o)

    world.engine.script = script
    _run_id, report_id, _job_id, code = world.launch(SWEEP)
    row = world.report(report_id)
    assert (code, row["status"]) == (0, "completed")
    sweep = row["report"]["margin_sweep"]
    failed = margin_cell(row["report"], "cross", 5, 1.0)
    assert (failed["status"], failed["error"]) == ("failed", "RuntimeError: boom")
    assert (failed["eligible"], failed["ineligible_reason"]) == (False, "failed")
    assert "total_return" not in failed
    assert sweep["status"] == "completed"
    assert sweep["safest"] == {"mode": "cross", "leverage": 3, "max_margin_pct": 0.5,
                               "move": CROSS_MOVE}  # fmt: skip


@pytest.mark.parametrize("cause", ["move_raises", "venue_not_json"])
def test_margin_stage_error_keeps_v1(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cause: str,
) -> None:
    """R23c: an error escaping the per-cell catch (the liquidating move raises, or the
    recorded venue is not JSON) fails only the sweep: rows and recommendation equal the
    sweep-off report. The log names the type, not the message."""
    from vibe_quant.bankroll import liq_move

    venue = "{not json" if cause == "venue_not_json" else VENUE
    _r, off_id, _j, code = world.launch(venue=venue)
    assert code == 0

    def boom(**kwargs: object) -> None:
        raise RuntimeError("boom-marker-7f3")

    if cause == "move_raises":
        monkeypatch.setattr(liq_move, "cell_move", boom)
    _r, on_id, _j, code = world.launch(SWEEP, venue=venue)
    on = world.report(on_id)
    assert (code, on["status"]) == (0, "completed")
    off_report = world.report(off_id)["report"]
    assert off_report["rows"]
    assert on["report"]["rows"] == off_report["rows"]
    assert on["report"]["recommendation"] == off_report["recommendation"]
    sweep = on["report"]["margin_sweep"]
    reason = sweep.pop("reason")
    assert sweep == {
        "status": "failed", "recommendation_reason": None, "k": 1.0, "scenario": None,
        "cells": [], "safest": None, "safest_reason": None,
    }  # fmt: skip
    if cause == "move_raises":
        assert reason == "RuntimeError: boom-marker-7f3"
        assert "margin sweep failed (RuntimeError)" in caplog.text
        assert "boom-marker-7f3" not in caplog.text
    else:
        assert reason.startswith("JSONDecodeError: ")
        assert "margin sweep failed (JSONDecodeError)" in caplog.text


def test_margin_cell_reuses_the_k_row_run(world: World) -> None:
    """The margin cell equal to the k-row (parent isolated 10x, margin limit 50%) runs no
    engine and reports the k-row's own run (here the baseline: k=1 is the baseline)."""

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        return result_of(total_return=0.21) if "leverage" in o else default_cell(k, o)

    world.engine.script = script
    params = {**PARENT_PARAMS, "max_margin_pct": 0.5}
    _run_id, report_id, _job_id, code = world.launch(SWEEP, params=params)
    assert code == 0
    report = world.report(report_id)["report"]
    assert margin_calls(world) == [c for c in GRID if c != ("isolated", 10, 0.5)]
    reused = margin_cell(report, "isolated", 10, 0.5)
    assert (reused["reused"], reused["status"], reused["total_return"]) == (True, "ok", 0.25)
    other = margin_cell(report, "isolated", 5, 0.5)
    assert (other["reused"], other["total_return"]) == (False, 0.21)


@pytest.mark.parametrize("venue", [
    json.dumps({"name": "BINANCE"}), None,
    json.dumps({"starting_balance_usdt": 0}), json.dumps({"starting_balance_usdt": -5}),
])  # fmt: skip
def test_margin_sweep_no_balance(world: World, venue: str | None) -> None:
    """R24: the parent recorded no starting balance (or one <= 0, R-bk-2): the cells still
    run and report, every move is unavailable (``no_starting_balance``), the scenario
    shows no balance or notional and no setting is named safest."""
    _run_id, report_id, _job_id, code = world.launch(SWEEP, venue=venue)
    assert code == 0
    sweep = world.report(report_id)["report"]["margin_sweep"]
    assert sweep["status"] == "completed" and len(sweep["cells"]) == 12
    assert sweep["scenario"]["starting_balance"] is None
    assert sweep["scenario"]["notional"] is None
    for cell in sweep["cells"]:
        assert cell["status"] == "ok" and cell["total_return"] == 0.25
        assert cell["move"] == {"status": "unavailable", "value": None, "side": None,
                                "legs": [], "reason": "no_starting_balance",
                                "path_dependent": False}  # fmt: skip
        assert (cell["positions"], cell["remainder"]) == (None, None)
    assert (sweep["safest"], sweep["safest_reason"]) == (None, "no_safe_setting")


# --- C26 (coordinator part): claim, fence, SIGTERM ------------------------------------


def test_refused_claim_does_no_engine_work(world: World) -> None:
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)
    assert world.sm.end_bankroll_report(report_id, "failed", "spawn failed", owner_pid=None)
    assert world.run(report_id, job_id) == 2
    assert world.engine.calls == []
    job = world.job(job_id)
    assert job["status"] == "failed"
    assert job["error_message"].startswith("claim refused: report")
    assert world.report(report_id)["error_message"] == "spawn failed"


def test_claim_records_the_owner(world: World) -> None:
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)
    owners: list[object] = []

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        owners.append(world.report(report_id)["owner_pid"])
        return default_cell(k, o)

    world.engine.script = script
    assert world.run(report_id, job_id) == 0
    assert owners == [os.getpid()] * 3
    job = world.job(job_id)
    assert job["pid"] == os.getpid()
    assert job["pid_start_time"] == process_start_time(os.getpid())


@pytest.mark.parametrize(
    ("req", "stage", "ks"),
    [
        ({"multiples": [0.5, 1.0, 2.0, 3.0]}, "size", [1.0, 0.5]),
        ({**SWEEP, "multiples": [0.5, 1.0, 2.0]}, "margin", [1.0, 0.5, 2.0, 1.0]),
    ],
)
def test_cancel_fences_the_next_cell(
    world: World, req: dict[str, object], stage: str, ks: list[float]
) -> None:
    """A cancel that lands while a cell runs (the size cell k=0.5, or the first margin
    cell): the next cell is never handed out, nothing is published, and the reconcile
    ends the report ``cancelled``."""
    run_id = world.parent()
    report_id, job_id = world.admit(run_id, req)

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        if is_margin(o, "isolated", 3, 0.5) if stage == "margin" else k == 0.5:
            conn = get_connection(world.db)
            try:
                run_lifecycle.finish(conn, job_id, run_lifecycle.JobStatus.KILLED, None,
                                     reconcile=False, owner=None)  # fmt: skip
            finally:
                conn.close()
        return default_cell(k, o)

    world.engine.script = script
    assert world.run(report_id, job_id) == 1
    assert world.engine.ks == ks
    assert world.report(report_id)["report"] is None
    assert world.job(job_id)["status"] == "killed"
    assert world.sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal) == [
        report_id
    ]
    row = world.report(report_id)
    assert (row["status"], row["error_message"]) == ("failed", "cancelled")
    assert not world.root(report_id).exists()


def test_sigterm_cleans_up(world: World) -> None:
    """SIGTERM mid-cell raises SystemExit: the scratch root is removed, the report ends
    ``failed`` and the job is marked failed."""
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)

    def script(k: float, o: Mapping[str, object]) -> ValidationResult:
        if k == 0.5:
            assert world.root(report_id).is_dir()
            signal.raise_signal(signal.SIGTERM)
        return default_cell(k, o)

    world.engine.script = script
    with pytest.raises(SystemExit) as exc:
        world.run(report_id, job_id)
    assert exc.value.code == 128 + signal.SIGTERM
    row = world.report(report_id)
    assert (row["status"], row["error_message"]) == ("failed", "terminated (SystemExit)")
    assert world.job(job_id)["status"] == "failed"
    assert not world.root(report_id).exists()


def test_publish_error_fails_the_report(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """The publish transaction itself raises (lock past busy_timeout): the report and the
    job end ``failed`` with the error instead of a traceback leaving them running."""

    def locked(*a: object, **k: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(StateManager, "publish_bankroll_report", locked)
    _run_id, report_id, job_id, code = world.launch()
    row = world.report(report_id)
    assert (code, row["status"], row["report"]) == (1, "failed", None)
    assert row["error_message"] == "OperationalError: database is locked"
    assert (world.job(job_id)["status"], world.job(job_id)["error_message"]) == (
        "failed", "OperationalError: database is locked",
    )  # fmt: skip
    assert not world.root(report_id).exists()


def test_refused_claim_leaves_a_finished_job_alone(world: World) -> None:
    """A reconcile already ended the job: the refused claim keeps its error and time."""
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)
    assert world.sm.end_bankroll_report(report_id, "failed", "cancelled", owner_pid=None)
    conn = get_connection(world.db)
    try:
        run_lifecycle.finish(conn, job_id, run_lifecycle.JobStatus.FAILED, "reaped",
                             reconcile=False, owner=None)  # fmt: skip
    finally:
        conn.close()
    before = dict(world.job(job_id))
    assert world.run(report_id, job_id) == 2
    after = dict(world.job(job_id))
    assert (after["status"], after["error_message"]) == ("failed", "reaped")
    assert after == before
    assert world.engine.calls == []


def test_refused_claim_leaves_another_attempts_job_alone(world: World) -> None:
    """The job row records another live process: refused, and that job keeps running."""
    run_id = world.parent()
    report_id, job_id = world.admit(run_id)
    other = os.getppid()
    world.write("UPDATE background_jobs SET status = 'running', pid = ? WHERE run_id = ?",
                (other, job_id))  # fmt: skip
    assert world.run(report_id, job_id) == 2
    job = world.job(job_id)
    assert (job["status"], job["pid"], job["error_message"]) == ("running", other, None)
    assert world.report(report_id)["status"] == "running"
    assert world.engine.calls == []


@pytest.mark.parametrize("layout", ["unnamespaced", "other_db"])
@pytest.mark.parametrize("other_id", [1, 7])
def test_two_dbs_share_a_tmp(world: World, layout: str, other_id: int) -> None:
    """Main backend and a ui-check/E2E stack on a DB copy share one tmp: this DB's report
    (id 1) neither reclaims nor removes the other DB's live root, even with the same id."""
    other = (
        world.scratch
        if layout == "unnamespaced"
        else db_scratch(world.scratch, world.tmp / "copy" / "state.db")
    )
    live = report_root(other, other_id)
    (live / "seed").mkdir(parents=True)
    (live / "archive.db").write_bytes(b"live")
    _run_id, report_id, _job_id, code = world.launch()
    assert (report_id, code, world.report(report_id)["status"]) == (1, 0, "completed")
    assert (live / "archive.db").read_bytes() == b"live" and (live / "seed").is_dir()
    assert not world.root(report_id).exists()


def test_db_scratch_is_per_resolved_db(tmp_path: Path) -> None:
    a = db_scratch(tmp_path, tmp_path / "a" / "state.db")
    assert a.parent == tmp_path and re.fullmatch(r"vq-bankroll-[0-9a-f]{12}", a.name)
    (tmp_path / "a").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "a")
    assert db_scratch(tmp_path, tmp_path / "link" / "state.db") == a
    assert db_scratch(tmp_path, tmp_path / "b" / "state.db") != a


# --- real spawn pool: worker crash, stop on fence ----------------------------------------

_REAL_EXECUTOR = rep._executor


def spawn_cell(
    inp: VariantInput, overrides: Mapping[str, object], catalog: Path, archive: Path
) -> CellRun:
    """``run_cell`` stand-in that runs in the spawned worker (``BK_SPAWN`` scripts it)."""
    k = k_of(overrides)
    action = json.loads(os.environ["BK_SPAWN"]).get(str(k))
    pid_file = Path(os.environ["BK_SPAWN_DIR"]) / "sleeper.pid"
    if action == "die":
        os.kill(os.getpid(), signal.SIGKILL)  # an OOM kill / panic=abort; no crash dialog
    if action == "slow":
        time.sleep(1.5)
    if action == "sleep":
        pid_file.write_text(str(os.getpid()))
        time.sleep(60)
    if action == "await_sleeper":
        deadline = time.monotonic() + 60
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    return CellRun.of(default_cell(k, overrides))


def _spawn_world(world: World, monkeypatch: pytest.MonkeyPatch, script: dict[str, str]) -> None:
    monkeypatch.setattr(rep, "_executor", _REAL_EXECUTOR)
    monkeypatch.setattr(rep, "run_cell", spawn_cell)
    monkeypatch.setenv("BK_SPAWN", json.dumps(script))
    monkeypatch.setenv("BK_SPAWN_DIR", str(world.tmp))


@pytest.mark.slow(reason="two reports on a real spawn pool, ~10 worker processes")
def test_worker_crash_fails_only_its_row(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """C23 for a worker that dies (no Python exception): only that cell's row fails, a
    cell broken alongside it is re-run alone, and serial/parallel rows are identical."""
    _spawn_world(world, monkeypatch, {"2.0": "die", "0.5": "slow"})
    reports = []
    for workers in (2, 1):
        _r, report_id, _j, code = world.launch({"workers": workers})
        row = world.report(report_id)
        assert (code, row["status"]) == (0, "completed"), row["error_message"]
        reports.append(row["report"])
    assert reports[0]["rows"] == reports[1]["rows"]
    rows = rows_by_k(reports[0])
    assert rows[2.0]["status"] == "failed" and rows[2.0]["eligible"] is False
    assert rows[2.0]["error"].startswith("BrokenProcessPool: ")
    assert rows[0.5]["status"] == "ok" and rows[0.5]["total_return"] == 0.06
    assert rows[1.0]["role"] == "baseline" and rows[1.0]["status"] == "ok"


def _running(pid: int) -> bool:
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return st.stdout.strip() not in ("",) and not st.stdout.strip().startswith("Z")


@pytest.mark.slow(reason="real spawn pool with a sleeping worker")
def test_fence_stops_running_workers(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fenced while a cell runs: the coordinator stops the in-flight worker process
    instead of leaving it to write into the removed root."""
    _spawn_world(world, monkeypatch, {"0.5": "await_sleeper", "2.0": "sleep"})
    owned = StateManager.bankroll_report_owned
    checks: list[bool] = []

    def fence(self: StateManager, report_id: int, pid: int) -> bool:
        checks.append(owned(self, report_id, pid) and len(checks) < 2)
        return checks[-1]

    monkeypatch.setattr(StateManager, "bankroll_report_owned", fence)
    pid_file = world.tmp / "sleeper.pid"
    try:
        _r, report_id, job_id, code = world.launch({"multiples": [0.5, 1.0, 2.0, 3.0],
                                                    "workers": 2})  # fmt: skip
        assert code == 1 and checks == [True, True, False]
        assert world.job(job_id)["error_message"] == (
            f"report {report_id} is no longer owned by pid {os.getpid()}"
        )
        sleeper = int(pid_file.read_text())
        deadline = time.monotonic() + 10
        while _running(sleeper) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _running(sleeper), "the in-flight worker outlived the fenced report"
    finally:
        if pid_file.exists():
            with contextlib.suppress(OSError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


# --- real engine, fresh process (slow) ---------------------------------------------------

_SITECUSTOMIZE = """
import os, sqlite3
_log = os.environ["BK_CONNECT_LOG"]
_connect = sqlite3.connect
def _recording_connect(database, *args, **kwargs):
    with open(_log, "a") as fh:
        fh.write(f"{os.getpid()}\\t{database}\\n")
    return _connect(database, *args, **kwargs)
sqlite3.connect = _recording_connect
"""
REAL_START, REAL_END = "2025-03-01", "2025-06-01"


def _real_archive(path: Path, symbol: str) -> Path:
    """Nonzero 8h funding + Vision intervals over the 3-month window (plus margins)."""
    from datetime import UTC, datetime

    from vibe_quant.data.archive import RawDataArchive

    day = 86_400_000
    start = int(datetime(2025, 3, 1, tzinfo=UTC).timestamp() * 1000) - 3 * day
    end = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp() * 1000) + 2 * day
    settlements = range(start, end, 8 * 3_600_000)
    archive = RawDataArchive(path)
    try:
        archive.insert_funding_rates(
            symbol,
            [(t, 0.0003 if i % 2 else -0.0001, None) for i, t in enumerate(settlements)],
            "test",
        )
        archive.insert_funding_intervals(symbol, [(t, 8) for t in settlements], "test")
    finally:
        archive.close()
    return path


def _tree_digest(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.slow(reason="real NT validation run() + two 3-engine reports in fresh processes")
def test_real_report_fresh_process(tmp_path: Path) -> None:
    """A real 1-symbol 4h 3-month funded parent (isolated, leverage 50: liquidated).

    Fresh CLI processes, spawned workers: every worker opens only the report's archive
    snapshot; the baseline equals the parent's stored headline exactly and the k=1 row is
    that baseline (ineligible ``liquidated``, Kelly not actionable); funding comes from
    each run (|k=2| > |k=1|); one worker and two workers publish identical rows; the
    source archive and catalog are unchanged and the scratch root is removed.
    """
    import hashlib
    import subprocess
    import sys

    from tests.unit.test_validation_variant import _BARS, _NEEDED, CATALOG, clone_catalog
    from vibe_quant.validation.runner import ValidationRunner

    mark = "BTCUSDT-PERP.BINANCE-1-MINUTE-MARK-EXTERNAL"
    if not all((_BARS / d).is_dir() for d in (*_NEEDED, mark)):
        pytest.skip("needs the BTCUSDT 4h, 1m and 1m MARK catalog")
    db = tmp_path / "state.db"
    catalog = clone_catalog(tmp_path / "catalog", ["BTCUSDT"])
    subprocess.run(
        ["cp", "-c", "-R", str(CATALOG.resolve() / "data" / "bar" / mark),
         str(catalog / "data" / "bar" / mark)],
        check=True,
    )  # fmt: skip
    archive = _real_archive(tmp_path / "raw_data.db", "BTCUSDT")
    hasher = dsl_hasher(NAME)
    params = {"liquidation_mode": "isolated", "leverage": 50, "max_position_pct": 1.0}
    sm = StateManager(db)
    sid = sm.create_strategy(NAME, dict(RSI_DSL))
    run_id = sm.create_backtest_run(
        sid, "validation", ["BTCUSDT"], "4h", REAL_START, REAL_END, params
    )
    sm.close()
    runner = ValidationRunner(
        db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog,
        funding_archive_path=archive,
    )  # fmt: skip
    try:
        parent = runner.run(run_id=run_id)
    finally:
        runner.close()
    assert parent.total_trades > 0 and parent.total_funding != 0.0
    assert parent.notes["liquidation"]["count"] > 0  # type: ignore[index]

    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    catalog_before = _tree_digest(catalog)
    scratch = tmp_path / "scratch"
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    reports = []
    for workers in (2, 1):
        sm = StateManager(db)
        try:
            assert sm.get_strategy(sid)["dsl_config"] == RSI_DSL  # type: ignore[index]
            stored_identity = json.loads(sm.get_backtest_result(run_id)["notes"])  # type: ignore[index]
            assert hasher(dict(RSI_DSL)) == stored_identity["validation_input"]["dsl_sha256"]
            report_id, job_id = sm.admit_bankroll_report(
                run_id, {"multiples": [0.5, 1.0, 2.0], "workers": workers},
                dsl_hash=hasher, job_alive=alive, terminate=never_signal,
            )  # fmt: skip
        finally:
            sm.close()
        conn = get_connection(db)
        try:
            run_lifecycle.open_attempt(conn, job_id, "bankroll_report", None, exists=False)
            conn.commit()
        finally:
            conn.close()
        log = tmp_path / f"connect-{workers}.log"
        env = {**os.environ, "PYTHONPATH": f"{site}{os.pathsep}{REPO}",
               "BK_CONNECT_LOG": str(log)}  # fmt: skip
        proc = subprocess.Popen(
            [sys.executable, "-m", "vibe_quant.bankroll", "report", "--report-id",
             str(report_id), "--job-id", str(job_id), "--db", str(db), "--catalog",
             str(catalog), "--archive", str(archive), "--scratch", str(scratch)],
            cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )  # fmt: skip
        out, _ = proc.communicate(timeout=900)
        assert proc.returncode == 0, out[-4000:]
        sm = StateManager(db)
        try:
            row = sm.get_bankroll_report(report_id)
        finally:
            sm.close()
        assert row is not None and row["status"] == "completed", row
        reports.append(row["report"])

        snapshot_db = report_root(db_scratch(scratch, db), report_id) / "archive.db"
        opened: dict[int, set[str]] = {}
        for line in log.read_text().splitlines():
            pid, path = line.split("\t")
            opened.setdefault(int(pid), set()).add(path)
        workers_opened = {p: s for p, s in opened.items() if p != proc.pid}
        assert len(workers_opened) == 3, opened  # one process per engine run
        for paths in workers_opened.values():
            resolved = {Path(x.removeprefix("file:").split("?")[0]).resolve() for x in paths}
            assert resolved == {snapshot_db.resolve()}, paths
        assert not report_root(db_scratch(scratch, db), report_id).exists()

    assert reports[0]["rows"] == reports[1]["rows"]
    report = reports[0]
    base = report["baseline"]
    assert (base["total_return"], base["max_drawdown"], base["sharpe_ratio"]) == (
        parent.total_return, parent.max_drawdown, parent.sharpe_ratio,
    )  # fmt: skip
    assert (base["trades"], base["total_fees"], base["total_funding"]) == (
        parent.total_trades, parent.total_fees, parent.total_funding,
    )  # fmt: skip
    rows = rows_by_k(report)
    assert rows[1.0]["role"] == "baseline"
    assert rows[1.0]["liquidation_count"] == parent.notes["liquidation"]["count"]  # type: ignore[index]
    assert rows[1.0]["eligible"] is False  # (its return is also non-positive here)
    assert report["kelly"]["actionable_reason"] in {"liquidated", "path_divergence"}
    assert report["recommendation"]["k"] is None
    assert abs(rows[2.0]["total_funding"]) > abs(rows[1.0]["total_funding"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_sha
    assert _tree_digest(catalog) == catalog_before
