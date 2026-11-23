"""Shadow forward tracker: enrolment and the update loop on a fake ``run_frozen`` seam.

Design: data/swarm/20261122-edge/designs/arch-c7-2.md §2, contract rows R14-R19, R22 (tracker
half), R25-R32, R36, R38, R42, R43. ``resolve_frozen``, ``run_frozen`` and ``coverage_day`` are
faked (T3a/T3b own them, T8 runs the real engine); the store, the catalog snapshot, estat and
the enrol history checks (catalog 1m start, archived funding) are real.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from array import array
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.objects import Price, Quantity

from vibe_quant.bankroll.report import dsl_hasher
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager, create_instrument, get_bar_type
from vibe_quant.db import state_manager
from vibe_quant.db.state_manager import StateManager, shadow_update_lock
from vibe_quant.metrics import DailyReturns
from vibe_quant.shadow import tracker
from vibe_quant.shadow.tracker import EnrolRefused, EnrolRequest
from vibe_quant.validation.results import MtmProbe, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_REPO = Path(__file__).resolve().parents[2]
PREREG = "Pre-registered: forward Sharpe > 1 on BTC 4h after costs."  # 57 chars
DSL: dict[str, object] = {
    "name": "shadow_probe_rsi",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"], "short": ["rsi > 60"]},
    "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 40"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}
_EPOCH = date(1970, 1, 1)
NOW = datetime(2026, 11, 22, 18, tzinfo=UTC)
F = date(2026, 11, 23)  # first forward day of a row enrolled at NOW
RUN_START = date(2026, 5, 27)  # F - 180 days
FAR = date(2030, 1, 1)  # `today` far ahead: the faked coverage day C decides
VENUE = '{"taker_fee":0.0005}'


def _idx(d: date) -> int:
    return (d - _EPOCH).days


def _ns(d: date, hour: int = 0) -> int:
    return (_idx(d) * 24 + hour) * 3_600 * 10**9


@functools.cache
def _noise(d: date) -> float:
    return float(np.random.default_rng(d.toordinal()).standard_normal())


def strong(d: date) -> float:
    """A clear edge: annual Sharpe ~ 7.6, passes E >= 100 in about half a year."""
    return 0.004 + 0.01 * _noise(d)


def small(d: date) -> float:
    return 0.001


def flat(d: date) -> float:
    return 0.0


@dataclass
class World:
    """Fake seams: the engine returns ``rets[name](day)`` for every day of its window."""

    rets: dict[str, Callable[[date], float]] = field(default_factory=dict)
    default: Callable[[date], float] = flat
    C: date = F
    reason: str | None = None
    venue: str = VENUE
    runs: list[dict[str, object]] = field(default_factory=list)
    coverage_calls: list[dict[str, object]] = field(default_factory=list)
    force_stop: str | None = None
    bust_ts: int | None = None
    daily_none: bool = False
    first_shift: int = 0
    short: int = 0
    fallback: int = 0
    perturb: dict[date, float] = field(default_factory=dict)

    def identity(
        self, name: str, rc: dict[str, object], detail: str | None, latency: str | None
    ) -> dict[str, object]:
        return {
            "v": 1,
            "strategy_name": name,
            # one DSL per strategy name: rows that differ only by name are different strategies
            "dsl_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "symbols": list(rc["symbols"]),  # type: ignore[call-overload]
            "timeframe": "4h",
            "start_date": rc["start_date"],
            "end_date": rc["end_date"],
            "detail_timeframe": detail or "1m",
            "latency_preset": latency or "cloud",
            "venue": self.venue,
            "launch": json.dumps(rc["parameters"], sort_keys=True, separators=(",", ":")),
            "entry_order": "market",
        }

    def resolve_frozen(self, **kw: object) -> dict[str, object]:
        assert set(kw) >= {"strategy_name", "dsl_json", "run_config", "catalog_path"}
        rc = kw["run_config"]
        assert isinstance(rc, dict)
        return self.identity(str(kw["strategy_name"]), rc, None, None)

    def run_frozen(self, **kw: object) -> tuple[ValidationResult, dict[str, object]]:
        self.runs.append(kw)
        rc = kw["run_config"]
        assert isinstance(rc, dict)
        name = str(kw["strategy_name"])
        ret = self.rets.get(name, self.default)
        start = date.fromisoformat(str(rc["start_date"]))
        end = date.fromisoformat(str(rc["end_date"]))
        first = start + timedelta(days=self.first_shift)
        days = [first + timedelta(days=i) for i in range((end - start).days - self.short)]
        vals = array("d", [self.perturb.get(d, ret(d)) for d in days])
        # like the engine: a bust or force-stop shows only in a window that reaches it
        end_ns = _ns(end)
        stop = self.force_stop
        stopped = stop is not None and datetime.fromisoformat(stop).timestamp() * 1e9 < end_ns
        bust_ts = self.bust_ts if self.bust_ts is not None and self.bust_ts < end_ns else None
        busted = stopped or bust_ts is not None
        result = ValidationResult(
            engine_force_stopped=stopped,
            mtm_probe=MtmProbe(
                paths="ok",
                daily=None if self.daily_none or busted else DailyReturns(_idx(first), vals),
                dd=0.0,
                bust_ts=bust_ts,
                path_divergence=False,
                ledger="ordinary",
            ),
        )
        result.funding_fallback_settlements = self.fallback
        if stopped:
            result.notes["engine_force_stop"] = {"stopped_at": stop}
        detail, latency = kw["detail_override"], kw["latency_override"]
        assert isinstance(detail, str) and isinstance(latency, str)
        return result, self.identity(name, rc, detail, latency)

    def coverage_day(
        self, sym_list: object, timeframes: object, **kw: object
    ) -> tuple[date, str | None]:
        self.coverage_calls.append({"symbols": list(sym_list), "timeframes": set(timeframes), **kw})  # type: ignore[call-overload]
        return self.C, self.reason


def install(monkeypatch: pytest.MonkeyPatch, world: World) -> None:
    monkeypatch.setattr(tracker, "resolve_frozen", world.resolve_frozen)
    monkeypatch.setattr(tracker, "run_frozen", world.run_frozen)
    monkeypatch.setattr(tracker, "coverage_day", world.coverage_day)


def build_data(
    root: Path,
    *,
    first_1m: datetime = datetime(2022, 1, 1, tzinfo=UTC),
    drop_funding: tuple[datetime, ...] = (),
) -> tuple[Path, Path]:
    """Catalog (BTC instrument + two 1m bars from ``first_1m``) and 8h funding 2022..2026."""
    cat, arch_path = root / "catalog", root / "archive.db"
    cm = CatalogManager(cat)
    cm.write_instrument(create_instrument("BTCUSDT"))
    bt = get_bar_type("BTCUSDT", "1m")
    px = Price.from_str("100.0")
    opens = [int(first_1m.timestamp()) * 10**9, _ns(date(2026, 12, 31))]
    cm.write_bars(
        [Bar(bt, px, px, px, px, Quantity.from_str("1.000"), o, o + 60 * 10**9 - 1) for o in opens]
    )
    dropped = {int(t.timestamp() * 1000) for t in drop_funding}
    t0 = int(datetime(2022, 1, 1, tzinfo=UTC).timestamp() * 1000)
    t1 = int(datetime(2027, 1, 1, tzinfo=UTC).timestamp() * 1000)
    arch = RawDataArchive(arch_path)
    try:
        arch.insert_funding_rates(
            "BTCUSDT",
            [(t, 0.0001, 100.0) for t in range(t0, t1, 8 * 3_600_000) if t not in dropped],
            "t",
        )
    finally:
        arch.close()
    return cat, arch_path


@dataclass
class Env:
    world: World
    sm: StateManager
    cat: Path
    arch: Path
    root: Path

    def enrol(self, *, now: datetime = NOW, sm: StateManager | None = None, **over: object) -> int:
        req: dict[str, object] = {
            "name": "rsi-btc",
            "source_kind": "strategy",
            "dsl_json": json.dumps(DSL),
            "symbols": ("BTCUSDT",),
            "parameters": {},
            "pre_registration": PREREG,
            "max_dd_limit": 0.5,
            **over,
        }
        return tracker.enrol(
            sm or self.sm,
            EnrolRequest(**req),  # type: ignore[arg-type]
            now=now,
            catalog=self.cat,
            archive=self.arch,
        )

    def update(
        self,
        c: date | None = None,
        *,
        sm: StateManager | None = None,
        failed: dict[str, set[str]] | None = None,
        budget_s: float = tracker.DEFAULT_BUDGET_S,
        today: date = FAR,
    ) -> tracker.UpdateSummary:
        if c is not None:
            self.world.C = c
        return tracker.update_all(
            sm or self.sm,
            today=today,
            trigger="test",
            failed_streams=failed or {},
            budget_s=budget_s,
            catalog=self.cat,
            archive=self.arch,
        )

    def row(self, sid: int, sm: StateManager | None = None) -> dict[str, object]:
        return next(r for r in (sm or self.sm).list_shadow_strategies() if r["id"] == sid)

    def days(self, sid: int, sm: StateManager | None = None) -> list[tuple[int, float]]:
        return (sm or self.sm).get_shadow_days(sid)

    def events(self, sid: int | None = None) -> list[tuple[str, str | None, str | None, str]]:
        rows = self.sm.conn.execute(
            "SELECT shadow_id, kind, from_state, to_state, message FROM shadow_events ORDER BY id"
        ).fetchall()
        return [(r[1], r[2], r[3], r[4]) for r in rows if sid is None or r[0] == sid]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    world = World()
    install(monkeypatch, world)
    cat, arch = build_data(tmp_path)
    sm = StateManager(tmp_path / "state.db")
    yield Env(world, sm, cat, arch, tmp_path)
    sm.close()


def _span(first: date, last: date, ret: Callable[[date], float]) -> list[tuple[int, float]]:
    return [
        (_idx(first + timedelta(days=i)), ret(first + timedelta(days=i)))
        for i in range((last - first).days + 1)
    ]


# --- enrolment -------------------------------------------------------------------------


def test_midday_enrol_zero_overlap(env: Env) -> None:
    """R15: the enrol day's own return (+5 %) is never stored; n counts from F = next midnight."""
    env.world.default = lambda d: 0.05 if d == date(2026, 11, 22) else 0.001
    sid = env.enrol()
    row = env.row(sid)
    assert (row["enrolled_at"], row["first_day"], row["run_start"]) == (
        "2026-11-22T18:00:00.000000+00:00",
        "2026-11-23",
        "2026-05-27",
    )
    env.update(date(2026, 11, 26))  # C = 11-26 carries the close-out: L = 11-25
    assert env.days(sid) == _span(F, date(2026, 11, 25), small)
    row = env.row(sid)
    assert row["committed_through"] == "2026-11-25"
    assert row["stats"]["days"] == 3  # type: ignore[index]
    (run,) = env.world.runs
    rc = run["run_config"]
    assert (rc["start_date"], rc["end_date"]) == ("2026-05-27", "2026-11-27")  # type: ignore[index]
    assert Path(str(run["catalog_path"])).name.startswith(".shadow-snap-")  # private snapshot


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 11, 22, 0, 0, 0, 0, tzinfo=UTC),
        datetime(2026, 11, 22, 23, 59, 59, 999_999, tzinfo=UTC),
    ],
)
def test_first_day_edges(env: Env, now: datetime) -> None:
    """R16: both ends of the enrol day give F = 11-23; enrolled_at keeps its microseconds."""
    row = env.row(env.enrol(now=now))
    assert row["first_day"] == "2026-11-23"
    assert row["enrolled_at"] == now.isoformat(timespec="microseconds")


def test_enrol_refuses_naive_now(env: Env) -> None:
    """A tz-naive ``now`` would be read as local time and shift F: refused, nothing written."""
    with pytest.raises(ValueError, match="^now must be timezone-aware, got 2026-11-22T23:30:00$"):
        env.enrol(now=datetime(2026, 11, 22, 23, 30))  # noqa: DTZ001
    assert env.sm.list_shadow_strategies() == []


@pytest.mark.parametrize(
    ("over", "field_name"),
    [
        ({"pre_registration": "  " + "x" * 39 + "  "}, "pre_registration"),
        ({"max_dd_limit": 0.0}, "max_dd_limit"),
        ({"max_dd_limit": 1.5}, "max_dd_limit"),
        ({"max_dd_limit": float("nan")}, "max_dd_limit"),
        ({"warmup_days": 29}, "warmup_days"),
        ({"warmup_days": 731}, "warmup_days"),
        ({"backdate_days": 0}, "backdate_days"),
        ({"backdate_days": 1501}, "backdate_days"),
    ],
)
def test_enrol_refusals(env: Env, over: dict[str, object], field_name: str) -> None:
    """R17: a bad field is refused by name; nothing is written."""
    with pytest.raises(EnrolRefused, match=field_name):
        env.enrol(**over)  # type: ignore[arg-type]
    assert env.sm.list_shadow_strategies() == []


@pytest.mark.parametrize(
    "over",
    [
        {"pre_registration": "  " + "x" * 40 + "  "},
        {"max_dd_limit": 1.0},
        {"warmup_days": 30},
        {"warmup_days": 730},
        {"backdate_days": 1},
        {"backdate_days": 1500},
    ],
)
def test_enrol_accepts_edges(env: Env, over: dict[str, object]) -> None:
    assert env.row(env.enrol(**over))["state"] == "tracking"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("first_1m", "drop", "text"),
    [
        (
            datetime(2026, 6, 1, tzinfo=UTC),
            (),
            "1m data for BTCUSDT starts 2026-06-01, after warm-up start 2026-05-27",
        ),
        (
            datetime(2026, 5, 27, 0, 1, tzinfo=UTC),
            (),
            "1m data for BTCUSDT starts 2026-05-28, after warm-up start 2026-05-27",
        ),
        (
            datetime(2024, 1, 1, tzinfo=UTC),
            (datetime(2026, 8, 1, 8, tzinfo=UTC),),
            "funding for BTCUSDT misses 1 settlement(s) in the warm-up, "
            "first 2026-08-01T08:00:00+00:00",
        ),
        (datetime(2026, 5, 27, tzinfo=UTC), (), None),  # starts exactly at the warm-up start
        (datetime(2026, 5, 26, 0, 1, tzinfo=UTC), (), None),  # first full day 05-27
        # a hole at or after F (the forward span) is the update's business, not enrol's
        (datetime(2024, 1, 1, tzinfo=UTC), (datetime(2026, 11, 23, tzinfo=UTC),), None),
    ],
)
def test_enrol_refuses_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_1m: datetime,
    drop: tuple[datetime, ...],
    text: str | None,
) -> None:
    """R18: the warm-up must be fully covered by catalog 1m bars and archived funding."""
    world = World()
    install(monkeypatch, world)
    cat, arch = build_data(tmp_path, first_1m=first_1m, drop_funding=drop)
    sm = StateManager(tmp_path / "state.db")
    try:
        e = Env(world, sm, cat, arch, tmp_path)
        if text is None:
            e.enrol()
            assert len(sm.list_shadow_strategies()) == 1
        else:
            with pytest.raises(EnrolRefused) as exc:
                e.enrol()
            assert str(exc.value) == text
            assert sm.list_shadow_strategies() == []
    finally:
        sm.close()


def test_enrol_refuses_duplicate_inside_the_insert(env: Env) -> None:
    """clmmc.27: same inputs as a live row -> refused verbatim; no row, no slot spent.

    The fake ``resolve_frozen`` hashes the strategy NAME, so the pre-check (real DSL hash)
    misses here and the in-transaction check of the insert is what refuses.
    """
    first = env.enrol()
    with pytest.raises(EnrolRefused, match=re.escape(f"already tracked as shadow #{first} (rsi-btc, tracking)")):
        env.enrol(source_ref={"run_id": 7}, symbols=["BTCUSDT"], parameters={})
    assert [r["id"] for r in env.sm.list_shadow_strategies()] == [first]
    other = env.enrol(parameters={"leverage": 3})
    assert (env.row(first)["slot"], env.row(other)["slot"]) == (1, 2)


def test_enrol_refuses_duplicate_before_resolving(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-check hashes the request's DSL as the run would and refuses before the engine."""
    first = env.enrol()
    env.sm.conn.execute(
        "UPDATE shadow_strategies SET dsl_sha256 = ?, state = 'futile' WHERE id = ?",
        (dsl_hasher("any name")(DSL), first),
    )
    env.sm.conn.commit()
    resolved: list[object] = []
    monkeypatch.setattr(tracker, "resolve_frozen", lambda **kw: resolved.append(kw))
    with pytest.raises(EnrolRefused, match=re.escape(f"already tracked as shadow #{first} (rsi-btc, futile)")):
        env.enrol(name="second try")
    assert resolved == []
    assert [r["id"] for r in env.sm.list_shadow_strategies()] == [first]


def test_unreadable_dsl_warns_as_shadow_not_bankroll(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """N2: the shadow path's unreadable-DSL warning names the shadow tracker."""
    bad = {"name": "shadow_probe_rsi", "timeframe": "4h"}  # valid JSON, invalid DSL
    with caplog.at_level(logging.WARNING):
        env.enrol(dsl_json=json.dumps(bad))
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("shadow: DSL of 'rsi-btc' is unreadable") for m in messages), messages
    assert not any("bankroll:" in m for m in messages), messages
    assert "Traceback" not in caplog.text  # no exc_info: no engine traceback in the log


_MISS = "funding for BTCUSDT misses 1 settlement(s) in the warm-up, first {}"


@pytest.mark.parametrize(
    ("now", "drop", "text"),
    [
        # a settlement later on the enrol day has not happened yet: not missing
        (datetime(2026, 11, 22, 13, tzinfo=UTC), datetime(2026, 11, 22, 16, tzinfo=UTC), None),
        (datetime(2026, 11, 22, 15, 59, 59, 999999, tzinfo=UTC),
         datetime(2026, 11, 22, 16, tzinfo=UTC), None),
        # one already past is still a hole
        (datetime(2026, 11, 22, 13, tzinfo=UTC), datetime(2026, 11, 22, 8, tzinfo=UTC),
         _MISS.format("2026-11-22T08:00:00+00:00")),
        (datetime(2026, 11, 22, 16, tzinfo=UTC), datetime(2026, 11, 22, 16, tzinfo=UTC),
         _MISS.format("2026-11-22T16:00:00+00:00")),
        (datetime(2026, 11, 22, 23, 59, tzinfo=UTC), datetime(2026, 11, 22, 16, tzinfo=UTC),
         _MISS.format("2026-11-22T16:00:00+00:00")),
    ],
)  # fmt: skip
def test_enrol_day_funding_checked_up_to_now(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    now: datetime,
    drop: datetime,
    text: str | None,
) -> None:
    """The warm-up's funding is checked through the enrol time, not the rest of that day."""
    world = World()
    install(monkeypatch, world)
    cat, arch = build_data(tmp_path, drop_funding=(drop,))
    sm = StateManager(tmp_path / "state.db")
    try:
        e = Env(world, sm, cat, arch, tmp_path)
        if text is None:
            assert e.row(e.enrol(now=now))["first_day"] == "2026-11-23"
        else:
            with pytest.raises(EnrolRefused) as exc:
                e.enrol(now=now)
            assert str(exc.value) == text
            assert sm.list_shadow_strategies() == []
    finally:
        sm.close()


def test_enrol_refuses_missing_archive(env: Env) -> None:
    """A wrong archive path is refused and never created (RawDataArchive would create it)."""
    wrong = env.root / "nope" / "archive.db"
    with pytest.raises(EnrolRefused, match="does not exist"):
        tracker.enrol(
            env.sm,
            EnrolRequest(
                name="x",
                source_kind="strategy",
                dsl_json=json.dumps(DSL),
                symbols=("BTCUSDT",),
                parameters={},
                pre_registration=PREREG,
                max_dd_limit=0.5,
            ),
            now=NOW,
            catalog=env.cat,
            archive=wrong,
        )
    assert not wrong.exists()
    assert env.sm.list_shadow_strategies() == []


def test_slot_exhausted(env: Env) -> None:
    """R19: the 6th non-test row of F1 gets no slot; E >= 100 keeps it tracking with a note."""
    env.world.default = strong
    ids = [env.enrol(name=f"s{i}") for i in range(6)]
    rows = [env.row(i) for i in ids]
    assert [r["slot"] for r in rows] == [1, 2, 3, 4, 5, None]
    assert rows[5]["status_note"] == "no evidence slot left; a new family needs your OK"
    assert env.events(ids[5]) == [
        ("note", None, None, "no evidence slot left; a new family needs your OK")
    ]
    env.update(F + timedelta(days=400))
    rows = [env.row(i) for i in ids]
    assert [r["state"] for r in rows] == ["passed"] * 5 + ["tracking"]
    assert rows[5]["status_note"] == "evidence ≥ 100 without a slot"
    assert rows[5]["verdict_day"] is None
    env.update(F + timedelta(days=407))
    notes = [e for e in env.events(ids[5]) if e[3] == "evidence ≥ 100 without a slot"]
    assert notes == [("note", None, None, "evidence ≥ 100 without a slot")]  # once


# --- update: data and engine output ---------------------------------------------------


@pytest.mark.parametrize(
    ("params", "streams", "frozen"),
    [
        ({}, {"klines"}, "klines"),
        ({}, {"funding"}, "funding"),
        ({}, {"mark"}, None),  # MARK is not a required stream with liquidation off
        ({"liquidation_mode": "isolated", "leverage": 3}, {"mark"}, "mark"),
    ],
)
def test_failed_stream_freezes(
    env: Env, params: dict[str, object], streams: set[str], frozen: str | None
) -> None:
    """R25: a required stream that failed in this data update stalls the row, 0 engine runs."""
    env.world.default = small
    sid = env.enrol(parameters=params)
    env.update(F + timedelta(days=10), failed={"BTCUSDT": streams, "ETHUSDT": {"klines"}})
    row = env.row(sid)
    if frozen is None:
        assert row["state"] == "tracking"
        assert len(env.world.runs) == 1
        return
    reason = f"data update failed for BTCUSDT ({frozen})"
    assert (row["state"], row["state_reason"]) == ("stalled", reason)
    assert env.world.runs == []
    assert env.days(sid) == []
    assert env.events(sid) == [("state_change", "tracking", "stalled", reason)]
    # the next update with the stream back advances and un-stalls
    env.update()
    row = env.row(sid)
    assert row["state"] == "tracking"
    assert env.days(sid) == _span(F, F + timedelta(days=9), small)
    assert env.events(sid)[-1][:3] == ("state_change", "stalled", "tracking")
    assert env.world.coverage_calls[-1]["mark"] is bool(params)


def test_gap_after_candidate_stalls(env: Env) -> None:
    """R22 (tracker half): days through L commit; the blocking gap stalls the row with its text."""
    env.world.default = small
    sid = env.enrol()
    gap = "data gap at 2026-12-03 (BTCUSDT)"
    env.world.reason = gap
    env.update(date(2026, 12, 2))
    row = env.row(sid)
    assert (row["state"], row["state_reason"]) == ("stalled", gap)
    assert env.days(sid) == _span(F, date(2026, 12, 1), small)
    assert row["committed_through"] == "2026-12-01"
    # gap persists, no new day: no engine run, still stalled
    env.update()
    assert len(env.world.runs) == 1
    assert env.row(sid)["state"] == "stalled"
    env.world.reason = None
    env.update(date(2026, 12, 6))
    assert env.row(sid)["state"] == "tracking"
    assert env.days(sid) == _span(F, date(2026, 12, 5), small)


@pytest.mark.parametrize(
    ("force_stop", "bust_ts", "state", "reason", "verdict", "last", "runs"),
    [
        # force-stopped and no daily series: the bust branch wins over "no daily series";
        # a re-run to the bust day commits the days a daily cadence saw before it (R14)
        (
            "2026-12-05T03:00:00+00:00",
            None,
            "futile",
            "account bust on 2026-12-05",
            "2026-12-05",
            date(2026, 12, 3),
            2,
        ),
        (
            None,
            _ns(date(2026, 11, 30), 12),
            "futile",
            "account bust on 2026-11-30",
            "2026-11-30",
            date(2026, 11, 28),
            2,
        ),
        # bust on F + 1: no day before it can commit, no re-run
        (
            "2026-11-24T00:00:00+00:00",
            None,
            "futile",
            "account bust on 2026-11-24",
            "2026-11-24",
            None,
            1,
        ),
        # bust on F + 2: day F commits on the re-run
        (
            "2026-11-25T00:00:00+00:00",
            None,
            "futile",
            "account bust on 2026-11-25",
            "2026-11-25",
            F,
            2,
        ),
        (
            "2026-11-23T00:00:00+00:00",
            None,
            "futile",
            "account bust on 2026-11-23",
            "2026-11-23",
            None,
            1,
        ),
        (
            "2026-11-22T23:59:00+00:00",
            None,
            "retired",
            "warm-up bust on 2026-11-22 not seen at enrol",
            None,
            None,
            1,
        ),
    ],
)
def test_bust_before_missing(
    env: Env,
    force_stop: str | None,
    bust_ts: int | None,
    state: str,
    reason: str,
    verdict: str | None,
    last: date | None,
    runs: int,
) -> None:
    """R26: a bust on/after F is futile, before F retires; checked before the missing series."""
    sid = env.enrol()
    env.world.force_stop, env.world.bust_ts = force_stop, bust_ts
    env.update(date(2026, 12, 10))
    row = env.row(sid)
    assert (row["state"], row["state_reason"], row["verdict_day"]) == (state, reason, verdict)
    assert env.days(sid) == ([] if last is None else _span(F, last, flat))
    assert row["committed_through"] == (None if last is None else last.isoformat())
    assert env.events(sid) == [("state_change", "tracking", state, reason)]
    assert len(env.world.runs) == runs
    ends = [str(r["run_config"]["end_date"]) for r in env.world.runs]  # type: ignore[index]
    assert ends == ["2026-12-11", verdict][:runs]
    env.update(date(2026, 12, 11))  # terminal: never run again
    assert len(env.world.runs) == runs


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"daily_none": True}, "no daily series"),
        (
            {"first_shift": 1},
            "engine returned 189 days from 2026-05-28, expected 189 from 2026-05-27",
        ),
        ({"short": 1}, "engine returned 188 days from 2026-05-27, expected 189 from 2026-05-27"),
        ({"fallback": 1}, "funding fallback charged on 1 settlement(s)"),
    ],
)
def test_output_shape(env: Env, over: dict[str, object], reason: str) -> None:
    """R27: a series that is not exactly run_start..C, or a fallback rate, stalls with 0 days."""
    env.world.default = small
    sid = env.enrol()
    for k, v in over.items():
        setattr(env.world, k, v)
    env.update(date(2026, 12, 1))  # C + 1 - run_start = 189 days
    row = env.row(sid)
    assert (row["state"], row["state_reason"]) == ("stalled", reason)
    assert env.days(sid) == []
    assert row["committed_through"] is None


@pytest.mark.parametrize(
    ("seam", "exc", "reason"),
    [
        ("run_frozen", RuntimeError("bad bar"), "engine error: RuntimeError: bad bar"),
        ("snapshot", TimeoutError("catalog busy"), "engine error: TimeoutError: catalog busy"),
    ],
)
def test_engine_error_stalls(
    env: Env, monkeypatch: pytest.MonkeyPatch, seam: str, exc: Exception, reason: str
) -> None:
    """A failing run or snapshot stalls that row only; the next row still runs."""
    env.world.default = small
    a, b = env.enrol(name="a"), env.enrol(name="b")
    real = getattr(tracker, seam)
    calls: list[int] = []

    def first_fails(*args: object, **kw: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return real(*args, **kw)

    monkeypatch.setattr(tracker, seam, first_fails)
    env.update(date(2026, 12, 1))
    row = env.row(a)
    assert (row["state"], row["state_reason"]) == ("stalled", reason)
    assert env.days(a) == []
    assert env.row(b)["state"] == "tracking"
    assert env.days(b) == _span(F, date(2026, 11, 30), small)


@pytest.mark.parametrize(
    ("dsl", "boom", "state", "reason"),
    [
        (
            {**DSL, "entry_conditions": {"long": ["foo < 45"]}},
            None,
            "retired",
            "invalid DSL: entry_conditions.long: Unknown indicator 'foo' in condition. "
            "Defined indicators: rsi: 'foo < 45'",
        ),
        (
            "{not json",
            None,
            "retired",
            "invalid DSL: Expecting property name enclosed in double quotes: "
            "line 1 column 2 (char 1)",
        ),
        (None, OSError("disk gone"), "stalled", "update error: OSError: disk gone"),
        (None, KeyError("ETHUSDT"), "stalled", "update error: KeyError: 'ETHUSDT'"),
    ],
)
def test_bad_row_never_blocks_others(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    dsl: object,
    boom: Exception | None,
    state: str,
    reason: str,
) -> None:
    """B1: a row whose stored DSL no longer validates retires, one whose coverage step raises
    stalls; it sorts first, yet the other rows advance in the same update."""
    env.world.default = small
    a, b, c = env.enrol(name="a"), env.enrol(name="b"), env.enrol(name="c")
    if dsl is not None:
        text = dsl if isinstance(dsl, str) else json.dumps(dsl)
        env.sm.conn.execute("UPDATE shadow_strategies SET dsl_json = ? WHERE id = ?", (text, a))
        env.sm.conn.commit()
    if boom is not None:
        real = env.world.coverage_day
        calls: list[int] = []

        def first_raises(*args: object, **kw: object) -> tuple[date, str | None]:
            calls.append(1)
            if len(calls) == 1:
                raise boom
            return real(*args, **kw)

        monkeypatch.setattr(tracker, "coverage_day", first_raises)
    env.update(F + timedelta(days=10))
    row = env.row(a)
    assert (row["state"], row["state_reason"], row["status_note"]) == (state, reason, reason)
    assert row["last_update_at"] is not None  # no longer sorts first forever
    assert env.days(a) == []
    assert env.events(a) == [("state_change", "tracking", state, reason)]
    for sid in (b, c):
        assert env.row(sid)["state"] == "tracking"
        assert env.days(sid) == _span(F, F + timedelta(days=9), small)


@pytest.mark.parametrize(
    "boom",
    [sqlite3.OperationalError("database is locked"), tracker.ShadowUpdateRunning("held")],
)
def test_db_error_aborts_update(env: Env, monkeypatch: pytest.MonkeyPatch, boom: Exception) -> None:
    """A state-DB error or a lock refusal inside a row fails the whole update closed."""
    env.world.default = small
    a, b = env.enrol(name="a"), env.enrol(name="b")

    def raises(*args: object, **kw: object) -> tuple[date, str | None]:
        raise boom

    monkeypatch.setattr(tracker, "coverage_day", raises)
    with pytest.raises(type(boom)):
        env.update(F + timedelta(days=10))
    for sid in (a, b):
        row = env.row(sid)
        assert (row["state"], row["last_update_at"]) == ("tracking", None)
        assert env.days(sid) == []


# --- reproduce: drift, inputs, epochs ---------------------------------------------------


def test_drift_retires(env: Env) -> None:
    """R28: one ulp on a stored day retires the row; stored days stay; no further runs."""
    env.world.default = small
    sid = env.enrol()
    env.update(F + timedelta(days=10))
    stored = env.days(sid)
    assert len(stored) == 10
    env.world.perturb = {F + timedelta(days=3): float(np.nextafter(0.001, 1.0))}
    env.update(F + timedelta(days=11))
    row = env.row(sid)
    reason = "day 2026-11-26: stored 0.001 now 0.0010000000000000002"
    assert (row["state"], row["state_reason"]) == ("retired", reason)
    assert env.days(sid) == stored
    assert row["committed_through"] == "2026-12-02"
    env.update(F + timedelta(days=12))
    assert len(env.world.runs) == 2


def test_negative_zero_reproduces(env: Env) -> None:
    """A stored 0.0 that comes back as -0.0 is the same day (IEEE ==), not drift."""
    sid = env.enrol()
    env.update(F + timedelta(days=5))
    env.world.perturb = {F + timedelta(days=1): -0.0}
    env.update(F + timedelta(days=6))
    assert env.row(sid)["state"] == "tracking"
    assert len(env.days(sid)) == 6


def test_flat_prefix_input_change(env: Env) -> None:
    """R29: every stored day reproduces, but a frozen input (the venue fee) changed: retired."""
    sid = env.enrol()
    env.update(F + timedelta(days=10))
    env.world.venue = '{"taker_fee":0.0004}'
    env.update(F + timedelta(days=11))
    row = env.row(sid)
    assert (row["state"], row["state_reason"]) == ("retired", "inputs changed: venue")
    assert len(env.days(sid)) == 10


def test_instrument_change_retires(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = env.enrol()
    env.update(F + timedelta(days=10))
    monkeypatch.setattr(tracker, "_instrument_identity", lambda syms: {"BTCUSDT": "{}"})
    env.update(F + timedelta(days=11))
    assert env.row(sid)["state_reason"] == "inputs changed: instrument"


def test_epoch_identical_keeps(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """R30: a declared engine change that reproduces every stored day keeps tracking."""
    env.world.default = small
    sid = env.enrol()
    env.update(F + timedelta(days=10))
    monkeypatch.setattr(tracker, "ENGINE_EPOCH", 2)
    env.update(F + timedelta(days=11))
    row = env.row(sid)
    assert (row["state"], row["engine_epoch"], row["engine_epoch_last"]) == ("tracking", 1, 2)
    assert env.events(sid) == [("note", None, None, "engine epoch 1→2: stored days reproduce")]
    assert len(env.days(sid)) == 11


def test_epoch_change_never_rewrites(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """R31: epoch moved + prefix differs -> retired; a passed row's verdict stays, marked stale once."""
    env.world.rets = {"a": small, "b": strong}
    a, b = env.enrol(name="a"), env.enrol(name="b")
    env.update(F + timedelta(days=400))
    passed = env.row(b)
    assert passed["state"] == "passed"
    a_days = env.days(a)
    monkeypatch.setattr(tracker, "ENGINE_EPOCH", 2)
    env.world.perturb = {F + timedelta(days=2): 0.002}
    env.update(F + timedelta(days=401))
    row = env.row(a)
    assert row["state"] == "retired"
    assert row["state_reason"] == (
        "day 2026-11-25: stored 0.001 now 0.002; declared engine change epoch 1→2"
    )
    assert env.days(a) == a_days
    stale = "evidence measured on engine epoch 1; paper decision must re-check under 2"
    env.update(F + timedelta(days=402))
    row_b = env.row(b)
    assert row_b["proposal_stale"] == 1
    assert (row_b["state"], row_b["verdict_day"], row_b["engine_epoch_last"]) == (
        "passed",
        passed["verdict_day"],
        1,
    )
    assert [e for e in env.events(b) if e[3] == stale] == [
        ("state_change", "passed", "passed", stale)
    ]


def test_reenrol_fresh_clock(env: Env) -> None:
    """R32: re-enrolling a retired strategy is a new row, new slot, new F, no days copied."""
    env.world.default = small
    old = env.enrol()
    env.update(F + timedelta(days=10))
    env.world.venue = '{"taker_fee":0.0004}'
    env.update(F + timedelta(days=11))
    assert env.row(old)["state"] == "retired"
    later = datetime(2026, 12, 24, 6, tzinfo=UTC)
    new = env.enrol(now=later)
    row = env.row(new)
    assert new != old
    assert (row["slot"], row["first_day"], row["state"]) == (2, "2026-12-25", "tracking")
    assert env.days(new) == []
    assert len(env.days(old)) == 10


# --- verdicts --------------------------------------------------------------------------


@pytest.mark.slow(reason="R14 needs ~180 daily updates up to the pass day")
def test_cadence_invariant(env: Env) -> None:
    """R14: daily updates and one batch give the same verdict day, state and stored days,
    also when the batch runs past a bust 149 days after the pass (the pass stands)."""
    env.world.default = strong
    env.world.bust_ts = _ns(F + timedelta(days=330), 12)  # past every run but bust_sm's
    daily_sm = StateManager(env.root / "daily.db")
    bust_sm = StateManager(env.root / "bust.db")
    try:
        a = env.enrol()
        b = env.enrol(sm=daily_sm)
        z = env.enrol(sm=bust_sm)
        end = F + timedelta(days=300)
        env.update(end)
        env.update(F + timedelta(days=340), sm=bust_sm)
        c = F + timedelta(days=1)
        while env.row(b, daily_sm)["state"] == "tracking" and c <= end:
            env.update(c, sm=daily_sm)
            c += timedelta(days=1)
        batch, daily, busted = env.row(a), env.row(b, daily_sm), env.row(z, bust_sm)
        keys = ("state", "verdict_day", "state_reason", "committed_through", "stats")
        assert {k: batch[k] for k in keys} == {k: daily[k] for k in keys}
        assert {k: busted[k] for k in keys} == {k: daily[k] for k in keys}
        assert (batch["state"], batch["verdict_day"]) == ("passed", "2027-05-23")
        assert batch["committed_through"] == batch["verdict_day"]
        assert env.days(a) == env.days(b, daily_sm) == env.days(z, bust_sm)
        verdict = date.fromisoformat(str(batch["verdict_day"]))
        assert ((verdict - F).days + 1) % 7 == 0  # decided at a block end
    finally:
        daily_sm.close()
        bust_sm.close()


def _dd_then_flat(d: date) -> float:
    return -0.02 if date(2026, 12, 3) <= d <= date(2026, 12, 5) else 0.0


@pytest.mark.parametrize(
    ("ret", "max_dd", "bust", "end", "state", "verdict", "reason"),
    [
        # drawdown past the limit 10 days before the bust: futile on the drawdown day
        (
            _dd_then_flat,
            0.05,
            date(2026, 12, 15),
            date(2026, 12, 25),
            "futile",
            "2026-12-05",
            "max drawdown 5.88% > limit 5.00%",
        ),
        # no verdict before the bust: futile on the bust day, days through bust - 2 stored
        (
            flat,
            0.5,
            date(2026, 12, 15),
            date(2026, 12, 25),
            "futile",
            "2026-12-15",
            "account bust on 2026-12-15",
        ),
    ],
)
def test_cadence_invariant_bust_in_batch(
    env: Env,
    ret: Callable[[date], float],
    max_dd: float,
    bust: date,
    end: date,
    state: str,
    verdict: str,
    reason: str,
) -> None:
    """R14 with a terminal event in the batch: a bust after an earlier verdict day never
    overrides it; batch and daily agree on state, day, reason, stored days and stats
    (pass-then-bust: test_cadence_invariant)."""
    env.world.default = ret
    env.world.bust_ts = _ns(bust, 12)
    daily_sm = StateManager(env.root / "daily.db")
    try:
        a = env.enrol(max_dd_limit=max_dd)
        b = env.enrol(sm=daily_sm, max_dd_limit=max_dd)
        env.update(end)
        c = F + timedelta(days=1)
        while env.row(b, daily_sm)["state"] == "tracking" and c <= end:
            env.update(c, sm=daily_sm)
            c += timedelta(days=1)
        batch, daily = env.row(a), env.row(b, daily_sm)
        keys = ("state", "verdict_day", "state_reason", "committed_through", "stats")
        assert {k: batch[k] for k in keys} == {k: daily[k] for k in keys}
        assert (batch["state"], batch["verdict_day"]) == (state, verdict)
        assert batch["state_reason"] == reason
        assert env.days(a) == env.days(b, daily_sm)
        bust_only = str(reason).startswith("account bust")
        stored_last = bust - timedelta(days=2) if bust_only else date.fromisoformat(verdict)
        assert env.days(a)[-1][0] == _idx(stored_last)
    finally:
        daily_sm.close()


def _paper_footprint(sm: StateManager, root: Path) -> tuple[dict[str, int], list[str]]:
    tables = [
        r[0]
        for r in sm.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not r[0].startswith(("shadow_", "sqlite_"))
    ]
    counts = {t: sm.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    return counts, sorted(p.name for p in (root / "data" / "state" / "paper_configs").iterdir())


def test_pass_only_records(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """R42: proposals off -> a real-slot pass records state only; nothing paper-side moves."""
    assert tracker.PROPOSALS_ENABLED is False
    monkeypatch.chdir(env.root)
    (env.root / "data" / "state" / "paper_configs").mkdir(parents=True)
    (env.root / "data" / "state" / "paper_configs" / "sentinel.json").write_text("{}")
    env.world.default = strong
    sid = env.enrol()
    before = _paper_footprint(env.sm, env.root)
    env.update(F + timedelta(days=400))
    row = env.row(sid)
    assert (row["state"], row["slot"]) == ("passed", 1)
    assert str(row["state_reason"]).startswith("evidence ")
    assert not any("paper_proposal" in e[3] for e in env.events())
    assert _paper_footprint(env.sm, env.root) == before
    assert env.days(sid)[-1][0] == _idx(date.fromisoformat(str(row["verdict_day"])))
    env.update(F + timedelta(days=401))  # same engine epoch: the pass is never flagged stale
    assert env.row(sid)["proposal_stale"] == 0
    assert not any("engine epoch" in e[3] for e in env.events(sid))


def test_test_row_never_proposes(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """R36: a backdated test row passes with no slot and no proposal even with proposals on."""
    monkeypatch.setattr(tracker, "PROPOSALS_ENABLED", True)
    env.world.default = strong
    test_id = env.enrol(name="bt", backdate_days=400)
    real_id = env.enrol(name="real")  # proposals on: a real-slot pass proposes (the gate is real)
    row = env.row(test_id)
    assert (row["is_test"], row["slot"], row["first_day"]) == (1, None, "2025-10-19")
    env.update(date(2026, 11, 21))
    row = env.row(test_id)
    assert row["state"] == "passed"
    events = env.events(test_id)
    assert [e[:3] for e in events] == [("state_change", "tracking", "passed")]
    assert events[0][3].startswith("[TEST] evidence ")
    assert not any("paper_proposal" in e[3] for e in events)
    env.update(F + timedelta(days=400))
    assert env.row(real_id)["state"] == "passed"
    assert any(e[3].startswith("paper_proposal") for e in env.events(real_id))


def test_futile_drawdown(env: Env) -> None:
    """A drawdown past the row's limit stops it on that day; days through it are stored."""
    env.world.default = lambda d: -0.02 if d >= date(2026, 11, 25) else 0.0
    sid = env.enrol(max_dd_limit=0.05)
    env.update(F + timedelta(days=30))
    row = env.row(sid)
    assert row["state"] == "futile"
    assert row["verdict_day"] == "2026-11-27"  # 1 - 0.98^3 = 5.88 % > 5 %
    assert row["state_reason"] == "max drawdown 5.88% > limit 5.00%"
    assert env.days(sid)[-1][0] == _idx(date(2026, 11, 27))


# --- budget, admission, cleanup, crash --------------------------------------------------


def test_budget_defers(env: Env) -> None:
    """R43: budget 0 -> 0 runs, both rows deferred, no state change; step-0 cleanup still runs."""
    a, b = env.enrol(name="a"), env.enrol(name="b")
    now = time.time()
    old, young = env.root / ".shadow-snap-old", env.root / ".shadow-snap-young"
    for path, age in ((old, 7200.0), (young, 60.0)):
        path.mkdir()
        os.utime(path, (now - age, now - age))
    summary = env.update(F + timedelta(days=10), budget_s=0)
    assert env.world.runs == []
    for sid in (a, b):
        row = env.row(sid)
        assert (row["state"], row["status_note"], row["last_update_at"]) == (
            "tracking",
            "deferred: update budget",
            None,
        )
    assert env.events() == []
    assert summary.runs == 0
    assert not old.exists()  # an orphan
    assert young.is_dir()  # maybe a peer tracker's snapshot before its owner lock


def test_update_refuses_while_locked(env: Env) -> None:
    assert tracker.ShadowUpdateRunning is state_manager.ShadowUpdateRunning
    env.enrol()
    db = env.sm.db_path
    assert db is not None
    with shadow_update_lock(db), pytest.raises(tracker.ShadowUpdateRunning):
        env.update(F + timedelta(days=10))
    assert env.world.runs == []


_KILL_CHILD = textwrap.dedent(
    """
    import signal, sys
    from datetime import date
    from pathlib import Path
    import pytest
    from tests.unit.test_shadow_tracker import World, install, small
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.shadow import tracker

    db, cat, arch, c = sys.argv[1:5]
    install(pytest.MonkeyPatch(), World(default=small, C=date.fromisoformat(c)))
    sm = StateManager(Path(db))
    updates = []
    def trace(sql):
        if sql.lstrip().startswith("UPDATE shadow_strategies"):
            updates.append(sql)
            if len(updates) == 2:  # row 2's transaction, after its days
                print("inside", flush=True)
                signal.pause()
    sm.conn.set_trace_callback(trace)
    tracker.update_all(sm, today=date(2030, 1, 1), trigger="test", failed_streams={},
                       catalog=Path(cat), archive=Path(arch))
    print("committed (BAD)", flush=True)
    """
)


def test_kill_between_rows(env: Env) -> None:
    """R38: SIGKILL inside row 2's transaction; the rerun says row 1 has no new data and
    processes row 2 once (no duplicate day)."""
    a, b = env.enrol(name="a"), env.enrol(name="b")
    c = F + timedelta(days=10)
    db = env.sm.db_path
    assert db is not None
    child = subprocess.Popen(
        [sys.executable, "-c", _KILL_CHILD, str(db), str(env.cat), str(env.arch), c.isoformat()],
        stdout=subprocess.PIPE,
        text=True,
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "inside"
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    assert child.returncode == -signal.SIGKILL
    expected = _span(F, c - timedelta(days=1), small)
    assert env.days(a) == expected
    assert env.days(b) == []
    env.world.default = small
    env.update(c)
    assert env.row(a)["status_note"] == "no new data since 2026-12-02"
    assert env.days(a) == expected
    assert env.days(b) == expected
    assert len(env.world.runs) == 1  # row 2 only
    dup = env.sm.conn.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM shadow_days GROUP BY shadow_id, day "
        "HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert dup == 0
    with pytest.raises(sqlite3.IntegrityError):
        env.sm.commit_shadow_update(a, new_days=expected[:1], fields={}, events=[])
