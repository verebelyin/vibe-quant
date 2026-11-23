"""The shadow hook at the end of a user-requested data update, and the shadow CLI.

Design arch-c7-2 §2 Hook / CLI / admission, contract row R40; user 2026-11-23: no
scheduler, the update refreshes MARK for every symbol of a tracked liquidation row first,
and a MARK failure stalls that row instead of aborting the update.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data import ingest
from vibe_quant.db.state_manager import (
    ShadowUpdateRunning,
    StateManager,
    shadow_update_lock,
)
from vibe_quant.shadow import notify, tracker

if TYPE_CHECKING:
    from collections.abc import Mapping

_REPO = Path(__file__).resolve().parents[2]
_PREREG = "Pre-registered: forward Sharpe > 1 on BTC 4h after costs."


def _row(
    name: str, symbols: list[str], params: dict[str, object], **over: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "source_kind": "strategy",
        "dsl_json": "{}",
        "dsl_sha256": "ab" * 32,
        "compiler_hash": "c0ffee",
        "symbols": symbols,
        "timeframe": "4h",
        "parameters": params,
        "frozen_identity": {},
        "enrolled_at": "2026-11-22T18:00:00.000000+00:00",
        "first_day": "2026-11-23",
        "run_start": "2026-05-27",
        "family": "F1",
        "pre_registration": _PREREG,
        "max_dd_limit": 0.2,
        "engine_epoch": 1,
        "engine_epoch_last": 1,
    }
    row.update(over)
    return row


def _db(path: Path, *rows: dict[str, object]) -> Path:
    sm = StateManager(path)
    sm.list_shadow_strategies()  # opens the connection: the DB file exists even with no rows
    for row in rows:
        sm.insert_shadow_strategy(row)
    sm.close()
    return path


class Hook:
    """Records what the shadow hook hands the tracker and the notifier."""

    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []
        self.notified: list[Path] = []
        self.raise_exc: Exception | None = None
        self.catalog_free: list[bool] = []

    def update_all(self, state: StateManager, **kw: object) -> tracker.UpdateSummary:
        try:
            with catalog_mod.catalog_write_lock(Path("data/catalog"), timeout_s=0):
                self.catalog_free.append(True)
        except TimeoutError:
            self.catalog_free.append(False)
        failed = kw["failed_streams"]
        assert isinstance(failed, dict)
        self.updates.append(
            {
                "db": Path(str(state.db_path)).resolve(),
                "trigger": kw["trigger"],
                "failed": {k: sorted(v) for k, v in failed.items()},
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return tracker.UpdateSummary(trigger=str(kw["trigger"]))

    def send_pending(self, state: StateManager) -> int:
        self.notified.append(Path(str(state.db_path)).resolve())
        return 0


@pytest.fixture
def hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Hook:
    """A private cwd (data/catalog, data/archive) and fake tracker/notifier/downloads."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "catalog").mkdir(parents=True)
    h = Hook()
    monkeypatch.setattr(tracker, "update_all", h.update_all)
    monkeypatch.setattr(notify, "send_pending", h.send_pending)
    return h


def _fake_update(results: Mapping[str, dict[str, object]], monkeypatch: pytest.MonkeyPatch) -> None:
    """Real ``ingest.update_all`` (catalog lock, download session) over fake symbols."""

    def update_symbol(symbol: str, **_: object) -> dict[str, object]:
        return {"new_klines": 0, "failed_ranges": [], **results.get(symbol, {})}

    monkeypatch.setattr(ingest, "update_symbol", update_symbol)


class Mark:
    def __init__(self, fail: dict[str, Exception | dict[str, object]] | None = None) -> None:
        self.fail = fail or {}
        self.fetched: list[tuple[str, str]] = []
        self.built: list[str] = []

    def ingest_mark_klines(self, symbol: str, start_date: object, **_: object) -> dict[str, object]:
        self.fetched.append((symbol, str(start_date)[:10]))
        bad = self.fail.get(symbol)
        if isinstance(bad, Exception):
            raise bad
        return {"failed_months": [], "failed_days": [], **(bad or {})}

    def build_mark_bars(self, catalog: object, archive: object, symbol: str) -> int:
        self.built.append(symbol)
        return 1


@pytest.fixture
def mark(monkeypatch: pytest.MonkeyPatch) -> Mark:
    m = Mark()
    monkeypatch.setattr(ingest, "ingest_mark_klines", m.ingest_mark_klines)
    monkeypatch.setattr(ingest, "build_mark_bars", m.build_mark_bars)
    return m


@pytest.mark.parametrize("db_name", ["alt.db", "nested/other-state.db"])
def test_state_db_explicit(
    hook: Hook, mark: Mark, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_name: str
) -> None:
    """R40: ``update --state-db X`` runs the tracker and the notifier on X, nothing else."""
    db = tmp_path / db_name
    db.parent.mkdir(parents=True, exist_ok=True)
    _db(db, _row("rsi-btc", ["BTCUSDT"], {}))
    _fake_update({}, monkeypatch)
    default_db = Path(os.environ["VIBE_QUANT_DB"])
    before = default_db.stat().st_mtime_ns if default_db.exists() else None
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)]) == 0
    assert hook.updates == [{"db": db.resolve(), "trigger": "data-update", "failed": {}}]
    assert hook.notified == [db.resolve()]
    assert (default_db.stat().st_mtime_ns if default_db.exists() else None) == before
    assert hook.catalog_free == [True]  # the hook runs after update_all released the lock


def test_failed_streams_reach_tracker(
    hook: Hook, mark: Mark, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed klines range / funding refresh of this update is handed on per symbol."""
    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT", "ETHUSDT"], {}))
    _fake_update(
        {
            "BTCUSDT": {"failed_ranges": ["2026-11-22T00:00..2026-11-22T01:00"]},
            "ETHUSDT": {"funding_failed": True},
        },
        monkeypatch,
    )
    code = ingest.main(["update", "--symbols", "BTCUSDT,ETHUSDT,SOLUSDT", "--state-db", str(db)])
    assert code == 1  # the data exit code is the update's own
    assert hook.updates[0]["failed"] == {"BTCUSDT": ["klines"], "ETHUSDT": ["funding"]}


def test_skip_shadow(
    hook: Hook, mark: Mark, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT"], {"liquidation_mode": "isolated"}))
    _fake_update({}, monkeypatch)
    assert (
        ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db), "--skip-shadow"]) == 0
    )
    assert hook.updates == []
    assert hook.notified == []
    assert mark.fetched == []


def test_update_error_skips_hook(
    hook: Hook, mark: Mark, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hook runs only after update_all RETURNS."""
    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT"], {}))

    def boom(symbol: str, **_: object) -> dict[str, object]:
        raise RuntimeError("binance down")

    monkeypatch.setattr(ingest, "update_symbol", boom)
    with pytest.raises(RuntimeError, match="binance down"):
        ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)])
    assert hook.updates == []


def test_hook_lock_held_keeps_exit_code(
    hook: Hook,
    mark: Mark,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT"], {}))
    _fake_update({}, monkeypatch)
    hook.raise_exc = ShadowUpdateRunning("shadow update already running; skipped")
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)]) == 0
    assert "shadow update already running; skipped" in capsys.readouterr().err
    assert hook.notified == []  # never notify while another updater runs


def test_hook_error_keeps_exit_code(
    hook: Hook,
    mark: Mark,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT"], {}))
    _fake_update({}, monkeypatch)
    hook.raise_exc = RuntimeError("engine fell over SECRET-9c1")
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)]) == 0
    assert "shadow update failed (RuntimeError)" in caplog.text
    assert "SECRET-9c1" not in caplog.text
    assert hook.notified == []


def test_mark_refresh_for_tracked_liquidation_rows(
    hook: Hook, mark: Mark, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MARK (archive + catalog) is refreshed for live liquidation rows' symbols only."""
    db = _db(
        tmp_path / "s.db",
        _row("liq-btc", ["BTCUSDT", "SOLUSDT"], {"liquidation_mode": "isolated", "leverage": 5}),
        _row("liq-btc-early", ["BTCUSDT"], {"liquidation_mode": "cross"}, run_start="2026-03-01"),
        _row("plain-eth", ["ETHUSDT"], {}),
        _row("done-xrp", ["XRPUSDT"], {"liquidation_mode": "cross"}),
    )
    sm = StateManager(db)
    xrp = next(r for r in sm.list_shadow_strategies() if r["name"] == "done-xrp")
    sm.commit_shadow_update(int(xrp["id"]), new_days=[], fields={"state": "retired"}, events=[])
    sm.close()
    _fake_update({}, monkeypatch)
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)]) == 0
    # earliest run_start per symbol; retired and liquidation-off rows add nothing
    assert sorted(mark.fetched) == [("BTCUSDT", "2026-03-01"), ("SOLUSDT", "2026-05-27")]
    assert sorted(mark.built) == ["BTCUSDT", "SOLUSDT"]
    assert hook.updates[0]["failed"] == {}


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("vision 503 SECRET-4be"),
        {"failed_months": ["2026-11"]},
        {"failed_days": ["2026-11-21"]},
    ],
    ids=["raises", "failed-month", "failed-day"],
)
def test_mark_failure_stalls_row_not_update(
    hook: Hook,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure: Exception | dict[str, object],
) -> None:
    m = Mark({"BTCUSDT": failure})
    monkeypatch.setattr(ingest, "ingest_mark_klines", m.ingest_mark_klines)
    monkeypatch.setattr(ingest, "build_mark_bars", m.build_mark_bars)
    db = _db(
        tmp_path / "s.db", _row("liq", ["BTCUSDT", "ETHUSDT"], {"liquidation_mode": "isolated"})
    )
    _fake_update({}, monkeypatch)
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)]) == 0
    assert hook.updates[0]["failed"] == {"BTCUSDT": ["mark"]}
    assert m.built == ["ETHUSDT"]  # the failed symbol's catalog is left as it was
    assert "MARK refresh failed for BTCUSDT" in caplog.text
    assert "SECRET-4be" not in caplog.text


def test_mark_build_failure_stalls_row(
    hook: Hook, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    m = Mark()

    def build(catalog: object, archive: object, symbol: str) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(ingest, "ingest_mark_klines", m.ingest_mark_klines)
    monkeypatch.setattr(ingest, "build_mark_bars", build)
    db = _db(tmp_path / "s.db", _row("liq", ["BTCUSDT"], {"liquidation_mode": "isolated"}))
    _fake_update({}, monkeypatch)
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(db)]) == 0
    assert hook.updates[0]["failed"] == {"BTCUSDT": ["mark"]}


_PROBE = textwrap.dedent(
    """
    import json, sys
    from vibe_quant.data import ingest
    ingest.update_all = lambda symbols=None, verbose=True: {}
    rc = ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", sys.argv[1]])
    print(json.dumps({
        "rc": rc,
        "tracker": "vibe_quant.shadow.tracker" in sys.modules,
        "runner": "vibe_quant.validation.runner" in sys.modules,
    }))
    """
)


def _probe(db: Path, cwd: Path) -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, str(db)],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return dict(json.loads(out.stdout.strip().splitlines()[-1]))


def test_no_rows_never_loads_tracker(tmp_path: Path) -> None:
    """A plain update with no shadow rows never imports the tracker or the validation runner."""
    db = _db(tmp_path / "empty.db")
    assert _probe(db, tmp_path) == {"rc": 0, "tracker": False, "runner": False}


def test_cli_lock_held_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from vibe_quant.shadow.__main__ import main

    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT"], {}))
    with shadow_update_lock(db):
        assert main(["update", "--db", str(db)]) == 2
    assert "shadow update already running; skipped" in capsys.readouterr().err


def test_cli_update_uses_db(hook: Hook, tmp_path: Path) -> None:
    from vibe_quant.shadow.__main__ import main

    db = _db(tmp_path / "cli.db", _row("rsi", ["BTCUSDT"], {}))
    assert main(["update", "--db", str(db)]) == 0
    assert hook.updates == [{"db": db.resolve(), "trigger": "cli", "failed": {}}]
    assert hook.notified == [db.resolve()]


def test_cli_job_id_records_skip(tmp_path: Path) -> None:
    """With ``--job-id`` a lock-held update completes the job as a skip (not a failure)."""
    from vibe_quant.jobs.manager import BacktestJobManager
    from vibe_quant.shadow.__main__ import main

    db = _db(tmp_path / "s.db", _row("rsi", ["BTCUSDT"], {}))
    jobs = BacktestJobManager(db)
    jobs.conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type, status) VALUES (-7, ?, 'shadow_update', 'running')",
        (os.getpid(),),
    )
    jobs.conn.commit()
    with shadow_update_lock(db):
        assert main(["update", "--db", str(db), "--job-id", "-7"]) == 2
    row = jobs.conn.execute(
        "SELECT status, error_message FROM background_jobs WHERE run_id = -7"
    ).fetchone()
    jobs.close()
    assert (row[0], row[1]) == ("completed", "skipped: shadow update already running")


@pytest.mark.simple_target(
    reason="R40: a dropped --state-db makes the API update advance the CLI default DB's "
    "shadow rows instead of the backend's; nothing else catches it"
)
def test_api_update_passes_state_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The API data-update job hands its own state DB to the hook (R40)."""
    import asyncio

    from vibe_quant.api.routers import data as data_router

    started: list[list[str]] = []

    class Jobs:
        db_path = tmp_path / "api.db"

        def start_job(self, job_id: int, job_type: str, command: list[str], log_file: str) -> int:
            started.append(command)
            return 4242

    monkeypatch.setattr(data_router, "log_dir", lambda: tmp_path)
    asyncio.run(data_router.start_update(Jobs()))  # type: ignore[arg-type]
    cmd = started[0]
    data_args = cmd[cmd.index("--") + 1 :]
    assert data_args == ["update", "--state-db", str(tmp_path / "api.db")]


def test_missing_state_db_refused(
    hook: Hook, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mistyped --state-db / --db refuses (rc 1) instead of creating an empty DB (rev N1)."""
    from vibe_quant.shadow.__main__ import main

    calls: list[object] = []
    monkeypatch.setattr(ingest, "update_all", lambda **kw: calls.append(kw) or {})
    missing = tmp_path / "typo" / "state.db"
    assert ingest.main(["update", "--symbols", "BTCUSDT", "--state-db", str(missing)]) == 1
    assert main(["update", "--db", str(missing)]) == 1
    assert not missing.exists() and not missing.parent.exists()
    assert calls == []  # nothing downloaded
    assert capsys.readouterr().err.count(f"state DB {missing} does not exist") == 2
