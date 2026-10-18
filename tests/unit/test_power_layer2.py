"""Power test L2 (T4a+T4b): scratch-catalog planted-edge builds, calibration, harvest.

``scripts/research/power/layer2.py`` plants a known edge into private copies of the
BTCUSDT bars and runs the real gate chain / real discovery on them. Rows R25-R32 of
``design-power.md`` §6 each have a named test here (R25/R27/R28 decision logic and the
screening worker on a tiny fixture catalog; the 6 real runs are the chief's).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import layer2  # noqa: E402


def test_planted_chromosome_dsl() -> None:
    """The planted rule: RSI(14) crosses_below 30 entry, crosses_above 60 exit, SL 4 / TP 8, LONG."""
    assert layer2.planted_dsl() == {
        "name": "genome_planted_rsi",
        "timeframe": "4h",
        "indicators": {
            "rsi_entry_0": {"type": "RSI", "period": 14},
            "rsi_exit_0": {"type": "RSI", "period": 14},
        },
        "entry_conditions": {"long": ["rsi_entry_0 crosses_below 30"]},
        "exit_conditions": {"long": ["rsi_exit_0 crosses_above 60"]},
        "stop_loss": {"type": "fixed_pct", "percent": 4.0},
        "take_profit": {"type": "fixed_pct", "percent": 8.0},
    }


# --- fixtures -------------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from typing import Any  # noqa: E402

import gatechain  # noqa: E402
import numpy as np  # noqa: E402
import plant  # noqa: E402
import pytest  # noqa: E402

from vibe_quant.data.archive import RawDataArchive  # noqa: E402
from vibe_quant.db.state_manager import StateManager  # noqa: E402
from vibe_quant.discovery.config_check import plan_windows  # noqa: E402

MIN_MS = 60_000
DAY0_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC
SCRIPT_DIR = Path(layer2.__file__).resolve().parent


def make_klines(days: int, seed: int = 5, sigma: float = 0.0012) -> plant.Klines:
    n = days * 1440
    rng = np.random.default_rng(seed)
    close = np.round(30_000.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n))), 1)
    open_ = np.concatenate(([30_000.0], close[:-1]))
    times = DAY0_MS + np.arange(n, dtype=np.int64) * MIN_MS
    return plant.Klines(
        open_time=times,
        open=open_,
        high=np.maximum(open_, close) + 0.5,
        low=np.minimum(open_, close) - 0.5,
        close=close,
        volume=rng.uniform(1.0, 10.0, n),
        close_time=times + 59_999,
    )


def make_source_archive(path: Path, days: int) -> None:
    arch = RawDataArchive(path)
    arch.insert_funding_rates(
        "BTCUSDT", [(DAY0_MS + i * 8 * 3_600_000, 0.0001, 30_000.0) for i in range(days * 3)], "t"
    )
    arch.insert_funding_intervals("BTCUSDT", [(DAY0_MS, 8)], "t")
    arch.close()


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 210-day scratch catalog written by the real ``plant.write_scratch``."""
    base = tmp_path_factory.mktemp("l2root")
    make_source_archive(base / "src.db", 210)
    plant.write_scratch(base / "root", make_klines(210), base / "src.db")
    return base / "root"


def make_data_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A miniature ``data/`` (BTCUSDT catalog files, raw archive) + a real-DB stand-in."""
    data = tmp_path / "data"
    bar = data / "catalog/data/bar/BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
    bar.mkdir(parents=True)
    (bar / "a.parquet").write_bytes(b"A" * 100)
    inst = data / "catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE"
    inst.mkdir(parents=True)
    (inst / "i.parquet").write_bytes(b"I" * 50)
    eth = data / "catalog/data/bar/ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
    eth.mkdir(parents=True)
    (eth / "e.parquet").write_bytes(b"E" * 10)
    (data / "archive").mkdir()
    make_source_archive(data / "archive/raw_data.db", 2)
    real_db = tmp_path / "real.db"
    conn = sqlite3.connect(real_db)
    conn.execute("CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, run_mode TEXT, created_at TEXT)")
    conn.executemany(
        "INSERT INTO backtest_runs (run_mode, created_at) VALUES (?, ?)",
        [("discovery", "2026-10-01 10:00:00"), ("screening", "2026-10-02 10:00:00"),
         ("discovery", "2026-10-03 10:00:00")],
    )
    conn.commit()
    conn.close()
    return data, real_db, data / "archive/raw_data.db"


# --- scratch DB pin (R26/R31) ---------------------------------------------------------------


@pytest.mark.parametrize("preset", ["unset", "real"])
def test_parent_pins_scratch_db_before_vibe_quant_import(preset: str) -> None:
    """The parent never keeps the real state DB: unset or real -> a scratch path, set at import."""
    env = {k: v for k, v in os.environ.items() if k != "VIBE_QUANT_DB"}
    if preset == "real":
        env["VIBE_QUANT_DB"] = str(layer2.REAL_DB)
    code = (
        "import sys, os; sys.path.insert(0, sys.argv[1]);"
        "import layer2;"
        "from vibe_quant.db.connection import DEFAULT_DB_PATH;"
        "print(os.environ['VIBE_QUANT_DB']); print(DEFAULT_DB_PATH.resolve()); print(layer2.REAL_DB)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT_DIR)], env=env, capture_output=True, text=True,
        check=True,
    ).stdout.split("\n")
    pinned, default_path, real = out[0], out[1], out[2]
    assert pinned != "" and Path(pinned).resolve() != Path(real)
    assert default_path == str(Path(pinned).resolve())  # vibe_quant saw the pin at import
    assert "layer2-parent-" in pinned


def test_child_env_forces_own_db_and_refuses_real(tmp_path: Path) -> None:
    db = tmp_path / "run.db"
    env = layer2.child_env(db, tmp_path / "logs")
    assert env["VIBE_QUANT_DB"] == str(db)
    assert env["VIBE_QUANT_LOG_DIR"] == str(tmp_path / "logs")
    with pytest.raises(layer2.Layer2Error, match="real state DB"):
        layer2.child_env(layer2.REAL_DB)
    with pytest.raises(layer2.Layer2Error, match="real state DB"):
        layer2.create_run(layer2.REAL_DB)


# --- R26 shared data untouched --------------------------------------------------------------


def test_snapshot_signatures_detect_any_change(tmp_path: Path) -> None:
    """R26: size/mtime_ns/sha256 of the BTCUSDT files + raw archive, discovery count + max id."""
    data, real_db, archive = make_data_tree(tmp_path)
    snap = layer2.take_snapshot(data, real_db, archive)
    assert sorted(snap["files"]) == [
        "archive/raw_data.db",
        "catalog/data/bar/BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL/a.parquet",
        "catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE/i.parquet",
    ]
    assert snap["discovery_rows"] == {"count": 2, "max_id": 3}
    assert snap["funding_source_count"] == 6
    assert layer2.compare_snapshot(snap, data, real_db, archive) == []

    # an untracked symbol may change freely
    (data / "catalog/data/bar/ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL/e.parquet").write_bytes(b"x")
    assert layer2.compare_snapshot(snap, data, real_db, archive) == []

    bar = data / "catalog/data/bar/BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL/a.parquet"
    st = bar.stat()
    bar.write_bytes(b"B" * 100)  # same size ...
    os.utime(bar, ns=(st.st_atime_ns, st.st_mtime_ns))  # ... same mtime: only the sha differs
    assert layer2.compare_snapshot(snap, data, real_db, archive) == [
        "file changed: catalog/data/bar/BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL/a.parquet"
    ]
    bar.write_bytes(b"A" * 100)
    os.utime(bar, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert layer2.compare_snapshot(snap, data, real_db, archive) == []
    os.utime(bar, ns=(st.st_atime_ns, st.st_mtime_ns + 1))  # mtime alone is a change
    assert len(layer2.compare_snapshot(snap, data, real_db, archive)) == 1
    os.utime(bar, ns=(st.st_atime_ns, st.st_mtime_ns))
    (bar.parent / "extra.parquet").write_bytes(b"n")
    assert any(p.startswith("new file") for p in layer2.compare_snapshot(snap, data, real_db, archive))


def test_new_discovery_row_is_reported_with_id_and_created_at(tmp_path: Path) -> None:
    """W-R3: another session's run shows up as a mismatch naming the new row."""
    data, real_db, archive = make_data_tree(tmp_path)
    snap = layer2.take_snapshot(data, real_db, archive)
    conn = sqlite3.connect(real_db)
    conn.execute(
        "INSERT INTO backtest_runs (run_mode, created_at) VALUES ('discovery', '2026-10-18 01:02:03')"
    )
    conn.commit()
    conn.close()
    problems = layer2.compare_snapshot(snap, data, real_db, archive)
    assert "new discovery row: id=4 created_at=2026-10-18 01:02:03" in problems
    assert any("discovery rows" in p for p in problems)


def test_sha_is_dropped_past_the_deadline_but_size_mtime_still_compare(tmp_path: Path) -> None:
    f = tmp_path / "big.db"
    f.write_bytes(b"z" * (20 << 20))
    sig = layer2.file_signature(f, sha_deadline_s=-1.0)
    assert sig["sha256"] is None and sig["size"] == 20 << 20
    assert layer2.file_signature(f)["sha256"] is not None
    assert layer2._sig_diff(sig, {**sig, "size": 1})
    assert not layer2._sig_diff(sig, {**sig, "sha256": "anything"})


def test_snapshot_and_verify_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, real_db, archive = make_data_tree(tmp_path)
    out = tmp_path / "out" / "layer2.json"
    base = ["--out", str(out), "--data-root", str(data), "--real-db", str(real_db),
            "--archive", str(archive)]
    assert layer2.main([*base, "verify-untouched"]) == 2  # no snapshot yet
    assert layer2.main([*base, "snapshot"]) == 0
    first = json.loads(out.read_text())["snapshot"]
    (data / "catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE/i.parquet").write_bytes(b"J" * 50)
    assert layer2.main([*base, "snapshot"]) == 0  # kept: the baseline is never silently moved
    assert json.loads(out.read_text())["snapshot"] == first
    capsys.readouterr()
    assert layer2.main([*base, "verify-untouched"]) == 3
    assert "R26 shared data CHANGED" in capsys.readouterr().out
    assert json.loads(out.read_text())["verify"]["ok"] is False
    (data / "catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE/i.parquet").write_bytes(b"I" * 50)
    os.utime(
        data / "catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE/i.parquet",
        ns=(1, first["files"]["catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE/i.parquet"]["mtime_ns"]),
    )
    assert layer2.main([*base, "verify-untouched"]) == 0


# --- R30 found ------------------------------------------------------------------------------


def test_found_criterion() -> None:
    """corr >= 0.5 AND S >= 0.5 x planted S; a mirror (corr -0.7) is not found but shows in max|corr|."""
    assert layer2.found_criterion(0.6, 1.2, 2.0) is True  # ratio 0.6
    assert layer2.found_criterion(0.4, 1.2, 2.0) is False
    assert layer2.found_criterion(-0.7, 1.2, 2.0) is False
    assert layer2.found_criterion(0.5, 1.0, 2.0) is True  # both edges inclusive
    assert layer2.found_criterion(0.4999, 1.0, 2.0) is False
    assert layer2.found_criterion(0.5, 0.9999, 2.0) is False
    assert layer2.found_criterion(None, 1.2, 2.0) is False  # undefined fails closed
    assert layer2.found_criterion(0.6, None, 2.0) is False
    assert layer2.found_criterion(0.6, 1.2, None) is False

    hit = layer2.assess_found([{"uid": "a", "corr": 0.6, "train_S": 1.2}], 2.0)
    assert hit["found"] is True and hit["max_abs_corr"] == 0.6
    assert hit["found_by"] == [{"uid": "a", "corr": 0.6, "train_S": 1.2}]
    weak = layer2.assess_found([{"uid": "b", "corr": 0.4, "train_S": 1.2}], 2.0)
    assert weak["found"] is False and weak["max_abs_corr"] == 0.4
    mirror = layer2.assess_found(
        [{"uid": "c", "corr": -0.7, "train_S": 1.2}, {"uid": "d", "corr": 0.1, "train_S": 3.0}], 2.0
    )
    assert mirror["found"] is False and mirror["max_abs_corr"] == 0.7
    assert layer2.assess_found([], 2.0) == {
        "found": False, "found_by": [], "max_abs_corr": None, "planted_train_S": 2.0,
    }


def test_signed_corr_and_ann_sharpe_use_common_days() -> None:
    a = {"first_day": 10, "values": [0.0, 1.0, 2.0, 3.0, 4.0]}
    b = {"first_day": 12, "values": [3.0, 2.0, 1.0, 0.0, 9.0]}  # days 12..16; common 12..14
    assert layer2.signed_corr(a, b) == pytest.approx(-1.0)
    assert layer2.signed_corr(a, {"first_day": 20, "values": [1.0, 2.0]}) is None
    assert layer2.signed_corr(a, {"first_day": 10, "values": [1.0] * 5}) is None  # flat
    assert layer2.signed_corr(None, a) is None
    # S: day 0 dropped, daily mean/std x sqrt(365)
    vals = [0.5] + [0.01, 0.03, 0.02, -0.01] * 15
    s = layer2.ann_sharpe({"first_day": 0, "values": vals})
    arr = np.array(vals[1:])
    assert s == pytest.approx(float(arr.mean() / arr.std(ddof=1)) * 365**0.5)
    assert layer2.ann_sharpe(None) is None
    assert layer2.ann_sharpe({"first_day": 0, "values": [0.5, 0.01, 0.03]}) is None  # too short


# --- R31 / R32 ------------------------------------------------------------------------------


def test_isolation_check_edges() -> None:
    ok = layer2.check_isolation({"prior_n": 0, "gate_n": 90, "raw": 90})
    assert ok == {"ok": True, "checked": True, "prior_n": 0, "gate_n": 90, "raw": 90}
    assert layer2.check_isolation({"prior_n": 1, "gate_n": 91, "raw": 90})["ok"] is False
    assert layer2.check_isolation({"prior_n": 0, "gate_n": 91, "raw": 90})["ok"] is False
    assert layer2.check_isolation({"prior_n": None, "gate_n": 90, "raw": 90})["ok"] is False
    assert layer2.check_isolation({})["checked"] is False


@pytest.mark.parametrize(
    ("row", "rc", "timed_out", "want"),
    [
        (("completed", None), 0, False, {"status": "completed"}),
        (("failed", "MissingBarDataError: no bars"), 1, False,
         {"status": "failed", "error": "MissingBarDataError: no bars"}),
        (("running", None), -9, False, {"status": "failed", "error": "exit code -9, run status 'running'"}),
        (("completed", None), 1, False, {"status": "failed", "error": "exit code 1, run status 'completed'"}),
        (("running", None), -9, True, {"status": "timeout", "error": "killed after 600s"}),
        (None, 1, False, {"status": "failed", "error": "run row missing (exit code 1)"}),
    ],
)
def test_classify_run_never_reports_a_failure_as_zero_champions(
    row: tuple[str, str | None] | None, rc: int, timed_out: bool, want: dict[str, str]
) -> None:
    got = layer2.classify_run(row, rc, timed_out, 600.0)
    assert got == want
    assert "champions" not in got


def _fake_command(script: str) -> Any:
    def command(run_id: int, db: Path, seed: int) -> list[str]:
        return [sys.executable, "-c", script, str(run_id), str(db), str(seed)]

    return command


FAKE_HEAD = (
    "import os, sqlite3, sys\n"
    "run_id, db, seed = int(sys.argv[1]), sys.argv[2], sys.argv[3]\n"
    "print('CWD=' + os.getcwd()); print('DB=' + os.environ['VIBE_QUANT_DB']); print('SEED=' + seed)\n"
    "con = sqlite3.connect(db)\n"
)


def test_run_discovery_isolation_and_polled_final_status(tmp_path: Path) -> None:
    """R31: the child runs with cwd = scratch root and its OWN DB; the record follows the run row."""
    root = tmp_path / "root"
    root.mkdir()
    done = FAKE_HEAD + (
        "con.execute(\"UPDATE backtest_runs SET status='completed' WHERE id=?\", (run_id,))\n"
        "con.commit()\n"
    )
    rec = layer2.run_discovery(root, tmp_path / "r1", 7, command_fn=_fake_command(done), poll_s=0.05)
    assert rec["status"] == "completed" and "error" not in rec and rec["seed"] == 7
    log = (tmp_path / "r1" / "run.log").read_text()
    assert f"CWD={root.resolve()}" in log
    assert f"DB={tmp_path / 'r1' / 'state.db'}" in log
    assert "SEED=7" in log
    assert layer2.REAL_DB.name not in log

    failed = FAKE_HEAD + (
        "con.execute(\"UPDATE backtest_runs SET status='failed', error_message=? WHERE id=?\","
        " ('MissingBarDataError: no bars', run_id))\ncon.commit()\nsys.exit(1)\n"
    )
    rec = layer2.run_discovery(root, tmp_path / "r2", 1, command_fn=_fake_command(failed), poll_s=0.05)
    assert rec["status"] == "failed" and rec["error"] == "MissingBarDataError: no bars"
    assert "champions" not in rec

    killed = FAKE_HEAD + "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"
    rec = layer2.run_discovery(root, tmp_path / "r3", 1, command_fn=_fake_command(killed), poll_s=0.05)
    assert rec["status"] == "failed" and rec["error"] == "exit code -9, run status 'pending'"


def test_run_over_timeout_is_killed_and_marked_timeout(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    hang = FAKE_HEAD + "import time\ntime.sleep(60)\n"
    t0 = time.monotonic()
    rec = layer2.run_discovery(
        root, tmp_path / "r", 1, timeout_s=1.0, command_fn=_fake_command(hang), poll_s=0.05
    )
    assert time.monotonic() - t0 < 30
    assert rec["status"] == "timeout" and rec["error"] == "killed after 1s"


def _save_notes(db: Path, run_id: int, notes: dict[str, Any]) -> None:
    state = StateManager(db)
    state.save_backtest_result(
        run_id,
        {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
         "total_trades": 0, "skewness": 0.0, "kurtosis": 3.0, "execution_time_seconds": 1.0,
         "notes": json.dumps(notes)},
    )
    state.close()


def test_harvest_run_reports_champions_stages_and_isolation(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    run_id = layer2.create_run(db)
    dsl = {"name": "genome_abc", "timeframe": "4h"}
    _save_notes(db, run_id, {
        "evaluated": 90, "outcome": None,
        "dsr_trials": {"raw": 90, "gate_n": 90, "prior_n": 0},
        "top_strategies": [{"dsl": dsl, "dsr": {"sharpe_daily": 0.1}}],
        "guardrail_rejections": [
            {"uid": "u1", "stage": "guardrails", "dsl": dsl},
            {"uid": "u2", "stage": "holdout", "dsl": dsl},
            {"uid": "u3", "stage": "guardrails", "dsl": dsl},
        ],
    })
    h = layer2.harvest_run(db, run_id)
    assert h["champions"] == 1
    assert h["rejections_by_stage"] == {"guardrails": 2, "holdout": 1}
    assert h["r31"]["ok"] is True and h["evaluated"] == 90
    assert [(c["source"], c["uid"]) for c in h["candidates"]] == [
        ("champion", "genome_abc"), ("rejected", "u1"), ("rejected", "u2"), ("rejected", "u3"),
    ]

    run2 = layer2.create_run(db)
    _save_notes(db, run2, {
        "evaluated": 90, "outcome": "no_viable_strategies", "top_strategies": [],
        "dsr_trials": {"raw": 90, "gate_n": 94, "prior_n": 4},
    })
    h2 = layer2.harvest_run(db, run2)
    assert h2["champions"] == 0 and h2["outcome"] == "no_viable_strategies"
    assert h2["r31"]["ok"] is False


def test_posthoc_p_rereads_the_stored_record_at_each_n() -> None:
    values = list(np.random.default_rng(11).normal(0.0005, 0.01, 875))
    _ok, record = gatechain.dsr_verdict(values, 167)
    out = layer2.posthoc_p([{"uid": "x", "dsr": record}, {"uid": "y", "dsr": None}], (1, 167))
    assert out["x"][1] == pytest.approx(0.7266370087847579)  # R29 / R1 at N=167
    assert out["x"][0] == pytest.approx(0.01786782317136858)  # R1 at N=1
    assert out["y"] == [None, None]


# --- R27 calibration ------------------------------------------------------------------------


def test_acceptance_edges() -> None:
    """btc2.0: |full S - 2.0| <= 0.15; btc5.0: TRAIN S >= 4.5; btc0 always accepted.

    A planted variant is accepted only when its kappa build converged (R23), so every
    record here carries ``converged: True``.
    """
    assert layer2.accepts("btc2.0", {"full_S": 2.0, "converged": True})
    assert layer2.accepts("btc2.0", {"full_S": 2.14, "converged": True})
    assert layer2.accepts("btc2.0", {"full_S": 1.86, "converged": True})
    assert not layer2.accepts("btc2.0", {"full_S": 2.16, "converged": True})
    assert not layer2.accepts("btc2.0", {"full_S": 1.84, "converged": True})
    assert not layer2.accepts("btc2.0", {"train_S": 2.0, "converged": True})  # wrong series
    assert not layer2.accepts("btc2.0", {"full_S": None, "converged": True})
    assert layer2.accepts("btc5.0", {"train_S": 4.5, "converged": True})
    assert not layer2.accepts("btc5.0", {"train_S": 4.4999999, "converged": True})
    assert layer2.accepts("btc5.0", {"train_S": 9.0, "converged": True})
    assert not layer2.accepts("btc5.0", {"full_S": 9.0, "converged": True})
    assert layer2.accepts("btc0", {})


def test_acceptance_requires_a_converged_build() -> None:
    """A non-converged kappa (|l_end| > KAPPA_TOL) fails closed, whatever S says (R23).

    The exact fail-open btc5.0 shipped: build b1 had train_S 5.4265 but converged False
    and was recorded as accepted.
    """
    assert not layer2.accepts("btc5.0", {"train_S": 5.4, "converged": False})
    assert not layer2.accepts("btc2.0", {"full_S": 2.0, "converged": False})
    assert layer2.accepts("btc5.0", {"train_S": 5.4, "converged": True})
    assert layer2.accepts("btc2.0", {"full_S": 2.0, "converged": True})
    # a missing or merely truthy `converged` is not True: fail closed, never `if value`
    assert not layer2.accepts("btc5.0", {"train_S": 5.4})
    assert not layer2.accepts("btc2.0", {"full_S": 2.0})
    assert not layer2.accepts("btc5.0", {"train_S": 5.4, "converged": 1})
    # btc0 has no plant: no kappa to converge, always accepted
    assert layer2.accepts("btc0", {})
    assert layer2.accepts("btc0", {"converged": False})


def test_next_delta_secant_proportional_and_clamped() -> None:
    def b(delta: float, full_s: float) -> dict[str, Any]:
        return {"delta": delta, "full_S": full_s}

    assert layer2.next_delta("btc2.0", [], 1.5e-5) == 1.5e-5
    # one point: S proportional to delta -> 1e-5 * 2.0 / 1.0
    assert layer2.next_delta("btc2.0", [b(1e-5, 1.0)], 9.9) == pytest.approx(2e-5)
    # clamp: a tiny S cannot send delta more than 4x up (or 4x down)
    assert layer2.next_delta("btc2.0", [b(1e-5, 0.01)], 9.9) == pytest.approx(4e-5)
    assert layer2.next_delta("btc2.0", [b(1e-5, 100.0)], 9.9) == pytest.approx(2.5e-6)
    # secant through (1e-5, 1.0) and (3e-5, 3.0): S(delta) = 1e5 * delta -> S=2 at 2e-5
    assert layer2.next_delta("btc2.0", [b(1e-5, 1.0), b(3e-5, 3.0)], 9.9) == pytest.approx(2e-5)
    # a non-positive S doubles delta; an equal-S pair falls back to proportional
    assert layer2.next_delta("btc2.0", [b(1e-5, -0.4)], 9.9) == pytest.approx(2e-5)
    assert layer2.next_delta("btc2.0", [b(1e-5, 1.0), b(2e-5, 1.0)], 9.9) == pytest.approx(4e-5)
    # btc5.0 calibrates on the TRAIN series toward 5.0
    assert layer2.next_delta("btc5.0", [{"delta": 1e-5, "train_S": 2.5}], 9.9) == pytest.approx(2e-5)


def test_build_one_plants_writes_scratch_and_reports_l_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = 40
    make_source_archive(tmp_path / "src.db", days)
    k = make_klines(days)
    start = DAY0_MS + 2 * 86_400_000
    end = DAY0_MS + (days - 2) * 86_400_000
    monkeypatch.setattr(layer2, "plant_window_ms", lambda: (start, end))
    monkeypatch.setattr(layer2, "measure_build", lambda root, db: {"train_S": 5.5, "full_S": 2.0})
    scratch = tmp_path / "scratch"
    rec = layer2.build_one("btc2.0", 1, 5e-5, k, tmp_path / "src.db", scratch, scratch / "p.db")
    root = scratch / "btc2.0-b1"
    assert rec["root"] == str(root) and rec["k"] == 1 and rec["delta"] == 5e-5
    assert rec["n_triggers"] >= 1 and rec["drift_minutes"] > 0
    assert rec["kappa"] > 0 and abs(rec["l_end"]) <= 0.02
    assert rec["converged"] == (abs(rec["l_end"]) <= 0.005)
    assert rec["price_ratio_gt_3"] is (rec["price_ratio"] > 3.0)
    assert rec["train_S"] == 5.5 and rec["full_S"] == 2.0
    from vibe_quant.data.catalog import CatalogManager

    cat = CatalogManager(root / "data" / "catalog")
    assert cat.get_bar_count("BTCUSDT", "1m") == days * 1440
    assert cat.get_bar_count("BTCUSDT", "4h") == days * 6
    layer2.assert_scratch_root(root, tmp_path / "data")
    # a stale root of a crashed attempt is replaced, never reused
    (root / "stale.txt").write_text("x")
    layer2.build_one("btc2.0", 1, 0.0, k, tmp_path / "src.db", scratch, scratch / "p.db")
    assert not (root / "stale.txt").exists()
    # delta 0 is the identity: no kappa, no triggers' drift
    again = layer2.build_one("btc0", 1, 0.0, k, tmp_path / "src.db", scratch, scratch / "p.db")
    assert (again["kappa"], again["l_end"], again["converged"]) == (0.0, 0.0, True)


def test_build_one_refines_kappa_only_past_kappa_tol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build asks solve_kappa for the phase-2 search gated at KAPPA_TOL, the same threshold
    that marks it non-converged, so a build that already met the bar keeps its phase-1 kappa."""
    days = 40
    make_source_archive(tmp_path / "src.db", days)
    k = make_klines(days)
    start = DAY0_MS + 2 * 86_400_000
    end = DAY0_MS + (days - 2) * 86_400_000
    monkeypatch.setattr(layer2, "plant_window_ms", lambda: (start, end))
    monkeypatch.setattr(layer2, "measure_build", lambda root, db: {"train_S": 5.5, "full_S": 2.0})
    seen: list[dict[str, Any]] = []
    real = plant.solve_kappa

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(plant, "solve_kappa", spy)
    scratch = tmp_path / "scratch"
    layer2.build_one("btc2.0", 1, 5e-5, k, tmp_path / "src.db", scratch, scratch / "p.db")
    assert len(seen) == 1 and seen[0]["refine_above"] == 0.005


def test_assert_scratch_root_refuses_the_shared_catalog(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "catalog").mkdir(parents=True)
    root = tmp_path / "scratch" / "r"
    (root / "data").mkdir(parents=True)
    with pytest.raises(layer2.Layer2Error, match="no catalog"):
        layer2.assert_scratch_root(root, data)
    (root / "data" / "catalog").symlink_to(data / "catalog")
    with pytest.raises(layer2.Layer2Error, match="shared catalog"):
        layer2.assert_scratch_root(root, data)
    with pytest.raises(layer2.Layer2Error, match="shared catalog"):
        layer2.assert_scratch_root(tmp_path, data)  # root/data/catalog IS the shared catalog


def _stub_builder(
    monkeypatch: pytest.MonkeyPatch, trains: list[float], fulls: list[float], ratio: float = 1.2
) -> list[float]:
    """Replace the heavy steps of ``cmd_build``; returns the deltas it was asked to build."""
    deltas: list[float] = []

    def fake_load(_archive: Path) -> object:
        return object()

    def fake_build(
        variant: str, k: int, delta: float, _klines: object, _archive: Path, scratch: Path, _db: Path
    ) -> dict[str, Any]:
        deltas.append(delta)
        root = layer2.build_root(scratch, variant, k)
        (root / "data" / "catalog").mkdir(parents=True, exist_ok=True)
        return {
            "k": k, "root": str(root), "delta": delta, "kappa": 0.0, "l_end": 0.0,
            "converged": True, "n_triggers": 9, "price_ratio": ratio, "price_ratio_gt_3": ratio > 3.0,
            "train_S": trains[k - 1], "full_S": fulls[k - 1],
        }

    monkeypatch.setattr(plant, "load_klines", fake_load)
    monkeypatch.setattr(layer2, "build_one", fake_build)
    return deltas


def _cli(tmp_path: Path) -> list[str]:
    return ["--out", str(tmp_path / "o" / "layer2.json"), "--data-root", str(tmp_path / "data"),
            "--real-db", str(tmp_path / "none.db"), "--archive", str(tmp_path / "src.db")]


def test_build_calibrates_within_four_builds_keeps_only_the_accepted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deltas = _stub_builder(monkeypatch, trains=[0, 0, 0, 0], fulls=[1.0, 1.5, 2.05, 9.9])
    base = _cli(tmp_path)
    assert layer2.main([*base, "build", "--variant", "btc2.0"]) == 0
    assert len(deltas) == 3  # stops at the first accepted build
    assert deltas[0] == 1.5e-5 and deltas[1] == pytest.approx(3e-5)
    state = json.loads((tmp_path / "o" / "layer2.json").read_text())
    v = state["variants"]["btc2.0"]
    assert v["status"] == "accepted" and v["accepted"]["k"] == 3
    assert [b["accepted"] for b in v["builds"]] == [False, False, True]
    scratch = tmp_path / "o" / "scratch"
    assert sorted(p.name for p in scratch.iterdir()) == ["btc2.0-b3"]
    # idempotent: a rerun builds nothing
    assert layer2.main([*base, "build", "--variant", "btc2.0"]) == 0
    assert len(deltas) == 3


def test_build_gives_up_after_four_builds_and_resumes_from_the_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deltas = _stub_builder(monkeypatch, trains=[0] * 8, fulls=[0.1, 0.2, 0.3, 0.4, 2.0, 2.0, 2.0, 2.0])
    base = _cli(tmp_path)
    assert layer2.main([*base, "build", "--variant", "btc2.0"]) == 2
    assert len(deltas) == 4
    state = json.loads((tmp_path / "o" / "layer2.json").read_text())
    assert state["variants"]["btc2.0"]["status"] == "not_calibrated"
    assert "accepted" not in state["variants"]["btc2.0"]
    # btc0 is a single build with no calibration target
    assert layer2.main([*base, "build", "--variant", "btc0"]) == 0
    assert len(deltas) == 5 and deltas[4] == 0.0


# --- R25 / R28 ------------------------------------------------------------------------------


def test_compare_exact_is_bitwise() -> None:
    want = layer2.EXPECTED_239
    got = {k: v for k, v in want.items()}
    assert layer2.compare_exact(got, want) == []
    got["sharpe"] = 0.7974716442710238  # one ulp off
    assert layer2.compare_exact(got, want) == [
        "sharpe: got 0.7974716442710238 want 0.7974716442710239"
    ]
    got["sharpe"], got["trades"] = want["sharpe"], 68
    assert layer2.compare_exact(got, want) == ["trades: got 68 want 67"]
    assert layer2.EXPECTED_PLANTED_W3 == {
        "sharpe": -0.3729222589307, "trades": 142, "return": -0.1734565196672771,
    }


def _chain_result(**over: Any) -> dict[str, Any]:
    ns = (1, 7, 9, 11, 12, 167, 752, 3901, 22121)
    res: dict[str, Any] = {
        "floor": True,
        "marginals": dict.fromkeys(
            ("min_trades", "min_return", "complexity", "bootstrap", "buy_hold", "holdout",
             "cross_window", "wfa"), True),
        "dsr": {str(n): True for n in ns},
        "chain_full": {str(n): True for n in ns},
        "chain_recent": {str(n): True for n in ns},
    }
    res.update(over)
    return res


def test_positive_control_failures_name_every_failed_gate() -> None:
    assert layer2.positive_control_failures(_chain_result()) == []
    assert layer2.positive_control_failures(_chain_result(floor=False)) == ["floor"]
    marg = _chain_result()["marginals"]
    marg["wfa"] = False
    assert layer2.positive_control_failures(_chain_result(marginals=marg)) == ["wfa"]
    dsr = _chain_result()["dsr"]
    dsr["167"] = False
    chain_full = _chain_result()["chain_full"]
    chain_full["167"] = False
    assert layer2.positive_control_failures(_chain_result(dsr=dsr, chain_full=chain_full)) == [
        "dsr[167]", "chain_full[167]",
    ]
    # an N outside the positive-control set does not fail it ...
    dsr = _chain_result()["dsr"]
    dsr["7"] = False
    assert layer2.positive_control_failures(_chain_result(dsr=dsr)) == []
    # ... but a missing key or empty marginals fail closed
    dsr = _chain_result()["dsr"]
    del dsr["22121"]
    assert layer2.positive_control_failures(_chain_result(dsr=dsr)) == ["dsr[22121]"]
    assert "marginals missing" in layer2.positive_control_failures(_chain_result(marginals={}))
    assert layer2.positive_control_failures({}) != []


@pytest.mark.parametrize(
    ("variant", "fail", "code"),
    [("btc5.0", True, 3), ("btc5.0", False, 0), ("btc2.0", True, 0)],
)
def test_direct_chain_exit_code_stops_only_on_a_failed_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, fail: bool, code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "o" / "layer2.json"
    root = tmp_path / "scratch" / "r"
    (root / "data" / "catalog").mkdir(parents=True)
    layer2.update_state(out, lambda s: layer2._variant(s, variant).__setitem__("accepted", {"root": str(root)}))
    result = _chain_result(floor=not fail)
    # evaluate_candidate's output has int N keys; the CLI must cope after the JSON round trip
    result["dsr"] = {int(k): v for k, v in result["dsr"].items()}
    result["chain_recent"] = {int(k): v for k, v in result["chain_recent"].items()}
    result["chain_full"] = {int(k): v for k, v in result["chain_full"].items()}
    monkeypatch.setattr(
        layer2, "run_worker",
        lambda kind, r, payload, db, **kw: {"fitness": {"train_S": 5.0}, "holdout": {}, "result": result},
    )
    assert layer2.main([*_cli(tmp_path), "direct-chain", "--variant", variant]) == code
    rec = json.loads(out.read_text())["variants"][variant]["direct_chain"]
    assert rec["failures"] == (["floor"] if fail else [])
    text = capsys.readouterr().out
    assert ("R28 positive control FAILED: floor" in text) == (fail and variant == "btc5.0")
    # recorded -> a rerun reports the same verdict without running the chain again
    monkeypatch.setattr(layer2, "run_worker", lambda *a, **k: pytest.fail("chain reran"))
    assert layer2.main([*_cli(tmp_path), "direct-chain", "--variant", variant]) == code


def test_check_btc0_cli_passes_only_on_bit_identical_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o" / "layer2.json"
    root = tmp_path / "scratch" / "btc0-b1"
    (root / "data" / "catalog").mkdir(parents=True)
    layer2.update_state(out, lambda s: layer2._variant(s, "btc0").__setitem__("accepted", {"root": str(root)}))
    monkeypatch.setattr(layer2, "read_strategy_239", lambda db: {"name": "s239"})
    good = {
        "s239": {"tag": "s239", "sharpe": 0.7974716442710239, "trades": 67,
                 "return": 0.159784359286041, "profit_factor": 1.6359612484989312,
                 "max_drawdown": 0.05716872229767501},
        "planted_w3": {"tag": "planted_w3", "sharpe": -0.3729222589307, "trades": 142,
                       "return": -0.1734565196672771, "profit_factor": 1.0, "max_drawdown": 0.3},
    }
    seen: list[list[str]] = []

    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> list[dict[str, Any]]:
        seen.append([j["tag"] for j in payload["jobs"]])
        assert r == root and payload["jobs"][1]["dsl"] == layer2.planted_dsl()
        assert payload["jobs"][0]["start"] == "2024-01-01" and payload["jobs"][0]["end"] == "2026-03-17"
        assert (payload["jobs"][1]["start"], payload["jobs"][1]["end"]) == ("2023-06-01", "2026-06-01")
        return [good["s239"], good["planted_w3"]]

    monkeypatch.setattr(layer2, "run_worker", worker)
    assert layer2.main([*_cli(tmp_path), "check-btc0"]) == 0
    assert seen == [["s239", "planted_w3"]]
    assert json.loads(out.read_text())["variants"]["btc0"]["check_btc0"]["passed"] is True

    good["planted_w3"]["trades"] = 141
    assert layer2.main([*_cli(tmp_path), "--force", "check-btc0"]) == 3
    rec = json.loads(out.read_text())["variants"]["btc0"]["check_btc0"]
    assert rec["passed"] is False and rec["diffs"]["planted_w3"] == ["trades: got 141 want 142"]


# --- screening worker + chain worker on a real (tiny) scratch catalog -----------------------


def test_screen_worker_runs_in_the_scratch_root(tiny_root: Path, tmp_path: Path) -> None:
    jobs = [
        {"tag": "a", "dsl": layer2.planted_dsl(), "start": "2024-01-10", "end": "2024-03-01",
         "collect": True, "bh": True},
        {"tag": "b", "dsl": layer2.planted_dsl(), "start": "2024-01-10", "end": "2024-03-01"},
    ]
    a, b = layer2.run_worker("screen", tiny_root, {"jobs": jobs}, tmp_path / "w.db")
    assert a["tag"] == "a" and a["trades"] > 0
    assert len(a["mtm"]["values"]) == 51 and a["mtm"]["first_day"] == 19732  # 2024-01-10
    assert isinstance(a["bh_daily_sharpe"], float)
    assert b["mtm"] is None and b["bh_daily_sharpe"] is None
    assert (b["sharpe"], b["trades"], b["return"]) == (a["sharpe"], a["trades"], a["return"])
    assert layer2.ann_sharpe(a["mtm"]) is not None
    # a worker started outside the scratch root refuses to run (cwd seam)
    with pytest.raises(layer2.Layer2Error, match="not the scratch root"):
        layer2.worker_screen({"root": str(tiny_root), "jobs": []})


def test_chain_worker_scores_the_planted_rule_through_every_real_gate(
    tiny_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = plan_windows("2024-01-05", "2024-07-20", train_test_split=0.8, eval_windows=2,
                        cross_window_months=(1,), wfa_oos_step_days=30)
    base = gatechain.settings_for("W3")
    cfg = replace(
        base.config, start_date=plan.train_start, end_date=plan.train_end,
        holdout_start_date=str(plan.holdout_start), holdout_end_date=str(plan.holdout_end),
        cross_window_months=[1], wfa_oos_step_days=30,
    )
    tiny = gatechain.ChainSettings(name="TINY", config=cfg, plan=plan, symbols=("BTCUSDT",))
    monkeypatch.setattr(gatechain, "settings_for", lambda name: tiny)
    monkeypatch.chdir(tiny_root)
    out = layer2.worker_chain({"root": str(tiny_root)})
    result = json.loads(json.dumps(out["result"], default=layer2._json_default))
    assert set(result["marginals"]) == {
        "min_trades", "min_return", "complexity", "bootstrap", "buy_hold", "holdout",
        "cross_window", "wfa",
    }
    assert out["holdout"]["total_trades"] > 0
    assert set(result["dsr"]) == {str(n) for n in gatechain.DSR_TRIALS}  # string keys after JSON
    # the 4-month fixture cannot pass the gates; each failing one is named
    fails = layer2.positive_control_failures(result)
    assert "min_trades" in fails and "floor" in fails


# --- discover / harvest CLI + verdict lines -------------------------------------------------


def test_discover_cli_records_a_failed_run_and_never_zero_champions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "o" / "layer2.json"
    root = tmp_path / "o" / "scratch" / "btc2.0-b2"
    (root / "data" / "catalog").mkdir(parents=True)
    layer2.update_state(out, lambda s: layer2._variant(s, "btc2.0").__setitem__("accepted", {"root": str(root)}))
    failed = FAKE_HEAD + (
        "con.execute(\"UPDATE backtest_runs SET status='failed', error_message=? WHERE id=?\","
        " ('MissingBarDataError: no 1m bars', run_id))\ncon.commit()\nsys.exit(1)\n"
    )
    monkeypatch.setattr(layer2, "discovery_command", _fake_command(failed))
    assert layer2.main([*_cli(tmp_path), "discover", "--variant", "btc2.0", "--seed", "2"]) == 0
    rec = json.loads(out.read_text())["variants"]["btc2.0"]["runs"]["2"]
    assert rec["status"] == "failed" and rec["error"] == "MissingBarDataError: no 1m bars"
    assert "champions" not in rec and "harvest" not in rec
    assert Path(rec["db"]) == tmp_path / "o" / "scratch" / "runs" / "btc2.0-seed2" / "state.db"
    # idempotent: recorded -> no second launch
    monkeypatch.setattr(layer2, "discovery_command", lambda *a: pytest.fail("relaunched"))
    assert layer2.main([*_cli(tmp_path), "discover", "--variant", "btc2.0", "--seed", "2"]) == 0
    # harvest skips it and the V5 line names the failure instead of 0 champions
    capsys.readouterr()
    assert layer2.main([*_cli(tmp_path), "harvest", "--variant", "btc2.0"]) == 0
    text = capsys.readouterr().out
    assert "V5 btc2.0: not found | seed 2: failed (MissingBarDataError: no 1m bars)" in text
    assert "champions=0" not in text


def test_discovery_command_flags_match_the_design() -> None:
    cmd = layer2.discovery_command(5, Path("/x/s.db"), 2)
    assert cmd[cmd.index("--seed") + 1] == "2" and cmd[cmd.index("--db") + 1] == "/x/s.db"
    flags = dict(zip(cmd[::2], cmd[1::2], strict=False))
    joined = " ".join(cmd)
    for needle in (
        "--indicator-pool ROC,DONCHIAN,RSI,CCI,NATR", "--population-size 12", "--max-generations 4",
        "--eval-windows 2", "--train-test-split 0.8", "--direction both", "--max-workers 3",
        "--timeframe 4h", "--start-date 2023-06-01", "--end-date 2026-06-01",
    ):
        assert needle in joined, needle
    assert "--mock" not in cmd and flags


def test_verdict_lines_v4_and_v5() -> None:
    chain: dict[str, Any] = {"failures": [], "result": {}}
    state = {"variants": {
        "btc5.0": {"direct_chain": chain, "runs": {
            "1": {"status": "completed", "harvest": {"champions": 0, "found": {"found": True, "max_abs_corr": 0.9}}},
            "2": {"status": "timeout", "error": "killed after 600s"}}},
        "btc2.0": {"direct_chain": {"failures": ["dsr[167]"], "result": {}}},
        "btc0": {"runs": {"1": {"status": "completed", "harvest": {"champions": 0, "found": {"found": False, "max_abs_corr": 0.2}}}}},
    }}
    lines = layer2.verdict_lines(state)
    assert lines[0] == "V4 btc5.0 direct chain [positive control]: PASS"
    assert lines[1] == "V4 btc2.0 direct chain [report only]: FAIL (dsr[167])"
    assert lines[2] == "V5 btc2.0: not found | no runs"
    assert lines[3] == (
        "V5 btc5.0: FOUND | seed 1: champions=0 found=True max|corr|=0.9; "
        "seed 2: timeout (killed after 600s)"
    )
    assert lines[4] == "V5 btc0 (null, info): seed 1: champions=0 found=False max|corr|=0.2"


# --- round 2: relative paths, rebuild invalidation, price ratio (W-R6) ----------------------


def test_relative_paths_survive_cwd_change_in_real_children(
    tiny_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default/relative --out: the screening worker and the discovery child (cwd = root) still
    find their root and open their DB, also with a root recorded relative in the state JSON."""
    monkeypatch.chdir(tiny_root.parent)
    rel_root = Path(tiny_root.name)
    jobs = [{"tag": "a", "dsl": layer2.planted_dsl(), "start": "2024-01-10", "end": "2024-02-01"}]
    res = layer2.run_worker("screen", rel_root, {"jobs": jobs}, Path("w.db"))
    assert res[0]["tag"] == "a"

    out = Path("o") / "layer2.json"
    layer2.update_state(
        out, lambda s: layer2._variant(s, "btc2.0").__setitem__("accepted", {"root": str(rel_root)})
    )
    base = ["--out", str(out), "--data-root", "data", "--real-db", "none.db", "--archive", "src.db"]
    assert layer2.main([*base, "discover", "--variant", "btc2.0", "--seed", "1", "--timeout", "120"]) == 0
    rec = json.loads(out.read_text())["variants"]["btc2.0"]["runs"]["1"]
    log = Path(rec["log"]).read_text()
    assert "unable to open database" not in log and "unable to open database" not in rec.get("error", "")
    assert rec["run_id"] == 1 and "run status 'pending'" not in rec.get("error", "")
    assert Path(rec["db"]).is_absolute() and Path(rec["root"]).is_absolute()


def _seed_dependents(out: Path, variant: str, root: Path) -> None:
    def seed(s: dict[str, Any]) -> None:
        v = layer2._variant(s, variant)
        v.update(
            accepted={"root": str(root), "k": 1, "train_S": 5.0},
            check_btc0={"passed": True},
            direct_chain={"failures": [], "result": {}},
            runs={"1": {"status": "completed", "harvest": {"found": {"found": True}}}},
            status="accepted",
        )

    layer2.update_state(out, seed)


def test_force_rebuild_drops_every_dependent_result_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o" / "layer2.json"
    old_root = tmp_path / "o" / "scratch" / "btc0-b1"
    (old_root / "data" / "catalog").mkdir(parents=True)
    _seed_dependents(out, "btc0", old_root)
    deltas = _stub_builder(monkeypatch, trains=[0] * 4, fulls=[0] * 4)
    assert layer2.main([*_cli(tmp_path), "--force", "build", "--variant", "btc0"]) == 0
    v = json.loads(out.read_text())["variants"]["btc0"]
    assert deltas == [0.0] and v["accepted"]["k"] == 1
    assert not {"check_btc0", "direct_chain", "runs"} & set(v)  # nothing cached for the new plant
    # a rebuild that fails still leaves no stale dependents behind
    _seed_dependents(out, "btc2.0", old_root)
    _stub_builder(monkeypatch, trains=[0] * 4, fulls=[0.1] * 4)
    assert layer2.main([*_cli(tmp_path), "--force", "build", "--variant", "btc2.0"]) == 2
    v = json.loads(out.read_text())["variants"]["btc2.0"]
    assert "accepted" not in v and not {"check_btc0", "direct_chain", "runs"} & set(v)


def test_rebuild_then_downstream_steps_run_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o" / "layer2.json"
    root = tmp_path / "o" / "scratch" / "btc5.0-b1"
    (root / "data" / "catalog").mkdir(parents=True)
    _seed_dependents(out, "btc5.0", root)
    _stub_builder(monkeypatch, trains=[9.0] * 4, fulls=[0] * 4)
    assert layer2.main([*_cli(tmp_path), "--force", "build", "--variant", "btc5.0"]) == 0
    calls: list[str] = []
    result = _chain_result()

    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        calls.append(kind)
        return {"fitness": {"train_S": 9.0}, "holdout": {}, "result": result}

    monkeypatch.setattr(layer2, "run_worker", worker)
    assert layer2.main([*_cli(tmp_path), "direct-chain", "--variant", "btc5.0"]) == 0
    assert calls == ["chain"]
    rec = json.loads(out.read_text())["variants"]["btc5.0"]["direct_chain"]
    assert rec["fitness"]["train_S"] == 9.0
    assert "runs" not in json.loads(out.read_text())["variants"]["btc5.0"]


def test_price_ratio_over_3x_fails_closed_and_blocks_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "o" / "layer2.json"
    deltas = _stub_builder(monkeypatch, trains=[5.0] * 4, fulls=[2.0] * 4, ratio=4.0)
    assert layer2.main([*_cli(tmp_path), "build", "--variant", "btc5.0"]) == 2
    assert len(deltas) == 1  # no further builds on a failed-closed variant
    v = json.loads(out.read_text())["variants"]["btc5.0"]
    assert v["status"] == "failed_price_ratio" and "accepted" not in v
    assert "price ratio 4.00 > 3x" in capsys.readouterr().out
    assert not (tmp_path / "o" / "scratch" / "btc5.0-b1").exists()
    # a rerun without --force stays failed; downstream refuses
    assert layer2.main([*_cli(tmp_path), "build", "--variant", "btc5.0"]) == 2
    assert len(deltas) == 1
    monkeypatch.setattr(layer2, "run_worker", lambda *a, **k: pytest.fail("ran downstream"))
    for cmd in (["direct-chain", "--variant", "btc5.0"], ["discover", "--variant", "btc5.0", "--seed", "1"]):
        assert layer2.main([*_cli(tmp_path), *cmd]) == 2
    # exactly 3.0 is still allowed
    _stub_builder(monkeypatch, trains=[5.0] * 4, fulls=[2.0] * 4, ratio=3.0)
    assert layer2.main([*_cli(tmp_path), "--force", "build", "--variant", "btc5.0"]) == 0
