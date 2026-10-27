"""Tests for scripts/agents/wfa_null.py — the no-edge pass rate of the rolling WFA gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _AGENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


burst = _load("burst")
wfa_null = _load("wfa_null")

# 184 days: K=6 -> 30d x 6, K=4 -> 46d x 4, K=3 -> 61d x 3
CELL: dict[str, Any] = {
    "symbol": "ZECUSDT", "timeframe": "1h", "start": "2025-07-01",
    "end": "2026-01-01", "leverage": 3, "pair": ["DONCHIAN", "ROC"],
    "direction": "long", "seed": 1,
}


def _base(k: int, cell: dict[str, Any] | None = None, key: str = "k", ledger: str = "L") -> dict[str, Any]:
    return {"key": key, "ledger": ledger, "K": k, "cell": dict(cell or CELL), "next_win": False}


def _stub_factory(rets: list[float], trades: int = 100) -> Any:
    """A factory handing out the next window return in call order (the gate asks per window)."""
    order = iter(range(len(rets)))

    def factory(_start: str, _end: str) -> Any:
        ret = rets[next(order)]

        def fn(_chrom: Any) -> dict[str, float | int]:
            return {"total_return": ret, "total_trades": trades, "sharpe_ratio": 1.0,
                    "max_drawdown": 0.1, "profit_factor": 1.5}

        return fn

    return factory


def _winner(key: str) -> dict[str, Any]:
    return {"key": key, "cell": dict(CELL), "best": {"chromosome": {"families": []}}}


def _rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


NEXT = {"start": "2026-01-01", "end": "2026-07-01", "ret": -0.1, "trades": 60}


def test_population_keeps_only_complete_failed_transfers() -> None:
    rows = [_winner(k) for k in ("kept", "gone", "failed", "no-next", "no-symbols")]
    tmap = {
        "kept": {"key": "kept", "symbols": {"ETHUSDT": {"ret": -0.2, "trades": 60}}, "next": NEXT},
        "gone": {"key": "gone", "symbols": {"ETHUSDT": {"ret": 0.2, "trades": 60}},
                 "next": {**NEXT, "ret": 0.2}},
        "failed": {"key": "failed", "failed": True, "symbols": {}, "next": None},
        "no-next": {"key": "no-next", "symbols": {"ETHUSDT": {"ret": -0.2, "trades": 60}},
                    "next": None},
        "no-symbols": {"key": "no-symbols", "symbols": {}, "next": NEXT},
    }
    included, excluded = wfa_null.population(rows, tmap)
    assert [r["key"] for r in included] == ["kept"]
    assert excluded == {"transfer_failed": 1, "no_next": 1, "no_symbols": 1,
                        "transferred": 1, "no_transfer": 0}


@pytest.mark.parametrize(
    ("start", "end", "k", "step", "n"),
    [
        ("2025-07-01", "2025-12-29", 6, 30, 6),  # 181 d, 1 d dropped
        ("2025-07-01", "2025-09-30", 3, 30, 3),  # 91 d, 1 d dropped
        ("2026-10-01", "2026-11-01", 4, 7, 4),  # 31 d, 3 d dropped
    ],
)
def test_window_plan_per_k(start: str, end: str, k: int, step: int, n: int) -> None:
    got_step, windows = wfa_null.window_plan({"start": start, "end": end}, k)
    assert (got_step, len(windows)) == (step, n)
    assert windows[0][0] == start
    assert all(windows[i][1] == windows[i + 1][0] for i in range(n - 1))
    assert windows[-1][1] == (date.fromisoformat(start) + timedelta(days=n * step)).isoformat()


@pytest.mark.parametrize(
    ("k", "rets", "profitable", "passed"),
    [
        (6, [0.1, -0.1, 0.1, 0.1, 0.1, 0.1], 5, True),
        (6, [0.1, -0.1, -0.1, -0.1, 0.1, 0.1], 3, False),
        (6, [0.1, 0.1, 0.1, 0.1, 0.0, -0.1], 4, False),  # 0.0 is not profitable
        (4, [0.1, -0.1, 0.1, 0.1], 3, True),
        (4, [0.1, 0.1, -0.1, -0.1], 2, False),
        (3, [-0.1, 0.1, 0.1], 2, False),
        (3, [0.1, 0.1, 0.1], 3, True),
    ],
)
def test_verdict_uses_production_wfa_rule(
    k: int, rets: list[float], profitable: int, passed: bool
) -> None:
    row = wfa_null.score(_base(k), SimpleNamespace(uid="t"), _stub_factory(rets))
    assert (row["K"], row["windows"], row["profitable"], row["passed"]) == (k, k, profitable, passed)
    assert row["rets"] == rets
    assert row["trades"] == [100] * k
    assert row["error"] is None


def test_window_exception_counts_as_loss() -> None:
    rets = [0.1, 0.1, 0.1, 0.1, 0.1, -0.1]
    seen: list[str] = []

    def factory(start: str, _end: str) -> Any:
        seen.append(start)

        def fn(_chrom: Any) -> dict[str, float | int]:
            if len(seen) == 2:
                raise ValueError("engine exploded")
            return {"total_return": rets[len(seen) - 1], "total_trades": 100}

        return fn

    row = wfa_null.score(_base(6), SimpleNamespace(uid="t"), factory)
    assert row["windows"] == 6
    assert row["rets"] == [0.1, 0.0, 0.1, 0.1, 0.1, -0.1]  # FAILED_WINDOW, not profitable
    assert (row["profitable"], row["passed"]) == (4, False)


def test_data_gap_fails_report_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from vibe_quant.errors import DataUnavailableError

    def factory(_start: str, _end: str) -> Any:
        def fn(_chrom: Any) -> dict[str, float | int]:
            raise DataUnavailableError("no 1m bars for ZECUSDT")

        return fn

    gap = wfa_null.score(_base(6, key="gap"), SimpleNamespace(uid="t"), factory)
    assert gap["error"].startswith("DataUnavailableError: no 1m bars")
    assert (gap["rets"], gap["windows"], gap["passed"]) == ([], 0, False)

    ok = {**_base(6, key="ok"), "step": 30, "windows": 6, "rets": [0.1] * 6,
          "trades": [100] * 6, "profitable": 6, "passed": True, "error": None}
    result = tmp_path / "r.jsonl"
    _rows(result, [ok, gap])
    rc = wfa_null.main(["report", "--result", str(result)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "errors=1" in out
    assert "L K=6 step=30 windows=6 1/1" in out  # the gap row is out of n, never a silent pass


def test_short_range_counts_as_fail() -> None:
    cell = {**CELL, "start": "2026-10-01", "end": "2026-10-04"}  # 3 d, K=6 -> step 0
    row = wfa_null.score(_base(6, cell=cell), SimpleNamespace(uid="t"), _stub_factory([]))
    assert (row["step"], row["windows"], row["rets"]) == (0, 0, [])
    assert (row["passed"], row["error"]) == (False, None)  # production fails closed


def test_factory_forwards_the_cell_timeframe() -> None:
    """Older ledgers (1m only) omit the field; the engine must still get the right bars."""
    for cell, expected in (({k: v for k, v in CELL.items() if k != "timeframe"}, "1m"),
                           (CELL, "1h")):
        fn = wfa_null._factory(cell)("2026-03-01", "2026-06-01")
        assert (fn.timeframe, fn.symbols) == (expected, [CELL["symbol"]])


def test_worker_refuses_the_real_state_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "VIBE_QUANT_DB", str(wfa_null.REPO_ROOT / "data" / "state" / "vibe_quant.db")
    )
    with pytest.raises(SystemExit, match="real state DB"):
        wfa_null.run_one({"key": "k"}, tmp_path / "r.json")


def test_clopper_pearson_bounds() -> None:
    assert wfa_null.cp_bounds(31, 58) == (0.39872490719438325, 0.666599650355953)
    assert wfa_null.cp_upper(0, 58) == 0.050339338330420105
    assert wfa_null.cp_upper(0, 59) == 0.04950760988822693
    assert wfa_null.cp_bounds(0, 58)[0] == 0.0
    assert wfa_null.cp_bounds(58, 58)[1] == 1.0
    assert wfa_null.cp_upper(58, 58) == 1.0


def test_resume_skips_done_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _rows(out / "ledger.jsonl", [_winner("k")])
    _rows(out / "transfer.jsonl", [
        {"key": "k", "symbols": {"ETHUSDT": {"ret": -0.2, "trades": 60}}, "next": NEXT},
    ])
    result = tmp_path / "r.jsonl"
    _rows(result, [{**_base(6, key="k"), "step": 30, "windows": 6, "rets": [0.1] * 6,
                    "trades": [100] * 6, "profitable": 6, "passed": True,
                    "error": None, "next_win": False}])
    launched: list[tuple[str, int]] = []

    class _Done:
        returncode = 0

        def poll(self) -> int:
            return 0

    def _launch(spec: dict[str, Any]) -> Any:
        launched.append((spec["key"], spec["K"]))
        return _Done(), tmp_path / "scratch", 0.0

    def _finish(spec: dict[str, Any], _tmp: Path, _rc: int, _wall: float) -> dict[str, Any]:
        return {**_base(spec["K"], key=spec["key"], ledger=spec["ledger"]), "step": 46,
                "windows": 4, "rets": [0.1] * 4, "trades": [100] * 4,
                "profitable": 4, "passed": True, "error": None}

    monkeypatch.setattr(wfa_null, "_launch", _launch)
    monkeypatch.setattr(wfa_null, "_finish", _finish)
    argv = ["run", "--out", str(out), "--windows", "6,4", "--result", str(result)]
    assert wfa_null.main(argv) == 0
    assert launched == [("k", 4)]  # (k, 6) was already measured
    assert len(burst.load_ledger(result)) == 2
    assert wfa_null.main(argv) == 0
    assert launched == [("k", 4)]  # nothing left to run
    assert len(burst.load_ledger(result)) == 2


def test_verdict_threshold() -> None:
    assert wfa_null.verdict(0, 58) == "FAIL"  # upper 0.050339 > 0.05
    assert wfa_null.verdict(0, 59) == "PASS"  # upper 0.049508 <= 0.05
    assert wfa_null.verdict(31, 58) == "FAIL"
    assert wfa_null.verdict(58, 58) == "FAIL"
    assert wfa_null.verdict(0, 0) == "FAIL"


def test_run_forwards_the_next_window_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crosstab's next_win is the transfer win rule: below the 50-trade floor is False."""
    out = tmp_path / "out"
    out.mkdir()
    _rows(out / "ledger.jsonl", [_winner("up"), _winner("few")])
    _rows(out / "transfer.jsonl", [
        {"key": "up", "symbols": {"ETHUSDT": {"ret": -0.1, "trades": 60}},
         "next": {**NEXT, "ret": 0.2, "trades": 60}},
        {"key": "few", "symbols": {"ETHUSDT": {"ret": -0.1, "trades": 60}},
         "next": {**NEXT, "ret": 0.2, "trades": 20}},
    ])
    launched: list[dict[str, Any]] = []

    class _Done:
        returncode = 0

        def poll(self) -> int:
            return 0

    def _launch(spec: dict[str, Any]) -> Any:
        launched.append(spec)
        return _Done(), tmp_path, 0.0

    monkeypatch.setattr(wfa_null, "_launch", _launch)
    monkeypatch.setattr(wfa_null, "_finish", lambda spec, _tmp, _rc, _wall: {
        **_base(spec["K"], key=spec["key"]), "step": 30, "windows": 6,
        "rets": [], "trades": [], "profitable": 0, "passed": False,
        "error": None, "next_win": spec["next_win"],
    })
    result = tmp_path / "r.jsonl"
    wfa_null.main(["run", "--out", str(out), "--windows", "6", "--result", str(result)])
    assert [spec["key"] for spec in launched] == ["up", "few"]
    assert [spec["next_win"] for spec in launched] == [True, False]


def test_report_prints_per_ledger_and_pooled_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def row(ledger: str, passed: bool, key: str) -> dict[str, Any]:
        return {"key": key, "ledger": ledger, "K": 6, "step": 30, "windows": 6,
                "rets": [0.1 if passed else -0.1] * 6, "trades": [100] * 6,
                "profitable": 6 if passed else 0, "passed": passed, "error": None,
                "next_win": not passed}

    result = tmp_path / "r.jsonl"
    _rows(result, [row("A", True, "a"), row("B", False, "b")])
    assert wfa_null.main(["report", "--result", str(result)]) == 0
    out = capsys.readouterr().out
    assert "A K=6 step=30 windows=6 1/1" in out
    assert "B K=6 step=30 windows=6 0/1" in out
    assert "pooled K=6 step=30 windows=6 1/2" in out
    assert "(T,F)=1" in out  # the single pass did not repeat in the next window
    assert "(F,T)=1" in out  # the single fail did
