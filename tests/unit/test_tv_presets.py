"""Hand-picked TradingView parameter presets: 20 per port, each inside the port's own domain."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vibe_quant.discovery.port import load_port

REPO = Path(__file__).resolve().parents[2]
TV_DIR = REPO / "scripts" / "research" / "tv_ports"
PORT_DIRS = sorted(p.parent for p in TV_DIR.glob("*/port.yaml"))
N_PRESETS = 20


def _entries(port_dir: Path) -> list[dict[str, object]]:
    raw = yaml.safe_load((port_dir / "presets.yaml").read_text())
    assert isinstance(raw, list)
    return raw


def test_sixteen_ports_found() -> None:
    assert len(PORT_DIRS) == 16


@pytest.mark.parametrize("port_dir", PORT_DIRS, ids=lambda p: p.name)
def test_every_port_has_twenty_valid_presets(port_dir: Path) -> None:
    port = load_port(port_dir)
    entries = _entries(port_dir)
    assert len(entries) == N_PRESETS
    names = [e["name"] for e in entries]
    assert len(set(names)) == N_PRESETS
    assert names[0] == "default"
    assert entries[0]["values"] == port.defaults()
    declared = {p.name for p in port.params}
    for entry in entries:
        assert set(entry) == {"name", "why", "values"}
        assert isinstance(entry["why"], str) and entry["why"].strip()
        values = entry["values"]
        assert isinstance(values, dict) and set(values) <= declared
        full = {**port.defaults(), **values}
        port.check_values(full)
        port.render(full)


# --------------------------------------------------------------------------- the grid runner

import importlib.util  # noqa: E402
import json  # noqa: E402
from datetime import date  # noqa: E402
from typing import Any  # noqa: E402

_spec = importlib.util.spec_from_file_location("preset_grid", TV_DIR / "preset_grid.py")
assert _spec is not None and _spec.loader is not None
grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grid)


def _row(preset: str, coin: str, ret: float, trades: int = 50, sharpe: float = 0.0,
         window: str = "search", error: str | None = None, port: str = "p", tf: str = "1h") -> dict[str, Any]:
    return {"kind": "run", "port": port, "tf": tf, "preset": preset, "coin": coin, "window": window,
            "ret": ret, "trades": trades, "sharpe": sharpe, "error": error}


ORDER = ["default", "fast", "slow"]


def test_pick_is_the_preset_with_most_positive_coins() -> None:
    rows = [
        _row("default", "BTC", 0.1), _row("default", "ETH", -0.1),
        _row("fast", "BTC", 0.1), _row("fast", "ETH", 0.2),
        _row("slow", "BTC", 0.5), _row("slow", "ETH", -0.5),
    ]
    assert grid.select_pick(rows, ORDER) == "fast"


def test_pick_tie_goes_to_higher_median_sharpe_then_file_order() -> None:
    tie = [
        _row("default", "BTC", 0.1, sharpe=0.5), _row("default", "ETH", 0.1, sharpe=0.7),
        _row("fast", "BTC", 0.1, sharpe=1.0), _row("fast", "ETH", 0.1, sharpe=1.4),
    ]
    assert grid.select_pick(tie, ORDER) == "fast"
    equal = [_row("fast", "BTC", 0.1, sharpe=1.0), _row("slow", "BTC", 0.1, sharpe=1.0)]
    assert grid.select_pick(equal, ORDER) == "fast"
    assert grid.select_pick(equal, ["slow", "fast"]) == "slow"


def test_positive_coin_needs_thirty_trades_and_a_return_above_zero() -> None:
    rows = [
        _row("default", "BTC", 0.1, trades=29), _row("default", "ETH", 0.0, trades=80),
        _row("fast", "BTC", 0.1, trades=30),
    ]
    assert grid.preset_score([r for r in rows if r["preset"] == "default"])[0] == 0
    assert grid.preset_score([r for r in rows if r["preset"] == "fast"])[0] == 1
    assert grid.select_pick(rows, ORDER) == "fast"


def test_errored_rows_never_count_and_no_valid_row_gives_no_pick() -> None:
    rows = [_row("default", "BTC", 9.0, sharpe=9.0, error="boom"), _row("fast", "BTC", -0.1)]
    assert grid.select_pick(rows, ORDER) == "fast"
    assert grid.select_pick([_row("default", "BTC", 1.0, error="x")], ORDER) is None
    assert grid.select_pick([], ORDER) is None


def test_lenient_and_strict_bars_at_their_edges() -> None:
    coins = ["A", "B", "C", "D", "E"]
    bh = dict.fromkeys(coins, 0.10)
    four = [_row("p", c, 0.11, window="test") for c in coins[:4]] + [_row("p", "E", 0.10, window="test")]
    out = grid.evaluate_pick(four, bh)
    assert (out["beats_bh"], out["strict"], out["lenient"]) == (4, True, True)  # equal to B&H is no beat
    three = four[:3] + [_row("p", "D", 0.10, window="test"), four[4]]
    out = grid.evaluate_pick(three, bh)
    assert (out["beats_bh"], out["strict"]) == (3, False)
    flat = grid.evaluate_pick([_row("p", "A", 0.0, window="test")], bh)
    assert (flat["lenient"], flat["positive"]) == (False, 0)
    assert grid.evaluate_pick([], bh)["mean"] is None


def test_chance_rates_use_every_clean_test_run() -> None:
    rows = [
        _row("a", "A", 0.2, window="test"), _row("b", "A", -0.1, window="test"),
        _row("a", "B", 0.05, window="test"), _row("b", "B", 0.3, window="test", error="x"),
    ]
    got = grid.chance_rates(rows, {"A": 0.1, "B": None})
    assert got == {"runs": 3, "share_positive": 2 / 3, "runs_with_bh": 2, "share_beats_bh": 0.5}


def test_done_keys_resume_and_retry_errors(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    ok = {"kind": "run", "key": "run|k1", "error": None}
    bad = {"kind": "run", "key": "run|k2", "error": "boom"}
    bh = {"kind": "bh", "key": "bh|A|test", "error": "ignored"}
    path.write_text("\n".join(json.dumps(r) for r in (ok, bad, bh)) + '\n{"kind": "run", "ke')
    assert grid.done_keys(path) == {"run|k1", "run|k2", "bh|A|test"}
    assert grid.done_keys(path, retry_errors=True) == {"run|k1", "bh|A|test"}
    assert grid.done_keys(tmp_path / "missing.jsonl") == set()


def test_clip_window_starts_the_day_after_the_first_close() -> None:
    first = date(2025, 5, 29)  # HYPE: first full day
    assert grid.clip_window("2023-01-01", "2026-01-01", first) == (date(2025, 5, 30), date(2026, 1, 1))
    assert grid.clip_window("2023-01-01", "2026-01-01", date(2020, 1, 1))[0] == date(2023, 1, 1)


def test_bh_return_is_last_close_over_the_close_before_start() -> None:
    def day(d: date) -> int:
        return (d - date(1970, 1, 1)).days

    closes = {day(date(2025, 12, 31)): 100.0, day(date(2026, 1, 1)): 105.0, day(date(2026, 1, 9)): 150.0}
    assert grid.bh_return(closes, date(2026, 1, 1), date(2026, 1, 10)) == 0.5
    assert grid.bh_return(closes, date(2026, 1, 2), date(2026, 1, 10)) == 150.0 / 105.0 - 1.0
    assert grid.bh_return(closes, date(2026, 1, 1), date(2026, 3, 1)) is None


def test_run_task_forwards_the_preset_values_and_records_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def fake_fn(coin: str, tf: str, start: str, end: str) -> Any:
        def call(genome: Any) -> dict[str, Any]:
            seen.append({"coin": coin, "tf": tf, "start": start, "end": end, "values": dict(genome.values)})
            return {"total_return": 0.25, "total_trades": 41, "sharpe_ratio": 1.5, "max_drawdown": 0.1}
        return call

    monkeypatch.setattr(grid, "_backtest_fn", fake_fn)
    for name, expect in (("st7_3", {"period": 7}), ("st14_2", {"period": 14, "multiplier": 2.0})):
        preset = next(p for p in grid.load_presets(TV_DIR / "03_supertrend") if p["name"] == name)
        row = grid.run_task({"port": "03_supertrend", "preset": name, "tf": "4h", "coin": "ETHUSDT",
                             "window": "test", "start": "2026-01-01", "end": "2026-10-20",
                             "values": preset["values"]})
        assert (row["ret"], row["trades"], row["sharpe"], row["dd"], row["error"]) == (0.25, 41, 1.5, 0.1, None)
        assert row["key"] == f"run|03_supertrend|{name}|4h|ETHUSDT|test"
        assert {k: seen[-1]["values"][k] for k in expect} == expect
        assert (seen[-1]["coin"], seen[-1]["tf"], seen[-1]["start"]) == ("ETHUSDT", "4h", "2026-01-01")


def test_run_task_error_becomes_a_row_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any) -> Any:
        raise RuntimeError("no bars")

    monkeypatch.setattr(grid, "_backtest_fn", boom)
    row = grid.run_task({"port": "03_supertrend", "preset": "default", "tf": "1h", "coin": "BTCUSDT",
                         "window": "search", "start": "2023-01-01", "end": "2026-01-01",
                         "values": grid.load_presets(TV_DIR / "03_supertrend")[0]["values"]})
    assert row["error"] == "RuntimeError: no bars"
    assert row["ret"] == 0.0 and row["trades"] == 0


def _fake_closes(first: date, last: date) -> dict[int, float]:
    start, end = (first - grid.EPOCH).days, (last - grid.EPOCH).days
    return {d: 100.0 + (d - start) for d in range(start, end + 1)}


def test_plan_resumes_skips_short_coins_and_refuses_port_timeframes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closes = {
        "AAAUSDT": _fake_closes(date(2022, 1, 1), date(2026, 10, 19)),
        "NEWUSDT": _fake_closes(date(2025, 10, 1), date(2026, 10, 19)),  # 92 search days
    }
    monkeypatch.setattr("vibe_quant.discovery.buyhold.load_daily_closes", lambda syms, _p: {s: closes[s] for s in syms})
    results = tmp_path / "results.jsonl"
    ports = [TV_DIR / "03_supertrend", TV_DIR / "14_ichimoku_hull_macd"]
    tasks = grid.plan(ports, ["15m", "1h"], ["AAAUSDT", "NEWUSDT"], 2, set(), results)
    # 03: 2 tf x 2 presets x (AAA search+test, NEW test) = 12; 14: 1h only = 6
    assert len(tasks) == 18
    assert {t["tf"] for t in tasks if t["port"] == "14_ichimoku_hull_macd"} == {"1h"}
    assert not [t for t in tasks if t["coin"] == "NEWUSDT" and t["window"] == "search"]
    assert [t for t in tasks if t["coin"] == "NEWUSDT"][0]["start"] == "2026-01-01"
    rows = grid.read_rows(results)
    skip = next(r for r in rows if r["kind"] == "skip")
    assert (skip["coin"], skip["window"]) == ("NEWUSDT", "search")
    bh = {r["key"]: r["ret"] for r in rows if r["kind"] == "bh"}
    assert bh["bh|AAAUSDT|test"] == (closes["AAAUSDT"][(date(2026, 10, 19) - grid.EPOCH).days]
                                     / closes["AAAUSDT"][(date(2025, 12, 31) - grid.EPOCH).days] - 1.0)
    done = {grid.run_key("03_supertrend", "default", "1h", "AAAUSDT", "search")}
    again = grid.plan(ports, ["15m", "1h"], ["AAAUSDT", "NEWUSDT"], 2, done | set(bh) | {skip["key"]}, results)
    assert len(again) == 17
    assert len(grid.read_rows(results)) == len(rows)  # nothing re-appended


def test_report_names_the_pick_and_the_bars(tmp_path: Path) -> None:
    rows = [
        {"kind": "bh", "key": f"bh|{c}|test", "coin": c, "window": "test", "ret": 0.0} for c in ("AAAUSDT", "BBBUSDT")
    ]
    for preset, rets in (("default", (-0.1, -0.1)), ("st7_3", (0.1, 0.2))):
        for coin, ret in zip(("AAAUSDT", "BBBUSDT"), rets, strict=True):
            rows.append(_row(preset, coin, ret, port="03_supertrend"))
            rows.append(_row(preset, coin, ret * 2, port="03_supertrend", window="test"))
    text, nums = grid.build_report(rows, {"03_supertrend": ["default", "st7_3"]})
    assert "| 03_supertrend | 1h | st7_3 | 2 |" in text
    assert (nums["picks"], nums["lenient"], nums["strict"]) == (1, 1, 0)
    assert nums["chance"]["share_positive"] == 0.5
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert grid.cmd_report(type("A", (), {"out": str(tmp_path)})()) == 0
    assert "st7_3" in (tmp_path / "report.md").read_text()
