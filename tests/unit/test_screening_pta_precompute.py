"""NTScreeningRunner installs precomputed compute_fn columns (vibe-quant-surto, S2).

The runner replays each strategy's compute_fn indicators once over the catalog bars
of the run window and installs them on the generated strategies (``_pta_pre``); the
generated feed then writes the precomputed value per bar instead of recomputing every
indicator over the buffer. Every metric must stay bit-identical to the live per-bar
path (``pta_precompute=False``), and a run reports what it fed in
``last_pta_precompute``. Real BTCUSDT/ETHUSDT catalog bars: skipped when absent.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vibe_quant.screening import nt_runner
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.screening.types import BacktestMetrics

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
_BAR_DIR = CATALOG / "data" / "bar"
needs_catalog = pytest.mark.skipif(
    not (_BAR_DIR / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir()
    or not (_BAR_DIR / "ETHUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir()
    or not (_BAR_DIR / "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL").is_dir(),
    reason="needs the BTCUSDT/ETHUSDT 1m and BTCUSDT 4h catalog",
)

START, END = "2026-03-01", "2026-03-08"
SHORT_START, SHORT_END = "2026-03-02", "2026-03-04"
LOGGER = "vibe_quant.screening.nt_runner"
_EXITS = {
    "stop_loss": {"type": "fixed_pct", "percent": 0.5},
    "take_profit": {"type": "fixed_pct", "percent": 0.6},
}


def rsi_rams_dsl() -> dict[str, Any]:
    """1m RSI (NT class) + RAMS (compute_fn): the strategy the bead measured at 93.83 s/month."""
    return {
        "name": "pta_pre_rsi_rams",
        "timeframe": "1m",
        "indicators": {
            "rsi": {"type": "RSI", "period": 14},
            "rams": {"type": "RAMS", "period": 50},
        },
        "entry_conditions": {"long": ["rsi < 45", "rams > 2.45"]},
        **_EXITS,
    }


def kama_adx_dsl() -> dict[str, Any]:
    return {
        "name": "pta_pre_kama_adx",
        "timeframe": "1m",
        "indicators": {
            "kama": {"type": "KAMA", "period": 10},
            "adx": {"type": "ADX", "period": 14},
        },
        "entry_conditions": {"long": ["close > kama", "adx > 20"]},
        **_EXITS,
    }


def multi_timeframe_dsl() -> dict[str, Any]:
    """1h strategy with a 1h KAMA and a 4h ADX: two buffers, two precomputed timeframes."""
    return {
        "name": "pta_pre_mtf",
        "timeframe": "1h",
        "additional_timeframes": ["4h"],
        "indicators": {
            "kama": {"type": "KAMA", "period": 10},
            "adx": {"type": "ADX", "period": 14, "timeframe": "4h"},
        },
        "entry_conditions": {"long": ["close > kama", "adx > 15"]},
        "stop_loss": {"type": "fixed_pct", "percent": 1.0},
        "take_profit": {"type": "fixed_pct", "percent": 1.5},
    }


def make_runner(
    dsl: dict[str, Any], symbols: list[str], start: str, end: str, **kwargs: Any
) -> NTScreeningRunner:
    return NTScreeningRunner(
        dsl,
        symbols,
        start,
        end,
        catalog_path=str(CATALOG),
        funding_archive_path=str(ARCHIVE),
        **kwargs,
    )


def run(
    dsl: dict[str, Any], symbols: list[str], start: str, end: str, **kwargs: Any
) -> tuple[NTScreeningRunner, BacktestMetrics]:
    runner = make_runner(dsl, symbols, start, end, **kwargs)
    return runner, runner({})


def assert_same_metrics(a: BacktestMetrics, b: BacktestMetrics) -> None:
    """Every BacktestMetrics field equal (exact ``==``) except the wall-clock one."""
    for field in dataclasses.fields(BacktestMetrics):
        if field.name == "execution_time_seconds":
            continue
        assert getattr(a, field.name) == getattr(b, field.name), field.name


def assert_fully_fed(runner: NTScreeningRunner, expect: set[tuple[str, str]]) -> None:
    """Each expected (instrument, tf) was installed, fed every bar and never failed."""
    got = {(str(e["instrument"]), str(e["tf"])) for e in runner.last_pta_precompute}
    assert got == expect
    for entry in runner.last_pta_precompute:
        assert entry["failed_at"] is None, entry
        assert entry["fed"] == entry["n"] and int(entry["n"]) > 0, entry  # type: ignore[call-overload]


@needs_catalog
def test_metrics_identical_with_and_without_precompute_rams_1m() -> None:
    off_runner, off = run(rsi_rams_dsl(), ["BTCUSDT"], START, END, pta_precompute=False)
    on_runner, on = run(rsi_rams_dsl(), ["BTCUSDT"], START, END)
    assert off.total_trades > 5  # the window exercises fills, not an empty run
    assert_same_metrics(on, off)
    assert off_runner.last_pta_precompute == []
    assert_fully_fed(on_runner, {("BTCUSDT-PERP.BINANCE", "1m")})


@needs_catalog
def test_metrics_identical_with_and_without_precompute_kama_adx_two_symbols() -> None:
    symbols = ["BTCUSDT", "ETHUSDT"]
    _, off = run(kama_adx_dsl(), symbols, START, END, pta_precompute=False)
    on_runner, on = run(kama_adx_dsl(), symbols, START, END)
    assert off.total_trades > 5
    assert_same_metrics(on, off)
    # one install per (strategy, tf): each symbol has its own strategy and its own bars
    assert_fully_fed(
        on_runner, {("BTCUSDT-PERP.BINANCE", "1m"), ("ETHUSDT-PERP.BINANCE", "1m")}
    )


@needs_catalog
def test_metrics_identical_with_and_without_precompute_multi_timeframe() -> None:
    start, end = "2026-02-01", "2026-03-08"
    _, off = run(multi_timeframe_dsl(), ["BTCUSDT"], start, end, pta_precompute=False)
    on_runner, on = run(multi_timeframe_dsl(), ["BTCUSDT"], start, end)
    assert off.total_trades > 0
    assert_same_metrics(on, off)
    assert_fully_fed(on_runner, {("BTCUSDT-PERP.BINANCE", "1h"), ("BTCUSDT-PERP.BINANCE", "4h")})
    by_tf = {str(e["tf"]): int(e["n"]) for e in on_runner.last_pta_precompute}  # type: ignore[call-overload]
    assert by_tf["1h"] > by_tf["4h"]  # two different frames


def _strategy_239_dsl() -> dict[str, Any] | None:
    """Strategy 239's DSL from the main checkout's state DB (read-only); None when absent."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True, cwd=REPO,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):  # not a git checkout (mutation sandbox)
        return None
    db = Path(common).parent / "data/state/vibe_quant.db"
    if not db.is_file():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute("select dsl_config from strategies where id=239").fetchone()
    finally:
        con.close()
    return None if row is None else dict(json.loads(row[0]))


@needs_catalog
def test_exactness_strategy_installs_precompute() -> None:
    """Strategy 239 (4h, PRICE_POSITION/ADX/KAMA) runs THROUGH the precompute with defaults."""
    dsl = _strategy_239_dsl()
    if dsl is None:
        pytest.skip("needs strategy 239 in the state DB")
    on_runner, on = run(dsl, ["BTCUSDT"], "2025-01-01", "2025-06-01")
    _, off = run(dsl, ["BTCUSDT"], "2025-01-01", "2025-06-01", pta_precompute=False)
    assert off.total_trades > 0
    assert len(on_runner.last_pta_precompute) >= 1
    assert_fully_fed(on_runner, {("BTCUSDT-PERP.BINANCE", "4h")})
    assert_same_metrics(on, off)


@needs_catalog
@pytest.mark.parametrize(
    ("value", "installed"), [("0", False), (None, True), ("", True), ("false", True)]
)
def test_env_switch(monkeypatch: pytest.MonkeyPatch, value: str | None, installed: bool) -> None:
    if value is None:
        monkeypatch.delenv("VIBE_QUANT_PTA_PRECOMPUTE", raising=False)
    else:
        monkeypatch.setenv("VIBE_QUANT_PTA_PRECOMPUTE", value)
    runner, _ = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END)
    if installed:
        assert_fully_fed(runner, {("BTCUSDT-PERP.BINANCE", "1m")})
    else:
        assert runner.last_pta_precompute == []


@needs_catalog
def test_report_is_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reused runner reports its last run only (a switched-off run leaves nothing behind)."""
    monkeypatch.delenv("VIBE_QUANT_PTA_PRECOMPUTE", raising=False)
    runner, _ = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END)
    assert len(runner.last_pta_precompute) == 1
    monkeypatch.setenv("VIBE_QUANT_PTA_PRECOMPUTE", "0")
    runner({})
    assert runner.last_pta_precompute == []


@needs_catalog
def test_install_error_keeps_live_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("VIBE_QUANT_PTA_PRECOMPUTE", raising=False)
    _, off = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END, pta_precompute=False)

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("vibe_quant.dsl.pta_replay.strategy_pta_columns", boom)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    runner, on = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END)
    assert "PTA precompute off for BTCUSDT-PERP.BINANCE 1m: RuntimeError: boom" in caplog.text
    assert "Traceback" not in caplog.text  # no exc_info: one line per tf
    assert all(r.exc_info is None for r in caplog.records)
    assert runner.last_pta_precompute == []
    assert_same_metrics(on, off)


@needs_catalog
def test_misaligned_columns_fall_back_exactly_and_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bar the precompute does not match is recomputed live: same numbers, one warning."""
    monkeypatch.delenv("VIBE_QUANT_PTA_PRECOMPUTE", raising=False)
    _, off = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END, pta_precompute=False)

    real = nt_runner._pta_bar_arrays

    def shifted(*args: Any) -> dict[str, np.ndarray]:
        arrays = dict(real(*args))
        close = arrays["close"].copy()
        close[100] += 1.0  # the engine feeds a different close at bar 100
        arrays["close"] = close
        return arrays

    monkeypatch.setattr(nt_runner, "_pta_bar_arrays", shifted)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    runner, on = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END)
    (entry,) = runner.last_pta_precompute
    assert entry["failed_at"] == 100 and entry["fed"] == 100
    assert (
        f"PTA precompute fell back for BTCUSDT-PERP.BINANCE 1m at bar 100 of {entry['n']}"
        in caplog.text
    )
    assert_same_metrics(on, off)


@needs_catalog
def test_precompute_longer_than_the_run_warns_without_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unfed trailing bars (fed != n, nothing failed) are reported too."""
    monkeypatch.delenv("VIBE_QUANT_PTA_PRECOMPUTE", raising=False)
    _, off = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END, pta_precompute=False)

    real = nt_runner._pta_bar_arrays

    def longer(*args: Any) -> dict[str, np.ndarray]:
        arrays = real(*args)
        extra = {k: np.append(v, v[-1] + 1) for k, v in arrays.items()}  # one bar the engine never feeds
        return extra

    monkeypatch.setattr(nt_runner, "_pta_bar_arrays", longer)
    caplog.set_level(logging.WARNING, logger=LOGGER)
    runner, on = run(rsi_rams_dsl(), ["BTCUSDT"], SHORT_START, SHORT_END)
    (entry,) = runner.last_pta_precompute
    n = int(entry["n"])  # type: ignore[call-overload]
    assert entry["failed_at"] is None and entry["fed"] == n - 1
    assert f"PTA precompute fell back for BTCUSDT-PERP.BINANCE 1m at bar {n - 1} of {n}" in caplog.text
    assert_same_metrics(on, off)


def _fake_catalog(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, ...]]) -> None:
    """Replace the catalog query by a deterministic stand-in: 3 Bars-like rows per key."""

    class _Bar:
        def __init__(self, i: int, salt: float) -> None:
            self.ts_init = 1_000 + i
            self.open = self.high = self.low = self.close = 10.0 + i + salt
            self.volume = 2.0

    class _Catalog:
        def __init__(self, path: str) -> None:
            self.path = path

        def bars(self, bar_types: list[str], start: str, end: str) -> list[_Bar]:
            calls.append((self.path, bar_types[0], start, end))
            return [_Bar(i, float(len(calls))) for i in range(3)]

    monkeypatch.setattr("nautilus_trader.persistence.catalog.ParquetDataCatalog", _Catalog)
    nt_runner._PTA_ARRAYS.clear()


def test_bar_arrays_are_cached_as_arrays_in_a_four_entry_lru(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    _fake_catalog(monkeypatch, calls)
    first = nt_runner._pta_bar_arrays("/cat", "A-1-MINUTE", "2026-01-01", "2026-01-02")
    assert set(first) == {"ts_init", "open", "high", "low", "close", "volume"}
    assert all(isinstance(v, np.ndarray) for v in first.values())  # arrays, never Bar objects
    assert first["ts_init"].tolist() == [1_000, 1_001, 1_002]
    assert first["close"].tolist() == [11.0, 12.0, 13.0]  # 10 + i + salt (first call)
    assert first["volume"].dtype == np.float64
    # same key: served from the cache, no second query
    assert nt_runner._pta_bar_arrays("/cat", "A-1-MINUTE", "2026-01-01", "2026-01-02") is first
    assert len(calls) == 1
    # another bar type / window / path are other keys
    second = nt_runner._pta_bar_arrays("/cat", "B-1-MINUTE", "2026-01-01", "2026-01-02")
    assert second["close"].tolist() == [12.0, 13.0, 14.0]
    nt_runner._pta_bar_arrays("/cat", "A-1-MINUTE", "2026-01-01", "2026-01-03")
    nt_runner._pta_bar_arrays("/other", "A-1-MINUTE", "2026-01-01", "2026-01-02")
    assert len(calls) == 4 and len(nt_runner._PTA_ARRAYS) == 4
    # a fifth key evicts the least recently used (the first), the others stay
    nt_runner._pta_bar_arrays("/cat", "C-1-MINUTE", "2026-01-01", "2026-01-02")
    assert len(nt_runner._PTA_ARRAYS) == 4
    assert ("/cat", "A-1-MINUTE", "2026-01-01", "2026-01-02") not in nt_runner._PTA_ARRAYS
    assert ("/cat", "B-1-MINUTE", "2026-01-01", "2026-01-02") in nt_runner._PTA_ARRAYS
    nt_runner._pta_bar_arrays("/cat", "A-1-MINUTE", "2026-01-01", "2026-01-02")
    assert len(calls) == 6  # the evicted key is queried again


def test_bar_arrays_hit_refreshes_recency(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []
    _fake_catalog(monkeypatch, calls)
    keys = [("/c", f"T{i}", "2026-01-01", "2026-01-02") for i in range(4)]
    for key in keys:
        nt_runner._pta_bar_arrays(*key)
    nt_runner._pta_bar_arrays(*keys[0])  # touch the oldest: it is now the newest
    nt_runner._pta_bar_arrays("/c", "T4", "2026-01-01", "2026-01-02")  # evicts keys[1]
    assert keys[0] in nt_runner._PTA_ARRAYS
    assert keys[1] not in nt_runner._PTA_ARRAYS


def test_default_runner_precompute_flags() -> None:
    dsl = rsi_rams_dsl()
    assert NTScreeningRunner(dsl, ["BTCUSDT"], START, END)._pta_precompute is True
    off = NTScreeningRunner(dsl, ["BTCUSDT"], START, END, pta_precompute=False)
    assert off._pta_precompute is False
    assert off.last_pta_precompute == []
