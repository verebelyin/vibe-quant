"""Aux-context indicators (FUNDING / FUNDING_Z) -- vibe-quant-5x7r8."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.unit.test_compiler_golden_source import (
    EXAMPLES_DIR,
    GOLDEN_DIR,
    _normalize,
)
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.validation.funding import clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

H = 3_600 * 1_000_000_000
MS = 1_000_000
DAY0 = 1_704_067_200 * 1_000_000_000  # 2024-01-01T00:00:00Z


def _make_archive(path: Path, rates_ms: list[tuple[int, float]], symbol: str = "BTCUSDT") -> Path:
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        for ft, rate in rates_ms:
            archive.conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, source) "
                "VALUES (?, ?, ?, 'test')",
                (symbol, ft, rate),
            )
        archive.conn.commit()
    finally:
        archive.close()
    return path


@pytest.fixture
def funding_archive(tmp_path: Path) -> Iterator[Path]:
    clear_rate_cache()
    day0_ms = DAY0 // MS
    rows = []
    for i in range(40):
        jitter = 4 if i % 2 == 0 else 0  # ...00:00.004 style jitter
        rows.append((day0_ms + i * 8 * 3_600_000 + jitter, 0.0001 * (i + 1)))
    path = _make_archive(tmp_path / "archive.db", rows)
    aux_data.configure(path, None)
    yield path
    aux_data.configure(None, None)
    clear_rate_cache()


def _funding_dsl(threshold: float = 0.5) -> dict[str, object]:
    return {
        "name": "funding_z_test",
        "timeframe": "4h",
        "indicators": {"fz": {"type": "FUNDING_Z", "period": 30}},
        "entry_conditions": {"long": [f"fz < {threshold}"]},
        "exit_conditions": {"long": ["fz > 1.5"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def test_non_context_strategy_source_unchanged() -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401  (registered; must not leak)

    files = sorted(EXAMPLES_DIR.glob("*.yaml"))
    assert files
    for yml in files:
        dsl = validate_strategy_dict(yaml.safe_load(yml.read_text()))
        src = StrategyCompiler().compile(dsl)
        golden = (GOLDEN_DIR / f"{yml.stem}.py.golden").read_text()
        assert _normalize(src) == _normalize(golden), yml.name


def test_context_strategy_source_buffers_close_ns() -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401

    src = StrategyCompiler().compile(validate_strategy_dict(_funding_dsl()))
    assert "self._pta_bufs: dict[str, PtaBuffer] = {" in src
    assert "with_close_ns=True" in src
    assert "((bar.ts_init + _step // 2) // _step) * _step," in src
    compile(src, "<generated>", "exec")


def test_funding_asof_no_lookahead_and_snaps_jitter(funding_archive: Path) -> None:
    # settlement 0 at 00:00:00.004 (1e-4), 1 at 08:00:00.000 (2e-4), 2 at 16:00:00.004 (3e-4)
    closes = np.array(
        [
            DAY0 + 8 * H - 1 * MS,  # 07:59:59.999 -> 00:00 rate
            DAY0 + 8 * H,  # 08:00 close -> 08:00 rate
            DAY0 + 16 * H,  # 16:00 close; archived 16:00:00.004 -> 16:00 rate
            DAY0 - 1 * MS,  # before first settlement -> NaN
        ],
        dtype=np.int64,
    )
    out = aux_data.funding_asof("BTCUSDT", closes)
    assert out[0] == pytest.approx(1e-4)
    assert out[1] == pytest.approx(2e-4)
    assert out[2] == pytest.approx(3e-4)
    assert np.isnan(out[3])


def test_funding_z_over_settlements_not_bars(funding_archive: Path) -> None:
    period = 5
    t = DAY0 + 20 * 8 * H  # settlement #20 (rate 2.1e-3)
    z_one = aux_data.funding_z_asof("BTCUSDT", np.array([t], dtype=np.int64), period)[0]
    # 60 one-minute bars inside the same settlement interval: identical value
    many = t + np.arange(0, 60, dtype=np.int64) * 60_000_000_000
    z_many = aux_data.funding_z_asof("BTCUSDT", many, period)
    assert np.all(z_many == z_one)
    window = np.array([0.0001 * (k + 1) for k in range(16, 21)])
    expected = (window[-1] - window.mean()) / window.std(ddof=1)
    assert z_one == pytest.approx(expected)


def test_missing_archive_raises(tmp_path: Path) -> None:
    clear_rate_cache()
    aux_data.configure(tmp_path / "nope.db", None)
    try:
        with pytest.raises(ValueError, match="funding archive"):
            aux_data.funding_asof("BTCUSDT", np.array([DAY0], dtype=np.int64))
        other = _make_archive(tmp_path / "other.db", [(DAY0 // MS, 1e-4)], symbol="ETHUSDT")
        aux_data.configure(other, None)
        with pytest.raises(ValueError, match="No funding rates archived"):
            aux_data.funding_asof("BTCUSDT", np.array([DAY0], dtype=np.int64))
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_context_spec_excluded_from_default_pool() -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401
    from vibe_quant.discovery.genome import build_indicator_pool

    spec = indicator_registry.get("FUNDING_Z")
    assert spec is not None and spec.needs_context
    default = build_indicator_pool()
    assert "FUNDING_Z" not in default
    assert "FUNDING" not in default
    assert "FUNDING_Z" in build_indicator_pool(include_context=True)


def test_invoke_requires_context() -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401

    spec = indicator_registry.get("FUNDING")
    assert spec is not None
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    with pytest.raises(ValueError, match="needs context"):
        invoke_compute_fn(spec, df, {})


def test_paper_rejects_context_indicator() -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401
    from vibe_quant.paper.config import ConfigurationError
    from vibe_quant.paper.node import reject_context_indicators

    with pytest.raises(ConfigurationError, match="FUNDING_Z"):
        reject_context_indicators(validate_strategy_dict(_funding_dsl()))
    plain = _funding_dsl()
    plain["indicators"] = {"rsi": {"type": "RSI", "period": 14}}
    plain["entry_conditions"] = {"long": ["rsi < 30"]}
    plain["exit_conditions"] = {"long": ["rsi > 70"]}
    reject_context_indicators(validate_strategy_dict(plain))


def test_funding_strategy_runs_in_screening() -> None:
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict

    if not DEFAULT_ARCHIVE_PATH.exists():
        pytest.skip("real funding archive not available")
    clear_rate_cache()
    dsl = validate_strategy_dict(_funding_dsl())

    runner = NTScreeningRunner(_dsl_to_dict(dsl), ["BTCUSDT"], "2024-01-01", "2024-06-30")
    metrics = runner({})
    assert metrics.total_trades > 0


def test_funding_z_compute_fn_over_settlements_not_bars(funding_archive: Path) -> None:
    """Plugin level: 1m bars inside one settlement interval share one z value."""
    import vibe_quant.dsl.plugins.funding  # noqa: F401

    spec = indicator_registry.get("FUNDING_Z")
    assert spec is not None
    t = DAY0 + 20 * 8 * H
    ns = t + np.arange(0, 60, dtype=np.int64) * 60_000_000_000
    df = pd.DataFrame(
        {c: np.arange(60, dtype=float) for c in ("open", "high", "low", "close", "volume")}
    )
    df.attrs["bar_close_ns"] = ns
    df.attrs["symbol"] = "BTCUSDT-PERP.BINANCE"
    out = invoke_compute_fn(spec, df, {"period": 5})
    assert isinstance(out, pd.Series)
    window = np.array([0.0001 * (k + 1) for k in range(16, 21)])
    expected = (window[-1] - window.mean()) / window.std(ddof=1)
    assert np.allclose(out.to_numpy(), expected)


def test_screening_uses_runner_archive_path(tmp_path: Path) -> None:
    """The runner's funding_archive_path must reach aux_data (no silent default)."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict

    clear_rate_cache()
    other = _make_archive(tmp_path / "eth_only.db", [(DAY0 // MS, 1e-4)], symbol="ETHUSDT")
    dsl = validate_strategy_dict(_funding_dsl())
    runner = NTScreeningRunner(
        _dsl_to_dict(dsl), ["BTCUSDT"], "2024-01-01", "2024-06-30",
        funding_archive_path=str(other),
    )
    try:
        # preflight raises before the engine; must NOT be masked as a -inf result
        with pytest.raises(AuxDataUnavailableError, match="No funding rates archived"):
            runner({})
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


@pytest.fixture
def _restore_ga_pool() -> Iterator[None]:
    """Rebuild the lazily-cached GA pool from scratch, restore afterwards."""
    from vibe_quant.discovery import operators as ops

    caches = [ops.INDICATOR_POOL, ops._INDICATOR_NAMES, ops.THRESHOLD_RANGES,
              ops._INT_PARAMS, ops._PARAM_DEFAULTS]
    saved = [(c, c.copy()) for c in caches]
    for c in caches:
        c.clear()
    yield
    for c, snap in saved:
        c.clear()
        if isinstance(c, dict):
            c.update(snap)  # type: ignore[arg-type]
        else:
            c.extend(snap)  # type: ignore[arg-type]


def _pipeline(indicator_pool: list[str] | None):  # type: ignore[no-untyped-def]
    from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    return DiscoveryPipeline(_make_config(indicator_pool=indicator_pool), _mock_backtest)


def test_explicit_pool_includes_funding_z(_restore_ga_pool: None) -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401
    from vibe_quant.discovery.operators import INDICATOR_POOL, THRESHOLD_RANGES

    _pipeline(["FUNDING_Z", "RSI"])._apply_indicator_pool_filter()
    assert set(INDICATOR_POOL) == {"FUNDING_Z", "RSI"}
    assert THRESHOLD_RANGES["FUNDING_Z"] == (-3.0, 3.0)


def test_default_pool_still_excludes_context(_restore_ga_pool: None) -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401
    from vibe_quant.discovery.operators import INDICATOR_POOL

    _pipeline(None)._apply_indicator_pool_filter()
    assert "FUNDING_Z" not in INDICATOR_POOL
    assert "FUNDING" not in INDICATOR_POOL
    assert "RSI" in INDICATOR_POOL


def _settlements(start_ns: int, n: int, skip: range = range(0)) -> list[tuple[int, float]]:
    return [
        ((start_ns + i * 8 * H) // MS, 1e-4)
        for i in range(n)
        if i not in skip
    ]


def _runner_for(archive: Path, start: str, end: str):  # type: ignore[no-untyped-def]
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict

    dsl = validate_strategy_dict(_funding_dsl())
    return NTScreeningRunner(
        _dsl_to_dict(dsl), ["BTCUSDT"], start, end, funding_archive_path=str(archive)
    )


def test_error_hierarchy() -> None:
    from vibe_quant.errors import DataUnavailableError
    from vibe_quant.screening.nt_runner import MissingBarDataError

    assert issubclass(MissingBarDataError, DataUnavailableError)
    assert issubclass(AuxDataUnavailableError, DataUnavailableError)
    assert issubclass(DataUnavailableError, ValueError)


def test_trimmed_archive_window_raises_through_runner(tmp_path: Path) -> None:
    clear_rate_cache()
    june1 = DAY0 + 152 * 24 * H  # 2024-06-01
    arch = _make_archive(tmp_path / "trim.db", _settlements(june1, 3 * 250))
    try:
        with pytest.raises(AuxDataUnavailableError, match="starts after"):
            _runner_for(arch, "2024-01-01", "2024-12-31")({})
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_window_past_archive_end_raises_through_runner(tmp_path: Path) -> None:
    clear_rate_cache()
    arch = _make_archive(tmp_path / "short.db", _settlements(DAY0, 3 * 60))  # to ~2024-02-29
    try:
        with pytest.raises(AuxDataUnavailableError, match="ends before"):
            _runner_for(arch, "2024-01-01", "2024-06-30")({})
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_preflight_allows_warmup_margin(tmp_path: Path) -> None:
    clear_rate_cache()
    arch = _make_archive(tmp_path / "ok.db", _settlements(DAY0 + 2 * 24 * H, 3 * 200))
    aux_data.configure(arch, None)
    try:  # first settlement 2 days after start: inside the 3-day margin
        aux_data.preflight(["FUNDING_Z"], ["BTCUSDT"], "2024-01-01", "2024-02-01")
        aux_data.preflight(["RSI"], ["BTCUSDT"], "1999-01-01", "2099-01-01")  # not context
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_validation_runner_preflights_context_funding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe_quant.data.archive as archive_mod
    from vibe_quant.validation.runner import ValidationRunner

    clear_rate_cache()
    june1 = DAY0 + 152 * 24 * H
    arch = _make_archive(tmp_path / "trim.db", _settlements(june1, 3 * 250))
    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", arch)
    runner = ValidationRunner(db_path=tmp_path / "state.db")
    dsl = validate_strategy_dict(_funding_dsl())
    try:
        with pytest.raises(AuxDataUnavailableError):
            runner._run_backtest(
                1, "funding_z_test", dsl, None, {  # type: ignore[arg-type]
                    "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2024-12-31"
                }, None,  # type: ignore[arg-type]
            )
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_archive_hole_returns_nan_not_stale(tmp_path: Path) -> None:
    clear_rate_cache()
    # settlements 0..9 then hole 10..19, then 20..29
    arch = _make_archive(tmp_path / "hole.db", _settlements(DAY0, 30, skip=range(10, 20)))
    aux_data.configure(arch, None)
    try:
        last_pre = DAY0 + 9 * 8 * H
        out = aux_data.funding_asof(
            "BTCUSDT",
            np.array([last_pre + 15 * H, last_pre + 17 * H, DAY0 + 25 * 8 * H], dtype=np.int64),
        )
        assert out[0] == pytest.approx(1e-4)  # within 2 periods: forward-fill ok
        assert np.isnan(out[1])  # stale beyond 2 periods
        assert out[2] == pytest.approx(1e-4)  # after the hole: live again
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_funding_z_series_cached(funding_archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy.lib.stride_tricks as st

    t = np.array([DAY0 + 20 * 8 * H], dtype=np.int64)
    first = aux_data.funding_z_asof("BTCUSDT", t, 5)
    calls = {"n": 0}
    real = st.sliding_window_view

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(st, "sliding_window_view", counting)
    assert aux_data.funding_z_asof("BTCUSDT", t, 5)[0] == first[0]
    assert calls["n"] == 0  # served from cache
    clear_rate_cache()  # invalidates together with the rate cache
    aux_data.funding_z_asof("BTCUSDT", t, 5)
    assert calls["n"] == 1


def _raising_backtest(chrom):  # type: ignore[no-untyped-def]
    raise AuxDataUnavailableError("simulated mid-run data loss")


def test_pipeline_preflight_fails_before_any_evaluation(
    funding_archive: Path, _restore_ga_pool: None
) -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401
    from tests.unit.test_discovery_pipeline import _make_config
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    calls: list[object] = []

    def counting(chrom):  # type: ignore[no-untyped-def]
        calls.append(chrom)
        raise AssertionError("evaluation must not start")

    cfg = _make_config(
        indicator_pool=["FUNDING_Z", "RSI"], symbols=["BTCUSDT"], max_workers=1,
        start_date="2024-01-01", end_date="2024-06-30",
    )
    with pytest.raises(AuxDataUnavailableError, match="refresh funding data"):
        DiscoveryPipeline(cfg, counting).run()
    assert calls == []


@pytest.mark.parametrize("max_workers", [1, 2])
def test_data_unavailable_in_evaluation_propagates_out_of_run(
    funding_archive: Path, _restore_ga_pool: None, max_workers: int
) -> None:
    from tests.unit.test_discovery_pipeline import _make_config
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    cfg = _make_config(
        indicator_pool=["RSI"], symbols=["BTCUSDT"], max_workers=max_workers,
        population_size=4, max_generations=1,
    )
    with pytest.raises(AuxDataUnavailableError, match="simulated"):
        DiscoveryPipeline(cfg, _raising_backtest).run()


def test_nt_backtest_fn_reraises_data_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    fn = NTBacktestFn(["BTCUSDT"], "4h", "2023-01-01", "2023-12-31")

    def boom(*_a: object, **_k: object) -> dict[str, float | int]:
        raise AuxDataUnavailableError("Funding archive for BTCUSDT starts after")

    monkeypatch.setattr(fn, "_run_single", boom)
    with pytest.raises(AuxDataUnavailableError):
        fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]
