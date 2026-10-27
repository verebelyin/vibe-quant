"""Catalog / funding-archive paths reach every tier's engine and fail closed.

WFA takes the catalog its caller names (as purged k-fold does), validation takes
catalog and funding archive from its constructor (as screening does), and a missing
or unreadable funding archive stops the run instead of charging the fallback rate.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.errors import DataUnavailableError

_TEMPLATES = Path(__file__).resolve().parents[2] / "vibe_quant" / "strategies" / "templates"


def _wfa_db(tmp_path: Path) -> Path:
    """Minimal state DB: one 4h strategy run on BTCUSDT for NTWFARunner to resolve."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT NOT NULL, dsl_config TEXT NOT NULL);
        CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, strategy_id INTEGER,
                                    symbols TEXT NOT NULL, parameters JSON NOT NULL);
        CREATE TABLE sweep_results (id INTEGER PRIMARY KEY, run_id INTEGER, starting_balance REAL);
        CREATE TABLE backtest_results (id INTEGER PRIMARY KEY, run_id INTEGER, starting_balance REAL);
        """
    )
    dsl = validate_strategy_dict(
        yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
    ).model_dump()
    conn.execute("INSERT INTO strategies VALUES (1, 'stub', ?)", (json.dumps(dsl),))
    conn.execute("INSERT INTO backtest_runs VALUES (42, 1, ?, '{}')", (json.dumps(["BTCUSDT"]),))
    conn.execute("INSERT INTO backtest_results (run_id, starting_balance) VALUES (42, 1000.0)")
    conn.commit()
    conn.close()
    return db


# --- nxmxd.8: WFA gets the catalog the caller names ---------------------------------


@pytest.mark.parametrize("name", ["empty_cat_a", "other_catalog_b"])
def test_wfa_backtest_runs_on_the_named_catalog(tmp_path: Path, name: str) -> None:
    """A non-default catalog with no bars stops the WFA window, naming that catalog."""
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    catalog = tmp_path / name
    catalog.mkdir()
    runner = NTWFARunner(42, _wfa_db(tmp_path), catalog_path=catalog)
    with pytest.raises(DataUnavailableError) as err:
        runner.backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})
    assert str(catalog) in str(err.value)


@pytest.mark.parametrize("name", ["empty_cat_c", "other_catalog_d"])
def test_wfa_engine_loads_bars_from_the_named_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """With the fill-tick set out of the way, the production runner itself reads the catalog."""
    from vibe_quant.overfitting.nt_runner import NTWFARunner
    from vibe_quant.screening.pipeline import ParentFillTicks

    monkeypatch.setattr(ParentFillTicks, "get", lambda self, start, end: None)
    catalog = tmp_path / name
    catalog.mkdir()
    runner = NTWFARunner(42, _wfa_db(tmp_path), catalog_path=catalog)
    with pytest.raises(DataUnavailableError, match="No catalog bars") as err:
        runner.backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})
    assert str(catalog) in str(err.value)


def test_wfa_without_a_catalog_keeps_the_default(tmp_path: Path) -> None:
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    runner = NTWFARunner(42, _wfa_db(tmp_path))
    assert runner._catalog_path is None  # noqa: SLF001


@pytest.mark.parametrize("name", ["cat_one", "cat_two"])
def test_overfitting_cli_gives_wfa_and_cv_the_same_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """``--catalog-path`` reaches BOTH real runners; without it neither gets one."""
    import vibe_quant.overfitting.__main__ as cli
    import vibe_quant.overfitting.nt_cv_runner as cv_mod
    import vibe_quant.overfitting.nt_runner as wfa_mod

    seen: dict[str, Any] = {}

    class _WFA:
        def __init__(self, run_id: int, db_path: Any, catalog_path: Any = None) -> None:
            seen["wfa"] = catalog_path

    class _CV:
        def __init__(self, run_id: int, db_path: Any, catalog_path: Any = None) -> None:
            seen["cv"] = catalog_path

    class _Pipeline:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def run(self, **k: Any) -> Any:
            raise RuntimeError("stop after wiring")

        def close(self) -> None:
            pass

    monkeypatch.setattr(wfa_mod, "NTWFARunner", _WFA)
    monkeypatch.setattr(cv_mod, "NTPurgedKFoldRunner", _CV)
    monkeypatch.setattr(cli, "OverfittingPipeline", _Pipeline)
    base = ["run", "--run-id", "1", "--db", str(tmp_path / "x.db"), "--real-wfa", "--real-cv"]

    with pytest.raises(RuntimeError, match="stop after wiring"):
        cli.main([*base, "--catalog-path", str(tmp_path / name)])
    assert seen == {"wfa": tmp_path / name, "cv": tmp_path / name}

    seen.clear()
    with pytest.raises(RuntimeError, match="stop after wiring"):
        cli.main(base)
    assert seen == {"wfa": None, "cv": None}


# --- nxmxd.6: validation takes catalog and funding archive like screening --------------

_H_MS = 3_600_000
_DAY0_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z


def _funding_archive(path: Path, first_ms: int, count: int, rate: float = 0.0001) -> Path:
    """An archive with ``count`` 8h BTCUSDT settlements starting at ``first_ms``."""
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        for i in range(count):
            archive.conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, source) "
                "VALUES ('BTCUSDT', ?, ?, 'test')",
                (first_ms + i * 8 * _H_MS, rate),
            )
        archive.conn.commit()
    finally:
        archive.close()
    return path


def _validation_dsl(indicator: str = "RSI") -> dict[str, object]:
    if indicator == "RSI":
        ind: dict[str, object] = {"rsi": {"type": "RSI", "period": 14}}
        cond = "rsi"
    else:
        ind = {"rsi": {"type": "FUNDING_Z", "period": 30}}
        cond = "rsi"
    return {
        "name": f"plumb_probe_{indicator.lower()}",
        "timeframe": "4h",
        "indicators": ind,
        "entry_conditions": {"long": [f"{cond} < 30"]},
        "exit_conditions": {"long": [f"{cond} > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


def _validation_run(tmp_path: Path, dsl: dict[str, object]) -> tuple[Path, int]:
    from vibe_quant.db.state_manager import StateManager

    db = tmp_path / "s.db"
    state = StateManager(db)
    sid = state.create_strategy(name=str(dsl["name"]), dsl_config=dsl)
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters={},
    )
    state.close()
    return db, run_id


class _Captured(Exception):  # noqa: N818 - control-flow sentinel, not an error
    pass


class _FakeNode:
    """Records the BacktestRunConfig and stops before any engine work."""

    seen: list[Any] = []  # noqa: RUF012 - per-test capture, reset by fixture

    def __init__(self, configs: list[Any]) -> None:
        _FakeNode.seen.extend(configs)

    def build(self) -> None:
        raise _Captured

    def get_engines(self) -> list[Any]:
        return []

    def dispose(self) -> None:
        pass


@pytest.fixture
def one_minute_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every catalog 'has' 1m coverage and the node only captures its config.

    The catalog PATH is what these tests observe, so coverage and bar checks are stubbed
    and the engine never builds.
    """
    from datetime import UTC, datetime

    import nautilus_trader.backtest.node as nt_node

    from vibe_quant.data.catalog import CatalogManager
    from vibe_quant.screening import nt_runner

    def date_range(self: Any, symbol: str, interval: str) -> Any:
        covered = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))
        return covered if interval == "1m" else None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(nt_node, "BacktestNode", _FakeNode)
    monkeypatch.setattr(nt_runner, "require_bars_in_window", lambda *a, **k: None)
    _FakeNode.seen = []


@pytest.fixture(autouse=True)
def _reset_aux_and_funding() -> Any:
    from vibe_quant.dsl import aux_data
    from vibe_quant.validation.funding import clear_rate_cache

    clear_rate_cache()
    yield
    aux_data.configure(None, None)
    clear_rate_cache()


@pytest.mark.usefixtures("one_minute_everywhere")
@pytest.mark.parametrize("name", ["val_cat_a", "elsewhere_b"])
def test_validation_loads_and_writes_the_named_catalog(tmp_path: Path, name: str) -> None:
    from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

    catalog = tmp_path / name
    db, run_id = _validation_run(tmp_path, _validation_dsl())
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog)
    with pytest.raises(ValidationRunnerError, match="_Captured"):
        runner.run(run_id=run_id)
    runner.close()
    (cfg,) = _FakeNode.seen
    assert {d.catalog_path for d in cfg.data} == {str(catalog.resolve())}
    assert (catalog / "data" / "crypto_perpetual" / "BTCUSDT-PERP.BINANCE").is_dir()


@pytest.mark.usefixtures("one_minute_everywhere")
def test_validation_does_not_rewrite_an_existing_instrument(tmp_path: Path) -> None:
    from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

    catalog = tmp_path / "cat"
    stamps: list[dict[str, int]] = []
    for _ in range(2):
        db, run_id = _validation_run(tmp_path / f"r{len(stamps)}", _validation_dsl())
        runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog)
        with pytest.raises(ValidationRunnerError, match="_Captured"):
            runner.run(run_id=run_id)
        runner.close()
        files = (catalog / "data" / "crypto_perpetual" / "BTCUSDT-PERP.BINANCE").rglob("*.parquet")
        stamps.append({str(f): f.stat().st_mtime_ns for f in files})
    assert stamps[0]
    assert stamps[0] == stamps[1]


@pytest.mark.parametrize("name", ["no_bars_a", "no_bars_b"])
def test_validation_window_is_clamped_against_the_named_catalog(
    tmp_path: Path, name: str
) -> None:
    """The default catalog has 1m BTC bars; the named empty one does not, so the run fails."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

    db, run_id = _validation_run(tmp_path, _validation_dsl())
    runner = ValidationRunner(
        db_path=db, logs_path=tmp_path / "logs", catalog_path=tmp_path / name
    )
    with pytest.raises(ValidationRunnerError, match="No 1m bars for BTCUSDT"):
        runner.run(run_id=run_id)
    runner.close()
    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"


@pytest.mark.usefixtures("one_minute_everywhere")
@pytest.mark.parametrize(
    ("first_ms", "count", "expected"),
    [
        (_DAY0_MS, 40, "ends before the requested window"),  # Jan 2024 only
        (_DAY0_MS + 400 * 24 * _H_MS, 200, "starts after the requested window"),
    ],
)
def test_validation_preflights_context_funding_on_the_named_archive(
    tmp_path: Path, first_ms: int, count: int, expected: str
) -> None:
    """The default archive covers Jan 2025; the named one does not, so the run is refused."""
    from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

    archive = _funding_archive(tmp_path / "named.db", first_ms, count)
    db, run_id = _validation_run(tmp_path, _validation_dsl("FUNDING_Z"))
    runner = ValidationRunner(
        db_path=db,
        logs_path=tmp_path / "logs",
        catalog_path=tmp_path / "cat",
        funding_archive_path=archive,
    )
    with pytest.raises(ValidationRunnerError, match=expected):
        runner.run(run_id=run_id)
    runner.close()
    assert _FakeNode.seen == []


@pytest.mark.parametrize(("rate", "other"), [(0.0004, 0.0009), (0.0013, 0.0002)])
def test_validation_extraction_accrues_funding_from_the_named_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: float, other: float
) -> None:
    import vibe_quant.validation.extraction as extraction
    from vibe_quant.validation.runner import ValidationRunner

    named = _funding_archive(tmp_path / "named.db", _DAY0_MS, 3, rate)
    _funding_archive(tmp_path / "unnamed.db", _DAY0_MS, 3, other)
    seen: dict[str, Any] = {}

    def fake_extract(*args: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "extracted"

    monkeypatch.setattr(extraction, "extract_results", fake_extract)
    runner = ValidationRunner(
        db_path=tmp_path / "s.db", logs_path=tmp_path / "logs", funding_archive_path=named
    )
    runner._extract_results(  # noqa: SLF001
        run_id=1, strategy_name="x", bt_result=None, engine=None, venue_config=None  # type: ignore[arg-type]
    )
    runner.close()
    assert seen["funding_calculator"].settlements("BTCUSDT")[1] == (rate, rate, rate)


@pytest.mark.parametrize("name", ["detail_cat_a", "detail_cat_b"])
def test_validation_sub_minute_detail_check_reads_the_named_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A 1m strategy uses 5s detail only if the run's catalog has it for every symbol."""
    from vibe_quant.data.catalog import CatalogManager
    from vibe_quant.validation.runner import ValidationRunner

    asked: list[tuple[Path, str, str]] = []

    def no_bars(self: CatalogManager, symbol: str, interval: str) -> None:
        asked.append((self._catalog_path, symbol, interval))  # noqa: SLF001

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", no_bars)
    runner = ValidationRunner(
        db_path=tmp_path / "s.db", logs_path=tmp_path / "logs", catalog_path=tmp_path / name
    )
    run_config: dict[str, object] = {
        "symbols": json.dumps(["BTCUSDT"]),
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
    }
    assert runner._resolve_detail_timeframe(run_config, "1m", None) is None  # noqa: SLF001
    runner.close()
    assert asked == [(tmp_path / name, "BTCUSDT", "5s")]


# --- nxmxd.1: a missing or unreadable funding archive fails closed ----------------------

_REL_ARCHIVE = Path("data/archive/raw_data.db")
_NS_PER_MS = 1_000_000


def _accrue_btc(calc: Any) -> Any:
    """One long BTC trade crossing the settlements at DAY0 + 8h and + 16h."""
    return calc.accrue(
        "BTCUSDT-PERP.BINANCE",
        "LONG",
        10_000.0,
        (_DAY0_MS + _H_MS) * _NS_PER_MS,
        (_DAY0_MS + 17 * _H_MS) * _NS_PER_MS,
    )


@pytest.mark.parametrize("subdir", ["wrong_cwd_a", "wrong_cwd_b"])
def test_missing_relative_archive_refuses_instead_of_charging_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, subdir: str
) -> None:
    """A run started from a cwd without the archive stops, naming the absolute path."""
    from vibe_quant.validation.funding import FundingArchiveUnavailableError, FundingCalculator

    cwd = tmp_path / subdir
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    calc = FundingCalculator(_REL_ARCHIVE)
    with pytest.raises(FundingArchiveUnavailableError) as err:
        _accrue_btc(calc)
    assert isinstance(err.value, DataUnavailableError)
    assert str((cwd / _REL_ARCHIVE).resolve()) in str(err.value)
    assert not (cwd / "data").exists()  # a lookup never creates an archive
    assert "fallback" not in caplog.text


def test_missing_archive_is_not_cached_as_empty(tmp_path: Path) -> None:
    """After the archive appears, the same path serves its rates (no poisoned cache)."""
    from vibe_quant.validation.funding import FundingArchiveUnavailableError, FundingCalculator

    path = tmp_path / "later.db"
    calc = FundingCalculator(path)
    with pytest.raises(FundingArchiveUnavailableError):
        calc.settlements("BTCUSDT")
    _funding_archive(path, _DAY0_MS, 3, 0.0004)
    assert calc.settlements("BTCUSDT")[1] == (0.0004, 0.0004, 0.0004)


@pytest.mark.parametrize("method", ["settlements", "settlement_hours", "accrue"])
def test_every_funding_read_refuses_a_missing_archive(tmp_path: Path, method: str) -> None:
    from vibe_quant.validation.funding import FundingArchiveUnavailableError, FundingCalculator

    calc = FundingCalculator(tmp_path / "nope.db")
    with pytest.raises(FundingArchiveUnavailableError, match="nope.db"):
        if method == "accrue":
            _accrue_btc(calc)
        else:
            getattr(calc, method)("BTCUSDT")


@pytest.mark.parametrize("marker", ["zq7-corrupt-marker", "k3x-other-bytes"])
def test_corrupt_archive_refuses_without_logging_its_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, marker: str
) -> None:
    from vibe_quant.validation.funding import FundingArchiveUnavailableError, FundingCalculator

    path = tmp_path / "corrupt.db"
    path.write_bytes((marker * 200).encode())
    calc = FundingCalculator(path)
    with pytest.raises(FundingArchiveUnavailableError, match="corrupt.db"):
        calc.settlements("BTCUSDT")
    assert marker not in caplog.text
    assert "fallback" not in caplog.text


def test_symbol_missing_from_an_existing_archive_still_charges_the_fallback(
    tmp_path: Path,
) -> None:
    """Unchanged: only an absent/unreadable archive refuses; a symbol hole is estimated."""
    from vibe_quant.validation.funding import FundingCalculator

    archive = _funding_archive(tmp_path / "other.db", _DAY0_MS, 3, 0.0004)
    accrual = _accrue_btc(FundingCalculator(archive, default_rate=0.0002))
    assert accrual.fallback_settlements == 0  # BTC is archived at 0.0004 for the 3 rows
    eth = FundingCalculator(archive, default_rate=0.0002).accrue(
        "ETHUSDT-PERP.BINANCE",
        "LONG",
        10_000.0,
        (_DAY0_MS + _H_MS) * _NS_PER_MS,
        (_DAY0_MS + 17 * _H_MS) * _NS_PER_MS,
    )
    assert eth.fallback_settlements == 2
    assert eth.total == pytest.approx(2 * 0.0002 * 10_000.0)


# --- An unavailable archive is an operator error at the overfitting CLI ----------------

_WFA_FLAGS = [
    "--start-date",
    "2024-01-01",
    "--end-date",
    "2024-06-01",
    "--wfa-is-days",
    "90",
    "--wfa-oos-days",
    "30",
    "--wfa-step-days",
    "30",
    "--wfa-min-windows",
    "1",
]


def _overfitting_run(tmp_path: Path) -> tuple[Path, int]:
    """A completed screening run with one sweep candidate for WFA/CV to re-check."""
    from vibe_quant.db.state_manager import StateManager

    db = tmp_path / "of.db"
    state = StateManager(db)
    try:
        dsl = validate_strategy_dict(
            yaml.safe_load((_TEMPLATES / "donchian_breakout.yaml").read_text())
        ).model_dump()
        strategy_id = state.create_strategy(name="plumb_of", dsl_config=dsl)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-06-01",
            parameters={},
        )
        state.save_sweep_result(
            run_id,
            {
                "parameters": {"donchian_period": 20},
                "sharpe_ratio": 1.2,
                "total_return": 0.1,
                "starting_balance": 1000.0,
            },
        )
        state.update_backtest_run_status(run_id, "completed")
    finally:
        state.close()
    return db, run_id


def _missing_archive_error(path: Path) -> DataUnavailableError:
    """The error the real funding read raises when the archive file is absent."""
    from vibe_quant.validation.funding import FundingArchiveUnavailableError, FundingCalculator

    try:
        FundingCalculator(path).settlements("BTCUSDT")
    except FundingArchiveUnavailableError as exc:
        return exc
    raise AssertionError(f"a missing archive at {path} was served instead of refused")


@pytest.mark.parametrize(
    ("flag", "filters", "extra", "column"),
    [
        ("--real-wfa", "wfa", _WFA_FLAGS, "passed_walk_forward"),
        ("--real-cv", "pkfold", [], "passed_purged_kfold"),
    ],
)
def test_overfitting_cli_reports_a_missing_funding_archive_as_an_operator_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    filters: str,
    extra: list[str],
    column: str,
) -> None:
    """The archive stops the run with one path-bearing line -- no traceback, no pass."""
    import vibe_quant.overfitting.__main__ as cli
    import vibe_quant.overfitting.nt_cv_runner as cv_mod
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    archive = tmp_path / "missing_archive.db"
    err = _missing_archive_error(archive)
    db, run_id = _overfitting_run(tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise err

    if flag == "--real-wfa":
        monkeypatch.setattr(NTWFARunner, "backtest", refuse)
    else:

        class _RefusingCV:
            n_samples = 200
            bars_per_day = None

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def bind_params(self, params: dict[str, object]) -> _RefusingCV:
                return self

            def run(self, train_indices: list[int], test_indices: list[int]) -> Any:
                raise err

        monkeypatch.setattr(cv_mod, "NTPurgedKFoldRunner", _RefusingCV)

    code = cli.main(
        ["run", "--run-id", str(run_id), "--db", str(db), "--filters", filters, flag, *extra]
    )
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert code != 0
    assert "Traceback" not in output
    assert str(archive) in output
    error_lines = [line for line in output.splitlines() if "not found" in line]
    assert len(error_lines) == 1
    assert str(archive) in error_lines[0]
    assert "FundingArchiveUnavailableError" in error_lines[0]

    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
        row = state.get_sweep_results(run_id)[0]
    finally:
        state.close()
    assert run is not None
    assert run["status"] == "completed"
    assert row[column] is None


@pytest.mark.parametrize("marker", ["iv-boom-1", "iv-boom-2"])
def test_unreadable_interval_table_refuses_without_logging_its_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    marker: str,
) -> None:
    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.validation.funding import FundingArchiveUnavailableError, FundingCalculator

    archive = _funding_archive(tmp_path / "iv.db", _DAY0_MS, 3)

    def boom(self: RawDataArchive, symbol: str) -> dict[int, int]:
        raise sqlite3.OperationalError(marker)

    monkeypatch.setattr(RawDataArchive, "get_funding_intervals", boom)
    with pytest.raises(FundingArchiveUnavailableError, match="iv.db"):
        FundingCalculator(archive).settlement_hours("BTCUSDT")
    assert marker not in caplog.text
