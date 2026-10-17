"""Carry CLI: ``run``, the six output files, the trials ledger, exit codes (task T6a).

Contract rows R39 (``--out`` exists without/with ``--force``), R40 (the trials
ledger path is injectable and the real ``data/portfolio/trials.jsonl`` is never
touched by tests), R52 (non-finite numbers become ``null``, so both JSON
payloads parse strictly) and R53 (``--force`` may replace a dir only if it is a
previous carry output), plus the failure-handling table of ``design.md`` §1.4
(exactly ``DataUnavailableError``/``CarryEquityTooSmall``/``CarryExecutionError``/
``ValueError`` -> ONE stderr line, exit 2; no ``except Exception``), §4's six
outputs and the §8-T6a stdout line. Round 2 adds the append-by-line test (SF1:
the ledger is appended, never rewritten) and the check-order test (SF2: an
existing ``--out`` is refused before ``load_carry_inputs``).

The fixture is synthetic: four weekly decisions over four Mondays (so the
headline weekly Sharpe is a real number, not the R50 null) with a funding-rate
switch every week (so the book flips each week), flat sizing ticks at the daily
boundary marks, and varying fill ticks so the equity path -- and the weekly
returns -- really move. It is copied from the engine/reference test builders on
purpose (not imported): a CLI test that reaches into another test file's fixture
breaks when that file changes.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import vibe_quant.portfolio.__main__ as carry_cli
from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.portfolio.metrics import carry_metrics
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.sizing import InstrumentSpec
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import CarryInputs, PriceTick
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.metrics import CarryReport

_MIN = 60 * 10**9
_HOUR = 3_600 * 10**9
_DAY = 86_400 * 10**9
_WEEK = 7 * _DAY
_EIGHT_HOURS = 8 * _HOUR
_BAR_TS_INIT = 59_999_000_000  # catalog 1m bar: ts_init = open + 59.999 s
_JITTER_MS = 3  # archived funding_time carries a few ms after the boundary
_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_SIZING = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0}  # flat: the daily boundary marks
_FILL = {
    "BTCUSDT": (51010.0, 52010.0, 51010.0, 52010.0),
    "ETHUSDT": (3010.0, 3110.0, 3010.0, 3110.0),
}
# Week-parity rates: the low-funding leg alternates every week, so the book flips.
_RATES = {"BTCUSDT": (0.0001, 0.0002), "ETHUSDT": (0.0002, 0.0001)}

_START_DATE = "2025-01-06"  # a Monday 00:00 UTC
_END_DATE = "2025-02-03"  # START + 4 weeks


def _ns(date: str) -> int:
    """``YYYY-MM-DD`` 00:00 UTC -> epoch ns."""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()) * 10**9


_START = _ns(_START_DATE)
_END = _ns(_END_DATE)
_DECISIONS = decision_times(_START, _END)
_SIX_FILES = {
    "summary.json",
    "daily.parquet",
    "weekly.parquet",
    "fills.parquet",
    "funding.parquet",
    "decisions.parquet",
}
_REAL_TRIALS = carry_cli.REPO_ROOT / "data/portfolio/trials.jsonl"


def _boundaries() -> list[int]:
    return list(range(_START - 9 * _DAY, _END, _EIGHT_HOURS))


def _rate(symbol: str, boundary: int) -> float:
    """The symbol's rate in the boundary's week: low and high swap every week."""
    return _RATES[symbol][((boundary - _START) // _WEEK) % 2]


def _marks(price: float) -> MarkSeries:
    """A flat 1m MARK series over the fixture window (one bar opening per minute)."""
    ts = np.arange(_START - _EIGHT_HOURS, _END, _MIN, dtype=np.int64)
    values = np.full(ts.shape, price, dtype=np.float64)
    return MarkSeries(ts=ts, low=values, high=values, close=values, open=values)


def _ticks() -> dict[str, tuple[PriceTick, ...]]:
    """Per decision: the sizing tick (``D-1 ms``) then the fill tick (``D+59.999 s``)."""
    return {
        symbol: tuple(
            tick
            for index, decision_ns in enumerate(_DECISIONS)
            for tick in (
                PriceTick(decision_ns - 1_000_000, _SIZING[symbol]),
                PriceTick(decision_ns + _BAR_TS_INIT, _FILL[symbol][index]),
            )
        )
        for symbol in _SYMBOLS
    }


def _archive(path: Path) -> FundingCalculator:
    """A tmp funding archive: 8h rows per symbol, ms-jittered like Binance."""
    boundaries = _boundaries()
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for symbol in _SYMBOLS:
        for boundary in boundaries:
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
                (symbol, boundary // 1_000_000 + _JITTER_MS, _rate(symbol, boundary)),
            )
    conn.commit()
    conn.close()
    return FundingCalculator(path)


def _inputs(tmp_path: Path) -> CarryInputs:
    """A four-week two-symbol ``CarryInputs``: four decisions, a flip every week."""
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=_START, end_ns=_END)
    boundaries = _boundaries()
    instruments = {symbol: create_instrument(symbol) for symbol in _SYMBOLS}
    return CarryInputs(
        spec=spec,
        decisions=tuple(_DECISIONS),
        instruments=instruments,
        specs={
            symbol: InstrumentSpec.from_instrument(instrument)
            for symbol, instrument in instruments.items()
        },
        ticks=_ticks(),
        settlements={
            symbol: Settlements(
                times_ns=tuple(boundary + _JITTER_MS * 10**6 for boundary in boundaries),
                rates=tuple(_rate(symbol, boundary) for boundary in boundaries),
            )
            for symbol in _SYMBOLS
        },
        marks={symbol: _marks(_SIZING[symbol]) for symbol in _SYMBOLS},
        calc=_archive(tmp_path / "raw_data.db"),
        # Every daily boundary, so a leg held at B is marked (metrics fails closed otherwise).
        boundary_prices={
            symbol: {boundary: _SIZING[symbol] for boundary in range(_START, _END + _DAY, _DAY)}
            for symbol in _SYMBOLS
        },
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _reject_constant(token: str) -> float:
    """``json.loads`` hook: a NaN/Infinity literal in our output is a bug (R52)."""
    raise AssertionError(f"non-finite JSON literal {token} in the CLI output")


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, inputs: CarryInputs) -> None:
    monkeypatch.setattr(
        carry_cli, "load_carry_inputs", lambda spec, catalog, archive: inputs
    )


def _raise(error: BaseException) -> Callable[..., CarryInputs]:
    def loader(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        raise error

    return loader


def _run_args(
    tmp_path: Path, *extra: str, out: Path | None = None, trials: Path | None = None
) -> list[str]:
    """A ``run`` argv; ``--out``/``--trials`` are omitted entirely when ``None``."""
    args = ["run", "--symbols", "BTCUSDT", "ETHUSDT", "--start", _START_DATE, "--end", _END_DATE]
    if out is not None:
        args += ["--out", str(out)]
    if trials is not None:
        args += ["--trials", str(trials)]
    return [*args, *extra]


@pytest.mark.parametrize("engine", ["reference", "nt"])
def test_run_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], engine: str
) -> None:
    """R39/R40 + §4: six files, the summary schema, the stdout line, one ledger line."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"
    real_before = _read(_REAL_TRIALS)

    code = carry_cli.main(_run_args(tmp_path, "--engine", engine, out=out_dir, trials=trials))

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert {path.name for path in out_dir.iterdir()} == _SIX_FILES

    summary = json.loads((out_dir / "summary.json").read_text())
    assert set(summary) == {
        "spec",
        "spec_hash",
        "engine",
        "git_sha",
        "liquidation",
        "headline",
        "counts",
        "decomposition",
        "by_year",
    }
    assert summary["engine"] == engine
    assert summary["liquidation"] == "not_modelled"
    assert summary["spec"] == inputs.spec.to_json()
    assert summary["spec_hash"] == inputs.spec.spec_hash()
    assert summary["git_sha"]
    assert set(summary["headline"]) == {
        "weekly_sharpe",
        "weekly_sharpe_lo_se",
        "weeks",
        "daily_sharpe_365",
        "total_return",
        "max_drawdown_daily",
    }
    assert set(summary["counts"]) == {
        "n_fills",
        "n_decisions",
        "flat_decisions",
        "band_skips",
        "min_notional_skips",
    }
    assert set(summary["decomposition"]["totals"]) == {
        "price_pnl_total",
        "funding_paid_total",
        "fees_total",
        "turnover_total",
        "turnover_annual",
    }
    assert set(summary["decomposition"]["funding"]) == {
        "funding_fallback_settlements",
        "funding_estimated",
        "funding_fallback_amount",
    }
    assert set(summary["decomposition"]["rounding_drift"]) == {
        "rounding_drift_mean",
        "rounding_drift_max",
    }
    # Four Mondays -> four weekly returns, so the headline is a number, not the R50 null.
    assert summary["headline"]["weekly_sharpe"] is not None
    assert len(summary["by_year"]) == 1  # the whole fixture is inside 2025

    daily = pd.read_parquet(out_dir / "daily.parquet")
    weekly = pd.read_parquet(out_dir / "weekly.parquet")
    fills = pd.read_parquet(out_dir / "fills.parquet")
    funding = pd.read_parquet(out_dir / "funding.parquet")
    decisions = pd.read_parquet(out_dir / "decisions.parquet")
    assert list(daily.columns) == ["ts", "equity", "return"]
    assert list(weekly.columns) == ["ts", "return"]
    assert list(fills.columns) == ["ts", "symbol", "steps", "qty", "price", "fee", "reduce_only"]
    assert list(funding.columns) == ["ts", "symbol", "amount", "fallback"]
    assert len(daily) == 28  # one row per day of the four-week window
    assert len(weekly) == 4
    assert len(decisions) == summary["counts"]["n_decisions"] == 4
    assert len(fills) == summary["counts"]["n_fills"]
    assert math.fsum(fills["fee"]) == summary["decomposition"]["totals"]["fees_total"]
    assert (
        math.fsum(funding["amount"]) == summary["decomposition"]["totals"]["funding_paid_total"]
    )
    assert (
        int(funding["fallback"].sum())
        == summary["decomposition"]["funding"]["funding_fallback_settlements"]
    )

    # The §8-T6a stdout line, verbatim, from the numbers the run actually wrote.
    headline = summary["headline"]
    totals = summary["decomposition"]["totals"]
    funding_block = summary["decomposition"]["funding"]
    counts = summary["counts"]
    assert captured.out == (
        f"carry[{engine}]: weekly_sharpe={headline['weekly_sharpe']!r} "
        f"total_return={headline['total_return']!r} "
        f"max_dd={headline['max_drawdown_daily']!r} "
        f"fills={counts['n_fills']!r} "
        f"funding_paid={totals['funding_paid_total']!r} "
        f"fees={totals['fees_total']!r} "
        f"fallback={funding_block['funding_fallback_settlements']!r} "
        f"out={out_dir}\n"
    )

    lines = trials.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["spec_hash"] == inputs.spec.spec_hash()
    assert record["spec"] == inputs.spec.to_json()
    assert record["engine"] == engine
    assert record["weeks"] == 4
    assert record["weekly_sharpe"] == headline["weekly_sharpe"]
    assert record["git_sha"] == summary["git_sha"]
    assert isinstance(record["ts"], int)
    assert record["created_at"].startswith("2025-") or record["created_at"].startswith("20")
    assert _read(_REAL_TRIALS) == real_before  # R40: the real ledger is untouched


@pytest.mark.parametrize(
    "error",
    [
        CarryDataError("BTCUSDT: no 1m bars in catalog /tmp/catalog"),
        CarryEquityTooSmall("<book>", _DECISIONS[0], 5.0, 40.0),
        CarryExecutionError("stale release BTCUSDT: queued at 1, released at 2"),
        ValueError("band must be in [0.0, 1.0), got 1.0"),
    ],
    ids=["data", "equity", "execution", "value"],
)
def test_known_error_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    """§1.4: each known error type -> one stderr line, exit 2, nothing written."""
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _raise(error))
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"

    assert carry_cli.main(_run_args(tmp_path, out=out_dir, trials=trials)) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert captured.err.startswith("carry[nt]: ")
    assert type(error).__name__ in captured.err
    assert str(error) in captured.err
    assert not out_dir.exists()
    assert not trials.exists()


def test_unexpected_error_is_not_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1.4: there is no ``except Exception`` -- an unexpected error propagates."""
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _raise(ZeroDivisionError("boom")))

    with pytest.raises(ZeroDivisionError, match="boom"):
        carry_cli.main(_run_args(tmp_path))


def test_out_exists_requires_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R39/R53 + SF2: the ``--out`` check runs before any load, and replaces only carry dirs."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"

    assert carry_cli.main(_run_args(tmp_path, out=out_dir, trials=trials)) == 0
    capsys.readouterr()
    assert {path.name for path in out_dir.iterdir()} == _SIX_FILES

    # SF2: an existing --out is refused BEFORE load_carry_inputs (no input load, no engine run).
    def loader(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        pytest.fail("load_carry_inputs must not run before the R39 check")

    monkeypatch.setattr(carry_cli, "load_carry_inputs", loader)
    assert carry_cli.main(_run_args(tmp_path, out=out_dir, trials=trials)) == 2

    captured = capsys.readouterr()
    assert len(captured.err.splitlines()) == 1
    assert "exists; pass --force" in captured.err
    assert {path.name for path in out_dir.iterdir()} == _SIX_FILES
    assert len(trials.read_text().splitlines()) == 1

    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("keep me")
    _patch_inputs(monkeypatch, inputs)
    assert carry_cli.main(_run_args(tmp_path, "--force", out=out_dir, trials=trials)) == 0

    capsys.readouterr()
    assert not sentinel.exists()
    assert {path.name for path in out_dir.iterdir()} == _SIX_FILES
    assert len(trials.read_text().splitlines()) == 2
    # the staging dir and the ``.old`` dir it is renamed to are both gone (R39 nit)
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []


def test_out_dir_appearing_during_the_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R39 late guard (SF2): a dir that appears at ``--out`` after the early check is not replaced.

    The early check cannot see a dir that another process creates while the data is
    loading, so the write path re-checks: without ``--force`` it must raise the same
    ``ValueError`` (exit 2) and leave the appeared dir's content untouched.
    """
    inputs = _inputs(tmp_path)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"

    def loader(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        out_dir.mkdir()  # appears AFTER the early _check_out_dir, during the input load
        (out_dir / "precious.txt").write_text("keep me")
        return inputs

    monkeypatch.setattr(carry_cli, "load_carry_inputs", loader)

    assert (
        carry_cli.main(_run_args(tmp_path, "--engine", "reference", out=out_dir, trials=trials)) == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert "exists; pass --force" in captured.err
    assert {path.name for path in out_dir.iterdir()} == {"precious.txt"}
    assert (out_dir / "precious.txt").read_text() == "keep me"
    assert not trials.exists()
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []


def test_replace_dir_restores_the_old_dir_when_the_second_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: a failed ``staging.rename(out_dir)`` restores the old dir and propagates.

    ``_replace_dir`` moves the existing ``out_dir`` aside, then renames the staged
    dir in. If that second rename fails (the ``finally`` in the fix would instead
    ``rmtree`` the aside) the old output must be renamed back into place -- the
    ``aside -> out_dir`` restore -- before the error is re-raised.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / carry_cli.SUMMARY_FILE).write_text("old")
    (out_dir / "keep.txt").write_text("keep me")
    staging = tmp_path / f".{out_dir.name}.tmp-x"
    staging.mkdir()
    (staging / carry_cli.SUMMARY_FILE).write_text("new")

    real_rename = Path.rename
    calls = {"n": 0}

    def flaky_rename(self: Path, target: Path) -> Path:
        calls["n"] += 1
        if calls["n"] == 2:  # the staging -> out_dir rename, after out_dir -> aside
            raise OSError("rename failed")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    with pytest.raises(OSError, match="rename failed"):
        carry_cli._replace_dir(staging, out_dir)

    assert calls["n"] == 3  # aside -> out_dir restore is the third rename
    assert (out_dir / "keep.txt").read_text() == "keep me"
    assert (out_dir / carry_cli.SUMMARY_FILE).read_text() == "old"


def test_replace_dir_restores_the_old_dir_on_a_non_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4b: the restore must trigger on ANY exception, not only ``OSError``.

    A ``RuntimeError`` raised by the second rename (a callback, an interrupt handler,
    a ``KeyboardInterrupt``-like abort) is not an ``OSError``, so an
    ``except OSError`` restore leaves ``out_dir`` missing and the old output stranded
    at the aside path. The old dir must be renamed back and the error re-raised.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / carry_cli.SUMMARY_FILE).write_text("old")
    (out_dir / "keep.txt").write_text("keep me")
    staging = tmp_path / f".{out_dir.name}.tmp-x"
    staging.mkdir()
    (staging / carry_cli.SUMMARY_FILE).write_text("new")

    real_rename = Path.rename
    calls = {"n": 0}

    def flaky_rename(self: Path, target: Path) -> Path:
        calls["n"] += 1
        if calls["n"] == 2:  # the staging -> out_dir rename, after out_dir -> aside
            raise RuntimeError("rename boom")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    with pytest.raises(RuntimeError, match="rename boom"):
        carry_cli._replace_dir(staging, out_dir)

    assert calls["n"] == 3  # aside -> out_dir restore is the third rename
    assert (out_dir / "keep.txt").read_text() == "keep me"
    assert (out_dir / carry_cli.SUMMARY_FILE).read_text() == "old"


def test_write_outputs_failure_leaves_the_old_dir_and_no_staging(tmp_path: Path) -> None:
    """R4: a frame writer that raises mid-write never touches the old ``out_dir``.

    The staged dir is built beside ``out_dir`` and only renamed in at the end, so a
    failure while writing one parquet file leaves the previous output intact and the
    staging dir cleaned up.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / carry_cli.SUMMARY_FILE).write_text("old")

    class _BoomFrame:
        def to_parquet(self, path: object, index: bool = False) -> None:
            raise RuntimeError("frame boom")

    with pytest.raises(RuntimeError, match="frame boom"):
        carry_cli._write_outputs(
            out_dir, {"weekly_sharpe": 1.0}, {"daily.parquet": _BoomFrame()}, force=True
        )

    assert {path.name for path in out_dir.iterdir()} == {carry_cli.SUMMARY_FILE}
    assert (out_dir / carry_cli.SUMMARY_FILE).read_text() == "old"
    assert list(tmp_path.glob(f".{out_dir.name}.tmp-*")) == []


def test_force_refuses_a_non_carry_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R53: ``--force`` replaces a dir only if it is a carry output (holds ``summary.json``)."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    keep = out_dir / "trials.jsonl"
    keep.write_text("previous run\n")
    trials = tmp_path / "trials.jsonl"

    assert carry_cli.main(_run_args(tmp_path, "--force", out=out_dir, trials=trials)) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert f"refusing to replace {out_dir}: not a carry output dir" in captured.err
    assert keep.read_text() == "previous run\n"
    assert {path.name for path in out_dir.iterdir()} == {"trials.jsonl"}
    assert not trials.exists()


def test_trials_ledger_appends_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SF1/R40: every successful run APPENDS one line; a failed run appends nothing."""
    inputs = _inputs(tmp_path)
    trials = tmp_path / "trials.jsonl"
    real_before = _read(_REAL_TRIALS)

    def loader(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        return replace(inputs, spec=spec)  # the run's own spec, so the two lines differ

    monkeypatch.setattr(carry_cli, "load_carry_inputs", loader)
    assert (
        carry_cli.main(
            _run_args(
                tmp_path, "--engine", "reference", "--band", "0.10", out=tmp_path / "a", trials=trials
            )
        )
        == 0
    )
    first = trials.read_text()

    assert (
        carry_cli.main(
            _run_args(
                tmp_path, "--engine", "reference", "--band", "0.05", out=tmp_path / "b", trials=trials
            )
        )
        == 0
    )
    capsys.readouterr()

    assert trials.read_text().startswith(first)  # the first line is untouched
    lines = trials.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["spec_hash"] != json.loads(lines[1])["spec_hash"]
    assert json.loads(lines[0])["spec"] != json.loads(lines[1])["spec"]

    monkeypatch.setattr(carry_cli, "load_carry_inputs", _raise(CarryDataError("no archive")))
    assert carry_cli.main(_run_args(tmp_path, out=tmp_path / "b", trials=trials)) == 2

    capsys.readouterr()
    assert len(trials.read_text().splitlines()) == 2  # a failed run appends nothing
    assert _read(_REAL_TRIALS) == real_before


def test_non_finite_numbers_are_written_as_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R52: NaN/inf become ``null`` so ``summary.json`` and the trial line stay valid JSON."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    real_metrics = carry_metrics

    def metrics(run: CarryRun, inputs: CarryInputs) -> CarryReport:
        report = real_metrics(run, inputs)
        return replace(
            report,
            summary={
                **report.summary,
                "weekly_sharpe": math.nan,
                "max_drawdown_daily": math.inf,
            },
            by_year=[{**row, "weekly_sharpe": math.nan} for row in report.by_year],
        )

    monkeypatch.setattr(carry_cli, "carry_metrics", metrics)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"

    assert (
        carry_cli.main(_run_args(tmp_path, "--engine", "reference", out=out_dir, trials=trials)) == 0
    )

    capsys.readouterr()
    summary = json.loads((out_dir / "summary.json").read_text(), parse_constant=_reject_constant)
    assert summary["headline"]["weekly_sharpe"] is None
    assert summary["headline"]["max_drawdown_daily"] is None
    assert summary["by_year"][0]["weekly_sharpe"] is None
    record = json.loads(trials.read_text(), parse_constant=_reject_constant)
    assert record["weekly_sharpe"] is None
    assert isinstance(record["ts"], int)


def test_relative_out_and_trials_resolve_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§8-T6a: relative ``--out``/``--trials`` are repo-root-relative, never cwd-relative."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    root = tmp_path / "root"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(carry_cli, "REPO_ROOT", root)
    monkeypatch.chdir(elsewhere)

    assert (
        carry_cli.main(
            _run_args(
                tmp_path,
                "--engine",
                "reference",
                out=Path("rel/out"),
                trials=Path("rel/trials.jsonl"),
            )
        )
        == 0
    )

    capsys.readouterr()
    assert {path.name for path in (root / "rel/out").iterdir()} == _SIX_FILES
    assert len((root / "rel/trials.jsonl").read_text().splitlines()) == 1
    assert not (elsewhere / "rel").exists()


def test_default_paths_resolve_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§8-T6a: the defaults, and a relative ``--catalog``/``--archive``, resolve from the root."""
    inputs = _inputs(tmp_path)
    seen: list[tuple[Path, Path]] = []

    def loader(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        seen.append((catalog, archive))
        return inputs

    monkeypatch.setattr(carry_cli, "load_carry_inputs", loader)
    monkeypatch.chdir(tmp_path)  # cwd must not change where the defaults point

    assert carry_cli.main(
        _run_args(
            tmp_path,
            "--engine",
            "reference",
            out=tmp_path / "out",
            trials=tmp_path / "trials.jsonl",
        )
    ) == 0
    assert carry_cli.main(
        _run_args(
            tmp_path,
            "--engine",
            "reference",
            "--catalog",
            "rel/catalog",
            "--archive",
            "rel/archive.db",
            out=tmp_path / "out2",
            trials=tmp_path / "trials2.jsonl",
        )
    ) == 0

    capsys.readouterr()
    assert seen == [
        (carry_cli.REPO_ROOT / "data/catalog", carry_cli.REPO_ROOT / "data/archive/raw_data.db"),
        (carry_cli.REPO_ROOT / "rel/catalog", carry_cli.REPO_ROOT / "rel/archive.db"),
    ]


def test_default_out_dir_and_ledger_derive_from_the_out_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§4/R40: default ``--out`` is ``<out root>/<engine>-<start>-<end>-<sha8>`` + ``trials.jsonl``."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    monkeypatch.setattr(carry_cli, "REPO_ROOT", tmp_path)

    args = _run_args(tmp_path, "--engine", "reference")
    assert "--out" not in args and "--trials" not in args  # both take their defaults
    assert carry_cli.main(args) == 0

    capsys.readouterr()
    expected = tmp_path / "data/portfolio" / f"reference-{_START_DATE}-{_END_DATE}-{inputs.spec.spec_hash()[:8]}"
    assert {path.name for path in expected.iterdir()} == _SIX_FILES
    ledger = tmp_path / "data/portfolio/trials.jsonl"
    assert len(ledger.read_text().splitlines()) == 1


def test_default_engine_is_nt() -> None:
    """§8-T6a: the live default is the NT engine (the reference is the oracle)."""
    args = carry_cli.build_parser().parse_args(
        ["run", "--symbols", "BTCUSDT", "ETHUSDT", "--start", _START_DATE, "--end", _END_DATE]
    )

    assert args.engine == "nt"
    assert args.force is False
    assert args.out is None
    assert args.trials is None
    assert args.catalog == "data/catalog"
    assert args.archive == "data/archive/raw_data.db"
