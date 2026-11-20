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
    """Six files, the summary schema, the stdout line, one ledger line."""
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
    assert _read(_REAL_TRIALS) == real_before  # The real ledger is untouched


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
    """The ``--out`` check runs before any load, and replaces only carry dirs."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"

    assert carry_cli.main(_run_args(tmp_path, out=out_dir, trials=trials)) == 0
    capsys.readouterr()
    assert {path.name for path in out_dir.iterdir()} == _SIX_FILES

    # An existing --out is refused BEFORE load_carry_inputs (no input load, no engine run).
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
    """A dir that appears at ``--out`` after the early check is not replaced.

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
    # the completed run was registered before publication and stays counted (R14d)
    assert [json.loads(line)["outcome"] for line in trials.read_text().splitlines()] == [
        "completed"
    ]
    assert list(out_dir.parent.glob(f".{out_dir.name}.tmp-*")) == []


def test_replace_dir_restores_the_old_dir_when_the_second_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``staging.rename(out_dir)`` restores the old dir and propagates.

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
    """The restore must trigger on ANY exception, not only ``OSError``.

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
    """A frame writer that raises mid-write never touches the old ``out_dir``.

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
    """``--force`` replaces a dir only if it is a carry output (holds ``summary.json``)."""
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
    """Every successful run APPENDS one line; a failed run appends nothing."""
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
    """NaN/inf become ``null`` so ``summary.json`` and the trial line stay valid JSON."""
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


# --- The DSR trial ledger: outcomes, failure rows, collisions (x0axf.14 T3) ---------------


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _no_load(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
    pytest.fail("load_carry_inputs must not run before the ledger/collision preflight")


def _evaluate_argv(*extra: str, out: Path, trials: Path) -> list[str]:
    return [
        "evaluate", "--symbols", "BTCUSDT", "ETHUSDT", "--start", _START_DATE, "--end",
        _END_DATE, "--draws", "1", "--out", str(out), "--trials", str(trials), *extra,
    ]  # fmt: skip


def test_trial_line_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R10: one completed line with ``failed_at null``, appended before the out dir exists."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"
    real_write = carry_cli._write_outputs
    seen: list[int] = []

    def write(out: Path, *args: object) -> None:
        assert not out.exists()
        seen.append(len(trials.read_text().splitlines()))
        real_write(out, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(carry_cli, "_write_outputs", write)
    assert carry_cli.main(_run_args(tmp_path, "--engine", "reference", out=out_dir, trials=trials)) == 0
    capsys.readouterr()

    assert seen == [1]
    (record,) = _records(trials)
    assert record["outcome"] == "completed"
    assert record["failed_at"] is None
    summary = json.loads((out_dir / "summary.json").read_text())
    assert record["weekly_sharpe"] == summary["headline"]["weekly_sharpe"]
    assert record["total_return"] == summary["headline"]["total_return"]
    assert record["max_drawdown_daily"] == summary["headline"]["max_drawdown_daily"]
    assert record["weeks"] == 4
    assert set(record) == {
        "spec_hash", "spec", "engine", "git_sha", "created_at", "ts", "weekly_sharpe",
        "weeks", "total_return", "max_drawdown_daily", "outcome", "failed_at",
    }  # fmt: skip


def _raising_runner(error: BaseException) -> Callable[[str], Callable[[CarryInputs], CarryRun]]:
    def runner(inputs: CarryInputs) -> CarryRun:
        raise error

    return lambda engine: runner


def test_ruined_run_is_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R11: a ruin at decision 2 exits 2 and records ONE ruined line, no out dir."""
    _patch_inputs(monkeypatch, _inputs(tmp_path))
    error = CarryEquityTooSmall("BTCUSDT", _DECISIONS[1], 3.0, 40.0)
    monkeypatch.setattr(carry_cli, "_engine_runner", _raising_runner(error))
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"

    assert carry_cli.main(_run_args(tmp_path, out=out_dir, trials=trials)) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("carry[nt]: CarryEquityTooSmall: ")
    assert not out_dir.exists()
    (record,) = _records(trials)
    assert record["outcome"] == "ruined"
    assert record["failed_at"] == _DECISIONS[1]
    assert record["engine"] == "nt"
    for key in ("weekly_sharpe", "weeks", "total_return", "max_drawdown_daily"):
        assert record[key] is None
    # a retry of the same ruin appends nothing (first row per (hash, outcome) is kept)
    assert carry_cli.main(_run_args(tmp_path, out=out_dir, trials=trials)) == 2
    capsys.readouterr()
    assert len(_records(trials)) == 1


class _StubLiquidated(CarryEquityTooSmall):
    """Stands in for T6's ``CarryLiquidated``."""


@pytest.mark.parametrize("liquidated", [False, True], ids=["ruin", "bust"])
@pytest.mark.parametrize("offset", [None, -1, 0, 1], ids=["null", "before_d1", "at_d1", "after_d1"])
def test_failure_classification_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    liquidated: bool,
    offset: int | None,
) -> None:
    """R11a: run and evaluate classify a failure identically; only t > D1 is a trial."""
    import vibe_quant.portfolio.research as research
    import vibe_quant.portfolio.trials as trials_module

    monkeypatch.setattr(trials_module, "_LIQUIDATED", (_StubLiquidated,))
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    at = None if offset is None else _DECISIONS[0] + offset
    kind = _StubLiquidated if liquidated else CarryEquityTooSmall
    error = kind("BTCUSDT", at, 3.0, 40.0)
    monkeypatch.setattr(carry_cli, "_engine_runner", _raising_runner(error))

    def simulate(inputs: CarryInputs, **kw: object) -> CarryRun:
        raise error

    monkeypatch.setattr(research, "simulate_reference", simulate)
    run_ledger = tmp_path / "run.jsonl"
    eval_ledger = tmp_path / "eval.jsonl"

    assert carry_cli.main(_run_args(tmp_path, out=tmp_path / "out", trials=run_ledger)) == 2
    assert carry_cli.main(_evaluate_argv(out=tmp_path / "res", trials=eval_ledger)) == 2
    capsys.readouterr()

    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "res" / "research.json").exists()
    if offset is None or offset <= 0:
        assert not run_ledger.exists()
        assert not eval_ledger.exists()
        return
    expected = [(inputs.spec.spec_hash(), "liquidated" if liquidated else "ruined", at)]
    for ledger in (run_ledger, eval_ledger):
        records = _records(ledger)
        assert [(r["spec_hash"], r["outcome"], r["failed_at"]) for r in records] == expected
        assert records[0]["weekly_sharpe"] is None


@pytest.mark.parametrize(
    "where",
    ["first_decision", "preflight_data", "execution"],
)
@pytest.mark.parametrize("existing", [False, True], ids=["absent", "existing"])
def test_input_failures_not_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    where: str,
    existing: bool,
) -> None:
    """R12: a stop at D1, a preflight data error or an execution error is not a trial."""
    inputs = _inputs(tmp_path)
    _patch_inputs(monkeypatch, inputs)
    if where == "preflight_data":
        monkeypatch.setattr(carry_cli, "load_carry_inputs", _raise(CarryDataError("no MARK")))
    else:
        error: BaseException = (
            CarryEquityTooSmall("<book>", _DECISIONS[0], 1.0, 2.0)
            if where == "first_decision"
            else CarryExecutionError("stale release BTCUSDT")
        )
        monkeypatch.setattr(carry_cli, "_engine_runner", _raising_runner(error))
    trials = tmp_path / "trials.jsonl"
    if existing:
        trials.write_text('{"spec_hash": "deadbeef", "weekly_sharpe": 0.5}\n')
    before = _read(trials)

    assert carry_cli.main(_run_args(tmp_path, out=tmp_path / "out", trials=trials)) == 2

    capsys.readouterr()
    assert _read(trials) == before
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("force", [False, True], ids=["plain", "force"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "dotdot"])
def test_ledger_output_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    alias: str,
    force: bool,
) -> None:
    """R14c: a ledger that IS research.json, or lives in the run's out dir, exits 2 first."""
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _no_load)
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "summary.json").write_text("{}")  # a previous carry output: --force may replace it
    research_file = out / "research.json"
    research_file.write_text("old research")
    (out / "x.jsonl").write_text('{"spec_hash": "deadbeef"}\n')
    (tmp_path / "alias").symlink_to(out, target_is_directory=True)
    base = {"direct": out, "symlink": tmp_path / "alias", "dotdot": out / "sub" / ".."}[alias]
    snapshot = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    extra = ("--force",) if force else ()

    for argv, ledger, target in (
        (_evaluate_argv(*extra, out=out, trials=base / "research.json"), "research.json", research_file),
        (_run_args(tmp_path, *extra, out=out, trials=base / "x.jsonl"), "x.jsonl", out),
        (_run_args(tmp_path, *extra, out=out, trials=base / "sub" / "deep.jsonl"), "sub/deep.jsonl", out),
        (_run_args(tmp_path, *extra, out=out, trials=base), "", out),
    ):
        assert carry_cli.main(argv) == 2
        err = capsys.readouterr().err
        assert len(err.splitlines()) == 1
        assert str(base / ledger if ledger else base) in err
        assert str(target) in err
        assert "trial ledger" in err

    assert {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()} == snapshot
    assert sorted(path.name for path in out.iterdir()) == [
        "research.json", "sub", "summary.json", "x.jsonl",
    ]  # fmt: skip


def test_ledger_alias_in_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R14c: --force must not delete a ledger alias kept in the out dir (leaf or traversed)."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("{}")  # a previous carry output: --force may replace it
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    ledger = tmp_path / "trials.jsonl"
    original = '{"spec_hash": "legacy", "weekly_sharpe": 0.1}\n'
    ledger.write_text(original)
    (elsewhere / "x.jsonl").write_text(original)
    (out / "x.jsonl").symlink_to(ledger)  # the leaf is an alias stored in out
    (out / "d").symlink_to(elsewhere, target_is_directory=True)
    (tmp_path / "via_out").symlink_to(out / "d", target_is_directory=True)
    (tmp_path / "direct").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _no_load)

    for alias in (out / "x.jsonl", tmp_path / "via_out" / "x.jsonl"):
        argv = _run_args(tmp_path, "--force", "--engine", "reference", out=out, trials=alias)
        assert carry_cli.main(argv) == 2
        err = capsys.readouterr().err
        assert f"ValueError: trial ledger {alias} lies in the output dir {out}" in err
    assert (out / "x.jsonl").is_symlink()
    assert (out / "d").is_symlink()
    assert ledger.read_text() == original
    assert (elsewhere / "x.jsonl").read_text() == original
    assert (out / "summary.json").read_text() == "{}"

    # control: an alias chain that never enters out is a normal ledger
    _patch_inputs(monkeypatch, _inputs(tmp_path))
    argv = _run_args(
        tmp_path, "--force", "--engine", "reference", out=out, trials=tmp_path / "direct" / "x.jsonl"
    )
    assert carry_cli.main(argv) == 0
    capsys.readouterr()
    assert [r.get("outcome") for r in _records(elsewhere / "x.jsonl")] == [None, "completed"]


@pytest.mark.parametrize(
    "sharpe", ["NaN", str(10**400), "1e308"], ids=["nan", "big_int", "variance_overflow"]
)
def test_ledger_bad_numbers_publish_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sharpe: str
) -> None:
    """R14d: a ledger the DSR cannot read exits 2 before any load; bytes kept, no report."""
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _no_load)
    ledger = tmp_path / "trials.jsonl"
    original = (
        '{"spec_hash": "x", "weekly_sharpe": ' + sharpe + "}\n"
        '{"spec_hash": "y", "weekly_sharpe": 0.1}\n'
    )
    ledger.write_text(original)

    assert carry_cli.main(_evaluate_argv(out=tmp_path / "res", trials=ledger)) == 2
    assert carry_cli.main(_run_args(tmp_path, out=tmp_path / "out", trials=ledger)) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2
    assert all(f"CarryLedgerError: trial ledger {ledger}" in line for line in err)
    assert ledger.read_text() == original
    assert not (tmp_path / "res").exists()
    assert not (tmp_path / "out").exists()


def test_ledger_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R14d: a bad ledger or a failed append publishes no output; a registered trial stays."""
    import vibe_quant.portfolio.trials as trials_module

    inputs = _inputs(tmp_path)
    trials = tmp_path / "trials.jsonl"

    # (a) malformed ledger: refused before any load, nothing written
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _no_load)
    trials.write_text('{"spec_hash": "deadbeef"}\n{"spec_hash": ')
    before = trials.read_bytes()
    assert carry_cli.main(_run_args(tmp_path, out=tmp_path / "out", trials=trials)) == 2
    assert carry_cli.main(_evaluate_argv(out=tmp_path / "res", trials=trials)) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2
    assert all(f"CarryLedgerError: trial ledger {trials} line 2: truncated" in line for line in err)
    assert trials.read_bytes() == before
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "res").exists()

    # (b) the append fails: exit 2, no output, the ledger keeps its old bytes
    _patch_inputs(monkeypatch, inputs)
    trials.write_text('{"spec_hash": "deadbeef", "weekly_sharpe": 0.5}\n')
    before = trials.read_bytes()

    def fail(fd: int, payload: bytes) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trials_module, "_append_bytes", fail)
    assert carry_cli.main(_run_args(tmp_path, "--engine", "reference", out=tmp_path / "out", trials=trials)) == 2
    assert carry_cli.main(_evaluate_argv(out=tmp_path / "res", trials=trials)) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2
    assert all(f"CarryLedgerError: trial ledger {trials} append failed: " in line for line in err)
    assert trials.read_bytes() == before
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "res" / "research.json").exists()

    # (c) publication fails AFTER registration: the trial stays counted
    monkeypatch.undo()
    _patch_inputs(monkeypatch, inputs)

    def no_publish(*args: object) -> None:
        raise ValueError("publication failed")

    monkeypatch.setattr(carry_cli, "_write_outputs", no_publish)
    assert carry_cli.main(_run_args(tmp_path, "--engine", "reference", out=tmp_path / "out", trials=trials)) == 2
    capsys.readouterr()
    assert trials.read_bytes().startswith(before)
    assert [(r["spec_hash"], r.get("outcome")) for r in _records(trials)] == [
        ("deadbeef", None),
        (inputs.spec.spec_hash(), "completed"),
    ]


@pytest.mark.parametrize("weighting", ["equal", "inverse_vol"])
def test_decisions_vols_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    weighting: str,
) -> None:
    """R26: decisions.parquet gains a JSON ``vols`` column only under inverse_vol."""
    base = _inputs(tmp_path)
    inputs = base
    if weighting == "inverse_vol":
        days = range(_START - 3 * _DAY, _END + _DAY, _DAY)
        swing = {"BTCUSDT": 0.01, "ETHUSDT": 0.02}
        inputs = replace(
            base,
            spec=replace(base.spec, weighting="inverse_vol", vol_lookback_days=3),
            vol_closes={
                symbol: {
                    day: _SIZING[symbol] * (1.0 + swing[symbol] * (-1) ** index)
                    for index, day in enumerate(days)
                }
                for symbol in _SYMBOLS
            },
        )
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"
    extra = ["--weighting", weighting]
    if weighting == "inverse_vol":
        extra += ["--vol-lookback-days", "3"]

    assert carry_cli.main(_run_args(tmp_path, *extra, out=out_dir, trials=trials)) == 0
    capsys.readouterr()

    decisions = pd.read_parquet(out_dir / "decisions.parquet")
    line = _records(trials)[0]
    assert line["spec_hash"] == inputs.spec.spec_hash()
    if weighting == "equal":
        assert "vols" not in decisions.columns
        assert "weighting" not in json.loads((out_dir / "summary.json").read_text())["spec"]
        return
    assert list(decisions.columns)[-1] == "vols"
    vols = [json.loads(cell) for cell in decisions["vols"]]
    assert [sorted(row) for row in vols] == [list(_SYMBOLS)] * 4
    # one leg a side: weights stay the equal split (R20) and vols are sigma > 0
    weights = [json.loads(cell) for cell in decisions["weights"]]
    assert {abs(w) for row in weights for w in row.values()} == {0.5}
    assert all(v > 0.0 for row in vols for v in row.values())
    summary = json.loads((out_dir / "summary.json").read_text())
    assert (summary["spec"]["weighting"], summary["spec"]["vol_lookback_days"]) == ("inverse_vol", 3)


# --- The cross liquidation overlay (x0axf.14 T6) -------------------------------------------

_SPIKE = _DECISIONS[0] + _DAY  # a bar open inside week 1 (short BTC, long ETH)


def _spiked(tmp_path: Path, bars: dict[int, dict[str, float]]) -> CarryInputs:
    """The fixture under ``liquidation_mode=cross`` with BTC MARK bars overridden."""
    base = _inputs(tmp_path)
    btc = base.marks["BTCUSDT"]
    columns = {name: getattr(btc, name).copy() for name in ("low", "high", "close", "open")}
    for at, values in bars.items():
        (index,) = np.flatnonzero(btc.ts == at)
        for name, value in values.items():
            columns[name][index] = value
    return replace(
        base,
        spec=replace(base.spec, liquidation_mode="cross"),
        marks={**base.marks, "BTCUSDT": MarkSeries(ts=btc.ts, **columns)},
    )


@pytest.mark.parametrize("engine", ["reference", "nt"])
def test_liquidated_run_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], engine: str
) -> None:
    """R42: a 3.1x BTC MARK spike (BTC is the short leg) busts the cross account in both executors: exit 2, one
    ``liquidated`` line at the bust ts, no out dir."""
    inputs = _spiked(tmp_path, {_SPIKE: {"high": 51010.0 * 3.1}})
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"
    argv = _run_args(tmp_path, "--engine", engine, "--liquidation-mode", "cross", out=out_dir, trials=trials)

    assert carry_cli.main(argv) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        f"carry[{engine}]: CarryLiquidated: account bust (cross) at {_SPIKE}: "
        "legs BTCUSDT,ETHUSDT margin balance "
    )
    assert len(captured.err.splitlines()) == 1
    assert not out_dir.exists()
    (record,) = _records(trials)
    assert (record["outcome"], record["failed_at"], record["engine"]) == ("liquidated", _SPIKE, engine)
    assert record["spec"]["liquidation_mode"] == "cross"
    assert record["weekly_sharpe"] is None
    # the same spike with the overlay off completes (the overlay is the only difference)
    _patch_inputs(monkeypatch, replace(inputs, spec=replace(inputs.spec, liquidation_mode="off")))
    assert carry_cli.main(_run_args(tmp_path, "--engine", engine, out=out_dir, trials=trials)) == 0
    capsys.readouterr()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["liquidation"] == "not_modelled"


def test_cross_run_without_bust_writes_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R39 via the CLI: summary.json carries the cross notes, the ledger one completed line."""
    inputs = _spiked(tmp_path, {})
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "out"
    trials = tmp_path / "trials.jsonl"
    argv = _run_args(tmp_path, "--engine", "reference", "--liquidation-mode", "cross", out=out_dir, trials=trials)

    assert carry_cli.main(argv) == 0
    capsys.readouterr()

    summary = json.loads((out_dir / "summary.json").read_text())
    notes = summary["liquidation"]
    assert (notes["mode"], notes["count"], notes["bust_ts"], notes["tier_table_version"]) == (
        "cross", 0, None, "2026-10-17",
    )  # fmt: skip
    assert summary["spec"]["liquidation_mode"] == "cross"
    assert [r["outcome"] for r in _records(trials)] == ["completed"]


def test_abort_before_completion_not_reclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R45: an executor abort before completion propagates as itself (recorded ``ruined``),
    even when the MARK path would have busted the account earlier."""
    _patch_inputs(monkeypatch, _spiked(tmp_path, {_SPIKE: {"high": 51010.0 * 3.1}}))
    error = CarryEquityTooSmall("BTCUSDT", _DECISIONS[2], 3.0, 40.0)
    monkeypatch.setattr(carry_cli, "_engine_runner", _raising_runner(error))
    trials = tmp_path / "trials.jsonl"

    argv = _run_args(tmp_path, "--liquidation-mode", "cross", out=tmp_path / "out", trials=trials)
    assert carry_cli.main(argv) == 2

    assert capsys.readouterr().err.startswith("carry[nt]: CarryEquityTooSmall: ")
    (record,) = _records(trials)
    assert (record["outcome"], record["failed_at"]) == ("ruined", _DECISIONS[2])


@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_liquidation_tier_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """R46: under cross a symbol without a tier table exits 2 before any load, no trial
    line; with the overlay off the same symbols reach the loader."""
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _no_load)
    trials = tmp_path / "trials.jsonl"
    out = tmp_path / "out"
    argv = (
        _run_args(tmp_path, out=out, trials=trials)
        if command == "run"
        else _evaluate_argv(out=out, trials=trials)
    )
    argv[argv.index("ETHUSDT")] = "ZZZUSDT"

    assert carry_cli.main([*argv, "--liquidation-mode", "cross"]) == 2

    err = capsys.readouterr().err
    assert err.endswith("CarryDataError: no liquidation tier table for ZZZUSDT in 2026-10-17\n")
    assert not trials.exists()
    assert not out.exists()
    monkeypatch.setattr(carry_cli, "load_carry_inputs", _raise(CarryDataError("loaded")))
    assert carry_cli.main(argv) == 2
    assert capsys.readouterr().err.endswith("CarryDataError: loaded\n")


def test_liquidation_data_error_not_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R46: an out-of-tier MARK after an earlier bust is a data error: exit 2, no trial."""
    bars = {_SPIKE: {"high": 51010.0 * 3.1}, _SPIKE + _HOUR: {"high": 1e12}}
    _patch_inputs(monkeypatch, _spiked(tmp_path, bars))
    trials = tmp_path / "trials.jsonl"
    argv = _run_args(tmp_path, "--engine", "reference", "--liquidation-mode", "cross", out=tmp_path / "out", trials=trials)

    assert carry_cli.main(argv) == 2

    assert "carry[reference]: CarryDataError: BTCUSDT: notional " in capsys.readouterr().err
    assert not trials.exists()
    assert not (tmp_path / "out").exists()
