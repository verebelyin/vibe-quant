"""The carry DSR trial ledger: outcomes, registration rules, locking, reducer.

Contract rows R11a, R14a, R14b, R14d (module half), R15 and R15a of design
x0axf.14. Every ledger here is a ``tmp_path`` file; the real
``data/portfolio/trials.jsonl`` is never touched.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from typing import TYPE_CHECKING, Any

import pytest

import vibe_quant.portfolio.trials as trials
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarryExecutionError,
    CarryLiquidated,
    CarrySpec,
)
from vibe_quant.portfolio.trials import (
    N_PRIOR_TRIALS,
    CarryLedgerError,
    completion_line,
    failure_line,
    failure_outcome,
    read_trials,
    record_completion,
    record_failure,
    trial_counts,
)

if TYPE_CHECKING:
    from pathlib import Path

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_SPEC_A = CarrySpec.from_dates(_SYMBOLS, "2025-01-06", "2025-03-03", band=0.10)
_SPEC_B = CarrySpec.from_dates(_SYMBOLS, "2025-01-06", "2025-03-03", band=0.20)
_A = _SPEC_A.spec_hash()
_B = _SPEC_B.spec_hash()
_D1 = 1_736_121_900_000_000_000  # Monday 2025-01-06 00:05 UTC
_D2 = _D1 + 7 * 86_400 * 10**9
_DECISIONS = (_D1, _D2, _D2 + 7 * 86_400 * 10**9)
# R15a: per-period variance of legacy A=0.1 and B=2.0 annualised weekly Sharpes
_AB_VARIANCE = 0.03471153846153847


def _summary(sharpe: float | None) -> dict[str, Any]:
    return {"weekly_sharpe": sharpe, "weeks": 8, "total_return": 0.01, "max_drawdown_daily": 0.02}


def _legacy(spec_hash: str, sharpe: float | None) -> str:
    """A pre-outcome ledger line, as ``__main__._trial_line`` wrote it on main."""
    return json.dumps(
        {
            "spec_hash": spec_hash,
            "spec": {},
            "engine": "nt",
            "git_sha": "abc",
            "created_at": "2026-10-12T00:00:00+00:00",
            "ts": 1,
            "weekly_sharpe": sharpe,
            "weeks": 8,
            "total_return": 0.0,
            "max_drawdown_daily": 0.0,
        },
        sort_keys=True,
    )


def _write(path: Path, *lines: str) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _counts(path: Path, own: str) -> tuple[int, float | None]:
    return trial_counts(read_trials(path), own)


def _outcomes(path: Path) -> list[tuple[str, str]]:
    return [(r["spec_hash"], r.get("outcome", "legacy")) for r in read_trials(path)]


def _completed(spec: CarrySpec, sharpe: float | None) -> str:
    return completion_line("reference", spec, _summary(sharpe), "sha")


def _failed(spec: CarrySpec, outcome: str, at: int = _D2) -> str:
    return failure_line("reference", spec, outcome, at, "sha")


# --- R15a: the DSR sample survives failures and retries ------------------------------


def test_dsr_sample_survives_retries(tmp_path: Path) -> None:
    """R15a: a ruined row (and its retry) never erases A's finite Sharpe; either order."""
    ledger = tmp_path / "trials.jsonl"
    _write(ledger, _legacy(_A, 0.1), _completed(_SPEC_B, 2.0))
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)

    assert record_failure(ledger, _failed(_SPEC_A, "ruined")) is True
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)
    before = ledger.read_bytes()
    assert record_failure(ledger, _failed(_SPEC_A, "ruined", _D2 + 1)) is False
    assert ledger.read_bytes() == before
    assert _outcomes(ledger).count((_A, "ruined")) == 1
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)

    reverse = tmp_path / "reverse.jsonl"
    _write(reverse, _completed(_SPEC_B, 2.0))
    assert record_failure(reverse, _failed(_SPEC_A, "ruined")) is True
    assert _counts(reverse, _A) == (5, None)  # the failure counts in N, not the variance
    assert record_completion(reverse, _completed(_SPEC_A, 0.1), history=False) is True
    assert _counts(reverse, _A) == (5, _AB_VARIANCE)


# --- R15 / R14a: reducer -------------------------------------------------------------


def test_null_sharpe_counts_not_variance(tmp_path: Path) -> None:
    """R15: completed 1.0, liquidated null, ruined null -> N = 3 + 3, variance None."""
    spec_c = CarrySpec.from_dates(_SYMBOLS, "2025-01-06", "2025-03-03", band=0.30)
    ledger = tmp_path / "trials.jsonl"
    _write(
        ledger,
        _completed(_SPEC_A, 1.0),
        _failed(_SPEC_B, "liquidated"),
        _failed(spec_c, "ruined"),
    )
    assert N_PRIOR_TRIALS == 3
    assert _counts(ledger, _A) == (6, None)
    # the own hash joins the set once: a fourth distinct hash makes 7
    assert _counts(ledger, "f" * 64)[0] == 7


@pytest.mark.parametrize("failure", ["ruined", "liquidated"])
def test_mixed_outcomes_reducer(tmp_path: Path, failure: str) -> None:
    """R14a: one hash = one trial; variance = its most recent finite completed Sharpe."""
    ledger = tmp_path / "trials.jsonl"
    finite_then_failed = [
        _legacy(_B, 2.0),
        _completed(_SPEC_A, 0.1),
        _failed(_SPEC_A, failure),
        _completed(_SPEC_A, None),
    ]
    _write(ledger, *finite_then_failed)
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)

    _write(ledger, *reversed(finite_then_failed))  # failure and null rows first
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)

    # the most recent finite completed row wins over an older finite one
    _write(ledger, _legacy(_B, 2.0), _legacy(_A, 5.0), _completed(_SPEC_A, 0.1))
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)

    # a failure-only hash adds to N only
    _write(ledger, _legacy(_B, 2.0), _legacy(_A, 0.1), _failed(_SPEC_A, failure))
    other = CarrySpec.from_dates(_SYMBOLS, "2025-01-06", "2025-03-03", band=0.40)
    record_failure(ledger, _failed(other, failure))
    assert _counts(ledger, _A) == (6, _AB_VARIANCE)
    # a repeat of the same (hash, outcome) failure is not appended
    before = ledger.read_bytes()
    assert record_failure(ledger, _failed(other, failure, _D2 + 99)) is False
    assert ledger.read_bytes() == before


def test_all_finite_ledger_keeps_last_observation(tmp_path: Path) -> None:
    """R14a: an all-finite ledger reduces exactly as main did (last row per hash)."""
    ledger = tmp_path / "trials.jsonl"
    _write(ledger, _legacy(_A, 3.0), _legacy(_B, 2.0), _legacy(_A, 0.1))
    assert _counts(ledger, _B) == (5, _AB_VARIANCE)
    assert _counts(tmp_path / "absent.jsonl", _A) == (4, None)


# --- R14b: registration rules ----------------------------------------------------------


def test_record_completion_rules(tmp_path: Path) -> None:
    """R14b: evaluate appends only to add a hash's first (finite) completion; run always."""
    ledger = tmp_path / "trials.jsonl"
    _write(ledger, _failed(_SPEC_A, "ruined"))
    assert record_completion(ledger, _completed(_SPEC_A, 0.1), history=False) is True
    assert _outcomes(ledger) == [(_A, "ruined"), (_A, "completed")]

    # finite after an earlier completed-null row -> appended
    _write(ledger, _completed(_SPEC_A, None))
    assert record_completion(ledger, _completed(_SPEC_A, 0.1), history=False) is True
    assert len(read_trials(ledger)) == 2

    # finite after a finite completion -> ledger bytes unchanged
    before = ledger.read_bytes()
    assert record_completion(ledger, _completed(_SPEC_A, 0.7), history=False) is False
    assert ledger.read_bytes() == before
    # null after a null-only completion -> unchanged too
    _write(ledger, _completed(_SPEC_A, None))
    before = ledger.read_bytes()
    assert record_completion(ledger, _completed(_SPEC_A, None), history=False) is False
    assert ledger.read_bytes() == before
    # a legacy row is a completion
    _write(ledger, _legacy(_A, 0.1))
    before = ledger.read_bytes()
    assert record_completion(ledger, _completed(_SPEC_A, 0.2), history=False) is False
    assert ledger.read_bytes() == before

    # run: one history row per completed run, every time
    assert record_completion(ledger, _completed(_SPEC_A, 0.1), history=True) is True
    assert record_completion(ledger, _completed(_SPEC_A, 0.1), history=True) is True
    assert len(read_trials(ledger)) == 3

    # a fresh ledger (and its parent dir) is created on first append
    fresh = tmp_path / "new" / "trials.jsonl"
    assert record_completion(fresh, _completed(_SPEC_B, 2.0), history=False) is True
    assert _outcomes(fresh) == [(_B, "completed")]


def test_line_fields(tmp_path: Path) -> None:
    """R10/R11 line shape: outcome + failed_at; failure lines null every metric."""
    done = json.loads(_completed(_SPEC_A, 1.5))
    assert done["outcome"] == "completed"
    assert done["failed_at"] is None
    assert done["weekly_sharpe"] == 1.5
    assert (done["weeks"], done["total_return"], done["max_drawdown_daily"]) == (8, 0.01, 0.02)
    assert done["spec_hash"] == _A
    assert done["spec"] == _SPEC_A.to_json()
    assert set(done) == {
        "spec_hash", "spec", "engine", "git_sha", "created_at", "ts",
        "weekly_sharpe", "weeks", "total_return", "max_drawdown_daily",
        "outcome", "failed_at",
    }  # fmt: skip
    failed = json.loads(_failed(_SPEC_B, "liquidated", _D2 + 5))
    assert failed["outcome"] == "liquidated"
    assert failed["failed_at"] == _D2 + 5
    for key in ("weekly_sharpe", "weeks", "total_return", "max_drawdown_daily"):
        assert failed[key] is None
    assert json.loads(_completed(_SPEC_A, math.inf))["weekly_sharpe"] is None
    with pytest.raises(ValueError, match="outcome"):
        failure_line("reference", _SPEC_A, "completed", _D2, "sha")


# --- R11a: failure classification ------------------------------------------------------


class _StubLiquidated(CarryEquityTooSmall):
    """Stands in for T6's ``CarryLiquidated`` (a ``CarryEquityTooSmall`` subclass)."""


def test_failure_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """R11a: t null or <= D1 -> not a trial; t > D1 -> ruined / liquidated (subclass first)."""
    monkeypatch.setattr(trials, "_LIQUIDATED", (_StubLiquidated,))

    def ruin(at: int | None) -> CarryEquityTooSmall:
        return CarryEquityTooSmall("BTCUSDT", at, 1.0, 2.0)

    def bust(at: int | None) -> CarryEquityTooSmall:
        return _StubLiquidated("BTCUSDT", at, 1.0, 2.0)

    for make in (ruin, bust):
        assert failure_outcome(make(None), _DECISIONS) is None
        assert failure_outcome(make(_D1 - 1), _DECISIONS) is None
        assert failure_outcome(make(_D1), _DECISIONS) is None
    assert failure_outcome(ruin(_D1 + 1), _DECISIONS) == "ruined"
    assert failure_outcome(ruin(_D2), _DECISIONS) == "ruined"
    assert failure_outcome(bust(_D1 + 1), _DECISIONS) == "liquidated"
    assert failure_outcome(bust(_D2), _DECISIONS) == "liquidated"
    for error in (
        CarryDataError("no MARK"),
        CarryExecutionError("stale"),
        CarryLedgerError("bad ledger"),
        ValueError("other"),
    ):
        assert failure_outcome(error, _DECISIONS) is None
    assert failure_outcome(ruin(_D2), ()) is None


def test_failure_outcome_real_liquidation_class() -> None:
    """R11a with T6's real ``CarryLiquidated``: after D1 -> liquidated, at D1 -> no trial;
    any other ``CarryEquityTooSmall`` subclass stays ruined."""
    assert failure_outcome(CarryLiquidated(("ETHUSDT",), _D2, -502.0, 62.0), _DECISIONS) == "liquidated"
    assert failure_outcome(CarryLiquidated(("ETHUSDT",), _D1 + 1, 0.0, 1.0), _DECISIONS) == "liquidated"
    assert failure_outcome(CarryLiquidated(("ETHUSDT",), _D1, -502.0, 62.0), _DECISIONS) is None
    assert failure_outcome(_StubLiquidated("X", _D2, 1.0, 2.0), _DECISIONS) == "ruined"


# --- R14d: ledger errors are atomic ---------------------------------------------------


def test_ledger_errors_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R14d: unreadable / malformed / truncated ledgers and failed appends raise, mutate nothing."""
    unreadable = tmp_path / "dir.jsonl"
    unreadable.mkdir()
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {unreadable} unreadable: "):
        read_trials(unreadable)
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {unreadable} unreadable: "):
        record_completion(unreadable, _completed(_SPEC_A, 1.0), history=True)
    assert issubclass(CarryLedgerError, ValueError)

    ledger = tmp_path / "trials.jsonl"
    cases = {
        "not json": _legacy(_A, 0.1) + "\n{not json\n",
        "not an object": _legacy(_A, 0.1) + "\n[1, 2]\n",
        "no spec_hash": _legacy(_A, 0.1) + '\n{"weekly_sharpe": 1.0}\n',
        "bad outcome": _legacy(_A, 0.1) + '\n{"spec_hash": "x", "outcome": "won"}\n',
    }
    for content in cases.values():
        ledger.write_text(content, encoding="utf-8")
        before = ledger.read_bytes()
        with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 2: malformed"):
            read_trials(ledger)
        with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 2: malformed"):
            record_failure(ledger, _failed(_SPEC_B, "ruined"))
        assert ledger.read_bytes() == before

    ledger.write_text(_legacy(_A, 0.1) + "\n" + _legacy(_B, 2.0)[:-5], encoding="utf-8")
    before = ledger.read_bytes()
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 2: truncated"):
        read_trials(ledger)
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 2: truncated"):
        record_completion(ledger, _completed(_SPEC_B, 2.0), history=True)
    assert ledger.read_bytes() == before

    ledger.write_bytes(b"\xff\xfe\n")
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 1: malformed"):
        read_trials(ledger)

    # a write that lands half the record and then fails is cut back to the old size
    _write(ledger, _legacy(_A, 0.1))
    before = ledger.read_bytes()

    def half_then_fail(fd: int, payload: bytes) -> None:
        os.write(fd, payload[: len(payload) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trials, "_append_bytes", half_then_fail)
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} append failed: "):
        record_completion(ledger, _completed(_SPEC_B, 2.0), history=True)
    assert ledger.read_bytes() == before
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} append failed: "):
        record_failure(ledger, _failed(_SPEC_B, "ruined"))
    assert ledger.read_bytes() == before
    assert _counts(ledger, _A) == (4, None)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Whitespace-only lines are not records (main's reader skipped them too)."""
    ledger = tmp_path / "trials.jsonl"
    ledger.write_text(_legacy(_A, 0.1) + "\n\n  \n" + _legacy(_B, 2.0) + "\n", encoding="utf-8")
    assert _counts(ledger, _A) == (5, _AB_VARIANCE)


_MAX_INT = 2**1024 - 2**971  # == int(sys.float_info.max), the largest float as an int


def _raw(spec_hash: str, sharpe: str) -> str:
    """A ledger line with ``weekly_sharpe`` spelled verbatim (any JSON-ish token)."""
    return '{"spec_hash": "' + spec_hash + '", "weekly_sharpe": ' + sharpe + "}"


def test_ledger_numbers_are_finite_floats(tmp_path: Path) -> None:
    """R14d: non-JSON constants and unrepresentable Sharpes are malformed, before any write."""
    ledger = tmp_path / "trials.jsonl"
    for bad in (
        _raw("x", "NaN"),
        _raw("x", "-Infinity"),
        '{"spec_hash": "x", "weekly_sharpe": 0.1, "total_return": Infinity}',
        _raw("x", str(_MAX_INT + 1)),  # an int beyond float range
        _raw("x", '"0.5"'),
        _raw("x", "true"),
    ):
        _write(ledger, _legacy(_A, 0.1), bad)
        before = ledger.read_bytes()
        with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 2: malformed"):
            read_trials(ledger)
        with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger} line 2: malformed"):
            record_completion(ledger, _completed(_SPEC_B, 2.0), history=True)
        assert ledger.read_bytes() == before

    # the edges that stay valid: the largest float, an int inside float range, null, absent
    _write(
        ledger,
        _raw("x", "1.7976931348623157e308"),
        _raw("y", str(-_MAX_INT)),
        _raw("z", "null"),
        '{"spec_hash": "w"}',
    )
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger}: weekly_sharpe variance"):
        read_trials(ledger)  # each line is valid; their variance is not a float
    _write(ledger, _raw("z", "null"), '{"spec_hash": "w"}', _raw("x", "1.7976931348623157e308"))
    assert _counts(ledger, _A) == (7, None)

    # variance edge: Sharpes v and 0 give (v/sqrt(52))**2 / 2 -> overflows between these
    _write(ledger, _raw("x", "1e155"), _raw("y", "0"))
    assert _counts(ledger, _A)[1] == pytest.approx((1e155 / math.sqrt(104)) ** 2, rel=1e-12)
    _write(ledger, _raw("x", "1.4e155"), _raw("y", "0"))
    before = ledger.read_bytes()
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger}: weekly_sharpe variance"):
        read_trials(ledger)
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger}: weekly_sharpe variance"):
        record_failure(ledger, _failed(_SPEC_B, "ruined"))
    assert ledger.read_bytes() == before

    # an append that would make the variance unrepresentable is refused, ledger unchanged
    _write(ledger, _raw("x", "1e308"))
    before = ledger.read_bytes()
    with pytest.raises(CarryLedgerError, match=f"^trial ledger {ledger}: weekly_sharpe variance"):
        record_completion(ledger, _completed(_SPEC_B, 0.1), history=True)
    assert ledger.read_bytes() == before
    assert record_failure(ledger, _failed(_SPEC_B, "ruined")) is True  # null Sharpe: no change
    assert _counts(ledger, _A) == (6, None)


# --- R14b: concurrent registration ----------------------------------------------------


_ROUNDS = 40


def _register_many(path: str, band: float) -> None:
    """A child process: ``_ROUNDS`` run completions with a padded spec (one hash each)."""
    from pathlib import Path

    spec = CarrySpec.from_dates(
        tuple(f"S{i:03d}USDT" for i in range(60)), "2025-01-06", "2025-03-03", band=band
    )
    for index in range(_ROUNDS):
        record_completion(
            Path(path),
            completion_line("reference", spec, _summary(float(index)), "sha"),
            history=True,
        )


def test_concurrent_register(tmp_path: Path) -> None:
    """R14b: two writers never interleave; every concurrent read is one coherent snapshot."""
    ledger = tmp_path / "trials.jsonl"
    ctx = mp.get_context("spawn")
    writers = [ctx.Process(target=_register_many, args=(str(ledger), band)) for band in (0.1, 0.2)]
    for writer in writers:
        writer.start()
    seen: list[int] = []
    while any(writer.is_alive() for writer in writers):
        seen.append(len(read_trials(ledger)))  # raises on a torn line
    for writer in writers:
        writer.join(60)
        assert writer.exitcode == 0
    records = read_trials(ledger)
    assert len(records) == 2 * _ROUNDS
    assert seen == sorted(seen)
    by_hash: dict[str, list[float]] = {}
    for record in records:
        by_hash.setdefault(record["spec_hash"], []).append(record["weekly_sharpe"])
    assert sorted(len(values) for values in by_hash.values()) == [_ROUNDS, _ROUNDS]
    for values in by_hash.values():
        assert values == [float(index) for index in range(_ROUNDS)]
    assert all(line for line in ledger.read_text(encoding="utf-8").split("\n")[:-1])


_WAIT = 60.0  # bound on every cross-process wait (spawned children import the package)


def _held_register(path: str, band: float, history: bool, at_append: Any, release: Any) -> None:
    """A child: register one completion, parked right before its append until ``release``."""
    from pathlib import Path

    original = trials._append_bytes

    def parked(fd: int, payload: bytes) -> None:
        at_append.set()
        if not release.wait(_WAIT):
            raise TimeoutError("never released")
        original(fd, payload)

    trials._append_bytes = parked
    spec = CarrySpec.from_dates(_SYMBOLS, "2025-01-06", "2025-03-03", band=band)
    line = completion_line("reference", spec, _summary(0.5), "sha")
    record_completion(Path(path), line, history=history)


def _plain_register(path: str, band: float, ready: Any) -> None:
    """A child: signal ``ready`` once imported, then register like ``evaluate`` does."""
    from pathlib import Path

    spec = CarrySpec.from_dates(_SYMBOLS, "2025-01-06", "2025-03-03", band=band)
    line = completion_line("reference", spec, _summary(0.5), "sha")
    ready.set()
    record_completion(Path(path), line, history=False)


def _half_then_fail(path: str, at_append: Any, release: Any) -> None:
    """A child: write half a record under the lock, park, then fail (rollback truncates)."""
    from pathlib import Path

    def torn(fd: int, payload: bytes) -> None:
        os.write(fd, payload[: len(payload) // 2])
        at_append.set()
        release.wait(_WAIT)
        raise OSError(28, "No space left on device")

    trials._append_bytes = torn
    with pytest.raises(CarryLedgerError, match="append failed"):
        record_failure(Path(path), _failed(_SPEC_B, "ruined"))


def test_concurrent_same_hash_appends_once(tmp_path: Path) -> None:
    """R14b: while one evaluate-style registration is parked before its append, a second
    for the same hash must wait for the lock and then see it: exactly one record."""
    ledger = tmp_path / "trials.jsonl"
    ctx = mp.get_context("spawn")
    at_append, release, ready = ctx.Event(), ctx.Event(), ctx.Event()
    first = ctx.Process(target=_held_register, args=(str(ledger), 0.1, False, at_append, release))
    second = ctx.Process(target=_plain_register, args=(str(ledger), 0.1, ready))
    try:
        first.start()
        assert at_append.wait(_WAIT)
        second.start()
        assert ready.wait(_WAIT)
        second.join(1.0)  # without the exclusive lock it reads, decides and appends here
        release.set()
        for child in (first, second):
            child.join(_WAIT)
            assert child.exitcode == 0
    finally:
        release.set()
        for child in (first, second):
            if child.is_alive():
                child.kill()
    assert _outcomes(ledger) == [(_A, "completed")]


def test_reader_never_sees_a_rolled_back_write(tmp_path: Path) -> None:
    """R14b: a reader during a half-written, later rolled-back append waits for the
    writer's lock and sees the coherent before-snapshot (never the torn line)."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    ledger = tmp_path / "trials.jsonl"
    _write(ledger, _legacy(_A, 0.1))
    before = read_trials(ledger)
    ctx = mp.get_context("spawn")
    at_append, release = ctx.Event(), ctx.Event()
    writer = ctx.Process(target=_half_then_fail, args=(str(ledger), at_append, release))
    with ThreadPoolExecutor(1) as pool:
        try:
            writer.start()
            assert at_append.wait(_WAIT)
            reading = pool.submit(read_trials, ledger)
            with pytest.raises(FutureTimeout):
                reading.result(timeout=1.0)  # blocked on the shared lock
            release.set()
            assert reading.result(timeout=_WAIT) == before
            writer.join(_WAIT)
            assert writer.exitcode == 0
        finally:
            release.set()
            if writer.is_alive():
                writer.kill()
    assert read_trials(ledger) == before


def test_git_sha_fallback_logs_without_the_error_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No git binary: ``"unknown"``, one warning naming the class, never the error text."""

    def no_git(*args: object, **kw: object) -> object:
        raise OSError("marker-7f3a no such file: git")

    monkeypatch.setattr(trials.subprocess, "run", no_git)
    with caplog.at_level("WARNING", logger="vibe_quant.portfolio.trials"):
        assert trials.git_sha() == "unknown"
    assert "git rev-parse failed (OSError)" in caplog.text
    assert "marker-7f3a" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
