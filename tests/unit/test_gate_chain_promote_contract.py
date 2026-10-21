"""Characterization contract for the promote-time gates (lane 4 T4, rows R50-R63).

Pins today's behaviour of the 1m short opposing-regime promote gate
(``vibe_quant/api/routers/discovery.py``) and of the promote-time overfitting DSR
re-check (``vibe_quant/overfitting/pipeline.py``) before rabmo.6 extracts
``vibe_quant/discovery/gate_chain.py``. Every row below calls ONLY the ADAPTER
block at the top; it holds every private access and every patch path, so T11
(regime gate) and T12 (re-check verdict) flip the adapter without touching a row.

The gate fails closed: the 409 detail and the reject reason texts are pinned
verbatim. The re-check rows drive ``OverfittingPipeline`` on a temp DB, never the
real ``data/state/vibe_quant.db``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import HTTPException

from vibe_quant.api.routers import discovery as _router
from vibe_quant.db.state_manager import StateManager
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig

#: The one 409 detail the regime gate raises (pinned verbatim, R52).
DETAIL_409 = (
    "1m short champions must pass at least one opposing-regime cross-window "
    "validation before promotion. Re-run discovery with shifted windows "
    "covering an opposite BTC market regime."
)

#: A neutral "not given" marker for optional payload keys / adapter arguments.
_MISSING = object()

#: Bear train range, bull shifted window: the shape that lets a short through.
_BEAR_BASE = {
    ("2024-01-01", "2025-01-01"): -1,
    ("2024-02-01", "2025-02-01"): 1,
}


def _window(
    sharpe: object = 0.5,
    return_pct: object = 0.1,
    start: str = "2024-02-01",
    end: str = "2025-02-01",
) -> dict[str, object]:
    return {"sharpe": sharpe, "return_pct": return_pct, "dates": [start, end]}


def _short(*windows: dict[str, object], direction: str = "short") -> dict[str, object]:
    return {
        "chromosome": {"direction": direction},
        "cross_window": {"windows": list(windows)},
    }


def _payload(min_sharpe: object = _MISSING) -> dict[str, object]:
    payload: dict[str, object] = {
        "cross_window_months": [1],
        "train_dates": ["2024-01-01", "2025-01-01"],
    }
    if min_sharpe is not _MISSING:
        payload["cross_window_min_sharpe"] = min_sharpe
    return payload


# ---------------------------------------------------------------------------
# ADAPTER: the only private access and the only patch paths in this file.
# T11 flips the regime half (gate_chain.passes_opposing_regime_gate +
# window_regime_sign), T12 the re-check half (recheck_dsr_verdict). Rows below
# must stay unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Raises:
    """A payload loader that must not be called (or must fail) for this row."""

    error: BaseException


@dataclass(frozen=True)
class _GateOut:
    """Outcome of one promote-gate call: "pass" or "409" + its pinned detail."""

    status: str
    detail: str | None
    payload_loads: int
    regime_calls: list[tuple[str, str, str]]


def _gate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    timeframe: str = "1m",
    entry: dict[str, object] | None = None,
    payload: dict[str, object] | _Raises | None = None,
    symbols: object = None,
    start_date: str = "2024-01-01",
    end_date: str = "2025-01-01",
    signs: dict[tuple[str, str], int] | None = None,
    default_sign: int = 0,
    real_regime: bool = False,
) -> _GateOut:
    payload_loads = 0

    def fake_load(_state: object, _run_id: int) -> dict[str, object]:
        nonlocal payload_loads
        payload_loads += 1
        if isinstance(payload, _Raises):
            raise payload.error
        return payload or {}

    monkeypatch.setattr(_router, "_load_discovery_payload", fake_load)

    regime_calls: list[tuple[str, str, str]] = []
    if real_regime:

        def down(*_args: object, **_kwargs: object) -> list[object]:
            raise RuntimeError("catalog down")

        # The real classifier keeps its own exception handling; only its data
        # loader is replaced. ``_window_regime_sign`` is not patched here.
        monkeypatch.setattr("vibe_quant.validation.random_baseline.load_ohlc", down)
    else:

        def fake_sign(symbol: str, start: str, end: str) -> int:
            regime_calls.append((symbol, start, end))
            return (signs or {}).get((start, end), default_sign)

        monkeypatch.setattr(_router, "_window_regime_sign", fake_sign)

    discovery_run: dict[str, object] = {
        "timeframe": timeframe,
        "symbols": ["BTCUSDT"] if symbols is None else symbols,
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        _router._enforce_short_1m_cross_regime_gate(
            object(),  # type: ignore[arg-type]  # the patched loader never reads it
            1,
            discovery_run,
            entry or {},
        )
    except HTTPException as exc:
        return _GateOut("409", str(exc.detail), payload_loads, regime_calls)
    return _GateOut("pass", None, payload_loads, regime_calls)


def _trades_short(entry: dict[str, object]) -> bool:
    return _router._discovery_entry_trades_short(entry)


#: Daily MTM Sharpe whose deflated statistic is 1.0 at N=88, T=875 (probe:
#: 0.9999999999999996, printed 1.0000 by the q95 reject text).
STAT_1_SHARPE_DAILY = 0.11800941586219002
_STAT_1_REASON = (
    "DSR q95 not significant: stat=1.0000 < spread=1.2000 (ann SR=1.873, T=875, N=88)"
)
_LEGACY_STAT_1_REASON = "DSR not significant: p=0.1587 (T=875, N=88)"


@dataclass(frozen=True)
class _RecheckOut:
    passed_dsr: bool | None
    fail_reason: str | None
    p_value: float | None
    deflated_sharpe: float | None


def _mtm_record(**over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "input": "pooled_mtm_daily_returns",
        "observations": 875,
        "sharpe_daily": STAT_1_SHARPE_DAILY,
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    record.update(over)
    return record


_ABSENT_MTM_REASON = (
    "DSR input missing: no mark-to-market statistics "
    "(mtm_sharpe_daily, mtm_skewness, mtm_kurtosis, mtm_days)"
)


def _recheck(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    record: Any = _MISSING,
    dsr_trials: dict[str, Any] | None = None,
    symbols: list[str] | None = None,
    symbol_agg: str = "portfolio",
    evaluated: int = 88,
    norm_sf: float | None = None,
) -> _RecheckOut:
    if record is _MISSING:
        record = _mtm_record()

    db = tmp_path / "disc.db"
    state = StateManager(db)
    syms = symbols or ["BTCUSDT"]
    run_id = state.create_backtest_run(
        None, "discovery", syms, "4h", "2022-01-01", "2026-03-17", {}
    )
    champion: dict[str, Any] = {"sharpe": 2.0}
    if record is not None:
        champion["dsr"] = record
    notes: dict[str, Any] = {
        "type": "discovery",
        "evaluated": evaluated,
        "symbols": syms,
        "symbol_agg": symbol_agg,
        "train_dates": ["2022-01-01", "2026-03-17"],
        "top_strategies": [champion],
    }
    if dsr_trials is not None:
        notes["dsr_trials"] = dsr_trials
    state.save_backtest_result(
        run_id,
        {
            "sharpe_ratio": 2.0,
            "total_return": 0.4,
            "max_drawdown": 0.1,
            "profit_factor": 1.5,
            "total_trades": 120,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "notes": json.dumps(notes),
        },
    )
    state.close()

    if norm_sf is not None:
        monkeypatch.setattr("vibe_quant.overfitting.dsr._norm_sf", lambda _x: norm_sf)

    pipeline = OverfittingPipeline(db)
    try:
        result = pipeline.run(run_id=run_id, config=FilterConfig.dsr_only())
    finally:
        pipeline.close()
    candidate = result.candidates[0]
    dsr = candidate.dsr_result
    return _RecheckOut(
        candidate.passed_dsr,
        candidate.dsr_fail_reason,
        dsr.p_value if dsr is not None else None,
        dsr.deflated_sharpe if dsr is not None else None,
    )


# ---------------------------------------------------------------------------
# R50-R51: the gate is lazy -- a non-1m run and a long-only champion never even
# load the discovery payload.
# ---------------------------------------------------------------------------


def test_non_1m_never_loads_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """R50: timeframe 4h returns None and never reads the stored notes."""
    out = _gate(
        monkeypatch,
        timeframe="4h",
        payload=_Raises(AssertionError("payload must not load")),
        entry=_short(_window()),
    )
    assert out.status == "pass"
    assert out.payload_loads == 0


def test_long_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """R51: a 1m direction=long champion is exempt and loads no payload."""
    out = _gate(
        monkeypatch,
        payload=_Raises(AssertionError("payload must not load")),
        entry=_short(_window(), direction="long"),
    )
    assert out.status == "pass"
    assert out.payload_loads == 0


# ---------------------------------------------------------------------------
# R52-R58: the opposing-regime window check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"payload": {}},
        {"payload": {"cross_window_months": []}},
        {
            "payload": {"cross_window_months": [1]},
            "entry": {"chromosome": {"direction": "short"}, "cross_window": ["x"]},
        },
        {
            "payload": {"cross_window_months": [1]},
            "entry": {"chromosome": {"direction": "short"}, "cross_window": {"windows": []}},
        },
        {"payload": _payload(), "entry": _short(_window()), "symbols": []},
        {
            "payload": {"cross_window_months": [1]},
            "entry": _short(_window()),
            "start_date": "",
            "end_date": "",
        },
    ],
    ids=[
        "no-offsets",
        "empty-offsets",
        "cross-window-not-dict",
        "empty-windows",
        "no-symbol",
        "no-dates",
    ],
)
def test_fail_closed_shapes(
    monkeypatch: pytest.MonkeyPatch, over: dict[str, object]
) -> None:
    """R52: every unreadable shape is a 409 with the pinned detail."""
    args: dict[str, Any] = {"entry": _short(_window()), "payload": _payload()}
    args.update(over)
    out = _gate(monkeypatch, **args)
    assert out.status == "409"
    assert out.detail == DETAIL_409
    assert out.payload_loads == 1
    assert out.regime_calls == []


@pytest.mark.parametrize(("return_pct", "status"), [(0.1, "pass"), (0.0, "409")])
def test_regime_sharpe_tie_and_zero_return(
    monkeypatch: pytest.MonkeyPatch, return_pct: float, status: str
) -> None:
    """R53: sharpe == min_sharpe passes; a non-positive window return does not."""
    out = _gate(
        monkeypatch,
        payload=_payload(),
        entry=_short(_window(0.5, return_pct)),
        signs=_BEAR_BASE,
    )
    assert out.status == status
    assert out.detail == (None if status == "pass" else DETAIL_409)


@pytest.mark.parametrize(("shifted_sign", "status"), [(-1, "409"), (1, "pass")])
def test_neutral_base_requires_bull(
    monkeypatch: pytest.MonkeyPatch, shifted_sign: int, status: str
) -> None:
    """R54: a neutral base demands a BULL window; a bear window never passes."""
    out = _gate(
        monkeypatch,
        payload=_payload(),
        entry=_short(_window(9.0, 1.0)),
        signs={("2024-01-01", "2025-01-01"): 0, ("2024-02-01", "2025-02-01"): shifted_sign},
    )
    assert out.status == status
    assert out.detail == (None if status == "pass" else DETAIL_409)


def test_loader_error_is_neutral(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R55: a failing OHLC load is a neutral sign (0), logged at WARNING."""
    with caplog.at_level("WARNING"):
        out = _gate(monkeypatch, payload=_payload(), entry=_short(_window()), real_regime=True)
    assert out.status == "409"
    assert out.detail == DETAIL_409
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert [r.getMessage() for r in warnings] == [
        "Regime sign: failed to load OHLC for BTCUSDT 2024-01-01→2025-01-01, treating as neutral",
        "Regime sign: failed to load OHLC for BTCUSDT 2024-02-01→2025-02-01, treating as neutral",
    ]


@pytest.mark.parametrize(
    "min_sharpe", [_MISSING, True, "x"], ids=["absent", "bool", "string"]
)
def test_min_sharpe_fallback(
    monkeypatch: pytest.MonkeyPatch, min_sharpe: object
) -> None:
    """R56: a missing / bool / non-numeric minimum falls back to 0.5."""
    out = _gate(
        monkeypatch,
        payload=_payload(min_sharpe),
        entry=_short(_window(0.5, 0.1)),
        signs=_BEAR_BASE,
    )
    assert out.status == "pass"
    assert out.detail is None


def test_legacy_window_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    """R57: legacy windows pair by offset: w1 <- start+1mo, w2 <- start+3mo."""
    legacy_in_sample = {"sharpe": 9.0, "return_pct": 9.0}
    legacy_w1 = {"sharpe": 0.0, "return_pct": 0.0}
    legacy_w2 = {"sharpe": 0.5, "return_pct": 0.1}
    out = _gate(
        monkeypatch,
        payload={"cross_window_months": [1, 3], "train_dates": ["2024-01-01", "2025-01-01"]},
        entry=_short(legacy_in_sample, legacy_w1, legacy_w2),
        signs={
            ("2024-01-01", "2025-01-01"): -1,
            ("2024-02-01", "2025-02-01"): 0,
            ("2024-04-01", "2025-04-01"): 1,
        },
    )
    assert out.status == "pass"
    assert out.regime_calls == [
        ("BTCUSDT", "2024-01-01", "2025-01-01"),
        ("BTCUSDT", "2024-02-01", "2025-02-01"),
        ("BTCUSDT", "2024-04-01", "2025-04-01"),
    ]


@pytest.mark.parametrize(
    ("symbols", "expected"), [('["BTCUSDT"]', "pass"), ("BTCUSDT", "pass"), ([], "409")]
)
def test_symbol_shapes(
    monkeypatch: pytest.MonkeyPatch, symbols: object, expected: str
) -> None:
    """R58: a JSON list and a bare string both give BTCUSDT; [] fails closed."""
    out = _gate(
        monkeypatch,
        payload=_payload(),
        entry=_short(_window()),
        symbols=symbols,
        signs=_BEAR_BASE,
    )
    assert out.status == expected
    if expected == "pass":
        assert {call[0] for call in out.regime_calls} == {"BTCUSDT"}
    else:
        assert out.detail == DETAIL_409
        assert out.regime_calls == []


# ---------------------------------------------------------------------------
# R59: which entries can place SHORT orders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"chromosome": {"direction": "short"}}, True),
        ({"chromosome": {"direction": "long"}}, False),
        (
            {
                "chromosome": {"direction": "both"},
                "dsl": {"entry_conditions": {"short": ["s"], "long": ["l"]}},
            },
            True,
        ),
        (
            {
                "chromosome": {"direction": "both"},
                "dsl": {"entry_conditions": {"short": ["same"], "long": ["same"]}},
            },
            False,
        ),
        ({"chromosome": {"direction": "both"}}, True),
        ({}, False),
    ],
    ids=["short", "long", "both-distinct", "both-identical", "both-no-dsl", "no-direction"],
)
def test_trades_short_table(entry: dict[str, object], expected: bool) -> None:
    """R59: the legacy same-bar tie resolves long; an unreadable both fails closed."""
    assert _trades_short(entry) is expected


# ---------------------------------------------------------------------------
# R60-R63: the promote-time overfitting DSR re-check
# ---------------------------------------------------------------------------


def test_recheck_legacy_tie_passes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R60: the legacy inclusive rule passes p == 0.05 (discovery's strict rule rejects it)."""
    out = _recheck(tmp_path, monkeypatch, norm_sf=0.05)
    assert out.passed_dsr is True
    assert out.fail_reason is None
    assert out.p_value == 0.05


def _q95_trials(spread: Any) -> dict[str, Any]:
    return {"n_rule": "cumulative_effective_same_tf_overlapping_train",
            "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": spread}  # fmt: skip


def test_recheck_spread_rows(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R61: the run's q95 spread decides; bool / overflow spreads fall back to legacy."""
    (tmp_path / "a").mkdir()
    rejected = _recheck(tmp_path / "a", monkeypatch, dsr_trials=_q95_trials(1.2))
    assert rejected.passed_dsr is False
    assert rejected.deflated_sharpe is not None
    assert f"{rejected.deflated_sharpe:.4f}" == "1.0000"
    assert rejected.fail_reason == _STAT_1_REASON

    (tmp_path / "b").mkdir()
    bool_spread = _recheck(tmp_path / "b", monkeypatch, dsr_trials=_q95_trials(True))
    assert bool_spread.passed_dsr is False
    assert bool_spread.fail_reason == _LEGACY_STAT_1_REASON

    (tmp_path / "c").mkdir()
    with caplog.at_level("WARNING"):
        huge = _recheck(tmp_path / "c", monkeypatch, dsr_trials=_q95_trials(10**400))
    assert huge.passed_dsr is False
    assert huge.fail_reason == _LEGACY_STAT_1_REASON
    assert "q95 re-check: neff_ev_q95_spread too large for a float; using legacy rule" in (
        caplog.text
    )


def _floor_record(psr_b: float) -> dict[str, Any]:
    return _mtm_record(
        symbol_psr_floor=0.8,
        symbols={
            "A": {"psr": 0.9, "observations": 875},
            "B": {"psr": psr_b, "observations": 875},
        },
    )


@pytest.mark.parametrize(
    ("psr_b", "floor_reason"),
    [
        (0.7, "; DSR per-symbol floor: B PSR(SR>0)=0.700 < 0.80"),
        (0.8, None),
    ],
    ids=["below-floor", "floor-tie"],
)
def test_recheck_floor_rows(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    psr_b: float,
    floor_reason: str | None,
) -> None:
    """R62: worst-mode re-applies the persisted per-symbol floor after the DSR reason."""
    out = _recheck(
        tmp_path,
        monkeypatch,
        record=_floor_record(psr_b),
        dsr_trials=_q95_trials(1.2),
        symbols=["A", "B"],
        symbol_agg="worst",
    )
    assert out.passed_dsr is False
    expected = _STAT_1_REASON if floor_reason is None else _STAT_1_REASON + floor_reason
    assert out.fail_reason == expected


def test_recheck_missing_mtm(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """R63: a champion without MTM inputs fails closed with the unchanged mtm reason."""
    out = _recheck(tmp_path, monkeypatch, record=None)
    assert out.passed_dsr is False
    assert out.fail_reason == _ABSENT_MTM_REASON
