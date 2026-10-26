"""Cumulative DSR gate N + deterministic N_eff cap (bd vibe-quant-yul7u.28, .30; megq1).

The discovery DSR gate's trial count N is this run's EV-matched effective count
(``notes.dsr_trials.run_n_eff``) PLUS the ``run_n_eff`` of earlier non-mock
discovery runs with the SAME timeframe whose train window OVERLAPS this run's
train window, recorded under ``n_rule``
``cumulative_effective_same_tf_overlapping_train`` (decision A3). Earlier runs
with no such rule are tallied (``prior_legacy_runs`` / ``prior_legacy_raw``) but
are not charged. ``notes.dsr_trials`` records the rule
(``gate_n`` / ``prior_n`` / ``prior_run_ids`` / ``n_rule``).

The informational N_eff eigendecomposition (never used by the gate) is capped
deterministically at 500 series so O(m^3) cannot blow up.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from array import array
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.gate_chain import TrialBudget
from vibe_quant.discovery.neff import N_RULE_EFFECTIVE
from vibe_quant.metrics import DailyReturns

_ANN = math.sqrt(252.0)


def _new_rule(run_n_eff: Any) -> dict[str, Any]:
    """A ``notes.dsr_trials`` payload recorded under the post-switch n_rule."""
    return {"n_rule": N_RULE_EFFECTIVE, "run_n_eff": run_n_eff}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(sr_ann: float, n: int = 1264, seed: int = 1, first_day: int = 19_000) -> DailyReturns:
    """Gaussian daily returns standardized to EXACTLY ``sr_ann`` annualized Sharpe."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    sigma = 0.01
    x = z * sigma + sr_ann / _ANN * sigma
    return DailyReturns(first_day, array("d", x.tobytes()))


def _chrom(**kw: Any) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 0.1, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 200, "total_return": 0.3, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


def _pipeline(**cfg: Any) -> Any:
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2022-06-01", "end_date": "2025-11-17",
        "require_bootstrap_ci": False, "require_dsr": True,
    }  # fmt: skip
    base.update(cfg)
    fn = NTBacktestFn(["BTCUSDT"], "4h", base["start_date"], base["end_date"])
    return DiscoveryPipeline(DiscoveryConfig(**base), backtest_fn=fn)


def _conn(rows: list[tuple[Any, ...]]) -> sqlite3.Connection:
    """(id, run_mode, timeframe, start_date, end_date, notes) -> in-memory DB."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE backtest_runs (id INTEGER, run_mode TEXT, timeframe TEXT,"
        " start_date TEXT, end_date TEXT)"
    )
    conn.execute("CREATE TABLE backtest_results (run_id INTEGER, notes TEXT)")
    for rid, mode, tf, sd, ed, notes in rows:
        conn.execute("INSERT INTO backtest_runs VALUES (?,?,?,?,?)", (rid, mode, tf, sd, ed))
        raw = notes if (notes is None or isinstance(notes, str)) else json.dumps(notes)
        conn.execute("INSERT INTO backtest_results VALUES (?,?)", (rid, raw))
    return conn


def _state(conn: sqlite3.Connection) -> Any:
    """Real StateManager (locked query methods) over the pre-built ``conn``."""
    from vibe_quant.db.state_manager import StateManager

    mgr = StateManager()
    mgr._conn = conn  # noqa: SLF001 -- skip get_connection/init_schema
    return mgr


# ---------------------------------------------------------------------------
# Prior-trial counting on a tmp DB
# ---------------------------------------------------------------------------

THIS_WINDOW = ("2024-03-01", "2024-09-01")


def _rows() -> list[tuple[Any, ...]]:
    return [
        # same tf, train window overlaps THIS_WINDOW, new rule -> counted (79)
        (1, "discovery", "4h", "2024-01-01", "2024-06-01",
         {"evaluated": 79, "dsr_trials": _new_rule(79)}),
        # same tf, disjoint window (all before) -> not counted
        (2, "discovery", "4h", "2022-01-01", "2022-02-01", {"evaluated": 500}),
        # other timeframe -> not counted here, but counted for a 1h lookup (40)
        (3, "discovery", "1h", "2024-01-01", "2024-06-01",
         {"evaluated": 40, "dsr_trials": _new_rule(40)}),
        # mock -> not a real trial
        (4, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 500, "mock": True}),
        # not discovery -> not counted
        (5, "screening", "4h", "2024-01-01", "2024-06-01", {"evaluated": 77}),
        # same tf, overlaps, new rule -> counted (21)
        (6, "discovery", "4h", "2024-04-01", "2024-10-01",
         {"evaluated": 21, "dsr_trials": _new_rule(21)}),
        # garbled notes -> skip
        (7, "discovery", "4h", "2024-01-01", "2024-06-01", "not json"),
        # notes carry train_dates: b window disjoint but train_dates overlap, new rule -> 5
        (8, "discovery", "4h", "2022-01-01", "2022-02-01",
         {"evaluated": 5, "train_dates": ["2024-02-01", "2024-04-01"],
          "dsr_trials": _new_rule(5)}),
        # train_dates present and DISJOINT even though b window overlaps -> not counted
        (9, "discovery", "4h", "2024-04-01", "2024-08-01",
         {"evaluated": 999, "train_dates": ["2022-01-01", "2022-02-01"]}),
        # malformed train_dates -> garbled, skip
        (10, "discovery", "4h", "2024-01-01", "2024-06-01",
         {"evaluated": 3, "train_dates": "oops"}),
        # later id (this run and beyond) -> excluded
        (11, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 12345}),
    ]


def test_prior_trials_counts_only_same_tf_overlapping_train() -> None:
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(_rows())
    n, run_ids, legacy_runs, legacy_raw = prior_discovery_trials(
        _state(conn), 11, "4h", THIS_WINDOW
    )
    assert n == 79 + 21 + 5
    assert run_ids == [1, 6, 8]
    assert legacy_runs == 0 and legacy_raw == 0


def test_prior_trials_default_timeframe_and_symbol_agnostic() -> None:
    from vibe_quant.discovery.record import prior_discovery_trials

    # A prior run with a DIFFERENT symbol set still counts (symbols not filtered).
    conn = _conn(
        [(1, "discovery", "4h", "2024-01-01", "2024-06-01",
          {"evaluated": 12, "dsr_trials": _new_rule(12)})]
    )
    n, run_ids, _lr, _lraw = prior_discovery_trials(_state(conn), 2, "4h", THIS_WINDOW)
    assert n == 12 and run_ids == [1]


def test_prior_trials_db_error_returns_none_not_zero() -> None:
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = sqlite3.connect(":memory:")  # no tables -> DB error
    n, run_ids, legacy_runs, legacy_raw = prior_discovery_trials(
        _state(conn), 5, "4h", ("2024-01-01", "2024-02-01")
    )
    assert n is None and run_ids == [] and (legacy_runs, legacy_raw) == (0, 0)


@pytest.mark.parametrize(
    ("prior_window", "why"),
    [
        (("2024-01-01", "2024-03-01"), "prior ends exactly where this window starts"),
        (("2024-09-01", "2024-12-01"), "prior starts exactly where this window ends"),
        (("2024-10-01", "2024-12-01"), "prior starts strictly after this window ends"),
        (("2023-10-01", "2024-02-01"), "prior ends strictly before this window starts"),
    ],
)
def test_prior_trials_window_boundaries_not_overlapping(
    prior_window: tuple[str, str], why: str
) -> None:
    """Half-open overlap: touching or disjoint windows contribute nothing."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(
        [(1, "discovery", "4h", prior_window[0], prior_window[1],
          {"evaluated": 7, "dsr_trials": _new_rule(7)})]
    )
    n, run_ids, legacy_runs, legacy_raw = prior_discovery_trials(_state(conn), 2, "4h", THIS_WINDOW)
    assert (n, run_ids, legacy_runs, legacy_raw) == (0, [], 0, 0), why


def test_prior_trials_one_day_overlap_counts() -> None:
    """The smallest overlap (1 day at each edge) IS counted -- guards over-tight filters."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(
        [
            (1, "discovery", "4h", "2024-01-01", "2024-03-02",
             {"evaluated": 3, "dsr_trials": _new_rule(3)}),
            (2, "discovery", "4h", "2024-08-31", "2024-12-01",
             {"evaluated": 4, "dsr_trials": _new_rule(4)}),
        ]
    )
    n, run_ids, _lr, _lraw = prior_discovery_trials(_state(conn), 3, "4h", THIS_WINDOW)
    assert (n, run_ids) == (7, [1, 2])


def test_prior_trials_garbled_evaluated_skips_only_that_row() -> None:
    """One unusable ``evaluated`` must not discard the other rows' count."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(
        [
            (1, "discovery", "4h", "2024-01-01", "2024-06-01",
             {"evaluated": 10, "dsr_trials": _new_rule(10)}),
            (2, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": "n/a"}),
            (3, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": [1, 2]}),
            (4, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": {"x": 1}}),
            (5, "discovery", "4h", "2024-01-01", "2024-06-01",
             {"evaluated": 5, "dsr_trials": _new_rule(5)}),
        ]
    )
    n, run_ids, _lr, _lraw = prior_discovery_trials(_state(conn), 6, "4h", THIS_WINDOW)
    assert (n, run_ids) == (15, [1, 5])


def test_prior_trials_garbled_rows_warn_and_do_not_change_n(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both skip reasons are logged (with the run id) and N is unchanged."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(
        [
            (1, "discovery", "4h", "2024-01-01", "2024-06-01",
             {"evaluated": 10, "dsr_trials": _new_rule(10)}),
            # garbled notes JSON -> skipped, logged
            (2, "discovery", "4h", "2024-01-01", "2024-06-01", "not json"),
            # overlapping window but non-integer evaluated -> skipped, logged
            (3, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": "n/a"}),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="vibe_quant.discovery.__main__"):
        n, run_ids, _lr, _lraw = prior_discovery_trials(_state(conn), 4, "4h", THIS_WINDOW)

    assert (n, run_ids) == (10, [1])
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2, messages
    assert any(
        "prior trial count" in m and "run 2" in m and "garbled notes" in m for m in messages
    ), messages
    assert any(
        "prior trial count" in m and "run 3" in m and "garbled evaluated" in m
        for m in messages
    ), messages


@pytest.mark.parametrize(
    "window", [("", "2024-09-01"), ("2024-03-01", ""), ("n/a", "2024-09-01"), ("2024-03", "x")]
)
def test_prior_trials_unusable_current_window_is_an_error(window: tuple[str, str]) -> None:
    """Unusable current window -> None (the pipeline records prior_error), not a silent 0."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn([(1, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 9})])
    assert prior_discovery_trials(_state(conn), 2, "4h", window) == (None, [], 0, 0)


def test_build_prior_trials_fn_mock_guard_and_run_ids() -> None:
    from vibe_quant.discovery.record import build_prior_trials_fn

    conn = _conn(_rows())
    state = _state(conn)

    fn, ids, legacy = build_prior_trials_fn(state, 11, "4h", mock=True)
    assert fn is None and ids == [] and legacy == {}  # mock runs keep the run-only N

    fn, ids, legacy = build_prior_trials_fn(state, 11, "4h", mock=False)
    assert fn is not None and ids == []  # nothing looked up until the pipeline asks
    assert fn(*THIS_WINDOW) == 79 + 21 + 5
    assert ids == [1, 6, 8]  # the contributing ids ride along in the SAME list object
    assert legacy == {"runs": 0, "raw": 0}
    # a later lookup that matches nothing must not leave stale ids behind
    assert fn("2030-01-01", "2030-02-01") == 0
    assert ids == [] and legacy == {"runs": 0, "raw": 0}
    # a different timeframe is honoured (closure carries it)
    fn_1h, ids_1h, _legacy_1h = build_prior_trials_fn(state, 11, "1h", mock=False)
    assert fn_1h is not None and fn_1h(*THIS_WINDOW) == 40 and ids_1h == [3]


# ---------------------------------------------------------------------------
# The gate uses gate_n = run N + prior_n
# ---------------------------------------------------------------------------


def test_gate_default_uses_run_n_only() -> None:
    chrom = _chrom()
    pipe = _pipeline()
    kept = pipe.gates.screen([(chrom, _fit(mtm_daily_returns=_series(1.6)))], TrialBudget(10, {}))
    assert [c.uid for c, _ in kept] == [chrom.uid]
    assert pipe.gates.dsr_records[chrom.uid]["trials"] == 10  # run-only N
    assert pipe.gates.dsr_trials is not None
    # no retained series -> the estimator fails closed to raw, so run N == 10
    assert pipe.gates.dsr_trials["run_n_eff"] == 10
    assert pipe.gates.dsr_trials["neff_ev_source"] == "fallback_raw"
    assert pipe.gates.dsr_trials["prior_n"] == 0 and pipe.gates.dsr_trials["gate_n"] == 10


def test_gate_rejects_candidate_that_fails_at_cumulative_n() -> None:
    # SR 1.6 annualized clears DSR at N=10 but not at N=1010.
    chrom = _chrom()
    only = _pipeline()
    assert only.gates.screen(
        [(chrom, _fit(mtm_daily_returns=_series(1.6)))], TrialBudget(10, {})
    )
    cumulative = _pipeline(prior_trials_fn=lambda s, e: 1000)
    kept = cumulative.gates.screen(
        [(chrom, _fit(mtm_daily_returns=_series(1.6)))], TrialBudget(10, {})
    )
    assert kept == []
    assert cumulative.gates.dsr_records[chrom.uid]["trials"] == 1010
    assert cumulative.gates.dsr_trials is not None
    assert cumulative.gates.dsr_trials["prior_n"] == 1000
    assert cumulative.gates.dsr_trials["gate_n"] == 1010
    reasons = cumulative.gates.rejections[0]["reasons"]
    assert any(r.startswith("DSR not significant") and "N=1010" in r for r in reasons)


def test_pipeline_passes_its_train_window_to_prior_fn() -> None:
    """The fn must be asked about THIS run's train window (config start/end), in order."""
    calls: list[tuple[str, str]] = []

    def record(start: str, end: str) -> int:
        calls.append((start, end))
        return 5

    pipe = _pipeline(
        prior_trials_fn=record,
        train_test_split=0.8,
        holdout_start_date="2025-11-18",
        holdout_end_date="2026-03-17",
    )
    pipe.gates.screen([(_chrom(), _fit(mtm_daily_returns=_series(1.6)))], TrialBudget(10, {}))
    assert calls == [("2022-06-01", "2025-11-17")]
    assert pipe.gates.dsr_trials is not None and pipe.gates.dsr_trials["gate_n"] == 15


def test_prior_fn_error_uses_zero_and_records_it() -> None:
    def boom(_s: str, _e: str) -> int | None:
        raise RuntimeError("db down")

    for bad in (boom, lambda s, e: None):
        pipe = _pipeline(prior_trials_fn=bad)
        kept = pipe.gates.screen(
            [(_chrom(), _fit(mtm_daily_returns=_series(1.6)))], TrialBudget(10, {})
        )
        assert len(kept) == 1  # fail open to run-only N
        trials = pipe.gates.dsr_trials
        assert trials is not None
        assert trials["prior_n"] is None and trials["prior_error"] is True
        # fail open keeps the RUN's effective count, not zero
        assert trials["gate_n"] == trials["run_n_eff"] == 10


def test_no_dsr_run_is_charged_at_raw() -> None:
    """R19b (B1): a --no-dsr run records n_rule NEW at raw, so A3 still charges it."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(
        [
            (
                1,
                "discovery",
                "4h",
                "2024-01-01",
                "2024-06-01",
                {
                    "evaluated": 40,
                    "dsr_trials": {
                        "raw": 40,
                        "run_n_eff": 40,
                        "neff_ev_source": "dsr_disabled",
                        "n_rule": N_RULE_EFFECTIVE,
                        "gate_n": None,
                    },
                },
            )
        ]
    )
    n, ids, legacy_runs, legacy_raw = prior_discovery_trials(_state(conn), 2, "4h", THIS_WINDOW)
    assert n == 40 and ids == [1]
    assert legacy_runs == 0 and legacy_raw == 0


def test_prior_counts_effective_rule_runs_only() -> None:
    """R20 (A3): only new-rule runs are charged; legacy runs are tallied, not counted."""
    from vibe_quant.discovery.record import prior_discovery_trials

    conn = _conn(
        [
            # new rule: charged at run_n_eff 12 though evaluated is 40
            (1, "discovery", "4h", "2024-01-01", "2024-06-01",
             {"evaluated": 40, "dsr_trials": _new_rule(12)}),
            # legacy (no dsr_trials): evaluated 127 is NOT charged, only tallied
            (2, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 127}),
            # pre-switch rows carry the OLD n_rule: also legacy, never charged
            (3, "discovery", "4h", "2024-01-01", "2024-06-01",
             {"evaluated": 9,
              "dsr_trials": {"n_rule": "cumulative_same_tf_overlapping_train",
                             "run_n_eff": 9}}),
        ]
    )
    n, ids, legacy_runs, legacy_raw = prior_discovery_trials(_state(conn), 4, "4h", THIS_WINDOW)
    assert n == 12
    assert ids == [1]
    assert legacy_runs == 2 and legacy_raw == 127 + 9


@pytest.mark.parametrize(
    "bad",
    [None, True, "12", 12.5, 0, 41],
    ids=["null", "true", "str", "float", "zero", "over-evaluated"],
)
def test_prior_bad_run_n_eff_counts_evaluated(
    bad: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """R21: a new-rule run_n_eff that is absent/null/bool/str/float/0/>evaluated fails closed."""
    from vibe_quant.discovery.record import prior_discovery_trials

    payload: dict[str, Any] = {"n_rule": N_RULE_EFFECTIVE}
    if bad is not None:
        payload["run_n_eff"] = bad
    conn = _conn([(1, "discovery", "4h", "2024-01-01", "2024-06-01",
                   {"evaluated": 40, "dsr_trials": payload})])
    with caplog.at_level(logging.WARNING, logger="vibe_quant.discovery.__main__"):
        n, ids, legacy_runs, legacy_raw = prior_discovery_trials(
            _state(conn), 2, "4h", THIS_WINDOW
        )
    assert n == 40 and ids == [1]
    assert legacy_runs == 0 and legacy_raw == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("invalid run_n_eff" in m and "run 1" in m for m in warnings), warnings


# ---------------------------------------------------------------------------
# notes.dsr_trials fields
# ---------------------------------------------------------------------------


def test_dsr_trials_note_records_cumulative_rule() -> None:
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.discovery.record import dsr_trials_note

    result = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79,
        converged=False, convergence_generation=None,
        dsr_trials={"raw": 79, "non_flat": 3, "prior_n": 921, "gate_n": 1000},
    )  # fmt: skip
    note = dsr_trials_note(result, [1, 6], {"runs": 4, "raw": 22104})
    assert note["raw"] == 79
    assert note["prior_n"] == 921
    assert note["prior_run_ids"] == [1, 6]
    assert note["gate_n"] == 1000
    assert note["prior_legacy_runs"] == 4 and note["prior_legacy_raw"] == 22104
    assert note["n_rule"] == N_RULE_EFFECTIVE
    # the old informational-only keys must be gone (no contradicting numbers)
    assert "cumulative_n" not in note and "p_value_at_cumulative_n" not in note


def test_dsr_trials_note_error_and_missing_prior_n() -> None:
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.discovery.record import dsr_trials_note

    errored = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79,
        converged=False, convergence_generation=None,
        dsr_trials={"raw": 79, "prior_n": None, "prior_error": True, "gate_n": 79},
    )  # fmt: skip
    note = dsr_trials_note(errored, [])
    assert note["prior_n"] is None and note["prior_error"] is True
    assert note["gate_n"] == 79 and note["prior_run_ids"] == []
    # no legacy run was reported -> the tally defaults to zero, never missing
    assert note["prior_legacy_runs"] == 0 and note["prior_legacy_raw"] == 0

    legacy = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79,
        converged=False, convergence_generation=None, dsr_trials={"raw": 79},
    )  # fmt: skip
    fallback = dsr_trials_note(legacy, [])
    assert fallback["prior_n"] == 0 and fallback["gate_n"] == 79


# ---------------------------------------------------------------------------
# Multi-seed + main() wiring: the notes must carry the cumulative-N record
# ---------------------------------------------------------------------------


def _mock_cfg(**kw: Any) -> Any:
    from vibe_quant.discovery.pipeline import DiscoveryConfig

    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 1, "elite_count": 1, "top_k": 1,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2022-06-01", "end_date": "2025-11-17",
        "require_bootstrap_ci": False, "require_dsr": True,
    }  # fmt: skip
    base.update(kw)
    return DiscoveryConfig(**base)


def test_multi_seed_result_carries_dsr_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pooled multi-seed result records raw=pooled N, gate_n and prior_n (not None)."""
    from vibe_quant.discovery import pooling, record
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    champ = (_chrom(), _fit(mtm_daily_returns=_series(5.0)))
    seeds_seen: list[int] = []

    def fake_run(self: Any) -> DiscoveryResult:
        # each seed retains its own non-flat series (feeds the informational N_eff)
        seeds_seen.append(len(seeds_seen) + 1)
        self._fitness_cache[f"seed{seeds_seen[-1]}"] = _fit(
            mtm_daily_returns=_series(1.0, seed=seeds_seen[-1])
        )
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=400,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 30)
    res = pooling._run_multi_seed(3, cfg, lambda c: {}, "/tmp/vq_disc_progress.json")
    assert res.total_candidates_evaluated == 1200
    trials = res.dsr_trials
    assert trials is not None
    assert trials["raw"] == 1200  # POOLED trials, not one seed's
    assert trials["prior_n"] == 30 and trials["gate_n"] == 1230
    assert trials["non_flat"] == 3  # the seeds' retained series were pooled
    note = record.dsr_trials_note(res, [4, 9])
    assert note["prior_run_ids"] == [4, 9] and note["gate_n"] == 1230


def _multi_seed_result(
    monkeypatch: pytest.MonkeyPatch, champ: tuple[Any, Any]
) -> tuple[Any, Any]:
    """Run the 3x400 mock multi-seed path with ``champ`` returned by every seed."""
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery import pooling
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    def fake_run(self: Any) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=400,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 30)
    res = pooling._run_multi_seed(3, cfg, lambda c: {}, "/tmp/vq_disc_progress.json")
    return cli, res


def test_multi_seed_result_carries_pooled_dsr_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A surviving multi-seed champion gets the POOLED record (trials = pooled gate N)."""
    champ = (_chrom(), _fit(mtm_daily_returns=_series(5.0)))
    _, res = _multi_seed_result(monkeypatch, champ)
    uid = champ[0].uid
    assert uid in {c.uid for c, _ in res.top_strategies}
    # C1: 1230 = 3x400 pooled evaluated + 30 prior; a per-seed record (400/430) is wrong.
    assert res.dsr_records[uid]["trials"] == 1230


def test_multi_seed_rejected_champion_carries_pooled_dsr_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pooled-rejected multi-seed champion is dropped but keeps its pooled record."""
    champ = (_chrom(), _fit(mtm_daily_returns=_series(0.3)))
    _, res = _multi_seed_result(monkeypatch, champ)
    uid = champ[0].uid
    # C2: not a champion, yet its DSR record (and the rejection row's dsr) survive.
    assert uid not in {c.uid for c, _ in res.top_strategies}
    assert res.dsr_records[uid]["trials"] == 1230
    rows = [
        r for r in res.guardrail_rejections
        if r.get("uid") == uid and r.get("stage") == "pooled_guardrails"
    ]  # fmt: skip
    assert len(rows) == 1
    row_dsr = rows[0]["dsr"]
    assert isinstance(row_dsr, dict) and row_dsr["trials"] == 1230


def test_multi_seed_champion_entry_carries_dsr(monkeypatch: pytest.MonkeyPatch) -> None:
    """C4: champion_entry for a multi-seed champion has key "dsr" with pooled trials."""
    champ = (_chrom(), _fit(mtm_daily_returns=_series(5.0)))
    from vibe_quant.discovery import record

    _, res = _multi_seed_result(monkeypatch, champ)
    chrom, fitness = res.top_strategies[0]
    entry = record.champion_entry(
        chrom, fitness, timeframe="4h",
        dsr=res.dsr_records.get(chrom.uid),
        buy_hold=res.buy_hold_records.get(chrom.uid),
    )  # fmt: skip
    assert entry["dsr"]["trials"] == 1230


def _main_notes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, num_seeds: int
) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    """Run the mock CLI with a faked builder; return (persisted notes, builder calls)."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery import record

    db_path = tmp_path / "state.db"
    state = StateManager(db_path)
    sid = state.create_strategy(
        name="__dsr_cum__", dsl_config={"type": "discovery"}, description="t",
        strategy_type="discovery",
    )  # fmt: skip
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="discovery", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2025-01-01", end_date="2025-02-01", parameters={},
    )  # fmt: skip
    state.close()

    builder_calls: list[tuple[Any, ...]] = []

    def builder(
        state_: Any, run_id_: int, timeframe: str, *, mock: bool
    ) -> tuple[Any, list[int], dict[str, int]]:
        builder_calls.append((run_id_, timeframe, mock))
        ids: list[int] = []  # the real helper fills this shared list when the fn is called
        legacy: dict[str, int] = {"runs": 0, "raw": 0}

        def fn(s: str, e: str) -> int:
            ids[:] = [7, 9]
            legacy["runs"], legacy["raw"] = 2, 500
            return 50

        return fn, ids, legacy

    monkeypatch.setattr(record, "build_prior_trials_fn", builder)
    argv = [
        "prog", "--run-id", str(run_id), "--population-size", "6", "--max-generations", "2",
        "--elite-count", "1", "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2025-01-01", "--end-date", "2025-02-01",
        "--num-seeds", str(num_seeds), "--db", str(db_path), "--mock",
    ]  # fmt: skip
    monkeypatch.setattr("sys.argv", argv)
    assert cli.main() == 0
    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    return json.loads(result["notes"]), builder_calls


@pytest.mark.parametrize("num_seeds", [1, 2])
def test_main_persists_cumulative_dsr_trials(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, num_seeds: int
) -> None:
    notes, builder_calls = _main_notes(tmp_path, monkeypatch, num_seeds)
    # main() hands the helper this run's id + timeframe and the mock flag
    assert len(builder_calls) == 1
    run_id, timeframe, mock = builder_calls[0]
    assert timeframe == "1h" and mock is True and isinstance(run_id, int)
    trials = notes["dsr_trials"]
    # R22: the exact persisted key set of a new-rule run (mock -> no series).
    assert set(trials) == {
        "raw", "non_flat", "neff_mean_corr", "neff_eig_pr", "neff_subsampled_from",
        "neff_ev", "neff_ev_q95_spread", "run_n_eff", "neff_ev_source",
        "neff_fallback_reason", "neff_ev_draws", "neff_ev_seed", "dsr_rule",
        "prior_n", "prior_run_ids", "prior_legacy_runs", "prior_legacy_raw",
        "gate_n", "n_rule",
    }, trials
    assert trials["raw"] == notes["evaluated"]
    assert trials["run_n_eff"] == trials["raw"]  # fallback: no retained series
    assert trials["neff_ev_source"] == "fallback_raw" and trials["neff_ev"] is None
    assert trials["neff_ev_draws"] == 4096 and trials["neff_ev_seed"] == 20261018
    assert trials["prior_n"] == 50
    assert trials["prior_run_ids"] == [7, 9]
    assert trials["prior_legacy_runs"] == 2 and trials["prior_legacy_raw"] == 500
    assert trials["gate_n"] == trials["run_n_eff"] + 50
    # F2: the JSON round-trip must preserve a plain int gate_n. A float gate_n
    # (948.0) is rejected by prior_discovery_trials (T4) as an invalid
    # run_n_eff and falls back to evaluated, losing the effective/cumulative N.
    assert type(trials["gate_n"]) is int
    assert trials["n_rule"] == N_RULE_EFFECTIVE


# ---------------------------------------------------------------------------
# yul7u.30: deterministic N_eff eig cap at 500 series
# ---------------------------------------------------------------------------


def test_neff_eig_capped_at_500_deterministic() -> None:
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 0.01, 1264)
    pipe = _pipeline()
    pool: dict[str, FitnessResult] = {}
    m = 600
    for i in range(m):
        x = common + rng.normal(0.0, 0.001, 1264)
        pool[f"k{i:04d}"] = _fit(
            mtm_daily_returns=DailyReturns(19_000, array("d", x.tobytes()))
        )

    trials = pipe.gates.trial_counts(TrialBudget(600, pool))
    assert trials["raw"] == 600
    assert trials["non_flat"] == 600
    assert trials["neff_subsampled_from"] == 600
    neff = trials["neff_eig_pr"]
    assert isinstance(neff, float)
    # deterministic across calls
    assert pipe.gates.trial_counts(TrialBudget(600, pool))["neff_eig_pr"] == neff

    # exactly the evenly-spaced 500-subset of the sorted-uid order was used
    items = sorted(pool.items(), key=lambda kv: str(kv[0]))
    rows = [
        np.frombuffer(fr.mtm_daily_returns.values, dtype=np.float64)  # type: ignore[union-attr]
        for _, fr in items
    ]
    idx = np.unique(np.linspace(0, m - 1, 500).astype(int))
    assert len(idx) == 500
    sel = np.vstack([rows[i] for i in idx])
    corr = np.nan_to_num(np.corrcoef(sel))
    ev = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    expected = float(ev.sum() ** 2 / (ev**2).sum()) * 600 / 500
    assert neff == pytest.approx(expected, rel=1e-12)


def test_neff_eig_uncapped_below_limit() -> None:
    rng = np.random.default_rng(5)
    pipe = _pipeline()
    pool: dict[str, FitnessResult] = {}
    for i in range(10):
        x = rng.normal(0.0005, 0.01, 1264)
        pool[f"k{i:04d}"] = _fit(
            mtm_daily_returns=DailyReturns(19_000, array("d", x.tobytes()))
        )
    trials = pipe.gates.trial_counts(TrialBudget(10, pool))
    assert trials["neff_subsampled_from"] is None
    assert trials["non_flat"] == 10


# ---------------------------------------------------------------------------
# F1 (B1): a NON-mock --no-dsr run persists its new-rule note through main()
# ---------------------------------------------------------------------------


def test_non_mock_no_dsr_run_persists_its_new_rule_note(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 (B1): a real (non-mock) --no-dsr run writes ``notes.dsr_trials`` through
    ``main()`` at ``run_n_eff == raw`` with ``gate_n`` null, so the NEXT
    same-timeframe overlapping-train run charges its raw trials (40).

    End to end: public CLI write -> private DB reload -> real prior lookup. The
    unit tests around ``record.dsr_trials_note`` and ``prior_discovery_trials`` pass a
    manufactured payload, so deleting ``__main__.py``'s ``elif not use_mock:``
    branch survived them -- this test kills that mutant.
    """
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery import record
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    db_path = tmp_path / "state.db"
    state = StateManager(db_path)
    sid = state.create_strategy(
        name="__no_dsr_main__", dsl_config={"type": "discovery"}, description="t",
        strategy_type="discovery",
    )  # fmt: skip
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="discovery", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2024-06-01", parameters={},
    )  # fmt: skip
    state.close()

    def fake_run(self: DiscoveryPipeline) -> DiscoveryResult:
        # Stubbed evaluation: a real, NON-mock run that evaluated 40 strategies
        # and found no champion (the zero-champion path still persists run_notes).
        return DiscoveryResult(
            generations=[], top_strategies=[], total_candidates_evaluated=40,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    monkeypatch.setattr("vibe_quant.discovery.run._check_data_available", lambda symbols: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id), "--population-size", "4",
            "--max-generations", "1", "--elite-count", "1", "--symbols", "BTCUSDT",
            "--timeframe", "4h", "--start-date", "2024-01-01", "--end-date", "2024-06-01",
            "--eval-windows", "1", "--train-test-split", "0", "--max-workers", "1",
            "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate", "--db", str(db_path),
        ],
    )  # fmt: skip
    assert cli.main() == 0

    # Reload the PUBLIC write (a fresh StateManager over the same private DB).
    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    assert result is not None
    notes = json.loads(result["notes"])
    assert notes["mock"] is False  # --mock was NOT passed
    trials = notes["dsr_trials"]
    assert trials["n_rule"] == N_RULE_EFFECTIVE
    assert trials["raw"] == notes["evaluated"] == 40
    assert trials["run_n_eff"] == 40  # == raw, not a computed effective N
    assert trials["neff_ev_source"] == "dsr_disabled"
    assert trials["gate_n"] is None  # no gate ran

    # The later run (same timeframe, overlapping train window) charges this one
    # at its raw count through the REAL prior lookup.
    counted, ids, legacy_runs, legacy_raw = record.prior_discovery_trials(
        state, run_id + 1, "4h", ("2024-03-01", "2024-09-01")
    )
    state.close()
    assert (counted, ids, legacy_runs, legacy_raw) == (40, [run_id], 0, 0)
