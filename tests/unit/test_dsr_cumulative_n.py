"""Cumulative DSR gate N + deterministic N_eff cap (bd vibe-quant-yul7u.28, .30).

The discovery DSR gate's trial count N is this run's distinct evaluated
strategies PLUS the evaluated counts of earlier non-mock discovery runs with
the SAME timeframe whose train window OVERLAPS this run's train window
(counts only, no de-dup). ``notes.dsr_trials`` records the rule
(``gate_n`` / ``prior_n`` / ``prior_run_ids`` / ``n_rule``).

The informational N_eff eigendecomposition (never used by the gate) is capped
deterministically at 500 series so O(m^3) cannot blow up.
"""

from __future__ import annotations

import json
import math
import sqlite3
from array import array
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.metrics import DailyReturns

_ANN = math.sqrt(252.0)


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
    """Duck-typed StateManager: the prior-trial query only touches ``.conn``."""
    return SimpleNamespace(conn=conn)


# ---------------------------------------------------------------------------
# Prior-trial counting on a tmp DB
# ---------------------------------------------------------------------------

THIS_WINDOW = ("2024-03-01", "2024-09-01")


def _rows() -> list[tuple[Any, ...]]:
    return [
        # same tf, train window overlaps THIS_WINDOW -> counted (79)
        (1, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 79}),
        # same tf, disjoint window (all before) -> not counted
        (2, "discovery", "4h", "2022-01-01", "2022-02-01", {"evaluated": 500}),
        # other timeframe -> not counted
        (3, "discovery", "1h", "2024-01-01", "2024-06-01", {"evaluated": 40}),
        # mock -> not a real trial
        (4, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 500, "mock": True}),
        # not discovery -> not counted
        (5, "screening", "4h", "2024-01-01", "2024-06-01", {"evaluated": 77}),
        # same tf, overlaps -> counted (21)
        (6, "discovery", "4h", "2024-04-01", "2024-10-01", {"evaluated": 21}),
        # garbled notes -> skip
        (7, "discovery", "4h", "2024-01-01", "2024-06-01", "not json"),
        # notes carry train_dates: b window disjoint but train_dates overlap -> counted (5)
        (8, "discovery", "4h", "2022-01-01", "2022-02-01",
         {"evaluated": 5, "train_dates": ["2024-02-01", "2024-04-01"]}),
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
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = _conn(_rows())
    n, run_ids = _prior_discovery_trials(
        _state(conn), 11, "4h", THIS_WINDOW
    )
    assert n == 79 + 21 + 5
    assert run_ids == [1, 6, 8]


def test_prior_trials_default_timeframe_and_symbol_agnostic() -> None:
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    # A prior run with a DIFFERENT symbol set still counts (symbols not filtered).
    conn = _conn([(1, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 12})])
    n, run_ids = _prior_discovery_trials(_state(conn), 2, "4h", THIS_WINDOW)
    assert n == 12 and run_ids == [1]


def test_prior_trials_db_error_returns_none_not_zero() -> None:
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = sqlite3.connect(":memory:")  # no tables -> DB error
    n, run_ids = _prior_discovery_trials(
        _state(conn), 5, "4h", ("2024-01-01", "2024-02-01")
    )
    assert n is None and run_ids == []


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
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = _conn([(1, "discovery", "4h", prior_window[0], prior_window[1], {"evaluated": 7})])
    n, run_ids = _prior_discovery_trials(_state(conn), 2, "4h", THIS_WINDOW)
    assert (n, run_ids) == (0, []), why


def test_prior_trials_one_day_overlap_counts() -> None:
    """The smallest overlap (1 day at each edge) IS counted -- guards over-tight filters."""
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = _conn(
        [
            (1, "discovery", "4h", "2024-01-01", "2024-03-02", {"evaluated": 3}),
            (2, "discovery", "4h", "2024-08-31", "2024-12-01", {"evaluated": 4}),
        ]
    )
    n, run_ids = _prior_discovery_trials(_state(conn), 3, "4h", THIS_WINDOW)
    assert (n, run_ids) == (7, [1, 2])


def test_prior_trials_garbled_evaluated_skips_only_that_row() -> None:
    """One unusable ``evaluated`` must not discard the other rows' count."""
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = _conn(
        [
            (1, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 10}),
            (2, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": "n/a"}),
            (3, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": [1, 2]}),
            (4, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": {"x": 1}}),
            (5, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 5}),
        ]
    )
    n, run_ids = _prior_discovery_trials(_state(conn), 6, "4h", THIS_WINDOW)
    assert (n, run_ids) == (15, [1, 5])


@pytest.mark.parametrize(
    "window", [("", "2024-09-01"), ("2024-03-01", ""), ("n/a", "2024-09-01"), ("2024-03", "x")]
)
def test_prior_trials_unusable_current_window_is_an_error(window: tuple[str, str]) -> None:
    """Unusable current window -> None (the pipeline records prior_error), not a silent 0."""
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = _conn([(1, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 9})])
    assert _prior_discovery_trials(_state(conn), 2, "4h", window) == (None, [])


def test_build_prior_trials_fn_mock_guard_and_run_ids() -> None:
    from vibe_quant.discovery.__main__ import _build_prior_trials_fn

    conn = _conn(_rows())
    state = _state(conn)

    fn, ids = _build_prior_trials_fn(state, 11, "4h", mock=True)
    assert fn is None and ids == []  # mock runs keep the run-only N

    fn, ids = _build_prior_trials_fn(state, 11, "4h", mock=False)
    assert fn is not None and ids == []  # nothing looked up until the pipeline asks
    assert fn(*THIS_WINDOW) == 79 + 21 + 5
    assert ids == [1, 6, 8]  # the contributing ids ride along in the SAME list object
    # a later lookup that matches nothing must not leave stale ids behind
    assert fn("2030-01-01", "2030-02-01") == 0
    assert ids == []
    # a different timeframe is honoured (closure carries it)
    fn_1h, ids_1h = _build_prior_trials_fn(state, 11, "1h", mock=False)
    assert fn_1h is not None and fn_1h(*THIS_WINDOW) == 40 and ids_1h == [3]


# ---------------------------------------------------------------------------
# The gate uses gate_n = run N + prior_n
# ---------------------------------------------------------------------------


def test_gate_default_uses_run_n_only() -> None:
    chrom = _chrom()
    pipe = _pipeline()
    kept = pipe._validate_top_strategies([(chrom, _fit(daily_returns=_series(1.6)))], 10)
    assert [c.uid for c, _ in kept] == [chrom.uid]
    assert pipe._dsr_records[chrom.uid]["trials"] == 10  # run-only N
    assert pipe._dsr_trials is not None
    assert pipe._dsr_trials["prior_n"] == 0 and pipe._dsr_trials["gate_n"] == 10


def test_gate_rejects_candidate_that_fails_at_cumulative_n() -> None:
    # SR 1.6 annualized clears DSR at N=10 but not at N=1010.
    chrom = _chrom()
    only = _pipeline()
    assert only._validate_top_strategies(
        [(chrom, _fit(daily_returns=_series(1.6)))], 10
    )
    cumulative = _pipeline(prior_trials_fn=lambda s, e: 1000)
    kept = cumulative._validate_top_strategies(
        [(chrom, _fit(daily_returns=_series(1.6)))], 10
    )
    assert kept == []
    assert cumulative._dsr_records[chrom.uid]["trials"] == 1010
    assert cumulative._dsr_trials is not None
    assert cumulative._dsr_trials["prior_n"] == 1000
    assert cumulative._dsr_trials["gate_n"] == 1010
    reasons = cumulative._guardrail_rejections[0]["reasons"]
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
    pipe._validate_top_strategies([(_chrom(), _fit(daily_returns=_series(1.6)))], 10)
    assert calls == [("2022-06-01", "2025-11-17")]
    assert pipe._dsr_trials is not None and pipe._dsr_trials["gate_n"] == 15


def test_prior_fn_error_uses_zero_and_records_it() -> None:
    def boom(_s: str, _e: str) -> int | None:
        raise RuntimeError("db down")

    for bad in (boom, lambda s, e: None):
        pipe = _pipeline(prior_trials_fn=bad)
        kept = pipe._validate_top_strategies(
            [(_chrom(), _fit(daily_returns=_series(1.6)))], 10
        )
        assert len(kept) == 1  # fail open to run-only N
        trials = pipe._dsr_trials
        assert trials is not None
        assert trials["prior_n"] is None and trials["prior_error"] is True
        assert trials["gate_n"] == 10


# ---------------------------------------------------------------------------
# notes.dsr_trials fields
# ---------------------------------------------------------------------------


def test_dsr_trials_note_records_cumulative_rule() -> None:
    from vibe_quant.discovery.__main__ import _dsr_trials_note
    from vibe_quant.discovery.pipeline import DiscoveryResult

    result = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79,
        converged=False, convergence_generation=None,
        dsr_trials={"raw": 79, "non_flat": 3, "prior_n": 921, "gate_n": 1000},
    )  # fmt: skip
    note = _dsr_trials_note(result, [1, 6])
    assert note["raw"] == 79
    assert note["prior_n"] == 921
    assert note["prior_run_ids"] == [1, 6]
    assert note["gate_n"] == 1000
    assert note["n_rule"] == "cumulative_same_tf_overlapping_train"
    # the old informational-only keys must be gone (no contradicting numbers)
    assert "cumulative_n" not in note and "p_value_at_cumulative_n" not in note


def test_dsr_trials_note_error_and_missing_prior_n() -> None:
    from vibe_quant.discovery.__main__ import _dsr_trials_note
    from vibe_quant.discovery.pipeline import DiscoveryResult

    errored = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79,
        converged=False, convergence_generation=None,
        dsr_trials={"raw": 79, "prior_n": None, "prior_error": True, "gate_n": 79},
    )  # fmt: skip
    note = _dsr_trials_note(errored, [])
    assert note["prior_n"] is None and note["prior_error"] is True
    assert note["gate_n"] == 79 and note["prior_run_ids"] == []

    legacy = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79,
        converged=False, convergence_generation=None, dsr_trials={"raw": 79},
    )  # fmt: skip
    fallback = _dsr_trials_note(legacy, [])
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
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    champ = (_chrom(), _fit(daily_returns=_series(5.0)))
    seeds_seen: list[int] = []

    def fake_run(self: Any) -> DiscoveryResult:
        # each seed retains its own non-flat series (feeds the informational N_eff)
        seeds_seen.append(len(seeds_seen) + 1)
        self._fitness_cache[f"seed{seeds_seen[-1]}"] = _fit(
            daily_returns=_series(1.0, seed=seeds_seen[-1])
        )
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=400,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 30)
    res = cli._run_multi_seed(3, cfg, lambda c: {}, "/tmp/vq_disc_progress.json")
    assert res.total_candidates_evaluated == 1200
    trials = res.dsr_trials
    assert trials is not None
    assert trials["raw"] == 1200  # POOLED trials, not one seed's
    assert trials["prior_n"] == 30 and trials["gate_n"] == 1230
    assert trials["non_flat"] == 3  # the seeds' retained series were pooled
    note = cli._dsr_trials_note(res, [4, 9])
    assert note["prior_run_ids"] == [4, 9] and note["gate_n"] == 1230


def _main_notes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, num_seeds: int
) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    """Run the mock CLI with a faked builder; return (persisted notes, builder calls)."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli

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

    def builder(state_: Any, run_id_: int, timeframe: str, *, mock: bool) -> tuple[Any, list[int]]:
        builder_calls.append((run_id_, timeframe, mock))
        ids: list[int] = []  # the real helper fills this shared list when the fn is called

        def fn(s: str, e: str) -> int:
            ids[:] = [7, 9]
            return 50

        return fn, ids

    monkeypatch.setattr(cli, "_build_prior_trials_fn", builder)
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
    assert trials["prior_n"] == 50
    assert trials["prior_run_ids"] == [7, 9]
    assert trials["gate_n"] == trials["raw"] + 50
    assert trials["n_rule"] == "cumulative_same_tf_overlapping_train"


# ---------------------------------------------------------------------------
# yul7u.30: deterministic N_eff eig cap at 500 series
# ---------------------------------------------------------------------------


def test_neff_eig_capped_at_500_deterministic() -> None:
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 0.01, 1264)
    pipe = _pipeline()
    m = 600
    for i in range(m):
        x = common + rng.normal(0.0, 0.001, 1264)
        pipe._fitness_cache[f"k{i:04d}"] = _fit(
            daily_returns=DailyReturns(19_000, array("d", x.tobytes()))
        )

    trials = pipe._dsr_trial_counts(600)
    assert trials["raw"] == 600
    assert trials["non_flat"] == 600
    assert trials["neff_subsampled_from"] == 600
    neff = trials["neff_eig_pr"]
    assert isinstance(neff, float)
    # deterministic across calls
    assert pipe._dsr_trial_counts(600)["neff_eig_pr"] == neff

    # exactly the evenly-spaced 500-subset of the sorted-uid order was used
    items = sorted(pipe._fitness_cache.items(), key=lambda kv: str(kv[0]))
    rows = [
        np.frombuffer(fr.daily_returns.values, dtype=np.float64)  # type: ignore[union-attr]
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
    for i in range(10):
        x = rng.normal(0.0005, 0.01, 1264)
        pipe._fitness_cache[f"k{i:04d}"] = _fit(
            daily_returns=DailyReturns(19_000, array("d", x.tobytes()))
        )
    trials = pipe._dsr_trial_counts(10)
    assert trials["neff_subsampled_from"] is None
    assert trials["non_flat"] == 10
