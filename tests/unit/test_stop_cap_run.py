"""The stop cap through a whole discovery run (arch-c29-1 T2): draws, notes, refusal, log.

With liquidation off the GA draws stops only inside the isolated liquidation distance at the run
leverage (10x BTCUSDT: 9.08 %). The mock runs record every genome the backtest adapter receives;
``test_discovery_notes_golden.py`` pins the full notes bytes of the capped default.
"""

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_discovery_notes_golden import GOLDENS, MOCK_ARGV, MOCK_CASES
from tests.unit.test_run_discovery import drive_run_discovery
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.pooling import _CellFns

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vibe_quant.discovery.operators import StrategyChromosome

BTC_NOTE = {
    "max_pct": 9.08,
    "leverage": 10.0,
    "symbol": "BTCUSDT",
    "side": "short",
    "liquidation_distance_pct": 9.561752988047814,
    "margin": 0.95,
    "reference_notional_usdt": 1000.0,
    "tier_snapshot": "2026-10-17",
}
# A larger population than the golden argv: enough uniform stop draws that a 10 % draw above
# 9.08 is certain for the seed, so the uncapped control proves the cap is what moves them.
BASE = [
    "--mock", "--seed", "7", "--population-size", "20", "--max-generations", "2",
    "--max-workers", "-1", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
    "--timeframe", "4h", "--symbols", "BTCUSDT", "--direction", "long",
]  # fmt: skip


class _Recording(run.MockBacktests):
    """The mock adapter, recording every genome any of its backtest functions receives."""

    def __init__(self) -> None:
        self.genomes: list[StrategyChromosome] = []

    def cell_fns(self, **kwargs: Any) -> _CellFns:
        fns = super().cell_fns(**kwargs)

        def wrap(fn: object) -> Callable[[StrategyChromosome], Any]:
            def record(chrom: StrategyChromosome) -> Any:
                self.genomes.append(chrom)
                return fn(chrom)  # type: ignore[operator]

            return record

        factory = fns.backtest_fn_factory
        return _CellFns(
            backtest_fn=wrap(fns.backtest_fn),
            holdout_backtest_fn=None
            if fns.holdout_backtest_fn is None
            else wrap(fns.holdout_backtest_fn),
            backtest_fn_factory=None if factory is None else (lambda s, e: wrap(factory(s, e))),  # type: ignore[operator]
        )


def _drive(tmp_path: Path, argv: list[str], adapter: run.BacktestAdapter) -> run.RunOutcome:
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2025-01-01", {}
    )
    state.close()
    return drive_run_discovery(
        db, run_id, [*argv, "--db", str(db), "--run-id", str(run_id)], adapter
    )


def _max_stop(genomes: list[StrategyChromosome]) -> float:
    return max(
        stop
        for g in genomes
        for stop in (g.stop_loss_pct, g.stop_loss_long_pct, g.stop_loss_short_pct)
        if stop is not None
    )


@pytest.mark.parametrize("extra", [[], ["--num-seeds", "2"]], ids=["one-seed", "two-seeds"])
def test_default_run_caps_stops(tmp_path: Path, extra: list[str]) -> None:
    """R20/R22: the default launch (off, 10x) never draws a 10 % stop, and says why in the notes."""
    adapter = _Recording()
    outcome = _drive(tmp_path, [*BASE, *extra], adapter)

    assert outcome.exit_code == 0, outcome.message
    assert outcome.notes is not None
    assert outcome.notes["stop_loss_cap"] == BTC_NOTE
    assert len(adapter.genomes) > 30
    assert _max_stop(adapter.genomes) <= 9.08
    assert max(chromosome_to_dsl(g)["stop_loss"]["percent"] for g in adapter.genomes) <= 9.08
    assert _max_stop(adapter.genomes) > 8.0  # rescaled across the range, not squeezed low


@pytest.mark.parametrize(
    "extra",
    [
        ["--liquidation-mode", "isolated"],
        ["--leverage", "9"],
        ["--exit-genes", "extended", "--sizing-mode", "atr", "--leverage", "1"],
    ],
    ids=["liquidation-on", "leverage-9", "extended-exits-1x"],
)
def test_uncapped_runs(tmp_path: Path, extra: list[str]) -> None:
    """R21/R25: liquidation on, a leverage whose cap is 10, and 1x extended exits keep today's range."""
    adapter = _Recording()
    outcome = _drive(tmp_path, [*BASE, *extra], adapter)

    assert outcome.exit_code == 0, outcome.message
    assert outcome.notes is not None
    assert "stop_loss_cap" not in outcome.notes
    legacy = [g for g in adapter.genomes if g.exit_rules is None]
    if legacy:  # an extended-exit genome draws its stops from the exit genes
        assert _max_stop(legacy) > 9.08  # the control: the same seed does draw past the cap


@pytest.mark.parametrize("case", sorted(MOCK_CASES))
def test_uncapped_notes_are_the_pre_cap_notes(tmp_path: Path, case: str) -> None:
    """R21b: at --leverage 9 (no cap) the notes are the pre-cap golden plus the leverage key.

    ``discovery_notes_uncapped_<case>.json`` were captured at the stop-cap T2 tree and verified
    equal to ``{**<T1-tree golden>, "leverage": 9.0}`` BEFORE the capped goldens were re-pinned,
    so every byte the cap moved in ``discovery_notes_<case>.json`` comes from the cap alone.
    """
    from tests.unit.test_discovery_notes_golden import notes_bytes

    notes = notes_bytes(
        tmp_path, [*MOCK_ARGV, *MOCK_CASES[case], "--leverage", "9"], run.MockBacktests()
    )
    golden = (GOLDENS / f"discovery_notes_uncapped_{case}.json").read_text(encoding="utf-8")

    assert notes == golden.rstrip("\n")
    assert json.loads(notes)["leverage"] == 9.0
    assert "stop_loss_cap" not in notes


def test_stop_range_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.run"):
        outcome = _drive(tmp_path, BASE, _Recording())

    assert outcome.exit_code == 0, outcome.message
    assert (
        "Stop-loss range 0.50-9.08% (liquidation off, 10x: BTCUSDT short liquidates 9.56% "
        "from entry)" in caplog.text
    )


def test_cli_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R23: a leverage that leaves no room for a stop is exit 2 before any pipeline is built."""
    built: list[object] = []

    def _no_pipeline(*args: object, **kwargs: object) -> object:
        built.append(args)
        raise AssertionError("pipeline built for a refused launch")

    monkeypatch.setattr(run, "DiscoveryPipeline", _no_pipeline)
    outcome = _drive(tmp_path, [*BASE, "--leverage", "107"], _Recording())

    assert outcome.exit_code == 2
    assert outcome.message == (
        "Discovery failed: --leverage 107 with liquidation off liquidates a BTCUSDT short 0.53% "
        "from entry; a 95% stop cap (0.50%) leaves no room above the smallest stop discovery "
        "draws (0.5%). Turn liquidation on (--liquidation-mode isolated or cross) or lower "
        "--leverage."
    )
    assert not built


def test_v2_plateau_neighbours_stay_inside_the_cap() -> None:
    """R20b: ``--fitness v2`` on ZECUSDT (cap 8.46): every genome evaluated, neighbours included."""
    import hashlib
    from array import array

    import numpy as np

    from vibe_quant.discovery.pipeline import DiscoveryPipeline
    from vibe_quant.discovery.run_spec import DiscoveryRunSpec
    from vibe_quant.metrics import DailyReturns

    seen: list[StrategyChromosome] = []

    def backtest(chrom: StrategyChromosome) -> dict[str, Any]:
        seen.append(chrom)
        seed = int.from_bytes(hashlib.sha256(repr(chrom).encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0004 + 0.0003 * rng.random(), 0.01, 300)
        return {
            "sharpe_ratio": 1.0 + float(rng.random()), "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 60 + int(rng.integers(0, 40)), "total_return": float(np.prod(1 + x) - 1),
            "mtm_daily_returns": DailyReturns(19000, array("d", x.tobytes())),
        }  # fmt: skip

    spec = DiscoveryRunSpec.from_argv(
        ["--symbols", "ZECUSDT", "--fitness", "v2", "--evaluator", "nt", "--timeframe", "4h",
         "--start-date", "2024-01-01", "--end-date", "2025-01-01", "--train-test-split", "0",
         "--population-size", "20", "--max-generations", "3", "--elite-count", "2",
         "--max-workers", "-1", "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"]
    )  # fmt: skip
    config = spec.to_config(spec.plan(), symbols=["ZECUSDT"], prior_trials_fn=None)
    assert config.stop_loss_max == 8.46

    random.seed(11)
    result = DiscoveryPipeline(config, backtest).run()

    assert result.plateau is not None and result.plateau["neighbour_evals"] > 0  # type: ignore[operator]
    assert len(seen) > 20
    assert _max_stop(seen) <= 8.46
