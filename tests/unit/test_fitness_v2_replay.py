"""R28 at the endpoint: a v2 champion's /replay child reproduces its stored numbers exactly.

T3 proved R28 at the backtest-fn layer (a fresh ``NTBacktestFn`` of the champion equals the
stored fitness); its handoff left the endpoint check open. Here a real ``--fitness v2``
discovery run writes the champion record, the shipped ``POST /results/{id}/replay/{i}``
creates the screening child and the child runs: the sweep row must equal the champion's
stored sharpe / trades / return / PF / max drawdown bit for bit.

``eval_windows=1`` and no split, so the stored champion numbers ARE the full-range continuous
backtest (``needs_full_range`` stays False and ``full_range_*`` echoes them), which is exactly
what a replay measures. Real NautilusTrader over the local catalog; skipped without one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.db.state_manager import StateManager

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CATALOG_PATH / "data" / "bar").exists(), reason="no data catalog"
)

START, END = "2024-01-01", "2024-04-01"


def _seed_entry() -> dict[str, object]:
    """A real seed genome whose step neighbours stay in range (RSI 5, thresholds 40/60)."""
    from vibe_quant.discovery.genome import chromosome_to_serializable
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 5.0}, ConditionType.LT, 40.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 5.0}, ConditionType.GT, 60.0, None)],
        stop_loss_pct=1.0, take_profit_pct=2.0, direction=Direction.LONG,
    )  # fmt: skip
    return chromosome_to_serializable(chrom)


def _seed_run(state: StateManager) -> int:
    """A finished run carrying one chromosome, the warm start of the v2 run."""
    seed_run = state.create_backtest_run(None, "discovery", ["BTCUSDT"], "4h", START, END, {})
    state.update_backtest_run_status(seed_run, "completed")
    state.save_backtest_result(seed_run, {"total_trades": 0}, trades=[])
    state.update_result_notes(
        seed_run,
        json.dumps(
            {
                "type": "discovery",
                "top_strategies": [{"chromosome": _seed_entry()}],
            }
        ),
    )
    return seed_run


def _v2_discovery(db: Path, seed_run: int) -> int:
    """Run one seeded v2 NT discovery; returns its run id (the champion record is persisted)."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run as discovery_run
    from vibe_quant.discovery.run_spec import build_parser
    from vibe_quant.jobs.manager import BacktestJobManager

    state = StateManager(db)
    run_id = state.create_backtest_run(None, "discovery", ["BTCUSDT"], "4h", START, END, {})
    state.close()
    args = build_parser().parse_args([
        "--run-id", str(run_id), "--db", str(db),
        "--symbols", "BTCUSDT", "--timeframe", "4h",
        "--start-date", START, "--end-date", END,
        "--population-size", "6", "--max-generations", "2", "--elite-count", "1",
        "--max-workers", "-1", "--seed", "1",
        # the launch surface under test
        "--fitness", "v2",
        "--gate-pool", "all", "--eval-windows", "1", "--train-test-split", "0",
        "--indicator-pool", "RSI", "--direction", "long",
        "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate",
        "--seed-from-run", str(seed_run),
    ])  # fmt: skip
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    try:
        outcome = discovery_run.run_discovery(
            args,
            discovery_run.NTBacktests(),
            discovery_run.StateRecordStore(state, jobs),
            run_id=run_id,
            started_at=0.0,
        )
    finally:
        state.close()
        jobs.close()
    assert outcome.exit_code == 0, outcome.message
    return run_id


def _notes(db: Path, run_id: int) -> dict[str, Any]:
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db)
    try:
        result = state.get_backtest_result(run_id)
    finally:
        state.close()
    assert result is not None
    return json.loads(str(result["notes"]))


@pytest.mark.slow(
    reason="a real seeded v2 NT discovery (pop 6, 2 gens) plus its replay child, 3 months of 4h BTC"
)
async def test_v2_champion_replays_exactly(tmp_path: Path) -> None:
    """R28: the /replay child of a v2 champion reproduces sharpe, trades, return, PF and max DD."""
    from tests.unit.test_pooled_replay import _client, _row, _run_child
    from vibe_quant.db.state_manager import StateManager

    db = tmp_path / "state.db"
    state = StateManager(db)
    seed_run = _seed_run(state)
    state.close()

    discovery_id = _v2_discovery(db, seed_run)
    notes = _notes(db, discovery_id)
    assert notes["fitness"] == "v2"
    # The plateau really ran (only the top 5% keys are re-scored, so the champion itself need
    # not be one of them); the replay must hold whatever the record says, plateau or not.
    assert notes["plateau"]["scored"] >= 1 and notes["plateau"]["neighbour_evals"] > 0
    champions = notes["top_strategies"]
    assert champions, "toothless: the seeded v2 run produced no champion"
    champ = champions[0]
    assert champ["fitness_v2"]["final"] == champ["score"]

    async with _client(db) as (client, _state):
        r = await client.post(f"/api/discovery/results/{discovery_id}/replay/0")
        assert r.status_code == 201, r.text
        replay_id = r.json()["replay_run_id"]
        # One window, no split: the stored numbers ARE a continuous full-range backtest, so
        # replay must not excuse a difference (no metrics_note).
        assert r.json()["metrics_note"] is None

    assert _run_child(db, replay_id) == 0
    rows, run = _row(db, replay_id)
    assert run["status"] == "completed"
    (row,) = rows
    assert row["sharpe_ratio"] == champ["full_range_sharpe"]
    assert row["total_trades"] == champ["full_range_trades"]
    assert row["total_return"] == champ["full_range_return_pct"]
    assert row["max_drawdown"] == champ["full_range_max_dd"]
    assert row["profit_factor"] == champ["full_range_pf"]
    assert row["total_trades"] > 0, "a champion that never trades proves nothing"
