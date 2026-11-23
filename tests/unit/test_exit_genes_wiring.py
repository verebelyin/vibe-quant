"""GA exit genes, run wiring: ``--exit-genes extended`` from launch to notes.

Launch refusals (the user's rulings of 2026-11-23: ATR sizing only, 1x with liquidation off)
are checked at both public edges with the same texts: the API launch/preflight (422, no run
row) and the CLI run (exit code 2, run row failed). The default launch (flag absent or
``fixed``) stays byte-identical. An extended run reaches its genomes through the pipeline,
its notes and its champions' DSL; promote and replay run that DSL on real NautilusTrader.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_discovery_run_spec import (
    API_END,
    API_START,
    _api_client,
    _api_spec,
    _run_count,
)
from tests.unit.test_run_discovery import drive_run_discovery
from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run as discovery_run
from vibe_quant.discovery.run_spec import DiscoveryRunSpec, build_parser

if TYPE_CHECKING:
    from pathlib import Path

SIZING_TEXT = (
    "--exit-genes extended lets the stop pick the position size (risk sizing divides the "
    "risk budget by the stop distance: a 50% safety stop sizes below the BTC minimum and "
    "trades nothing). Use --sizing-mode atr"
)


def _leverage_text(leverage: str) -> str:
    return (
        "--exit-genes extended draws a 50% safety stop and ATR stops of any width; at "
        f"leverage {leverage} with liquidation off a position is liquidated before such a "
        "stop (isolated 2x: about 49.8%). Turn liquidation on (--liquidation-mode isolated "
        "or cross) or use --leverage 1."
    )


_GRID = {"modes": ["isolated"], "leverages": [5.0], "max_margin_pcts": [0.5]}
_CLI_GRID = [
    "--bankroll-modes", "isolated", "--bankroll-leverages", "5", "--bankroll-max-margin-pcts", "0.5",
]  # fmt: skip

# name -> (API body extras, CLI extras, refusal detail or None = launches)
_CASES: dict[str, tuple[dict[str, Any], list[str], str | None]] = {
    "atr-isolated": (
        {"sizing_mode": "atr", "liquidation_mode": "isolated"},
        ["--sizing-mode", "atr", "--liquidation-mode", "isolated"],
        None,
    ),
    "atr-off-leverage-1": (
        {"sizing_mode": "atr", "leverage": 1},
        ["--sizing-mode", "atr", "--leverage", "1"],
        None,
    ),
    "atr-cross-default-leverage": (
        {"sizing_mode": "atr", "liquidation_mode": "cross"},
        ["--sizing-mode", "atr", "--liquidation-mode", "cross"],
        None,
    ),
    "atr-grid": (
        {"sizing_mode": "atr", "bankroll_grid": _GRID},
        ["--sizing-mode", "atr", *_CLI_GRID],
        None,
    ),
    # both rules fail on a default launch: sizing text first, then leverage (R21f)
    "default-flags": ({}, [], f"{SIZING_TEXT}; {_leverage_text('10')}"),
    "atr-off-leverage-2": (
        {"sizing_mode": "atr", "leverage": 2},
        ["--sizing-mode", "atr", "--leverage", "2"],
        _leverage_text("2"),
    ),
    "default-sizing-off-leverage-1": ({"leverage": 1}, ["--leverage", "1"], SIZING_TEXT),
    # R21e: risk 0.25 against the 0.5 cap makes every stop up to 50% size the same, but an
    # ATR stop of 60% of price does not: only ATR sizing is stop-independent
    "risk-0.25-isolated": (
        {"risk_per_trade": 0.25, "liquidation_mode": "isolated"},
        ["--risk-per-trade", "0.25", "--liquidation-mode", "isolated"],
        SIZING_TEXT,
    ),
    "grid-default-sizing": ({"bankroll_grid": _GRID}, _CLI_GRID, SIZING_TEXT),
}
_IDS = list(_CASES)

_MOCK_ARGV = [
    "--mock", "--seed", "5", "--population-size", "6", "--max-generations", "2",
    "--elite-count", "1", "--max-workers", "-1", "--symbols", "BTCUSDT",
    "--timeframe", "4h", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
]  # fmt: skip


def _api_body(**extra: Any) -> dict[str, Any]:
    return {
        "symbols": ["BTCUSDT"], "timeframes": ["4h"], "start_date": API_START,
        "end_date": API_END, "exit_gene_space": "extended", **extra,
    }  # fmt: skip


@pytest.mark.parametrize("case", _IDS)
async def test_launch_refusal_leverage(tmp_path: Path, case: str) -> None:
    """R21/R21b/R21e/R21f: the API refuses (422, no run row) or launches; preflight agrees."""
    extra, _cli, detail = _CASES[case]
    body = _api_body(**extra)
    async with _api_client(tmp_path / "x.db") as (client, state):
        r = await client.post("/api/discovery/launch", json=body)
        if detail is not None:
            assert (r.status_code, r.json()["detail"]) == (422, detail)
            assert _run_count(state) == 0
            pre = await client.post("/api/discovery/preflight", json=body)  # same texts
            assert (pre.status_code, pre.json()["detail"]) == (422, detail)
            return
        assert r.status_code == 201, r.text
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert run["parameters"]["exit_gene_space"] == "extended"


@pytest.mark.parametrize("case", _IDS)
def test_cli_launch_refusal_exits_2_with_the_same_text(tmp_path: Path, case: str) -> None:
    """The CLI run: exit 2 and the run row failed with the API's text; the launches run."""
    _api, cli, detail = _CASES[case]
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2025-01-01", {}
    )
    state.close()
    argv = [*_MOCK_ARGV, "--exit-genes", "extended", *cli, "--db", str(db), "--run-id", str(run_id)]
    outcome = drive_run_discovery(db, run_id, argv, discovery_run.MockBacktests())
    state = StateManager(db)
    try:
        row = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert row is not None
    if detail is None:
        assert (outcome.exit_code, row["status"]) == (0, "completed"), outcome.message
        return
    assert outcome.exit_code == 2
    assert detail in outcome.message
    assert row["status"] == "failed"


@pytest.mark.parametrize(("space", "sent"), [("fixed", False), ("extended", True)])
def test_exit_gene_space_is_stored_and_forwarded_only_when_extended(space: str, sent: bool) -> None:
    """R1: a fixed launch keeps its params and argv byte-identical (no key); the forwarded argv
    parses back to the space the run reads."""
    spec = _api_spec(exit_gene_space=space)
    assert ("exit_gene_space" in spec.to_params()) is sent
    assert spec.to_argv().count("--exit-genes") == int(sent)
    assert build_parser().parse_args(["--run-id", "0", *spec.to_argv()]).exit_gene_space == space
    assert DiscoveryRunSpec.from_argv(spec.to_argv()).to_argv() == spec.to_argv()


def test_an_absent_flag_is_the_fixed_space() -> None:
    assert build_parser().parse_args(["--run-id", "1"]).exit_gene_space == "fixed"
    assert "--exit-genes" not in DiscoveryRunSpec.from_argv([]).to_argv()


@pytest.mark.simple_target(
    reason="launch-table row: a bogus --exit-genes value must exit 2, not fall back to fixed"
)
def test_bogus_exit_genes_value_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--run-id", "1", "--exit-genes", "bogus"])
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


async def test_bogus_exit_gene_space_is_a_422(tmp_path: Path) -> None:
    async with _api_client(tmp_path / "x.db") as (client, state):
        r = await client.post("/api/discovery/launch", json=_api_body(exit_gene_space="bogus"))
        assert r.status_code == 422
        assert "exit_gene_space" in json.dumps(r.json()["detail"])
        assert _run_count(state) == 0


# ---------------------------------------------------------------------------
# The pipeline: every genome of an extended run carries exit rules
# ---------------------------------------------------------------------------


def _extended_pipeline_run(**overrides: Any) -> tuple[list[Any], Any]:
    """A 2-generation extended mock run: every chromosome the backtest fn sees, and the result."""
    from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    seen: list[Any] = []

    def backtest(chrom: Any) -> Any:
        seen.append(chrom)
        return _mock_backtest(chrom)

    # entropy_threshold above any entropy: every generation injects immigrants (ops.random)
    cfg = _make_config(
        population_size=12, max_generations=2, elite_count=1, timeframe="4h",
        mutation_rate=0.5, immigrant_fraction=0.5, entropy_threshold=2.0,
        exit_gene_space="extended", **overrides,
    )  # fmt: skip
    import random

    random.seed(5)
    return seen, DiscoveryPipeline(cfg, backtest).run()


def test_extended_run_never_breeds_a_legacy_genome() -> None:
    """N4: immigrants (``ops.random``), mutation and crossover keep exit rules; a legacy genome
    in the mix raises ``crossover of mixed exit gene spaces`` and fails the run."""
    seen, result = _extended_pipeline_run()
    assert len(seen) > 12, "needs a second generation (immigrants, children)"
    assert all(chrom.exit_rules is not None for chrom in seen)
    assert result.top_strategies


def test_seeded_extended_population_draws_every_exit_shape() -> None:
    """R23 (population side): seed 5, pop 12 x 2 reaches a trailing stop, a safety stop and a
    time exit, and every genome renders a parser-valid DSL."""
    from vibe_quant.discovery.genome import chromosome_to_dsl
    from vibe_quant.dsl.schema import StrategyDSL

    seen, _ = _extended_pipeline_run()
    stops = {chrom.exit_rules.long.stop_type for chrom in seen}
    assert {"atr_trailing", "safety"} <= stops
    assert any(chrom.exit_rules.max_bars is not None for chrom in seen)
    for chrom in seen[:30]:
        StrategyDSL.model_validate(chromosome_to_dsl(chrom))


def test_top_strategies_log_prints_the_exit_rules(caplog: pytest.LogCaptureFixture) -> None:
    """N4: the pinned dead sl/tp (2.0 / 3.0) of an extended genome are not what the log shows."""
    import logging

    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.pipeline"):
        _extended_pipeline_run()
    top = [
        r.getMessage()
        for r in caplog.records
        if " uid=" in r.getMessage() and "dir=" in r.getMessage()
    ]
    assert top
    assert all("exits=long:" in line for line in top)
    assert not any("sl=2.0%" in line for line in top)


def test_breed_syncs_the_short_block_to_the_direction_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1: parents BOTH (long + short block) bred under a LONG constraint hand LONG children
    whose exit rules already fit the direction to the validity check (``sync_rules`` draws no
    RNG; before it, only the repair mutation fixed them)."""
    from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
    from vibe_quant.discovery.exit_genes import ExitBlock, ExitRules, rules_problems
    from vibe_quant.discovery.operators import Direction, _random_chromosome
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    block = ExitBlock("fixed_pct", 2.0, "fixed_pct", 4.0, None)
    parents = []
    for _ in range(2):
        chrom = _random_chromosome(direction_constraint=Direction.BOTH, exit_timeframe="4h")
        chrom.exit_rules = ExitRules(long=block, short=block, max_bars=None)
        parents.append(chrom)
    cfg = _make_config(
        population_size=12, timeframe="4h", mutation_rate=0.0, crossover_rate=0.0,
        exit_gene_space="extended", direction="long",
    )  # fmt: skip
    pipe = DiscoveryPipeline(cfg, _mock_backtest)
    pipe._direction_constraint = Direction.LONG
    checked: list[list[str]] = []
    real_is_valid = pipe._ops.is_valid

    def spy(chrom: Any) -> bool:
        checked.append(rules_problems(chrom.exit_rules, chrom.direction))
        return bool(real_is_valid(chrom))

    monkeypatch.setattr(pipe._ops, "is_valid", spy)
    children, _retries, _fallbacks = pipe._breed(parents[0], parents[1])
    assert checked[:2] == [[], []]  # the two children, as first handed to the validity check
    assert [c.direction for c in children] == [Direction.LONG, Direction.LONG]
    assert all(c.exit_rules is not None and c.exit_rules.short is None for c in children)


# ---------------------------------------------------------------------------
# A whole extended run through the CLI: notes, replay, seeds
# ---------------------------------------------------------------------------

_EXT = [
    "--exit-genes", "extended", "--sizing-mode", "atr", "--liquidation-mode", "isolated",
]  # fmt: skip
_OPEN_GATES = ["--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"]
_SEED_TEXT = "seed run used --exit-genes extended; pass --exit-genes extended to warm-start from it"


def _cli_run(db: Path, *extra: str, population: int = 12) -> tuple[int, discovery_run.RunOutcome]:
    """One mock ``run_discovery`` into ``db`` (pop ``population`` x 2, seed 5, 4h)."""
    state = StateManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2025-01-01", {}
    )
    state.close()
    argv = [
        "--mock", "--seed", "5", "--population-size", str(population), "--max-generations", "2",
        "--elite-count", "1", "--max-workers", "-1", "--symbols", "BTCUSDT", "--timeframe", "4h",
        "--start-date", "2024-01-01", "--end-date", "2025-01-01", *_OPEN_GATES, *extra,
        "--db", str(db), "--run-id", str(run_id),
    ]  # fmt: skip
    return run_id, drive_run_discovery(db, run_id, argv, discovery_run.MockBacktests())


def _notes(db: Path, run_id: int) -> dict[str, Any]:
    state = StateManager(db)
    try:
        result = state.get_backtest_result(run_id)
    finally:
        state.close()
    assert result is not None
    notes: dict[str, Any] = json.loads(result["notes"])
    return notes


def _without_timing(obj: Any) -> Any:
    from tests.unit.test_discovery_cli import _strip_timing

    return _strip_timing(obj)


def test_seeded_extended_run_replays(tmp_path: Path) -> None:
    """R23: the same seed twice gives the same notes; they say ``extended`` and each champion
    carries its exit rules in its chromosome and its DSL."""
    runs = []
    for tag in ("a", "b"):
        db = tmp_path / f"{tag}.db"
        run_id, outcome = _cli_run(db, *_EXT)
        assert outcome.exit_code == 0, outcome.message
        runs.append(_notes(db, run_id))
    assert json.dumps(_without_timing(runs[0]), sort_keys=True) == json.dumps(
        _without_timing(runs[1]), sort_keys=True
    )
    notes = runs[0]
    assert notes["exit_gene_space"] == "extended"
    assert notes["top_strategies"], "toothless compare: no champions"
    for entry in notes["top_strategies"]:
        assert entry["chromosome"]["exit_rules"]["long"]["stop_type"] in (
            "fixed_pct", "atr_fixed", "atr_trailing", "safety"
        )  # fmt: skip
        assert "stop_loss" in entry["dsl"]


def test_a_fixed_run_never_writes_the_extended_note(tmp_path: Path) -> None:
    db = tmp_path / "fixed.db"
    run_id, outcome = _cli_run(db, "--exit-genes", "fixed")
    assert outcome.exit_code == 0, outcome.message
    notes = _notes(db, run_id)
    assert "exit_gene_space" not in notes
    assert all("exit_rules" not in entry["chromosome"] for entry in notes["top_strategies"])


def test_extended_seed_into_a_fixed_run_is_refused_and_the_rest_launch(tmp_path: Path) -> None:
    """R22b (CLI): an extended champion cannot warm-start a fixed run (exit 2, the loader's
    text); extended -> extended and a fixed seed -> extended launch."""
    db = tmp_path / "seed.db"
    ext_id, outcome = _cli_run(db, *_EXT)
    assert outcome.exit_code == 0, outcome.message
    fixed_id, outcome = _cli_run(db, "--exit-genes", "fixed")
    assert outcome.exit_code == 0, outcome.message

    _id, outcome = _cli_run(db, "--seed-from-run", str(ext_id))
    assert outcome.exit_code == 2
    assert _SEED_TEXT in outcome.message
    _id, outcome = _cli_run(db, *_EXT, "--seed-from-run", str(ext_id))
    assert outcome.exit_code == 0, outcome.message
    _id, outcome = _cli_run(db, *_EXT, "--seed-from-run", str(fixed_id))
    assert outcome.exit_code == 0, outcome.message


async def test_extended_seed_into_a_fixed_launch_is_a_422(tmp_path: Path) -> None:
    """R22b (API): the launch refuses with the loader's text before any run row; an extended
    launch from the same seed is accepted."""
    db = tmp_path / "seed.db"
    ext_id, outcome = _cli_run(db, *_EXT)
    assert outcome.exit_code == 0, outcome.message
    async with _api_client(db) as (client, state):
        before = _run_count(state)
        fixed = {"symbols": ["BTCUSDT"], "timeframes": ["4h"], "start_date": API_START,
                 "end_date": API_END, "seed_run_id": ext_id}  # fmt: skip
        r = await client.post("/api/discovery/launch", json=fixed)
        assert (r.status_code, r.json()["detail"]) == (422, _SEED_TEXT)
        assert _run_count(state) == before
        r = await client.post(
            "/api/discovery/launch",
            json=_api_body(seed_run_id=ext_id, sizing_mode="atr", liquidation_mode="isolated"),
        )
        assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# Promote and replay an extended champion on real NautilusTrader
# ---------------------------------------------------------------------------


def _discovery_row(state: StateManager, entries: list[dict[str, Any]], **run: Any) -> int:
    """A finished discovery run whose notes hold ``entries`` as its champions."""
    from vibe_quant.dsl.compiler import compiler_version_hash

    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=run["symbols"], timeframe=run["timeframe"],
        start_date=run["start"], end_date=run["end"], parameters={},
    )  # fmt: skip
    state.update_backtest_run_status(run_id, "completed")
    state.save_backtest_result(run_id, {"total_trades": 0}, trades=[])
    notes = {"type": "discovery", "top_strategies": entries,
             "compiler_version": compiler_version_hash(), **run.get("notes", {})}  # fmt: skip
    state.update_result_notes(run_id, json.dumps(notes))
    return run_id


def _extended_chrom() -> Any:
    """A real extended genome (RSI entry/exit genes, trailing stop + risk-reward target, 12-bar
    time exit); the entry rule is plain RSI so a three-month 4h BTC window trades."""
    from vibe_quant.discovery.exit_genes import ExitBlock, ExitRules, canonical_block
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    block = canonical_block(ExitBlock("atr_trailing", 2.5, "risk_reward", 2.0, 14))
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 45.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 60.0, None)],
        stop_loss_pct=2.0, take_profit_pct=3.0, direction=Direction.LONG,
        exit_rules=ExitRules(long=block, short=None, max_bars=12),
    )  # fmt: skip


_REAL = pytest.mark.skipif(
    not (DEFAULT_CATALOG_PATH / "data" / "bar").exists(), reason="no data catalog"
)


@_REAL
@pytest.mark.slow(
    reason="real NT discovery evaluation, then a replay child, over 3 months of BTC 4h"
)
async def test_replay_extended_champion_exact(tmp_path: Path) -> None:
    """R24: replay of an extended champion runs its stored DSL and writes one sweep row equal to
    the discovery full-range headline (Sharpe, trades, return, drawdown, profit factor)."""
    from tests.unit.test_pooled_replay import _client, _row, _run_child
    from vibe_quant.discovery.backtest_fn import NTBacktestFn, full_range_headline
    from vibe_quant.discovery.genome import chromosome_to_dsl
    from vibe_quant.sizing_settings import SizingSettings

    start, end = "2024-01-01", "2024-04-01"
    sizing = SizingSettings(sizing_mode="atr")
    chrom = _extended_chrom()
    bt = NTBacktestFn(["BTCUSDT"], "4h", start, end, sizing=sizing)(chrom)
    assert "error" not in bt
    head = full_range_headline(
        bt, fallback_sharpe=0.0, fallback_trades=0, fallback_max_dd=0.0, fallback_pf=0.0,
        fallback_return=0.0,
    )  # fmt: skip
    assert head["full_range_trades"] > 0, "toothless replay: the champion never trades"
    entry: dict[str, Any] = {"dsl": {**chromosome_to_dsl(chrom), "timeframe": "4h"}, **head}
    assert entry["dsl"]["stop_loss"]["type"] == "atr_trailing"
    assert entry["dsl"]["max_bars_in_trade"] == 12
    db = tmp_path / "state.db"
    async with _client(db) as (client, state):
        did = _discovery_row(
            state, [entry], symbols=["BTCUSDT"], timeframe="4h", start=start, end=end,
            notes={"sizing": sizing.to_params()},
        )  # fmt: skip
        r = await client.post(f"/api/discovery/results/{did}/replay/0")
        assert r.status_code == 201, r.text
        replay_id = r.json()["replay_run_id"]
    assert _run_child(db, replay_id) == 0
    rows, run = _row(db, replay_id)
    assert run["status"] == "completed"
    (row,) = rows
    assert row["sharpe_ratio"] == head["full_range_sharpe"]
    assert row["total_trades"] == head["full_range_trades"]
    assert row["total_return"] == head["full_range_return_pct"]
    assert row["max_drawdown"] == head["full_range_max_dd"]
    assert row["profit_factor"] == head["full_range_pf"]


@pytest.mark.slow(reason="screening and validation NT engine runs on a toy catalog")
async def test_promoted_trailing_champion_validates_without_a_collapse(tmp_path: Path) -> None:
    """R26: promote an extended champion through the API -> a strategy whose DSL keeps its
    ``atr_trailing`` stop and ATR multiple; its validation run reaches ``completed`` with a trade
    closed by a STOP_MARKET whose trigger had moved, and the consistency check (against the
    champion's own screening numbers) raises no ``validation-collapse`` flag."""
    from tests.unit.test_exit_genes_tier_parity import _block, _ga_dsl, _path, _read_tap
    from tests.unit.test_exit_genes_tier_parity import _write_path_catalog as write_toy
    from tests.unit.test_max_bars_exit_parity import _fills_from_engine
    from tests.unit.test_screening_fill_mode import _END, _START
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.validation.consistency import assess_consistency, lookup_screening_reference
    from vibe_quant.validation.runner import ValidationRunner

    toy = write_toy(tmp_path / "toy", _path, ("BTCUSDT",))
    dsl = _ga_dsl("genome_trail", _block("atr_trailing", 2.0, "fixed_pct", 90.0, 3))
    screening = NTScreeningRunner(dsl, ["BTCUSDT"], _START, _END, catalog_path=str(toy))({})
    assert screening.sharpe_ratio > 0 and screening.total_trades > 0, "toothless: no reference edge"
    head = {"sharpe": screening.sharpe_ratio, "trades": screening.total_trades}

    db = tmp_path / "state.db"
    async with _api_client(db) as (client, state):
        did = _discovery_row(
            state, [{"dsl": dsl, **head}], symbols=["BTCUSDT"], timeframe="4h",
            start=_START, end=_END,
        )  # fmt: skip
        r = await client.post(f"/api/discovery/results/{did}/promote/0?mode=validation")
        assert r.status_code == 201, r.text
        promoted = r.json()
    stored = StateManager(db)
    try:
        strategy = stored.get_strategy(promoted["strategy_id"])
        assert strategy is not None
        stop = strategy["dsl_config"]["stop_loss"]
        assert (stop["type"], stop["atr_multiplier"]) == ("atr_trailing", 2.0)

        seen: list[Any] = []
        orig = ValidationRunner._extract_results

        def spy(self: ValidationRunner, **kwargs: Any) -> Any:
            seen.append((_fills_from_engine(kwargs["engine"]), _read_tap(kwargs["engine"])))
            return orig(self, **kwargs)

        runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=toy)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ValidationRunner, "_extract_results", spy)
            try:
                result = runner.run(run_id=promoted["run_id"])
            finally:
                runner.close()
        run_row = stored.get_backtest_run(promoted["run_id"])
        assert run_row is not None and run_row["status"] == "completed", run_row
        fills, tap = seen[0]
        stopped = [p for p in fills.positions if p.closing == "STOP_MARKET"]
        assert stopped, "no trade closed by a stop"
        first = stopped[0]
        assert first.px_close is not None and first.px_close > first.px_open + 500, (
            "the stop closed at a loss: the trigger never moved up with price"
        )
        triggers = [lv.level for lv in tap.protective if lv.type == "STOP_MARKET"]
        assert len(set(triggers)) > 1

        notes = json.loads(str(stored.get_backtest_result(promoted["run_id"])["notes"]))  # type: ignore[index]
        flags = notes.get("consistency", {}).get("flags", [])
        assert not any(flag.startswith("validation-collapse") for flag in flags)
        reference, refused = lookup_screening_reference(
            stored, promoted["strategy_id"], promoted["name"], val_window=(_START, _END),
            val_symbols=["BTCUSDT"], val_sizing={},
        )  # fmt: skip
        assert reference is not None and not refused
        assert reference.source == f"discovery_run:{did}"
        report = assess_consistency(
            reference, result.sharpe_ratio, result.total_trades, (_START, _END)
        )
        assert not [f for f in report.flags if f.startswith("validation-collapse")]
        assert result.sharpe_ratio >= 0.5 * reference.sharpe  # the collapse bar itself
    finally:
        stored.close()
