"""Characterization goldens for the discovery notes readers (bead vibe-quant-rabmo.8, T2).

Every ``tests/fixtures/discovery_notes/era_*.json`` is a discovery run's notes text, verbatim
from the real state DB (one per notes era), or a synthetic champion era built from the
current writers (``dump_synthetic_fixtures``). For each one the readers' outputs (API
responses, created run parameters, loader results) are compared with
``responses_golden.json``, captured ONCE on the pre-refactor base:

    PYTHONPATH=$PWD .venv/bin/python -c \
        "from tests.unit.test_discovery_record_golden import dump_golden; dump_golden()"

Characterization only: when a refactor changes a golden, the refactor is wrong (or declares a
comparability break). The adapter block below is the only place that names the readers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers.results import _enrich_result_with_notes
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.__main__ import _prior_discovery_trials
from vibe_quant.discovery.campaign import extract_top_chromosomes
from vibe_quant.discovery.genome import chromosome_to_serializable
from vibe_quant.discovery.seed import load_seed_chromosomes
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.screening.replay_drift import _load_discovery_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# --- adapter block: the only names a reader move (rabmo.8 T10/T11) flips -------------------

ROUTER_REGIME_SIGN = "vibe_quant.api.routers.discovery._window_regime_sign"


def adapter_prior_trials(
    state: StateManager, run_id: int, timeframe: str, window: tuple[str, str]
) -> tuple[int | None, list[int], int, int]:
    return _prior_discovery_trials(state, run_id, timeframe, window)


# ------------------------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "discovery_notes"
GOLDEN_PATH = FIXTURE_DIR / "responses_golden.json"

REAL_ERA_RUN_IDS = (1, 42, 799, 840, 877, 913, 919, 950, 956, 981)
SYNTHETIC_ERA_RUN_IDS = (9950, 9981)  # synthetic copies of 950 / 981 plus champions

_BASE_ROW_KEYS = (
    "cross_window_results",
    "wfa_sharpe_consistency",
    "bootstrap_sharpe_lower",
    "bootstrap_sharpe_upper",
    "bootstrap_ci_level",
    "cross_regime_results",
    "random_short_baseline_pct",
)


def load_eras() -> list[dict[str, Any]]:
    """Every fixture era, ordered by the run id it gets in a test DB."""
    eras = [json.loads(p.read_text()) for p in sorted(FIXTURE_DIR.glob("era_*.json"))]
    for era in eras:
        era.setdefault("run_id", era["source_run_id"])
    return sorted(eras, key=lambda e: int(e["run_id"]))


def notes_object(era: dict[str, Any]) -> dict[str, Any] | None:
    try:
        data = json.loads(era["notes"])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def champion_count(era: dict[str, Any]) -> int:
    notes = notes_object(era)
    top = notes.get("top_strategies") if notes else None
    return len(top) if isinstance(top, list) else 0


def insert_discovery_run(
    state: StateManager,
    run_id: int,
    *,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    parameters: dict[str, object],
    notes: str | None,
    job: bool = True,
) -> None:
    """A completed discovery run with an explicit id, its result row and a finished job row."""
    state.conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe, start_date, "
        "end_date, parameters, status) VALUES (?, NULL, 'discovery', ?, ?, ?, ?, ?, 'completed')",
        (
            run_id,
            json.dumps(symbols),
            timeframe,
            start_date,
            end_date,
            json.dumps(parameters),
        ),
    )
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, notes)
    )
    if job:
        state.conn.execute(
            "INSERT INTO background_jobs (run_id, pid, job_type, status, started_at, "
            "completed_at) VALUES (?, 1, 'discovery', 'completed', '2026-01-01 00:00:00', "
            "'2026-01-01 01:00:00')",
            (run_id,),
        )
    state.conn.commit()


def seed_eras(state: StateManager, eras: list[dict[str, Any]]) -> None:
    for era in eras:
        run = era["run"]
        insert_discovery_run(
            state,
            int(era["run_id"]),
            symbols=run["symbols"],
            timeframe=run["timeframe"],
            start_date=run["start_date"],
            end_date=run["end_date"],
            parameters=run["parameters"],
            notes=era["notes"],
        )


def _regime_sign_stub(base_window: tuple[str, str] | None) -> Any:
    """Deterministic stand-in for the catalog-reading regime classifier.

    The base (train) window is a bear market, every other window a bull market; with no base
    window everything is neutral. The real classifier reads local market data.
    """

    def sign(symbol: str, start_date: str, end_date: str) -> int:
        if base_window is None:
            return 0
        return -1 if (start_date, end_date) == base_window else 1

    return sign


@asynccontextmanager
async def api_client(
    db_path: Path, base_window: tuple[str, str] | None = None
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager]]:
    """The real app on a temp DB; job launches return pid 4242 (no subprocess)."""
    app = create_app()
    state = StateManager(db_path=db_path)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db_path)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        with (
            patch.object(jobs, "start_job", return_value=4242),
            patch(ROUTER_REGIME_SIGN, side_effect=_regime_sign_stub(base_window)),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac, state, jobs
    finally:
        await ws.stop()
        jobs.close()
        state.close()


def digest(value: object) -> dict[str, object]:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    size = len(value) if isinstance(value, (list, dict, str)) else None
    return {"len": size, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def body_of(response: Any) -> object:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


def row_counts(state: StateManager) -> list[int]:
    return [
        int(state.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608
        for table in ("strategies", "backtest_runs")
    ]


def created_parameters(state: StateManager, run_id: int) -> object:
    run = state.get_backtest_run(run_id)
    assert run is not None
    params = run["parameters"]
    if isinstance(params, dict) and "dsl_override" in params:
        # the translated DSL is the translator's output, not a notes reader's
        params = {**params, "dsl_override": "<translated dsl>"}
    return params


async def post_recorded(ac: AsyncClient, state: StateManager, url: str) -> dict[str, object]:
    before = row_counts(state)
    response = await ac.post(url)
    body = body_of(response)
    record: dict[str, object] = {
        "status": response.status_code,
        "body": body,
        "rows_added": [a - b for a, b in zip(row_counts(state), before, strict=True)],
    }
    if response.status_code == 201 and isinstance(body, dict):
        created = body.get("run_id", body.get("replay_run_id"))
        if isinstance(created, int):
            record["parameters"] = created_parameters(state, created)
    return record


def raised_or(fn: Any) -> object:
    try:
        return {"value": fn()}
    except Exception as exc:
        return {"raises": type(exc).__name__}


def serialized_chromosomes(state: StateManager, run_id: int) -> object:
    def load() -> object:
        loaded = load_seed_chromosomes(state, run_id)
        if loaded is None:
            return None
        return digest([chromosome_to_serializable(c) for c in loaded])

    return raised_or(load)


def metrics_row(state: StateManager, run_id: int, index: int) -> object:
    return raised_or(lambda: _load_discovery_metrics(state, run_id, index))


def enrich_row(era: dict[str, Any]) -> dict[str, object]:
    row: dict[str, object] = {"id": 1, "run_id": era["run_id"], "notes": era["notes"]}
    row.update({key: None for key in _BASE_ROW_KEYS})
    enriched = _enrich_result_with_notes(row)
    enriched.pop("notes")
    return enriched


async def record_era(era: dict[str, Any], tmp_dir: Path) -> dict[str, object]:
    rid = int(era["run_id"])
    n = champion_count(era)
    notes = notes_object(era)
    train = notes.get("train_dates") if notes else None
    base_window = (str(train[0]), str(train[1])) if isinstance(train, list) and len(train) == 2 else None
    out: dict[str, object] = {}
    async with api_client(tmp_dir / f"era_{rid}.db", base_window) as (ac, state, _jobs):
        seed_eras(state, [era])

        response = await ac.get(f"/api/discovery/results/{rid}")
        body = body_of(response)
        assert isinstance(body, dict)
        out["results"] = {
            "status": response.status_code,
            "strategies": digest(body.get("strategies")),
            "reason": body.get("reason"),
            "guardrail_rejections": digest(body.get("guardrail_rejections"))
            if body.get("guardrail_rejections") is not None
            else None,
        }

        response = await ac.get("/api/discovery/jobs")
        entries = [e for e in response.json() if e["run_id"] == rid]
        out["jobs_entry"] = entries[0] if entries else None

        indices = sorted({0, n} | set(range(n)))
        out["export"] = {
            f"{label}:{idx}": await post_recorded(
                ac, state, f"/api/discovery/results/{rid}/export/{idx}"
            )
            for label, idx in [("first", 0), ("first", -1), ("first", n), ("again", 0)]
        }
        out["promote"] = {
            f"{mode}/{vrange}:{idx}": await post_recorded(
                ac,
                state,
                f"/api/discovery/results/{rid}/promote/{idx}"
                f"?mode={mode}&validation_range={vrange}",
            )
            for idx in indices
            for mode, vrange in (
                ("screening", "holdout"),
                ("validation", "holdout"),
                ("validation", "full"),
            )
        }
        out["replay"] = {
            str(idx): await post_recorded(ac, state, f"/api/discovery/results/{rid}/replay/{idx}")
            for idx in sorted({0, n})
        }

        out["load_seed_chromosomes"] = serialized_chromosomes(state, rid)
        out["extract_top_chromosomes"] = raised_or(
            lambda: digest(extract_top_chromosomes(rid, state, 3))
        )
        out["load_discovery_metrics"] = {
            str(idx): metrics_row(state, rid, idx) for idx in range(-1, n + 1)
        }
    out["enrich"] = enrich_row(era)
    return out


async def record_set(eras: list[dict[str, Any]], tmp_dir: Path) -> dict[str, object]:
    """Readers that look across runs: /results/latest and the prior-trial scan."""
    out: dict[str, object] = {}
    async with api_client(tmp_dir / "set.db") as (ac, state, _jobs):
        seed_eras(state, eras)
        response = await ac.get("/api/discovery/results/latest")
        body = body_of(response)
        assert isinstance(body, dict)
        out["latest"] = {
            "status": response.status_code,
            "strategies": digest(body.get("strategies")),
        }
        priors: dict[str, object] = {}
        for era in eras:
            run = era["run"]
            notes = notes_object(era)
            train = notes.get("train_dates") if notes else None
            window = (
                (str(train[0]), str(train[1]))
                if isinstance(train, list) and len(train) == 2
                else (run["start_date"], run["end_date"])
            )
            priors[str(era["run_id"])] = list(
                adapter_prior_trials(state, int(era["run_id"]), run["timeframe"], window)
            )
        out["prior_trials"] = priors
        out["prior_trials_wide_window"] = {
            timeframe: list(
                adapter_prior_trials(state, 100000, timeframe, ("2000-01-01", "2100-01-01"))
            )
            for timeframe in ("1m", "1h", "4h", "1d")
        }
    return out


async def compute_golden(tmp_dir: Path) -> dict[str, object]:
    eras = load_eras()
    golden: dict[str, object] = {"set": await record_set(eras, tmp_dir)}
    for era in eras:
        golden[f"era_{era['run_id']}"] = await record_era(era, tmp_dir)
    golden["replay_note_rows"] = await replay_note_rows(tmp_dir)
    return golden


def dump_golden() -> None:
    """Write ``responses_golden.json`` (run ONCE on the base; see the module docstring)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["VIBE_QUANT_DB"] = str(Path(tmp) / "pin.db")
        os.environ["VIBE_QUANT_LOG_DIR"] = str(Path(tmp) / "logs")  # no real progress files
        golden = asyncio.run(compute_golden(Path(tmp)))
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")


def dump_synthetic_fixtures() -> None:
    """Write the two synthetic champion eras from the CURRENT champion writers (run once).

    No champion exists in the real DB for recent notes eras, so: run 950's notes (bankroll
    grid) plus one champion carrying ``BankrollCell.to_note()``, and run 981's notes plus four
    champions from ``_champion_entry`` -- one long with a current ``cross_window`` (shifted
    windows with ``dates``), two shorts (current and LEGACY ``cross_window``: index 0 is the
    in-sample window, no dates) and one short without any.
    """
    from vibe_quant.bankroll_grid import BankrollCell
    from vibe_quant.discovery.__main__ import _champion_entry, cross_window_entry
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )
    from vibe_quant.discovery.pipeline import CrossWindowResult, HoldoutResult

    def chromosome(direction: Direction, uid: str) -> StrategyChromosome:
        return StrategyChromosome(
            entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
            exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=3.0,
            direction=direction,
            uid=uid,
        )

    def fitness(sharpe: float, trades: int) -> FitnessResult:
        return FitnessResult(
            sharpe_ratio=sharpe,
            max_drawdown=0.05,
            profit_factor=1.5,
            total_trades=trades,
            total_return=0.12,
            complexity_penalty=0.0,
            overtrade_penalty=0.0,
            sl_tp_penalty=0.0,
            raw_score=sharpe,
            adjusted_score=sharpe,
            passed_filters=True,
            filter_results={},
        )

    def champion(direction: Direction, uid: str, sharpe: float, trades: int) -> dict[str, object]:
        return _champion_entry(
            chromosome(direction, uid),
            fitness(sharpe, trades),
            timeframe="1m",
            dsr=None,
            buy_hold=None,
        )

    def window(sharpe: float, ret: float) -> HoldoutResult:
        return HoldoutResult(
            sharpe_ratio=sharpe,
            max_drawdown=0.04,
            profit_factor=1.3,
            total_trades=25,
            total_return=ret,
        )

    def real(run_id: int) -> dict[str, Any]:
        path = next(FIXTURE_DIR.glob(f"era_{run_id}_*.json"))
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    # 950 + one bankroll-grid champion
    base = real(950)
    notes = json.loads(base["notes"])
    entry = champion(Direction.LONG, "bg0000000001", 1.4, 80)
    entry["bankroll"] = BankrollCell("isolated", 10.0, 0.5).to_note()
    for key in ("outcome", "reason", "best_raw_score"):
        notes.pop(key, None)
    notes["top_strategies"] = [entry]
    out = {
        **base,
        "run_id": 9950,
        "synthetic": "run 950 notes + one champion carrying BankrollCell.to_note()",
        "notes": json.dumps(notes),
    }
    (FIXTURE_DIR / "era_9950_synthetic_bankroll_champion.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )

    # 981 + four champions (cross_window current / legacy / absent)
    base = real(981)
    notes = json.loads(base["notes"])
    current = CrossWindowResult(
        window_results=[window(1.2, 0.04), window(0.2, -0.01)],
        windows_passed=1,
        total_windows=2,
        passed=True,
        window_dates=[("2026-04-01", "2026-10-03"), ("2026-05-01", "2026-11-03")],
        offsets_months=[1, 2],
        required=1,
    )
    long_champ = champion(Direction.LONG, "cw0000000001", 1.6, 90)
    long_champ["cross_window"] = cross_window_entry(current)
    short_current = champion(Direction.SHORT, "cw0000000002", 1.3, 70)
    short_current["cross_window"] = cross_window_entry(current)
    short_legacy = champion(Direction.SHORT, "cw0000000003", 1.1, 60)
    short_legacy["cross_window"] = {
        "windows_passed": 1,
        "total_windows": 3,
        "required": 1,
        "passed": True,
        "windows": [
            {"sharpe": 1.1, "max_dd": 0.03, "pf": 1.4, "trades": 60, "return_pct": 0.1},
            {"sharpe": 0.9, "max_dd": 0.04, "pf": 1.3, "trades": 22, "return_pct": 0.03},
            {"sharpe": 0.1, "max_dd": 0.05, "pf": 1.0, "trades": 20, "return_pct": -0.02},
        ],
    }
    short_bare = champion(Direction.SHORT, "cw0000000004", 1.0, 50)
    for key in ("outcome", "reason", "best_raw_score"):
        notes.pop(key, None)
    notes["cross_window_months"] = [1, 2]
    notes["cross_window_min_sharpe"] = 0.5
    notes["top_strategies"] = [long_champ, short_current, short_legacy, short_bare]
    out = {
        **base,
        "run_id": 9981,
        "synthetic": (
            "run 981 notes + four champions from _champion_entry: long with current "
            "cross_window, short with current, short with LEGACY (index 0 in-sample, no dates), "
            "short without"
        ),
        "notes": json.dumps(notes),
    }
    (FIXTURE_DIR / "era_9981_synthetic_cross_window_champions.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def tmp_module_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("discovery_record_golden")


def _era_ids() -> list[str]:
    return [f"era_{e['run_id']}" for e in load_eras()]


def _json_round_trip(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


_computed: dict[str, object] = {}


async def _computed_golden(tmp_dir: Path) -> dict[str, object]:
    if not _computed:
        _computed.update(await compute_golden(tmp_dir))
    return _computed


def test_every_listed_era_present() -> None:
    eras = {int(e["run_id"]): e for e in load_eras()}
    assert set(eras) == set(REAL_ERA_RUN_IDS) | set(SYNTHETIC_ERA_RUN_IDS)
    # the era each real run stands for
    assert notes_object(eras[1]) is None  # non-JSON text
    assert eras[1]["notes"].startswith("discovery: generations=5")
    assert notes_object(eras[42])["compiler_version"]  # type: ignore[index]
    assert notes_object(eras[42])["top_strategies"]  # type: ignore[index]
    assert notes_object(eras[799])["holdout_dates"]  # type: ignore[index]
    assert "wfa" in notes_object(eras[799])["top_strategies"][0]  # type: ignore[index]
    assert "full_range_sharpe" in notes_object(eras[840])["top_strategies"][0]  # type: ignore[index]
    assert isinstance(notes_object(eras[877])["seed"], int)  # type: ignore[index]
    assert champion_count(eras[877]) == 0
    assert (
        notes_object(eras[913])["dsr_trials"]["n_rule"]  # type: ignore[index]
        == "cumulative_same_tf_overlapping_train"
    )
    assert "fee_schedule" in notes_object(eras[919])  # type: ignore[operator]
    assert "buy_hold_gate" in notes_object(eras[919])  # type: ignore[operator]
    assert "bankroll_grid" in notes_object(eras[950])  # type: ignore[operator]
    assert champion_count(eras[950]) == 0
    assert (
        notes_object(eras[956])["dsr_trials"]["n_rule"]  # type: ignore[index]
        == "cumulative_effective_same_tf_overlapping_train"
    )
    for key in ("liquidation", "sizing", "indicator_pool", "evaluator", "ind_memo"):
        assert key in notes_object(eras[981])  # type: ignore[operator]
    # synthetic eras
    assert notes_object(eras[9950])["top_strategies"][0]["bankroll"] == {  # type: ignore[index]
        "liquidation_mode": "isolated",
        "leverage": 10.0,
        "max_margin_pct": 0.5,
    }
    windows = [
        c["cross_window"]["windows"] for c in notes_object(eras[9981])["top_strategies"]  # type: ignore[index]
        if "cross_window" in c
    ]
    assert all("dates" in w for w in windows[0])  # current writer
    assert all("dates" not in w for w in windows[-1])  # legacy: index 0 = in-sample


async def test_promote_responses_match_golden(tmp_module_dir: Path) -> None:
    computed = await _computed_golden(tmp_module_dir)
    golden = _golden()
    for key in _era_ids():
        assert _json_round_trip(computed[key]["promote"]) == golden[key]["promote"], key  # type: ignore[index]


async def test_replay_responses_match_golden(tmp_module_dir: Path) -> None:
    computed = await _computed_golden(tmp_module_dir)
    golden = _golden()
    for key in _era_ids():
        assert _json_round_trip(computed[key]["replay"]) == golden[key]["replay"], key  # type: ignore[index]


async def test_export_responses(tmp_module_dir: Path) -> None:
    """R38c: index 0 created then exists, -1 and out-of-range 404."""
    computed = await _computed_golden(tmp_module_dir)
    golden = _golden()
    for key in _era_ids():
        assert _json_round_trip(computed[key]["export"]) == golden[key]["export"], key  # type: ignore[index]
    one = golden["era_840"]["export"]
    assert one["first:0"]["status"] == 201 and one["first:0"]["body"]["status"] == "created"
    assert one["again:0"]["body"]["status"] == "exists"
    assert one["first:-1"]["status"] == 404
    assert one["first:4"]["status"] == 404


async def test_results_and_jobs_entries_match_golden(tmp_module_dir: Path) -> None:
    computed = await _computed_golden(tmp_module_dir)
    golden = _golden()
    for key in _era_ids():
        for section in ("results", "jobs_entry"):
            assert _json_round_trip(computed[key][section]) == golden[key][section], (key, section)  # type: ignore[index]


async def test_loader_outputs_match_golden(tmp_module_dir: Path) -> None:
    computed = await _computed_golden(tmp_module_dir)
    golden = _golden()
    for key in _era_ids():
        for section in ("load_seed_chromosomes", "extract_top_chromosomes", "load_discovery_metrics", "enrich"):
            assert _json_round_trip(computed[key][section]) == golden[key][section], (key, section)  # type: ignore[index]


async def test_cross_run_readers_match_golden(tmp_module_dir: Path) -> None:
    """/results/latest over the fixture set and the prior-trial scan (R30, R36)."""
    computed = await _computed_golden(tmp_module_dir)
    golden = _golden()
    assert _json_round_trip(computed["set"]) == golden["set"]


async def test_replay_note_rows_match_golden(tmp_path: Path) -> None:
    """R27: the replay metrics_note text for each symbol_agg source and symbol count."""
    golden = _golden()["replay_note_rows"]
    assert _json_round_trip(await replay_note_rows(tmp_path)) == golden


async def replay_note_rows(tmp_dir: Path) -> dict[str, object]:
    era = next(e for e in load_eras() if e["run_id"] == 840)
    dsl = notes_object(era)["top_strategies"][0]["dsl"]  # type: ignore[index]
    rows: dict[str, object] = {}
    cases: list[tuple[str, dict[str, object], str | None]] = [
        ("params_worst", {"symbol_agg": "worst"}, None),
        ("notes_worst", {}, "worst"),
        ("neither", {}, None),
        ("params_portfolio_notes_worst", {"symbol_agg": "portfolio"}, "worst"),
        ("worst_with_eval_windows", {"symbol_agg": "worst", "eval_windows": 3}, None),
        ("eval_windows_only", {"eval_windows": 3}, None),
    ]
    for label, params, notes_agg in cases:
        for symbols in (["BTCUSDT"], ["BTCUSDT", "ETHUSDT"]):
            async with api_client(tmp_dir / f"note_{label}_{len(symbols)}.db") as (ac, state, _j):
                payload: dict[str, object] = {"top_strategies": [{"dsl": dsl}]}
                if notes_agg is not None:
                    payload["symbol_agg"] = notes_agg
                insert_discovery_run(
                    state,
                    7,
                    symbols=symbols,
                    timeframe="4h",
                    start_date="2024-01-01",
                    end_date="2025-01-01",
                    parameters=params,
                    notes=json.dumps(payload),
                )
                response = await ac.post("/api/discovery/results/7/replay/0")
                body = body_of(response)
                assert isinstance(body, dict)
                rows[f"{label}/{len(symbols)}"] = {
                    "status": response.status_code,
                    "metrics_note": body.get("metrics_note"),
                }
    return rows
