"""ADAPTIVE_RSI alpha floor: refused below 0.7 at strategy save, GA draws 0.7-0.9.

User ruling on bead vibe-quant-b9uet: ADAPTIVE_RSI(14, alpha 0.3) needs ~154,000 bars
to settle, so its value depends on how much history exists.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.genome import INDICATOR_POOL, chromosome_to_serializable
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.seed import load_seed_chromosomes
from vibe_quant.dsl.parser import DSLValidationError, validate_strategy_dict
from vibe_quant.dsl.plugins.example_adaptive_rsi import compute_adaptive_rsi
from vibe_quant.dsl.translator import translate_dsl_config

if TYPE_CHECKING:
    from pathlib import Path

R1 = "ADAPTIVE_RSI alpha must be between 0.7 and 1.0"


def _dsl(**arsi: Any) -> dict[str, Any]:
    return {
        "name": "arsi_floor",
        "timeframe": "1h",
        "indicators": {"arsi": {"type": "ADAPTIVE_RSI", "period": 14, **arsi}},
        "entry_conditions": {"long": ["arsi < 30"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def _refusal(dsl: dict[str, Any]) -> str:
    with pytest.raises(DSLValidationError) as exc:
        validate_strategy_dict(dsl)
    return " ".join([str(exc.value), *exc.value.details])


@pytest.mark.parametrize(("alpha", "stored"), [(0.7, 0.7), (0.85, 0.85), (1.0, 1.0), (1, 1.0)])
def test_alpha_in_range_is_accepted(alpha: float, stored: float) -> None:
    dsl = validate_strategy_dict(_dsl(alpha=alpha))
    value = dsl.indicators["arsi"].model_extra["alpha"]  # type: ignore[index]
    assert type(value) is float
    assert value == stored


@pytest.mark.parametrize(
    ("alpha", "shown"),
    [
        (0.69999, "0.69999"),
        (0.3, "0.3"),
        (0.0, "0.0"),
        (-0.5, "-0.5"),
        (1.0001, "1.0001"),
        (0, "0.0"),
    ],
)
def test_alpha_out_of_range_is_refused(alpha: float, shown: str) -> None:
    text = _refusal(_dsl(alpha=alpha))
    assert f"{R1} (got {shown})" in text
    assert "154,000" in text


@pytest.mark.parametrize("alpha", [True, False, "0.8"])
def test_non_number_alpha_keeps_type_refusal(alpha: object) -> None:
    text = _refusal(_dsl(alpha=alpha))
    assert "param 'alpha' expected float" in text
    assert R1 not in text


@pytest.mark.parametrize("alpha", [math.inf, math.nan])
def test_non_finite_alpha_is_refused_naming_alpha(alpha: float) -> None:
    text = _refusal(_dsl(alpha=alpha))
    assert "param 'alpha' must be finite" in text


def test_omitted_or_null_alpha_uses_default_0_7() -> None:
    validate_strategy_dict(_dsl())
    validate_strategy_dict(translate_dsl_config(_dsl(alpha=None)))
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 400))
    df = pd.DataFrame({"close": close})
    omitted = compute_adaptive_rsi(df, {"period": 14}).to_numpy()
    explicit = compute_adaptive_rsi(df, {"period": 14, "alpha": 0.7}).to_numpy()
    np.testing.assert_array_equal(omitted, explicit)
    assert not np.array_equal(
        explicit, compute_adaptive_rsi(df, {"period": 14, "alpha": 0.5}).to_numpy()
    )


def test_ga_draws_alpha_only_from_0_7_to_0_9() -> None:
    assert INDICATOR_POOL["ADAPTIVE_RSI"].param_ranges["alpha"] == (0.7, 0.9)


def _arsi_seed(alpha: float) -> StrategyChromosome:
    gene = StrategyGene(
        indicator_type="ADAPTIVE_RSI",
        parameters={"period": 14.0, "alpha": alpha},
        condition=ConditionType.LT,
        threshold=30.0,
    )
    return StrategyChromosome(
        entry_genes=[gene],
        exit_genes=[gene.clone()],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def test_warm_start_drops_old_seed_below_floor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = StateManager(db_path=tmp_path / "seed.db")
    try:
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="discovery",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2025-01-01",
            end_date="2025-04-01",
            parameters={},
        )
        seeds = [_arsi_seed(0.3), _arsi_seed(0.8), _arsi_seed(0.69999), _arsi_seed(0.7)]
        notes = {"top_strategies": [{"chromosome": chromosome_to_serializable(c)} for c in seeds]}
        state.save_backtest_result(run_id, {"notes": json.dumps(notes)})
        with caplog.at_level(logging.WARNING, logger="vibe_quant.discovery.seed"):
            loaded = load_seed_chromosomes(state, run_id)
    finally:
        state.close()
    assert loaded is not None
    assert [c.entry_genes[0].parameters["alpha"] for c in loaded] == [0.8, 0.7]
    assert "dropped 2 seed chromosome(s)" in caplog.text
    assert "outside the current GA parameter ranges" in caplog.text


@pytest.mark.parametrize(("alpha", "refused"), [(0.5, True), (0.69999, True), (0.7, False)])
def test_sweep_override_below_floor_is_refused_when_strategy_is_built(
    alpha: float, refused: bool
) -> None:
    """A sweep/override value never passes strategy validation: the compiled strategy
    refuses it from its EFFECTIVE params when it is built, before any bar."""
    from vibe_quant.dsl.compiler import StrategyCompiler

    module = StrategyCompiler().compile_to_module(validate_strategy_dict(_dsl(alpha=0.8)))
    config_cls = module.ArsiFloorConfig
    strategy_cls = module.ArsiFloorStrategy
    config = config_cls(instrument_id="BTCUSDT-PERP.BINANCE", arsi_alpha=alpha)
    if refused:
        with pytest.raises(ValueError, match=R1):
            strategy_cls(config)
    else:
        assert strategy_cls(config)._pta_lookback["arsi"] == 28


def test_explicit_null_alpha_counts_as_omitted_at_validation_and_save(tmp_path: Path) -> None:
    """Chief ruling B1: canonical ``alpha: null`` is omitted (default 0.7), not a type error."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vibe_quant.api import deps
    from vibe_quant.api.routers import strategies
    from vibe_quant.dsl.compiler import StrategyCompiler

    def compiled_alpha(dsl: dict[str, Any]) -> object:
        module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
        return module.ArsiFloorConfig(instrument_id="BTCUSDT-PERP.BINANCE").arsi_alpha

    assert compiled_alpha(_dsl(alpha=None)) == 0.7

    app = FastAPI()
    app.include_router(strategies.router)
    state = StateManager(db_path=tmp_path / "state.db")
    app.dependency_overrides[deps.get_state_manager] = lambda: state
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/strategies", json={"name": "arsi_null", "dsl_config": _dsl(alpha=None)}
            )
    finally:
        state.close()
    assert response.status_code == 201, response.text
    assert compiled_alpha(response.json()["dsl_config"]) == 0.7


def test_warm_start_floor_leaves_other_indicator_seeds_alone(tmp_path: Path) -> None:
    """Chief ruling S1: only ADAPTIVE_RSI alpha is filtered; a historical BBANDS std_dev
    outside today's GA range is kept, as before this change."""
    seed = _arsi_seed(0.8)
    for gene in seed.entry_genes + seed.exit_genes:
        gene.indicator_type = "BBANDS"
        gene.parameters = {"period": 14.0, "std_dev": 3.5}
        gene.sub_value = "percent_b"
        gene.threshold = 0.5
    assert not (
        INDICATOR_POOL["BBANDS"].param_ranges["std_dev"][0]
        <= 3.5
        <= INDICATOR_POOL["BBANDS"].param_ranges["std_dev"][1]
    )
    state = StateManager(db_path=tmp_path / "seed.db")
    try:
        run_id = _seed_run(state, [chromosome_to_serializable(seed)])
        loaded = load_seed_chromosomes(state, run_id)
    finally:
        state.close()
    assert loaded is not None
    assert [c.entry_genes[0].parameters["std_dev"] for c in loaded] == [3.5]


def test_warm_start_counts_legacy_both_conversion_only_for_kept_seeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chief ruling S2: a dropped direction=both seed is not reported as loaded."""
    dropped = _arsi_seed(0.3)
    dropped.direction = Direction.BOTH
    kept_both = _arsi_seed(0.8)
    kept_both.direction = Direction.BOTH
    legacy = [
        {k: v for k, v in chromosome_to_serializable(c).items() if not k.startswith("short_")}
        for c in (dropped, kept_both)
    ]
    state = StateManager(db_path=tmp_path / "seed.db")
    try:
        run_id = _seed_run(state, [*legacy, chromosome_to_serializable(_arsi_seed(0.85))])
        with caplog.at_level(logging.WARNING, logger="vibe_quant.discovery.seed"):
            loaded = load_seed_chromosomes(state, run_id)
    finally:
        state.close()
    assert loaded is not None
    assert [c.entry_genes[0].parameters["alpha"] for c in loaded] == [0.8, 0.85]
    assert "dropped 1 seed chromosome(s)" in caplog.text
    assert "loaded 1 legacy direction=both" in caplog.text


def _seed_run(state: StateManager, blobs: list[dict[str, Any]]) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-04-01",
        parameters={},
    )
    notes = {"top_strategies": [{"chromosome": blob} for blob in blobs]}
    state.save_backtest_result(run_id, {"notes": json.dumps(notes)})
    return run_id
