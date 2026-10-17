"""T7 (bead vibe-quant-q7v3f): a strategy save rejects indicator params its type does not use.

Contract rows R60-R68 for job 20261016-safety-bankroll (design.md § B4). The save check is
indicator-only (SF1): an unrelated risk/condition error cannot block a save (R67) and a config
with no indicators is a no-op (R67b).
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.genome import (
    StrategyChromosome,
    StrategyGene,
    chromosome_to_dsl,
)
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.schema import (
    IndicatorConfig,
    StrategyDSL,
    check_strategy_indicator_params,
)
from vibe_quant.dsl.translator import translate_dsl_config
from vibe_quant.jobs.manager import BacktestJobManager

EMA_STD_DEV_MSG = "Indicator 'EMA' does not use param 'std_dev'. Valid params: ['period']"


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(tmp_db: Path):
    """Create app with temp DB, manually wire up lifespan deps (mirrors the api tests)."""
    app = create_app()

    state_mgr = StateManager(db_path=tmp_db)
    _ = state_mgr.conn  # force schema init

    job_mgr = BacktestJobManager(tmp_db)  # never the real default DB
    catalog_mgr = CatalogManager()
    ws_mgr = ConnectionManager()
    await ws_mgr.start()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.catalog_manager = catalog_mgr
    app.state.ws_manager = ws_mgr

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _ui(indicators: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A frontend DslConfig body (``indicators`` as a list) with an optional risk/conditions."""
    return {"general": {"timeframe": "1h"}, "indicators": indicators, **extra}


# --- R67: the save checks indicators only -----------------------------------


@pytest.mark.parametrize(
    ("name", "dsl_config"),
    [
        (
            "t7_trailing",
            _ui([{"type": "EMA", "params": {"period": 20}}], risk={"trailing_stop_pct": 1}),
        ),
        (
            "t7_bad_condition",
            _ui(
                [{"type": "EMA", "params": {"period": 20}}],
                conditions={
                    "entry": [
                        {"left": "EMA(20)", "operator": ">", "right": "price"},
                        {"left": "EMA(20)", "operator": ">", "right": "price", "logic": "or"},
                    ]
                },
            ),
        ),
    ],
)
async def test_save_ignores_non_indicator_errors(
    client: AsyncClient, name: str, dsl_config: dict[str, Any]
) -> None:
    """A clean EMA saves even with an unsupported risk key / a condition the launch rejects."""
    r = await client.post("/api/strategies", json={"name": name, "dsl_config": dsl_config})
    assert r.status_code == 201, r.text


# --- R67b: a config with no indicators is a no-op ---------------------------


@pytest.mark.parametrize(
    ("name", "dsl_config"),
    [
        ("t7_no_indicators", {"general": {"timeframe": "1h"}}),
        ("t7_null_indicators", {"general": {"timeframe": "1h"}, "indicators": None}),
        ("t7_empty_indicators", {"general": {"timeframe": "1h"}, "indicators": []}),
    ],
)
async def test_save_without_indicators_noop(
    client: AsyncClient, name: str, dsl_config: dict[str, Any]
) -> None:
    """Stored configs with no ``indicators`` (ids 222/227/228) must still save."""
    r = await client.post("/api/strategies", json={"name": name, "dsl_config": dsl_config})
    assert r.status_code == 201, r.text


# --- R66: a save rejects an unused param, in both config spellings ----------


@pytest.mark.parametrize(
    ("name", "dsl_config"),
    [
        ("t7_ui_unused", _ui([{"type": "EMA", "params": {"period": 20, "std_dev": 2}}])),
        ("t7_dsl_unused", {"indicators": {"ema": {"type": "EMA", "period": 20, "std_dev": 2}}}),
    ],
)
async def test_save_rejects_unused_param(
    client: AsyncClient, name: str, dsl_config: dict[str, Any]
) -> None:
    r = await client.post("/api/strategies", json={"name": name, "dsl_config": dsl_config})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == EMA_STD_DEV_MSG
    listing = await client.get("/api/strategies")
    assert all(s["name"] != name for s in listing.json()["strategies"])


async def test_update_rejects_unused_param(client: AsyncClient) -> None:
    """``PUT`` runs the same check (design: create/update call it)."""
    cr = await client.post(
        "/api/strategies",
        json={"name": "t7_update", "dsl_config": _ui([{"type": "EMA", "params": {"period": 20}}])},
    )
    assert cr.status_code == 201, cr.text
    sid = cr.json()["id"]
    r = await client.put(
        f"/api/strategies/{sid}",
        json={"dsl_config": _ui([{"type": "EMA", "params": {"period": 20, "std_dev": 2}}])},
    )
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == EMA_STD_DEV_MSG


# --- R60-R65, R68: the IndicatorConfig / check unit behaviour ----------------


def test_ema_std_dev_rejected() -> None:
    """R60: EMA declares only ``period``; ``std_dev`` must be rejected, not ignored."""
    with pytest.raises(ValidationError) as exc:
        IndicatorConfig(type="EMA", period=20, std_dev=2)
    assert EMA_STD_DEV_MSG in str(exc.value)


def test_stoch_spellings_accepted() -> None:
    """R61: STOCH ``period``/``d_period`` and the UI ``k_period``/``d_period`` both pass."""
    direct = IndicatorConfig(type="STOCH", period=14, d_period=3)
    assert (direct.period, direct.d_period) == (14, 3)
    check_strategy_indicator_params(
        {"indicators": [{"type": "STOCH", "params": {"k_period": 14, "d_period": 3}}]}
    )


def test_macd_period_rejected() -> None:
    """R62: MACD has no ``period`` field; the message lists its three real params."""
    with pytest.raises(ValidationError) as exc:
        IndicatorConfig(type="MACD", period=14)
    assert (
        "Indicator 'MACD' does not use param 'period'. "
        "Valid params: ['fast_period', 'signal_period', 'slow_period']"
    ) in str(exc.value)


def test_declared_builtins_accepted() -> None:
    """R63: KC ``atr_multiplier``, BBANDS ``std_dev`` and the UI ``BB`` alias all pass."""
    IndicatorConfig(type="KC", period=20, atr_multiplier=2.0)
    IndicatorConfig(type="BBANDS", period=20, std_dev=2.0)
    check_strategy_indicator_params(
        {"indicators": [{"type": "BB", "params": {"period": 20, "std_dev": 2}}]}
    )


def test_none_valued_ignored() -> None:
    """R64: an explicit ``None`` is not a used param (the editor emits ``std_dev: null``)."""
    assert IndicatorConfig(type="EMA", period=20, std_dev=None).std_dev is None


def test_paramless_type_rejects_period() -> None:
    """R65: OBV takes no params at all."""
    with pytest.raises(ValidationError) as exc:
        IndicatorConfig(type="OBV", period=14)
    assert "Indicator 'OBV' does not use param 'period'. Valid params: []" in str(exc.value)


def test_check_runs_on_user_fields_only() -> None:
    """R68: the check runs before defaults fill, so only user-set fields are inspected."""
    cfg = IndicatorConfig(type="MACD")
    assert (cfg.fast_period, cfg.slow_period, cfg.signal_period) == (12, 26, 9)
    with pytest.raises(ValidationError) as exc:
        IndicatorConfig(type="MACD", period=14)
    assert "does not use param 'period'" in str(exc.value)
    # A default-filled field is invisible: the check keys off model_fields_set, not attributes.
    probe = IndicatorConfig.model_construct(type="EMA", period=14, std_dev=2.0)
    probe.__pydantic_fields_set__ = {"type"}  # as if the caller gave only ``type``
    IndicatorConfig.validate_used_params(probe)  # must not reject the prefilled std_dev


# --- T7-R2: the GA must not inject a dead ``period`` (vibe-quant-q7v3f) -------

# The three enrolled indicator types whose registry param_schema has no ``period``.
_DEAD_PERIOD_TYPES: list[tuple[str, dict[str, float]]] = [
    ("SQUEEZE_MOM", {"length": 20}),
    ("SQUEEZE_RATIO", {"length": 20, "bb_mult": 2.0, "kc_mult": 1.5}),
    ("TREND_ENSEMBLE", {"max_lookback": 250}),
]


@pytest.mark.parametrize(("ind_type", "params"), _DEAD_PERIOD_TYPES)
def test_genome_gene_drops_only_the_dead_period(
    ind_type: str, params: dict[str, float]
) -> None:
    """A generated gene carries exactly the spec params — the injected period is gone.

    Pre-fix ``_gene_to_indicator_config`` injected ``period=14`` for every type not
    MACD/STOCH/BBANDS. The compute path never reads that key (the compiler emitted
    ``<name>_period`` but the indicator is built from its spec params), so the only
    change is the dead key disappearing: the config equals ``{type, **spec params}``,
    it still validates, and the generated module has no field to read a period from.
    """
    gene = StrategyGene(ind_type, dict(params), "less_than", 50.0)
    chrom = StrategyChromosome(
        entry_genes=[gene], exit_genes=[gene], stop_loss_pct=2.0, take_profit_pct=4.0
    )
    dsl_dict = chromosome_to_dsl(chrom)
    name = f"{ind_type.lower()}_entry_0"
    assert dsl_dict["indicators"][name] == {"type": ind_type, **params}

    dsl = StrategyDSL.model_validate(dsl_dict)
    source = StrategyCompiler().compile(dsl)
    assert f"{name}_period" not in source  # nothing could read the removed period
    for param in params:
        assert f"{name}_{param}" in source  # the real params still reach the strategy
    StrategyCompiler().compile_to_module(dsl)



# --- B1 (R64): an explicit null param saves on every endpoint and format ----


@pytest.mark.parametrize(
    "dsl_config",
    [
        _ui([{"type": "EMA", "params": {"period": 20, "std_dev": None}}]),
        {"indicators": {"ema": {"type": "EMA", "period": 20, "std_dev": None}}},
    ],
    ids=["ui_list", "canonical_dict"],
)
async def test_null_param_saves_on_post_and_put(
    client: AsyncClient, dsl_config: dict[str, Any]
) -> None:
    """``std_dev: null`` on EMA is ignored: POST 201 and PUT 200 (was HTTP 500 in UI format)."""
    r = await client.post("/api/strategies", json={"name": "t7_null", "dsl_config": dsl_config})
    assert r.status_code == 201, r.text
    assert "'std_dev': 2" not in str(r.json()["dsl_config"])  # no real std_dev was stored
    clean = _ui([{"type": "EMA", "params": {"period": 20}}])
    cr = await client.post("/api/strategies", json={"name": "t7_null_put", "dsl_config": clean})
    assert cr.status_code == 201, cr.text
    pr = await client.put(f"/api/strategies/{cr.json()['id']}", json={"dsl_config": dsl_config})
    assert pr.status_code == 200, pr.text


# --- SF1 (T7-R2): computed-value parity of the genome period fix ------------


@pytest.mark.parametrize(("ind_type", "params"), _DEAD_PERIOD_TYPES)
def test_dead_period_does_not_change_computed_values(
    ind_type: str, params: dict[str, float]
) -> None:
    """Indicator values (incl. NaN positions) are identical with and without the old period=14."""
    import numpy as np

    from vibe_quant.dsl.compiler import _to_class_name

    gene = StrategyGene(ind_type, dict(params), "less_than", 50.0)
    chrom = StrategyChromosome(
        entry_genes=[gene], exit_genes=[gene], stop_loss_pct=2.0, take_profit_pct=4.0
    )
    new_dsl = StrategyDSL.model_validate(chromosome_to_dsl(chrom))
    name = f"{ind_type.lower()}_entry_0"
    # model_copy skips validation: rebuild the pre-fix config that carried period=14.
    old_cfg = new_dsl.indicators[name].model_copy(update={"period": 14})
    old_dsl = new_dsl.model_copy(update={"indicators": {**new_dsl.indicators, name: old_cfg}})

    rng = np.random.default_rng(727)
    n = 800
    close = 100 + np.cumsum(rng.normal(0, 0.9, n))
    rows = np.column_stack(
        (
            close - 0.1,
            close + rng.uniform(0.1, 1, n),
            close - rng.uniform(0.1, 1, n),
            close,
            rng.uniform(50, 500, n),
        )
    )

    def run(dsl: StrategyDSL) -> Any:
        mod = StrategyCompiler().compile_to_module(dsl)
        cls = _to_class_name(dsl.name)
        strat = getattr(mod, cls + "Strategy")(
            getattr(mod, cls + "Config")(instrument_id="BTCUSDT-PERP.BINANCE")
        )
        tf = dsl.timeframe
        vals = []
        for row in rows:
            strat._pta_bufs[tf].append(*map(float, row))
            strat._update_pta_indicators(tf)
            vals.append(strat._pta_values.get(name, float("nan")))
        return np.array(vals)

    old_vals, new_vals = run(old_dsl), run(new_dsl)
    assert np.isfinite(new_vals).sum() > 100  # the indicator really computed
    assert np.array_equal(np.isnan(old_vals), np.isnan(new_vals))
    assert np.array_equal(old_vals, new_vals, equal_nan=True)


# --- T7-R4 (B1 residual): the translator drops None params everywhere --------

_EMA_ENTRY = {"entry": [{"left": "EMA(20)", "operator": ">", "right": "price"}]}
_EMA_NULL_UI = _ui(
    [{"type": "EMA", "params": {"period": 20, "std_dev": None}}], conditions=_EMA_ENTRY
)
_EMA_CLEAN_UI = _ui([{"type": "EMA", "params": {"period": 20}}], conditions=_EMA_ENTRY)


async def test_stored_null_param_validates(client: AsyncClient) -> None:
    """T7-R4: a stored ``std_dev: null`` no longer breaks /validate (was valid:false float(None))."""
    r = await client.post(
        "/api/strategies", json={"name": "t7_null_validate", "dsl_config": _EMA_NULL_UI}
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    stored = (await client.get(f"/api/strategies/{sid}")).json()["dsl_config"]
    assert stored["indicators"][0]["params"]["std_dev"] is None  # the null payload is preserved
    v = await client.post(f"/api/strategies/{sid}/validate")
    assert v.status_code == 200, v.text
    assert v.json() == {"valid": True, "errors": []}, v.text


def test_translate_drops_null_param_like_absent_key() -> None:
    """T7-R4: a UI-format null param translates to exactly the absent-key result."""
    assert translate_dsl_config(_EMA_NULL_UI, strategy_name="t7_parity") == translate_dsl_config(
        _EMA_CLEAN_UI, strategy_name="t7_parity"
    )


def test_translate_drops_null_builtin_in_canonical_format() -> None:
    """T7-R4: a canonical-format config with an explicit null builtin equals the absent-key form."""
    canonical_null = {
        "name": "t7_canon",
        "timeframe": "1h",
        "indicators": {"ema": {"type": "EMA", "period": 20, "std_dev": None}},
        "entry_conditions": {"long": ["EMA(20) > close"], "short": []},
        "stop_loss": {"type": "fixed_pct", "value": 2.0},
        "take_profit": {"type": "fixed_pct", "value": 4.0},
    }
    canonical_clean = {
        **canonical_null,
        "indicators": {"ema": {"type": "EMA", "period": 20}},
    }
    translated = translate_dsl_config(canonical_null)
    assert "std_dev" not in translated["indicators"]["ema"]  # the null builtin is gone
    assert translated == translate_dsl_config(canonical_clean)
