"""Coverage for the indicator catalog router (P8).

Pins the wire contract so plugin authors can rely on
``GET /api/indicators/catalog`` surfacing their spec as soon as it's
registered, without any schema edits.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# 1. Catalog returns every registered indicator
# ---------------------------------------------------------------------------


def test_catalog_returns_all_registered_indicators(client: TestClient) -> None:
    """Every name in the registry must appear in the response."""
    resp = client.get("/api/indicators/catalog")
    assert resp.status_code == 200
    body = resp.json()
    api_names = {entry["type_name"] for entry in body["indicators"]}
    registry_names = set(indicator_registry.list_indicators())
    assert api_names == registry_names


# ---------------------------------------------------------------------------
# 2. Response matches the pydantic shape (every field present)
# ---------------------------------------------------------------------------


def test_catalog_response_schema_matches_pydantic_model(client: TestClient) -> None:
    """Every entry must carry the fields IndicatorCatalogEntry declares."""
    resp = client.get("/api/indicators/catalog")
    body = resp.json()
    assert body["indicators"], "Empty catalog body"
    required_keys = {
        "type_name",
        "display_name",
        "description",
        "category",
        "popular",
        "chart_placement",
        "default_params",
        "param_schema",
        "output_names",
        "requires_high_low",
        "requires_volume",
        "source_file",
        "is_proposed",
    }
    for entry in body["indicators"]:
        missing = required_keys - set(entry)
        assert not missing, f"Missing keys on {entry.get('type_name')}: {missing}"
    assert "categories" in body
    assert set(body["categories"]) >= {"Trend", "Momentum", "Volatility", "Volume"}


# ---------------------------------------------------------------------------
# 3. Custom plugin spec appears in the response
# ---------------------------------------------------------------------------


def test_custom_plugin_appears_in_catalog(client: TestClient) -> None:
    """Register a throwaway spec at test time — it must immediately show
    up in a fresh catalog request. Confirms the "drop a plugin, restart,
    done" contract with zero schema edits."""
    def _noop_compute(df, params):  # noqa: ARG001
        return df["close"]

    spec = IndicatorSpec(
        name="TESTP8CUSTOM",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 12},
        param_schema={"period": int},
        compute_fn=_noop_compute,
        display_name="P8 Test Custom",
        description="Used by tests to validate the catalog endpoint.",
        category="Momentum",
        chart_placement="oscillator",
    )
    indicator_registry.register_spec(spec)
    try:
        resp = client.get("/api/indicators/catalog")
        assert resp.status_code == 200
        body = resp.json()
        entry = next(
            (e for e in body["indicators"] if e["type_name"] == "TESTP8CUSTOM"),
            None,
        )
        assert entry is not None, "Custom spec missing from catalog"
        assert entry["display_name"] == "P8 Test Custom"
        assert entry["category"] == "Momentum"
        assert entry["chart_placement"] == "oscillator"
        assert entry["default_params"] == {"period": 12}
        assert entry["param_schema"] == {"period": "int"}
    finally:
        indicator_registry._indicators.pop("TESTP8CUSTOM", None)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. Standard category buckets are all present
# ---------------------------------------------------------------------------


def test_categories_include_all_four_standard_buckets(client: TestClient) -> None:
    """The UI needs all five buckets (four canonical + Custom) to render
    a stable grouped picker — even if the current registry has zero
    entries in one of them."""
    resp = client.get("/api/indicators/catalog")
    body = resp.json()
    for bucket in ("Trend", "Momentum", "Volatility", "Volume", "Custom"):
        assert bucket in body["categories"], f"Missing bucket: {bucket}"


# ---------------------------------------------------------------------------
# 5. Multi-output indicators expose their full output_names list
# ---------------------------------------------------------------------------


def test_multi_output_indicators_list_output_names_correctly(
    client: TestClient,
) -> None:
    """BBANDS exposes upper/middle/lower/percent_b/bandwidth; MACD
    exposes macd/signal/histogram; STOCH exposes k/d. These match
    spec.output_names verbatim."""
    resp = client.get("/api/indicators/catalog")
    entries = {e["type_name"]: e for e in resp.json()["indicators"]}

    bbands = entries["BBANDS"]
    assert set(bbands["output_names"]) >= {
        "upper",
        "middle",
        "lower",
        "percent_b",
        "bandwidth",
    }

    macd = entries["MACD"]
    assert set(macd["output_names"]) == {"macd", "signal", "histogram"}

    stoch = entries["STOCH"]
    assert set(stoch["output_names"]) == {"k", "d"}


# ---------------------------------------------------------------------------
# 6. source_file + is_proposed surface for plugin specs
# ---------------------------------------------------------------------------


def test_plugin_indicators_carry_source_file_and_is_proposed_flag(
    client: TestClient,
) -> None:
    """Plugins under ``vibe_quant/dsl/plugins/`` surface their file path
    so the UI can offer promote actions on ``proposed_*`` entries."""
    resp = client.get("/api/indicators/catalog")
    entries = {e["type_name"]: e for e in resp.json()["indicators"]}

    # The example_adaptive_rsi plugin ships in the repo.
    arsi = entries.get("ADAPTIVE_RSI")
    if arsi is not None:
        assert arsi["source_file"] is not None
        assert arsi["source_file"].endswith("example_adaptive_rsi.py")
        assert arsi["is_proposed"] is False

    # Any indicator whose plugin file basename starts with ``proposed_``
    # must surface is_proposed=True. We can't depend on a specific
    # proposed_* indicator existing at test time, so the assertion is
    # conditional on whether one is registered.
    proposed = [
        e
        for e in entries.values()
        if e.get("source_file") and "proposed_" in e["source_file"].rsplit("/", 1)[-1]
    ]
    for entry in proposed:
        assert entry["is_proposed"] is True


def test_builtin_indicators_have_null_source_file(client: TestClient) -> None:
    """Built-in (NT-native / pandas-ta) indicators have no plugin file —
    they should report ``source_file=None`` so the UI doesn't try to
    promote them."""
    resp = client.get("/api/indicators/catalog")
    entries = {e["type_name"]: e for e in resp.json()["indicators"]}
    # RSI is a built-in spec (registered by core, not by a plugin file).
    rsi = entries.get("RSI")
    assert rsi is not None
    assert rsi["source_file"] is None
    assert rsi["is_proposed"] is False
