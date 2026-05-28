"""Pydantic schemas for the indicator catalog API.

Mirrors the public surface of ``IndicatorSpec`` as pure JSON so the
frontend can render an indicator picker without the DSL python layer.
New indicators (plugins dropped into ``vibe_quant/dsl/plugins/``) show up
automatically once the process restarts — no schema edits here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndicatorCatalogEntry(BaseModel):
    """One indicator row in the catalog response.

    Every field maps to an ``IndicatorSpec`` attribute, with ``param_schema``
    rendered as ``{name: type_name}`` because types aren't JSON-native.
    """

    type_name: str = Field(..., description="Canonical uppercase indicator name (RSI, MACD, ...)")
    display_name: str = Field(
        "", description="Human-readable label for the UI (e.g. 'Relative Strength Index')"
    )
    description: str = Field("", description="One-line description for tooltips")
    category: str = Field(
        "Custom", description="UI category: Trend | Momentum | Volatility | Volume | Custom"
    )
    popular: bool = Field(False, description="Highlight in the UI as a commonly-used indicator")
    chart_placement: str = Field(
        "oscillator",
        description="Rendering hint: 'overlay' (draws on price) or 'oscillator' (own pane)",
    )
    default_params: dict[str, float | int | str | bool] = Field(
        default_factory=dict, description="Default parameter values"
    )
    param_schema: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter name → Python type name (e.g. {'period': 'int'})",
    )
    output_names: list[str] = Field(
        default_factory=lambda: ["value"],
        description="Names of output series (multi-output indicators expose several)",
    )
    requires_high_low: bool = Field(
        False, description="Indicator needs high/low series (ATR, STOCH, ADX, …)"
    )
    requires_volume: bool = Field(
        False, description="Indicator needs the volume series (MFI, OBV, VWAP, VOLSMA)"
    )
    source_file: str | None = Field(
        None,
        description=(
            "Repo-relative path to the plugin file this spec was registered "
            "from, or null for built-ins. Used by the UI to surface promote "
            "actions on proposed_* plugins."
        ),
    )
    is_proposed: bool = Field(
        False,
        description=(
            "True iff the spec was registered from a "
            "``vibe_quant/dsl/plugins/proposed_*.py`` file — i.e. an "
            "LLM-scaffolded indicator pending human promotion."
        ),
    )


class PluginLoadErrorEntry(BaseModel):
    """One plugin that failed to import at startup."""

    module: str = Field(..., description="Fully-qualified module path (e.g. vibe_quant.dsl.plugins.foo)")
    error_type: str = Field(..., description="Exception class name")
    message: str = Field(..., description="Exception message")


class IndicatorCatalogResponse(BaseModel):
    """Wrapper response returned by ``GET /api/indicators/catalog``."""

    indicators: list[IndicatorCatalogEntry]
    categories: list[str] = Field(
        default_factory=lambda: ["Trend", "Momentum", "Volatility", "Volume", "Custom"],
        description="Ordered category list for grouped rendering in the UI",
    )
    plugin_errors: list[PluginLoadErrorEntry] = Field(
        default_factory=list,
        description="Plugins that failed to import at startup; empty on healthy installs",
    )
