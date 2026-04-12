"""Indicator catalog router.

Exposes ``GET /api/indicators/catalog`` so the frontend can render its
indicator picker off the live registry. Every built-in spec plus every
plugin dropped into ``vibe_quant/dsl/plugins/`` is surfaced here with
zero per-indicator code in this file — the spec IS the catalog entry.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from vibe_quant.api.schemas.indicators import (
    IndicatorCatalogEntry,
    IndicatorCatalogResponse,
)
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/indicators", tags=["indicators"])


def _spec_to_entry(spec: IndicatorSpec) -> IndicatorCatalogEntry:
    """Project an IndicatorSpec into the API catalog shape.

    ``param_schema``'s values are Python ``type`` objects, which aren't
    JSON-native — we send the type's ``__name__`` instead so the UI can
    decide whether to render an int / float / string input.
    """
    return IndicatorCatalogEntry(
        type_name=spec.name,
        display_name=spec.display_name or spec.name,
        description=spec.description,
        category=spec.category,
        popular=spec.popular,
        chart_placement=spec.chart_placement,
        default_params={
            k: v  # type: ignore[misc]
            for k, v in spec.default_params.items()
            if isinstance(v, (int, float, str, bool))
        },
        param_schema={k: t.__name__ for k, t in spec.param_schema.items()},
        output_names=list(spec.output_names),
        requires_high_low=spec.requires_high_low,
        requires_volume=spec.requires_volume,
    )


@router.get("/catalog", response_model=IndicatorCatalogResponse)
def get_catalog() -> IndicatorCatalogResponse:
    """Return every registered indicator spec as a catalog entry.

    Ordered alphabetically by ``type_name`` via
    ``indicator_registry.all_specs()``. The UI is expected to re-group
    by ``category`` using the ``categories`` field on the response.
    """
    entries = [_spec_to_entry(spec) for spec in indicator_registry.all_specs()]
    return IndicatorCatalogResponse(indicators=entries)
