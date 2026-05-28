"""Indicator catalog router.

Exposes ``GET /api/indicators/catalog`` so the frontend can render its
indicator picker off the live registry. Every built-in spec plus every
plugin dropped into ``vibe_quant/dsl/plugins/`` is surfaced here with
zero per-indicator code in this file — the spec IS the catalog entry.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

from fastapi import APIRouter

from vibe_quant.api.schemas.indicators import (
    IndicatorCatalogEntry,
    IndicatorCatalogResponse,
    PluginLoadErrorEntry,
)
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugin_loader import get_load_errors, reload_plugins

_PLUGINS_DIR = Path(__file__).resolve().parent.parent.parent / "dsl" / "plugins"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/indicators", tags=["indicators"])


def _spec_source_file(spec: IndicatorSpec) -> tuple[str | None, bool]:
    """Return ``(source_file, is_proposed)`` for a spec.

    Resolves the spec's compute_fn source path via ``inspect.getfile``.
    Only specs registered from a plugin under
    ``vibe_quant/dsl/plugins/`` get a ``source_file`` — built-ins with
    a generic dispatcher compute_fn return ``(None, False)``.

    ``is_proposed`` is true iff the basename starts with ``proposed_``.
    """
    fn = spec.compute_fn
    if fn is None:
        return None, False
    try:
        src = inspect.getfile(fn)
    except (TypeError, OSError):
        return None, False
    p = Path(src).resolve()
    try:
        rel_to_plugins = p.relative_to(_PLUGINS_DIR.resolve())
    except ValueError:
        return None, False
    # Only return a source_file for top-level plugin .py files (no
    # subdirs) — keeps the surface honest and matches the scaffold
    # contract that lays everything flat in ``dsl/plugins/``.
    if rel_to_plugins.parent != Path():
        return None, False
    try:
        repo_rel = str(p.relative_to(Path.cwd()))
    except ValueError:
        repo_rel = str(p)
    return repo_rel, p.name.startswith("proposed_")


def _spec_to_entry(spec: IndicatorSpec) -> IndicatorCatalogEntry:
    """Project an IndicatorSpec into the API catalog shape.

    ``param_schema``'s values are Python ``type`` objects, which aren't
    JSON-native — we send the type's ``__name__`` instead so the UI can
    decide whether to render an int / float / string input.
    """
    source_file, is_proposed = _spec_source_file(spec)
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
        source_file=source_file,
        is_proposed=is_proposed,
    )


@router.get("/catalog", response_model=IndicatorCatalogResponse)
def get_catalog() -> IndicatorCatalogResponse:
    """Return every registered indicator spec as a catalog entry.

    Ordered alphabetically by ``type_name`` via
    ``indicator_registry.all_specs()``. The UI is expected to re-group
    by ``category`` using the ``categories`` field on the response.
    """
    entries = [_spec_to_entry(spec) for spec in indicator_registry.all_specs()]
    plugin_errors = [
        PluginLoadErrorEntry(
            module=err.module,
            error_type=err.error_type,
            message=err.message,
        )
        for err in get_load_errors()
    ]
    return IndicatorCatalogResponse(
        indicators=entries, plugin_errors=plugin_errors
    )


@router.post("/reload")
def reload_plugin_catalog() -> dict[str, object]:
    """Re-import every plugin module under ``vibe_quant/dsl/plugins/``.

    Dev-mode convenience: edit a plugin file, POST to this endpoint, and
    the registry picks up the new spec without a backend restart.
    Built-in indicators are left untouched; only plugin-registered specs
    are unregistered and reimported. The response echoes the loaded
    module list and any load errors for quick diagnosis.
    """
    loaded = reload_plugins()
    errors = [
        {
            "module": err.module,
            "error_type": err.error_type,
            "message": err.message,
        }
        for err in get_load_errors()
    ]
    return {"loaded": loaded, "errors": errors}
