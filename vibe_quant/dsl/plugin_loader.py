"""Best-effort auto-importer for indicator plugins.

Walks ``vibe_quant.dsl.plugins`` with ``pkgutil.iter_modules`` at startup
and imports every non-underscore-prefixed module. Each plugin is
responsible for calling ``indicator_registry.register(name)`` or
``register_spec(spec)`` at module scope — the loader just triggers the
import.

A failing plugin is logged (not raised) so one broken plugin can never
take down the entire registry. Built-ins must be registered before this
runs; ``vibe_quant/dsl/indicators.py`` calls ``load_builtin_plugins``
from the bottom of the file, after every ``@indicator_registry.register``
has executed.

Plugin name collisions with a built-in raise ``KeyError`` by default;
plugins that intentionally shadow a built-in must call
``indicator_registry.register_spec(spec, override=True)`` explicitly.
Collisions that do go through (via ``override=True``) are still logged
at INFO level so the override is visible in prod logs.

Failed-load surfacing
---------------------
Plugin import errors are recorded on the module-level ``_load_errors``
list (cleared at the start of each ``load_builtin_plugins`` call). Read
via :func:`get_load_errors`; surfaced to the frontend through the
``/api/indicators/catalog`` response. Set ``VQ_PLUGINS_STRICT=1`` in the
environment to re-raise the first failure instead — useful in CI.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PluginLoadError:
    """One plugin that failed to import, with the error message."""

    module: str
    error_type: str
    message: str


_load_errors: list[PluginLoadError] = []


def _strict_mode() -> bool:
    """Return True iff ``VQ_PLUGINS_STRICT`` is set to a truthy value."""
    return os.environ.get("VQ_PLUGINS_STRICT", "").lower() in ("1", "true", "yes")


def get_load_errors() -> list[PluginLoadError]:
    """Return a copy of the most recent load-error list."""
    return list(_load_errors)


def load_builtin_plugins() -> list[str]:
    """Import every non-underscore module under ``vibe_quant.dsl.plugins``.

    Returns:
        List of fully-qualified module names that were successfully
        imported. Callers don't usually need this; the side effect (spec
        registration) is the whole point. Tests consume the return value
        to assert which plugins loaded.

    Raises:
        Exception: If ``VQ_PLUGINS_STRICT=1`` and any plugin raises
            during import, the first error is re-raised after logging.
    """
    # Import the package lazily so callers (e.g., tests) that want to
    # monkey-patch the package ``__path__`` before the first load can do
    # so without a stale reference.
    from vibe_quant.dsl import plugins

    _load_errors.clear()
    loaded: list[str] = []
    strict = _strict_mode()

    for module_info in pkgutil.iter_modules(plugins.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        qualified = f"{plugins.__name__}.{name}"
        try:
            importlib.import_module(qualified)
        except Exception as exc:
            # Plugins are third-party / experimental code; any import
            # failure is logged and recorded so the rest of the registry
            # stays usable. VQ_PLUGINS_STRICT re-raises to fail CI.
            logger.error(
                "Failed to load indicator plugin %s: %s", qualified, exc
            )
            _load_errors.append(
                PluginLoadError(
                    module=qualified,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            if strict:
                raise
            continue
        loaded.append(qualified)
        logger.info("Loaded indicator plugin: %s", qualified)

    return loaded
