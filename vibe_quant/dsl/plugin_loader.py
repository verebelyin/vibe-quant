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

Plugin name collisions with a built-in are logged as a warning. The
behavior on collision is "last write wins" because ``register_spec``
unconditionally overwrites the registry entry — documented here so
future debuggers know where to look.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def load_builtin_plugins() -> list[str]:
    """Import every non-underscore module under ``vibe_quant.dsl.plugins``.

    Returns:
        List of fully-qualified module names that were successfully
        imported. Callers don't usually need this; the side effect (spec
        registration) is the whole point. Tests consume the return value
        to assert which plugins loaded.
    """
    # Import the package lazily so callers (e.g., tests) that want to
    # monkey-patch the package ``__path__`` before the first load can do
    # so without a stale reference.
    from vibe_quant.dsl import plugins
    from vibe_quant.dsl.indicators import indicator_registry

    loaded: list[str] = []
    pre_existing = set(indicator_registry.list_indicators())

    for module_info in pkgutil.iter_modules(plugins.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        qualified = f"{plugins.__name__}.{name}"
        try:
            importlib.import_module(qualified)
        except Exception as exc:  # pragma: no cover - defensive catch
            # Plugins are third-party / experimental code; any import
            # failure is logged and swallowed so the rest of the
            # registry stays usable.
            logger.error(
                "Failed to load indicator plugin %s: %s", qualified, exc
            )
            continue
        loaded.append(qualified)
        logger.info("Loaded indicator plugin: %s", qualified)

    # Collision smoke check: any built-in whose compute_fn's module now
    # lives inside a plugin we just loaded has been silently overwritten.
    # We can't fully prove the overwrite (register_spec doesn't track
    # history) but we can flag it when it happens via module path.
    plugin_module_prefix = plugins.__name__ + "."
    for ind_name in pre_existing:
        spec = indicator_registry.get(ind_name)
        if spec is None:  # shouldn't happen but be defensive
            continue
        fn = spec.compute_fn
        if fn is not None and fn.__module__.startswith(plugin_module_prefix):
            logger.warning(
                "Plugin overwrote built-in indicator %r (now sourced from %s)",
                ind_name,
                fn.__module__,
            )

    return loaded
