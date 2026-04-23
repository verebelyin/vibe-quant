"""Best-effort auto-importer for indicator plugins.

Discovers plugins from three sources on every ``load_builtin_plugins``
call:

1. ``vibe_quant.dsl.plugins`` (built-in drop-in directory) —
   ``pkgutil.iter_modules`` over every non-underscore ``.py`` file.
2. ``$VQ_PLUGIN_PATH`` — colon-separated directories (``:`` on POSIX,
   ``;`` on Windows follows ``os.pathsep``). Every ``.py`` file in each
   directory is loaded via ``importlib.util.spec_from_file_location``
   under a unique ``vibe_quant.dsl.plugins_ext.<dir>.<stem>`` name so
   cached modules never collide across directories.
3. Python ``entry_points`` group ``vibe_quant.indicators`` — a
   third-party pip package can publish::

       [project.entry-points."vibe_quant.indicators"]
       my_ind = "my_pkg.my_ind"

   and the module will be imported at startup. The side effect
   (``register_spec``) is the whole point — the entry point target can
   be a module or a callable.

Each source is responsible for calling ``indicator_registry.register(name)``
or ``register_spec(spec)`` at module scope — the loader just triggers
the import.

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
import importlib.metadata
import importlib.util
import logging
import os
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PluginLoadError:
    """One plugin that failed to import, with the error message."""

    module: str
    error_type: str
    message: str


_load_errors: list[PluginLoadError] = []

# Names of indicator specs registered during the most recent plugin load.
# Used by :func:`reload_plugins` to know which registry entries to
# unregister before reimporting. Excludes built-ins (registered before
# ``load_builtin_plugins`` ever runs).
_plugin_registered_names: set[str] = set()


def _strict_mode() -> bool:
    """Return True iff ``VQ_PLUGINS_STRICT`` is set to a truthy value."""
    return os.environ.get("VQ_PLUGINS_STRICT", "").lower() in ("1", "true", "yes")


def get_load_errors() -> list[PluginLoadError]:
    """Return a copy of the most recent load-error list."""
    return list(_load_errors)


def get_plugin_registered_names() -> set[str]:
    """Return a copy of indicator names registered by the last plugin load."""
    return set(_plugin_registered_names)


def reload_plugins() -> list[str]:
    """Unregister plugin-registered specs and reimport every plugin module.

    Useful for a dev-mode live-reload loop: edit a plugin file, POST to
    ``/api/indicators/reload``, and the registry picks up the new spec
    without a backend restart. Built-in specs are left untouched.

    Returns:
        List of fully-qualified module names that successfully reloaded.
    """
    from vibe_quant.dsl import plugins
    from vibe_quant.dsl.indicators import indicator_registry

    # Unregister prior plugin specs so collision-checks don't fire on
    # reimport.
    for name in list(_plugin_registered_names):
        indicator_registry.unregister(name)
    _plugin_registered_names.clear()

    # Evict plugin modules from sys.modules so the next import actually
    # re-executes the module body (plain ``import`` is a no-op once a
    # module is cached). We evict proactively rather than calling
    # ``importlib.reload`` because reload keeps stale module-level state
    # around that can cause spurious collisions. Covers:
    #   vibe_quant.dsl.plugins.*          — drop-in directory
    #   vibe_quant.dsl.plugins_ext.*      — VQ_PLUGIN_PATH directories
    # Entry-point targets aren't under a known prefix, so we skip them
    # here — ``importlib.reload`` semantics for external packages are
    # the caller's responsibility.
    for prefix in (plugins.__name__ + ".", "vibe_quant.dsl.plugins_ext."):
        for mod_name in [m for m in sys.modules if m.startswith(prefix)]:
            del sys.modules[mod_name]

    # Drop importlib's finder/path caches too — without this, a freshly
    # edited file on disk may be served from the previous import's
    # cached bytecode path.
    importlib.invalidate_caches()

    # Wipe any ``__pycache__`` directories inside the plugin search path
    # so a same-second edit can't be masked by stale .pyc bytecode. This
    # only runs on explicit reload (not the normal startup path), so the
    # filesystem churn is acceptable.
    import shutil as _shutil
    from pathlib import Path as _Path

    for search_root in plugins.__path__:
        cache_dir = _Path(search_root) / "__pycache__"
        if cache_dir.is_dir():
            _shutil.rmtree(cache_dir, ignore_errors=True)

    return load_builtin_plugins()


def _record_failure(
    qualified: str, exc: BaseException, strict: bool
) -> None:
    """Log + record + (optionally) re-raise a plugin-load exception."""
    logger.error("Failed to load indicator plugin %s: %s", qualified, exc)
    _load_errors.append(
        PluginLoadError(
            module=qualified,
            error_type=type(exc).__name__,
            message=str(exc),
        )
    )
    if strict:
        raise exc


def _import_file_as_module(path: Path, qualified_name: str) -> None:
    """Import a ``.py`` file at an arbitrary path under a synthetic name.

    Used for VQ_PLUGIN_PATH discovery — the files aren't inside a
    Python package, so we can't use ``importlib.import_module``.
    Registering the module in ``sys.modules`` under a synthetic name
    under ``vibe_quant.dsl.plugins_ext.*`` makes it identifiable for
    reload + eviction.
    """
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not build import spec for {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise


def _discover_env_path_plugins(
    indicator_registry_module: object, strict: bool, loaded: list[str]
) -> None:
    """Load every ``.py`` under each ``VQ_PLUGIN_PATH`` directory."""
    raw = os.environ.get("VQ_PLUGIN_PATH", "").strip()
    if not raw:
        return
    from vibe_quant.dsl.indicators import indicator_registry

    for raw_dir in raw.split(os.pathsep):
        dir_path = Path(raw_dir).expanduser()
        if not dir_path.is_dir():
            logger.warning(
                "VQ_PLUGIN_PATH entry is not a directory: %s", dir_path
            )
            continue
        # Derive a stable per-directory namespace so plugins in two
        # different VQ_PLUGIN_PATH dirs with the same filename don't
        # collide in sys.modules.
        dir_tag = "_".join(
            filter(None, str(dir_path.resolve()).replace(":", "_").split("/"))
        ) or "root"
        for py_file in sorted(dir_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            qualified = f"vibe_quant.dsl.plugins_ext.{dir_tag}.{py_file.stem}"
            before = set(indicator_registry.list_indicators())
            try:
                _import_file_as_module(py_file, qualified)
            except Exception as exc:
                _record_failure(qualified, exc, strict)
                continue
            after = set(indicator_registry.list_indicators())
            _plugin_registered_names.update(after - before)
            loaded.append(qualified)
            logger.info(
                "Loaded VQ_PLUGIN_PATH indicator plugin: %s (%s)",
                qualified,
                py_file,
            )


def _discover_entry_point_plugins(strict: bool, loaded: list[str]) -> None:
    """Load every plugin published under the ``vibe_quant.indicators``
    entry-point group."""
    from vibe_quant.dsl.indicators import indicator_registry

    try:
        entries = importlib.metadata.entry_points(group="vibe_quant.indicators")
    except TypeError:
        # Py <3.10 used a different API; we target 3.13 so this path is
        # only a defensive fallback.
        entries = importlib.metadata.entry_points().get(  # type: ignore[attr-defined]
            "vibe_quant.indicators", []
        )

    for ep in entries:
        qualified = f"entry_point:{ep.name}={ep.value}"
        before = set(indicator_registry.list_indicators())
        try:
            # ``ep.load()`` imports the target module or callable. If it's
            # a callable, we call it so it can register specs imperatively.
            obj = ep.load()
            if callable(obj):
                obj()
        except Exception as exc:
            _record_failure(qualified, exc, strict)
            continue
        after = set(indicator_registry.list_indicators())
        _plugin_registered_names.update(after - before)
        loaded.append(qualified)
        logger.info("Loaded entry-point indicator plugin: %s", qualified)


def load_builtin_plugins() -> list[str]:
    """Import plugin modules from built-in dir, VQ_PLUGIN_PATH, entry points.

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
    from vibe_quant.dsl.indicators import indicator_registry

    _load_errors.clear()
    loaded: list[str] = []
    strict = _strict_mode()

    # Source 1: built-in drop-in directory.
    for module_info in pkgutil.iter_modules(plugins.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        qualified = f"{plugins.__name__}.{name}"
        before = set(indicator_registry.list_indicators())
        try:
            importlib.import_module(qualified)
        except Exception as exc:
            _record_failure(qualified, exc, strict)
            continue
        after = set(indicator_registry.list_indicators())
        _plugin_registered_names.update(after - before)
        loaded.append(qualified)
        logger.info("Loaded indicator plugin: %s", qualified)

    # Source 2: VQ_PLUGIN_PATH directories.
    _discover_env_path_plugins(indicator_registry, strict, loaded)

    # Source 3: Python entry-point group.
    _discover_entry_point_plugins(strict, loaded)

    return loaded
