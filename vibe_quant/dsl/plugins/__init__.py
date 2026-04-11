"""Drop-in directory for custom indicator plugins.

Every non-underscore ``.py`` file in this directory is auto-imported at
package load time via ``vibe_quant.dsl.plugin_loader.load_builtin_plugins``,
which is called once at the bottom of ``vibe_quant/dsl/indicators.py``
after the built-in specs have all registered. Each plugin module should
use ``indicator_registry.register(name)`` or ``register_spec(spec)`` at
module scope; the loader itself holds no references to the modules.

Files prefixed with an underscore are skipped (useful for helpers or
work-in-progress plugins you don't want auto-loaded).

Plugin loading is best-effort: a plugin that raises on import logs an
error but does NOT crash the process, so the built-in registry stays
available even if one plugin is broken.

See ``example_adaptive_rsi.py`` (Phase 9) for a fully-worked example of
a plugin that declares ``compute_fn``, ``param_ranges``, and
``threshold_range`` to auto-enroll in the GA indicator pool and the
frontend catalog API.
"""
