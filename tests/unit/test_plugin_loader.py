"""Tests for the indicator plugin auto-loader.

Covers the four behaviors the P6 loader has to guarantee:

1. It discovers and imports every non-underscore ``.py`` file in the
   target plugins package.
2. Files prefixed with an underscore are skipped.
3. A plugin that raises on import is logged and swallowed, not raised.
4. A plugin whose spec collides with a built-in triggers a warning
   (but the overwrite itself is allowed, "last write wins").

The tests use ``tmp_path`` + ``monkeypatch.syspath_prepend`` to mount a
fake plugins package so the real ``vibe_quant/dsl/plugins/`` directory
stays untouched.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import TYPE_CHECKING

import pytest

from vibe_quant.dsl import plugin_loader
from vibe_quant.dsl.indicators import indicator_registry

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(autouse=True)
def _restore_plugin_registry_state():
    """Snapshot and restore the registry + plugin-tracking state.

    ``reload_plugins()`` in particular will unregister whatever is in
    ``_plugin_registered_names``, which at module import time got
    populated from the REAL plugin dir. Tests in this file monkeypatch
    the plugins package to a throwaway fake, so without this fixture a
    ``reload_plugins()`` call would unregister real-world KAMA/VIDYA/
    FRAMA/ADAPTIVE_RSI specs and never restore them — poisoning later
    tests in the sweep.
    """
    saved_registry = dict(indicator_registry._indicators)  # noqa: SLF001
    saved_names = set(plugin_loader._plugin_registered_names)  # noqa: SLF001
    saved_errors = list(plugin_loader._load_errors)  # noqa: SLF001
    # Snapshot cached plugin modules too — reload_plugins() evicts them
    # from sys.modules, which leaks across tests: a later test that
    # imports e.g. ``vibe_quant.dsl.plugins.example_adaptive_rsi`` would
    # re-execute its module body and collide on ADAPTIVE_RSI since we
    # restore the registry entry below. Restore both or neither.
    plugin_prefixes = (
        "vibe_quant.dsl.plugins.",
        "vibe_quant.dsl.plugins_ext.",
    )
    saved_modules = {
        name: mod for name, mod in sys.modules.items()
        if name.startswith(plugin_prefixes)
    }
    try:
        yield
    finally:
        indicator_registry._indicators.clear()  # noqa: SLF001
        indicator_registry._indicators.update(saved_registry)  # noqa: SLF001
        plugin_loader._plugin_registered_names.clear()  # noqa: SLF001
        plugin_loader._plugin_registered_names.update(saved_names)  # noqa: SLF001
        plugin_loader._load_errors.clear()  # noqa: SLF001
        plugin_loader._load_errors.extend(saved_errors)  # noqa: SLF001
        for mod_name in [m for m in sys.modules if m.startswith(plugin_prefixes)]:
            del sys.modules[mod_name]
        sys.modules.update(saved_modules)


# ---------------------------------------------------------------------------
# Fake-package scaffolding
# ---------------------------------------------------------------------------


def _install_fake_plugin_pkg(
    tmp_path: Path, module_name: str, monkeypatch: MonkeyPatch
) -> Path:
    """Create ``<tmp>/<module_name>/__init__.py`` and wire it into sys.path.

    Also point ``plugin_loader``'s internal ``vibe_quant.dsl.plugins``
    import at this throwaway package so ``load_builtin_plugins()`` walks
    the fake directory instead of the real one.

    Returns:
        The path to the fake package directory. Callers drop ``.py``
        files in here to simulate plugins.
    """
    pkg_dir = tmp_path / module_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        '"""Fake plugins package for test."""\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    # Build a synthetic ``vibe_quant.dsl.plugins`` module whose __path__
    # points at the temp directory. Patch it into BOTH sys.modules AND
    # the parent package attribute — the loader does
    # ``from vibe_quant.dsl import plugins``, which reads the attribute
    # off ``vibe_quant.dsl`` rather than going through sys.modules.
    fake_pkg = types.ModuleType("vibe_quant.dsl.plugins")
    fake_pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    fake_pkg.__name__ = "vibe_quant.dsl.plugins"
    monkeypatch.setitem(sys.modules, "vibe_quant.dsl.plugins", fake_pkg)

    import vibe_quant.dsl as _dsl_pkg

    monkeypatch.setattr(_dsl_pkg, "plugins", fake_pkg)

    return pkg_dir


# ---------------------------------------------------------------------------
# 1. Discovery of drop-in files
# ---------------------------------------------------------------------------


def test_load_builtin_plugins_discovers_dropin_files(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A ``.py`` file dropped into the plugins package is auto-imported."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_discover", monkeypatch)
    (pkg_dir / "my_plugin.py").write_text(
        'MARKER = "loaded-by-test_load_builtin_plugins_discovers_dropin_files"\n'
    )

    loaded = plugin_loader.load_builtin_plugins()

    assert "vibe_quant.dsl.plugins.my_plugin" in loaded
    mod = sys.modules.get("vibe_quant.dsl.plugins.my_plugin")
    assert mod is not None
    assert getattr(mod, "MARKER", None) == (
        "loaded-by-test_load_builtin_plugins_discovers_dropin_files"
    )


# ---------------------------------------------------------------------------
# 2. Underscore-prefixed files are skipped
# ---------------------------------------------------------------------------


def test_underscore_prefixed_files_are_skipped(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """``_helper.py`` must not be auto-imported (reserved for internal use)."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_underscore", monkeypatch)
    (pkg_dir / "_private_helper.py").write_text("MARKER = 'should-not-load'\n")
    (pkg_dir / "visible_plugin.py").write_text("MARKER = 'loaded'\n")

    loaded = plugin_loader.load_builtin_plugins()

    assert "vibe_quant.dsl.plugins.visible_plugin" in loaded
    assert "vibe_quant.dsl.plugins._private_helper" not in loaded
    assert "vibe_quant.dsl.plugins._private_helper" not in sys.modules


# ---------------------------------------------------------------------------
# 3. Broken plugin is logged, not raised
# ---------------------------------------------------------------------------


def test_plugin_exception_is_logged_and_swallowed(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A plugin that raises on import must not crash the loader.

    One working plugin + one broken plugin in the same directory: the
    working one must still load, and the loader must return normally
    with the broken plugin's error logged.
    """
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_broken", monkeypatch)
    (pkg_dir / "broken.py").write_text(
        "raise RuntimeError('intentional failure from test')\n"
    )
    (pkg_dir / "working.py").write_text("MARKER = 'ok'\n")

    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)
    with caplog.at_level(logging.ERROR, logger="vibe_quant.dsl.plugin_loader"):
        loaded = plugin_loader.load_builtin_plugins()

    assert "vibe_quant.dsl.plugins.working" in loaded
    assert "vibe_quant.dsl.plugins.broken" not in loaded
    # Error log must mention the broken module so debugging isn't a
    # guessing game.
    assert any(
        "broken" in record.getMessage() for record in caplog.records
    ), "Expected a log entry referencing the broken plugin"


# ---------------------------------------------------------------------------
# 5. Failed loads are surfaced via get_load_errors()
# ---------------------------------------------------------------------------


def test_failed_plugin_is_recorded_in_load_errors(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A broken plugin must be queryable via ``get_load_errors()`` so the
    catalog API can surface it to the frontend."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_errors", monkeypatch)
    (pkg_dir / "broken.py").write_text(
        "raise ValueError('boom from test_failed_plugin_is_recorded')\n"
    )
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()
    errors = plugin_loader.get_load_errors()

    assert len(errors) == 1
    err = errors[0]
    assert err.module == "vibe_quant.dsl.plugins.broken"
    assert err.error_type == "ValueError"
    assert "boom from test_failed_plugin_is_recorded" in err.message


def test_load_errors_are_cleared_between_runs(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Two consecutive ``load_builtin_plugins()`` calls must not
    accumulate errors — the second run snapshots the current state."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_clear", monkeypatch)
    (pkg_dir / "broken.py").write_text("raise RuntimeError('first')\n")
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()
    assert len(plugin_loader.get_load_errors()) == 1

    # Remove broken plugin and rerun — the list should be empty.
    (pkg_dir / "broken.py").unlink()
    plugin_loader.load_builtin_plugins()
    assert plugin_loader.get_load_errors() == []


# ---------------------------------------------------------------------------
# 6. VQ_PLUGINS_STRICT re-raises on broken plugin
# ---------------------------------------------------------------------------


def test_strict_mode_reraises_plugin_load_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """``VQ_PLUGINS_STRICT=1`` converts the log-and-swallow behavior into
    a hard failure — CI-friendly."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_strict", monkeypatch)
    (pkg_dir / "broken.py").write_text(
        "raise RuntimeError('strict-mode failure')\n"
    )
    monkeypatch.setenv("VQ_PLUGINS_STRICT", "1")

    import pytest

    with pytest.raises(RuntimeError, match="strict-mode failure"):
        plugin_loader.load_builtin_plugins()


# ---------------------------------------------------------------------------
# 7. reload_plugins picks up edits without polluting the registry
# ---------------------------------------------------------------------------


def test_reload_plugins_picks_up_edited_spec(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Edit a plugin file on disk, call ``reload_plugins()``, and the
    updated spec must replace the previous one without re-raising a
    collision error."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_reload", monkeypatch)
    (pkg_dir / "my_ind.py").write_text(
        "from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n"
        "def _c(df, params):  # noqa: ARG001\n"
        "    return df['close']\n"
        "indicator_registry.register_spec(\n"
        "    IndicatorSpec(\n"
        "        name='RELOAD_TEST',\n"
        "        nt_class=None,\n"
        "        pandas_ta_func=None,\n"
        "        default_params={'period': 10},\n"
        "        param_schema={'period': int},\n"
        "        compute_fn=_c,\n"
        "    )\n"
        ")\n"
    )
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()
    first = indicator_registry.get("RELOAD_TEST")
    assert first is not None
    assert first.default_params == {"period": 10}

    # Rewrite the plugin with a different default period. reload_plugins
    # wipes the __pycache__ directory so stale .pyc bytecode can't be
    # served even on filesystems whose mtime resolution is too coarse to
    # distinguish two writes within the same millisecond.
    plugin_path = pkg_dir / "my_ind.py"
    plugin_path.write_text(
        "from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n"
        "def _c(df, params):  # noqa: ARG001\n"
        "    return df['close']\n"
        "indicator_registry.register_spec(\n"
        "    IndicatorSpec(\n"
        "        name='RELOAD_TEST',\n"
        "        nt_class=None,\n"
        "        pandas_ta_func=None,\n"
        "        default_params={'period': 42},\n"
        "        param_schema={'period': int},\n"
        "        compute_fn=_c,\n"
        "    )\n"
        ")\n"
    )

    plugin_loader.reload_plugins()
    second = indicator_registry.get("RELOAD_TEST")
    assert second is not None
    assert second.default_params == {"period": 42}
    # Clean up so other tests don't see a leaked spec.
    indicator_registry.unregister("RELOAD_TEST")


def test_reload_plugins_does_not_touch_builtins(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Built-in specs must survive a plugin reload — only
    plugin-registered names get unregistered."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_reload_builtins", monkeypatch)
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    real_rsi = indicator_registry.get("RSI")
    assert real_rsi is not None

    plugin_loader.reload_plugins()

    assert indicator_registry.get("RSI") is real_rsi


# ---------------------------------------------------------------------------
# 8. VQ_PLUGIN_PATH discovery
# ---------------------------------------------------------------------------


def test_vq_plugin_path_loads_external_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A ``.py`` file under a directory listed in VQ_PLUGIN_PATH is
    loaded and registers its spec just like a built-in plugin."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_envpath_builtin", monkeypatch)

    ext_dir = tmp_path / "ext_plugins"
    ext_dir.mkdir()
    (ext_dir / "external_ind.py").write_text(
        "from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n"
        "def _c(df, params):  # noqa: ARG001\n"
        "    return df['close']\n"
        "indicator_registry.register_spec(\n"
        "    IndicatorSpec(\n"
        "        name='EXT_IND',\n"
        "        nt_class=None,\n"
        "        pandas_ta_func=None,\n"
        "        default_params={'period': 7},\n"
        "        param_schema={'period': int},\n"
        "        compute_fn=_c,\n"
        "    )\n"
        ")\n"
    )
    monkeypatch.setenv("VQ_PLUGIN_PATH", str(ext_dir))
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()

    spec = indicator_registry.get("EXT_IND")
    assert spec is not None
    assert spec.default_params == {"period": 7}


def test_vq_plugin_path_multiple_dirs_with_colon_separator(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Two directories separated by os.pathsep both get scanned."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_envpath_multi", monkeypatch)

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    for dir_path, name in [(dir_a, "EXT_A"), (dir_b, "EXT_B")]:
        (dir_path / "mod.py").write_text(
            "from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n"
            "def _c(df, params):  # noqa: ARG001\n"
            "    return df['close']\n"
            "indicator_registry.register_spec(\n"
            "    IndicatorSpec(\n"
            f"        name='{name}',\n"
            "        nt_class=None,\n"
            "        pandas_ta_func=None,\n"
            "        default_params={'period': 7},\n"
            "        param_schema={'period': int},\n"
            "        compute_fn=_c,\n"
            "    )\n"
            ")\n"
        )
    monkeypatch.setenv(
        "VQ_PLUGIN_PATH", f"{dir_a}{os.pathsep}{dir_b}"
    )
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()

    assert indicator_registry.get("EXT_A") is not None
    assert indicator_registry.get("EXT_B") is not None


def test_vq_plugin_path_invalid_dir_logs_warning(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A non-existent VQ_PLUGIN_PATH entry logs a warning but doesn't
    take down the rest of the load."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_envpath_bad", monkeypatch)

    monkeypatch.setenv(
        "VQ_PLUGIN_PATH", str(tmp_path / "does_not_exist")
    )
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    with caplog.at_level(
        logging.WARNING, logger="vibe_quant.dsl.plugin_loader"
    ):
        plugin_loader.load_builtin_plugins()

    assert any(
        "not a directory" in r.getMessage() for r in caplog.records
    )


def test_vq_plugin_path_broken_file_records_load_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A broken external plugin shows up in get_load_errors()."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_envpath_broken", monkeypatch)

    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "broken_ext.py").write_text(
        "raise RuntimeError('external failure')\n"
    )
    monkeypatch.setenv("VQ_PLUGIN_PATH", str(ext_dir))
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()

    errors = plugin_loader.get_load_errors()
    assert any(
        "broken_ext" in e.module and e.error_type == "RuntimeError"
        for e in errors
    ), f"Expected external plugin failure in load_errors, got: {errors}"


# ---------------------------------------------------------------------------
# 9. entry_points discovery
# ---------------------------------------------------------------------------


def test_entry_points_plugin_is_loaded(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """An entry-point registered under 'vibe_quant.indicators' is loaded
    — simulated here by stubbing ``importlib.metadata.entry_points``."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_ep_builtin", monkeypatch)

    registrations: list[str] = []

    def _register_via_entry_point() -> None:
        from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

        def _c(df, params):  # noqa: ARG001, ANN001
            return df["close"]

        indicator_registry.register_spec(
            IndicatorSpec(
                name="EP_IND",
                nt_class=None,
                pandas_ta_func=None,
                default_params={"period": 3},
                param_schema={"period": int},
                compute_fn=_c,
            )
        )
        registrations.append("EP_IND")

    class _FakeEP:
        name = "my_ep"
        value = "stub:_register_via_entry_point"

        def load(self):  # noqa: ANN202
            return _register_via_entry_point

    def _fake_entry_points(*, group: str):  # noqa: ANN202
        if group == "vibe_quant.indicators":
            return [_FakeEP()]
        return []

    monkeypatch.setattr(
        "importlib.metadata.entry_points", _fake_entry_points
    )
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()

    assert registrations == ["EP_IND"]
    assert indicator_registry.get("EP_IND") is not None


def test_entry_points_load_failure_is_recorded(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """If an entry-point raises on load, the error is logged and
    surfaced via get_load_errors() without taking down the rest."""
    _install_fake_plugin_pkg(tmp_path, "_p6test_ep_fail", monkeypatch)

    class _BrokenEP:
        name = "broken_ep"
        value = "stub:raises"

        def load(self):  # noqa: ANN202
            msg = "entry-point load boom"
            raise RuntimeError(msg)

    def _fake_entry_points(*, group: str):  # noqa: ANN202
        if group == "vibe_quant.indicators":
            return [_BrokenEP()]
        return []

    monkeypatch.setattr(
        "importlib.metadata.entry_points", _fake_entry_points
    )
    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)

    plugin_loader.load_builtin_plugins()

    errors = plugin_loader.get_load_errors()
    assert any(
        "broken_ep" in e.module and e.error_type == "RuntimeError"
        for e in errors
    ), f"Expected broken entry-point in load_errors, got: {errors}"


# ---------------------------------------------------------------------------
# 4. Plugin that overwrites a built-in emits a warning
# ---------------------------------------------------------------------------


def test_plugin_cannot_overwrite_builtin_silently(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A plugin that calls ``register_spec(spec)`` with a colliding name
    raises KeyError and is therefore logged as a load failure — not
    silently shadowed. The error is surfaced via ``get_load_errors()``."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_shadow", monkeypatch)
    (pkg_dir / "shadow_rsi.py").write_text(
        "from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n"
        "\n"
        "def _shadow_compute(df, params):  # noqa: ARG001\n"
        "    return df['close']\n"
        "\n"
        "indicator_registry.register_spec(\n"
        "    IndicatorSpec(\n"
        "        name='RSI',\n"
        "        nt_class=None,\n"
        "        pandas_ta_func=None,\n"
        "        default_params={'period': 14},\n"
        "        param_schema={'period': int},\n"
        "        compute_fn=_shadow_compute,\n"
        "    )\n"
        ")\n"
    )

    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)
    real_rsi = indicator_registry.get("RSI")
    assert real_rsi is not None
    with caplog.at_level(
        logging.ERROR, logger="vibe_quant.dsl.plugin_loader"
    ):
        plugin_loader.load_builtin_plugins()

    # The shadow plugin should have failed to load and be recorded as an
    # error. RSI must remain the built-in.
    errors = plugin_loader.get_load_errors()
    assert any(
        e.module.endswith("shadow_rsi") and e.error_type == "KeyError"
        for e in errors
    ), f"Expected KeyError on shadow_rsi, got: {errors}"
    assert indicator_registry.get("RSI") is real_rsi


def test_plugin_can_override_builtin_with_explicit_flag(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """``override=True`` opts into shadowing a built-in; the registry
    logs the override at INFO so it's auditable."""
    pkg_dir = _install_fake_plugin_pkg(tmp_path, "_p6test_override", monkeypatch)
    (pkg_dir / "override_rsi.py").write_text(
        "from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry\n"
        "\n"
        "def _custom_compute(df, params):  # noqa: ARG001\n"
        "    return df['close']\n"
        "\n"
        "indicator_registry.register_spec(\n"
        "    IndicatorSpec(\n"
        "        name='RSI',\n"
        "        nt_class=None,\n"
        "        pandas_ta_func=None,\n"
        "        default_params={'period': 14},\n"
        "        param_schema={'period': int},\n"
        "        compute_fn=_custom_compute,\n"
        "    ),\n"
        "    override=True,\n"
        ")\n"
    )

    monkeypatch.delenv("VQ_PLUGINS_STRICT", raising=False)
    real_rsi = indicator_registry.get("RSI")
    assert real_rsi is not None
    try:
        with caplog.at_level(
            logging.INFO, logger="vibe_quant.dsl.indicators"
        ):
            plugin_loader.load_builtin_plugins()

        shadowed = indicator_registry.get("RSI")
        assert shadowed is not None
        assert shadowed is not real_rsi, "RSI should have been overridden"
        assert any(
            "RSI" in r.getMessage() and "overriding" in r.getMessage()
            for r in caplog.records
        ), "Expected INFO log recording the override"
    finally:
        indicator_registry.register_spec(real_rsi, override=True)
