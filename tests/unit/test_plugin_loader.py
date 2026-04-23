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
import sys
import types
from typing import TYPE_CHECKING

from vibe_quant.dsl import plugin_loader
from vibe_quant.dsl.indicators import indicator_registry

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


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
