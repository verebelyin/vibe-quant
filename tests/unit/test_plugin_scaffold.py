"""Tests for the indicator plugin scaffolding CLI."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.dsl.plugin_scaffold import (
    VALID_CATEGORIES,
    _normalize_name,
    main,
    scaffold_plugin,
)


def test_normalize_name_uppercases() -> None:
    upper, lower = _normalize_name("my_ind")
    assert upper == "MY_IND"
    assert lower == "my_ind"


def test_normalize_name_rejects_invalid_start() -> None:
    with pytest.raises(ValueError, match="must start with a letter"):
        _normalize_name("1bad")


def test_normalize_name_rejects_dashes() -> None:
    with pytest.raises(ValueError, match="must start with a letter"):
        _normalize_name("my-ind")


def test_scaffold_plugin_creates_both_files(tmp_path: Path) -> None:
    written = scaffold_plugin(
        "my_test_ind",
        category="Momentum",
        plugin_dir=tmp_path / "plugins",
        tests_dir=tmp_path / "tests",
    )
    assert len(written) == 2
    plugin_path, test_path = written
    assert plugin_path.name == "my_test_ind.py"
    assert test_path.name == "test_my_test_ind.py"
    assert plugin_path.exists()
    assert test_path.exists()

    plugin_src = plugin_path.read_text()
    assert 'name="MY_TEST_IND"' in plugin_src
    assert "def compute_my_test_ind(" in plugin_src
    assert 'category="Momentum"' in plugin_src

    test_src = test_path.read_text()
    assert 'indicator_registry.get("MY_TEST_IND")' in test_src
    assert "def test_my_test_ind_registered()" in test_src


def test_scaffold_plugin_skip_tests(tmp_path: Path) -> None:
    written = scaffold_plugin(
        "only_plugin",
        plugin_dir=tmp_path / "plugins",
        tests_dir=tmp_path / "tests",
        skip_tests=True,
    )
    assert len(written) == 1
    assert not (tmp_path / "tests" / "test_only_plugin.py").exists()


def test_scaffold_plugin_refuses_existing_without_force(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "existing.py").write_text("# placeholder\n")

    with pytest.raises(FileExistsError):
        scaffold_plugin(
            "existing",
            plugin_dir=plugin_dir,
            tests_dir=tmp_path / "tests",
        )


def test_scaffold_plugin_force_overwrites(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "existing.py").write_text("# placeholder\n")

    scaffold_plugin(
        "existing",
        plugin_dir=plugin_dir,
        tests_dir=tmp_path / "tests",
        force=True,
    )
    assert 'name="EXISTING"' in (plugin_dir / "existing.py").read_text()


def test_scaffold_plugin_rejects_bad_category(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid category"):
        scaffold_plugin(
            "bad_cat",
            category="NotARealCategory",
            plugin_dir=tmp_path / "plugins",
            tests_dir=tmp_path / "tests",
        )


def test_main_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "new",
            "MAIN_IND",
            "--plugin-dir",
            str(tmp_path / "plugins"),
            "--tests-dir",
            str(tmp_path / "tests"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "main_ind.py" in out
    assert "test_main_ind.py" in out


def test_main_reports_existing_file_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "dup.py").write_text("")

    rc = main(
        [
            "new",
            "DUP",
            "--plugin-dir",
            str(tmp_path / "plugins"),
            "--tests-dir",
            str(tmp_path / "tests"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "already exists" in err


def test_scaffold_output_imports_cleanly(tmp_path: Path) -> None:
    """The scaffold should produce a syntactically valid, importable file."""
    import importlib.util

    written = scaffold_plugin(
        "smoke_ind",
        plugin_dir=tmp_path / "plugins",
        tests_dir=tmp_path / "tests",
    )
    plugin_path = written[0]

    spec = importlib.util.spec_from_file_location(
        "_scaffold_smoke_test", plugin_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    from vibe_quant.dsl.indicators import indicator_registry

    indicator_registry.unregister("SMOKE_IND")
    try:
        spec.loader.exec_module(module)
        registered = indicator_registry.get("SMOKE_IND")
        assert registered is not None
        assert registered.compute_fn is not None
    finally:
        indicator_registry.unregister("SMOKE_IND")


def test_scaffold_generated_test_passes(tmp_path: Path) -> None:
    """The generated test stub should pass end-to-end in a subprocess.

    Uses ``VQ_PLUGIN_PATH`` so the plugin file in the tmp dir is
    auto-discovered and registered when the subprocess imports
    ``vibe_quant.dsl``.
    """
    import os
    import sys

    written = scaffold_plugin(
        "selftest_ind",
        plugin_dir=tmp_path / "plugins",
        tests_dir=tmp_path / "tests",
    )
    _plugin_path, test_path = written

    env = os.environ.copy()
    env["VQ_PLUGIN_PATH"] = str(tmp_path / "plugins")
    env["VQ_PLUGINS_STRICT"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Generated test failed.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_valid_categories_are_reasonable() -> None:
    assert "Trend" in VALID_CATEGORIES
    assert "Custom" in VALID_CATEGORIES
