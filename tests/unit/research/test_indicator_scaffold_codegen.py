"""Unit tests for slice 2 of bd-3p1k.1 — codegen + AST safety + file write.

The claude-p subprocess is mocked by monkeypatching the module-level
``_run_claude_codegen``; the AST gate / render / mypy / ruff steps are
exercised against the real subprocess invocations against a tmp path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.research import indicator_scaffold
from vibe_quant.research.indicator_scaffold import (
    CodegenError,
    IndicatorSpecArgs,
    _ast_safety_check,
    proposed_to_spec_args,
    render_plugin_file,
    synthesize_and_write,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------- AST safety gate ----------


def _good_body() -> str:
    return (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        '    import pandas as pd\n'
        '    period = int(params.get("period", 14) or 14)\n'
        '    return df["close"].rolling(period).mean()\n'
    )


def test_ast_check_accepts_well_formed_body() -> None:
    _ast_safety_check(_good_body(), "compute_my_ind")


def test_ast_check_rejects_syntax_error() -> None:
    with pytest.raises(CodegenError) as e:
        _ast_safety_check("def x(:\n", "compute_my_ind")
    assert e.value.code == "syntax_error"


def test_ast_check_rejects_non_function_top_level() -> None:
    with pytest.raises(CodegenError) as e:
        _ast_safety_check("x = 5\n", "compute_my_ind")
    assert e.value.code == "non_function"


def test_ast_check_rejects_empty_body() -> None:
    with pytest.raises(CodegenError) as e:
        _ast_safety_check("", "compute_my_ind")
    assert e.value.code == "non_function"


def test_ast_check_rejects_wrong_function_name() -> None:
    body = (
        "def compute_other(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    return df\n"
    )
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "missing_signature"


def test_ast_check_rejects_missing_annotations() -> None:
    body = "def compute_my_ind(df, params):\n    return df\n"
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "missing_signature"


def test_ast_check_rejects_banned_import_os() -> None:
    body = (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    import os\n"
        "    return df['close']\n"
    )
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "banned_import"
    assert e.value.detail == "os"


def test_ast_check_rejects_banned_subprocess() -> None:
    body = (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    from subprocess import run\n"
        "    run(['ls'])\n"
        "    return df['close']\n"
    )
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "banned_import"


def test_ast_check_rejects_exec_call() -> None:
    body = (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    exec('print(1)')\n"
        "    return df['close']\n"
    )
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "banned_call"
    assert e.value.detail == "exec"


def test_ast_check_rejects_eval_call_in_nested_scope() -> None:
    # Banned calls must be caught even when buried inside a helper —
    # ast.walk traverses the whole subtree, not just top-level statements.
    body = (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    def helper():\n"
        "        return eval('1+1')\n"
        "    return df['close']\n"
    )
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "banned_call"


def test_ast_check_rejects_multiple_top_level_statements() -> None:
    body = (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    return df['close']\n"
        "\n"
        "x = 5\n"
    )
    with pytest.raises(CodegenError) as e:
        _ast_safety_check(body, "compute_my_ind")
    assert e.value.code == "non_function"


# ---------- render_plugin_file ----------


def _spec(name: str = "MY_IND") -> IndicatorSpecArgs:
    return proposed_to_spec_args(
        {
            "name": name.lower(),
            "formula": "ema(close, period)",
            "parameters": {"period": {"default": 14, "range": [5, 30]}},
            "output_range": "0..100",
        }
    )


def test_render_plugin_file_contains_required_header() -> None:
    rendered = render_plugin_file(
        _spec(), _good_body(), extraction_id=42, source_quote="from u/x"
    )
    assert "AUTO-GENERATED FROM EXTRACTION 42" in rendered
    assert "RANGES: period=llm" in rendered  # range came from LLM dict


def test_render_plugin_file_includes_register_spec_call() -> None:
    rendered = render_plugin_file(_spec(), _good_body(), extraction_id=1)
    assert "indicator_registry.register_spec" in rendered
    assert "name='MY_IND'" in rendered
    assert "compute_fn=compute_my_ind" in rendered


def test_render_plugin_file_threshold_range_from_output_hint() -> None:
    rendered = render_plugin_file(_spec(), _good_body(), extraction_id=1)
    # output_range "0..100" → threshold_range (20.0, 80.0)
    assert "threshold_range=(20.0, 80.0)" in rendered


def test_render_plugin_file_no_threshold_when_unbounded() -> None:
    spec = proposed_to_spec_args(
        {
            "name": "u",
            "formula": "f",
            "parameters": {"period": {"default": 14}},
            "output_range": "unbounded",
        }
    )
    rendered = render_plugin_file(spec, _good_body(), extraction_id=1)
    assert "threshold_range=None" in rendered


# ---------- synthesize_and_write end-to-end ----------


def _isolate_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "plugins"
    target.mkdir()
    monkeypatch.setattr(indicator_scaffold, "PLUGINS_DIR", target)
    return target


def test_synthesize_writes_file_when_body_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _isolate_plugins(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", lambda _p: _good_body()
    )
    # mypy + ruff are subprocesses — keep them green by stubbing too,
    # the goal of this test is to verify the happy path wiring, not to
    # validate the toolchain (covered separately).
    monkeypatch.setattr(indicator_scaffold, "run_mypy", lambda _p: (True, ""))
    monkeypatch.setattr(indicator_scaffold, "run_ruff", lambda _p: (True, ""))

    path = synthesize_and_write(
        _spec(), formula="ema(close, period)", extraction_id=7, source_quote="q"
    )
    assert path == plugins / "proposed_my_ind.py"
    assert path.read_text(encoding="utf-8").startswith('"""AUTO-GENERATED')


def test_synthesize_raises_banned_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_plugins(tmp_path, monkeypatch)
    bad = (
        "def compute_my_ind(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        "    import os\n"
        "    return df['close']\n"
    )
    monkeypatch.setattr(indicator_scaffold, "_run_claude_codegen", lambda _p: bad)
    with pytest.raises(CodegenError) as e:
        synthesize_and_write(
            _spec(), formula="f", extraction_id=1, source_quote=None
        )
    assert e.value.code == "banned_import"
    # No file should have been written.
    assert not (tmp_path / "plugins" / "proposed_my_ind.py").exists()


def test_synthesize_raises_syntax_error_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_plugins(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", lambda _p: "def x(:\n"
    )
    with pytest.raises(CodegenError) as e:
        synthesize_and_write(
            _spec(), formula="f", extraction_id=1, source_quote=None
        )
    assert e.value.code == "syntax_error"
    assert not (tmp_path / "plugins" / "proposed_my_ind.py").exists()


def test_synthesize_raises_non_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_plugins(tmp_path, monkeypatch)
    monkeypatch.setattr(indicator_scaffold, "_run_claude_codegen", lambda _p: "x = 5\n")
    with pytest.raises(CodegenError) as e:
        synthesize_and_write(
            _spec(), formula="f", extraction_id=1, source_quote=None
        )
    assert e.value.code == "non_function"


def test_synthesize_deletes_file_on_mypy_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _isolate_plugins(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", lambda _p: _good_body()
    )
    monkeypatch.setattr(
        indicator_scaffold, "run_mypy", lambda _p: (False, "incompatible types")
    )
    monkeypatch.setattr(indicator_scaffold, "run_ruff", lambda _p: (True, ""))

    with pytest.raises(CodegenError) as e:
        synthesize_and_write(
            _spec(), formula="f", extraction_id=1, source_quote=None
        )
    assert e.value.code == "mypy_fail"
    assert not (plugins / "proposed_my_ind.py").exists()


def test_synthesize_deletes_file_on_ruff_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins = _isolate_plugins(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", lambda _p: _good_body()
    )
    monkeypatch.setattr(indicator_scaffold, "run_mypy", lambda _p: (True, ""))
    monkeypatch.setattr(
        indicator_scaffold, "run_ruff", lambda _p: (False, "F401 unused import")
    )

    with pytest.raises(CodegenError) as e:
        synthesize_and_write(
            _spec(), formula="f", extraction_id=1, source_quote=None
        )
    assert e.value.code == "ruff_fail"
    assert not (plugins / "proposed_my_ind.py").exists()


def test_synthesize_raises_empty_body() -> None:
    spec = _spec()
    with pytest.raises(CodegenError) as e:
        synthesize_and_write(
            spec,
            formula="f",
            extraction_id=1,
            source_quote=None,
            runner=lambda _p: "   ",
        )
    assert e.value.code == "empty_body"


def test_run_claude_codegen_maps_timeout_to_codegen_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 60s subprocess timeout must surface as code=timeout, not bare TimeoutExpired."""
    import subprocess

    from vibe_quant.research.indicator_scaffold import _run_claude_codegen

    def fake_run(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="claude", timeout=60)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/claude")
    with pytest.raises(CodegenError) as e:
        _run_claude_codegen("prompt")
    assert e.value.code == "timeout"
