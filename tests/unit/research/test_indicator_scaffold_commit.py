"""Unit tests for slice 3 of bd-3p1k.1 — contract test gen + pytest + commit.

Subprocesses (pytest, git) are monkeypatched so the suite stays
hermetic. A focused integration test against a real ``git init`` repo
covers the happy commit path end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.research import indicator_scaffold
from vibe_quant.research.indicator_scaffold import (
    CO_AUTHOR_TRAILER,
    ScaffoldError,
    ScaffoldResult,
    git_commit_scaffold,
    proposed_to_spec_args,
    render_contract_test,
    run_contract_test,
    scaffold_full,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------- render_contract_test ----------


def _spec(name: str = "MY_IND", outputs: list[str] | None = None):
    proposal = {
        "name": name.lower(),
        "formula": "ema(close, period)",
        "parameters": {"period": {"default": 14, "range": [5, 30]}},
        "output_range": "0..100",
    }
    if outputs:
        proposal["outputs"] = outputs
    return proposed_to_spec_args(proposal)


def test_render_contract_test_single_output_has_three_tests() -> None:
    src = render_contract_test(_spec())
    assert "def test_my_ind_registered" in src
    assert "def test_my_ind_contract_length_and_index" in src
    assert "def test_my_ind_not_all_nan_past_warmup" in src
    # Single-output uses the Series template (no dict assertion).
    assert "isinstance(out, pd.Series)" in src
    assert "isinstance(out, dict)" not in src


def test_render_contract_test_multi_output_uses_dict_branch() -> None:
    src = render_contract_test(_spec("DUAL", outputs=["fast", "slow"]))
    assert "def test_dual_contract_dict_and_alignment" in src
    assert "isinstance(out, dict)" in src
    assert "('fast', 'slow')" in src


def test_render_contract_test_compiles_as_python() -> None:
    import ast

    src = render_contract_test(_spec("COMPILES"))
    ast.parse(src)  # no SyntaxError


# ---------- run_contract_test ----------


def test_run_contract_test_returns_passed_true_on_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    class FakeProc:
        returncode = 0
        stdout = "1 passed"
        stderr = ""

    monkeypatch.setattr(sp, "run", lambda *_a, **_kw: FakeProc())
    ok, output = run_contract_test(tmp_path / "test_x.py")
    assert ok is True
    assert "passed" in output


def test_run_contract_test_returns_passed_false_on_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    class FakeProc:
        returncode = 1
        stdout = "FAILED test_foo"
        stderr = "AssertionError"

    monkeypatch.setattr(sp, "run", lambda *_a, **_kw: FakeProc())
    ok, output = run_contract_test(tmp_path / "test_x.py")
    assert ok is False
    assert "FAILED" in output
    assert "AssertionError" in output


def test_run_contract_test_truncates_output_to_2kb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    class FakeProc:
        returncode = 1
        stdout = "x" * 5000
        stderr = ""

    monkeypatch.setattr(sp, "run", lambda *_a, **_kw: FakeProc())
    _, output = run_contract_test(tmp_path / "test_x.py")
    assert len(output) <= 2048


def test_run_contract_test_timeout_returns_passed_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    def boom(*_a, **_kw):
        raise sp.TimeoutExpired(cmd="pytest", timeout=30)

    monkeypatch.setattr(sp, "run", boom)
    ok, output = run_contract_test(tmp_path / "test_x.py", timeout_seconds=30)
    assert ok is False
    assert "timed out" in output.lower()


# ---------- git_commit_scaffold (real git, integration) ----------


def _init_repo(tmp_path: Path) -> Path:
    import subprocess as sp

    sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    # An initial commit so HEAD exists and pre-commit hooks (if any) can
    # diff against a parent.
    (tmp_path / "README").write_text("seed\n")
    sp.run(["git", "-C", str(tmp_path), "add", "README"], check=True)
    sp.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "seed"],
        check=True,
    )
    return tmp_path


def test_git_commit_scaffold_creates_one_commit_with_both_files(
    tmp_path: Path,
) -> None:
    import subprocess as sp

    repo = _init_repo(tmp_path)
    plugin = repo / "plugin.py"
    test = repo / "test_plugin.py"
    plugin.write_text("# plugin\n")
    test.write_text("# test\n")

    sha = git_commit_scaffold(plugin, test, name="MY_IND", repo_root=repo)
    assert len(sha) == 40  # full SHA

    # Commit message exact
    msg = sp.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert msg == "chore: scaffold proposed indicator MY_IND (bd-3p1k)"

    # Co-Authored-By trailer
    body = sp.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert CO_AUTHOR_TRAILER in body

    # Both files in the commit
    stat = sp.run(
        ["git", "-C", str(repo), "show", "--stat", "--format=", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "plugin.py" in stat
    assert "test_plugin.py" in stat

    # NOT pushed — branch ahead by 1 (or no upstream at all in this fresh repo).
    status = sp.run(
        ["git", "-C", str(repo), "status", "--porcelain=v2", "--branch"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # No "+0" upstream tracking → at minimum no remote push happened.
    assert "branch.upstream" not in status  # fresh repo has none


def test_git_commit_scaffold_reverts_staging_on_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    plugin = repo / "plugin.py"
    test = repo / "test_plugin.py"
    plugin.write_text("# plugin\n")
    test.write_text("# test\n")

    # Mock _git so 'add' succeeds, 'commit' fails, 'restore' is called.
    calls: list[list[str]] = []
    real_git = indicator_scaffold._git

    def fake_git(*args, cwd=None):
        calls.append(list(args))
        if args[0] == "commit":
            return 1, "", "pre-commit hook rejected"
        return real_git(*args, cwd=cwd)

    monkeypatch.setattr(indicator_scaffold, "_git", fake_git)

    with pytest.raises(ScaffoldError) as excinfo:
        git_commit_scaffold(plugin, test, name="X", repo_root=repo)
    assert excinfo.value.code == "commit_failed"
    assert "pre-commit" in excinfo.value.output

    # A 'restore --staged' call must have happened to unstage the files.
    assert any(a[0] == "restore" and "--staged" in a for a in calls)


# ---------- scaffold_full orchestrator ----------


def _good_body_for(prompt: str) -> str:
    import re

    m = re.search(r"def (compute_[a-z0-9_]+)\(", prompt)
    fn_name = m.group(1) if m else "compute_x"
    return (
        f"def {fn_name}(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:\n"
        f'    import pandas as pd\n'
        f'    period_raw = params.get("period", 14)\n'
        f'    period = int(period_raw) if isinstance(period_raw, (int, float)) else 14\n'
        f'    return df["close"].rolling(period).mean()\n'
    )


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    tests = tmp_path / "tests_out"
    tests.mkdir()
    monkeypatch.setattr(indicator_scaffold, "PLUGINS_DIR", plugins)
    monkeypatch.setattr(indicator_scaffold, "TESTS_DIR", tests)
    monkeypatch.setattr(indicator_scaffold, "run_mypy", lambda _p: (True, ""))
    monkeypatch.setattr(indicator_scaffold, "run_ruff", lambda _p: (True, ""))
    return plugins, tests


def test_scaffold_full_happy_path_returns_result_and_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins, tests = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", _good_body_for
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "run_contract_test",
        lambda _p, **_kw: (True, "ok"),
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "git_commit_scaffold",
        lambda _pp, _tp, *, name, repo_root=None: "deadbeef" * 5,  # 40-char fake
    )

    result = scaffold_full(
        _spec(),
        formula="ema(close, period)",
        extraction_id=1,
        source_quote="q",
    )
    assert isinstance(result, ScaffoldResult)
    assert result.plugin_path == plugins / "proposed_my_ind.py"
    assert result.test_path == tests / "test_proposed_my_ind.py"
    assert result.commit_sha == "deadbeef" * 5
    assert result.plugin_path.exists()
    assert result.test_path.exists()


def test_scaffold_full_test_failure_deletes_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins, tests = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", _good_body_for
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "run_contract_test",
        lambda _p, **_kw: (False, "FAILED test_my_ind_contract"),
    )

    with pytest.raises(ScaffoldError) as excinfo:
        scaffold_full(
            _spec(),
            formula="f",
            extraction_id=1,
            source_quote=None,
        )
    assert excinfo.value.code == "test_failed"
    assert "FAILED" in excinfo.value.output
    assert not (plugins / "proposed_my_ind.py").exists()
    assert not (tests / "test_proposed_my_ind.py").exists()


def test_scaffold_full_commit_failure_deletes_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugins, tests = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold, "_run_claude_codegen", _good_body_for
    )
    monkeypatch.setattr(
        indicator_scaffold,
        "run_contract_test",
        lambda _p, **_kw: (True, ""),
    )

    def fail_commit(*_a, **_kw):
        raise ScaffoldError("commit_failed", "pre-commit hook rejected")

    monkeypatch.setattr(indicator_scaffold, "git_commit_scaffold", fail_commit)

    with pytest.raises(ScaffoldError) as excinfo:
        scaffold_full(
            _spec(),
            formula="f",
            extraction_id=1,
            source_quote=None,
        )
    assert excinfo.value.code == "commit_failed"
    assert not (plugins / "proposed_my_ind.py").exists()
    assert not (tests / "test_proposed_my_ind.py").exists()


def test_scaffold_full_codegen_failure_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codegen-stage failure must NOT swallow into ScaffoldError —
    the endpoint distinguishes codegen_failed from test_failed."""
    plugins, tests = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        indicator_scaffold,
        "_run_claude_codegen",
        lambda _p: "def x(:\n",  # syntax error
    )
    from vibe_quant.research.indicator_scaffold import CodegenError

    with pytest.raises(CodegenError) as excinfo:
        scaffold_full(
            _spec(),
            formula="f",
            extraction_id=1,
            source_quote=None,
        )
    assert excinfo.value.code == "syntax_error"
    assert not (plugins / "proposed_my_ind.py").exists()
    assert not (tests / "test_proposed_my_ind.py").exists()
