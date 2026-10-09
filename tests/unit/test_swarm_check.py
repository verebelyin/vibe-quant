"""Tests for scripts/agents/swarm_check.py — the pre-review gate helper.

Pure functions and orchestration run against a throwaway git repo in tmp_path; the
tool subprocesses (ruff/mypy/pytest/exactness/frontend) are stubbed with a fake
runner, so no real tools ever run. One integration-ish test drives the .sh wrapper
with SWARM_CHECK_RUFF/SWARM_CHECK_MYPY/SWARM_CHECK_PYTEST stubs instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "swarm_check.py"
_spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
assert _spec is not None and _spec.loader is not None
swarm_check = importlib.util.module_from_spec(_spec)
sys.modules["swarm_check"] = swarm_check
_spec.loader.exec_module(swarm_check)


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "config", "user.email", "gate@test")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "keep.py").write_text("X = 1\n")
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 1\n")
    tests = repo / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_foo.py").write_text("def test_foo() -> None:\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


class FakeRunner:
    """Delegates git to the real binary; answers tool calls from `responses`.

    `responses` maps a needle matched against any command argument to either a
    (rc, output) tuple or a callable(cmd, cwd) -> (rc, output). First match wins.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[int] = []
        self.envs: list[Mapping[str, str] | None] = []
        self.responses: dict[
            str, tuple[int, str] | Callable[[Sequence[str], Path], tuple[int, str]]
        ] = {}

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int = 600,
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        self.calls.append(c)
        self.timeouts.append(timeout)
        self.envs.append(env)
        if c[0] == "git":
            proc = subprocess.run(c, cwd=str(cwd), capture_output=True, text=True)
            return proc.returncode, proc.stdout + proc.stderr
        for needle, resp in self.responses.items():
            if any(needle in Path(arg).name for arg in c):
                if callable(resp):
                    return resp(c, Path(cwd))
                return resp
        return 0, ""


def _called_with(runner: FakeRunner, needle: str) -> bool:
    return any(needle in Path(arg).name for cmd in runner.calls for arg in cmd)


def _cfg(tmp_path: Path, repo: Path, **kwargs: object) -> object:
    defaults: dict[str, object] = {
        "worktree": repo,
        "main_root": tmp_path,
        "base": "main",
    }
    defaults.update(kwargs)
    return swarm_check.Config(**defaults)  # type: ignore[arg-type]


def _main_exactness(tmp_path: Path, repo: Path | None = None, text: str = "# main\n") -> Path:
    """Main checkout's exactness script (cfg.main_root == tmp_path); optionally mirrored."""
    script = tmp_path / "scripts" / "agents" / "exactness_239.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text)
    if repo is not None:
        (repo / "scripts" / "agents").mkdir(parents=True, exist_ok=True)
        (repo / "scripts" / "agents" / "exactness_239.py").write_text(text)
    return script


class TestScope:
    def test_offenders_listed(self) -> None:
        offenders = swarm_check.scope_offenders(
            ["vibe_quant/a.py", "docs/x.md", "README.md"], ["vibe_quant/*", "tests/*"]
        )
        assert offenders == ["docs/x.md", "README.md"]

    def test_all_match(self) -> None:
        assert swarm_check.scope_offenders(
            ["vibe_quant/a.py", "tests/unit/test_x.py"], ["vibe_quant/*", "tests/*"]
        ) == []

    def test_empty_scope_matches_nothing(self) -> None:
        assert swarm_check.scope_offenders(["a.py"], []) == ["a.py"]


class TestGitState:
    def test_diff_uncommitted_untracked(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-qb", "feature")
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "change foo")
        (repo / "vibe_quant" / "keep.py").write_text("X = 2\n")  # uncommitted
        (repo / "stray.txt").write_text("x\n")  # untracked
        (repo / "cmd-report.md").write_text("report\n")  # untracked, excluded
        state = swarm_check.git_state(repo, "main", FakeRunner())
        assert "vibe_quant/foo.py" in state.changed  # committed on the branch
        assert "vibe_quant/keep.py" in state.changed  # uncommitted
        assert "stray.txt" in state.changed  # untracked
        assert "cmd-report.md" not in state.changed  # untracked report excluded
        assert state.untracked == ["stray.txt"]

    def test_rename_reports_both_paths(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _git(repo, "mv", "vibe_quant/foo.py", "vibe_quant/bar.py")
        state = swarm_check.git_state(repo, "HEAD", FakeRunner())
        assert state.error == ""
        assert "vibe_quant/bar.py" in state.changed
        assert "vibe_quant/foo.py" in state.changed

    def test_rename_old_path_out_of_scope_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _git(repo, "mv", "vibe_quant/foo.py", "vibe_quant/bar.py")
        cfg = _cfg(tmp_path, repo, base="HEAD", scope=("vibe_quant/bar.py",),
                   exactness="never", frontend="never")
        scope = {c.name: c for c in swarm_check.run_checks(cfg, FakeRunner())}["scope"]
        assert scope.status == "FAIL" and "vibe_quant/foo.py" in scope.detail

    def test_non_ascii_path_unquoted(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "caf\u00e9.py").write_text("X = 1\n")
        state = swarm_check.git_state(repo, "HEAD", FakeRunner())
        assert "vibe_quant/caf\u00e9.py" in state.changed

    def test_bad_base_is_error(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        state = swarm_check.git_state(repo, "mian", FakeRunner())
        assert state.error.startswith("bad base ref: mian")
        assert state.changed == []

    def test_git_diff_failure_is_error(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)

        def run(cmd: Sequence[str], cwd: Path, env: object = None, timeout: int = 1) -> tuple[int, str]:
            if "diff" in cmd:
                return 129, "usage: git diff"
            return FakeRunner()(cmd, cwd)

        state = swarm_check.git_state(repo, "main", run)
        assert "git diff failed" in state.error and state.changed == []

    def test_git_status_failure_is_error(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)

        def run(cmd: Sequence[str], cwd: Path, env: object = None, timeout: int = 1) -> tuple[int, str]:
            if "status" in cmd:
                return 128, "fatal: boom"
            return FakeRunner()(cmd, cwd)

        state = swarm_check.git_state(repo, "main", run)
        assert "git status failed" in state.error


class TestTestSelection:
    def test_k_expr_from_source_basenames(self) -> None:
        expr = swarm_check.k_expr(
            ["vibe_quant/discovery/fitness.py", "vibe_quant/data/kama.py"]
        )
        assert expr == "fitness or kama"

    def test_k_expr_dedupes_and_skips_tests(self) -> None:
        expr = swarm_check.k_expr(
            [
                "vibe_quant/screening/fitness.py",
                "vibe_quant/discovery/fitness.py",
                "tests/unit/test_fitness.py",
            ]
        )
        assert expr == "fitness"

    def test_changed_test_files(self) -> None:
        files = swarm_check.changed_test_files(
            ["tests/unit/test_a.py", "tests/api/test_b.py", "tests/conftest.py", "vibe_quant/x.py"]
        )
        assert files == ["tests/unit/test_a.py", "tests/api/test_b.py"]

    def test_k_expr_drops_init_conftest_and_odd_stems(self) -> None:
        expr = swarm_check.k_expr(
            [
                "vibe_quant/__init__.py",
                "vibe_quant/__main__.py",
                "tests/conftest.py",
                "vibe_quant/my file.py",
                "vibe_quant/real.py",
            ]
        )
        assert expr == "real"

    def test_no_selection_without_python(self) -> None:
        assert swarm_check.k_expr(["frontend/App.tsx"]) == ""
        assert swarm_check.changed_test_files(["frontend/App.tsx"]) == []


class TestAutoTriggers:
    def test_exactness_dirs(self) -> None:
        for p in [
            "vibe_quant/dsl/parser.py",
            "vibe_quant/screening/pipeline.py",
            "vibe_quant/discovery/fitness.py",
            "vibe_quant/validation/gate.py",
            "vibe_quant/data/catalog.py",
            "vibe_quant/db/schema.py",  # inverted rule: anything not allowlisted
            "vibe_quant/overfitting/wfa.py",
            "vibe_quant/errors.py",
            "vibe_quant/metrics.py",
        ]:
            assert swarm_check.should_run_exactness([p]), p

    def test_exactness_extra_files(self) -> None:
        for p in [
            "vibe_quant/nt_compat.py",
            "vibe_quant/utils.py",
            "pyproject.toml",
            "uv.lock",
            "scripts/agents/exactness_239.py",
        ]:
            assert swarm_check.should_run_exactness([p]), p

    def test_exactness_not_elsewhere(self) -> None:
        for p in [
            "vibe_quant/api/app.py",
            "vibe_quant/paper/guard.py",
            "vibe_quant/research/x.py",
            "vibe_quant/risk/limits.py",
            "frontend/x.ts",
            "tests/unit/test_x.py",
        ]:
            assert not swarm_check.should_run_exactness([p]), p

    def test_frontend_trigger(self) -> None:
        assert swarm_check.should_run_frontend(["frontend/src/App.tsx"])
        assert not swarm_check.should_run_frontend(["vibe_quant/api/app.py"])


class TestLint:
    def test_parse_error_counts(self) -> None:
        out = "src/a.ts check\nFound 5 errors.\nFound 7 warnings.\n"
        assert swarm_check.parse_lint_errors(out) == 5

    def test_count_anchored(self) -> None:
        assert swarm_check.parse_lint_errors("see: Found 9 errors. in log\n") == 0
        assert not swarm_check.has_lint_count("lint blew up")

    def test_parse_zero(self) -> None:
        assert swarm_check.parse_lint_errors("Found 2 warnings.\n") == 0

    def test_regression_only_when_increased(self) -> None:
        assert swarm_check.lint_regressed("Found 5 errors.", "Found 3 errors.")
        assert not swarm_check.lint_regressed("Found 3 errors.", "Found 3 errors.")
        assert not swarm_check.lint_regressed("Found 1 error.", "Found 3 errors.")


class TestAggregation:
    def test_gate_pass_on_pass_and_skip(self) -> None:
        checks = [
            swarm_check.Check("scope", "SKIP", "no --scope given"),
            swarm_check.Check("ruff", "PASS", "ok"),
        ]
        gate, n = swarm_check.gate_summary(checks)
        assert (gate, n) == ("GATE PASS", 0)

    def test_warn_detail_does_not_fail(self) -> None:
        checks = [swarm_check.Check("clean", "WARN", "1 untracked: stray.txt")]
        gate, n = swarm_check.gate_summary(checks)
        assert (gate, n) == ("GATE PASS", 0)

    def test_gate_fail_counts(self) -> None:
        checks = [
            swarm_check.Check("ruff", "FAIL", "x"),
            swarm_check.Check("tests", "FAIL", "y"),
            swarm_check.Check("mypy", "PASS", "ok"),
        ]
        gate, n = swarm_check.gate_summary(checks)
        assert (gate, n) == ("GATE FAIL (2 failed)", 2)

    def test_format_line(self) -> None:
        line = swarm_check.format_line(swarm_check.Check("ruff", "PASS", "All checks passed!"))
        assert line == "PASS ruff: All checks passed!"


class TestJson:
    def test_structure(self) -> None:
        checks = [
            swarm_check.Check("scope", "SKIP", "no --scope given"),
            swarm_check.Check("ruff", "FAIL", "boom"),
        ]
        data = swarm_check.to_json_dict(checks)
        assert data["scope"] == {"status": "SKIP", "detail": "no --scope given"}
        assert data["ruff"] == {"status": "FAIL", "detail": "boom"}
        assert data["gate"] == "FAIL"

    def test_gate_pass_value(self) -> None:
        data = swarm_check.to_json_dict([swarm_check.Check("ruff", "PASS", "ok")])
        assert data["gate"] == "PASS"


class TestRunChecks:
    def test_skips_when_disabled(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        runner = FakeRunner()
        runner.responses = {"ruff": (0, "All checks passed!"), "mypy": (0, "Success")}
        (repo / "notes.txt").write_text("x\n")  # non-empty diff
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by_name = {c.name: c for c in checks}
        assert [c.name for c in checks] == [
            "scope", "clean", "ruff", "mypy", "tests", "exactness", "frontend",
        ]
        assert by_name["scope"].status == "SKIP"
        assert by_name["tests"].status == "SKIP"
        assert by_name["exactness"].status == "SKIP"
        assert by_name["frontend"].status == "SKIP"
        assert not _called_with(runner, "pytest")

    def test_scope_fail_lists_offenders(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        runner = FakeRunner()
        (repo / "docs").mkdir()
        (repo / "docs" / "x.md").write_text("x\n")
        cfg = _cfg(tmp_path, repo, scope=("vibe_quant/*",), exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        scope = {c.name: c for c in checks}["scope"]
        assert scope.status == "FAIL"
        assert "docs/x.md" in scope.detail

    def test_clean_warns_but_passes(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "stray.txt").write_text("x\n")
        (repo / "cmd-report.md").write_text("r\n")
        runner = FakeRunner()
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        clean = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["clean"]
        assert clean.status == "WARN"
        assert swarm_check.format_line(clean).startswith("WARN clean:")
        assert "stray.txt" in clean.detail
        assert "cmd-report.md" not in clean.detail

    def test_ruff_failure_marks_check(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        runner = FakeRunner()
        runner.responses = {"ruff": (1, "E501 line too long"), "mypy": (0, "Success")}
        (repo / "notes.txt").write_text("x\n")  # non-empty diff
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by_name = {c.name: c for c in checks}
        assert by_name["ruff"].status == "FAIL"
        assert swarm_check.gate_summary(checks) == ("GATE FAIL (1 failed)", 1)

    def test_tests_selection_changed_test_and_module(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "discovery").mkdir()
        (repo / "vibe_quant" / "discovery" / "fitness.py").write_text("F = 1\n")
        (repo / "tests" / "unit" / "test_bar.py").write_text("def test_bar() -> None:\n    pass\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "2 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["tests"].status == "PASS"
        pytest_cmds = [" ".join(c) for c in runner.calls if any("pytest" in Path(a).name for a in c)]
        assert any("tests/unit/test_bar.py" in c for c in pytest_cmds)
        assert any("tests/unit -k fitness" in c for c in pytest_cmds)

    def test_tests_failure_marks_check(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (1, "1 failed, 1 passed in 0.02s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        tests = {c.name: c for c in checks}["tests"]
        assert tests.status == "FAIL"
        assert "1 failed" in tests.detail

    def test_exactness_auto_runs_on_dsl_change(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "dsl").mkdir()
        (repo / "vibe_quant" / "dsl" / "parser.py").write_text("P = 1\n")
        main_script = _main_exactness(tmp_path, repo)
        runner = FakeRunner()
        runner.responses = {"exactness": (0, "IDENTICAL")}
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        ex = {c.name: c for c in checks}["exactness"]
        assert ex.status == "PASS" and "IDENTICAL" in ex.detail
        # always the MAIN checkout's script, never the worktree copy
        assert [str(main_script)] in [c[-1:] for c in runner.calls]
        ex_envs = [e for c, e in zip(runner.calls, runner.envs, strict=True) if c[-1] == str(main_script)]
        assert ex_envs and ex_envs[0] is not None
        assert ex_envs[0]["PYTHONPATH"] == str(repo)  # exercises the WORKTREE's code
        assert str(repo / "scripts" / "agents" / "exactness_239.py") not in sum(runner.calls, [])

    def test_exactness_drift_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "data").mkdir()
        (repo / "vibe_quant" / "data" / "catalog.py").write_text("C = 1\n")
        _main_exactness(tmp_path, repo)
        runner = FakeRunner()
        runner.responses = {"exactness": (1, "DRIFT: sharpe 1.3 -> 1.4")}
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["exactness"].status == "FAIL"

    def test_exactness_identical_but_nonzero_rc_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_exactness(tmp_path, repo)
        runner = FakeRunner()
        runner.responses = {"exactness": (1, "IDENTICAL (then crashed)")}
        cfg = _cfg(tmp_path, repo, exactness="always", frontend="never", base="HEAD")
        ex = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["exactness"]
        assert ex.status == "FAIL"

    def test_exactness_modified_script_in_worktree_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_exactness(tmp_path, repo)
        (repo / "scripts" / "agents" / "exactness_239.py").write_text("EXPECTED_SHARPE = 9\n")
        runner = FakeRunner()
        runner.responses = {"exactness": (0, "IDENTICAL")}
        cfg = _cfg(tmp_path, repo, exactness="always", frontend="never", base="HEAD")
        ex = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["exactness"]
        assert ex.status == "FAIL" and "modified in worktree" in ex.detail
        assert not _called_with(runner, "exactness_239.py")

    def test_exactness_missing_main_script_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        cfg = _cfg(tmp_path, repo, exactness="always", frontend="never", base="HEAD")
        ex = {c.name: c for c in swarm_check.run_checks(cfg, FakeRunner())}["exactness"]
        assert ex.status == "FAIL" and "missing" in ex.detail

    def test_exactness_skips_without_trigger(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "api").mkdir()
        (repo / "vibe_quant" / "api" / "app.py").write_text("A = 1\n")
        runner = FakeRunner()
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["exactness"].status == "SKIP"
        assert not _called_with(runner, "exactness_239.py")

    def test_frontend_lint_regression_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "frontend").mkdir()
        (repo / "frontend" / "App.tsx").write_text("export {}\n")
        main_fe = tmp_path / "frontend"
        main_fe.mkdir()

        def lint(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            if "build" in cmd:
                return 0, "built"
            if "lint" in cmd:
                return 1, "Found 5 errors." if cwd == repo / "frontend" else "Found 3 errors."
            return 0, "ok"

        runner = FakeRunner()
        runner.responses = {"tsc": (0, ""), "pnpm": lint}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never")
        checks = swarm_check.run_checks(cfg, runner)
        fe = {c.name: c for c in checks}["frontend"]
        assert fe.status == "FAIL"
        assert "5" in fe.detail and "3" in fe.detail

    def test_frontend_ok_when_lint_flat(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "frontend").mkdir()
        (repo / "frontend" / "App.tsx").write_text("export {}\n")
        (tmp_path / "frontend").mkdir()
        runner = FakeRunner()
        runner.responses = {"tsc": (0, ""), "pnpm": (0, "Found 3 errors.")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["frontend"].status == "PASS"

    def _fe_status(
        self, tmp_path: Path, responses: dict[str, object]
    ) -> tuple[str, str]:
        repo = _init_repo(tmp_path)
        (tmp_path / "frontend").mkdir()
        runner = FakeRunner()
        runner.responses = responses  # type: ignore[assignment]
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="always")
        fe = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["frontend"]
        return fe.status, fe.detail

    def test_frontend_tsc_failure_fails(self, tmp_path: Path) -> None:
        status, detail = self._fe_status(tmp_path, {"tsc": (1, "TS2322"), "pnpm": (0, "")})
        assert status == "FAIL" and "tsc" in detail

    def test_frontend_build_failure_fails(self, tmp_path: Path) -> None:
        def pnpm(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            return (1, "build exploded") if "build" in cmd else (0, "")

        status, detail = self._fe_status(tmp_path, {"tsc": (0, ""), "pnpm": pnpm})
        assert status == "FAIL" and "build" in detail

    def test_frontend_lint_crash_without_count_fails(self, tmp_path: Path) -> None:
        status, detail = self._fe_status(tmp_path, {"tsc": (0, ""), "pnpm": (1, "biome: crashed")})
        assert status == "FAIL" and "without an error count" in detail

    def test_frontend_always_runs_without_changes(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (tmp_path / "frontend").mkdir()
        runner = FakeRunner()
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="always")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["frontend"].status == "PASS"
        assert _called_with(runner, "tsc")


class TestFailClosed:
    def test_bad_base_fails_gate(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 5\n")
        _git(repo, "commit", "-qam", "change")
        runner = FakeRunner()
        cfg = _cfg(tmp_path, repo, base="mian", exactness="auto", frontend="auto")
        checks = swarm_check.run_checks(cfg, runner)
        by = {c.name: c for c in checks}
        assert by["scope"].status == "FAIL" and "bad base ref: mian" in by["scope"].detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_bad_base_script_exit_1(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SWARM_")}
        proc = subprocess.run(
            ["bash", str(_SCRIPT.parent / "swarm-check.sh"), str(repo), "--base", "mian",
             "--exactness", "never", "--frontend", "never"],
            capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 1, proc.stdout
        assert "FAIL scope: bad base ref: mian" in proc.stdout
        assert "GATE FAIL" in proc.stdout.strip().splitlines()[-1]

    def test_pytest_rc5_on_k_group_is_skip_not_fail(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (5, "5 deselected in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "SKIP" and "no tests matched -k foo" in tests.detail

    def test_pytest_rc5_on_changed_test_file_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_bar.py").write_text("X = 1\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (5, "no tests ran")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "FAIL"

    def test_real_run_timeout(self, tmp_path: Path) -> None:
        rc, out = swarm_check.real_run(["sleep", "5"], tmp_path, None, 1)
        assert rc != 0 and "timed out" in out

    def test_real_run_missing_binary(self, tmp_path: Path) -> None:
        rc, out = swarm_check.real_run(["/nonexistent/bin/tool"], tmp_path, None, 5)
        assert rc != 0 and "cannot run" in out

    def test_missing_binary_still_yields_gate(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)  # main_root=tmp_path has no .venv -> real_run hits ENOENT
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, swarm_check.real_run)
        by = {c.name: c for c in checks}
        assert by["ruff"].status == "FAIL" and "cannot run" in by["ruff"].detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_tests_and_exactness_get_long_timeout(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_exactness(tmp_path, repo)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "1 passed"), "exactness": (0, "IDENTICAL")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="always", frontend="never")
        swarm_check.run_checks(cfg, runner)
        assert swarm_check.TEST_TIMEOUT in runner.timeouts

    def test_stubs_refused_without_allow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _init_repo(tmp_path)
        monkeypatch.delenv("SWARM_CHECK_ALLOW_STUBS", raising=False)
        monkeypatch.setenv("SWARM_CHECK_RUFF", "/bin/true")
        runner = FakeRunner()
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        stubs = {c.name: c for c in checks}["stubs"]
        assert stubs.status == "FAIL"
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")
        assert not any(c[0] == "/bin/true" for c in runner.calls)  # real tool used instead

    def test_stubs_allowed_are_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _init_repo(tmp_path)
        monkeypatch.setenv("SWARM_CHECK_ALLOW_STUBS", "1")
        monkeypatch.setenv("SWARM_CHECK_RUFF", "/bin/true")
        runner = FakeRunner()
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        stubs = {c.name: c for c in checks}["stubs"]
        assert stubs.status == "WARN" and "NOTE stub active: SWARM_CHECK_RUFF=/bin/true" in stubs.detail
        assert any(c[0] == "/bin/true" for c in runner.calls)
        assert swarm_check.to_json_dict(checks)["stubs"]["status"] == "WARN"

    def test_empty_diff_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, FakeRunner())
        scope = {c.name: c for c in checks}["scope"]
        assert scope.status == "FAIL" and "empty diff vs HEAD" in scope.detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    @pytest.mark.parametrize("path", ["tests/conftest.py", "vibe_quant/__init__.py"])
    def test_conftest_or_init_only_runs_full_unit_suite(self, tmp_path: Path, path: str) -> None:
        repo = _init_repo(tmp_path)
        (repo / path).write_text("X = 1\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "9 passed")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        pytest_cmds = [c for c in runner.calls if any("pytest" in Path(a).name for a in c)]
        assert [c[c.index("pytest") + 1:] if "pytest" in c else c[-1:] for c in pytest_cmds] == [["tests/unit"]]

    def test_internal_crash_is_fail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _init_repo(tmp_path)

        def boom(cfg: object) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(swarm_check, "run_checks", boom)
        out_json = tmp_path / "o.json"
        rc = swarm_check.main([str(repo), "--json", str(out_json)])
        assert rc == 1
        data = json.loads(out_json.read_text())
        assert data["internal"]["status"] == "FAIL" and "kaboom" in data["internal"]["detail"]
        assert data["gate"] == "FAIL"

    def test_json_clean_is_warn(self) -> None:
        data = swarm_check.to_json_dict([swarm_check.Check("clean", "WARN", "1 untracked: x")])
        assert data["clean"]["status"] == "WARN" and data["gate"] == "PASS"


class TestIntegration:
    def _repo_with_stubs(self, tmp_path: Path) -> tuple[Path, dict[str, str]]:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        (repo / "stray.txt").write_text("x\n")
        (repo / "cmd-report.md").write_text("r\n")
        stub_ruff = tmp_path / "stub_ruff.sh"
        stub_ruff.write_text("#!/bin/sh\necho 'All checks passed!'\n")
        stub_mypy = tmp_path / "stub_mypy.sh"
        stub_mypy.write_text("#!/bin/sh\necho 'Success: no issues found'\n")
        stub_pytest = tmp_path / "stub_pytest.sh"
        stub_pytest.write_text(
            "#!/bin/sh\necho \"pytest $@\" >> \"$STUB_LOG\"\necho '3 passed in 0.05s'\n"
        )
        for s in (stub_ruff, stub_mypy, stub_pytest):
            s.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k != "SWARM_BUS"}
        env.update(
            {
                "SWARM_CHECK_RUFF": str(stub_ruff),
                "SWARM_CHECK_MYPY": str(stub_mypy),
                "SWARM_CHECK_PYTEST": str(stub_pytest),
                "STUB_LOG": str(tmp_path / "stub.log"),
                "SWARM_CHECK_ALLOW_STUBS": "1",
            }
        )
        return repo, env

    def test_script_gate_pass(self, tmp_path: Path) -> None:
        repo, env = self._repo_with_stubs(tmp_path)
        out_json = tmp_path / "out.json"
        proc = subprocess.run(
            [
                "bash",
                str(_SCRIPT.parent / "swarm-check.sh"),
                str(repo),
                "--base", "main",
                "--scope", "vibe_quant/*,*.txt",
                "--exactness", "never",
                "--frontend", "never",
                "--json", str(out_json),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PASS scope:" in proc.stdout
        assert "WARN" in proc.stdout
        assert "PASS ruff:" in proc.stdout
        assert "PASS tests:" in proc.stdout
        assert "SKIP exactness:" in proc.stdout
        assert "SKIP frontend:" in proc.stdout
        assert proc.stdout.strip().endswith("GATE PASS")
        data = json.loads(out_json.read_text())
        assert data["gate"] == "PASS"
        assert data["tests"]["status"] == "PASS"
        log = (tmp_path / "stub.log").read_text()
        assert "tests/unit -k foo" in log

    def test_script_gate_fail(self, tmp_path: Path) -> None:
        repo, env = self._repo_with_stubs(tmp_path)
        stub_mypy_fail = tmp_path / "stub_mypy_fail.sh"
        stub_mypy_fail.write_text("#!/bin/sh\necho '3 errors'\nexit 1\n")
        stub_mypy_fail.chmod(0o755)
        env["SWARM_CHECK_MYPY"] = str(stub_mypy_fail)
        out_json = tmp_path / "out.json"
        proc = subprocess.run(
            [
                "bash",
                str(_SCRIPT.parent / "swarm-check.sh"),
                str(repo),
                "--base", "main",
                "--scope", "docs/*",
                "--exactness", "never",
                "--frontend", "never",
                "--json", str(out_json),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "FAIL scope:" in proc.stdout
        assert "FAIL mypy:" in proc.stdout
        assert proc.stdout.strip().endswith("GATE FAIL (2 failed)")
        data = json.loads(out_json.read_text())
        assert data["gate"] == "FAIL"
