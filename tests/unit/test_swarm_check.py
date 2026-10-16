"""Tests for scripts/agents/swarm_check.py — the pre-review gate helper.

Pure functions and orchestration run against a throwaway git repo in tmp_path; the
tool subprocesses (ruff/mypy/pytest/exactness/frontend) are stubbed with a fake
runner, so no real tools ever run. One integration-ish test drives the .sh wrapper
with SWARM_CHECK_RUFF/SWARM_CHECK_MYPY/SWARM_CHECK_PYTEST stubs instead.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "swarm_check.py"
_spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
assert _spec is not None and _spec.loader is not None
swarm_check = importlib.util.module_from_spec(_spec)
sys.modules["swarm_check"] = swarm_check
_spec.loader.exec_module(swarm_check)

# Hermetic, non-interactive git: no user/system config (so gpgsign/hooks/aliases cannot
# change behaviour) and an identity supplied via the environment, so a repo needs no
# `git config` calls at all. Every git call in this file is one subprocess, the most
# expensive thing these tests do, so the template below exists to spawn as few as possible.
_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "gate",
    "GIT_AUTHOR_EMAIL": "gate@test",
    "GIT_COMMITTER_NAME": "gate",
    "GIT_COMMITTER_EMAIL": "gate@test",
}


def _git_env() -> dict[str, str]:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV_OVERRIDES)
    return env


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_git_env(),
    )
    return out.stdout


_TEMPLATE: Path | None = None


def _template_repo() -> Path:
    """One read-only repo template per test process: `git init` + commit are 3 of the ~25
    git spawns a cache test used to pay for its throwaway repo, and every test paid them.
    A `copytree` of a committed repo is a valid independent clone (see _init_repo)."""
    global _TEMPLATE
    if _TEMPLATE is None or not _TEMPLATE.exists():
        root = Path(tempfile.mkdtemp(prefix="swarm-check-template-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        repo = root / "wt"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=_git_env())
        _populate_repo(repo)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        _TEMPLATE = repo
    return _TEMPLATE


def _populate_repo(repo: Path) -> None:
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "keep.py").write_text("X = 1\n")
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 1\n")
    tests = repo / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_foo.py").write_text("def test_foo() -> None:\n    assert True\n")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "wt"
    shutil.copytree(_template_repo(), repo, symlinks=True)
    return repo


# Tests in TestPassCache that only assert a stamp is NOT reused otherwise spend a full gate
# run writing the stamp they then invalidate. The template below is the same repo state
# (_pass_repo) with a valid PASS stamp already present, produced by the production writer
# (cache_key + write_stamp), so those tests start from "would be reused" and pay for one run.
_PASS_CHECKS = (
    swarm_check.Check("scope", "SKIP", "no --scope given"),
    swarm_check.Check("clean", "PASS", "no untracked files"),
    swarm_check.Check("ruff", "PASS", "All checks passed!"),
    swarm_check.Check("mypy", "PASS", "Success"),
    swarm_check.Check("tests", "PASS", "1 passed in 0.01s | 1 passed in 0.01s"),
    swarm_check.Check("exactness", "SKIP", "disabled"),
    swarm_check.Check("exactness-carry", "SKIP", "disabled"),
    swarm_check.Check("frontend", "SKIP", "disabled"),
)
_PASS_SCOPES = ((), ("vibe_quant/*",))  # the two scopes TestPassCache gates under

_STAMPED_TEMPLATE: Path | None = None


def _stamped_template() -> Path:
    """_pass_repo's state (feature branch, foo changed) with a valid PASS stamp for every
    scope TestPassCache uses, written by swarm_check.write_stamp and checked by load_stamp."""
    global _STAMPED_TEMPLATE
    if _STAMPED_TEMPLATE is None or not _STAMPED_TEMPLATE.exists():
        root = Path(tempfile.mkdtemp(prefix="swarm-check-stamped-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        repo = root / "wt"
        shutil.copytree(_template_repo(), repo, symlinks=True)
        _git(repo, "checkout", "-qb", "feature")
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "change foo")
        runner = FakeRunner()
        head = _git(repo, "rev-parse", "HEAD").strip()
        base_sha = _git(repo, "rev-parse", "main^{commit}").strip()
        for scope in _PASS_SCOPES:
            cfg = swarm_check.Config(
                worktree=repo, main_root=root, base="main",
                scope=scope, exactness="never", frontend="never",
            )
            key = swarm_check.cache_key(head, base_sha, cfg, swarm_check.tool_hash())
            swarm_check.write_stamp(repo, runner, cfg, key, _PASS_CHECKS)
            assert swarm_check.load_stamp(repo, runner, key) is not None
        _STAMPED_TEMPLATE = repo
    return _STAMPED_TEMPLATE


def _init_pass_repo(tmp_path: Path) -> Path:
    """A copy of _stamped_template: a clean feature-branch repo whose current tree already
    has a valid PASS stamp, so a gate run would reuse it unless something changed."""
    repo = tmp_path / "wt"
    shutil.copytree(_stamped_template(), repo, symlinks=True)
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
            proc = subprocess.run(
                c, cwd=str(cwd), capture_output=True, text=True, env=_git_env()
            )
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


def _main_carry(tmp_path: Path, repo: Path | None = None, text: str = "# main carry\n") -> Path:
    """Main checkout's exactness_carry.py (cfg.main_root == tmp_path); optionally mirrored
    into the worktree, as the carry engine's own commit does."""
    script = tmp_path / "scripts" / "agents" / "exactness_carry.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text)
    if repo is not None:
        (repo / "scripts" / "agents").mkdir(parents=True, exist_ok=True)
        (repo / "scripts" / "agents" / "exactness_carry.py").write_text(text)
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


_KNOWN_MODULES = frozenset({"vibe_quant", "vibe_quant.a", "vibe_quant.a.b"})


class TestImportGraph:
    """The reverse import closure: which tests/unit files reach a changed vibe_quant module.

    Selection by test NAME (`-k <stem>`) cannot see a caller test whose names never mention
    the module: tests/unit/test_symbol_agg.py::test_topk_ranks_gate_passers_first ran red
    while the gate PASSed (job 20261015-funding-carry pqon2 r1). These are pure functions.
    """

    _GRAPH: dict[str, set[str]] = {
        "vibe_quant/__init__.py": set(),
        "vibe_quant/a/__init__.py": set(),
        "vibe_quant/a/b.py": {"vibe_quant.c"},
        "vibe_quant/c.py": set(),
        "tests/unit/test_direct.py": {"vibe_quant.a.b"},
        "tests/unit/test_pkg.py": {"vibe_quant.a"},
        "tests/unit/test_other.py": {"vibe_quant.c"},
        "tests/unit/test_gone.py": {"vibe_quant.gone"},
        "tests/unit/test_unrelated.py": {"vibe_quant.other"},
    }

    def test_module_name_from_path(self) -> None:
        assert swarm_check.module_name("vibe_quant/discovery/pipeline.py") == (
            "vibe_quant.discovery.pipeline"
        )
        assert swarm_check.module_name("vibe_quant/discovery/__init__.py") == "vibe_quant.discovery"
        assert swarm_check.module_name("vibe_quant/__init__.py") == "vibe_quant"
        assert swarm_check.module_name("vibe_quant/metrics.py") == "vibe_quant.metrics"
        # only vibe_quant modules seed the closure (scripts/ and tests/ changes are out of it)
        assert swarm_check.module_name("tests/unit/test_x.py") == ""
        assert swarm_check.module_name("scripts/agents/swarm_check.py") == ""
        assert swarm_check.module_name("vibe_quant/grammar.lark") == ""

    @pytest.mark.parametrize(
        ("source", "module", "expected"),
        [
            ("import vibe_quant.a.b\n", "vibe_quant.c.d", {"vibe_quant.a.b"}),
            ("import vibe_quant.a.b as x\n", "vibe_quant.c.d", {"vibe_quant.a.b"}),
            ("from vibe_quant.a.b import y\n", "vibe_quant.c.d", {"vibe_quant.a.b"}),
            ("import numpy\nimport vibe_quant.a.b\n", "vibe_quant.c.d", {"vibe_quant.a.b"}),
            # a lazy import inside a function still counts (common in this repo)
            ("def f():\n    from vibe_quant.a.b import y\n", "vibe_quant.c.d", {"vibe_quant.a.b"}),
            (
                "if TYPE_CHECKING:\n    from vibe_quant.a.b import y\n",
                "vibe_quant.c.d",
                {"vibe_quant.a.b"},
            ),
            ("from vibe_quant.a import b\n", "vibe_quant.c.d", {"vibe_quant.a", "vibe_quant.a.b"}),
            ("from .b import y\n", "vibe_quant.a.c", {"vibe_quant.a.b"}),
            ("from ..a import b\n", "vibe_quant.x.y", {"vibe_quant.a", "vibe_quant.a.b"}),
            # climbing above the repo root resolves to nothing: never a crash
            ("from ...a import b\n", "vibe_quant.x.y", set()),
        ],
    )
    def test_import_forms(self, source: str, module: str, expected: set[str]) -> None:
        assert swarm_check.parse_imports(source, module, _KNOWN_MODULES) == expected

    def test_package_relative_import(self) -> None:
        found = swarm_check.parse_imports(
            "from . import b\n", "vibe_quant.a", _KNOWN_MODULES, is_package=True
        )
        assert found == {"vibe_quant.a", "vibe_quant.a.b"}

    def test_submodule_import_has_no_parent_edge(self) -> None:
        # No implicit parent edge: otherwise EVERY vibe_quant.a test would be selected for
        # any change to one of its submodules.
        found = swarm_check.parse_imports("import vibe_quant.a.c\n", "vibe_quant.x.y", _KNOWN_MODULES)
        assert found == {"vibe_quant.a.c"}

    def test_from_package_of_a_non_module_keeps_the_package(self) -> None:
        found = swarm_check.parse_imports(
            "from vibe_quant.a import X\n", "vibe_quant.x.y", _KNOWN_MODULES
        )
        assert found == {"vibe_quant.a"}

    def test_unparseable_source_is_none(self) -> None:
        assert swarm_check.parse_imports("def (:\n", "vibe_quant.x", _KNOWN_MODULES) is None

    def _tree(self, tmp_path: Path) -> dict[str, set[str]]:
        """A tiny worktree: two caller tests (one nested), a broken file, a binary file."""
        (tmp_path / "vibe_quant" / "discovery").mkdir(parents=True)
        (tmp_path / "vibe_quant" / "discovery" / "pipeline.py").write_text("P = 1\n")
        (tmp_path / "vibe_quant" / "broken.py").write_text("def (:\n")
        (tmp_path / "vibe_quant" / "unreadable.py").write_bytes(b"\xff\xfe\x00\x01")
        (tmp_path / "tests" / "unit" / "api").mkdir(parents=True)
        for rel, source in (
            ("tests/unit/test_caller.py", "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n"),
            ("tests/unit/api/test_nested.py", "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n"),
            ("tests/unit/test_plainer.py", "import json\n"),
        ):
            (tmp_path / rel).write_text(source)
        return swarm_check.build_import_graph(tmp_path)

    def test_graph_walks_nested_test_dirs(self, tmp_path: Path) -> None:
        graph = self._tree(tmp_path)
        assert graph["tests/unit/test_caller.py"] == {"vibe_quant.discovery.pipeline"}
        assert graph["tests/unit/api/test_nested.py"] == {"vibe_quant.discovery.pipeline"}
        assert graph["tests/unit/test_plainer.py"] == set()
        assert swarm_check.importer_test_files(["vibe_quant/discovery/pipeline.py"], graph) == [
            "tests/unit/api/test_nested.py",
            "tests/unit/test_caller.py",
        ]

    def test_graph_skips_unparseable_and_unreadable_files(self, tmp_path: Path) -> None:
        graph = self._tree(tmp_path)
        assert graph["vibe_quant/broken.py"] == set()
        assert graph["vibe_quant/unreadable.py"] == set()
        assert swarm_check.importer_test_files(["vibe_quant/broken.py"], graph) == []

    def test_direct_and_transitive_importers(self) -> None:
        # test_other imports vibe_quant.c directly; test_direct reaches it through
        # vibe_quant.a.b (a two-module chain: the caller test in the bead's report).
        assert swarm_check.importer_test_files(["vibe_quant/c.py"], self._GRAPH) == [
            "tests/unit/test_direct.py",
            "tests/unit/test_other.py",
        ]

    def test_deleted_module_seeds_the_closure_by_name(self) -> None:
        # vibe_quant.gone.py is in the diff but not in the graph: it still seeds by name
        assert swarm_check.importer_test_files(["vibe_quant/gone.py"], self._GRAPH) == [
            "tests/unit/test_gone.py",
        ]

    def test_package_change_selects_only_its_explicit_importers(self) -> None:
        # test_direct imports vibe_quant.a.b, never the package itself
        assert swarm_check.importer_test_files(["vibe_quant/a/__init__.py"], self._GRAPH) == [
            "tests/unit/test_pkg.py",
        ]

    def test_no_selection_for_a_change_outside_vibe_quant(self) -> None:
        for path in ("scripts/agents/swarm_check.py", "tests/unit/test_direct.py", "README.md"):
            assert swarm_check.importer_test_files([path], self._GRAPH) == []
        assert swarm_check.importer_test_files([], self._GRAPH) == []


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
            "scope", "clean", "bus-inbox", "ruff", "mypy", "tests", "exactness",
            "exactness-carry", "frontend",
        ]
        assert by_name["scope"].status == "SKIP"
        assert by_name["bus-inbox"].status == "SKIP"
        assert by_name["tests"].status == "SKIP"
        assert by_name["exactness"].status == "SKIP"
        assert by_name["exactness-carry"].status == "SKIP"
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


def _importer_repo(tmp_path: Path) -> Path:
    """The bead's red fixture: a PRE-EXISTING test file imports the module under change.

    test_symbol_agg.py is committed on the feature branch, so the diff carries only an edit
    to vibe_quant/discovery/pipeline.py. No test name contains "pipeline", so the old
    `-k pipeline` name match can never reach that file.
    """
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "feature")
    (repo / "vibe_quant" / "discovery").mkdir()
    (repo / "vibe_quant" / "discovery" / "pipeline.py").write_text("P = 1\n")
    (repo / "tests" / "unit" / "test_symbol_agg.py").write_text(
        "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n\n\n"
        "def test_topk_ranks_gate_passers_first() -> None:\n"
        "    assert DiscoveryPipeline is not None\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add the pipeline and its caller test")
    (repo / "vibe_quant" / "discovery" / "pipeline.py").write_text("P = 2\n")  # the change
    return repo


def _lazy_importer_repo(tmp_path: Path) -> Path:
    """G2 fixture: pipeline imports selection INSIDE a function (lazy imports are common)."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "feature")
    (repo / "vibe_quant" / "discovery").mkdir()
    (repo / "vibe_quant" / "discovery" / "selection.py").write_text("def f() -> int:\n    return 1\n")
    (repo / "vibe_quant" / "discovery" / "pipeline.py").write_text(
        "def run() -> int:\n    from vibe_quant.discovery.selection import f\n\n    return f()\n"
    )
    (repo / "tests" / "unit" / "test_caller.py").write_text(
        "from vibe_quant.discovery.pipeline import run\n\n\n"
        "def test_run() -> None:\n    assert run() == 1\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "lazy selection import")
    (repo / "vibe_quant" / "discovery" / "selection.py").write_text(
        "def f() -> int:\n    return 2\n"
    )  # the change
    return repo


class TestImporterTestSelection:
    """run_checks: the reverse import closure drives real pytest calls."""

    @staticmethod
    def _pytest_cmds(runner: FakeRunner) -> list[list[str]]:
        return [
            c[c.index("pytest") + 1:]
            for c in runner.calls
            if any("pytest" in Path(a).name for a in c)
        ]

    @staticmethod
    def _fails_on(needle: str) -> Callable[[Sequence[str], Path], tuple[int, str]]:
        def run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            if any(needle in Path(a).name for a in cmd):
                return 1, "1 failed in 0.02s"
            return 0, "1 passed in 0.01s"

        return run

    def test_importing_test_file_is_selected_and_fails_the_gate(self, tmp_path: Path) -> None:
        repo = _importer_repo(tmp_path)
        runner = FakeRunner()
        runner.responses = {"pytest": self._fails_on("test_symbol_agg.py")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by = {c.name: c for c in checks}
        assert by["tests"].status == "FAIL"
        assert "1 failed" in by["tests"].detail
        assert _called_with(runner, "test_symbol_agg.py")
        assert swarm_check.gate_summary(checks) == ("GATE FAIL (1 failed)", 1)

    def test_importer_files_run_as_one_parallel_call_plus_a_serial_timing_call(
        self, tmp_path: Path
    ) -> None:
        repo = _importer_repo(tmp_path)
        runner = FakeRunner()

        def respond(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            # rc5 on the importer files' serial timing call only: "no timing tests" is a SKIP note
            if "tests/unit/test_symbol_agg.py" in cmd and cmd[-1] == "timing":
                return 5, "no tests ran"
            return 0, "1 passed in 0.01s"

        runner.responses = {"pytest": respond}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit/test_symbol_agg.py", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit/test_symbol_agg.py", "-m", "timing"],
            ["tests/unit", "-k", "pipeline", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-k", "pipeline", "-m", "timing"],
        ]
        assert "1 importer files" in tests.detail
        assert "no timing tests in the importer files" in tests.detail

    def test_deleted_module_still_selects_its_lazy_importers_chain(self, tmp_path: Path) -> None:
        # B1: deleting selection.py (on disk) must keep the `from vibe_quant.discovery import
        # selection` submodule edge, or the pipeline's caller test is never selected.
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-qb", "feature")
        (repo / "vibe_quant" / "discovery").mkdir()
        (repo / "vibe_quant" / "discovery" / "__init__.py").write_text("")
        (repo / "vibe_quant" / "discovery" / "selection.py").write_text("def f() -> int:\n    return 1\n")
        (repo / "vibe_quant" / "discovery" / "pipeline.py").write_text(
            "def run() -> int:\n    from vibe_quant.discovery import selection\n\n"
            "    return selection.f()\n"
        )
        (repo / "tests" / "unit" / "test_caller.py").write_text(
            "from vibe_quant.discovery.pipeline import run\n\n\n"
            "def test_run() -> None:\n    assert run() == 1\n"
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "lazy submodule import")
        (repo / "vibe_quant" / "discovery" / "selection.py").unlink()  # the change: deletion
        runner = FakeRunner()
        runner.responses = {"pytest": self._fails_on("test_caller.py")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["tests"].status == "FAIL"
        assert _called_with(runner, "test_caller.py")
        assert swarm_check.gate_summary(checks)[1] == 1

    def test_a_changed_test_file_is_not_repeated_as_an_importer(self, tmp_path: Path) -> None:
        repo = _importer_repo(tmp_path)
        (repo / "tests" / "unit" / "test_symbol_agg.py").write_text(
            "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n\n\n"
            "def test_topk_ranks_gate_passers_first() -> None:\n    assert DiscoveryPipeline\n"
        )  # now also a changed test file
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit/test_symbol_agg.py"],
            ["tests/unit", "-k", "pipeline", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-k", "pipeline", "-m", "timing"],
        ]
        assert "importer files" not in tests.detail

    def test_transitive_importer_through_a_function_level_import(self, tmp_path: Path) -> None:
        repo = _lazy_importer_repo(tmp_path)
        runner = FakeRunner()
        runner.responses = {"pytest": self._fails_on("test_caller.py")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "FAIL"
        assert _called_with(runner, "test_caller.py")

    def test_importer_count_at_the_cap_still_runs_the_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _importer_repo(tmp_path)
        monkeypatch.setattr(swarm_check, "MAX_IMPORTER_FILES", 1)  # boundary: 1 > 1 is false
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert _called_with(runner, "test_symbol_agg.py")
        assert "1 importer files" in tests.detail

    def test_the_configured_cap_trips_on_one_file_over_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _importer_repo(tmp_path)
        (repo / "vibe_quant" / "hub.py").write_text("H = 1\n")  # the changed seed
        n = swarm_check.MAX_IMPORTER_FILES + 1
        assert n < 10_000  # a cap this high could never trip: fail here, not by hanging
        graph = {"vibe_quant/hub.py": set()} | {
            f"tests/unit/test_imp{i:03d}.py": {"vibe_quant.hub"} for i in range(n)
        }
        monkeypatch.setattr(swarm_check, "build_import_graph", lambda worktree, extra=(): graph)
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "9 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-m", "timing"],
        ]
        assert f"{n} importer files > cap {swarm_check.MAX_IMPORTER_FILES}" in tests.detail
        assert "full tests/unit" in tests.detail
        assert not _called_with(runner, "test_imp")

    def test_scripts_agents_change_runs_no_importer_selection(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "scripts" / "agents").mkdir(parents=True)
        (repo / "scripts" / "agents" / "swarm_check.py").write_text("X = 2\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit", "-k", "swarm_check", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-k", "swarm_check", "-m", "timing"],
        ]


class TestExactnessCarry:
    """exactness-carry: a SECOND exactness proof, for the funding-carry engine.

    It runs the MAIN checkout's scripts/agents/exactness_carry.py with the worktree's code
    on PYTHONPATH, exactly like exactness_239. It must SKIP (never FAIL) while the carry
    script has not landed on main yet, and it must never run a worktree copy.
    """

    def _check(
        self,
        tmp_path: Path,
        repo: Path,
        responses: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[swarm_check.Check, FakeRunner]:
        runner = FakeRunner()
        if responses:
            runner.responses = responses  # type: ignore[assignment]
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never", **kwargs)
        checks = swarm_check.run_checks(cfg, runner)
        return {c.name: c for c in checks}["exactness-carry"], runner

    def test_triggers(self) -> None:
        for p in [
            "vibe_quant/portfolio/sizing.py",
            "vibe_quant/portfolio/deep/nested.py",
            "vibe_quant/validation/funding.py",
            "scripts/agents/exactness_carry.py",
            "pyproject.toml",
            "uv.lock",
        ]:
            assert swarm_check.should_run_exactness_carry([p]), p

    def test_no_trigger_elsewhere(self) -> None:
        for p in [
            "vibe_quant/validation/runner.py",
            "vibe_quant/validation/funding_extra.py",
            "vibe_quant/portfolios/x.py",
            "vibe_quant/dsl/parser.py",
            "vibe_quant/api/app.py",
            "scripts/agents/exactness_239.py",
            "scripts/agents/swarm_check.py",
            "frontend/x.ts",
            "tests/unit/test_x.py",
        ]:
            assert not swarm_check.should_run_exactness_carry([p]), p

    def test_skips_without_trigger_and_names_the_trigger_set(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "api").mkdir()
        (repo / "vibe_quant" / "api" / "app.py").write_text("A = 1\n")
        _main_carry(tmp_path)  # present on main: only the missing trigger can skip it
        carry, runner = self._check(tmp_path, repo)
        assert carry.status == "SKIP"
        assert "vibe_quant/portfolio/" in carry.detail
        assert "vibe_quant/validation/funding.py" in carry.detail
        assert "scripts/agents/exactness_carry.py" in carry.detail
        assert not _called_with(runner, "exactness_carry.py")

    def test_skips_when_main_lacks_the_script(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "portfolio").mkdir()
        (repo / "vibe_quant" / "portfolio" / "sizing.py").write_text("S = 1\n")
        carry, runner = self._check(tmp_path, repo)  # no _main_carry: not on main yet
        assert carry.status == "SKIP"
        assert "main has no exactness_carry.py yet" in carry.detail
        assert not _called_with(runner, "exactness_carry.py")

    def test_skips_when_worktree_has_it_but_main_does_not(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "portfolio").mkdir()
        (repo / "vibe_quant" / "portfolio" / "sizing.py").write_text("S = 1\n")
        (repo / "scripts" / "agents").mkdir(parents=True, exist_ok=True)
        (repo / "scripts" / "agents" / "exactness_carry.py").write_text("# worktree only\n")
        carry, runner = self._check(tmp_path, repo)
        assert carry.status == "SKIP"
        assert "main has no exactness_carry.py yet" in carry.detail
        assert not _called_with(runner, "exactness_carry.py")

    def test_modified_worktree_copy_fails_and_is_never_run(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_carry(tmp_path)
        (repo / "scripts" / "agents").mkdir(parents=True, exist_ok=True)
        (repo / "scripts" / "agents" / "exactness_carry.py").write_text("EXPECTED = 9\n")
        carry, runner = self._check(tmp_path, repo)
        assert carry.status == "FAIL"
        assert "exactness_carry script modified in worktree" in carry.detail
        assert not _called_with(runner, "exactness_carry.py")

    def test_triggered_run_passes_on_identical_via_main_with_worktree_code(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "portfolio").mkdir()
        (repo / "vibe_quant" / "portfolio" / "sizing.py").write_text("S = 1\n")
        main_script = _main_carry(tmp_path, repo)
        seen: list[Path] = []

        def capture(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            seen.append(cwd)
            return 0, "IDENTICAL"

        carry, runner = self._check(tmp_path, repo, responses={"exactness_carry.py": capture})
        assert carry.status == "PASS" and "IDENTICAL" in carry.detail
        assert seen == [repo]  # cwd is the worktree, not main
        called = [c for c in runner.calls if c[-1] == str(main_script)]
        assert called and called[0][0] == str(tmp_path / ".venv" / "bin" / "python")
        envs = [e for c, e in zip(runner.calls, runner.envs, strict=True) if c[-1] == str(main_script)]
        assert envs and envs[0] is not None
        assert envs[0]["PYTHONPATH"] == str(repo)  # exercises the WORKTREE's code
        timeouts = [
            t for c, t in zip(runner.calls, runner.timeouts, strict=True) if c[-1] == str(main_script)
        ]
        assert timeouts == [swarm_check.TEST_TIMEOUT]
        # never the worktree copy
        assert str(repo / "scripts" / "agents" / "exactness_carry.py") not in sum(runner.calls, [])

    def test_rc_zero_without_identical_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_carry(tmp_path, repo)
        carry, _ = self._check(
            tmp_path, repo, responses={"exactness_carry.py": (0, "no evidence line")},
            exactness="always",
        )
        assert carry.status == "FAIL"
        assert "drift (rc=0)" in carry.detail and "no evidence line" in carry.detail

    def test_nonzero_rc_with_identical_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_carry(tmp_path, repo)
        carry, _ = self._check(
            tmp_path, repo, responses={"exactness_carry.py": (3, "IDENTICAL then crashed")},
            exactness="always",
        )
        assert carry.status == "FAIL"
        assert "drift (rc=3)" in carry.detail

    def test_disabled_by_exactness_never(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "portfolio").mkdir()
        (repo / "vibe_quant" / "portfolio" / "sizing.py").write_text("S = 1\n")
        _main_carry(tmp_path, repo)  # script present and a trigger changed: still disabled
        carry, runner = self._check(tmp_path, repo, exactness="never")
        assert carry.status == "SKIP" and carry.detail == "disabled"
        assert not _called_with(runner, "exactness_carry.py")

    def test_exactness_always_runs_it_without_a_trigger(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_carry(tmp_path, repo)
        carry, runner = self._check(
            tmp_path, repo, responses={"exactness": (0, "IDENTICAL")}, exactness="always"
        )
        assert carry.status == "PASS"
        assert _called_with(runner, "exactness_carry.py")

    def test_exactness_always_skips_when_main_lacks_the_script(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        carry, runner = self._check(tmp_path, repo, exactness="always")
        assert carry.status == "SKIP"
        assert "main has no exactness_carry.py yet" in carry.detail
        assert not _called_with(runner, "exactness_carry.py")


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
        # A big selection runs under xdist with the relative-speed tests deselected,
        # then those `timing` tests run serially (they flake under worker contention).
        assert [c[c.index("pytest") + 1:] for c in pytest_cmds] == [
            ["tests/unit", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-m", "timing"],
        ]

    def test_conftest_full_suite_rc5_is_fail_not_skip(self, tmp_path: Path) -> None:
        """rc 5 means "nothing matched" only for a -k group, where a smaller selection is
        legitimate. On the FULL suite (a conftest/__init__ change) it means the suite did
        not run: FAIL, never SKIP."""
        repo = _init_repo(tmp_path)
        (repo / "tests" / "conftest.py").write_text("X = 1\n")
        runner = FakeRunner()

        def pytest_by_parallel(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            if "-n" in cmd:  # the xdist full-suite pass
                return 5, "no tests ran in 0.01s"
            return 0, "1 passed in 0.01s"

        runner.responses = {"pytest": pytest_by_parallel}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by = {c.name: c for c in checks}
        assert by["tests"].status == "FAIL"
        assert "no tests ran" in by["tests"].detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_k_group_runs_parallel_then_timing_serially(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "1 passed")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        pytest_cmds = [c for c in runner.calls if any("pytest" in Path(a).name for a in c)]
        assert [c[c.index("pytest") + 1:] for c in pytest_cmds] == [
            ["tests/unit", "-k", "foo", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-k", "foo", "-m", "timing"],
        ]

    def test_targeted_files_run_serially_without_xdist(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_bar.py").write_text("def test_bar() -> None:\n    pass\n")
        runner = FakeRunner()
        runner.responses = {"pytest": (0, "1 passed")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        swarm_check.run_checks(cfg, runner)
        pytest_cmds = [c for c in runner.calls if any("pytest" in Path(a).name for a in c)]
        # A handful of files: xdist startup costs more than it saves, so no -n/-m.
        assert [c[c.index("pytest") + 1:] for c in pytest_cmds] == [["tests/unit/test_bar.py"]]

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


class TestCacheKey:
    def test_scope_order_insensitive(self, tmp_path: Path) -> None:
        a = _cfg(tmp_path, tmp_path, scope=("b/*", "a/*"))
        b = _cfg(tmp_path, tmp_path, scope=("a/*", "b/*"))
        assert swarm_check.cache_key("h", "s", a, "t") == swarm_check.cache_key("h", "s", b, "t")

    def test_every_field_matters(self, tmp_path: Path) -> None:
        base = _cfg(tmp_path, tmp_path)
        key = swarm_check.cache_key("h", "s", base, "t")
        assert key != swarm_check.cache_key("h2", "s", base, "t")
        assert key != swarm_check.cache_key("h", "s2", base, "t")
        assert key != swarm_check.cache_key("h", "s", _cfg(tmp_path, tmp_path, scope=("x",)), "t")
        assert key != swarm_check.cache_key("h", "s", _cfg(tmp_path, tmp_path, exactness="always"), "t")
        assert key != swarm_check.cache_key("h", "s", _cfg(tmp_path, tmp_path, frontend="always"), "t")
        assert key != swarm_check.cache_key("h", "s", base, "t2")

    def test_main_root_inputs_matter(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, tmp_path)
        none = swarm_check.cache_key("h", "s", cfg, "t")  # all three files missing: fixed token
        assert none == swarm_check.cache_key("h", "s", cfg, "t")
        seen = {none}
        for rel in swarm_check.MAIN_KEY_FILES:
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("v1\n")
            first = swarm_check.cache_key("h", "s", cfg, "t")
            path.write_text("v2\n")
            second = swarm_check.cache_key("h", "s", cfg, "t")
            assert first != second, rel
            assert first not in seen and second not in seen, rel
            seen.update({first, second})

    def test_main_key_files_cover_exactness_and_lockfiles(self) -> None:
        assert set(swarm_check.MAIN_KEY_FILES) == {
            "scripts/agents/exactness_239.py", "scripts/agents/exactness_carry.py",
            "uv.lock", "frontend/pnpm-lock.yaml",
        }

    def test_carry_pin_change_on_main_invalidates_the_key(self, tmp_path: Path) -> None:
        """A pin edit to MAIN's exactness_carry.py invalidates stamps, as 239's does."""
        cfg = _cfg(tmp_path, tmp_path)
        script = tmp_path / "scripts" / "agents" / "exactness_carry.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("EXPECTED = 1\n")
        first = swarm_check.cache_key("h", "s", cfg, "t")
        script.write_text("EXPECTED = 2\n")
        assert swarm_check.cache_key("h", "s", cfg, "t") != first

    def test_parse_check_line_roundtrip(self) -> None:
        check = swarm_check.Check("tests", "PASS", "1 passed | x: y")
        line = swarm_check.format_line(check)
        assert swarm_check.parse_check_line(line) == check
        assert swarm_check.parse_check_line("garbage") is None
        assert swarm_check.parse_check_line("BOGUS name: detail") is None


class TestPassCache:
    def _unstamped_pass_repo(self, tmp_path: Path) -> Path:
        """A clean feature-branch repo with NO stamp yet, so a run must really write one."""
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-qb", "feature")
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "change foo")
        return repo

    def _pass_repo(self, tmp_path: Path) -> Path:
        """The same repo state with its valid PASS stamp already present (see _stamped_template),
        so tests that only assert reuse is refused pay for one gate run instead of two."""
        repo = _init_pass_repo(tmp_path)
        assert len(self._stamps(repo)) == len(_PASS_SCOPES)  # else the test passes for free
        return repo

    def _runner(self) -> FakeRunner:
        runner = FakeRunner()
        runner.responses = {
            "ruff": (0, "All checks passed!"),
            "mypy": (0, "Success"),
            "pytest": (0, "1 passed in 0.01s"),
        }
        return runner

    def _main(self, repo: Path, runner: FakeRunner, *extra: str) -> int:
        rc: int = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never", *extra],
            run=runner,
        )
        return rc

    def _stamps(self, repo: Path) -> list[Path]:
        return sorted((repo / ".git" / "swarm-check").glob("pass-*.json"))

    def test_pass_writes_stamp_then_reuses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._unstamped_pass_repo(tmp_path)
        first = self._runner()
        assert self._main(repo, first) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert _called_with(first, "pytest")  # expensive tools really ran
        assert len(self._stamps(repo)) == 1
        assert _git(repo, "status", "--porcelain").strip() == ""  # stamp invisible to git

        second = self._runner()
        out_json = tmp_path / "cached.json"
        assert self._main(repo, second, "--json", str(out_json)) == 0
        out = capsys.readouterr().out
        assert "(cached) PASS ruff: All checks passed!" in out
        assert "PASS cache: reused stamp from" in out
        assert out.strip().endswith("GATE PASS")
        assert not _called_with(second, "pytest")
        assert not _called_with(second, "ruff")
        assert not _called_with(second, "mypy")
        data = json.loads(out_json.read_text())
        assert data["gate"] == "PASS" and data["cached"] is True
        assert data["tests"] == {
            "status": "PASS",
            "detail": "1 passed in 0.01s | 1 passed in 0.01s",
        }

    def test_untampered_stamped_template_is_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Positive control for the not-reused tests below: if _stamped_template's key ever
        diverged from main()'s key, they would all pass without proving anything."""
        repo = self._pass_repo(tmp_path)
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert "PASS cache: reused" in capsys.readouterr().out
        assert not _called_with(runner, "pytest")

    def test_new_head_no_reuse(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        repo = self._pass_repo(tmp_path)  # already stamped at the feature HEAD
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 3\n")
        _git(repo, "commit", "-qam", "bump")
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_base_moved_no_reuse(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        repo = self._pass_repo(tmp_path)  # already stamped at the feature HEAD
        _git(repo, "checkout", "-q", "main")
        (repo / "vibe_quant" / "keep.py").write_text("X = 9\n")
        _git(repo, "commit", "-qam", "advance main")
        _git(repo, "checkout", "-q", "feature")
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_different_scope_no_reuse(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._pass_repo(tmp_path)  # stamped for scope "vibe_quant/*"
        runner = self._runner()
        assert self._main(repo, runner, "--scope", "vibe_quant/*,tests/*") == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_untracked_file_blocks_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._unstamped_pass_repo(tmp_path)
        (repo / "stray.txt").write_text("x\n")
        assert self._main(repo, self._runner()) == 0
        assert not self._stamps(repo)
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")

    def test_uncommitted_change_blocks_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._unstamped_pass_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 7\n")
        assert self._main(repo, self._runner()) == 0
        assert not self._stamps(repo)

    def test_fail_writes_no_stamp(self, tmp_path: Path) -> None:
        repo = self._unstamped_pass_repo(tmp_path)
        runner = self._runner()
        runner.responses["ruff"] = (1, "E501 boom")
        assert self._main(repo, runner) == 1
        assert not self._stamps(repo)

    def test_no_cache_flag_forces_full_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._pass_repo(tmp_path)  # already stamped: without --no-cache it would be reused
        runner = self._runner()
        assert self._main(repo, runner, "--no-cache") == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_corrupt_stamp_forces_full_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._pass_repo(tmp_path)
        for stamp in self._stamps(repo):
            stamp.write_text("{not json")
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        stamps = self._stamps(repo)
        assert len(stamps) == len(_PASS_SCOPES)
        # the full run rewrote its own stamp; the other corrupt one was never served
        assert len([p for p in stamps if "not json" not in p.read_text()]) == 1

    def test_stamp_with_fail_line_not_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._pass_repo(tmp_path)
        for stamp in self._stamps(repo):
            data = json.loads(stamp.read_text())
            data["checks"] = ["FAIL ruff: injected"] + data["checks"]
            stamp.write_text(json.dumps(data))
        runner = self._runner()
        assert self._main(repo, runner) == 0  # must re-run, never serve a FALSE PASS
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_tool_hash_change_no_reuse(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = self._pass_repo(tmp_path)
        monkeypatch.setattr(swarm_check, "tool_hash", lambda: "0" * 64)
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_commit_during_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """B1: a commit made while pytest runs must not be stamped (GATE PASS is for old HEAD)."""
        repo = self._unstamped_pass_repo(tmp_path)
        committed = False

        def commit_mid_run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            nonlocal committed
            if not committed:  # the gate runs pytest twice (parallel + timing)
                committed = True
                (repo / "vibe_quant" / "foo.py").write_text("def foo(:\n")  # broken, never gated
                _git(repo, "commit", "-qam", "broken mid-run")
            return 0, "1 passed in 0.01s"

        first = self._runner()
        first.responses["pytest"] = commit_mid_run
        assert self._main(repo, first) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert not self._stamps(repo)
        second = self._runner()
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")  # re-gated, never served from cache
        assert "PASS cache:" not in capsys.readouterr().out

    def test_dirty_start_committed_mid_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """B1: untracked at start (WARN) + committed during the run -> end state is clean."""
        repo = self._unstamped_pass_repo(tmp_path)
        (repo / "stray.py").write_text("X = 1\n")
        committed = False

        def commit_mid_run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            nonlocal committed
            if not committed:  # the gate runs pytest twice (parallel + timing)
                committed = True
                _git(repo, "add", "-A")
                _git(repo, "commit", "-qm", "stray")
            return 0, "1 passed in 0.01s"

        first = self._runner()
        first.responses["pytest"] = commit_mid_run
        assert self._main(repo, first) == 0
        assert not self._stamps(repo)
        second = self._runner()
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")

    def test_commit_and_reset_mid_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """N1: commit + `reset --hard` puts HEAD and the tree back where they started, so the
        pre/post key compare matches; the HEAD reflog still moved, so no stamp is written."""
        repo = self._unstamped_pass_repo(tmp_path)
        done = False

        def commit_then_reset(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            nonlocal done
            if not done:
                done = True
                (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 5\n")
                _git(repo, "commit", "-qam", "temp")
                _git(repo, "reset", "-q", "--hard", "HEAD~1")
            return 0, "1 passed in 0.01s"

        first = self._runner()
        first.responses["pytest"] = commit_then_reset
        assert self._main(repo, first) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert not self._stamps(repo)
        second = self._runner()
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")  # re-gated, never served from cache

    def test_unreadable_reflog_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """N1 fail closed: when git cannot report the HEAD reflog length the verdict is still
        printed, but no stamp is written (the mid-run guard could not run)."""

        class NoReflog(FakeRunner):
            def __call__(
                self,
                cmd: Sequence[str],
                cwd: Path,
                env: Mapping[str, str] | None = None,
                timeout: int = 600,
            ) -> tuple[int, str]:
                if "--walk-reflogs" in [str(x) for x in cmd]:
                    return 128, "fatal: reflog unavailable"
                return super().__call__(cmd, cwd, env, timeout)

        repo = self._unstamped_pass_repo(tmp_path)
        first = NoReflog()
        first.responses = self._runner().responses
        assert self._main(repo, first) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert not self._stamps(repo)
        assert swarm_check.head_reflog_count(repo, first) is None

    def test_stamp_with_warn_line_not_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._pass_repo(tmp_path)
        for stamp in self._stamps(repo):
            data = json.loads(stamp.read_text())
            data["checks"] = ["WARN clean: 1 untracked: x"] + data["checks"]
            stamp.write_text(json.dumps(data))
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_wrong_version_stamp_not_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A stamp written by another CACHE_VERSION is ignored, never served."""
        repo = self._pass_repo(tmp_path)
        for stamp in self._stamps(repo):
            data = json.loads(stamp.read_text())
            data["version"] = swarm_check.CACHE_VERSION + 1
            stamp.write_text(json.dumps(data))
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_stamp_from_before_the_carry_check_not_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No CACHE_VERSION bump is needed for the new exactness-carry line: every stamp
        written before that check existed was keyed on a different tool hash (the key
        hashes this script's own bytes), so an old stamp - which has no carry line - can
        never match the key and is never served as a PASS."""
        repo = self._unstamped_pass_repo(tmp_path)
        head = _git(repo, "rev-parse", "HEAD").strip()
        base_sha = _git(repo, "rev-parse", "main^{commit}").strip()
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        pre_carry = tuple(c for c in _PASS_CHECKS if c.name != "exactness-carry")
        assert len(pre_carry) == len(_PASS_CHECKS) - 1
        old_key = swarm_check.cache_key(head, base_sha, cfg, "0" * 64)  # pre-change bytes
        swarm_check.write_stamp(repo, FakeRunner(), cfg, old_key, pre_carry)
        assert len(self._stamps(repo)) == 1
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")  # re-gated, never served from the old stamp
        assert "PASS cache:" not in capsys.readouterr().out

    def test_stored_fail_gate_not_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A stamp whose stored gate is FAIL is never replayed as a PASS, even with a
        matching key and only PASS/SKIP check lines."""
        repo = self._pass_repo(tmp_path)
        for stamp in self._stamps(repo):
            data = json.loads(stamp.read_text())
            data["gate"] = "FAIL"
            stamp.write_text(json.dumps(data))
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_active_stubs_never_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S1: stubs honoured (ALLOW_STUBS=1) -> the PASS came from fakes -> no stamp."""
        repo = self._unstamped_pass_repo(tmp_path)
        monkeypatch.setenv("SWARM_CHECK_ALLOW_STUBS", "1")
        monkeypatch.setenv("SWARM_CHECK_RUFF", "ruff-stub")
        assert self._main(repo, self._runner()) == 0
        assert "stubs" in capsys.readouterr().out
        assert not self._stamps(repo)

    def test_stamp_write_is_atomic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """N2: the stamp appears only via os.replace of a temp file; no leftovers on failure."""
        repo = self._unstamped_pass_repo(tmp_path)

        def boom(src: object, dst: object) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(swarm_check.os, "replace", boom)
        assert self._main(repo, self._runner()) == 0  # a stamp problem never fails the gate
        assert not self._stamps(repo)
        cache_dir = repo / ".git" / "swarm-check"
        assert not cache_dir.exists() or not list(cache_dir.iterdir())
        monkeypatch.undo()
        assert self._main(repo, self._runner()) == 0
        assert len(self._stamps(repo)) == 1
        assert list((repo / ".git" / "swarm-check").iterdir()) == self._stamps(repo)

    def test_cache_hit_posts_to_bus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """N3: a cached PASS is announced to the chief like a normal run."""
        repo = self._unstamped_pass_repo(tmp_path)
        bus = tmp_path / "bus"
        bus.mkdir()  # a real job bus exists; the gate never creates one
        monkeypatch.setenv("SWARM_BUS", str(bus))
        assert self._main(repo, self._runner()) == 0
        head = _git(repo, "rev-parse", "HEAD").strip()
        assert self._main(repo, self._runner()) == 0
        bodies = [
            json.loads(ln)["body"]
            for ln in (bus / "messages.jsonl").read_text().splitlines()
        ]
        assert bodies == ["wt: GATE PASS", f"wt: GATE PASS (cached {head[:12]})"]


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


class TestBusInbox:
    """The gate must refuse a hand-back while the chief has an unread DM to the agent."""

    def _bus(self, tmp_path: Path, messages: list[dict[str, object]] | None = None) -> Path:
        d = tmp_path / "bus"
        d.mkdir()
        if messages:
            (d / "messages.jsonl").write_text(
                "".join(json.dumps(m) + "\n" for m in messages), encoding="utf-8"
            )
        return d

    def _check(self, tmp_path: Path, **kwargs: object) -> tuple[Any, Path]:
        repo = _init_repo(tmp_path)
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never", **kwargs)
        chk = {c.name: c for c in swarm_check.run_checks(cfg, FakeRunner())}["bus-inbox"]
        return chk, repo

    def test_skip_without_agent_or_bus(self, tmp_path: Path) -> None:
        assert self._check(tmp_path)[0].status == "SKIP"

    def test_pass_when_no_unread_chief_message(self, tmp_path: Path) -> None:
        d = self._bus(tmp_path, [{"id": "p1", "from": "peer", "to": "sa", "kind": "info", "body": "hi"}])
        chk = self._check(tmp_path, bus=str(d), bus_agent="sa")[0]
        assert chk.status == "PASS"
        assert not (d / "cursors" / "sa.offset").exists()  # peek never advances a cursor

    def test_fail_on_unread_chief_message(self, tmp_path: Path) -> None:
        d = self._bus(tmp_path, [{"id": "m1", "from": "chief", "to": "sa", "kind": "question", "body": "x" * 100}])
        chk = self._check(tmp_path, bus=str(d), bus_agent="sa")[0]
        assert chk.status == "FAIL"
        assert "m1" in chk.detail and "x" * 80 in chk.detail and "x" * 81 not in chk.detail
        assert not (d / "cursors" / "sa.offset").exists()  # a FAIL must not consume the DM

    def test_cached_pass_plus_unread_chief_dm_is_gate_fail(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _init_pass_repo(tmp_path)  # a clean tree with a valid PASS stamp: reuse path
        d = self._bus(tmp_path, [{"id": "m1", "from": "chief", "to": "sa", "kind": "question", "body": "reread the brief"}])
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(d), "--bus-agent", "sa"],
            run=FakeRunner(),
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "PASS cache: reused stamp from" in out  # the tool checks really were cached
        assert "FAIL bus-inbox:" in out and "m1" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")


class TestBusInboxHardening:
    """Review findings: late DMs, misconfigured buses, corrupt cursors, --bus telemetry."""

    @staticmethod
    def _chief_dm(bus_dir: Path, mid: str = "late1") -> None:
        with (bus_dir / "messages.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": mid, "from": "chief", "to": "sa", "kind": "info", "body": "stop"}) + "\n")

    def test_chief_dm_arriving_while_the_tools_run_fails_the_gate(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        d = tmp_path / "bus"
        d.mkdir()
        runner = FakeRunner()

        def ruff_then_dm(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            self._chief_dm(d)  # lands after any early peek, before the verdict
            return 0, "All checks passed!"

        runner.responses["ruff"] = ruff_then_dm
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never", bus=str(d), bus_agent="sa")
        chk = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["bus-inbox"]
        assert chk.status == "FAIL" and "late1" in chk.detail

    def test_missing_configured_bus_dir_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        cfg = _cfg(tmp_path, repo, bus=str(tmp_path / "nope"), bus_agent="sa")
        chk = swarm_check.bus_inbox_check(cfg)
        assert chk.status == "FAIL" and "nope" in chk.detail
        assert not (tmp_path / "nope").exists()  # the check must not create it

    def test_agent_without_a_bus_fails_but_bus_alone_skips(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        assert swarm_check.bus_inbox_check(_cfg(tmp_path, repo, bus_agent="sa")).status == "FAIL"
        assert swarm_check.bus_inbox_check(_cfg(tmp_path, repo, bus=str(tmp_path))).status == "SKIP"

    def test_corrupt_cursor_is_reported_as_a_fail_with_the_reason(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        d = tmp_path / "bus"
        (d / "cursors").mkdir(parents=True)
        (d / "cursors" / "sa.offset").write_text("")
        chk = swarm_check.bus_inbox_check(_cfg(tmp_path, repo, bus=str(d), bus_agent="sa"))
        assert chk.status == "FAIL" and "sa.offset" in chk.detail

    def test_gate_telemetry_goes_to_the_bus_flag_not_the_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SWARM_BUS", raising=False)
        monkeypatch.delenv("SWARM_AGENT", raising=False)
        repo = _init_pass_repo(tmp_path)
        d = tmp_path / "bus"
        d.mkdir()
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(d), "--bus-agent", "sa"],
            run=FakeRunner(),
        )
        assert rc == 0
        bodies = [json.loads(ln)["body"] for ln in (d / "messages.jsonl").read_text().splitlines()]
        assert len(bodies) == 1 and "GATE PASS" in bodies[0]

    def test_mistyped_bus_is_never_created_and_keeps_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("SWARM_BUS", raising=False)
        repo = _init_pass_repo(tmp_path)
        typo = tmp_path / "jobs" / "20261012-typo" / "bus"
        for _ in range(2):  # the first run must not create the dir that makes the second one pass
            rc = swarm_check.main(
                [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
                 "--bus", str(typo), "--bus-agent", "sa"],
                run=FakeRunner(),
            )
            cap = capsys.readouterr()
            assert rc == 1 and "FAIL bus-inbox:" in cap.out and cap.out.strip().endswith("GATE FAIL (1 failed)")
            assert "not posting" in cap.err and str(typo) in cap.err
            assert not typo.exists() and not typo.parent.exists()


class TestBusInboxCursorBounds:
    @pytest.mark.parametrize("value", ["999999", "-5"])
    def test_out_of_bounds_cursor_is_a_fail_naming_file_and_value(self, tmp_path: Path, value: str) -> None:
        repo = _init_repo(tmp_path)
        d = tmp_path / "bus"
        (d / "cursors").mkdir(parents=True)
        (d / "messages.jsonl").write_text(
            json.dumps({"id": "m1", "from": "chief", "to": "sa", "kind": "info", "body": "must-read"}) + "\n"
        )
        (d / "cursors" / "sa.offset").write_text(value)
        chk = swarm_check.bus_inbox_check(_cfg(tmp_path, repo, bus=str(d), bus_agent="sa"))
        assert chk.status == "FAIL" and "sa.offset" in chk.detail and value in chk.detail
