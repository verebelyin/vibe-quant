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
    from collections.abc import Callable, Iterator, Mapping, Sequence

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
    # data/swarm/ is gitignored in the real repo: the per-run gate logs live there and must
    # not show as untracked (the fixtures' main checkout IS the gated repo).
    (repo / ".gitignore").write_text("data/swarm/\n")
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "keep.py").write_text("X = 1\n")
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 1\n")
    tests = repo / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_foo.py").write_text(
        "from vibe_quant.foo import foo\n\n\ndef test_foo() -> None:\n    assert foo()\n"
    )


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
    swarm_check.Check("tests", "PASS", "1 passed in 0.01s | 1 importer files"),
    swarm_check.Check("exactness", "SKIP", "disabled"),
    swarm_check.Check("exactness-carry", "SKIP", "disabled"),
    swarm_check.Check("frontend", "SKIP", "disabled"),
)
_PASS_SCOPES = ((), ("vibe_quant/*",))  # the two scopes TestPassCache gates under
_COMMITTED_FOO = "def foo() -> int:\n    return 2\n"  # foo.py's content on the feature branch
_HEAD = "a" * 40  # the DeclaredGit HEAD: constant, so a key does not depend on a real sha

_STAMPED_TEMPLATE: Path | None = None


def _pass_runner(repo: Path, **declared: Any) -> DeclaredGit:
    """`_pass_repo`'s git state: foo.py committed on the feature branch, clean tree.

    Declared, not spawned: the state is fixed and one gate run asks git ~20 times.
    """
    return DeclaredGit(repo, committed=["vibe_quant/foo.py"], **declared)


def _stamped_template() -> Path:
    """_pass_repo's state (feature branch, foo changed) with a valid PASS stamp for every
    scope TestPassCache uses, written by swarm_check.write_stamp and checked by load_stamp."""
    global _STAMPED_TEMPLATE
    if _STAMPED_TEMPLATE is None or not _STAMPED_TEMPLATE.exists():
        root = Path(tempfile.mkdtemp(prefix="swarm-check-stamped-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        repo = root / "wt"
        shutil.copytree(_template_repo(), repo, symlinks=True)
        (repo / "vibe_quant" / "foo.py").write_text(_COMMITTED_FOO)
        runner = _pass_runner(repo)
        for scope in _PASS_SCOPES:
            cfg = swarm_check.Config(
                worktree=repo, main_root=root, base="main",
                scope=scope, exactness="never", frontend="never",
            )
            key = swarm_check._cache_probe(cfg, runner)[0]
            assert key
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


@pytest.fixture(autouse=True)
def _no_real_t3_history_db() -> Iterator[None]:
    """No gate test may read the caller's real T3 history DB (a ~4 s, 10 GB sqlite file).

    The stale-reads check looks the maker up in T3's state; here the default points nowhere,
    so an unrelated gate test SKIPs it. A test that wants a DB sets $SWARM_T3_DB in its body
    (a function-scoped autouse fixture runs first, so the body's value wins). Own
    MonkeyPatch instance: a test calling ``monkeypatch.undo()`` must not restore the real path.
    """
    guard = pytest.MonkeyPatch()
    guard.delenv("SWARM_T3_DB", raising=False)
    guard.setattr(swarm_check, "T3_DB_DEFAULT", Path(tempfile.gettempdir()) / "no-t3-state.sqlite")
    yield
    guard.undo()


class DeclaredGit(FakeRunner):
    """Answers the gate's git questions from a DECLARED tree; tools still come from `responses`.

    A run asks git six distinct questions but spawns it ~20 times (merge-base and
    `rev-parse <base>^{commit}` three times each, diff and status before AND after the tools)
    and on a loaded host each spawn costs 25-40 ms - the spawns were most of this suite's wall.
    Real git output parsing stays pinned where it belongs: TestGitState builds real repos
    (rename, non-ASCII, bad ref, git errors), the .sh wrapper test runs the real script and the
    reworded-commit cache case rewords a real commit.

    `committed` is what `git diff <base>...HEAD` lists, `dirty`/`untracked` are working-tree
    entries. With none of the three given the state FOLLOWS THE FILES on disk (lazily, at the
    first git call, so a test writes files in any order and the diff is what git would report
    against the template's committed content). Anything else is delegated to real git.

    It answers by COMMAND, not by arguments: `rev-list`, `merge-base`, `diff`, `status` and
    `show <ref>:<path>` ignore the ref and flags. So every such question also has a test
    against real git (reflog count: test_a_commit_then_reset_moves_the_head_reflog_count;
    base ref: TestPinnedText real-git test; the rest: TestGitState / TestBaseSelection).
    """

    def __init__(
        self,
        repo: Path,
        *,
        branch: str = "feature",
        origin: bool = False,
        committed: Sequence[str] | None = None,
        dirty: Sequence[str] | None = None,
        untracked: Sequence[str] | None = None,
        head: str = "a" * 40,
        tree: str = "b" * 40,
        base_sha: str = "c" * 40,
        merge_base: str = "c" * 40,
        reflog: str = "1",
        blobs: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.repo = repo
        self.branch = branch
        self.origin = origin
        self.head, self.tree = head, tree
        self.base_sha, self.merge_base, self.reflog = base_sha, merge_base, reflog
        self.committed: list[str] | None = None
        self.porcelain: list[str] | None = None
        if not (committed is None and dirty is None and untracked is None):
            self.committed = list(committed or ())
            self.porcelain = [f" M {p}" for p in dirty or ()] + [
                f"?? {p}" for p in untracked or ()
            ]
        self.blobs = dict(blobs or {})

    def _from_disk(self) -> None:
        if self.committed is not None and self.porcelain is not None:
            return
        committed, dirty, untracked = _declared_state(self.repo)
        self.committed = committed
        self.porcelain = [f" M {p}" for p in dirty] + [f"?? {p}" for p in untracked]

    def _git(self, args: list[str]) -> tuple[int, str] | None:
        """(rc, output) for a git command, or None to let real git answer it."""
        if not args:
            return None
        self._from_disk()
        assert self.committed is not None and self.porcelain is not None
        if args[0] == "symbolic-ref":
            return 0, self.branch + "\n"
        if args[0] == "merge-base":
            return 0, self.merge_base + "\n"
        if args[0] == "diff":
            return 0, "".join(p + "\0" for p in self.committed)
        if args[0] == "status":
            return 0, "".join(e + "\0" for e in self.porcelain)
        if args[0] == "rev-list":
            return 0, self.reflog + "\n"
        if args[0] == "show" and len(args) > 1 and ":" in args[1]:
            path = args[1].split(":", 1)[1]
            if path in self.blobs:
                return 0, self.blobs[path]
            disk = self.repo / path
            return (0, disk.read_text(encoding="utf-8")) if disk.is_file() else (1, "")
        if args[0] == "rev-parse":
            for ref in args[1:]:
                if ref == "--absolute-git-dir":
                    return 0, str(self.repo / ".git") + "\n"
                if ref.startswith("-"):
                    continue
                if ref.startswith("origin/"):
                    # no remote unless the test declares one: resolve_base keeps the local base
                    return (0, self.base_sha + "\n") if self.origin else (1, "")
                if ref.endswith("^{tree}"):
                    return 0, self.tree + "\n"
                if ref == "HEAD" or ref.endswith("^{commit}") or ref == "HEAD^{commit}":
                    return 0, (self.head if ref == "HEAD" else self.base_sha) + "\n"
            return 1, ""
        return None

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int = 600,
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        if c[0] == "git" and len(c) > 3 and c[1] == "-C":
            self.calls.append(c)
            self.timeouts.append(timeout)
            self.envs.append(env)
            declared = self._git(c[3:])
            if declared is not None:
                return declared
        return super().__call__(cmd, cwd, env, timeout)


#: The template's committed content; `_declared_state` diffs a test's tree against it, so a
#: test keeps writing files the way it always did and the declared diff follows the files.
_TEMPLATE_FILES = (
    ".gitignore",
    "vibe_quant/keep.py",
    "vibe_quant/foo.py",
    "tests/unit/test_foo.py",
)


def _declared_state(repo: Path) -> tuple[list[str], list[str], list[str]]:
    """(committed, dirty, untracked) for a copied template repo, from the files on disk."""
    template = _template_repo()
    dirty: list[str] = []
    for rel in _TEMPLATE_FILES:
        if (repo / rel).read_text(encoding="utf-8") != (template / rel).read_text(encoding="utf-8"):
            dirty.append(rel)
    known = set(_TEMPLATE_FILES)
    untracked: list[str] = []
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo).as_posix()
        if not path.is_file() or rel in known or "__pycache__" in path.parts:
            continue
        if rel.startswith(".git/") or rel.startswith("data/swarm/"):
            continue  # .git is not the worktree; data/swarm/ is gitignored
        untracked.append(rel)
    return [], dirty, untracked


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


_TIERS_SCRIPT = "scripts/agents/exactness_tiers.py"
_GATE_SCRIPT = "scripts/agents/gate_golden.py"
_GOLDENS = ("data", "swarm", "_goldens")
_TIERS_OK = "C1-defaults: IDENTICAL\nC8-1d-isolated-125: IDENTICAL\n"
_GATE_OK = "IDENTICAL\n"


def _install_goldens(
    tmp_path: Path,
    repo: Path,
    *,
    main_scripts: bool = True,
    worktree_scripts: bool = True,
    main_text: str = "# main\n",
    worktree_text: str | None = None,
) -> tuple[Path, Path]:
    """Main checkout's goldens (cfg.main_root == tmp_path) plus both golden scripts.

    The worktree scripts are written too unless `worktree_scripts` is False (they are the
    untracked files that make the diff non-empty in these tests)."""
    for rel in (_TIERS_SCRIPT, _GATE_SCRIPT):
        if main_scripts:
            main_script = tmp_path / rel
            main_script.parent.mkdir(parents=True, exist_ok=True)
            main_script.write_text(main_text)
        if worktree_scripts:
            wt_script = repo / rel
            wt_script.parent.mkdir(parents=True, exist_ok=True)
            wt_script.write_text(main_text if worktree_text is None else worktree_text)
    tiers_path = tmp_path.joinpath(*_GOLDENS, "tiers.json")
    gate_path = tmp_path.joinpath(*_GOLDENS, "gate_golden.json")
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text("{}\n")
    gate_path.write_text("{}\n")
    return tiers_path, gate_path


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

    def test_partial_scope_fail_lists_every_offender(self, tmp_path: Path) -> None:
        """A --scope narrower than the diff must name EVERY changed path outside it: the maker
        fixes the scope from this line, not from a truncated first five (bead aq9zy item 3)."""
        repo = _init_repo(tmp_path)
        names = ("b", "c", "d", "e", "f", "g")
        for name in names:
            (repo / "vibe_quant" / f"{name}.py").write_text(f"X = '{name}'\n")
        cfg = _cfg(tmp_path, repo, scope=("vibe_quant/foo.py",),
                   exactness="never", frontend="never")
        scope = {c.name: c for c in swarm_check.run_checks(cfg, DeclaredGit(repo))}["scope"]
        assert scope.status == "FAIL"
        assert "6/6 paths out of scope" in scope.detail
        for name in names:  # the 6th offender is dropped by a first-five cut
            assert f"vibe_quant/{name}.py" in scope.detail
        assert "more)" not in scope.detail


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

    def _tree(self, tmp_path: Path, bad: str = "") -> dict[str, set[str]]:
        """A tiny worktree: two caller tests (one nested); `bad` adds a broken or binary file."""
        (tmp_path / "vibe_quant" / "discovery").mkdir(parents=True)
        (tmp_path / "vibe_quant" / "discovery" / "pipeline.py").write_text("P = 1\n")
        if bad == "broken":
            (tmp_path / "vibe_quant" / "broken.py").write_text("def (:\n")
        if bad == "unreadable":
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

    @pytest.mark.parametrize(("bad", "why"), [("broken", "cannot parse"), ("unreadable", "cannot read")])
    def test_an_unparseable_or_unreadable_file_raises_and_the_plan_goes_full(
        self, tmp_path: Path, bad: str, why: str
    ) -> None:
        """Review rev-gate-speed-1 F4: a file contributing NO edges hid its importers, so a
        bad file is an unknown selection and the plan grows to the full suite."""
        with pytest.raises(swarm_check.GraphError, match=why):
            self._tree(tmp_path, bad)
        cfg = swarm_check.Config(worktree=tmp_path, main_root=tmp_path)
        state = swarm_check.GitState(changed=["vibe_quant/discovery/pipeline.py"], untracked=[])
        plan = swarm_check.plan_tests(cfg, state)
        assert plan.full and plan.files == ["tests/unit"]
        assert plan.notes[0].startswith("full tests/unit: selection unknown") and why in plan.notes[0]

    def test_a_scanner_read_error_also_goes_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._tree(tmp_path)
        real = Path.read_text

        def flaky(path: Path, *args: Any, **kwargs: Any) -> str:
            if path.name == "test_plainer.py" and kwargs.get("errors") == "replace":
                raise OSError("injected")  # the graph read passed; a mention scan fails
            return real(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky)
        cfg = swarm_check.Config(worktree=tmp_path, main_root=tmp_path)
        state = swarm_check.GitState(changed=["vibe_quant/discovery/pipeline.py"], untracked=[])
        assert swarm_check.plan_tests(cfg, state).full

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


def _path_reader_repo(tmp_path: Path) -> Path:
    """The bead's red fixture: a test READS two vibe_quant modules by path, never importing them.

    tests/unit/test_state_manager_locking.py's real pattern is the ``_EXTERNAL_MODULES`` tuple
    of repo-relative paths plus ``(_REPO_ROOT / rel).read_text()``. A second test builds the
    same path from a directory and a file name, so both needle forms are exercised. Neither
    file imports the modules, so the reverse import closure never reaches them.
    """
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "feature")
    (repo / "vibe_quant" / "screening").mkdir()
    (repo / "vibe_quant" / "screening" / "replay_drift.py").write_text("R = 1\n")
    (repo / "vibe_quant" / "validation").mkdir()
    (repo / "vibe_quant" / "validation" / "consistency.py").write_text("C = 1\n")
    (repo / "tests" / "unit" / "test_state_manager_locking.py").write_text(
        "_EXTERNAL_MODULES = (\n"
        '    "vibe_quant/screening/replay_drift.py",\n'
        '    "vibe_quant/validation/consistency.py",\n'
        ")\n\n\n"
        "def test_state_consumers_never_touch_the_shared_connection() -> None:\n"
        "    for rel in _EXTERNAL_MODULES:\n"
        "        (_REPO_ROOT / rel).read_text()\n"
    )
    (repo / "tests" / "unit" / "test_split_path.py").write_text(
        "from pathlib import Path\n\n"
        '_ROOT = Path(__file__).parents[2] / "vibe_quant" / "screening"\n\n\n'
        "def test_read() -> None:\n"
        '    (_ROOT / "replay_drift.py").read_text()\n'
    )
    (repo / "tests" / "unit" / "test_unrelated.py").write_text(
        "def test_x() -> None:\n    assert True\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a test that reads the source by path")
    return repo


class TestPathMentionTestSelection:
    """tests that read a changed source file BY PATH are invisible to the import graph.

    tests/unit/test_state_manager_locking.py reads vibe_quant/screening/replay_drift.py and
    vibe_quant/validation/consistency.py through ``read_text()``, so neither `-k <stem>` nor
    the reverse import closure selects it when those modules change (bead olizd).
    """

    def test_the_import_closure_cannot_see_them(self, tmp_path: Path) -> None:
        repo = _path_reader_repo(tmp_path)
        changed = ["vibe_quant/screening/replay_drift.py"]
        assert swarm_check.importer_test_files(
            changed, swarm_check.build_import_graph(repo)
        ) == []

    @pytest.mark.parametrize(
        ("changed", "expected"),
        [
            (["vibe_quant/screening/replay_drift.py"],  # the literal relative path
             ["tests/unit/test_split_path.py", "tests/unit/test_state_manager_locking.py"]),
            (["vibe_quant/validation/consistency.py"],  # path literal only in the lock test
             ["tests/unit/test_state_manager_locking.py"]),
            (["vibe_quant/other.py"], []),
            (["tests/unit/test_state_manager_locking.py"], []),  # a changed test selects nothing
            (["scripts/agents/swarm_check.py"], []),  # only vibe_quant modules are needles
            ([], []),
        ],
    )
    def test_path_mentions_select_reading_tests(
        self, tmp_path: Path, changed: list[str], expected: list[str]
    ) -> None:
        repo = _path_reader_repo(tmp_path)
        assert swarm_check.path_mention_test_files(changed, repo) == expected

    def test_a_path_reading_test_fails_the_gate(self, tmp_path: Path) -> None:
        repo = _path_reader_repo(tmp_path)
        (repo / "vibe_quant" / "screening" / "replay_drift.py").write_text("R = 2\n")  # the change

        def respond(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            if any("test_state_manager_locking.py" in Path(a).name for a in cmd):
                return 1, "1 failed in 0.02s"
            return 0, "1 passed in 0.01s"

        runner = DeclaredGit(repo, dirty=["vibe_quant/screening/replay_drift.py"])
        runner.responses = {"pytest": respond}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["tests"].status == "FAIL"
        assert _called_with(runner, "test_state_manager_locking.py")
        assert swarm_check.gate_summary(checks) == ("GATE FAIL (1 failed)", 1)

    def test_path_reading_files_join_the_one_parallel_call(self, tmp_path: Path) -> None:
        repo = _path_reader_repo(tmp_path)
        (repo / "vibe_quant" / "screening" / "replay_drift.py").write_text("R = 2\n")
        runner = DeclaredGit(repo, dirty=["vibe_quant/screening/replay_drift.py"])
        runner.responses = {"pytest": (0, "9 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        pytest_cmds = [
            c[c.index("pytest") + 1:]
            for c in runner.calls
            if any("pytest" in Path(a).name for a in c)
        ]
        assert pytest_cmds == [
            ["tests/unit/test_split_path.py", "tests/unit/test_state_manager_locking.py",
             "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
        ]
        assert "2 importer files" in tests.detail


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
        runner = DeclaredGit(repo)
        runner.responses = {"ruff": (0, "All checks passed!"), "mypy": (0, "Success")}
        (repo / "notes.md").write_text("x\n")  # non-empty diff (.md: no reader test here)
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by_name = {c.name: c for c in checks}
        assert [c.name for c in checks] == [
            "scope", "clean", "bus-inbox", "stale-reads", "ruff", "mypy", "tests", "pinned-text",
            "exactness", "exactness-carry", "exactness-tiers", "gate-golden", "parity",
            "frontend",
        ]  # fmt: skip
        assert by_name["scope"].status == "SKIP"
        assert by_name["bus-inbox"].status == "SKIP"
        assert by_name["stale-reads"].status == "SKIP"
        assert by_name["tests"].status == "SKIP"
        assert by_name["pinned-text"].status == "SKIP"
        assert by_name["exactness"].status == "SKIP"
        assert by_name["exactness-carry"].status == "SKIP"
        assert by_name["exactness-tiers"].status == "SKIP"
        assert by_name["gate-golden"].status == "SKIP"
        assert by_name["parity"].status == "SKIP" and by_name["parity"].detail == "disabled"
        assert by_name["frontend"].status == "SKIP"
        assert not _called_with(runner, "pytest")

    def test_scope_fail_lists_offenders(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        clean = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["clean"]
        assert clean.status == "WARN"
        assert swarm_check.format_line(clean).startswith("WARN clean:")
        assert "stray.txt" in clean.detail
        assert "cmd-report.md" not in clean.detail

    def test_ruff_failure_marks_check(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        runner = DeclaredGit(repo)
        runner.responses = {"ruff": (1, "E501 line too long"), "mypy": (0, "Success")}
        (repo / "notes.txt").write_text("x\n")  # non-empty diff
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by_name = {c.name: c for c in checks}
        assert by_name["ruff"].status == "FAIL"
        assert swarm_check.gate_summary(checks) == ("GATE FAIL (1 failed)", 1)

    def test_tests_selection_changed_test_and_module(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_fitness_names.py").write_text(
            "def test_fitness_rank() -> None:\n    pass\n"
        )  # committed, no import edge, names the stem only
        (repo / "vibe_quant" / "discovery").mkdir()
        (repo / "vibe_quant" / "discovery" / "fitness.py").write_text("F = 1\n")
        (repo / "tests" / "unit" / "test_bar.py").write_text("def test_bar() -> None:\n    pass\n")
        # base=HEAD: test_fitness_names.py is committed and unchanged (never a target)
        runner = DeclaredGit(
            repo, untracked=["tests/unit/test_bar.py", "vibe_quant/discovery/fitness.py"]
        )
        runner.responses = {"pytest": (0, "2 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["tests"].status == "PASS"
        pytest_cmds = [
            c[c.index("pytest") + 1:] for c in runner.calls if any("pytest" in Path(a).name for a in c)
        ]
        # a vibe_quant module gets no `-k <stem>` group: its import closure + path/string
        # references cover it, and a stem mostly matched namesakes of other modules
        assert pytest_cmds == [["tests/unit/test_bar.py", *_PAR]]  # no timing mark: no timing call

    def test_tests_failure_marks_check(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)
        runner.responses = {"exactness": (1, "DRIFT: sharpe 1.3 -> 1.4")}
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        assert {c.name: c for c in checks}["exactness"].status == "FAIL"

    def test_exactness_identical_but_nonzero_rc_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_exactness(tmp_path, repo)
        runner = DeclaredGit(repo)
        runner.responses = {"exactness": (1, "IDENTICAL (then crashed)")}
        cfg = _cfg(tmp_path, repo, exactness="always", frontend="never", base="HEAD")
        ex = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["exactness"]
        assert ex.status == "FAIL"

    def test_exactness_modified_script_in_worktree_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_exactness(tmp_path, repo)
        (repo / "scripts" / "agents" / "exactness_239.py").write_text("EXPECTED_SHARPE = 9\n")
        runner = DeclaredGit(repo)
        runner.responses = {"exactness": (0, "IDENTICAL")}
        cfg = _cfg(tmp_path, repo, exactness="always", frontend="never", base="HEAD")
        ex = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["exactness"]
        assert ex.status == "FAIL" and "modified in worktree" in ex.detail
        assert not _called_with(runner, "exactness_239.py")

    def test_exactness_missing_main_script_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        cfg = _cfg(tmp_path, repo, exactness="always", frontend="never", base="HEAD")
        ex = {c.name: c for c in swarm_check.run_checks(cfg, DeclaredGit(repo))}["exactness"]
        assert ex.status == "FAIL" and "missing" in ex.detail

    def test_exactness_skips_without_trigger(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "api").mkdir()
        (repo / "vibe_quant" / "api" / "app.py").write_text("A = 1\n")
        runner = DeclaredGit(repo)
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

        runner = DeclaredGit(repo)
        runner.responses = {"tsc": (0, ""), "pnpm": lint}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never")
        checks = swarm_check.run_checks(cfg, runner)
        fe = {c.name: c for c in checks}["frontend"]
        assert fe.status == "FAIL"
        assert "5" in fe.detail and "3" in fe.detail

    def test_frontend_equal_nonzero_lint_fails(self, tmp_path: Path) -> None:
        """Flat is no longer clean: main is expected at 0 lint errors (vibe-quant-g3z0q)."""
        status, detail = self._fe_status(tmp_path, {"tsc": (0, ""), "pnpm": (0, "Found 3 errors.")})
        assert status == "FAIL"
        assert "must be 0" in detail and "3" in detail

    def test_frontend_one_error_each_fails(self, tmp_path: Path) -> None:
        status, detail = self._fe_status(tmp_path, {"tsc": (0, ""), "pnpm": (0, "Found 1 error.")})
        assert status == "FAIL" and "must be 0" in detail

    def test_frontend_zero_lint_ok_when_main_has_errors(self, tmp_path: Path) -> None:
        def pnpm(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            return 0, "Found 3 errors." if cwd == tmp_path / "frontend" else "Found 0 errors."

        status, detail = self._fe_status(tmp_path, {"tsc": (0, ""), "pnpm": pnpm})
        assert status == "PASS"
        assert "lint errors 0 (main 3)" in detail

    def _fe_status(
        self, tmp_path: Path, responses: dict[str, object]
    ) -> tuple[str, str]:
        repo = _init_repo(tmp_path)
        (tmp_path / "frontend").mkdir()
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo, dirty=["vibe_quant/discovery/pipeline.py"])
        runner.responses = {"pytest": self._fails_on("test_symbol_agg.py")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by = {c.name: c for c in checks}
        assert by["tests"].status == "FAIL"
        assert "1 failed" in by["tests"].detail
        assert _called_with(runner, "test_symbol_agg.py")
        assert swarm_check.gate_summary(checks) == ("GATE FAIL (1 failed)", 1)

    def test_importer_files_run_as_one_parallel_call_without_timing_tests(
        self, tmp_path: Path
    ) -> None:
        """Per round the importer files run once, timing tests deselected (they run in
        --full); the -k group collects only files that can match it - here none is left
        once the importer file (the only one mentioning "pipeline") already runs."""
        repo = _importer_repo(tmp_path)
        runner = DeclaredGit(repo, dirty=["vibe_quant/discovery/pipeline.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit/test_symbol_agg.py", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
        ]
        assert "1 importer files" in tests.detail

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
        runner = DeclaredGit(repo, committed=["vibe_quant/discovery/selection.py"])
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
        runner = DeclaredGit(repo, dirty=["vibe_quant/discovery/pipeline.py", "tests/unit/test_symbol_agg.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit/test_symbol_agg.py", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
        ]
        assert "importer files" not in tests.detail

    def test_transitive_importer_through_a_function_level_import(self, tmp_path: Path) -> None:
        repo = _lazy_importer_repo(tmp_path)
        runner = DeclaredGit(repo, dirty=["vibe_quant/discovery/selection.py"])
        runner.responses = {"pytest": self._fails_on("test_caller.py")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "FAIL"
        assert _called_with(runner, "test_caller.py")

    def test_a_hub_module_runs_its_importer_files_never_the_full_suite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No cap any more (bead q2ijs): 400 importer files - more than tests/unit has today -
        still run as ONE file-list call; the full suite is the --full run's job."""
        repo = _importer_repo(tmp_path)
        (repo / "vibe_quant" / "hub.py").write_text("H = 1\n")  # the changed seed
        n = 400
        graph = {"vibe_quant/hub.py": set()} | {
            f"tests/unit/test_imp{i:03d}.py": {"vibe_quant.hub"} for i in range(n)
        }
        # the plan reads `import_graph_pair` (the graph AND its reverse index, both read-only)
        importers = {"vibe_quant.hub": {f"tests.unit.test_imp{i:03d}" for i in range(n)}}
        monkeypatch.setattr(
            swarm_check,
            "import_graph_pair",
            lambda worktree, extra=(), snapshot=None: (graph, importers),
        )
        runner = DeclaredGit(repo, dirty=["vibe_quant/hub.py"])
        runner.responses = {"pytest": (0, "9 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        cmds = self._pytest_cmds(runner)  # the importer files + the -k group's one file
        assert len(cmds) == 1 and "tests/unit" not in cmds[0]
        assert cmds[0][:n] == [f"tests/unit/test_imp{i:03d}.py" for i in range(n)]
        assert f"{n} importer files" in tests.detail
        assert "full tests/unit" not in tests.detail

    def test_scripts_agents_change_runs_no_importer_selection(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_gate_names.py").write_text(
            "def test_swarm_check_ok() -> None:\n    pass\n"
        )  # names the stem only: `-k swarm_check` reaches it, nothing else does
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "name-only test")
        (repo / "scripts" / "agents").mkdir(parents=True)
        (repo / "scripts" / "agents" / "swarm_check.py").write_text("X = 2\n")
        runner = DeclaredGit(repo, untracked=["scripts/agents/swarm_check.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit/test_gate_names.py", "-k", "swarm_check", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
        ]


_PAR = ["-n", "auto", "--dist", "worksteal", "-m", "not timing"]  # the per-round xdist tail

#: The always-run policy set, written out literally: _policy_repo builds exactly these files
#: and the selection must forward exactly them, so a rename/drop in swarm_check.POLICY_TESTS
#: is caught (asserting against the constant itself would move both sides together).
_POLICY_TESTS = [
    "tests/unit/test_exception_hygiene.py",
    "tests/unit/test_steering_budget.py",
    "tests/unit/test_api_routers_conn_locking.py",
    "tests/unit/test_discovery_liquidation.py",
    "tests/unit/test_script_exec_bits.py",
    "tests/unit/test_swarm_check_loaders.py",
]


#: The policy scans over all of vibe_quant/**/*.py (POLICY_TRIGGERS "vibe_quant/*.py").
_VQ_WIDE = ["tests/unit/test_discovery_liquidation.py", "tests/unit/test_swarm_check_loaders.py"]


def _policy_repo(tmp_path: Path) -> Path:
    """The policy tests committed on a feature branch; the diff changes ONE unrelated module
    that none of them imports (they scan the tree instead of the module under change)."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "feature")
    for rel in _POLICY_TESTS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_policy() -> None:\n    assert True\n")
    (repo / "vibe_quant" / "unrelated.py").write_text("U = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "policy tests and an unrelated module")
    (repo / "vibe_quant" / "unrelated.py").write_text("U = 2\n")  # the change
    return repo


def _policy_docs_repo(tmp_path: Path) -> Path:
    """`_policy_repo`'s tree with a DOC-only diff: no .py changes, but the policy tests SCAN
    the docs, so a steering-doc edit can turn test_steering_budget.py red with no python change
    - the selection must run POLICY_TESTS instead of skipping."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "feature")
    for rel in _POLICY_TESTS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_policy() -> None:\n    assert True\n")
    (repo / "docs" / "claude").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "claude" / "beads.md").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "policy tests and a steering doc")
    (repo / "docs" / "claude" / "beads.md").write_text("b2\n")  # the docs-only change
    return repo


def _importer_policy_repo(tmp_path: Path) -> Path:
    """`_importer_repo`'s tree plus the committed policy tests: the change (a vibe_quant
    module) queues importer commands BEFORE the targeted/policy call, so where the targets
    command lands in `commands` is observable."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-qb", "feature")
    for rel in _POLICY_TESTS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_policy() -> None:\n    assert True\n")
    (repo / "vibe_quant" / "discovery").mkdir()
    (repo / "vibe_quant" / "discovery" / "pipeline.py").write_text("P = 1\n")
    (repo / "tests" / "unit" / "test_symbol_agg.py").write_text(
        "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n\n\n"
        "def test_topk_ranks_gate_passers_first() -> None:\n"
        "    assert DiscoveryPipeline is not None\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "policy tests, the pipeline and its caller test")
    (repo / "vibe_quant" / "discovery" / "pipeline.py").write_text("P = 2\n")  # the change
    return repo


class TestPolicyTestSelection:
    """Repo-wide policy tests SCAN the tree instead of importing the changed module, so the
    reverse import closure can never select them: on 2026-10-18 a new silent handler in
    vibe_quant/discovery/neff.py went red in tests/unit/test_exception_hygiene.py although
    every gate PASSed. Every non-full selection appends the ones present (POLICY_TESTS).
    """

    @staticmethod
    def _pytest_cmds(runner: FakeRunner) -> list[list[str]]:
        return [
            c[c.index("pytest") + 1:]
            for c in runner.calls
            if any("pytest" in Path(a).name for a in c)
        ]

    def test_unrelated_module_change_still_runs_every_policy_test(self, tmp_path: Path) -> None:
        repo = _policy_repo(tmp_path)
        runner = DeclaredGit(repo, dirty=["vibe_quant/unrelated.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        # only the scans over vibe_quant/** (unrelated.py is outside the money packages, the
        # routers/discovery and the docs/scripts trees)
        assert self._pytest_cmds(runner)[0] == [*_VQ_WIDE, *_PAR]
        assert "2 policy tests" in tests.detail

    def test_policy_tests_follow_a_changed_test_file(self, tmp_path: Path) -> None:
        repo = _policy_repo(tmp_path)
        (repo / "tests" / "unit" / "test_touched.py").write_text(
            "def test_touched() -> None:\n    assert True\n"
        )
        runner = DeclaredGit(repo, dirty=["vibe_quant/unrelated.py"], untracked=["tests/unit/test_touched.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        swarm_check.run_checks(cfg, runner)
        assert self._pytest_cmds(runner)[0] == ["tests/unit/test_touched.py", *_VQ_WIDE, *_PAR]

    def test_full_suite_run_does_not_append_them_separately(self, tmp_path: Path) -> None:
        repo = _policy_repo(tmp_path)
        (repo / "tests" / "conftest.py").write_text("X = 1\n")  # forces the whole tests/unit
        runner = DeclaredGit(repo, dirty=["vibe_quant/unrelated.py"], untracked=["tests/conftest.py"])
        runner.responses = {"pytest": (0, "9 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [
            ["tests/unit", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
            ["tests/unit", "-m", "timing"],
        ]
        assert not _called_with(runner, "test_exception_hygiene.py")

    def test_absent_policy_files_are_not_forwarded(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)  # this tree carries no policy tests
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 3\n")
        runner = DeclaredGit(repo, dirty=["vibe_quant/foo.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner) == [["tests/unit/test_foo.py", *_PAR]]
        assert "policy tests" not in tests.detail

    def test_docs_only_change_still_runs_every_policy_test(self, tmp_path: Path) -> None:
        """A diff with no .py file (a steering-doc edit) can still turn a tree-scanning policy
        test red, so the selection must run POLICY_TESTS instead of returning an early SKIP."""
        repo = _policy_docs_repo(tmp_path)
        runner = DeclaredGit(repo, dirty=["docs/claude/beads.md"])
        runner.responses = {"pytest": (0, "4 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        # docs/claude/beads.md is a steering doc: only the word-budget scan reads it
        assert self._pytest_cmds(runner) == [["tests/unit/test_steering_budget.py", *_PAR]]
        assert "1 policy tests" in tests.detail

    def test_changed_policy_file_runs_once_plus_the_other_three(self, tmp_path: Path) -> None:
        """A changed POLICY file is itself a changed test file: it must appear exactly once,
        followed by the other three - the dedupe filter (drop it and the file runs twice)."""
        repo = _policy_repo(tmp_path)
        (repo / "tests" / "unit" / "test_steering_budget.py").write_text(
            "def test_policy() -> None:\n    assert True\n\n\n"
            "def test_another() -> None:\n    assert True\n"
        )  # the change: a policy file changed
        runner = DeclaredGit(repo, dirty=["tests/unit/test_steering_budget.py", "vibe_quant/unrelated.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        assert self._pytest_cmds(runner)[0] == [
            "tests/unit/test_steering_budget.py", *_VQ_WIDE, *_PAR,
        ]
        assert "2 policy tests" in tests.detail

    def test_policy_call_runs_ahead_of_the_importer_closure(self, tmp_path: Path) -> None:
        """Order pin: the policy files lead the one call, the importer closure follows."""
        repo = _importer_policy_repo(tmp_path)
        runner = DeclaredGit(repo, dirty=["vibe_quant/discovery/pipeline.py"])
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        cmds = self._pytest_cmds(runner)
        # discovery/pipeline.py: the money-package, routers/discovery and vibe_quant-wide scans
        expected = [
            "tests/unit/test_exception_hygiene.py",
            "tests/unit/test_api_routers_conn_locking.py",
            *_VQ_WIDE,
        ]
        assert cmds == [[*expected, "tests/unit/test_symbol_agg.py", *_PAR]]
        assert "4 policy tests" in tests.detail

    def test_policy_set_is_pinned(self) -> None:
        """The always-run set is what reopens the gap when a file is dropped: pin it."""
        assert list(swarm_check.POLICY_TESTS) == _POLICY_TESTS


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
        runner = DeclaredGit(repo)
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


class TestGoldenProofs:
    """exactness-tiers and gate-golden: the two golden proofs no gate ran before.

    Both compare against goldens kept OUT of git in the MAIN checkout's data/swarm/_goldens/
    and both run the WORKTREE's script (unlike exactness_239/carry, which run main's) with the
    worktree on PYTHONPATH. They run in --full (or --exactness always) only: a per-round gate
    must not pay ~90 s of real-catalog cells. A missing golden FAILs closed (never a SKIP):
    that is the drift 98cc4b86 shipped unnoticed.
    """

    def _checks(
        self,
        tmp_path: Path,
        repo: Path,
        responses: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[dict[str, swarm_check.Check], list[swarm_check.Check], FakeRunner]:
        runner = DeclaredGit(repo)
        if responses:
            runner.responses = responses  # type: ignore[assignment]
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never", **kwargs)
        checks = swarm_check.run_checks(cfg, runner)
        return {c.name: c for c in checks}, checks, runner

    def test_per_round_skips_both(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo)  # present and runnable: only the mode skips them
        by, _checks, runner = self._checks(tmp_path, repo, full=False)
        for name in ("exactness-tiers", "gate-golden"):
            assert by[name] == swarm_check.Check(name, "SKIP", "per-round: runs in --full")
        assert not _called_with(runner, "exactness_tiers.py")
        assert not _called_with(runner, "gate_golden.py")

    def test_exactness_always_runs_both_without_a_full_run(self, tmp_path: Path) -> None:
        """`--exactness always` is the other way in (a maker's explicit golden probe): it must
        run BOTH proofs, not only the ones a `--full` integration run would.
        Kills `if cfg.full or cfg.exactness == "always":` -> `if cfg.full:`."""
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo)
        _main_exactness(tmp_path, repo)  # --exactness always also runs exactness_239
        by, checks, runner = self._checks(
            tmp_path,
            repo,
            {
                "exactness_239.py": (0, "IDENTICAL"),
                "exactness_tiers.py": (0, _TIERS_OK),
                "gate_golden.py": (0, _GATE_OK),
            },
            full=False,
            exactness="always",
        )
        assert swarm_check.format_line(by["exactness-tiers"]) == (
            "PASS exactness-tiers: IDENTICAL"
        )
        assert swarm_check.format_line(by["gate-golden"]) == "PASS gate-golden: IDENTICAL"
        assert _called_with(runner, "exactness_tiers.py")
        assert _called_with(runner, "gate_golden.py")
        assert swarm_check.gate_summary(checks) == ("GATE PASS", 0)

    def test_exactness_never_disables_both(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo)
        by, _checks, runner = self._checks(tmp_path, repo, full=True, exactness="never")
        for name in ("exactness-tiers", "gate-golden"):
            assert by[name] == swarm_check.Check(name, "SKIP", "disabled")
        assert not _called_with(runner, "exactness_tiers.py")

    def test_full_runs_the_worktree_scripts_against_main_goldens(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        tiers_path, gate_path = _install_goldens(tmp_path, repo)
        by, checks, runner = self._checks(
            tmp_path,
            repo,
            {"exactness_tiers.py": (0, _TIERS_OK), "gate_golden.py": (0, _GATE_OK)},
            full=True,
        )
        assert by["exactness-tiers"] == swarm_check.Check("exactness-tiers", "PASS", "IDENTICAL")
        assert by["gate-golden"] == swarm_check.Check("gate-golden", "PASS", "IDENTICAL")
        assert swarm_check.gate_summary(checks) == ("GATE PASS", 0)

        tiers_call = next(
            c for c in runner.calls if any(str(a).endswith("exactness_tiers.py") for a in c)
        )
        assert tiers_call == [
            str(tmp_path / ".venv" / "bin" / "python"),
            str(repo / _TIERS_SCRIPT),
            "--check",
            str(tiers_path),
        ]
        index = runner.calls.index(tiers_call)
        assert runner.envs[index] == {"PYTHONPATH": str(repo)}  # the WORKTREE's code
        assert runner.timeouts[index] == swarm_check.TEST_TIMEOUT

        gate_call = next(
            c for c in runner.calls if any(str(a).endswith("gate_golden.py") for a in c)
        )
        assert gate_call[1:] == [str(repo / _GATE_SCRIPT), "--against", str(gate_path)]

    def test_tiers_drift_fails_with_the_capture_hint(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        tiers_path, _gate_path = _install_goldens(tmp_path, repo)
        by, checks, _runner = self._checks(
            tmp_path,
            repo,
            {
                "exactness_tiers.py": (
                    1,
                    "C1-defaults: IDENTICAL\nC3-isolated-125: DRIFT\n"
                    "  first diff $.screening.metrics.pf: 1.0 != 1.1\n",
                ),
                "gate_golden.py": (0, _GATE_OK),
            },
            full=True,
        )
        tiers_check = by["exactness-tiers"]
        assert tiers_check.status == "FAIL"
        assert "drift (rc=1): C3-isolated-125: DRIFT" in tiers_check.detail
        assert f"PYTHONPATH={repo}" in tiers_check.detail
        assert f"--capture {tiers_path}" in tiers_check.detail
        assert swarm_check.gate_summary(checks) == ("GATE FAIL (1 failed)", 1)

    def test_first_diff_line_is_the_evidence_when_no_drift_line_is_printed(self) -> None:
        assert (
            swarm_check._tiers_evidence("  first diff $.screening.metrics.pf: 1.0 != 1.1")
            == "first diff $.screening.metrics.pf: 1.0 != 1.1"
        )

    def test_tiers_rc_zero_with_a_drift_line_still_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo)
        by, _checks, _runner = self._checks(
            tmp_path,
            repo,
            {"exactness_tiers.py": (0, "C3-isolated-125: DRIFT\n"), "gate_golden.py": (0, _GATE_OK)},
            full=True,
        )
        assert by["exactness-tiers"].status == "FAIL"
        assert "C3-isolated-125: DRIFT" in by["exactness-tiers"].detail

    def test_tiers_identical_output_with_a_nonzero_rc_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo)
        by, _checks, _runner = self._checks(
            tmp_path,
            repo,
            {
                "exactness_tiers.py": (1, _TIERS_OK + "Traceback: capture crashed\n"),
                "gate_golden.py": (0, _GATE_OK),
            },
            full=True,
        )
        tiers_check = by["exactness-tiers"]
        assert tiers_check.status == "FAIL"
        assert "drift (rc=1)" in tiers_check.detail
        assert "Traceback: capture crashed" in tiers_check.detail

    def test_gate_golden_without_an_identical_line_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo)
        by, _checks, _runner = self._checks(
            tmp_path,
            repo,
            {"exactness_tiers.py": (0, _TIERS_OK), "gate_golden.py": (0, "no evidence line")},
            full=True,
        )
        gate_check = by["gate-golden"]
        assert gate_check.status == "FAIL"
        assert "drift (rc=0)" in gate_check.detail and "no evidence line" in gate_check.detail

    def test_gate_golden_drift_names_the_first_changed_line_and_the_recapture(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        _tiers_path, gate_path = _install_goldens(tmp_path, repo)
        by, _checks, _runner = self._checks(
            tmp_path,
            repo,
            {
                "exactness_tiers.py": (0, _TIERS_OK),
                "gate_golden.py": (
                    1,
                    "--- stored\n+++ current\n@@ -1 +1 @@\n"
                    '-  "gate_n": 55,\n+  "gate_n": 56,\n',
                ),
            },
            full=True,
        )
        gate_check = by["gate-golden"]
        assert gate_check.status == "FAIL"
        assert '-  "gate_n": 55,' in gate_check.detail
        assert f"--out {gate_path}" in gate_check.detail

    def test_missing_main_golden_fails_closed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        tiers_path, _gate_path = _install_goldens(tmp_path, repo)
        tiers_path.unlink()
        by, _checks, runner = self._checks(
            tmp_path,
            repo,
            {"exactness_tiers.py": (0, _TIERS_OK), "gate_golden.py": (0, _GATE_OK)},
            full=True,
        )
        tiers_check = by["exactness-tiers"]
        assert tiers_check.status == "FAIL"
        assert "main golden missing" in tiers_check.detail and str(tiers_path) in tiers_check.detail
        assert not _called_with(runner, "exactness_tiers.py")

    def test_missing_worktree_script_fails_closed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo, worktree_scripts=False)
        (repo / "notes.md").write_text("x\n")  # a diff without the scripts
        by, _checks, runner = self._checks(
            tmp_path,
            repo,
            {"exactness_tiers.py": (0, _TIERS_OK), "gate_golden.py": (0, _GATE_OK)},
            full=True,
        )
        assert by["exactness-tiers"].status == "FAIL"
        assert "worktree script missing" in by["exactness-tiers"].detail
        assert not _called_with(runner, "exactness_tiers.py")

    def test_script_changed_in_this_diff_is_named(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _install_goldens(tmp_path, repo, worktree_text="# worktree edit\n")
        by, _checks, _runner = self._checks(
            tmp_path,
            repo,
            {"exactness_tiers.py": (0, _TIERS_OK), "gate_golden.py": (0, _GATE_OK)},
            full=True,
        )
        assert by["exactness-tiers"] == swarm_check.Check(
            "exactness-tiers", "PASS", "IDENTICAL (script changed in this diff)"
        )
        assert by["gate-golden"].detail == "IDENTICAL (script changed in this diff)"


class TestParityGate:
    """parity: a test-deleting diff must not LOSE a mutant kill (job 20261028-prune).

    13 of 16 reviews found a mutant the deleted tests killed on base and the remaining tests
    no longer kill. The check runs the WORKTREE's parity_mutants.py against the worktree (the
    change under review is what gets checked) and FAILs only on a loss the script confirmed
    against every test file that names the module: a screening candidate ends with
    "  LOST KILL" and must never be quoted as the verdict.
    """

    _LOST = "LOST KILL vibe_quant/foo.py:2:cmp (killed at base by tests/unit/test_foo.py::test_foo)"

    def _check(
        self,
        tmp_path: Path,
        repo: Path,
        responses: dict[str, object] | None = None,
        **kwargs: object,
    ) -> tuple[swarm_check.Check, list[swarm_check.Check], DeclaredGit]:
        runner = DeclaredGit(repo)
        if responses:
            runner.responses = responses  # type: ignore[assignment]
        cfg = _cfg(tmp_path, repo, base="HEAD", frontend="never", **kwargs)
        checks = swarm_check.run_checks(cfg, runner)
        return {c.name: c for c in checks}["parity"], checks, runner

    def _changed_test_repo(self, tmp_path: Path) -> Path:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_foo.py").write_text(
            "from vibe_quant.foo import foo\n\n\ndef test_foo() -> None:\n"
            "    assert foo()\n\n\n# reviewed: unchanged behaviour\n"
        )
        return repo

    def test_skips_per_round(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        parity, _, runner = self._check(tmp_path, repo)
        assert parity.status == "SKIP"
        assert parity.detail == "per-round: runs in --full"
        assert not _called_with(runner, "parity_mutants.py")

    def test_skips_when_exactness_is_never(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        parity, _, runner = self._check(tmp_path, repo, full=True, exactness="never")
        assert parity.status == "SKIP" and parity.detail == "disabled"
        assert not _called_with(runner, "parity_mutants.py")

    def test_full_with_no_changed_test_file_skips(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "docs").mkdir()
        (repo / "docs" / "notes.md").write_text("- a note\n")
        parity, _, runner = self._check(tmp_path, repo, full=True)
        assert parity.status == "SKIP" and parity.detail == "no test file changed"
        assert not _called_with(runner, "parity_mutants.py")

    def test_full_pass_quotes_the_parity_line(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        parity, _, runner = self._check(
            tmp_path,
            repo,
            full=True,
            responses={
                "parity_mutants.py": (0, "base=KILLED head=KILLED x\nPARITY compared=1 lost=0 errors=0")
            },
        )
        assert parity.status == "PASS"
        assert parity.detail == "PARITY compared=1 lost=0 errors=0"
        call = next(c for c in runner.calls if "parity_mutants.py" in Path(c[-1]).name or
                    any(a.endswith("parity_mutants.py") for a in c))
        assert call[0] == str(tmp_path / ".venv" / "bin" / "python")
        assert call[1] == str(repo / "scripts" / "agents" / "parity_mutants.py")
        assert call[2:] == ["--repo", str(repo), "--base", "HEAD"]
        index = runner.calls.index(call)
        assert runner.timeouts[index] == swarm_check.TEST_TIMEOUT
        assert runner.envs[index] == {"PYTHONPATH": str(repo)}

    def test_full_pass_quotes_the_parity_line_not_trailing_stderr(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        summary = "PARITY compared=0 lost=0 errors=0 (no reachable candidate: skipped none)"
        output = f"{summary}\ncandidate vibe_quant/data/mark_bars.py:63:del (priority 2)"
        parity, _, _ = self._check(
            tmp_path, repo, full=True, responses={"parity_mutants.py": (0, output)}
        )
        assert parity.status == "PASS"
        assert parity.detail == summary

    def test_full_lost_kill_fails_the_gate(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        screened = "base=KILLED head=SURVIVED vibe_quant/foo.py:2:cmp  LOST KILL"
        output = "\n".join([screened, self._LOST, "PARITY compared=1 lost=1 errors=0"])
        parity, checks, _ = self._check(
            tmp_path, repo, full=True, responses={"parity_mutants.py": (1, output)}
        )
        assert parity.status == "FAIL"
        assert self._LOST in parity.detail
        assert "base=KILLED" not in parity.detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_full_fail_quotes_at_most_three_confirmed_losses(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        confirmed = [f"LOST KILL m{i} (killed at base by tests/unit/test_foo.py::test_foo)" for i in range(5)]
        screened = ["base=KILLED head=SURVIVED x  LOST KILL" for _ in range(4)]
        output = "\n".join([*screened, *confirmed, "PARITY compared=5 lost=5 errors=0"])
        parity, _, _ = self._check(
            tmp_path, repo, full=True, responses={"parity_mutants.py": (1, output)}
        )
        assert parity.status == "FAIL"
        assert "LOST KILL m2 " in parity.detail
        assert "LOST KILL m3 " not in parity.detail
        assert "base=KILLED" not in parity.detail

    def test_full_error_fails_with_the_error_line(self, tmp_path: Path) -> None:
        repo = self._changed_test_repo(tmp_path)
        parity, _, _ = self._check(
            tmp_path,
            repo,
            full=True,
            responses={"parity_mutants.py": (2, "boom\nPARITY error: unknown base rev: nope")},
        )
        assert parity.status == "FAIL"
        assert parity.detail == "PARITY error: unknown base rev: nope"


class TestFailClosed:
    def test_bad_base_fails_gate(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 5\n")
        _git(repo, "commit", "-qam", "change")
        # git rejects the ref itself, so this one keeps the real binary
        checks = swarm_check.run_checks(cfg, FakeRunner()) if (cfg := _cfg(
            tmp_path, repo, base="mian", exactness="auto", frontend="auto")) else []
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

    def test_pytest_rc5_on_importer_call_is_skip_not_fail(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = DeclaredGit(repo)
        runner.responses = {"pytest": (5, "5 deselected in 0.01s")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "SKIP" and "no non-timing tests in the selected files" in tests.detail

    def test_pytest_rc5_on_changed_test_file_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_bar.py").write_text("X = 1\n")
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)
        runner.responses = {"pytest": (0, "1 passed"), "exactness": (0, "IDENTICAL")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="always", frontend="never")
        swarm_check.run_checks(cfg, runner)
        assert swarm_check.TEST_TIMEOUT in runner.timeouts

    def test_stubs_refused_without_allow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _init_repo(tmp_path)
        monkeypatch.delenv("SWARM_CHECK_ALLOW_STUBS", raising=False)
        monkeypatch.setenv("SWARM_CHECK_RUFF", "/bin/true")
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        stubs = {c.name: c for c in checks}["stubs"]
        assert stubs.status == "WARN" and "NOTE stub active: SWARM_CHECK_RUFF=/bin/true" in stubs.detail
        assert any(c[0] == "/bin/true" for c in runner.calls)
        assert swarm_check.to_json_dict(checks)["stubs"]["status"] == "WARN"

    def test_empty_diff_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, DeclaredGit(repo))
        scope = {c.name: c for c in checks}["scope"]
        assert scope.status == "FAIL" and "empty diff vs HEAD" in scope.detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    @pytest.mark.parametrize(
        "path",
        [
            "tests/conftest.py",
            "vibe_quant/__init__.py",
            "pyproject.toml",  # pytest config + deps
            "uv.lock",
            "scripts/tool/__main__.py",  # no module name: no import closure, -k drops the stem
        ],
    )
    def test_conftest_or_init_only_runs_full_unit_suite(self, tmp_path: Path, path: str) -> None:
        repo = _init_repo(tmp_path)
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("X = 1\n")
        runner = DeclaredGit(repo)
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
        runner = DeclaredGit(repo)

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

    def test_vibe_quant_main_module_is_no_full_suite_trigger(self, tmp_path: Path) -> None:
        """discovery/__main__.py is a module: it seeds the import closure (bead q2ijs - it
        used to force the full suite every round)."""
        assert swarm_check.full_suite_reason(["vibe_quant/discovery/__main__.py"]) == ""
        assert swarm_check.full_suite_reason(["vibe_quant/discovery/pipeline.py", "x.md"]) == ""
        assert "uv.lock" in swarm_check.full_suite_reason(["a.md", "uv.lock"])
        # exact path only: elsewhere it is an unmodelled resource, not the pytest config
        assert "non-Python resource" in swarm_check.full_suite_reason(["docs/pyproject.toml"])

    def test_importer_call_runs_parallel_without_timing_tests(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 9\n")
        runner = DeclaredGit(repo)
        runner.responses = {"pytest": (0, "1 passed")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "PASS"
        pytest_cmds = [c for c in runner.calls if any("pytest" in Path(a).name for a in c)]
        assert [c[c.index("pytest") + 1:] for c in pytest_cmds] == [["tests/unit/test_foo.py", *_PAR]]

    def test_changed_test_file_runs_in_the_parallel_call_then_its_timing_tests(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_bar.py").write_text(
            "import pytest\n\n\n@pytest.mark.timing\ndef test_bar() -> None:\n    pass\n"
        )
        (repo / "tests" / "unit" / "test_plain.py").write_text("def test_p() -> None:\n    pass\n")
        runner = DeclaredGit(repo)
        runner.responses = {"pytest": (0, "1 passed")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        swarm_check.run_checks(cfg, runner)
        pytest_cmds = [c for c in runner.calls if any("pytest" in Path(a).name for a in c)]
        # a changed test file runs whole: its relative-speed tests serially, uncontended;
        # only the file that has a timing mark gets the serial call
        assert [c[c.index("pytest") + 1:] for c in pytest_cmds] == [
            ["tests/unit/test_bar.py", "tests/unit/test_plain.py", *_PAR],
            ["tests/unit/test_bar.py", "-m", "timing"],
        ]

    @pytest.mark.parametrize(
        ("text", "marked"),
        [
            ("@pytest.mark.timing\ndef test_x(): ...\n", True),
            ('pytestmark = [pytest.mark.timing]\n', True),
            ('pytestmark = getattr(pytest.mark, "timing")\n', True),
            ("def test_timing_of_fills(): ...\n", False),
        ],
    )
    def test_has_timing_mark(self, tmp_path: Path, text: str, marked: bool) -> None:
        (tmp_path / "test_t.py").write_text(text)
        assert swarm_check.has_timing_mark(tmp_path / "test_t.py") is marked

    def test_internal_crash_is_fail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _init_repo(tmp_path)

        def boom(cfg: object, run: object = None, facts: object = None) -> None:
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
        assert key != swarm_check.cache_key("h", "s", base, "t", "merge-base-2")

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
            "data/swarm/_goldens/tiers.json", "data/swarm/_goldens/gate_golden.json",
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
    """The PASS stamp: written only for a clean tree whose key still holds, never written by a
    run that could not be trusted, and never served when it no longer matches the tree.

    The cases that assert a stamp is NOT reused assert `load_stamp`/`cache_key` - the same
    verdict a whole gate run pays ~20 git spawns to reach; the reuse path stays end to end in
    test_pass_writes_stamp_then_reused, the mid-run guards in one run each, and the key's
    inputs in TestCacheKey.
    """

    def _stub(self, repo: Path) -> DeclaredGit:
        return self._declared(repo)

    def _declared(self, repo: Path, **declared: Any) -> DeclaredGit:
        """A stubbed DeclaredGit: the mid-run guard tests move its state while the gate runs."""
        runner = _pass_runner(repo, **declared)
        runner.responses = self._responses()
        return runner

    def _pass_repo(self, tmp_path: Path) -> tuple[Path, FakeRunner]:
        """A clean feature-branch repo whose tree already has a valid PASS stamp for every
        scope below, so a run reuses it unless the test breaks that."""
        repo = _init_pass_repo(tmp_path)
        assert len(self._stamps(repo)) == len(_PASS_SCOPES)  # else the test passes for free
        return repo, self._stub(repo)

    def _unstamped_repo(self, tmp_path: Path) -> tuple[Path, FakeRunner]:
        """The same clean tree with NO stamp yet, so a run must really write one."""
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text(_COMMITTED_FOO)
        return repo, self._stub(repo)

    @staticmethod
    def _responses() -> dict[str, Any]:
        return {
            "ruff": (0, "All checks passed!"),
            "mypy": (0, "Success"),
            "pytest": (0, "1 passed in 0.01s"),
        }

    def _runner(self, repo: Path | None = None, **declared: Any) -> FakeRunner:
        """A real-git runner; `repo` + kwargs build a declared one (see DeclaredGit)."""
        runner: FakeRunner = FakeRunner() if repo is None else _pass_runner(repo, **declared)
        runner.responses = self._responses()
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
        repo, first = self._unstamped_repo(tmp_path)
        assert self._main(repo, first) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert _called_with(first, "pytest")  # expensive tools really ran
        assert len(self._stamps(repo)) == 1
        assert "swarm-check" not in _git(repo, "status", "--porcelain")  # stamp unseen by git

        second = self._runner(repo)
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
        assert data["tests"] == {"status": "PASS", "detail": "1 passed in 0.01s | 1 importer files"}

    def test_untampered_stamped_template_is_reused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Positive control for every not-reused test below: if the stamped template's key ever
        diverged from main()'s key, they would all pass without proving anything."""
        repo, runner = self._pass_repo(tmp_path)
        assert self._main(repo, runner) == 0
        assert "PASS cache: reused" in capsys.readouterr().out
        assert not _called_with(runner, "pytest")

    def test_reworded_commit_reuses_the_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """HANDOFF R3: the key is HEAD^{tree} + base + merge-base, so rewording the commit after
        a PASS (17 makers did, ~14.5 min of re-runs) is served from the stamp. Real git: this is
        the one cache case where the history itself has to move."""
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-qb", "feature")
        (repo / "vibe_quant" / "foo.py").write_text(_COMMITTED_FOO)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "change foo")
        assert self._main(repo, self._runner()) == 0
        assert len(self._stamps(repo)) == 1
        old_head = _git(repo, "rev-parse", "HEAD").strip()
        _git(repo, "commit", "--amend", "-qm", "reworded message only")
        assert _git(repo, "rev-parse", "HEAD").strip() != old_head
        runner = self._runner()
        assert self._main(repo, runner) == 0
        assert "PASS cache: reused" in capsys.readouterr().out
        assert not _called_with(runner, "pytest")

    def test_a_moved_tree_is_not_served_from_the_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A commit that changes the tree is a new key: the stamp of the old tree is never
        served (the reverse of test_untampered_stamped_template_is_reused)."""
        repo, _ = self._pass_repo(tmp_path)
        runner = self._runner(repo, tree="d" * 40)
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    @pytest.mark.parametrize("moved", ["base", "merge-base", "scope", "tool-hash"])
    def test_a_stamp_is_not_served_when_a_key_input_moved(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        moved: str,
    ) -> None:
        """Through the gate's own entry point: each input `_cache_probe` hands to cache_key
        (base sha, merge-base, scope, this script's hash) retires a stamp written before it
        moved. TestCacheKey pins cache_key's own inputs; this pins that main() passes them.
        The same row covers a stamp written by an older gate script (its tool hash differs)."""
        repo, _ = self._pass_repo(tmp_path)  # stamped for the declared base/merge-base/tool
        declared: dict[str, Any] = {}
        extra: tuple[str, ...] = ()
        if moved == "base":
            declared["base_sha"] = "d" * 40  # main advanced after the PASS
        elif moved == "merge-base":
            declared["merge_base"] = "e" * 40  # the branch was rebased onto a newer main
        elif moved == "scope":
            extra = ("--scope", "vibe_quant/*,tests/*")  # stamped for () and vibe_quant/*
        else:
            monkeypatch.setattr(swarm_check, "tool_hash", lambda: "0" * 64)
        runner = self._runner(repo, **declared)
        assert self._main(repo, runner, *extra) == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_a_commit_then_reset_moves_the_head_reflog_count(self, tmp_path: Path) -> None:
        """Real git: `commit` + `reset --hard` puts HEAD and the tree back, so only the reflog
        shows it moved - the question the N1 mid-run guard asks (`rev-list --walk-reflogs`).
        Without the flag git counts commits reachable from HEAD and the count stays equal."""
        repo = _init_repo(tmp_path)
        runner = FakeRunner()
        head = _git(repo, "rev-parse", "HEAD")
        tree = _git(repo, "rev-parse", "HEAD^{tree}")
        before = swarm_check.head_reflog_count(repo, runner)
        assert before is not None
        (repo / "vibe_quant" / "foo.py").write_text(_COMMITTED_FOO)
        _git(repo, "commit", "-qam", "temp")
        _git(repo, "reset", "-q", "--hard", "HEAD~1")
        assert _git(repo, "rev-parse", "HEAD") == head  # the key inputs are back where they were
        assert _git(repo, "rev-parse", "HEAD^{tree}") == tree
        after = swarm_check.head_reflog_count(repo, runner)
        assert after is not None and after > before

    def test_full_mode_does_not_reuse_a_round_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, _ = self._pass_repo(tmp_path)  # stamped by per-round runs
        runner = self._runner(repo)
        assert self._main(repo, runner, "--full") == 0
        assert "PASS cache:" not in capsys.readouterr().out
        assert any(
            c[c.index("pytest") + 1:][:1] == ["tests/unit"] for c in runner.calls if "pytest" in c
        )

    def test_no_cache_flag_forces_full_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, runner = self._pass_repo(tmp_path)  # already stamped: without --no-cache it is reused
        assert self._main(repo, runner, "--no-cache") == 0
        assert _called_with(runner, "pytest")
        assert "PASS cache:" not in capsys.readouterr().out

    def test_untracked_file_blocks_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, _ = self._unstamped_repo(tmp_path)
        (repo / "stray.txt").write_text("x\n")
        first = self._runner(repo, untracked=["stray.txt"])
        assert self._main(repo, first) == 0
        assert not self._stamps(repo)
        second = self._runner(repo, untracked=["stray.txt"])
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")  # nothing cached: the tree was never clean

    def test_uncommitted_change_blocks_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, _ = self._unstamped_repo(tmp_path)
        runner = self._runner(repo, dirty=["vibe_quant/foo.py"])
        assert self._main(repo, runner) == 0
        assert not self._stamps(repo)

    def test_fail_writes_no_stamp(self, tmp_path: Path) -> None:
        repo, runner = self._unstamped_repo(tmp_path)
        runner.responses["ruff"] = (1, "E501 boom")
        assert self._main(repo, runner) == 1
        assert not self._stamps(repo)

    def test_bus_only_fail_writes_stamp_and_next_run_is_cached(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """B3/B4: a bus-inbox-only FAIL still stamps the tool PASS, so the re-run after the DM
        is read reuses it instead of re-running the whole suite."""
        repo, _ = self._unstamped_repo(tmp_path)
        bus = tmp_path / "bus"
        bus.mkdir()

        def ruff_then_dm(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            # lands after the pre-check, before the end-of-run bus check: the tools really run
            # and only the (time-dependent) bus-inbox check fails.
            with (bus / "messages.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"id": "late1", "from": "chief", "to": "sa", "kind": "info",
                                "body": "stop"}) + "\n"
                )
            return 0, "All checks passed!"

        first = self._runner(repo)
        first.responses["ruff"] = ruff_then_dm
        assert self._main(repo, first, "--bus", str(bus), "--bus-agent", "sa") == 1
        out = capsys.readouterr().out
        assert "FAIL bus-inbox:" in out and "late1" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")
        assert _called_with(first, "pytest")  # the tool checks really ran (and failed only on bus)
        assert len(self._stamps(repo)) == 1  # B3: the tool PASS is stamped despite the bus FAIL

        # read the DM: advance the agent's cursor past the message; the tree is unchanged
        (bus / "cursors").mkdir(exist_ok=True)
        (bus / "cursors" / "sa.offset").write_text(str((bus / "messages.jsonl").stat().st_size))

        second = self._runner(repo)
        assert self._main(repo, second, "--bus", str(bus), "--bus-agent", "sa") == 0
        out2 = capsys.readouterr().out
        assert "PASS cache: reused stamp from" in out2  # B4: cached PASS, no suite re-run
        assert "FAIL bus-inbox:" not in out2
        assert out2.strip().endswith("GATE PASS")
        assert not _called_with(second, "pytest")
        assert not _called_with(second, "ruff") and not _called_with(second, "mypy")

    def test_commit_during_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """B1: a commit made while pytest runs must not be stamped (GATE PASS is for old HEAD).

        The commit moves the tree sha mid-run - declared, not spawned: the guard compares the
        pre/post keys, and this is exactly the state that comparison reads."""
        repo, _ = self._unstamped_repo(tmp_path)
        runner = self._declared(repo)
        done = False

        def commit_mid_run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            nonlocal done
            if not done:  # the gate runs pytest twice (parallel + timing)
                done = True
                runner.tree = "e" * 40  # the commit: a new tree for the same HEAD name
            return 0, "1 passed in 0.01s"

        runner.responses["pytest"] = commit_mid_run
        assert self._main(repo, runner) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert not self._stamps(repo)
        second = self._runner(repo, tree="e" * 40)
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")  # re-gated, never served from cache
        assert "PASS cache:" not in capsys.readouterr().out

    def test_dirty_start_committed_mid_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """B1: untracked at start (WARN) + committed during the run -> end state is clean, so
        only the PRE-run key (empty: the tree was dirty) keeps the stamp away."""
        repo, _ = self._unstamped_repo(tmp_path)
        (repo / "stray.py").write_text("X = 1\n")
        runner = self._declared(repo, untracked=["stray.py"])
        done = False

        def commit_mid_run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            nonlocal done
            if not done:  # the gate runs pytest twice (parallel + timing)
                done = True
                runner.porcelain = []  # the stray file got committed mid-run
            return 0, "1 passed in 0.01s"

        runner.responses["pytest"] = commit_mid_run
        assert self._main(repo, runner) == 0
        assert not self._stamps(repo)
        second = self._runner(repo, tree="e" * 40)
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")

    def test_commit_and_reset_mid_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """N1: commit + `reset --hard` puts HEAD and the tree back where they started, so the
        pre/post key compare matches; the HEAD reflog still moved, so no stamp is written."""
        repo, _ = self._unstamped_repo(tmp_path)
        runner = self._declared(repo)
        done = False

        def commit_then_reset(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            nonlocal done
            if not done:
                done = True
                runner.reflog = "3"  # commit + reset: the tree is back, the reflog moved
            return 0, "1 passed in 0.01s"

        runner.responses["pytest"] = commit_then_reset
        assert self._main(repo, runner) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert not self._stamps(repo)
        second = self._runner(repo)
        assert self._main(repo, second) == 0
        assert _called_with(second, "pytest")  # re-gated, never served from cache

    def test_unreadable_reflog_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """N1 fail closed: when git cannot report the HEAD reflog length the verdict is still
        printed, but no stamp is written (the mid-run guard could not run)."""

        class NoReflog(DeclaredGit):
            def _git(self, args: list[str]) -> tuple[int, str] | None:
                if args[:1] == ["rev-list"]:
                    return 128, "fatal: reflog unavailable"
                return super()._git(args)

        repo, _ = self._unstamped_repo(tmp_path)
        runner = NoReflog(repo, committed=["vibe_quant/foo.py"])
        runner.responses = self._runner().responses
        assert self._main(repo, runner) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert not self._stamps(repo)
        assert swarm_check.head_reflog_count(repo, runner) is None

    @pytest.mark.parametrize(
        "mutation", ["fail-line", "warn-line", "version", "gate"]
    )
    def test_a_stamp_that_does_not_match_is_never_served(
        self, tmp_path: Path, mutation: str
    ) -> None:
        """The decision main() makes for each of these is `load_stamp` -> None: a
        FAIL or WARN line, another CACHE_VERSION, a stored gate FAIL. Asserted at the edge (a
        full run per case bought nothing); a corrupt file and main()'s own no-reuse path are
        pinned by test_corrupt_stamp_forces_full_run and
        test_a_moved_tree_is_not_served_from_the_stamp.
        """
        repo, _ = self._pass_repo(tmp_path)
        runner = _pass_runner(repo)
        cfg = _cfg(tmp_path, repo, base="main", exactness="never", frontend="never")
        key = swarm_check._cache_probe(cfg, runner)[0]
        assert key and swarm_check.load_stamp(repo, runner, key) is not None
        path = swarm_check.stamp_path(repo, runner, key)
        assert path is not None
        data = json.loads(path.read_text())
        if mutation == "fail-line":
            data["checks"] = ["FAIL ruff: injected"] + data["checks"]
        elif mutation == "warn-line":
            data["checks"] = ["WARN clean: 1 untracked: x"] + data["checks"]
        elif mutation == "version":
            data["version"] = swarm_check.CACHE_VERSION + 1
        else:
            data["gate"] = "FAIL"
        path.write_text(json.dumps(data))
        assert swarm_check.load_stamp(repo, runner, key) is None, mutation

    def test_corrupt_stamp_forces_full_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, _ = self._pass_repo(tmp_path)
        for stamp in self._stamps(repo):
            stamp.write_text("{not json")
        runner = self._declared(repo)
        assert self._main(repo, runner) == 0
        assert _called_with(runner, "pytest")
        stamps = self._stamps(repo)
        assert len(stamps) == len(_PASS_SCOPES)
        # the full run rewrote its own stamp; the other corrupt one was never served
        assert len([p for p in stamps if "not json" not in p.read_text()]) == 1

    def test_active_stubs_never_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S1: stubs honoured (ALLOW_STUBS=1) -> the PASS came from fakes -> no stamp."""
        repo, runner = self._unstamped_repo(tmp_path)
        monkeypatch.setenv("SWARM_CHECK_ALLOW_STUBS", "1")
        monkeypatch.setenv("SWARM_CHECK_RUFF", "ruff-stub")
        assert self._main(repo, runner) == 0
        assert "stubs" in capsys.readouterr().out
        assert not self._stamps(repo)

    def test_stamp_write_is_atomic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """N2: the stamp appears only via os.replace of a temp file; no leftovers on failure."""
        repo, _ = self._unstamped_repo(tmp_path)

        def boom(src: object, dst: object) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(swarm_check.os, "replace", boom)
        assert self._main(repo, self._runner(repo)) == 0  # a stamp problem never fails the gate
        assert not self._stamps(repo)
        cache_dir = repo / ".git" / "swarm-check"
        assert not cache_dir.exists() or not list(cache_dir.iterdir())
        monkeypatch.undo()
        assert self._main(repo, self._runner(repo)) == 0
        assert len(self._stamps(repo)) == 1
        assert list((repo / ".git" / "swarm-check").iterdir()) == self._stamps(repo)

    def test_neither_a_run_nor_a_cache_hit_posts_to_the_bus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The verdict lives in the LOG (read by handoff_lint), never on the bus: gate posts were
        dead DMs nobody read (100 in one job's bus)."""
        repo, first = self._unstamped_repo(tmp_path)
        bus = tmp_path / "bus"
        bus.mkdir()  # a real job bus exists
        monkeypatch.setenv("SWARM_BUS", str(bus))
        assert self._main(repo, first) == 0
        assert self._main(repo, self._runner(repo)) == 0
        assert "PASS cache: reused stamp" in capsys.readouterr().out
        assert list(bus.iterdir()) == []


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
        assert "tests/unit -n auto" in log  # notes.txt: an unmodelled resource runs the full suite

    def test_script_gate_fail(self, tmp_path: Path) -> None:
        """The wrapper's FAIL path: an out-of-scope diff stops every tool (the failing mypy
        stub never runs) and the JSON says FAIL. One script run per direction (pass/fail);
        a tool FAIL's own line is pinned in-process by test_tool_fail_line_points_at_the_log.
        """
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
        # an out-of-scope diff stops before the tools: the failing mypy stub never runs
        assert "SKIP mypy: scope FAIL: fix scope first" in proc.stdout
        assert "SKIP tests: scope FAIL: fix scope first" in proc.stdout
        assert proc.stdout.strip().endswith("GATE FAIL (1 failed)")
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
        # B5: the FULL body is listed (no 80-char truncation) and the fix is named
        assert "m1" in chk.detail and "x" * 100 in chk.detail and "x" * 101 not in chk.detail
        assert chk.detail.endswith("— run: bus inbox")
        assert not (d / "cursors" / "sa.offset").exists()  # a FAIL must not consume the DM

    def test_fail_lists_every_unread_dm_with_its_full_body(self, tmp_path: Path) -> None:
        d = self._bus(
            tmp_path,
            [
                {"id": "a1", "from": "chief", "to": "sa", "kind": "info", "body": "alpha" * 40},
                {"id": "b2", "from": "chief", "to": "sa", "kind": "info", "body": "bravo" * 40},
            ],
        )
        chk = self._check(tmp_path, bus=str(d), bus_agent="sa")[0]
        assert chk.status == "FAIL"
        assert chk.detail.startswith("2 unread from chief: ")
        assert f"a1 {'alpha' * 40}" in chk.detail and f"b2 {'bravo' * 40}" in chk.detail
        assert chk.detail.endswith("— run: bus inbox")  # each id + full body, then the command

    def test_cached_pass_plus_unread_chief_dm_is_gate_fail(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A cached PASS does not excuse an unread chief DM: the pre-check sees an empty bus,
        then the DM lands and the cached-path bus check still fails the gate."""
        repo = _init_pass_repo(tmp_path)  # a clean tree with a valid PASS stamp: reuse path
        d = self._bus(tmp_path)  # empty bus, so the pre-check does not short-circuit
        calls: list[int] = []

        def dm_after_precheck(cfg: object) -> Any:
            calls.append(1)
            if len(calls) == 1:
                return swarm_check.Check("bus-inbox", "PASS", "pre-check: no unread")
            return swarm_check.Check(
                "bus-inbox", "FAIL", "1 unread from chief: m1 reread the brief — run: bus inbox"
            )

        monkeypatch.setattr(swarm_check, "bus_inbox_check", dm_after_precheck)
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(d), "--bus-agent", "sa"],
            run=_pass_runner(repo),
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "PASS cache: reused stamp from" in out  # the tool checks really were cached
        assert "FAIL bus-inbox:" in out and "m1" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")


class TestBusInboxPrecheck:
    """B1: an unread chief DM already waiting refuses the hand-back before any tool runs."""

    def test_unread_dm_fails_before_any_tool_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        d = tmp_path / "bus"
        d.mkdir()
        (d / "messages.jsonl").write_text(
            json.dumps(
                {"id": "m1", "from": "chief", "to": "sa", "kind": "question",
                 "body": "read the brief " * 20}
            )
            + "\n"
        )
        runner = DeclaredGit(repo)
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(d), "--bus-agent", "sa"],
            run=runner,
        )
        out = capsys.readouterr().out
        assert rc == 1
        bus_line = next(ln for ln in out.splitlines() if ln.startswith("FAIL bus-inbox:"))
        assert "m1" in bus_line and bus_line.endswith("— run: bus inbox")
        assert out.strip().endswith("GATE FAIL (1 failed)")
        assert runner.calls == []  # no tool (not even git) ran: the waiting DM refused it up front

    def test_precheck_does_not_consume_the_dm(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        d = tmp_path / "bus"
        d.mkdir()
        (d / "messages.jsonl").write_text(
            json.dumps({"id": "m1", "from": "chief", "to": "sa", "kind": "info", "body": "hi"}) + "\n"
        )
        assert swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(d), "--bus-agent", "sa"],
            run=DeclaredGit(repo),
        ) == 1
        # the FAIL is a peek: reading the DM is the worker's job (`bus inbox`), not the gate's
        assert not (d / "cursors" / "sa.offset").exists()


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
        runner = DeclaredGit(repo)

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

    def test_gate_result_never_goes_to_the_bus_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("SWARM_BUS", raising=False)
        monkeypatch.delenv("SWARM_AGENT", raising=False)
        repo = _init_pass_repo(tmp_path)
        d = tmp_path / "bus"
        d.mkdir()
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(d), "--bus-agent", "sa"],
            run=_pass_runner(repo),
        )
        assert rc == 0 and capsys.readouterr().out.strip().endswith("GATE PASS")
        assert not (d / "messages.jsonl").exists()

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
                run=_pass_runner(repo),
            )
            cap = capsys.readouterr()
            assert rc == 1 and "FAIL bus-inbox:" in cap.out and cap.out.strip().endswith("GATE FAIL (1 failed)")
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


class TestInheritedSwarmEnv:
    """A unit test must never read or post to a real swarm bus through the caller's env.

    The gate runs ``pytest tests/unit`` as a child that inherits the worker shell's env
    (swarm_check.py:218 merges ``os.environ``) while ``main`` resolves ``cfg.bus`` from
    ``$SWARM_BUS`` -- so the ``<worktree>: GATE PASS`` DMs that landed in the gate's own live
    job bus during a worker's gate run were written from inside these tests: every class above
    that called ``swarm_check.main()`` without ``--bus`` posted to whatever bus the env named
    (the gate no longer posts at all; an inherited SWARM_AGENT still makes bus-inbox read the
    real bus). tests/unit/conftest.py's
    ``_forbid_inherited_swarm_env`` strips the caller's SWARM_* vars before each test body.
    """

    def test_no_swarm_env_in_a_unit_test_body(self) -> None:
        """The guard. Run with SWARM_BUS/SWARM_AGENT exported this is the leak condition; the
        child-pytest test below recreates that condition, so the pin holds in a clean env too."""
        inherited = {k: v for k, v in os.environ.items() if k.startswith("SWARM_")}
        assert inherited == {}, f"a unit test inherited swarm env: {inherited}"

    def test_gate_run_from_a_test_posts_no_dm_even_with_an_explicit_bus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The leak path stays closed twice: the stripped env leaves the gate no bus, and a
        bus set deliberately (as cmd-task.sh exports it) receives nothing either."""
        repo = _init_pass_repo(tmp_path)
        argv = [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never"]
        assert swarm_check.main(argv, run=FakeRunner()) == 0
        assert "GATE PASS" in capsys.readouterr().out
        assert list(tmp_path.glob("**/messages.jsonl")) == []

        explicit = tmp_path / "explicit_bus"
        explicit.mkdir()
        monkeypatch.setenv("SWARM_BUS", str(explicit))
        monkeypatch.setenv("SWARM_AGENT", "sa")
        assert swarm_check.main(argv, run=FakeRunner()) == 0
        assert list(explicit.iterdir()) == []

    def test_no_swarm_env_resurrection_by_a_monkeypatch_undo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap the guard must not fall into: TestPassCache::test_stamp_write_is_atomic calls
        ``monkeypatch.undo()`` mid-test, which undoes everything the *shared* monkeypatch recorded
        -- so a guard built on it resurrects the caller's bus and the gate runs after the undo post
        into it (that test produced the last live-bus DM of a worker's gate run)."""
        monkeypatch.setenv("SWARM_BUS", "/nonexistent/live-bus")  # stands in for cmd-task.sh
        monkeypatch.undo()
        assert "SWARM_BUS" not in os.environ, "monkeypatch.undo() resurrected the caller's bus"

    def test_guard_holds_in_a_child_pytest_with_swarm_env_exported(self, tmp_path: Path) -> None:
        """End-to-end: the leak path is a *child* pytest of the gate, so run one with the worker
        shell's SWARM_BUS/SWARM_AGENT exported -- it selects the two ``no_swarm_env`` guard pins
        above plus ``gate_run_from_a_test``, the gate-running test whose bus asserts are the ones
        that have to fire, since the env they all need is exactly what this test provides -- and
        assert the child passes, that it really selected that gate test, and that the bus it
        points at stays empty. A guard that is missing, or that the shared ``monkeypatch`` can
        undo, fails there even when the outer run has a clean env."""
        live_bus = tmp_path / "live_bus"
        live_bus.mkdir()
        root = Path(__file__).resolve().parents[2]
        env = {
            **os.environ,
            "SWARM_BUS": str(live_bus),
            "SWARM_AGENT": "probe",
            "PYTHONPATH": str(root),
            # The gate exports PYTEST_ADDOPTS="-q ..."; its -q would cancel the -v below.
            "PYTEST_ADDOPTS": "",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(root / "tests/unit/test_swarm_check.py"),
             "-k", "no_swarm_env or gate_run_from_a_test", "-v", "-p", "no:randomly",
             "-o", "addopts="],
            cwd=str(root), env=env, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        # -v, so the selection itself is pinned: under a bare ``-k no_swarm_env`` the gate
        # test -- the only child test that runs the gate and could touch ``live_bus`` --
        # was never selected, and this is the assertion that a revert of the -k fails.
        assert "test_gate_run_from_a_test_posts_no_dm_even_with_an_explicit_bus" in proc.stdout
        assert not (live_bus / "messages.jsonl").exists()


def _stub_declared(
    repo: Path, git_for: dict[str, Any] | None = None, **responses: Any
) -> DeclaredGit:
    runner = DeclaredGit(repo, **(git_for or {}))
    runner.responses = {
        "ruff": (0, "All checks passed!"),
        "mypy": (0, "Success"),
        "pytest": (0, "1 passed in 0.01s"),
        **responses,
    }
    return runner


class TestBaseSelection:
    """1. No --base on branch `main` gates origin/main; an explicit --base is untouched."""

    def _repo_on_main_with_origin(self, tmp_path: Path) -> Path:
        repo = _init_repo(tmp_path)
        head = _git(repo, "rev-parse", "HEAD").strip()
        _git(repo, "update-ref", "refs/remotes/origin/main", head)
        return repo

    def test_picks_origin_main_when_on_main(self, tmp_path: Path) -> None:
        repo = self._repo_on_main_with_origin(tmp_path)
        base, info = swarm_check.resolve_base(repo, None, FakeRunner())
        assert base == "origin/main"
        assert info == "INFO base: on main, using origin/main (pass --base to override)"

    def test_keeps_main_without_an_origin_ref(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)  # no refs/remotes/origin/main
        assert swarm_check.resolve_base(repo, None, FakeRunner()) == ("main", "")

    def test_explicit_base_is_never_rewritten(self, tmp_path: Path) -> None:
        repo = self._repo_on_main_with_origin(tmp_path)
        assert swarm_check.resolve_base(repo, "main", FakeRunner()) == ("main", "")
        assert swarm_check.resolve_base(repo, "origin/main", FakeRunner()) == ("origin/main", "")

    def test_off_main_branch_keeps_main(self, tmp_path: Path) -> None:
        repo = self._repo_on_main_with_origin(tmp_path)
        _git(repo, "checkout", "-qb", "feature")
        assert swarm_check.resolve_base(repo, None, FakeRunner()) == ("main", "")

    def test_main_prints_info_and_the_unpushed_empty_diff_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo_on_main_with_origin(tmp_path)  # HEAD == origin/main, clean tree
        rc = swarm_check.main(
            [str(repo), "--exactness", "never", "--frontend", "never"],
            run=_stub_declared(repo, git_for={"branch": "main", "origin": True}),
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "INFO base: on main, using origin/main (pass --base to override)" in out
        assert (
            "FAIL scope: empty diff vs origin/main: nothing unpushed to gate; "
            "pass --base <sha> to gate pushed commits"
        ) in out

    def test_explicit_base_main_keeps_todays_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo_on_main_with_origin(tmp_path)
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never"],
            run=_stub_declared(repo, git_for={"branch": "main", "origin": True}),
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "INFO base:" not in out
        assert "FAIL scope: empty diff vs main (nothing to review)" in out


class TestBoardPaths:
    """2. The bus's board/telemetry mirror files are never the worker's diff."""

    def test_board_file_is_not_a_scope_offender(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        board = repo / "docs" / "orchestration" / "board"
        board.mkdir(parents=True)
        (board / "messages.jsonl").write_text("{}\n")
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        cfg = _cfg(tmp_path, repo, base="HEAD", scope=("vibe_quant/*",),
                   exactness="never", frontend="never")
        scope = {c.name: c for c in swarm_check.run_checks(cfg, _stub_declared(repo))}["scope"]
        assert scope.status == "PASS" and "messages.jsonl" not in scope.detail

    def test_board_only_change_never_makes_a_non_empty_diff(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        telemetry = repo / "docs" / "orchestration" / "telemetry"
        telemetry.mkdir(parents=True)
        (telemetry / "tasks.jsonl").write_text("{}\n")
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        scope = {c.name: c for c in swarm_check.run_checks(cfg, _stub_declared(repo))}["scope"]
        assert scope.status == "FAIL" and "empty diff" in scope.detail


class TestGateLog:
    """3+4. Every run writes the tool transcript, prints LOG before GATE, counts re-runs."""

    def _repo(self, tmp_path: Path) -> Path:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        return repo

    def _main(self, repo: Path, runner: FakeRunner, *extra: str) -> int:
        rc: int = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never", *extra],
            run=runner,
        )
        return rc

    def test_log_line_precedes_gate_and_file_holds_the_transcript(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        runner = _stub_declared(repo, pytest=(0, "3 passed in 0.05s"))
        assert self._main(repo, runner) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[-1] == "GATE PASS"
        assert lines[-2].startswith("LOG ")
        log = Path(lines[-2][len("LOG "):])
        assert log.parent == repo / "data" / "swarm" / "_gate-logs"
        head12 = _HEAD[:12]
        assert log.name.startswith(f"{repo.name}-{head12}-")
        body = log.read_text()
        assert "All checks passed!" in body and "3 passed in 0.05s" in body
        assert "$ " in body and "pytest" in body and "[exit 0]" in body

    def test_tool_fail_line_points_at_the_log(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        runner = _stub_declared(repo, mypy=(1, "boom: 2 errors"))
        assert self._main(repo, runner) == 1
        out = capsys.readouterr().out
        fail = next(ln for ln in out.splitlines() if ln.startswith("FAIL mypy:"))
        assert fail.endswith(" (full output: LOG above)")
        assert "boom: 2 errors" in fail  # the useful tail line is preserved

    def test_non_tool_fail_line_has_no_log_pointer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        (repo / "docs").mkdir()
        (repo / "docs" / "x.md").write_text("x\n")
        assert self._main(repo, _stub_declared(repo), "--scope", "vibe_quant/*") == 1
        out = capsys.readouterr().out
        fail = next(ln for ln in out.splitlines() if ln.startswith("FAIL scope:"))
        assert "full output: LOG above" not in fail

    def test_scope_fail_log_lists_every_out_of_scope_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scope FAIL skips every tool, so the transcript is empty: the log must still carry
        the out-of-scope paths, or the maker hands back "fix scope first" with nothing to read."""
        repo = self._repo(tmp_path)
        (repo / "docs").mkdir()
        names = ("a", "b", "c", "d", "e", "f")
        for name in names:
            (repo / "docs" / f"{name}.md").write_text("x\n")
        assert self._main(repo, _stub_declared(repo), "--scope", "vibe_quant/*") == 1
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[-2].startswith("LOG ")
        body = Path(lines[-2][len("LOG "):]).read_text()
        assert body.startswith("FAIL scope: 6/7 paths out of scope: ")
        for name in names:  # the 6th offender is dropped by a first-five cut
            assert f"docs/{name}.md" in body

    def test_cached_run_still_writes_a_log(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _init_pass_repo(tmp_path)  # a valid PASS stamp: the second call reuses it
        assert self._main(repo, _pass_runner(repo)) == 0
        capsys.readouterr()
        assert self._main(repo, _pass_runner(repo)) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[-1] == "GATE PASS"
        assert lines[-2].startswith("LOG ")
        assert "cached run" in Path(lines[-2][len("LOG "):]).read_text()

    def test_log_ends_with_what_was_gated_and_the_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """handoff_lint matches a maker's cited commit against these two lines."""
        repo = self._repo(tmp_path)
        tree = "e" * 40
        clean = {"committed": ["vibe_quant/foo.py"], "tree": tree}
        pass_run = _stub_declared(repo, clean, pytest=(0, "3 passed in 0.05s"))
        assert self._main(repo, pass_run) == 0
        log = Path(capsys.readouterr().out.strip().splitlines()[-2][len("LOG "):])
        assert log.read_text().splitlines()[-2:] == [f"HEAD {_HEAD} TREE {tree}", "GATE PASS"]
        fail_run = _stub_declared(repo, clean, mypy=(1, "boom: 2 errors"))
        assert self._main(repo, fail_run, "--no-cache") == 1
        log = Path(capsys.readouterr().out.strip().splitlines()[-2][len("LOG "):])
        assert log.read_text().splitlines()[-1] == "GATE FAIL (1 failed)"

    def test_log_marks_a_gate_over_uncommitted_edits_dirty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The tools read the working tree: with tracked edits no commit holds what was gated."""
        repo = self._repo(tmp_path)
        runner = _stub_declared(
            repo, {"committed": ["vibe_quant/foo.py"], "dirty": ["vibe_quant/foo.py"]}
        )
        self._main(repo, runner)
        log = Path(capsys.readouterr().out.strip().splitlines()[-2][len("LOG "):])
        assert log.read_text().splitlines()[-2] == f"HEAD {_HEAD} TREE dirty"

    def test_cached_log_ends_with_the_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _init_pass_repo(tmp_path)
        assert self._main(repo, _pass_runner(repo)) == 0
        capsys.readouterr()
        assert self._main(repo, _pass_runner(repo)) == 0
        log = Path(capsys.readouterr().out.strip().splitlines()[-2][len("LOG "):])
        assert log.read_text().splitlines()[-2:] == [f"HEAD {_HEAD} TREE {'b' * 40}", "GATE PASS"]

    def test_log_marks_an_edit_made_while_the_tools_ran_moved(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Clean when the gate started, edited during the test run: the pre-run tree was not
        what the tools read, so the LOG must not vouch for it."""
        repo = self._repo(tmp_path)
        runner = _stub_declared(repo, {"committed": ["vibe_quant/foo.py"], "tree": "e" * 40})

        def edit_during_tests(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            assert runner.porcelain is not None
            runner.porcelain.append(" M vibe_quant/foo.py")
            return 0, "3 passed in 0.05s"

        runner.responses["pytest"] = edit_during_tests
        self._main(repo, runner)
        log = Path(capsys.readouterr().out.strip().splitlines()[-2][len("LOG "):])
        assert log.read_text().splitlines()[-2] == f"HEAD {_HEAD} TREE moved"

    def test_run_counter_reports_the_second_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _init_pass_repo(tmp_path)
        head12 = _HEAD[:12]
        log_dir = repo / "data" / "swarm" / "_gate-logs"
        log_dir.mkdir(parents=True)
        (log_dir / f"{repo.name}-{head12}-20240101T000000Z.log").write_text("$ x\n")
        assert self._main(repo, _stub_declared(repo)) == 0
        assert "INFO gate run #2 on this HEAD" in capsys.readouterr().out


def _gate_runs_ledger(bus: Path) -> Path:
    return bus.parent / "gate-runs.jsonl"


def _seed_gate_runs(bus: Path, agent: str, count: int) -> Path:
    """`count` uncached runs already on the book for `agent`: the cap counts these."""
    ledger = _gate_runs_ledger(bus)
    ledger.write_text(
        "".join(
            json.dumps(
                {"ts": f"t{i}", "agent": agent, "worktree": "wt", "log": f"/logs/r{i}.log"}
            )
            + "\n"
            for i in range(count)
        )
    )
    return ledger


class TestGateRunsCap:
    """The gate runs once per round, enforced: a maker's 4th uncached run is refused.

    A cached PASS costs nothing, is never refused and never counted; verifiers, reviewers,
    `--full` and a run with no bus stay outside the cap. The check sits after the cached-stamp
    branch and before `run_checks`, so a refused run starts no tool and writes no LOG.
    """

    def _repo(self, tmp_path: Path) -> Path:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        return repo

    @staticmethod
    def _argv(repo: Path, bus: Path, agent: str, *extra: str) -> list[str]:
        return [
            str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
            "--bus", str(bus), "--bus-agent", agent, "--no-cache", *extra,
        ]  # fmt: skip

    def test_fourth_uncached_maker_run_fails_without_running_a_tool(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True)
        runner = _stub_declared(repo)
        argv = self._argv(repo, bus, "cc-x-1")
        for _ in range(3):
            assert swarm_check.main(argv, run=runner) == 0
        ledger = _gate_runs_ledger(bus)
        log_dir = repo / "data" / "swarm" / "_gate-logs"
        logs_before = len(list(log_dir.glob("*.log")))
        capsys.readouterr()
        calls_before = len(runner.calls)

        assert swarm_check.main(argv, run=runner) == 1
        out = capsys.readouterr().out
        assert "FAIL gate-runs: cc-x-1 ran the gate 3 times (cap 3)" in out
        assert "ask the chief (bus ask --to chief)" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")
        assert all(c[0] == "git" for c in runner.calls[calls_before:])  # no tool for it
        assert len(list(log_dir.glob("*.log"))) == logs_before  # a refused run writes no LOG
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        assert len(rows) == 3  # and appends nothing
        assert [r["agent"] for r in rows] == ["cc-x-1"] * 3
        assert [r["worktree"] for r in rows] == [repo.name] * 3
        assert rows[-1]["log"] in out  # the FAIL names the last LOG to read

    def test_cached_pass_after_three_runs_is_not_refused_and_appends_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _init_pass_repo(tmp_path)  # a valid PASS stamp: this run costs no tool
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True)
        ledger = _seed_gate_runs(bus, "cc-x-1", 3)
        before = ledger.read_text()
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(bus), "--bus-agent", "cc-x-1"],
            run=_pass_runner(repo),
        )  # fmt: skip
        out = capsys.readouterr().out
        assert rc == 0
        assert "PASS cache: reused stamp from" in out
        assert "FAIL gate-runs" not in out
        assert ledger.read_text() == before

    def test_full_integration_run_is_never_capped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True)
        ledger = _seed_gate_runs(bus, "cc-x-1", 3)
        rc = swarm_check.main(
            self._argv(repo, bus, "cc-x-1", "--full"), run=_stub_declared(repo)
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "FAIL gate-runs" not in out
        assert len(ledger.read_text().splitlines()) == 3

    def test_verifier_is_never_capped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True)
        ledger = _seed_gate_runs(bus, "ver-x-1", 3)
        assert swarm_check.main(
            self._argv(repo, bus, "ver-x-1"), run=_stub_declared(repo)
        ) == 0
        assert "FAIL gate-runs" not in capsys.readouterr().out
        assert len(ledger.read_text().splitlines()) == 3

    def test_no_bus_is_never_capped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = self._repo(tmp_path)
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus-agent", "cc-x-1"],
            run=_stub_declared(repo),
        )  # fmt: skip
        out = capsys.readouterr().out
        assert "FAIL gate-runs" not in out
        assert list(tmp_path.rglob("gate-runs.jsonl")) == []
        # the bus-inbox check alone refuses it: an agent id with no bus to peek
        assert rc == 1 and "FAIL bus-inbox:" in out

    @pytest.mark.parametrize("raw", ["0", "-2"])
    def test_a_zero_or_negative_cap_refuses_the_first_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        raw: str,
    ) -> None:
        monkeypatch.setenv("SWARM_GATE_CAP", raw)
        repo = self._repo(tmp_path)
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True)
        runner = _stub_declared(repo)
        assert swarm_check.main(self._argv(repo, bus, "cc-x-1"), run=runner) == 1
        out = capsys.readouterr().out
        assert "FAIL gate-runs: cc-x-1 ran the gate 0 times (cap 0)" in out and "no earlier run" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")
        assert all(c[0] == "git" for c in runner.calls)
        assert not _gate_runs_ledger(bus).exists()
        assert not list((repo / "data" / "swarm" / "_gate-logs").glob("*.log"))

    def test_swarm_gate_cap_allows_a_fourth_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("SWARM_GATE_CAP", "5")
        repo = self._repo(tmp_path)
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True)
        ledger = _seed_gate_runs(bus, "cc-x-1", 3)
        assert swarm_check.main(self._argv(repo, bus, "cc-x-1"), run=_stub_declared(repo)) == 0
        assert "FAIL gate-runs" not in capsys.readouterr().out
        assert len(ledger.read_text().splitlines()) == 4


_PIN = "the quick brown fox jumps over the lazy dog"


_PIN_SRC = f'MESSAGE = "{_PIN}"\n\n\ndef foo() -> int:\n    return 1\n'
_PIN_EDIT = 'MESSAGE = "a totally different message"\n\n\ndef foo() -> int:\n    return 1\n'


def _pinned_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """foo.py's base content carries _PIN; the working tree edit no longer does.

    Declared, not committed: `git show HEAD:vibe_quant/foo.py` is the only base content the
    check reads, so its blob comes from memory and the repo needs no history.
    """
    repo = _init_repo(tmp_path)
    (repo / "vibe_quant" / "foo.py").write_text(_PIN_EDIT)
    (repo / "tests" / "unit" / "test_pin.py").write_text(
        f'def test_pin() -> None:\n    assert "{_PIN}"\n'
    )
    return repo, {"vibe_quant/foo.py": _PIN_SRC}


class TestPinnedText:
    """5. A test the gate does NOT run that pins a string the diff removed is a FAIL."""

    def _pin(
        self, tmp_path: Path, repo: Path, blobs: dict[str, str], **state: Any
    ) -> Any:
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        runner = DeclaredGit(repo, blobs=blobs, dirty=["vibe_quant/foo.py"], **state)
        return {c.name: c for c in swarm_check.run_checks(cfg, runner)}["pinned-text"]

    def test_unrun_test_pinning_a_removed_literal_fails(self, tmp_path: Path) -> None:
        repo, blobs = _pinned_repo(tmp_path)
        pin = self._pin(tmp_path, repo, blobs)
        assert pin.status == "FAIL"
        assert "tests/unit/test_pin.py" in pin.detail
        assert f'pins removed text "{_PIN}"' in pin.detail

    def test_the_removed_literal_is_read_from_the_base_ref_not_from_head(
        self, tmp_path: Path
    ) -> None:
        """Real git: `git show <base>:<path>` must read the BASE, which here is two commits
        behind HEAD (the first branch commit already dropped the literal)."""
        repo = _init_repo(tmp_path)
        foo = repo / "vibe_quant" / "foo.py"
        foo.write_text(_PIN_SRC)
        (repo / "tests" / "unit" / "test_pin.py").write_text(
            f'def test_pin() -> None:\n    assert "{_PIN}"\n'
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "pin the message")
        _git(repo, "checkout", "-qb", "feature")
        foo.write_text(_PIN_EDIT)
        _git(repo, "commit", "-qam", "drop the message")
        (repo / "vibe_quant" / "keep.py").write_text("X = 2\n")
        _git(repo, "commit", "-qam", "unrelated follow-up")
        cfg = _cfg(tmp_path, repo, base="main", exactness="never", frontend="never")
        pin = {c.name: c for c in swarm_check.run_checks(cfg, FakeRunner())}["pinned-text"]
        assert pin.status == "FAIL"
        assert f'pins removed text "{_PIN}"' in pin.detail

    def test_literal_still_present_elsewhere_passes(self, tmp_path: Path) -> None:
        repo, blobs = _pinned_repo(tmp_path)
        (repo / "vibe_quant" / "keep.py").write_text(f'X = "{_PIN}"\n')  # still in production
        pin = self._pin(tmp_path, repo, blobs, untracked=["vibe_quant/keep.py"])
        assert pin.status == "PASS" and "0 removed literals" in pin.detail

    def test_unchanged_implicitly_concatenated_fstring_is_not_removed(
        self, tmp_path: Path
    ) -> None:
        """ast merges `"a, " f"candidates={x}"` into the constant "a, candidates=", which no
        source text holds: a comment-only diff must not read it as removed (a
        discovery/__main__.py bench diff FAILed on ", candidates=")."""
        repo = _init_repo(tmp_path)
        src = 'def f(x: int) -> str:\n    return (\n        "done: "\n        f"candidates={x}"\n    )\n'
        (repo / "vibe_quant" / "foo.py").write_text(src + "# comment only\n")
        (repo / "tests" / "unit" / "test_unrun.py").write_text('X = "done: candidates="\n')
        pin = self._pin(
            tmp_path,
            repo,
            {"vibe_quant/foo.py": src},
            untracked=["tests/unit/test_unrun.py"],
        )
        assert pin.status == "PASS" and "0 removed literals" in pin.detail

    def test_pinning_test_already_run_passes(self, tmp_path: Path) -> None:
        repo, blobs = _pinned_repo(tmp_path)
        (repo / "tests" / "unit" / "test_pin.py").write_text(
            f'def test_pin() -> None:\n    # touched\n    assert "{_PIN}"\n'
        )  # the changed test file runs whole, so its pin is covered
        pin = self._pin(tmp_path, repo, blobs, untracked=["tests/unit/test_pin.py"])
        assert pin.status == "PASS" and "1 removed literals" in pin.detail

    def test_pin_is_not_reported_when_the_full_unit_suite_runs(self, tmp_path: Path) -> None:
        """The other half of ``test_already_run``: with TestRun.full_unit the whole tests/unit
        ran, so a tests/unit pin IS covered by the gate's own selection and must not be a
        false-positive FAIL. The pinning test is NOT in ``explicit`` here - a conftest change
        is not a changed test file - so only the ``full_unit and path.startswith('tests/unit/')``
        branch can spare it (a mutant returning False there reports the pin)."""
        repo, blobs = _pinned_repo(tmp_path)
        (repo / "tests" / "conftest.py").write_text("X = 1\n")  # forces the full tests/unit run
        pin = self._pin(tmp_path, repo, blobs, untracked=["tests/conftest.py"])
        assert pin.status == "PASS"
        assert "test_pin.py" not in pin.detail

    def test_test_already_run_explicit_full_unit_and_outside(self) -> None:
        """Direct pin of the branch: explicit membership, the full_unit tests/unit case, and
        the two negatives (a tests/unit path without full_unit, and a path outside tests/unit)."""
        check = swarm_check.Check("tests", "PASS", "x")
        explicit = swarm_check.TestRun(check, frozenset({"tests/unit/test_a.py"}))
        assert swarm_check.test_already_run("tests/unit/test_a.py", explicit)
        assert not swarm_check.test_already_run("tests/unit/test_b.py", explicit)
        full = swarm_check.TestRun(check, frozenset(), True)
        assert swarm_check.test_already_run("tests/unit/test_b.py", full)
        assert not swarm_check.test_already_run("tests/api/test_b.py", full)
        assert not swarm_check.test_already_run("tests/unit/test_b.py", swarm_check.TestRun(check))

    def test_skips_without_a_vibe_quant_change(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "notes.txt").write_text("x\n")
        runner = DeclaredGit(repo, untracked=["notes.txt"])
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        pin = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["pinned-text"]
        assert pin.status == "SKIP" and "no vibe_quant" in pin.detail


def test_gate_caps_xdist_auto_when_the_host_is_already_loaded() -> None:
    workers = swarm_check.xdist_auto_workers
    assert workers({}, 1.0, 10) is None
    assert workers({}, 5.0, 10) is None
    assert workers({}, 6.4, 10) == "5"  # floor: half the cores
    assert workers({}, 29.5, 10) == "5"
    assert workers({}, 3.0, 10) is None
    assert workers({"SWARM_CHECK_WORKERS": "3"}, 1.0, 10) == "3"


def test_failed_tests_names_the_first_failing_node_ids() -> None:
    out = (
        "....F.E\n=== short test summary info ===\n"
        "FAILED tests/unit/test_a.py::test_x - AssertionError: boom\n"
        "ERROR tests/unit/test_b.py::test_y - fixture missing\n"
        "FAILED tests/unit/test_c.py::test_z[1] - assert 1 == 2\n"
        "FAILED tests/unit/test_d.py::test_w\n2 failed, 1 error, 4 passed in 1.0s\n"
    )
    assert swarm_check.failed_tests(out) == [
        "FAILED tests/unit/test_a.py::test_x, tests/unit/test_b.py::test_y, "
        "tests/unit/test_c.py::test_z[1] (+1 more)"
    ]
    assert swarm_check.failed_tests("3 passed in 0.1s") == []


def test_gate_pytest_runs_quiet_and_the_fail_line_names_the_test(tmp_path: Path) -> None:
    repo = _path_reader_repo(tmp_path)
    (repo / "vibe_quant" / "screening" / "replay_drift.py").write_text("R = 2\n")
    runner = FakeRunner()
    runner.responses = {
        "pytest": (1, "FAILED tests/unit/test_r.py::test_drift - assert 1 == 2\n1 failed in 0.1s")
    }
    cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
    tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
    assert tests.status == "FAIL"
    assert "FAILED tests/unit/test_r.py::test_drift" in tests.detail
    pytest_envs = [
        e for c, e in zip(runner.calls, runner.envs, strict=True)
        if any("pytest" in Path(a).name for a in c)
    ]  # fmt: skip
    assert pytest_envs
    assert all(e is not None and "-q -rfE" in e["PYTEST_ADDOPTS"] for e in pytest_envs)


def test_parse_scope_strips_the_new_file_marker() -> None:
    """Briefs mark new files `+path`; the gate must accept the same scope string verbatim."""
    assert swarm_check.parse_scope("+a.py, b/*,,+frontend/c.ts ") == ("a.py", "b/*", "frontend/c.ts")
    assert swarm_check.parse_scope("") == ()


_STALE_THREAD = (
    "thread:delegated-task:command%3Amcp%3Aaaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    "%3Adelegate-task%3A"
)


_STALE_DB_SEQ = 0


def _stale_db(tmp_path: Path, agent: str, *commands: str) -> Path:
    """A tiny T3 state DB holding one command_execution row per command, for `agent`."""
    import sqlite3

    global _STALE_DB_SEQ
    _STALE_DB_SEQ += 1
    path = tmp_path / f"t3-{agent}-{_STALE_DB_SEQ}.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "create table orchestration_v2_projection_turn_items "
        "(thread_id text, type text, ordinal integer, payload_json text)"
    )
    for ordinal, command in enumerate(commands):
        con.execute(
            "insert into orchestration_v2_projection_turn_items values (?, ?, ?, ?)",
            (_STALE_THREAD + agent, "command_execution", ordinal, json.dumps({"input": command})),
        )
    con.commit()
    con.close()
    return path


class TestStaleReads:
    """A maker round whose commands read or ran MAIN's code instead of its worktree FAILs.

    T3 children start in the main checkout: 59 of 342 in Oct 14-28 read or tested main's
    stale code. The check names the command (user decision: no PATH wrapper), applies to the
    same cc-/impl- makers as the gate-runs cap, and is time-dependent like bus-inbox: it is
    re-evaluated on a cached PASS and never stored in the PASS stamp.
    """

    def _argv(self, tmp_path: Path, repo: Path, agent: str, *extra: str) -> list[str]:
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True, exist_ok=True)
        return [
            str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
            "--bus", str(bus), "--bus-agent", agent, "--no-cache", *extra,
        ]  # fmt: skip

    def _repo(self, tmp_path: Path) -> Path:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
        return repo

    def test_maker_stale_read_fails_the_gate_and_names_the_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = self._repo(tmp_path)
        stale = "cd main && set -o pipefail; sed -n 1,80p scripts/agents/handoff_lint.py"
        db = _stale_db(
            tmp_path, "cc-x-1", stale, f"cd {repo} && sed -n 1p scripts/agents/bus.py"
        )
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        assert swarm_check.main(self._argv(tmp_path, repo, "cc-x-1"), run=_stub_declared(repo)) == 1
        out = capsys.readouterr().out
        assert f"FAIL stale-reads: 1 stale command(s), e.g. {stale}; re-read from {repo}" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")

    def test_the_quoted_command_is_cut_to_120_chars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = self._repo(tmp_path)
        stale = "cat vibe_quant/" + "x" * 200 + ".py"
        db = _stale_db(tmp_path, "cc-x-1", stale)
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        assert swarm_check.main(self._argv(tmp_path, repo, "cc-x-1"), run=_stub_declared(repo)) == 1
        out = capsys.readouterr().out
        assert f", e.g. {stale[:120]}; re-read from {repo}" in out

    def test_a_command_run_from_the_worktree_passes_with_the_count(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = self._repo(tmp_path)
        db = _stale_db(
            tmp_path,
            "cc-x-1",
            f"cd {repo} && sed -n 1p scripts/agents/bus.py",
            f"cd {repo} && PYTHONPATH=$PWD .venv/bin/python -m pytest tests/unit/test_foo.py",
        )
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        assert swarm_check.main(self._argv(tmp_path, repo, "cc-x-1"), run=_stub_declared(repo)) == 0
        out = capsys.readouterr().out
        assert "PASS stale-reads: 0 of 2 commands" in out
        assert out.strip().endswith("GATE PASS")

    def test_a_reviewer_agent_skips_the_per_agent_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = self._repo(tmp_path)
        db = _stale_db(tmp_path, "rev-x-1", "cat vibe_quant/metrics.py")
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        assert swarm_check.main(self._argv(tmp_path, repo, "rev-x-1"), run=_stub_declared(repo)) == 0
        out = capsys.readouterr().out
        assert "SKIP stale-reads:" in out and "maker" in out

    def test_full_integration_run_skips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = self._repo(tmp_path)
        db = _stale_db(tmp_path, "cc-x-1", "cat vibe_quant/metrics.py")
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        rc = swarm_check.main(
            self._argv(tmp_path, repo, "cc-x-1", "--full"), run=_stub_declared(repo)
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "SKIP stale-reads:" in out

    def test_a_missing_db_skips_with_the_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = self._repo(tmp_path)
        missing = tmp_path / "no-such" / "statev2.sqlite"
        monkeypatch.setenv("SWARM_T3_DB", str(missing))
        assert swarm_check.main(self._argv(tmp_path, repo, "cc-x-1"), run=_stub_declared(repo)) == 0
        out = capsys.readouterr().out
        assert f"SKIP stale-reads: no T3 history db at {missing}" in out

    def test_a_stale_only_fail_does_not_block_the_stamp_and_a_cached_run_reuses_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Time-dependent like bus-inbox: the tool PASS is stamped, so a fixed log reuses it."""
        repo = _init_pass_repo(tmp_path)
        shutil.rmtree(repo / ".git" / "swarm-check", ignore_errors=True)
        runner = _stub_declared(repo, git_for={"committed": ["vibe_quant/foo.py"]})
        db = _stale_db(tmp_path, "cc-x-1", "cat vibe_quant/metrics.py")
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True, exist_ok=True)
        argv = [
            str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
            "--bus", str(bus), "--bus-agent", "cc-x-1",
        ]  # fmt: skip

        monkeypatch.setenv("SWARM_T3_DB", str(db))
        assert swarm_check.main(argv, run=runner) == 1
        first = capsys.readouterr().out
        assert "FAIL stale-reads:" in first and "PASS cache:" not in first
        stamps = list((repo / ".git" / "swarm-check").glob("pass-*.json"))
        assert stamps, "a stale-reads-only FAIL must still write the tool PASS stamp"

        clean = _stale_db(tmp_path, "cc-x-1")
        monkeypatch.setenv("SWARM_T3_DB", str(clean))
        assert swarm_check.main(argv, run=runner) == 0
        second = capsys.readouterr().out
        assert "PASS cache: reused stamp from" in second
        assert "PASS stale-reads: 0 of 0 commands" in second

    def test_a_cached_pass_with_a_stale_read_fails_again(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = _init_pass_repo(tmp_path)
        db = _stale_db(tmp_path, "cc-x-1", "cat vibe_quant/metrics.py")
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        bus = tmp_path / "job" / "bus"
        bus.mkdir(parents=True, exist_ok=True)
        rc = swarm_check.main(
            [str(repo), "--base", "main", "--exactness", "never", "--frontend", "never",
             "--bus", str(bus), "--bus-agent", "cc-x-1"],
            run=_pass_runner(repo),
        )  # fmt: skip
        out = capsys.readouterr().out
        assert rc == 1
        assert "PASS cache: reused stamp from" in out
        assert "FAIL stale-reads: 1 stale command(s), e.g. cat vibe_quant/metrics.py" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")
