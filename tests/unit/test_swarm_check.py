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
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.conftest import free_port as _free_port

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from tests.unit.conftest import BoardServer

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
                worktree=repo,
                main_root=root,
                base="main",
                scope=scope,
                exactness="never",
                frontend="never",
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
            proc = subprocess.run(c, cwd=str(cwd), capture_output=True, text=True, env=_git_env())
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
            self.porcelain = [f" M {p}" for p in dirty or ()] + [f"?? {p}" for p in untracked or ()]
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


class TestGitState:
    def test_git_diff_failure_is_error(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)

        def run(
            cmd: Sequence[str], cwd: Path, env: object = None, timeout: int = 1
        ) -> tuple[int, str]:
            if "diff" in cmd:
                return 129, "usage: git diff"
            return FakeRunner()(cmd, cwd)

        state = swarm_check.git_state(repo, "main", run)
        assert "git diff failed" in state.error and state.changed == []


class TestImportGraph:
    """The reverse import closure: which tests/unit files reach a changed vibe_quant module.

    Selection by test NAME (`-k <stem>`) cannot see a caller test whose names never mention
    the module: tests/unit/test_symbol_agg.py::test_topk_ranks_gate_passers_first ran red
    while the gate PASSed (job 20261015-funding-carry). These are pure functions.
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
            (
                "tests/unit/test_caller.py",
                "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n",
            ),
            (
                "tests/unit/api/test_nested.py",
                "from vibe_quant.discovery.pipeline import DiscoveryPipeline\n",
            ),
            ("tests/unit/test_plainer.py", "import json\n"),
        ):
            (tmp_path / rel).write_text(source)
        return swarm_check.build_import_graph(tmp_path)

    @pytest.mark.parametrize(
        ("bad", "why"), [("broken", "cannot parse"), ("unreadable", "cannot read")]
    )
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
        assert (
            plan.notes[0].startswith("full tests/unit: selection unknown") and why in plan.notes[0]
        )

    def test_direct_and_transitive_importers(self) -> None:
        # test_other imports vibe_quant.c directly; test_direct reaches it through
        # vibe_quant.a.b (a two-module chain: the caller test in the bead's report).
        assert swarm_check.importer_test_files(["vibe_quant/c.py"], self._GRAPH) == [
            "tests/unit/test_direct.py",
            "tests/unit/test_other.py",
        ]


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
    the reverse import closure selects it when those modules change.
    """

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


class TestAggregation:
    def test_gate_fail_counts(self) -> None:
        checks = [
            swarm_check.Check("ruff", "FAIL", "x"),
            swarm_check.Check("tests", "FAIL", "y"),
            swarm_check.Check("mypy", "PASS", "ok"),
        ]
        gate, n = swarm_check.gate_summary(checks)
        assert (gate, n) == ("GATE FAIL (2 failed)", 2)


class TestRunChecks:
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

    def test_approved_scope_add_admits_only_its_globs(self, tmp_path: Path) -> None:
        """A chief-approved SCOPE += reaches the gate without editing the brief's command."""
        repo = _init_repo(tmp_path)
        runner = DeclaredGit(repo)
        (repo / "docs").mkdir()
        (repo / "docs" / "x.md").write_text("x\n")
        (repo / "docs" / "y.md").write_text("y\n")
        job = tmp_path / "job"
        (job / "bus").mkdir(parents=True)
        (job / "scope").mkdir()
        (job / "scope" / "cc-t-1.txt").write_text("docs/x.md\n")
        bus = str(job / "bus")

        def scope_check(agent: str) -> swarm_check.Check:
            globs, _note = swarm_check.effective_scope("vibe_quant/*", bus, agent)
            cfg = _cfg(tmp_path, repo, scope=globs, exactness="never", frontend="never")
            return {c.name: c for c in swarm_check.run_checks(cfg, runner)}["scope"]

        mine = scope_check("cc-t-1")
        assert mine.status == "FAIL"
        assert "docs/y.md" in mine.detail
        assert "docs/x.md" not in mine.detail
        other = scope_check("cc-other-1")  # approvals are per agent
        assert "docs/x.md" in other.detail

    def test_gate_main_applies_the_approved_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() hands the approved globs to the checks; an unreadable file adds nothing."""
        repo = _init_repo(tmp_path)
        job = tmp_path / "job"
        (job / "bus").mkdir(parents=True)
        (job / "scope").mkdir()
        approvals = job / "scope" / "cc-t-1.txt"
        approvals.write_text("docs/x.md\n")
        seen: list[tuple[str, ...]] = []

        def capture(cfg: Any, run: object = None, facts: object = None) -> list[Any]:
            seen.append(tuple(cfg.scope))
            return [swarm_check.Check("scope", "PASS", "ok")]

        monkeypatch.setattr(swarm_check, "run_checks", capture)
        monkeypatch.setattr(
            swarm_check, "bus_inbox_check", lambda cfg: swarm_check.Check("bus-inbox", "PASS", "")
        )
        argv = [str(repo), "--scope", "vibe_quant/*", "--bus", str(job / "bus")]
        swarm_check.main([*argv, "--bus-agent", "cc-t-1", "--no-cache"])
        approvals.unlink()
        approvals.mkdir()  # unreadable as a file
        swarm_check.main([*argv, "--bus-agent", "cc-t-1", "--no-cache"])
        assert seen == [("vibe_quant/*", "docs/x.md"), ("vibe_quant/*",)]

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

    def test_missing_declared_pins_file_fails_closed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_exactness(tmp_path, repo)
        runner = DeclaredGit(repo)
        runner.responses = {"exactness": (0, "IDENTICAL")}
        cfg = _cfg(
            tmp_path,
            repo,
            exactness="always",
            frontend="never",
            base="HEAD",
            expect_pins=str(tmp_path / "nope.json"),
        )
        ex = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["exactness"]
        assert ex.status == "FAIL" and "declared pins file missing" in ex.detail
        assert not _called_with(runner, "exactness_239.py")

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


class TestImporterTestSelection:
    """run_checks: the reverse import closure drives real pytest calls."""

    @staticmethod
    def _pytest_cmds(runner: FakeRunner) -> list[list[str]]:
        return [
            c[c.index("pytest") + 1 :]
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

    def test_deleted_module_still_selects_its_lazy_importers_chain(self, tmp_path: Path) -> None:
        # Deleting selection.py (on disk) must keep the `from vibe_quant.discovery import
        # selection` submodule edge, or the pipeline's caller test is never selected.
        repo = _init_repo(tmp_path)
        _git(repo, "checkout", "-qb", "feature")
        (repo / "vibe_quant" / "discovery").mkdir()
        (repo / "vibe_quant" / "discovery" / "__init__.py").write_text("")
        (repo / "vibe_quant" / "discovery" / "selection.py").write_text(
            "def f() -> int:\n    return 1\n"
        )
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


_PAR = ["-n", "auto", "--dist", "worksteal", "-m", "not timing"]  # the per-round xdist tail
_POL = ["-m", "not timing"]  # the serial fail-fast policy call's tail

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
    "tests/unit/test_scripts_import_smoke.py",
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


class TestPolicyTestSelection:
    """Repo-wide policy tests SCAN the tree instead of importing the changed module, so the
    reverse import closure can never select them: on 2026-10-18 a new silent handler in
    vibe_quant/discovery/neff.py went red in tests/unit/test_exception_hygiene.py although
    every gate PASSed. Every non-full selection appends the ones present (POLICY_TESTS).
    """

    @staticmethod
    def _pytest_cmds(runner: FakeRunner) -> list[list[str]]:
        return [
            c[c.index("pytest") + 1 :]
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
        assert self._pytest_cmds(runner) == [[*_VQ_WIDE, *_POL]]
        assert "2 policy tests" in tests.detail

    @pytest.mark.parametrize("full", [False, True])
    def test_a_failing_policy_test_fails_the_gate_before_the_main_run(
        self, tmp_path: Path, full: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Job 20261120-board: a missing exec bit (test_script_exec_bits.py) surfaced after a
        4-minute suite run. A red policy call ends the tests step: the selection never runs."""
        repo = _policy_repo(tmp_path)
        (repo / "tests" / "unit" / "test_new.py").write_text("def test_new() -> None:\n    pass\n")
        runner = DeclaredGit(
            repo, dirty=["vibe_quant/unrelated.py"], untracked=["tests/unit/test_new.py"]
        )
        node = "tests/unit/test_discovery_liquidation.py::test_policy"

        def pytest_run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            if "tests/unit/test_discovery_liquidation.py" in cmd:
                return 1, f"FAILED {node} - assert False\n1 failed, 1 passed in 0.01s\n"
            return 0, "1 passed in 0.01s"

        runner.responses = {"pytest": pytest_run}
        monkeypatch.setattr(
            swarm_check, "_new_unmarked_tests", lambda *a: ["tests/unit/test_new.py::test_new"]
        )
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never", full=full)
        checks = {c.name: c for c in swarm_check.run_checks(cfg, runner)}
        tests = checks["tests"]
        assert tests.status == "FAIL"
        assert node in tests.detail
        assert "policy tests failed: selection skipped" in tests.detail
        # the changed test never ran: no timing verdict, and no second FAIL line for it
        assert checks["slow-tests"].status == "SKIP"
        # nor is the new test re-run under coverage: the gate already failed
        assert checks["simple-tests"].status == "SKIP"
        assert "policy tests failed" in checks["simple-tests"].detail
        cmds = self._pytest_cmds(runner)
        assert len(cmds) == 1 and "tests/unit/test_discovery_liquidation.py" in cmds[0]
        assert not any("tests/unit/test_new.py" in c or "tests/unit" in c for c in cmds)

    def test_a_changed_policy_file_runs_only_in_the_fail_fast_call(self, tmp_path: Path) -> None:
        """Review rev-gate-1 B7: a policy file that is itself a target (or an importer) left
        the first call, so its failure arrived only after the ordinary xdist run."""
        repo = _policy_repo(tmp_path)
        policy = "tests/unit/test_script_exec_bits.py"
        (repo / policy).write_text(
            "from vibe_quant.unrelated import U\n\ndef test_policy() -> None:\n    assert U\n"
        )
        (repo / "tests" / "unit" / "test_user.py").write_text(
            "from vibe_quant.unrelated import U\n"
        )
        runner = DeclaredGit(
            repo, dirty=["vibe_quant/unrelated.py", policy, "tests/unit/test_user.py"]
        )
        node = f"{policy}::test_policy"

        def pytest_run(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            if policy in cmd:
                return 1, f"FAILED {node} - assert False\n1 failed in 0.01s\n"
            return 0, "1 passed in 0.01s"

        runner.responses = {"pytest": pytest_run}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "FAIL" and node in tests.detail
        cmds = self._pytest_cmds(runner)
        assert len(cmds) == 1 and policy in cmds[0] and "-n" not in cmds[0]
        # green: the ordinary call does not run it twice; the policy call keeps its junit
        # report, so slow-tests still times the changed policy test
        runner = DeclaredGit(
            repo, dirty=["vibe_quant/unrelated.py", policy, "tests/unit/test_user.py"]
        )
        runner.responses = {"pytest": (0, "1 passed in 0.01s")}
        swarm_check.run_checks(cfg, runner)
        cmds = self._pytest_cmds(runner)
        assert policy in cmds[0] and "tests/unit/test_user.py" in cmds[1]
        assert all(policy not in c for c in cmds[1:])
        pol_env = next(e for c, e in zip(runner.calls, runner.envs, strict=True) if policy in c)
        assert pol_env is not None and "--junitxml=" in pol_env["PYTEST_ADDOPTS"]


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

    def test_modified_worktree_copy_fails_and_is_never_run(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_carry(tmp_path)
        (repo / "scripts" / "agents").mkdir(parents=True, exist_ok=True)
        (repo / "scripts" / "agents" / "exactness_carry.py").write_text("EXPECTED = 9\n")
        carry, runner = self._check(tmp_path, repo)
        assert carry.status == "FAIL"
        assert "exactness_carry script modified in worktree" in carry.detail
        assert not _called_with(runner, "exactness_carry.py")

    def test_rc_zero_without_identical_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _main_carry(tmp_path, repo)
        carry, _ = self._check(
            tmp_path,
            repo,
            responses={"exactness_carry.py": (0, "no evidence line")},
            exactness="always",
        )
        assert carry.status == "FAIL"
        assert "drift (rc=0)" in carry.detail and "no evidence line" in carry.detail


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


class TestFailClosed:
    def test_bad_base_fails_gate(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 5\n")
        _git(repo, "commit", "-qam", "change")
        # git rejects the ref itself, so this one keeps the real binary
        checks = (
            swarm_check.run_checks(cfg, FakeRunner())
            if (cfg := _cfg(tmp_path, repo, base="mian", exactness="auto", frontend="auto"))
            else []
        )
        by = {c.name: c for c in checks}
        assert by["scope"].status == "FAIL" and "bad base ref: mian" in by["scope"].detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_pytest_rc5_on_changed_test_file_fails(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "tests" / "unit" / "test_bar.py").write_text("X = 1\n")
        runner = DeclaredGit(repo)
        runner.responses = {"pytest": (5, "no tests ran")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        tests = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["tests"]
        assert tests.status == "FAIL"

    def test_missing_binary_still_yields_gate(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)  # main_root=tmp_path has no .venv -> real_run hits ENOENT
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        checks = swarm_check.run_checks(cfg, swarm_check.real_run)
        by = {c.name: c for c in checks}
        assert by["ruff"].status == "FAIL" and "cannot run" in by["ruff"].detail
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_stubs_refused_without_allow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        # A big selection runs under xdist with the relative-speed tests deselected; the
        # serial `timing` call names its files, and this tree has none (no call).
        assert [c[c.index("pytest") + 1 :] for c in pytest_cmds] == [
            ["tests/unit", "-n", "auto", "--dist", "worksteal", "-m", "not timing"],
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


def _shared_dir(main: Path, link: Path) -> Path:
    """`main` as a real dir holding a marker, `link` a symlink to it (worktree.sh's shared link)."""
    main.mkdir(parents=True, exist_ok=True)
    (main / "marker").write_text("main's\n")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(main)
    return main / "marker"


class TestDependencyEnvs:
    """A worktree that changes its deps is gated against ITS deps, never main's (user ruling
    2026-11-24: no SWARM_CHECK_ALLOW_STUBS / --exactness never workarounds)."""

    def test_lock_differs_every_python_tool_runs_from_the_worktree_venv(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        (tmp_path / "uv.lock").write_text("old\n")
        (repo / "uv.lock").write_text("new\n")
        main_marker = _shared_dir(tmp_path / ".venv", repo / ".venv")
        _main_exactness(tmp_path, repo)
        runner = DeclaredGit(repo)
        runner.responses = {"exactness_239.py": (0, "IDENTICAL"), "pytest": (0, "1 passed")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="always", frontend="never")
        by = {c.name: c for c in swarm_check.run_checks(cfg, runner)}
        assert by["python-env"].status == "PASS"
        assert by["python-env"].detail == "worktree .venv (lock differs)"
        sync = [
            (c, e)
            for c, e in zip(runner.calls, runner.envs, strict=True)
            if c[:2] == ["uv", "sync"]
        ]
        assert len(sync) == 1 and "--locked" in sync[0][0] and "--all-extras" in sync[0][0]
        assert sync[0][1] == {"UV_PROJECT_ENVIRONMENT": str(repo / ".venv")}
        # the link to MAIN's venv is gone (a sync through it would rewrite main), main untouched
        assert not (repo / ".venv").is_symlink() and main_marker.read_text() == "main's\n"
        venv_bin = str(repo / ".venv" / "bin")
        tools = [c for c in runner.calls if c[0] not in ("git", "uv")]
        assert {Path(c[0]).name for c in tools} >= {"ruff", "mypy", "python"}
        assert all(c[0].startswith(venv_bin) for c in tools), tools
        assert by["exactness"].status == "PASS" and by["ruff"].status == "PASS"

    def test_same_lock_keeps_main_venv(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        for root in (tmp_path, repo):
            (root / "uv.lock").write_text("same\n")
        (repo / "notes.txt").write_text("x\n")
        runner = DeclaredGit(repo)
        cfg = _cfg(tmp_path, repo, exactness="never", frontend="never")
        by = {c.name: c for c in swarm_check.run_checks(cfg, runner)}
        assert by["python-env"].detail == "main .venv"
        assert not any(c[0] == "uv" for c in runner.calls)
        assert any(c[0] == str(tmp_path / ".venv" / "bin" / "ruff") for c in runner.calls)

    def test_uv_sync_failure_fails_closed_before_any_tool(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        runner = DeclaredGit(repo)
        runner.responses = {"uv": (2, "Resolving...\nerror: lockfile needs to be updated")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="always", frontend="never")
        checks = swarm_check.run_checks(cfg, runner)
        by = {c.name: c for c in checks}
        assert by["python-env"].status == "FAIL"
        assert "lockfile needs to be updated" in by["python-env"].detail
        assert by["ruff"].status == "SKIP" and by["tests"].status == "SKIP"
        assert [c[0] for c in runner.calls if c[0] != "git"] == ["uv"]  # no tool ran on main's venv
        assert swarm_check.gate_summary(checks)[0].startswith("GATE FAIL")

    def test_python_env_flip_changes_the_pass_key(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "uv.lock").write_text("new\n")
        cfg = _cfg(tmp_path, repo)
        (tmp_path / "pyproject.toml").write_text("p\n")
        (repo / "pyproject.toml").write_text("p\n")
        (tmp_path / "uv.lock").write_text("new\n")
        main_env = swarm_check.cache_key("h", "s", cfg, "t")
        (tmp_path / "pyproject.toml").write_text(
            "p2\n"
        )  # main moved: the worktree's env now differs
        assert swarm_check.cache_key("h", "s", cfg, "t") != main_env

    def _frontend(
        self, tmp_path: Path, responses: dict[str, object]
    ) -> tuple[swarm_check.Check, DeclaredGit, Path, Path]:
        repo = _init_repo(tmp_path)
        (tmp_path / "frontend").mkdir()
        (tmp_path / "frontend" / "package.json").write_text('{"v": 1}\n')
        (repo / "frontend").mkdir()
        (repo / "frontend" / "package.json").write_text('{"v": 2}\n')
        marker = _shared_dir(
            tmp_path / "frontend" / "node_modules", repo / "frontend" / "node_modules"
        )
        runner = DeclaredGit(repo)
        runner.responses = responses  # type: ignore[assignment]
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never")
        fe = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["frontend"]
        return fe, runner, repo, marker

    def test_deps_differ_replaces_the_shared_node_modules_link(self, tmp_path: Path) -> None:
        fe, runner, repo, marker = self._frontend(tmp_path, {"pnpm": (0, "ok")})
        assert fe.status == "PASS", fe.detail
        assert "private node_modules" in fe.detail
        assert not (repo / "frontend" / "node_modules").is_symlink()
        assert marker.read_text() == "main's\n"  # main's node_modules untouched
        pnpm = [c for c in runner.calls if c[0] in ("pnpm", "npx")]
        assert pnpm[0] == ["pnpm", "install", "--frozen-lockfile"]  # before tsc/build/vitest

    def test_private_install_failure_fails_closed(self, tmp_path: Path) -> None:
        def pnpm(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
            return (1, "ERR_PNPM_OUTDATED_LOCKFILE") if "install" in cmd else (0, "ok")

        fe, runner, _, marker = self._frontend(tmp_path, {"pnpm": pnpm})
        assert fe.status == "FAIL" and "ERR_PNPM_OUTDATED_LOCKFILE" in fe.detail
        assert marker.read_text() == "main's\n"
        assert not any(c[:2] == ["pnpm", "build"] for c in runner.calls)

    def test_vitest_failure_fails_frontend(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "frontend").mkdir()
        (repo / "frontend" / "App.tsx").write_text("export {}\n")
        (tmp_path / "frontend").mkdir()
        runner = DeclaredGit(repo)
        runner.responses = {"vitest": (1, "Tests  1 failed | 41 passed (42)"), "pnpm": (0, "ok")}
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never")
        fe = {c.name: c for c in swarm_check.run_checks(cfg, runner)}["frontend"]
        assert fe.status == "FAIL" and "vitest" in fe.detail and "1 failed" in fe.detail
        assert ["pnpm", "vitest", "run"] in runner.calls


class TestCacheKey:
    def test_every_field_matters(self, tmp_path: Path) -> None:
        base = _cfg(tmp_path, tmp_path)
        key = swarm_check.cache_key("h", "s", base, "t")
        assert key != swarm_check.cache_key("h2", "s", base, "t")
        assert key != swarm_check.cache_key("h", "s2", base, "t")
        assert key != swarm_check.cache_key("h", "s", _cfg(tmp_path, tmp_path, scope=("x",)), "t")
        assert key != swarm_check.cache_key(
            "h", "s", _cfg(tmp_path, tmp_path, exactness="always"), "t"
        )
        assert key != swarm_check.cache_key(
            "h", "s", _cfg(tmp_path, tmp_path, frontend="always"), "t"
        )
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

    @pytest.mark.simple_target(
        reason="gate fail-closed: main's load_flakes.txt must key the PASS stamp"
    )
    def test_main_load_flakes_list_matters(self, tmp_path: Path) -> None:
        """Review rev-flakeretry-1 S1: a stamped PASS (one earned by a retry too) is not reused
        after main's load_flakes.txt loses an entry or gains a bad line."""
        cfg = _cfg(tmp_path, tmp_path)
        _write_flakes(tmp_path, f"{_FOO}::test_flaky  # vibe-quant-ca3v8 xdist catalog leak")
        listed = swarm_check.cache_key("h", "s", cfg, "t")
        _write_flakes(tmp_path, f"{_FOO}::test_flaky  # no bead id")
        assert swarm_check.cache_key("h", "s", cfg, "t") != listed


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

    def test_full_mode_never_reads_or_writes_a_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Review rev-gate-1 B4: the verifier's --full run backstops every per-round shortcut,
        so a PASS stamp for the same tree must not stand in for it, nor may it write one."""
        repo, _ = self._unstamped_repo(tmp_path)
        runner = self._runner(repo)
        cfg = swarm_check.Config(
            worktree=repo,
            main_root=repo,
            base="main",
            exactness="never",
            frontend="never",
            full=True,
        )
        key = swarm_check._cache_probe(cfg, runner)[0]
        assert key
        swarm_check.write_stamp(repo, runner, cfg, key, _PASS_CHECKS)
        runner.responses["pytest"] = (
            1,
            "FAILED tests/unit/test_foo.py::test_foo - boom\n1 failed in 0.01s",
        )
        assert self._main(repo, runner, "--full") == 1
        assert "PASS cache:" not in capsys.readouterr().out
        for stamp in self._stamps(repo):
            stamp.unlink()
        assert self._main(repo, self._runner(repo), "--full") == 0
        assert self._stamps(repo) == []

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

    def test_full_mode_does_not_reuse_a_round_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, _ = self._pass_repo(tmp_path)  # stamped by per-round runs
        runner = self._runner(repo)
        assert self._main(repo, runner, "--full") == 0
        assert "PASS cache:" not in capsys.readouterr().out
        assert any(
            c[c.index("pytest") + 1 :][:1] == ["tests/unit"] for c in runner.calls if "pytest" in c
        )

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

    def test_commit_during_run_not_stamped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A commit made while pytest runs must not be stamped (GATE PASS is for old HEAD).

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
            '#!/bin/sh\necho "pytest $@" >> "$STUB_LOG"\necho \'3 passed in 0.05s\'\n'
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
                "--base",
                "main",
                "--scope",
                "vibe_quant/*,*.txt",
                "--exactness",
                "never",
                "--frontend",
                "never",
                "--json",
                str(out_json),
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
                "--base",
                "main",
                "--scope",
                "docs/*",
                "--exactness",
                "never",
                "--frontend",
                "never",
                "--json",
                str(out_json),
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

    def test_the_file_store_with_an_agent_fails_closed(self, tmp_path: Path) -> None:
        """R27: the file store has no worker inbox any more: FAIL naming the store, never SKIP
        or PASS, even with an unread-looking chief DM or none at all."""
        for msgs in (
            [],
            [{"id": "m1", "from": "chief", "to": "sa", "kind": "question", "body": "x"}],
        ):
            sub = tmp_path / str(len(msgs))
            sub.mkdir()
            d = self._bus(sub, msgs)
            chk = self._check(sub, bus=str(d), bus_agent="sa")[0]
            assert (chk.status, chk.detail) == (
                "FAIL",
                "the file store has no worker inbox: unset SWARM_STORE",
            )
            assert list(d.iterdir()) == ([d / "messages.jsonl"] if msgs else [])  # nothing written

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
                "bus-inbox",
                "FAIL",
                "1 unread from chief in thread t1: m1 reread the brief — run: vb read t1, then reply",
            )

        monkeypatch.setattr(swarm_check, "bus_inbox_check", dm_after_precheck)
        rc = swarm_check.main(
            [
                str(repo),
                "--base",
                "main",
                "--exactness",
                "never",
                "--frontend",
                "never",
                "--bus",
                str(d),
                "--bus-agent",
                "sa",
            ],
            run=_pass_runner(repo),
        )
        out = capsys.readouterr().out
        assert rc == 1
        assert "PASS cache: reused stamp from" in out  # the tool checks really were cached
        assert "FAIL bus-inbox:" in out and "m1" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")


class _FakeBoardStore:
    """The board_store surface `_board_inbox` reads: one task thread for `sa` in a fixed channel."""

    def __init__(self, thread: list[dict[str, object]]) -> None:
        self.thread = thread

    def job_slug(self, name: str) -> str:
        return name

    def ns(self) -> str:
        return ""

    def profile_of(self, agent: str) -> str:
        return agent

    def read_channel(self, channel: str) -> list[dict[str, object]]:
        return [{"id": "root", "labels": {"agent": "sa", "from": "chief"}}]

    def read_thread(self, tid: str) -> list[dict[str, object]]:
        return self.thread


class TestBoardInboxOlderThanLastCommitEdges:
    """C10's WARN rule on the board inbox at its edges (moved from the file-store twin, R27): a
    message in the SAME second as the last commit, or a git error, stays a FAIL (fail closed)."""

    @staticmethod
    def _check(tmp_path: Path, repo: Path, created: int, monkeypatch: pytest.MonkeyPatch) -> Any:
        thread = [
            {"id": "root", "labels": {"from": "chief"}, "text": "t1"},
            {"id": "p1", "labels": {"from": "sa"}, "text": "[progress] red"},
            {"id": "c1", "labels": {"from": "chief"}, "text": "ruling",
             "createdAt": datetime.fromtimestamp(created, UTC).isoformat()},
        ]  # fmt: skip
        monkeypatch.setattr(swarm_check, "_board_store_module", lambda: _FakeBoardStore(thread))
        (tmp_path / "job" / "bus").mkdir(parents=True, exist_ok=True)
        return swarm_check._board_inbox(
            _cfg(tmp_path, repo, bus=str(tmp_path / "job" / "bus"), bus_agent="sa")
        )

    def test_a_git_error_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        not_a_repo = tmp_path / "not-a-repo"
        not_a_repo.mkdir()
        chk = self._check(tmp_path, not_a_repo, int(time.time()) - 3600, monkeypatch)
        assert chk.status == "FAIL" and "c1 ruling" in chk.detail


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
        assert (
            swarm_check.main(argv, run=FakeRunner()) == 1
        )  # the file store has no worker inbox (R27)
        assert "FAIL bus-inbox: the file store has no worker inbox" in capsys.readouterr().out
        assert list(explicit.iterdir()) == []


@pytest.fixture
def _inbox_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bus-inbox check PASSes: a class that gates with an agent tests another check, and the
    file store (the unit suite's) has no worker inbox (R27)."""
    monkeypatch.setattr(
        swarm_check,
        "bus_inbox_check",
        lambda cfg: swarm_check.Check("bus-inbox", "PASS", "stubbed"),
    )


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

    def _pin(self, tmp_path: Path, repo: Path, blobs: dict[str, str], **state: Any) -> Any:
        cfg = _cfg(tmp_path, repo, base="HEAD", exactness="never", frontend="never")
        runner = DeclaredGit(repo, blobs=blobs, dirty=["vibe_quant/foo.py"], **state)
        return {c.name: c for c in swarm_check.run_checks(cfg, runner)}["pinned-text"]

    def test_unrun_test_pinning_a_removed_literal_fails(self, tmp_path: Path) -> None:
        repo, blobs = _pinned_repo(tmp_path)
        pin = self._pin(tmp_path, repo, blobs)
        assert pin.status == "FAIL"
        assert "tests/unit/test_pin.py" in pin.detail
        assert f'pins removed text "{_PIN}"' in pin.detail


_STALE_THREAD = (
    "thread:delegated-task:command%3Amcp%3Aaaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee%3Adelegate-task%3A"
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


@pytest.mark.usefixtures("_inbox_pass")
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
        """The branch's uncommitted edit to vibe_quant/foo.py makes main's copy stale."""
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
        stale = "cd main && set -o pipefail; sed -n 1,80p vibe_quant/foo.py"
        db = _stale_db(tmp_path, "cc-x-1", stale, f"cd {repo} && sed -n 1p scripts/agents/bus.py")
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        assert swarm_check.main(self._argv(tmp_path, repo, "cc-x-1"), run=_stub_declared(repo)) == 1
        out = capsys.readouterr().out
        assert f"FAIL stale-reads: 1 stale command(s), e.g. {stale}; re-read from {repo}" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")

    def test_a_cached_pass_with_a_stale_read_fails_again(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = _init_pass_repo(tmp_path)
        db = _stale_db(tmp_path, "cc-x-1", "cat vibe_quant/foo.py")
        monkeypatch.setenv("SWARM_T3_DB", str(db))
        # the stamped template is its own main: say the branch changed the file it reads
        monkeypatch.setattr(
            swarm_check._stale_reads_module(),
            "unchanged_paths",
            lambda wt, main, **_: lambda p, **__: False,
        )
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
        assert "FAIL stale-reads: 1 stale command(s), e.g. cat vibe_quant/foo.py" in out
        assert out.strip().endswith("GATE FAIL (1 failed)")


# ------------------------------------------------------------------ bus-inbox on the board store


def _board_mode(board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """SWARM_STORE=board on the fixture board in a fresh channel namespace; returns the job dir
    (its ``bus/`` exists, as a rendered job's does)."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, board_server.env[key])
    monkeypatch.setenv("SWARM_STORE", "board")
    monkeypatch.setenv("SWARM_BOARD_NS", "g" + uuid.uuid4().hex[:12])
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    job = tmp_path / "20261120-Gate"
    (job / "bus").mkdir(parents=True)
    return job


def _job_channel(job: Path) -> str:
    return f"{os.environ['SWARM_BOARD_NS']}/job/{job.name.lower()}"


def _vb_msg(board_server: BoardServer, *args: str, profile: str, text: str) -> dict[str, Any]:
    """One `vb post|reply ... - --json` through the real CLI; the board's message back."""
    proc = board_server.vb(*args, "-", "--json", profile=profile, stdin=text)
    assert proc.returncode == 0, proc.stderr
    msg: dict[str, Any] = json.loads(proc.stdout)["message"]
    return msg


def _thread_root(
    board_server: BoardServer, job: Path, agent: str, text: str = "t1: build it"
) -> str:
    root = _vb_msg(
        board_server,
        "post",
        _job_channel(job),
        "--label",
        f"agent={agent}",
        "--label",
        "task=t1",
        profile="chief",
        text=text,
    )
    return str(root["id"])


def _say(board_server: BoardServer, root: str, profile: str, text: str) -> str:
    return str(_vb_msg(board_server, "reply", root, profile=profile, text=text)["id"])


def _commit_at(repo: Path, epoch: int) -> str:
    """An empty commit dated ``epoch`` (no sleeping to order it against board messages)."""
    env = {
        **_git_env(),
        "GIT_COMMITTER_DATE": f"@{epoch} +0000",
        "GIT_AUTHOR_DATE": f"@{epoch} +0000",
    }
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "wip"], check=True, env=env
    )
    return _git(repo, "rev-parse", "--short", "HEAD").strip()


def _linted_handoff(root: Path) -> str:
    """A maker block citing a real commit and a real PASS LOG of its tree (handoff_lint opens both)."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "a.py").write_text("x = 1\n")
    _git(root, "add", "a.py")
    _git(root, "commit", "-q", "-m", "feat: x")
    sha, tree = (
        _git(root, "rev-parse", "HEAD").strip(),
        _git(root, "rev-parse", "HEAD^{tree}").strip(),
    )
    log = root / "data" / "swarm" / "_gate-logs" / f"wt-{sha[:12]}-x.log"
    log.parent.mkdir(parents=True)
    log.write_text(f"$ ruff check\n[exit 0]\nHEAD {sha} TREE {tree}\nGATE PASS\n")
    return (
        "## Handoff\nstatus: done\nrole: implementer\nevidence:\n- red then green\n"
        f"- `GATE PASS` (LOG {log}, commit {sha[:8]})\nimprove: none\n"
    )


class TestBusInboxBoard:
    """bus-inbox with SWARM_STORE=board (job 20261120-board R18/R22): the agent's task thread in
    job/<job> is its inbox; chief messages after the agent's newest message are unread."""

    @staticmethod
    def _check(tmp_path: Path, job: Path, agent: str = "sa") -> Any:
        repo = tmp_path / "wt"
        if not repo.exists():
            repo = _init_repo(tmp_path)
        return swarm_check.bus_inbox_check(
            _cfg(tmp_path, repo, bus=str(job / "bus"), bus_agent=agent)
        )

    @pytest.mark.parametrize("broken", ["unreachable", "refused", "raises"])
    def test_a_board_that_cannot_be_read_fails_naming_board_up(
        self,
        board_server: BoardServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        broken: str,
    ) -> None:
        job = _board_mode(board_server, monkeypatch, tmp_path)
        _thread_root(board_server, job, "sa")
        if broken == "unreachable":
            monkeypatch.setenv("BOARD_URL", f"http://127.0.0.1:{_free_port()}/api/board")
            monkeypatch.setenv("BOARD_TOKEN", "t")
        elif broken == "refused":
            monkeypatch.setenv("BOARD_TOKEN", "not-the-token")
        else:

            def boom(channel: str) -> list[dict[str, object]]:
                raise RuntimeError("socket reset")

            monkeypatch.setattr(swarm_check._board_store_module(), "read_channel", boom)
        swarm_check._board_store_module().invalidate()
        chk = self._check(tmp_path, job)
        assert chk.status == "FAIL"
        assert chk.detail.startswith("cannot read the board for sa: ")
        assert chk.detail.endswith("— start it: scripts/agents/board-up.sh")

    @pytest.mark.slow(
        reason="board_server start/stop ~1.5 s when run alone + 5 real vb processes, ~3 s"
    )
    def test_chief_messages_after_the_agents_newest_fail_with_id_and_full_text(
        self, board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        job = _board_mode(board_server, monkeypatch, tmp_path)
        repo = _init_repo(tmp_path)
        _commit_at(repo, int(time.time()) - 3600)  # the board's createdAt is after it
        root = _thread_root(board_server, job, "sa")
        _say(board_server, root, "sa", "[progress] red tests")
        first = _say(board_server, root, "chief", "alpha " * 40)
        _say(board_server, root, "peer-2", "peer note")
        second = _say(board_server, root, "chief", "stop: rebase first")
        chk = self._check(tmp_path, job)
        assert (chk.status, chk.detail) == (
            "FAIL",
            f"2 unread from chief in thread {root}: {first} {'alpha ' * 40}; {second} stop: rebase first"
            f" — run: vb read {root}, then reply",
        )


_NO_FILE_INBOX = "the file store has no worker inbox: unset SWARM_STORE"


class TestDoneNeedsHandoffInTheGate:
    """A done message by the gate's own agent without a handoff that lints FAILs bus-inbox,
    in both stores (read through bus._all, so store-independent)."""

    @staticmethod
    def _check(tmp_path: Path, job: Path, agent: str = "sa") -> Any:
        repo = tmp_path / "wt"
        if not repo.exists():
            repo = _init_repo(tmp_path)
        return swarm_check.bus_inbox_check(
            _cfg(tmp_path, repo, bus=str(job / "bus"), bus_agent=agent)
        )

    def test_jsonl_done_without_then_with_a_linted_handoff(self, tmp_path: Path) -> None:
        job = tmp_path / "20261120-jd"
        (job / "bus").mkdir(parents=True)
        (job / "bus" / "messages.jsonl").write_text("".join(json.dumps(m) + "\n" for m in [
            {"id": "d1", "ts": "2026-11-20T10:00:00", "from": "sa", "to": "chief", "kind": "done", "body": "ok"},
            {"id": "d2", "ts": "2026-11-20T10:00:01", "from": "peer", "to": "chief", "kind": "done", "body": "ok"},
        ]))  # fmt: skip
        handoff = job / "handoffs" / "sa.md"
        chk = self._check(tmp_path, job)
        assert (chk.status, chk.detail) == (
            "FAIL",
            f"sa posted done (d1) without a handoff: {handoff} is missing — write and lint {handoff}; {_NO_FILE_INBOX}",
        )
        handoff.parent.mkdir()
        handoff.write_text("## Handoff\nstatus: done\nimprove: none\n")
        chk = self._check(tmp_path, job)
        assert chk.status == "FAIL"
        assert chk.detail == (
            f"sa posted done (d1) without a handoff: {handoff} fails lint: "
            "MISSING evidence: no '- ' item follows 'evidence:' in the last '## Handoff' block; "
            "MISSING gate: no 'GATE PASS' line in the last '## Handoff' block; "
            "MISSING gate: no _gate-logs/ LOG path in the last '## Handoff' block"
            f" — write and lint {handoff}; {_NO_FILE_INBOX}"
        )
        handoff.write_text(_linted_handoff(tmp_path / "gated"))
        chk = self._check(tmp_path, job)
        assert (chk.status, chk.detail) == (
            "FAIL",
            _NO_FILE_INBOX,
        )  # no done hit left: peer's done is not the gate's


def test_gate_script_imports_only_the_stdlib() -> None:
    """swarm-check.sh runs swarm_check.py with the system python3 (no venv): a venv-only
    import anywhere, also inside a function, crashes the gate (`coverage`, 2026-11-20)."""
    import ast

    tree = ast.parse(_SCRIPT.read_text())
    names = {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    names |= {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }
    assert sorted(n for n in names if n.split(".")[0] not in sys.stdlib_module_names) == []


_FOO = "tests/unit/test_foo.py"
_FLAKES = "scripts/agents/load_flakes.txt"


def _write_flakes(root: Path, *lines: str) -> None:
    path = root / _FLAKES
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def _junit_path(env: Mapping[str, str] | None) -> str | None:
    opts = (env or {}).get("PYTEST_ADDOPTS", "").split()
    return next((o.removeprefix("--junitxml=") for o in opts if o.startswith("--junitxml=")), None)


def _junit_case(node: str, child: str) -> str:
    file, *cls, name = node.split("::")
    classname = ".".join([file[:-3].replace("/", "."), *cls])
    return f'<testcase classname="{classname}" name="{name}" time="0.01">{child}</testcase>'


def _write_junit(path: str, cases: str) -> None:
    Path(path).write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")


class FlakeRunner(DeclaredGit):
    """The main pytest call fails the node ids in `red` (a junit row each, with a failure;
    `errors` get an error row) and exits `rc` (1); `summary` replaces pytest's count line. A
    serial re-run of ONE node id (`-n0`) exits `alone[node]` (default 0, passes alone) and
    writes a junit row for it: `alone_junit[node]` = passed|failure|skipped|none (no row)."""

    def __init__(
        self,
        repo: Path,
        red: Sequence[str],
        alone: Mapping[str, int] | None = None,
        *,
        rc: int = 1,
        errors: Sequence[str] = (),
        summary: str | None = None,
        alone_junit: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(repo, dirty=["vibe_quant/foo.py"])
        self.red = list(red)
        self.errors = list(errors)
        self.rc, self.summary = rc, summary
        self.alone = dict(alone or {})
        self.alone_junit = dict(alone_junit or {})
        self.alone_runs: list[str] = []
        self.alone_timeouts: list[int] = []

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int = 600,
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        if not any(Path(a).name == "pytest" for a in c):
            return super().__call__(cmd, cwd, env, timeout)
        self.calls.append(c)
        self.envs.append(env)
        if "-n0" in c:
            node = next(a for a in c if "::" in a)
            self.alone_runs.append(node)
            self.alone_timeouts.append(timeout)
            rc = self.alone.get(node, 0)
            outcome = self.alone_junit.get(node, "failure" if rc else "passed")
            if (junit := _junit_path(env)) and outcome != "none":
                child = "" if outcome == "passed" else f"<{outcome} message='x'/>"
                _write_junit(junit, _junit_case(node, child))
            return rc, (f"FAILED {node} - again\n1 failed" if rc else "1 passed in 0.01s")
        if record := (env or {}).get(swarm_check.DESELECT_ENV):
            (Path(record) / "gw0.txt").write_text("")
        junit = _junit_path(env)
        assert junit is not None
        cases = _junit_case(f"{_FOO}::test_ok", "")
        cases += "".join(_junit_case(node, "<failure message='boom'/>") for node in self.red)
        cases += "".join(_junit_case(node, "<error message='boom'/>") for node in self.errors)
        _write_junit(junit, cases)
        if not self.red and not self.errors:
            return 0, "1 passed in 0.01s"
        lines = "".join(f"FAILED {node} - boom\n" for node in self.red)
        lines += "".join(f"ERROR {node} - boom\n" for node in self.errors)
        counts = [f"{len(self.red)} failed"] if self.red else []
        counts += [f"{len(self.errors)} error{'s' * (len(self.errors) > 1)}"] if self.errors else []
        return self.rc, lines + (self.summary or f"{', '.join(counts)}, 1 passed in 0.01s")


class TwoCallRunner(FlakeRunner):
    """Review rev-flakeretry-1 B1 repro: the FIRST main call exits `first_rc` with an empty
    junit (no failed case); every later one fails the listed ids in `red`."""

    def __init__(self, repo: Path, red: Sequence[str], first_rc: int) -> None:
        super().__init__(repo, red)
        self.first_rc = first_rc
        self.main_calls = 0

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int = 600,
    ) -> tuple[int, str]:
        c = [str(x) for x in cmd]
        if any(Path(a).name == "pytest" for a in c) and "-n0" not in c:
            self.main_calls += 1
            if self.main_calls == 1:
                self.calls.append(c)
                junit = _junit_path(env)
                assert junit is not None
                _write_junit(junit, "")
                return self.first_rc, "ERROR: usage / no tests ran\n"
        return super().__call__(cmd, cwd, env, timeout)


class TestLoadFlakeRetry:
    """User decision 2026-11-23 (q2=A): the gate re-runs a test known to be load-flaky once,
    alone, before calling it a FAIL; the first failure stays in the LOG; a second failure stays
    a FAIL. The list is scripts/agents/load_flakes.txt in the MAIN checkout (chief R-flr-1: a
    diff never whitelists its own failing test); both copies are format-checked."""

    @staticmethod
    def _gate(tmp_path: Path, runner: Any) -> dict[str, Any]:
        cfg = _cfg(tmp_path, runner.repo if isinstance(runner, FlakeRunner) else runner._run.repo,
                   base="HEAD", exactness="never", frontend="never")  # fmt: skip
        return {c.name: c for c in swarm_check.run_checks(cfg, runner)}

    def test_listed_failure_that_passes_alone_passes_the_row(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        node = f"{_FOO}::test_flaky"
        _write_flakes(
            tmp_path, "# load flakes", "", f"{node}  # vibe-quant-ca3v8 xdist catalog leak"
        )
        runner = FlakeRunner(repo, [node])
        log = swarm_check.ToolLog(runner)
        tests = self._gate(tmp_path, log)["tests"]
        assert tests.status == "PASS"
        assert tests.detail.startswith(
            f"RETRIED-ALONE: {node} failed in the gate, passed alone (load_flakes.txt) | "
        )
        assert runner.alone_runs == [node]
        retry = next(c for c in runner.calls if "-n0" in c)
        assert retry[-4:] == [node, "-p", "no:randomly", "-n0"]
        # the LOG keeps the first failure, then the retry under its note
        texts = [e if isinstance(e, str) else e[0] + "\n" + e[2] for e in log.entries]
        first = next(i for i, t in enumerate(texts) if f"FAILED {node} - boom" in t)
        note = texts.index(f"RETRY-ALONE {node}")
        assert first < note and "-n0" in texts[note + 1]

    def test_a_second_failure_stays_a_fail(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        node = f"{_FOO}::test_flaky"
        today = self._gate(tmp_path, FlakeRunner(repo, [node]))["tests"]  # no list yet
        _write_flakes(tmp_path, f"{node}  # vibe-quant-ca3v8 xdist catalog leak")
        tests = self._gate(tmp_path, FlakeRunner(repo, [node], alone={node: 1}))["tests"]
        assert tests.status == "FAIL"
        assert tests.detail == f"{today.detail} | retry alone also failed: {node}"

    @pytest.mark.parametrize("listed", [0, 1])
    def test_an_unlisted_failure_never_retries_and_fails_as_today(
        self, tmp_path: Path, listed: int
    ) -> None:
        repo = _init_repo(tmp_path)
        flaky, real = f"{_FOO}::test_flaky", f"{_FOO}::TestReal::test_real"
        red = [flaky, real][1 - listed :]
        today = self._gate(tmp_path, FlakeRunner(repo, red))["tests"]
        _write_flakes(tmp_path, f"{flaky}  # vibe-quant-ca3v8 xdist catalog leak")
        runner = FlakeRunner(repo, red)
        tests = self._gate(tmp_path, runner)["tests"]
        assert (tests.status, tests.detail) == ("FAIL", today.detail)
        assert runner.alone_runs == []

    @pytest.mark.parametrize(("count", "retried"), [(3, True), (4, False)])
    def test_at_most_three_failed_cases_retry(
        self, tmp_path: Path, count: int, retried: bool
    ) -> None:
        repo = _init_repo(tmp_path)
        red = [f"{_FOO}::test_flaky_{i}" for i in range(count)]
        _write_flakes(tmp_path, *(f"{n}  # vibe-quant-ca3v8 xdist catalog leak" for n in red))
        runner = FlakeRunner(repo, red)
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == ("PASS" if retried else "FAIL")
        assert runner.alone_runs == (red if retried else [])

    def test_the_text_cap_does_not_hide_an_unlisted_failure(self, tmp_path: Path) -> None:
        """failed_tests() names 3 ids; the 4th, unlisted, is only in the junit."""
        repo = _init_repo(tmp_path)
        listed = [f"{_FOO}::test_flaky_{i}" for i in range(3)]
        _write_flakes(
            tmp_path, *(f"{n}  # vibe-quant-ca3v8 xdist catalog leak" for n in listed[:2])
        )
        runner = FlakeRunner(repo, listed)
        assert self._gate(tmp_path, runner)["tests"].status == "FAIL"
        assert runner.alone_runs == []

    def test_a_policy_fail_fast_call_never_retries(self, tmp_path: Path) -> None:
        repo = _policy_repo(tmp_path)
        node = "tests/unit/test_discovery_liquidation.py::test_policy"
        _write_flakes(tmp_path, f"{node}  # vibe-quant-ca3v8 xdist catalog leak")
        runner = FlakeRunner(repo, [node])
        runner.committed, runner.porcelain = [], [" M vibe_quant/unrelated.py"]
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == "FAIL" and swarm_check.POLICY_STOP in tests.detail
        assert runner.alone_runs == []

    def test_a_diff_cannot_whitelist_its_own_failing_test(self, tmp_path: Path) -> None:
        """Chief R-flr-1: the worktree copy is format-checked, never trusted for entries."""
        repo = _init_repo(tmp_path)
        node = f"{_FOO}::test_flaky"
        _write_flakes(repo, f"{node}  # vibe-quant-ca3v8 xdist catalog leak")
        runner = FlakeRunner(repo, [node])
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == "FAIL" and f"FAILED {node}" in tests.detail
        assert runner.alone_runs == []

    @pytest.mark.parametrize("where", ["main", "worktree"])
    @pytest.mark.parametrize(
        ("line", "why"),
        [
            (f"{_FOO}::test_flaky  # xdist catalog leak", "has no bead id"),
            (f"{_FOO}::test_flaky", "has no bead id"),
            (f"{_FOO}::test_flaky  # vibe-quant-ca3v8", "has no reason"),
        ],
    )
    def test_a_line_without_bead_id_fails_closed(
        self, tmp_path: Path, where: str, line: str, why: str
    ) -> None:
        repo = _init_repo(tmp_path)
        good = f"{_FOO}::test_other  # vibe-quant-28p7p fill-tick build under xdist"
        _write_flakes(tmp_path if where == "main" else repo, "# header", good, "", line)
        runner = FlakeRunner(repo, [])  # a green run: the list alone fails the row
        tests = self._gate(tmp_path, runner)["tests"]
        assert (tests.status, tests.detail) == ("FAIL", f"load_flakes.txt line 4 {why}")

    def test_a_stale_entry_is_not_an_error(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _write_flakes(
            tmp_path, f"{_FOO}::test_renamed_long_ago  # vibe-quant-ca3v8 xdist catalog leak"
        )
        tests = self._gate(tmp_path, FlakeRunner(repo, []))["tests"]
        assert tests.status == "PASS" and "RETRIED" not in tests.detail

    @pytest.mark.parametrize(
        ("entry", "retried"),
        [
            ("TestX::test_p", ["TestX::test_p[a]", "TestX::test_p[b]"]),
            ("TestX::test_p[a]", None),
            ("test_p", None),
        ],
    )
    def test_parametrized_ids(self, tmp_path: Path, entry: str, retried: list[str] | None) -> None:
        """A parametrized id matches exactly; a bare file::name matches all its parameter ids
        (and only within its own class path)."""
        repo = _init_repo(tmp_path)
        red = [f"{_FOO}::TestX::test_p[a]", f"{_FOO}::TestX::test_p[b]"]
        _write_flakes(tmp_path, f"{_FOO}::{entry}  # vibe-quant-ca3v8 xdist catalog leak")
        runner = FlakeRunner(repo, red)
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == ("PASS" if retried else "FAIL")
        assert runner.alone_runs == [f"{_FOO}::{r}" for r in retried or []]

    def test_a_retried_run_stores_no_file_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _init_repo(tmp_path)
        node = f"{_FOO}::test_flaky"
        _write_flakes(tmp_path, f"{node}  # vibe-quant-ca3v8 xdist catalog leak")
        stored: list[list[str]] = []
        monkeypatch.setattr(
            swarm_check,
            "_file_cache_plan",
            lambda *a: swarm_check.FileCacheRun([_FOO], {}, "env", ("a" * 40, "b" * 40, 1)),
        )
        monkeypatch.setattr(
            swarm_check,
            "_store_file_passes",
            lambda cfg, run, cache, passed: stored.append(list(passed)),
        )
        assert self._gate(tmp_path, FlakeRunner(repo, []))["tests"].status == "PASS"
        assert stored == [[_FOO]]  # control: a green run stores its file
        stored.clear()
        assert self._gate(tmp_path, FlakeRunner(repo, [node]))["tests"].status == "PASS"
        assert stored == []


class TestLoadFlakeRetryFailsClosed:
    """Round 2 (review rev-flakeretry-1 B1, S1, N1-N3; chief F7-F12): a retry may only turn a
    FAIL into a PASS when every red pytest call is fully explained by listed failed cases."""

    _gate = staticmethod(TestLoadFlakeRetry._gate)
    _NODE = f"{_FOO}::test_flaky"
    _LINE = f"{_NODE}  # vibe-quant-ca3v8 xdist catalog leak"

    @staticmethod
    def _plan(monkeypatch: pytest.MonkeyPatch, *calls: list[str]) -> None:
        plan = swarm_check.TestPlan(
            [(args, False, "") for args in calls], frozenset(), False, [], []
        )
        monkeypatch.setattr(swarm_check, "plan_tests", lambda cfg, state: plan)

    @pytest.mark.parametrize("first_rc", [1, 2, 3, 4, 5])
    def test_another_red_call_without_a_failed_case_blocks_the_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_rc: int
    ) -> None:
        """F7: rc 2/3/4/5 (or rc 1 with an empty junit) in one call, a listed flake that passes
        alone in the other: FAIL exactly as without the list, no retry."""
        repo = _init_repo(tmp_path)
        self._plan(monkeypatch, ["tests/unit/test_new.py"], [_FOO])
        today = self._gate(tmp_path, TwoCallRunner(repo, [self._NODE], first_rc))["tests"]
        _write_flakes(tmp_path, self._LINE)
        runner = TwoCallRunner(repo, [self._NODE], first_rc)
        tests = self._gate(tmp_path, runner)["tests"]
        assert (tests.status, tests.detail) == ("FAIL", today.detail)
        assert runner.alone_runs == []

    def test_two_red_calls_of_listed_flakes_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control for F7: two calls, both rc 1 with only the listed case, still retry."""
        repo = _init_repo(tmp_path)
        self._plan(monkeypatch, ["tests/unit/test_new.py"], [_FOO])
        _write_flakes(tmp_path, self._LINE)
        runner = FlakeRunner(repo, [self._NODE])
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == "PASS" and tests.detail.startswith(f"RETRIED-ALONE: {self._NODE} ")
        assert runner.alone_runs == [self._NODE]

    @pytest.mark.parametrize("rc", [2, 3])
    def test_a_listed_failure_in_a_call_that_exited_other_than_one_never_retries(
        self, tmp_path: Path, rc: int
    ) -> None:
        """F7: an interrupted (2) or internal-error (3) call whose junit holds a listed case."""
        repo = _init_repo(tmp_path)
        today = self._gate(tmp_path, FlakeRunner(repo, [self._NODE], rc=rc))["tests"]
        _write_flakes(tmp_path, self._LINE)
        runner = FlakeRunner(repo, [self._NODE], rc=rc)
        tests = self._gate(tmp_path, runner)["tests"]
        assert (tests.status, tests.detail) == ("FAIL", today.detail)
        assert runner.alone_runs == []

    @pytest.mark.parametrize(
        ("summary", "retried"),
        [
            ("1 failed, 1 passed in 0.01s", True),
            ("1 passed in 0.01s", False),  # rc 1 but pytest counts no failure
            ("2 failed, 1 passed in 0.01s", False),
            ("1 failed, 1 passed, 1 error in 0.01s", False),  # a session-level error (N2)
            ("INTERNALERROR> boom", False),  # no count line at all
        ],
    )
    def test_the_summary_count_must_equal_the_listed_cases(
        self, tmp_path: Path, summary: str, retried: bool
    ) -> None:
        """F8: pytest's failed+errors count == the failed junit cases of that call."""
        repo = _init_repo(tmp_path)
        _write_flakes(tmp_path, self._LINE)
        runner = FlakeRunner(repo, [self._NODE], summary=summary)
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == ("PASS" if retried else "FAIL")
        assert runner.alone_runs == ([self._NODE] if retried else [])

    def test_errored_cases_count_as_errors(self, tmp_path: Path) -> None:
        """F8: `1 failed, 1 error` with one failed and one errored listed case retries both."""
        repo = _init_repo(tmp_path)
        other = f"{_FOO}::TestX::test_setup_flaky"
        _write_flakes(
            tmp_path, self._LINE, f"{other}  # vibe-quant-28p7p fixture build under xdist"
        )
        runner = FlakeRunner(repo, [self._NODE], errors=[other])
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == "PASS"
        assert runner.alone_runs == [self._NODE, other]

    @pytest.mark.parametrize(
        ("outcome", "passed"), [("passed", True), ("skipped", False), ("none", False)]
    )
    def test_passed_alone_means_a_passed_junit_case(
        self, tmp_path: Path, outcome: str, passed: bool
    ) -> None:
        """F9: rc 0 alone with the test skipped or not reported (deselected) is no pass."""
        repo = _init_repo(tmp_path)
        _write_flakes(tmp_path, self._LINE)
        runner = FlakeRunner(repo, [self._NODE], alone_junit={self._NODE: outcome})
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == ("PASS" if passed else "FAIL")
        assert passed or tests.detail.endswith(f"retry alone also failed: {self._NODE}")

    def test_the_alone_junit_must_name_the_retried_id(self, tmp_path: Path) -> None:
        """F9: a passed row for ANOTHER parameter id of the same test is no pass."""
        repo = _init_repo(tmp_path)
        node = f"{_FOO}::TestX::test_p[a]"
        _write_flakes(tmp_path, f"{node}  # vibe-quant-ca3v8 xdist catalog leak")

        class OtherId(FlakeRunner):
            def __call__(self, cmd: Sequence[str], cwd: Path, env: Mapping[str, str] | None = None,
                         timeout: int = 600) -> tuple[int, str]:  # fmt: skip
                rc, out = super().__call__(cmd, cwd, env, timeout)
                if "-n0" in [str(x) for x in cmd] and (junit := _junit_path(env)):
                    _write_junit(junit, _junit_case(f"{_FOO}::TestX::test_p[b]", ""))
                return rc, out

        assert self._gate(tmp_path, OtherId(repo, [node]))["tests"].status == "FAIL"

    def test_the_alone_rerun_is_capped_and_a_timeout_fails(self, tmp_path: Path) -> None:
        """F11: each alone re-run gets SLOW_RERUN_TIMEOUT; a timeout (run's rc 124) is a FAIL."""
        repo = _init_repo(tmp_path)
        _write_flakes(tmp_path, self._LINE)
        runner = FlakeRunner(
            repo, [self._NODE], {self._NODE: 124}, alone_junit={self._NODE: "none"}
        )
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == "FAIL" and tests.detail.endswith(
            f"retry alone also failed: {self._NODE}"
        )
        assert runner.alone_timeouts == [swarm_check.SLOW_RERUN_TIMEOUT]

    @pytest.mark.parametrize(("mark", "retried"), [("timing", False), ("not timing", True)])
    def test_a_timing_call_is_never_retry_eligible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mark: str, retried: bool
    ) -> None:
        """F12 (chief ruling): a red `-m timing` call never retries."""
        repo = _init_repo(tmp_path)
        self._plan(monkeypatch, [_FOO, "-m", mark])
        _write_flakes(tmp_path, self._LINE)
        runner = FlakeRunner(repo, [self._NODE])
        tests = self._gate(tmp_path, runner)["tests"]
        assert tests.status == ("PASS" if retried else "FAIL")
        assert runner.alone_runs == ([self._NODE] if retried else [])


@pytest.mark.simple_target(
    reason="gate base selection: ancestry direction only a real repo proves; a wrong base fails scope or hides diff"
)
def test_an_explicit_base_on_main_follows_a_later_main_merge(tmp_path: Path) -> None:
    """A brief pins `--base <sha>`; the chief later merges main into the worktree. Gating
    against the old sha FAILed scope on main's own files, so the base moves to the new
    merge-base. Before the merge, and for a base main never had, the --base stays verbatim."""
    repo = _init_repo(tmp_path)
    start = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-qb", "side")
    (repo / "side.py").write_text("S = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "side")
    side = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-qb", "feat")
    (repo / "feat.py").write_text("F = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat")
    run = swarm_check.real_run
    assert swarm_check.resolve_base(repo, start, run) == (start, "")

    _git(repo, "checkout", "-q", "main")
    (repo / "later.py").write_text("L = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "main moves on")
    later = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "feat")
    _git(repo, "merge", "-q", "--no-edit", "main")

    base, info = swarm_check.resolve_base(repo, start, run)
    assert base == later
    assert (
        info
        == f"INFO base: --base {start[:12]} is on main and the branch merged main since; using merge-base {later[:12]}"
    )
    assert swarm_check.resolve_base(repo, side, run) == (side, "")
    assert swarm_check.resolve_base(repo, "main", run) == ("main", "")
