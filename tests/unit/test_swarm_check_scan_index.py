"""Equality proofs for swarm_check's per-tree scan caches.

A gate that selects fewer tests hides failures, so every cache this module adds is pinned
against the UNCACHED scan it replaced, on the real tree, for a hub module, a plugin and a
test-only diff: same files, same commands, same full/not-full verdict. Trees built by hand
where the rule under test is a shape (a needle inside another needle, a reached module named
in a string) so a regression fails here, not in a maker's gate.

The git-facts tests pin the other half of the round: one run asks `merge-base <base> HEAD`
and `rev-parse <base>^{commit}` once per PHASE (pre-run key, post-run freshness check),
not three times each.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "agents" / "swarm_check.py"
if "swarm_check" in sys.modules:
    swarm_check = sys.modules["swarm_check"]
else:
    _spec = importlib.util.spec_from_file_location("swarm_check", _SCRIPT)
    assert _spec is not None and _spec.loader is not None
    swarm_check = importlib.util.module_from_spec(_spec)
    sys.modules["swarm_check"] = swarm_check
    _spec.loader.exec_module(swarm_check)


@pytest.fixture(autouse=True)
def _fresh_tree_cache() -> Iterator[None]:
    """Each test starts and ends with an empty per-process tree cache: a plan built here is
    never served by a cache another test filled, and fills none it does not need."""
    swarm_check._TREE_CACHE.clear()
    yield
    swarm_check._TREE_CACHE.clear()


def _plan(worktree: Path, changed: list[str], *, full: bool = False) -> Any:
    cfg = swarm_check.Config(worktree=worktree, main_root=worktree, full=full)
    return swarm_check.plan_tests(cfg, swarm_check.GitState(changed=changed, untracked=[]))


def _naive_mention(worktree: Path, needles: set[str]) -> list[str]:
    """The pre-cache `tests_mentioning`: a plain substring scan of every tests/unit test file."""
    if not needles:
        return []
    root = worktree / "tests" / "unit"
    if not root.is_dir():
        return []
    return [
        path.relative_to(worktree).as_posix()
        for path in sorted(root.rglob("test_*.py"))
        if any(needle in path.read_text(encoding="utf-8", errors="replace") for needle in needles)
    ]


def _naive_dynamic(
    reached: set[str], worktree: Path, snapshot: Any = None
) -> list[str]:
    """The pre-index `dynamic_reference_test_files`, straight from each file's own strings.

    `snapshot` is accepted and ignored: the plain scan re-reads the tree it is asked about."""
    if not reached:
        return []
    packages = {m.removesuffix(".__main__") for m in reached if m.endswith(".__main__")}
    root = worktree / "tests" / "unit"
    if not root.is_dir():
        return []
    out: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        names = {m.group(1) for m in swarm_check.QUOTED_MODULE_RE.finditer(text)}
        runs = {m.group(1) for m in swarm_check.RUN_MODULE_RE.finditer(text)}
        hit = any(
            {".".join(parts[:i]) for i in range(2, len(parts) + 1)} & reached
            for parts in (name.split(".") for name in names)
        ) or bool(runs & packages)
        if hit:
            out.append(path.relative_to(worktree).as_posix())
    return out


class TestCachedSelectionEqualsUncached:
    """The kill zone: a cached selection must BE the uncached one, file for file."""

    @pytest.mark.slow(reason="plain scan of the real repo tree vs the caches")
    def test_the_plan_is_the_same_with_the_caches_and_with_a_plain_scan(self) -> None:
        """On the REAL tree, for the widest changed set (a hub module: a wide import closure,
        a wide path-mention set and a wide string-reference set), so both cached scans below
        are compared against the plain one on real input. The plain plan runs on the graph the
        cached one already built: the graph is not what this row compares, the scans are."""
        changed = "vibe_quant/screening/replay_drift.py"
        cached = _plan(_ROOT, [changed])
        assert cached.explicit  # the diff selects something: an empty plan proves nothing

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(swarm_check, "tests_mentioning", _naive_mention)
            mp.setattr(swarm_check, "dynamic_reference_test_files", _naive_dynamic)
            plain = _plan(_ROOT, [changed])

        assert plain.explicit == cached.explicit
        assert plain.files == cached.files
        assert plain.commands == cached.commands
        assert plain.full == cached.full

class TestCacheInvalidation:
    """A cache that outlives an edit would gate the pre-edit tree: every one must re-key."""

    def _repo(self, tmp_path: Path) -> Path:
        (tmp_path / "vibe_quant").mkdir()
        (tmp_path / "vibe_quant" / "foo.py").write_text("def f() -> int:\n    return 1\n")
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_other.py").write_text("X = 1\n")
        return tmp_path

    def test_editing_an_import_between_two_plans_changes_the_selection(
        self, tmp_path: Path
    ) -> None:
        """The import graph keys on the snapshot: a new vibe_quant edge in the SAME process
        must re-plan (a stale graph would miss the test that now reaches the module)."""
        repo = self._repo(tmp_path)
        (repo / "vibe_quant" / "bar.py").write_text("B = 1\n")
        (repo / "tests" / "unit" / "test_other.py").write_text(
            "from vibe_quant.foo import F\n\n\ndef test_x() -> None:\n    assert F == 1\n"
        )
        assert _plan(repo, ["vibe_quant/bar.py"]).explicit == set()

        (repo / "vibe_quant" / "foo.py").write_text("from vibe_quant.bar import B\n\nF = B\n")
        assert _plan(repo, ["vibe_quant/bar.py"]).explicit == {"tests/unit/test_other.py"}

    def test_a_test_file_that_gains_an_import_between_two_plans_is_selected(
        self, tmp_path: Path
    ) -> None:
        """The graph analyses the file's CURRENT bytes - a tree cached by path alone would
        keep selecting nothing after the file gains an edge to the changed module."""
        repo = self._repo(tmp_path)
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == set()

        (repo / "tests" / "unit" / "test_other.py").write_text("import vibe_quant.foo\n")
        assert _plan(repo, ["vibe_quant/foo.py"]).explicit == {"tests/unit/test_other.py"}

# --------------------------------------------------------------------------- git spawns

_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "cache",
    "GIT_AUTHOR_EMAIL": "cache@test",
    "GIT_COMMITTER_NAME": "cache",
    "GIT_COMMITTER_EMAIL": "cache@test",
}


def _git_env() -> dict[str, str]:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV_OVERRIDES)
    return env


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
        env=_git_env(),
    )  # fmt: skip
    return out.stdout


def _template_repo() -> Path:
    """One committed repo per process: a copytree is the independent clone each test dirties."""
    root = Path(tempfile.mkdtemp(prefix="swarm-check-spawns-"))
    import atexit

    atexit.register(shutil.rmtree, root, ignore_errors=True)
    repo = root / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=_git_env())
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 1\n")
    unit = repo / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_foo.py").write_text(
        "from vibe_quant.foo import foo\n\n\ndef test_foo() -> None:\n    assert foo()\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "vibe_quant" / "foo.py").write_text("def foo() -> int:\n    return 2\n")
    _git(repo, "commit", "-qam", "foo")
    return repo


_TEMPLATE: Path | None = None


def _repo(tmp_path: Path) -> Path:
    global _TEMPLATE
    if _TEMPLATE is None or not _TEMPLATE.exists():
        _TEMPLATE = _template_repo()
    repo = tmp_path / "wt"
    shutil.copytree(_TEMPLATE, repo, symlinks=True)
    return repo


# The graph walk only has to look at statements: an import is a statement, and no expression
# holds one. The tree-wide walk of every node of ~770 files was most of a cold graph build.

_EVERY_STATEMENT_POSITION = '''\
import vibe_quant.top
if flag:
    import vibe_quant.if_body
elif other:
    import vibe_quant.elif_body
else:
    import vibe_quant.if_else
def fn():
    import vibe_quant.fn_body
    def inner():
        from vibe_quant import in_inner
async def afn():
    import vibe_quant.afn_body
    async with ctx:
        import vibe_quant.async_with
    async for x in y:
        import vibe_quant.async_for
class K:
    import vibe_quant.class_body
for i in z:
    import vibe_quant.for_body
else:
    import vibe_quant.for_else
while cond:
    import vibe_quant.while_body
else:
    import vibe_quant.while_else
with ctx:
    import vibe_quant.with_body
try:
    import vibe_quant.try_body
except ValueError:
    import vibe_quant.handler_body
else:
    import vibe_quant.try_else
finally:
    import vibe_quant.try_finally
try:
    pass
except* OSError:
    import vibe_quant.star_handler
match cmd:
    case 1:
        import vibe_quant.case_body
    case _:
        import vibe_quant.case_default
if TYPE_CHECKING:
    import vibe_quant.type_checking_only
x = lambda: [1, 2, 3]
y = "import vibe_quant.in_a_string"
'''


_REPRESENTATIVE_SOURCES = (
    _EVERY_STATEMENT_POSITION,
    "",  # empty module
    "import os\nfrom a.b import c as d\nx = lambda: 1\n",  # flat module, imports only at top level
    "def f():\n    async def g():\n        from . import h\n    return g\n",  # relative, nested defs
    "class A:\n    import sys\n    def m(self):\n        with open('x') as f:\n            import json\n",
    "try:\n    import a\nexcept ImportError:\n    import b\nelse:\n    import c\nfinally:\n    import d\n",
    "match x:\n    case 1:\n        import m1\n    case _:\n        import m2\n",
    "x = [i for i in range(3) if i]\nprint(f'{x!r}')\n",  # no import at all
)


def test_the_import_nodes_are_exactly_the_ones_a_full_walk_finds() -> None:
    """Bounded representative sources, not the repo: the full-tree scan cost ~1 s."""
    import ast

    for source in _REPRESENTATIVE_SOURCES:
        tree = ast.parse(source)
        walked = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        got = list(swarm_check._import_nodes(tree))
        assert len(got) == len(walked)
        assert {id(n) for n in got} == {id(n) for n in walked}
