"""Cap on how many vibe_quant modules make swarm_check's per-round plan fall back to the full
tests/unit.

Today only the 28 package `__init__.py` files go full (their import reach is everything). A
module that newly goes full - a dynamic loader the selection does not model, a graph edge it
cannot read, a module a conftest imports at module level - turns a seconds-long gate into the
whole suite for every diff touching it, so growing the count must be a decision, not an
accident.

A single-file diff reaches the full suite through THREE rules, all checked here over EVERY
module of the real tree: `full_suite_reason` (path/stem, pure), the tree-wide
`unmodelled_loaders` / `check_pytest_plugins` scans (once per tree), and the module-specific
`conftest_fixture_test_files` (a conftest at tests/ or tests/unit/ whose reached dependency
sends the subtree = the whole suite). The import graph is built ONCE and every module's
reach is read from it; `conftest_fixture_test_files` depends on `reached` only through
`reached & (every name a conftest imports or a fixture depends on)`, so modules are
grouped by that intersection and the real function runs once per group (242 modules -> ~10
calls). The composition into a real plan is pinned too, on a tree small enough to be free.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

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

MAX_FULL_MODULES = 28


def _modules(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in (root / "vibe_quant").rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _conftest_full_modules(root: Path, modules: Sequence[str]) -> list[str]:
    """The modules whose reach sends `conftest_fixture_test_files` to FullSuite, from ONE
    import graph. Modules sharing `reached & D` (D = every conftest import / fixture dep) get
    the same answer, so the real function runs once per distinct intersection."""
    snapshot = swarm_check.plan_snapshot(root)
    graph, importers = swarm_check.import_graph_pair(root, (), snapshot)
    index = swarm_check.conftest_index(root, graph, snapshot)
    deps: set[str] = set()
    for _scope, shared, fixtures in index.conftests:
        deps.update(shared)
        for fx in fixtures:
            deps.update(fx.deps)
    verdict: dict[frozenset[str], bool] = {}
    full: list[str] = []
    for rel in modules:
        reached = swarm_check.reached_modules([rel], graph, importers)
        key = frozenset(reached & deps)
        if key not in verdict:
            try:
                swarm_check.conftest_fixture_test_files(reached, graph, root, snapshot)
                verdict[key] = False
            except swarm_check.FullSuite:
                verdict[key] = True
        if verdict[key]:
            full.append(rel)
    return full


def _full_modules(modules: list[str], worktree: Path) -> list[str]:
    """The modules whose single-file diff plans the full suite (plan_tests itself)."""
    cfg = swarm_check.Config(worktree=worktree, main_root=worktree)
    return [
        rel
        for rel in modules
        if swarm_check.plan_tests(cfg, swarm_check.GitState(changed=[rel], untracked=[])).full
    ]


def _one_module_tree(tmp_path: Path, conftest: str = "") -> Path:
    """A tree of one module and the one test that imports it (`conftest`: tests/unit/'s)."""
    (tmp_path / "vibe_quant").mkdir()
    (tmp_path / "vibe_quant" / "__init__.py").write_text("")
    (tmp_path / "vibe_quant" / "mod.py").write_text("def f() -> int:\n    return 1\n")
    (tmp_path / "vibe_quant" / "other.py").write_text("X = 1\n")
    unit = tmp_path / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_mod.py").write_text(
        "from vibe_quant.mod import f\n\n\ndef test_f() -> None:\n    assert f() == 1\n"
    )
    if conftest:
        (unit / "conftest.py").write_text(conftest)
    return tmp_path


@pytest.fixture(scope="module")
def real_tree() -> dict[str, Any]:
    """The real tree's module list; the import graph (cached by swarm_check) is built here once."""
    modules = _modules(_ROOT)
    assert len(modules) > 200  # the walk found the tree, not an empty dir
    snapshot = swarm_check.plan_snapshot(_ROOT)
    swarm_check.import_graph_pair(_ROOT, (), snapshot)
    return {"modules": modules, "snapshot": snapshot}


@pytest.mark.slow(reason="builds the real repo import graph once, then plans every module from it")
def test_at_most_the_package_inits_plan_the_full_suite(
    tmp_path: Path, real_tree: dict[str, Any]
) -> None:
    modules: list[str] = real_tree["modules"]
    inits = [rel for rel in modules if rel.endswith("__init__.py")]

    # rule 1: the path/stem check every single-file diff goes through
    path_full = [rel for rel in modules if swarm_check.full_suite_reason([rel])]
    assert path_full == inits
    assert len(path_full) <= MAX_FULL_MODULES

    # rule 2: the selection's own tree-wide escape hatches, one scan each of the REAL tree
    assert swarm_check.unmodelled_loaders(_ROOT, real_tree["snapshot"]) == []
    swarm_check.check_pytest_plugins(_ROOT, real_tree["snapshot"])  # raises FullSuite

    # rule 3: no module a conftest reaches (imported at module level, or a fixture / hook
    # dependency) sends tests/ or tests/unit/ to the full suite; inits go full by rule 1
    conftest_full = [
        rel for rel in _conftest_full_modules(_ROOT, modules) if not rel.endswith("__init__.py")
    ]
    assert conftest_full == [], f"modules a conftest reaches plan the full suite: {conftest_full}"

    # and the composition, end to end (plan_tests, not its rules): an ordinary module diff
    # takes the selection path instead of the suite
    repo = _one_module_tree(tmp_path)
    plan = swarm_check.plan_tests(
        swarm_check.Config(worktree=repo, main_root=repo),
        swarm_check.GitState(changed=["vibe_quant/mod.py"], untracked=[]),
    )
    assert not plan.full
    assert plan.explicit == {"tests/unit/test_mod.py"}


def test_the_cap_counts_a_module_the_planner_sends_to_the_full_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teeth: with the planner's loader check reporting an unmodelled loader, a non-init
    module is counted (the cap test above would then fail)."""
    monkeypatch.setattr(
        swarm_check, "unmodelled_loaders", lambda _wt, snapshot=None: ["vibe_quant/x.py"]
    )
    assert _full_modules(["vibe_quant/mod.py"], _one_module_tree(tmp_path)) == ["vibe_quant/mod.py"]


def test_the_cap_counts_a_module_a_conftest_imports(tmp_path: Path) -> None:
    """Teeth: a module tests/unit/conftest.py imports at module level (and so its importers)
    plans the full suite; the grouped check agrees with plan_tests on every module."""
    repo = _one_module_tree(tmp_path, conftest="import vibe_quant.other\n")
    modules = _modules(repo)
    by_plan = [rel for rel in _full_modules(modules, repo) if not rel.endswith("__init__.py")]
    assert by_plan == ["vibe_quant/other.py"]
    assert _conftest_full_modules(repo, modules) == ["vibe_quant/other.py"]
