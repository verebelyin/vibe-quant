"""Cap on how many vibe_quant modules make swarm_check's per-round plan fall back to the full
tests/unit (the planner's own plan_tests, one single-file diff per module).

Today only the 28 package `__init__.py` files go full (their import reach is everything). A
module that newly goes full - a dynamic loader the selection does not model, a graph edge it
cannot read - turns a seconds-long gate into the whole suite for every diff touching it, so
growing the count must be a decision, not an accident (vibe-quant-pbjvh).
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

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


def _full_modules(monkeypatch: pytest.MonkeyPatch, modules: list[str]) -> list[str]:
    """The modules whose single-file diff plans the full suite. The tree-wide parse work is
    memoised (the tree does not change between plans) so 234 plans take seconds, not minutes."""
    graphs: dict[Path, dict[str, set[str]]] = {}
    build = swarm_check.build_import_graph

    def cached_graph(worktree: Path, extra: object = ()) -> dict[str, set[str]]:
        if worktree not in graphs:
            graphs[worktree] = build(worktree, extra)
        return graphs[worktree]

    plugins_checked: list[bool] = []
    check = swarm_check.check_pytest_plugins

    def check_once(worktree: Path) -> None:
        if not plugins_checked:
            check(worktree)
            plugins_checked.append(True)

    reads: dict[tuple[Path, str | None, str | None], str] = {}
    read_text = Path.read_text

    def cached_read(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if _ROOT not in self.parents:
            return read_text(self, encoding=encoding, errors=errors)
        key = (self, encoding, errors)
        if key not in reads:
            reads[key] = read_text(self, encoding=encoding, errors=errors)
        return reads[key]

    monkeypatch.setattr(Path, "read_text", cached_read)
    monkeypatch.setattr(swarm_check, "build_import_graph", cached_graph)
    monkeypatch.setattr(swarm_check, "check_pytest_plugins", check_once)
    monkeypatch.setattr(swarm_check, "unmodelled_loaders", functools.cache(swarm_check.unmodelled_loaders))
    parts = swarm_check._conftest_parts
    cached_parts: dict[tuple[str, str], object] = {}

    def conftest_parts(source: str, module: str, known: object, bare: object) -> object:
        if (source, module) not in cached_parts:
            cached_parts[source, module] = parts(source, module, known, bare)
        return cached_parts[source, module]

    monkeypatch.setattr(swarm_check, "_conftest_parts", conftest_parts)
    monkeypatch.setattr(swarm_check, "_requests", functools.cache(swarm_check._requests))
    cfg = swarm_check.Config(worktree=_ROOT, main_root=_ROOT)
    return [
        rel
        for rel in modules
        if swarm_check.plan_tests(cfg, swarm_check.GitState(changed=[rel], untracked=[])).full
    ]


def test_at_most_the_package_inits_plan_the_full_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = sorted(
        p.relative_to(_ROOT).as_posix()
        for p in (_ROOT / "vibe_quant").rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert len(modules) > 200  # the walk found the tree, not an empty dir
    full = _full_modules(monkeypatch, modules)
    assert len(full) <= MAX_FULL_MODULES, (
        f"{len(full)} modules plan the full tests/unit (cap {MAX_FULL_MODULES}); "
        f"non-__init__ ones: {[m for m in full if not m.endswith('__init__.py')]}"
    )
    assert [m for m in full if not m.endswith("__init__.py")] == []


def test_the_cap_counts_a_module_the_planner_sends_to_the_full_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teeth: with the planner's loader check reporting an unmodelled loader, a non-init
    module is counted (the cap test above would then fail)."""
    monkeypatch.setattr(swarm_check, "unmodelled_loaders", lambda _wt: ["vibe_quant/x.py"])
    assert _full_modules(monkeypatch, ["vibe_quant/data/archive.py"]) == ["vibe_quant/data/archive.py"]
