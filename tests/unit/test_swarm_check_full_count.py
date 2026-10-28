"""Cap on how many vibe_quant modules make swarm_check's per-round plan fall back to the full
tests/unit (the planner's own plan_tests, one single-file diff per module).

Today only the 28 package `__init__.py` files go full (their import reach is everything). A
module that newly goes full - a dynamic loader the selection does not model, a graph edge it
cannot read - turns a seconds-long gate into the whole suite for every diff touching it, so
growing the count must be a decision, not an accident (vibe-quant-pbjvh).
"""

from __future__ import annotations

import importlib.util
import sys
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

MAX_FULL_MODULES = 28


@pytest.fixture(scope="module", autouse=True)
def _memo_conftest_parse() -> Iterator[None]:
    """Perf shim, no assertion change: one plan parses every conftest's fixtures and deps AND
    every test file's fixture requests, so the 241 single-file plans below re-parse the same
    files 241 times (14 s of the run). Both functions are pure in their arguments, so memoise
    them here and each file is analysed once. Production should cache this itself (handoff
    `open:`: `_conftest_parts`/`_requests` are not in its per-process `_TREE_CACHE`).
    """
    real_parts = swarm_check._conftest_parts
    parts: dict[tuple[object, ...], Any] = {}

    def cached_parts(source: str, module: str, known: Any, bare: Any) -> Any:
        key = (source, module, frozenset(known), tuple(sorted(dict(bare).items())))
        if key not in parts:
            parts[key] = real_parts(source, module, known, bare)
        shared, fixtures = parts[key]
        return set(shared), list(fixtures)  # callers get their own copies

    real_requests = swarm_check._requests
    requests: dict[str, Any] = {}

    def cached_requests(text: str) -> Any:
        if text not in requests:
            requests[text] = real_requests(text)
        found, dynamic = requests[text]
        return set(found), dynamic

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(swarm_check, "_conftest_parts", cached_parts)
        mp.setattr(swarm_check, "_requests", cached_requests)
        yield


def _full_modules(modules: list[str]) -> list[str]:
    """The modules whose single-file diff plans the full suite (the planner caches its
    tree-wide scans per process, so 234 plans do not re-parse the tree)."""
    cfg = swarm_check.Config(worktree=_ROOT, main_root=_ROOT)
    return [
        rel
        for rel in modules
        if swarm_check.plan_tests(cfg, swarm_check.GitState(changed=[rel], untracked=[])).full
    ]


def test_at_most_the_package_inits_plan_the_full_suite() -> None:
    modules = sorted(
        p.relative_to(_ROOT).as_posix()
        for p in (_ROOT / "vibe_quant").rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert len(modules) > 200  # the walk found the tree, not an empty dir
    full = _full_modules(modules)
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
    monkeypatch.setattr(
        swarm_check, "unmodelled_loaders", lambda _wt, snapshot=None: ["vibe_quant/x.py"]
    )
    assert _full_modules(["vibe_quant/data/archive.py"]) == ["vibe_quant/data/archive.py"]
