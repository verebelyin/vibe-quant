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
    monkeypatch.setattr(swarm_check, "unmodelled_loaders", lambda _wt: ["vibe_quant/x.py"])
    assert _full_modules(["vibe_quant/data/archive.py"]) == ["vibe_quant/data/archive.py"]
