"""POLICY_TRIGGERS in scripts/agents/swarm_check.py match what each policy test really scans.

A policy test runs per round only when the diff touches its globs, so a glob narrower than
the test's scan would let a red policy test through (fail open). Each assertion below reads
the scan root out of the policy test itself, so editing a test's scan without its trigger
(or the reverse) goes red here.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import sys
from pathlib import Path

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

TRIGGERS = swarm_check.POLICY_TRIGGERS


def _source(rel: str) -> str:
    return (_ROOT / rel).read_text()


def test_every_policy_test_has_triggers() -> None:
    assert set(TRIGGERS) == set(swarm_check.POLICY_TESTS)


def test_exception_hygiene_triggers_are_its_scan_packages() -> None:
    mod = importlib.import_module("tests.unit.test_exception_hygiene")
    assert set(TRIGGERS["tests/unit/test_exception_hygiene.py"]) == {
        f"vibe_quant/{pkg}/*.py" for pkg in mod.SCAN_PACKAGES
    }
    assert 'directory.rglob("*.py")' in _source("tests/unit/test_exception_hygiene.py")


def test_steering_triggers_cover_every_file_it_reads_and_none_is_dead() -> None:
    mod = importlib.import_module("tests.unit.test_steering_budget")
    read = {p.relative_to(_ROOT).as_posix() for p in mod._steering_files()} | set(mod.BUDGETS)
    globs = TRIGGERS["tests/unit/test_steering_budget.py"]
    assert [r for r in read if not any(fnmatch.fnmatch(r, g) for g in globs)] == []
    assert [g for g in globs if not any(fnmatch.fnmatch(r, g) for r in read)] == []


def test_conn_locking_triggers_are_its_scanned_dirs() -> None:
    mod = importlib.import_module("tests.unit.test_api_routers_conn_locking")
    assert set(TRIGGERS["tests/unit/test_api_routers_conn_locking.py"]) == {
        f"vibe_quant/{d.relative_to(mod._PKG).as_posix()}/*.py" for d in mod._SCANNED
    }


def test_whole_package_scans_trigger_on_all_of_vibe_quant() -> None:
    # both walk rglob("*.py") from the vibe_quant package root
    assert "root = Path(vibe_quant.__file__).parent" in _source(
        "tests/unit/test_discovery_liquidation.py"
    )
    assert TRIGGERS["tests/unit/test_discovery_liquidation.py"] == ("vibe_quant/*.py",)
    assert TRIGGERS["tests/unit/test_swarm_check_loaders.py"] == ("vibe_quant/*.py",)


def test_exec_bits_triggers_on_anything_under_scripts() -> None:
    assert '"ls-files", "-s", "scripts"' in _source("tests/unit/test_script_exec_bits.py")
    assert TRIGGERS["tests/unit/test_script_exec_bits.py"] == ("scripts/*",)


def test_selection_follows_the_triggers(tmp_path: Path) -> None:
    for rel in swarm_check.POLICY_TESTS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")

    def picked(*changed: str) -> list[str]:
        return [Path(p).stem for p in swarm_check.policy_test_files(tmp_path, list(changed))]

    assert picked("docs/claude/beads.md") == ["test_steering_budget"]
    assert picked("scripts/agents/bus") == ["test_script_exec_bits"]
    assert picked("frontend/src/App.tsx") == []
    assert picked("vibe_quant/api/routers/data.py") == [
        "test_api_routers_conn_locking",
        "test_discovery_liquidation",
        "test_swarm_check_loaders",
    ]
    # its own file and the gate itself always trigger; None (no diff given) = all
    assert picked("tests/unit/test_script_exec_bits.py") == ["test_script_exec_bits"]
    assert len(picked("scripts/agents/swarm_check.py")) == len(swarm_check.POLICY_TESTS)
    assert len(swarm_check.policy_test_files(tmp_path)) == len(swarm_check.POLICY_TESTS)
