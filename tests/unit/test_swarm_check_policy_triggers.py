"""POLICY_TRIGGERS in scripts/agents/swarm_check.py match what each policy test really scans.

A policy test runs per round only when the diff touches its globs, so a glob narrower than
the test's scan would let a red policy test through (fail open). Each assertion below reads
the scan root out of the policy test itself, so editing a test's scan without its trigger
(or the reverse) goes red here.
"""

from __future__ import annotations

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


def test_every_policy_test_has_triggers() -> None:
    assert set(TRIGGERS) == set(swarm_check.POLICY_TESTS)


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
    assert picked("scripts/research/rsi_2026.py") == ["test_script_exec_bits", "test_scripts_import_smoke"]
    # its own file and the gate itself always trigger; None (no diff given) = all
    assert picked("tests/unit/test_script_exec_bits.py") == ["test_script_exec_bits"]
    assert len(picked("scripts/agents/swarm_check.py")) == len(swarm_check.POLICY_TESTS)
    assert len(swarm_check.policy_test_files(tmp_path)) == len(swarm_check.POLICY_TESTS)
