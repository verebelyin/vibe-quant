"""``swarm-check --full`` runs tests/unit only, so a test added under tests/integration/ never runs
after it lands (job 20261108-tv-ports: two makers put gate-judged runner tests there; one had no
junit timing, the other would have silently stopped running at land). New tests go in tests/unit.
"""

from __future__ import annotations

from pathlib import Path

_INTEGRATION = Path(__file__).resolve().parents[1] / "integration"
#: The pre-existing files; this list only ever shrinks (move a file to tests/unit, then drop it).
_KNOWN = {
    "__init__.py",
    "test_ma_cross_end_to_end.py",
    "test_plugin_contract.py",
    "test_plugin_end_to_end.py",
}


def test_no_new_test_files_under_tests_integration() -> None:
    present = {p.name for p in _INTEGRATION.glob("*.py")}
    assert present - _KNOWN == set(), (
        f"put new tests in tests/unit/, not tests/integration/: {sorted(present - _KNOWN)}"
    )
