"""The verifier agent runs the integration gate (`swarm-check.sh --full`) on a merged tree."""

from __future__ import annotations

from pathlib import Path

_VERIFIER = Path(__file__).resolve().parents[2] / ".claude" / "agents" / "verifier.md"


def test_verifier_integration_runs_swarm_check_full() -> None:
    lines = [ln for ln in _VERIFIER.read_text().splitlines() if "swarm-check.sh" in ln]
    assert any("--full" in ln and "integrat" in ln.lower() for ln in lines), lines
