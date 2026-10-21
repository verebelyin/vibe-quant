"""The gate golden fixture and runner (bead vibe-quant-rabmo.6 T5a/T5b).

Section (a) must reach every gate stage with a fixed verdict: its digest is the
contract every rabmo.6 refactor step has to reproduce. The runner's
``--against`` must be exact, including after one key is mutated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"
sys.path.insert(0, str(_AGENTS))

from gate_golden_fixture import run_single, stage_counts, tag  # noqa: E402

EXPECTED_STAGES = {"cross_window": 1, "guardrails": 1, "holdout": 1, "wfa": 1}


def test_fixture_covers_every_stage() -> None:
    result = run_single()
    assert [tag(chromosome) for chromosome, _ in result.top_strategies] == [1]
    assert stage_counts(result) == EXPECTED_STAGES
