"""The gate golden fixture and runner (bead vibe-quant-rabmo.6 T5a/T5b).

Section (a) must reach every gate stage with a fixed verdict: its digest is the
contract every rabmo.6 refactor step has to reproduce. The runner's
``--against`` must be exact, including after one key is mutated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest  # noqa: TC002  # annotations only

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"
sys.path.insert(0, str(_AGENTS))

import gate_golden  # noqa: E402
from gate_golden_fixture import run_single, stage_counts, tag  # noqa: E402

EXPECTED_STAGES = {"cross_window": 1, "guardrails": 1, "holdout": 1, "wfa": 1}


def test_fixture_covers_every_stage() -> None:
    result = run_single()
    assert [tag(chromosome) for chromosome, _ in result.top_strategies] == [1]
    assert stage_counts(result) == EXPECTED_STAGES


def test_against_identical_on_same_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "gate_golden.json"

    assert gate_golden.main(["--out", str(base)]) == 0
    assert gate_golden.main(["--against", str(base)]) == 0
    assert "IDENTICAL" in capsys.readouterr().out

    stored = json.loads(base.read_text())
    assert stored["a"]["dsr_trials"]["gate_n"] == 55
    assert stored["a"]["guardrail_rejections"][0]["reasons"] == [
        "DSR not significant: p=0.9705 >= 0.05 (pooled daily SR=0.500 ann, T=181, N=55)"
    ]
    if "skipped" not in stored["c"]:
        assert stored["c"]["877"][0]["passed_dsr"] is False
        assert len(stored["d"]) == 11

    stored["a"]["dsr_trials"]["gate_n"] += 1
    base.write_text(json.dumps(stored))

    assert gate_golden.main(["--against", str(base)]) == 1
