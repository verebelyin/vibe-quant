"""Exit-code contract of ``scripts/agents/exactness_carry.py`` (R6, task T6b).

``main()`` is the PASS/FAIL gate the swarm gate reads: it returns 0 when the NT
engine run equals the no-NT reference and every recorded headline pin holds, and
1 on any drift. The heavy run (real catalog + archive, four coins, 2023-2026) is
monkeypatched here; both codes are asserted from literals, so a mutation of the
return (always 0) fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts/agents/exactness_carry.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exactness_carry", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script module with the heavy run replaced by fakes that pass every pin."""
    module = _load_script()
    spec = module.CarrySpec.from_dates(module.SYMBOLS, module.START, module.END)
    inputs = SimpleNamespace(spec=spec, decisions=tuple(range(module.EXPECTED_DECISIONS)))
    report = SimpleNamespace(
        summary=dict(module.RECORDED), daily=[], weekly=[], legs=[], by_year=[]
    )
    engine = SimpleNamespace(fills=(), payments=())
    reference = SimpleNamespace()
    monkeypatch.setattr(module, "load_carry_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(module, "run_carry_engine", lambda inputs: engine)
    monkeypatch.setattr(module, "simulate_reference", lambda inputs: reference)
    monkeypatch.setattr(module, "carry_metrics", lambda *args, **kwargs: report)
    return module


def test_main_returns_zero_when_the_runs_are_identical(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing ``compare_runs`` (0 mismatches) and matching pins -> exit code 0."""
    monkeypatch.setattr(script, "compare_runs", lambda *args, **kwargs: [])

    assert script.main() == 0

    out = capsys.readouterr().out
    assert "0 mismatches" in out
    assert "carry exactness: IDENTICAL" in out


def test_main_returns_one_on_any_drift(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One mismatch from ``compare_runs`` -> exit code 1 (the gate is fail-closed)."""
    monkeypatch.setattr(
        script, "compare_runs", lambda *args, **kwargs: ["weekly_sharpe: 0.0 != 1.0"]
    )

    assert script.main() == 1

    out = capsys.readouterr().out
    assert "1 mismatches" in out
    assert "carry exactness: DRIFT" in out
