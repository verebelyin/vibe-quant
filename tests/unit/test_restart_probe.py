"""scripts/agents/restart_probe.py: the paper-node restart chain compared across git revs."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "agents" / "restart_probe.py"
_spec = importlib.util.spec_from_file_location("restart_probe", _SCRIPT)
assert _spec is not None and _spec.loader is not None
restart_probe = importlib.util.module_from_spec(_spec)
sys.modules["restart_probe"] = restart_probe
_spec.loader.exec_module(restart_probe)

# the revision before the restart-chain fixes: S1 loses the manual halt + risk there
_KNOWN_BAD = "915fa5ac"


# the CLI archives revs of the checkout it lives in: a non-git copy (mutate.py's) has none
needs_git = pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")


def _record(**override: object) -> dict[str, dict[str, object]]:
    base: dict[str, object] = {"state": "error", "halt_reason": "manual"}
    rec = {step: dict(base) for step in restart_probe.STEPS}
    for step, value in override.items():
        rec[step] = {**base, "halt_reason": value}
    return rec


def test_compare_names_the_steps_that_differ_from_the_first_rev() -> None:
    results = {"a": _record(), "b": _record(S2=None, S4=None)}
    assert restart_probe.compare(results) == ["S2", "S4"]
    assert restart_probe.compare({"a": _record(), "b": _record()}) == []


@needs_git
def test_cli_is_identical_for_the_same_rev_twice(capsys: pytest.CaptureFixture[str]) -> None:
    rc = restart_probe.main(["--repo", str(_ROOT), "--rev", "HEAD", "--rev", "HEAD"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[-1] == "IDENTICAL"
    assert "S1 IDENTICAL" in out and "LEAKS IDENTICAL" in out


@needs_git
def test_cli_differs_at_step_1_for_the_known_bad_rev(capsys: pytest.CaptureFixture[str]) -> None:
    if subprocess.run(
        ["git", "-C", str(_ROOT), "cat-file", "-e", f"{_KNOWN_BAD}^{{commit}}"], capture_output=True
    ).returncode:
        pytest.skip(f"{_KNOWN_BAD} not in this clone")
    rc = restart_probe.main(["--repo", str(_ROOT), "--rev", "HEAD", "--rev", _KNOWN_BAD])
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    assert out[-1] == "DIFFERS (first at S1)"
    assert "S1 DIFFERS" in out


@needs_git
def test_cli_exits_2_when_a_rev_does_not_exist(capsys: pytest.CaptureFixture[str]) -> None:
    rc = restart_probe.main(["--repo", str(_ROOT), "--rev", "no-such-rev-xyz"])
    assert rc == 2
    assert "no-such-rev-xyz" in capsys.readouterr().err
