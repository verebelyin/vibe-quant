"""scripts/agents/restart_probe.py: the paper-node restart chain compared across git revs."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

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


@pytest.fixture(scope="module", autouse=True)
def _one_run_per_rev() -> Iterator[None]:
    """`run_rev` is a pure function of (repo, rev) — a git archive plus one child that drives the
    real restart chain — and each test only reads the record, so do that work once per rev."""
    cache: dict[str, dict[str, dict[str, object]]] = {}
    real = restart_probe.run_rev

    def cached(repo: Path, rev: str) -> dict[str, dict[str, object]]:
        if rev not in cache:
            cache[rev] = real(repo, rev)
        return cache[rev]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(restart_probe, "run_rev", cached)
        yield


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


def test_child_interpreter_is_the_repo_venv_when_it_exists(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    assert restart_probe.child_python(tmp_path) == str(venv_python)
    assert restart_probe.child_python(tmp_path / "nowhere") == sys.executable


@needs_git
def test_script_runs_through_its_shebang(tmp_path: Path) -> None:
    """Invoked as `scripts/agents/restart_probe.py` (python3 on PATH has no pytest)."""
    if not (_ROOT / ".venv" / "bin" / "python").exists():
        pytest.skip("no .venv in this checkout")
    proc = subprocess.run(
        [str(_SCRIPT), "--repo", str(_ROOT), "--rev", "HEAD"],  # one rev: this is the entry test
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "IDENTICAL"
