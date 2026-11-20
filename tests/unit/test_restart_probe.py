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


@needs_git
def test_cli_is_identical_for_the_same_rev_twice(capsys: pytest.CaptureFixture[str]) -> None:
    rc = restart_probe.main(["--repo", str(_ROOT), "--rev", "HEAD", "--rev", "HEAD"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[-1] == "IDENTICAL"
    assert "S1 IDENTICAL" in out and "LEAKS IDENTICAL" in out


@needs_git
@pytest.mark.slow(reason="replays the restart probe on two real git revs (~12 s)")
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
