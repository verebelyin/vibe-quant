"""The gate golden fixture and runner (bead vibe-quant-rabmo.6 T5a/T5b).

Section (a) must reach every gate stage with a fixed verdict: its digest is the
contract every rabmo.6 refactor step has to reproduce. The runner's
``--against`` must be exact, including after one key is mutated.
"""

from __future__ import annotations

import json
import os
import subprocess
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


# ---- the default --against: <main checkout>/data/swarm/_goldens/gate_golden.json ----
_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "golden",
    "GIT_AUTHOR_EMAIL": "golden@test",
    "GIT_COMMITTER_NAME": "golden",
    "GIT_COMMITTER_EMAIL": "golden@test",
}


def _git_env() -> dict[str, str]:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV_OVERRIDES)
    return env


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
        env=_git_env(),
    )


def _git_pair(tmp_path: Path) -> tuple[Path, Path]:
    """(main checkout, linked worktree of it): the worktree's git-common-dir is main's .git."""
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(main)], check=True, env=_git_env())
    (main / "keep.txt").write_text("x\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "feature", str(wt))
    return main, wt


def test_default_golden_is_under_the_main_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main, wt = _git_pair(tmp_path)
    monkeypatch.setattr(gate_golden, "REPO_ROOT", wt)
    expected = main / "data" / "swarm" / "_goldens" / "gate_golden.json"
    assert gate_golden.default_golden() == expected  # from the linked worktree
    assert gate_golden.default_golden(main) == expected


def test_bare_against_compares_with_the_main_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main, wt = _git_pair(tmp_path)
    golden = main / "data" / "swarm" / "_goldens" / "gate_golden.json"
    golden.parent.mkdir(parents=True)
    golden.write_text(json.dumps({"a": 1}, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(gate_golden, "REPO_ROOT", wt)
    monkeypatch.setattr(gate_golden, "build", lambda db=None: {"a": 1})

    assert gate_golden.main(["--against"]) == 0
    assert "IDENTICAL" in capsys.readouterr().out


def test_explicit_against_path_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main, wt = _git_pair(tmp_path)
    golden = main / "data" / "swarm" / "_goldens" / "gate_golden.json"
    golden.parent.mkdir(parents=True)
    golden.write_text(json.dumps({"a": 1}, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(gate_golden, "REPO_ROOT", wt)
    monkeypatch.setattr(gate_golden, "build", lambda db=None: {"a": 1})
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"a": 2}, indent=2, sort_keys=True) + "\n")

    assert gate_golden.main(["--against", str(other)]) == 1
    assert "IDENTICAL" not in capsys.readouterr().out


def test_bare_against_without_the_main_golden_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main, wt = _git_pair(tmp_path)
    monkeypatch.setattr(gate_golden, "REPO_ROOT", wt)

    def no_build(db: Path | None = None) -> dict[str, object]:
        raise AssertionError("the digest was built before checking the golden exists")

    monkeypatch.setattr(gate_golden, "build", no_build)
    assert gate_golden.main(["--against"]) == 2
    assert str(main / "data" / "swarm" / "_goldens" / "gate_golden.json") in (
        capsys.readouterr().err
    )
