import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/hooks/rm_guard.py"
_spec = importlib.util.spec_from_file_location("rm_guard", _PATH)
assert _spec and _spec.loader
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

CWD = "/Users/someone/repo"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/x",
        "D=/tmp/rev-x; rm -rf $D && mkdir -p $D",
        "export D=/tmp/q; rm -rf ${D}/w",
        "D=/tmp/x D2=$D/y; rm -r $D2",
        'D=$(mktemp -d); rm -rf "$D"',
        "cd /tmp && rm -rf scratch",
    ],
)
def test_temp_only_rm_is_allowed(command: str) -> None:
    assert guard.decide(command, CWD)[0] == "allow"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        'rm -rf "$HOME/x"',
        "D=$HOME/x; rm -rf $D",
        "D=/tmp/x; D=$UNSET_VAR_FOR_TEST/y; rm -rf $D",
        "for d in a b; do rm -rf $d; done",
        "H=$(git rev-parse HEAD); rm -rf /tmp/x",
        "rm -rf /tmp",
    ],
)
def test_rm_outside_temp_or_unknown_asks(command: str) -> None:
    assert guard.decide(command, CWD)[0] == "ask"


def _hook(command: str, **extra: str) -> dict[str, dict[str, str]] | None:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": CWD, **extra}
    out = subprocess.run(
        [sys.executable, str(_PATH)], input=json.dumps(payload), capture_output=True, text=True
    )
    assert out.returncode == 0
    return json.loads(out.stdout) if out.stdout.strip() else None


def test_subagent_is_denied_with_a_rewrite_hint_instead_of_a_prompt_it_cannot_answer() -> None:
    res = _hook("rm -rf build", agent_id="a123", agent_type="implementer")
    assert res
    out = res["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "D=/tmp/x; rm -rf $D" in out["permissionDecisionReason"]


_JOB = guard._REPO + "/data/swarm"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (f"rm -f {_JOB}/20261121-bugs6/{{red,gate}}.log", "allow"),
        (f"rm -rf {_JOB}/20261121-bugs6/probes/x", "allow"),
        (f"rm -f {_JOB}/_goldens/gate_golden.json", "ask"),
        (f"rm -rf {_JOB}/20261121-bugs6", "ask"),
        (f"rm -rf {_JOB}/20261121-bugs6/../../state/vibe_quant.db", "ask"),
        ("rm -rf /tmp/{a,../../etc}", "ask"),
        (f"rm -rf {guard._REPO}/data/state/x", "ask"),
    ],
)
def test_job_scratch_is_free_but_goldens_state_and_brace_escapes_ask(
    command: str, expected: str
) -> None:
    assert guard.decide(command, CWD)[0] == expected


def test_t3_delegated_child_is_denied_not_asked() -> None:
    payload = {"tool_name": "Bash", "tool_input": {"command": "rm -rf build"}, "cwd": CWD}
    env = {**os.environ, "T3CODE_THREAD_ID": "thread:delegated-task:x"}
    out = subprocess.run(
        [sys.executable, str(_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(out.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 - <<'PY'\ns = 'rm -rf /tmp `x`'\nPY", None),
        ("cat > /tmp/n.md <<EOF\nrun rm -rf build\nEOF\nls", None),
        ("python3 - <<'PY'\nx = 1\nPY\nrm -rf build", "ask"),
        ("bash <<'EOF'\nrm -rf build\nEOF", "ask"),
        ("sh -s <<-EOF\n\trm -rf build\n\tEOF", "ask"),
    ],
)
def test_heredoc_body_is_data_unless_a_shell_runs_it(command: str, expected: str | None) -> None:
    assert guard.decide(command, CWD)[0] == expected


def test_untracked_file_in_a_linked_worktree_is_free_but_tracked_files_and_main_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewers write probe tests into a worktree's tests/unit (conftest needed) and could not
    delete them: 39 denials 2026-11-06..11-27. An untracked, non-ignored file in a LINKED worktree
    is disposable; a tracked file, a directory and anything in the main checkout still ask."""
    main, wt = tmp_path / "main", tmp_path / "wt"
    git = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q", str(main))
    (main / "kept.py").write_text("x = 1\n")
    git("-C", str(main), "add", "kept.py")
    git("-C", str(main), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    git("-C", str(main), "worktree", "add", "-q", str(wt))
    (wt / "test_zz_probe.py").write_text("")
    (main / "probe.py").write_text("")
    monkeypatch.setattr(guard, "_REPO", str(main.resolve()))
    monkeypatch.setattr(
        guard, "temp_roots", lambda: ["/nonexistent-temp-root"]
    )  # tmp_path is in $TMPDIR
    assert guard.decide("rm -f test_zz_probe.py", str(wt))[0] == "allow"
    assert guard.decide("rm kept.py", str(wt))[0] == "ask"
    assert guard.decide("rm -rf .", str(wt))[0] == "ask"
    assert guard.decide("rm probe.py", str(main))[0] == "ask"
