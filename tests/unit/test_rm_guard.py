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
