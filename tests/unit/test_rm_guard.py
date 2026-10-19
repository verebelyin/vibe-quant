import importlib.util
import json
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


def test_temp_rm_mixed_with_other_commands_defers_to_normal_flow() -> None:
    cmd = "S=/tmp/rev; rm -rf $S; mkdir -p $S && git archive HEAD | tar -x -C $S"
    assert guard.decide(cmd, CWD) == (None, "")


def _hook(command: str, **extra: str) -> dict[str, dict[str, str]] | None:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": CWD, **extra}
    out = subprocess.run(
        [sys.executable, str(_PATH)], input=json.dumps(payload), capture_output=True, text=True
    )
    assert out.returncode == 0
    return json.loads(out.stdout) if out.stdout.strip() else None


def test_main_session_is_asked() -> None:
    res = _hook("rm -rf build")
    assert res and res["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_subagent_is_denied_with_a_rewrite_hint_instead_of_a_prompt_it_cannot_answer() -> None:
    res = _hook("rm -rf build", agent_id="a123", agent_type="implementer")
    assert res
    out = res["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "D=/tmp/x; rm -rf $D" in out["permissionDecisionReason"]


def test_subagent_temp_rm_still_allowed() -> None:
    res = _hook("D=/tmp/x; rm -rf $D", agent_id="a123")
    assert res and res["hookSpecificOutput"]["permissionDecision"] == "allow"
