import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/hooks/bd_notes_guard.py"
_spec = importlib.util.spec_from_file_location("bd_notes_guard", _PATH)
assert _spec and _spec.loader
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _run(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_PATH)], input=stdin, capture_output=True, text=True)


@pytest.mark.parametrize(
    "command",
    [
        'bd update vibe-quant-x --notes "a"',
        'cd /repo && bd update x --status open --notes="a" >/dev/null',
        "bd update x \\\n  --notes a",
        "printf x | bd update x --notes a",
        "env X=1 bd update x --notes a",
        "X=1 Y=2 bd update x --notes a",
        "env -i X=1 bd update x --notes a",
        "echo hi; bd update x --notes a",
        "echo '; ' && bd update x --notes a",
        "cat <<'EOF'\nhi\nEOF\nbd update x --notes a",
        "(bd update x --notes a)",
    ],
)
def test_replacing_notes_is_blocked(command: str) -> None:
    assert guard.blocks(command)


@pytest.mark.parametrize(
    "command",
    [
        'bd update x --append-notes "a"',
        'bd create --title t --notes "first notes"',
        'bd update x --claim && echo "--notes"',
        "rg -- --notes docs/claude/beads.md",
        "git commit -F - <<'EOF'\nhook blocks `bd update x --notes`\nEOF",
        "cat <<'EOF'\nbd update x --notes a\nEOF",
        "cat <<-EOF\n\tbd update x --notes a\n\tEOF",
        "printf '%s' '; bd update x --notes a'",
        'echo "x && bd update x --notes a"',
        "echo 'a; bd update x --notes b; c'",
        "bd update x --append-notes 'avoid --notes replacement'",
        'bd update x --append-notes="use --notes"',
        "bd update x --notes-file f",
        "bd update x --status open 2>&1 | tail -1",
        "bd update 'unbalanced --notes",
    ],
)
def test_other_commands_pass(command: str) -> None:
    assert not guard.blocks(command)


def test_hook_exit_code_blocks_via_stdin() -> None:
    run = _run(json.dumps({"tool_input": {"command": 'bd update x --notes "a"'}}))
    assert run.returncode == 2
    assert "--append-notes" in run.stderr


@pytest.mark.parametrize(
    ("command", "code"),
    [
        ("printf x | bd update x --notes a", 2),
        ("cat <<'EOF'\nbd update x --notes a\nEOF", 0),
        ("printf '%s' '; bd update x --notes a'", 0),
        ("bd update x --append-notes 'avoid --notes replacement'", 0),
        ("env X=1 bd update x --notes a", 2),
        ("cat <<< EOF\nbd update x --notes a", 2),
        ("env -u X bd update x --notes a", 2),
        ("env -C /tmp bd update x --notes a", 2),
        ('env -S "bd update x --notes a"', 0),  # -S operand not inspected (documented)
        ("env --unset X bd update x --notes a", 2),
        ("env -u X bd update x --append-notes a", 0),
    ],
)
def test_hook_exit_codes_for_review_repros(command: str, code: int) -> None:
    assert _run(json.dumps({"tool_input": {"command": command}})).returncode == code


@pytest.mark.parametrize(
    "payload",
    [
        "null",
        "[]",
        "not json",
        "{}",
        '{"tool_input": null}',
        '{"tool_input": {"command": 1}}',
        '{"tool_input": {"command": []}}',
        '{"tool_input": {"command": null}}',
    ],
)
def test_malformed_input_fails_open(payload: str) -> None:
    assert _run(payload).returncode == 0


def test_invalid_utf8_stdin_fails_open() -> None:
    run = subprocess.run([sys.executable, str(_PATH)], input=b"\xff\xfe{", capture_output=True)
    assert run.returncode == 0
