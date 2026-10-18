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


@pytest.mark.parametrize(
    "command",
    [
        'bd update vibe-quant-x --notes "a"',
        'cd /repo && bd update x --status open --notes="a" >/dev/null',
        "bd update x \\\n  --notes a",
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
    ],
)
def test_other_commands_pass(command: str) -> None:
    assert not guard.blocks(command)


def test_hook_exit_code_blocks_via_stdin() -> None:
    payload = json.dumps({"tool_input": {"command": 'bd update x --notes "a"'}})
    run = subprocess.run([sys.executable, str(_PATH)], input=payload, capture_output=True, text=True)
    assert run.returncode == 2
    assert "--append-notes" in run.stderr
