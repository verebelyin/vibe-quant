import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/hooks/shell_guard.py"
_spec = importlib.util.spec_from_file_location("shell_guard", _PATH)
assert _spec and _spec.loader
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "command",
    [
        "pkill -f swarm-check",
        "cd /x && pkill -9f pytest",
        "killall python3",
        "while pgrep -f 'discovery run' >/dev/null; do sleep 5; done",
        "until ! pgrep -lf foo; do sleep 1; done",
        "cat > /tmp/b.md <<EOF\nrun `scripts/x.sh` first\nEOF",
        "cat <<-DONE\n\tsee `import`\n\tDONE",
    ],
)
def test_known_footguns_are_blocked(command: str) -> None:
    assert guard.reason(command)


@pytest.mark.parametrize(
    "command",
    [
        "kill 1234",
        "pkill -x vite",
        "pgrep -f uvicorn",
        "while kill -0 $pid; do sleep 5; done",
        "cat > /tmp/b.md <<'EOF'\nrun `scripts/x.sh` first\nEOF",
        'cat <<"EOF"\n`x`\nEOF',
        "cat <<EOF\nplain $HOME text\nEOF",
        "echo 'pkill -f is banned'",
        "rg -n 'pkill -f' docs",
        "python3 - <<<'print(1)'",
        "git commit -F - <<'EOF'\nblocks while/until pgrep -f loops and pkill -f\nEOF",
        'git commit -m "while pgrep -f x; do sleep 1; done"',
    ],
)
def test_safe_commands_pass(command: str) -> None:
    assert guard.reason(command) is None


def test_hook_blocks_with_exit_2_and_reason_on_stderr() -> None:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "pkill -f swarm-check"}})
    out = subprocess.run(
        [sys.executable, str(_PATH)], input=payload, capture_output=True, text=True
    )
    assert out.returncode == 2
    assert "kill the pid you started" in out.stderr


def test_hook_fails_open_on_garbage() -> None:
    out = subprocess.run(
        [sys.executable, str(_PATH)], input="not json", capture_output=True, text=True
    )
    assert out.returncode == 0
