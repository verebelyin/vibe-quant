import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/hooks/context_guard.py"
_spec = importlib.util.spec_from_file_location("context_guard", _PATH)
assert _spec and _spec.loader
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _transcript(tmp_path: Path, *contexts: int) -> Path:
    path = tmp_path / "t.jsonl"
    lines = [json.dumps({"type": "user", "message": {"content": "hi"}})]
    for ctx in contexts:
        usage = {
            "input_tokens": 10,
            "cache_read_input_tokens": ctx - 110,
            "cache_creation_input_tokens": 100,
        }
        lines.append(json.dumps({"type": "assistant", "message": {"usage": usage}}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_reads_the_newest_usage(tmp_path: Path) -> None:
    assert guard.last_context_tokens(_transcript(tmp_path, 120_000, 345_000)) == 345_000


def test_nudges_once_per_100k_step_from_300k() -> None:
    assert guard.nudge(299_999, 0) == (None, 0)
    msg, step = guard.nudge(345_000, 0)
    assert msg and msg.startswith("CONTEXT 345k tokens") and step == 300_000
    assert guard.nudge(399_999, step) == (None, 300_000)
    msg, step = guard.nudge(905_000, step)
    assert msg and step == 900_000


def test_hook_prints_once_per_session(tmp_path: Path) -> None:
    payload = json.dumps(
        {"transcript_path": str(_transcript(tmp_path, 652_000)), "session_id": f"t-{tmp_path.name}"}
    )
    run = lambda: subprocess.run(  # noqa: E731
        [sys.executable, str(_PATH)], input=payload, capture_output=True, text=True
    )
    first, second = run(), run()
    assert first.returncode == 0 and "CONTEXT 652k tokens" in first.stdout
    assert second.returncode == 0 and second.stdout == ""


def test_hook_is_silent_on_garbage() -> None:
    out = subprocess.run([sys.executable, str(_PATH)], input="{}", capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout == ""
