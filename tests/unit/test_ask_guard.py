"""ask_guard.py (Stop hook): a final message whose "Waiting on you" points at a question it never
asks is sent back once with the reason; a retry (stop_hook_active) always stops."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/hooks/ask_guard.py"


def _run(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_PATH)], input=json.dumps(payload), capture_output=True, text=True
    )


def test_bare_question_numbers_block_the_stop_once(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    entries = [
        {"type": "user", "message": {"content": "status?"}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "**Running:** c8"}]},
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "**Waiting on you:** 9 = A or B?"}]},
        },
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    blocked = _run({"transcript_path": str(transcript)})
    assert blocked.returncode == 0
    out = json.loads(blocked.stdout)
    assert out["decision"] == "block" and "question 9 is referenced but not asked" in out["reason"]
    retry = _run({"transcript_path": str(transcript), "stop_hook_active": True})
    assert retry.returncode == 0 and retry.stdout == ""
    asked = _run(
        {"last_assistant_message": "9. Keep BTC? A: keep. B: sell.\nWaiting on you: 9 = A or B?"}
    )
    assert asked.returncode == 0 and asked.stdout == ""
