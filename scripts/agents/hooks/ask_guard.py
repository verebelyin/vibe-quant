#!/usr/bin/env python3
"""Stop hook: a final message whose "Waiting on you" points at a question it never asks goes back.

Nov 2026: the chief's status messages ended "Waiting on you: 9 = A or B? 10 = A or B?" while the
questions sat in an earlier message or a questions file; the user asked 8 times to have them
restated. `ask_lint.py --status` finds each referenced number with no question text in the same
message; this hook runs it on the turn's final text and, on a problem, blocks the stop with the
lint lines as the reason so the model restates the questions.

Silent (exit 0, no output) on any error, when `stop_hook_active` is set (one retry only), and when
the message has no "Waiting on you". Stdlib only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TAIL_BYTES = 1024 * 1024
_LINT = Path(__file__).resolve().parents[1] / "ask_lint.py"


def final_text(transcript: Path) -> str:
    """Text blocks of the assistant entries after the last user/tool-result entry."""
    with transcript.open("rb") as fh:
        fh.seek(max(0, transcript.stat().st_size - TAIL_BYTES))
        lines = fh.read().decode("utf-8", errors="replace").splitlines()
    parts: list[str] = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        kind = entry.get("type")
        if kind == "user":
            break
        if kind != "assistant":
            continue
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            parts.extend(reversed(texts))
    return "\n".join(reversed(parts))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if payload.get("stop_hook_active"):
            return 0
        text = payload.get("last_assistant_message") or final_text(Path(payload["transcript_path"]))
        spec = importlib.util.spec_from_file_location("ask_lint", _LINT)
        assert spec and spec.loader
        lint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lint)
        problems = lint.status_problems(text)
    except Exception:  # noqa: BLE001 - a hook must never break the session
        return 0
    if problems:
        reason = "ask_guard: " + "; ".join(problems) + ". Repeat each question in full above the summary line."
        print(json.dumps({"decision": "block", "reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
