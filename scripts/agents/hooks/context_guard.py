#!/usr/bin/env python3
"""UserPromptSubmit hook: nudge a fresh session once the context passes 300k tokens.

Chief sessions ran several jobs in one context: 309k -> 905k tokens per turn and 353M
input tokens in one session (05ba, 10-16/17), 652k in another (603f), with the account
session limit hit twice. Every turn re-reads the whole context. Job state lives in
data/swarm/<job>/ (brief, ledger, handoffs) and beads, so a new session loses nothing.

Reads the last assistant `usage` from the tail of `transcript_path` and prints one line
(added to the model's context) the first time each 100k step from 300k is crossed per
session. Silent below the threshold or on any error: a hook must never block a prompt.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

THRESHOLD = 300_000
STEP = 100_000
TAIL_BYTES = 512 * 1024


def last_context_tokens(transcript: Path) -> int | None:
    """Prompt-side tokens (input + cache read + cache write) of the newest assistant turn."""
    with transcript.open("rb") as fh:
        fh.seek(max(0, transcript.stat().st_size - TAIL_BYTES))
        lines = fh.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if '"usage"' not in line:
            continue
        try:
            usage = json.loads(line)["message"]["usage"]
        except (ValueError, KeyError, TypeError):
            continue
        return int(
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
    return None


def nudge(tokens: int, last_step: int) -> tuple[str | None, int]:
    """(message or None, new last step) for a context of ``tokens``."""
    if tokens < THRESHOLD:
        return None, last_step
    step = THRESHOLD + (tokens - THRESHOLD) // STEP * STEP
    if step <= last_step:
        return None, last_step
    msg = (
        f"CONTEXT {tokens // 1000}k tokens: every turn re-reads all of it. If the current job "
        "has landed (or is between waves), write its summary and ask the user to start a fresh "
        "session; job state lives in data/swarm/<job>/ and beads."
    )
    return msg, step


def main() -> int:
    try:
        data = json.load(sys.stdin)
        transcript = Path(data["transcript_path"])
        session = str(data.get("session_id") or transcript.stem)
        tokens = last_context_tokens(transcript)
        if tokens is None:
            return 0
        state = Path(tempfile.gettempdir()) / f"vq-context-guard-{session}.json"
        last = int(json.loads(state.read_text())["step"]) if state.exists() else 0
        msg, step = nudge(tokens, last)
        if msg:
            state.write_text(json.dumps({"step": step}))
            print(msg)
    except (OSError, ValueError, KeyError, TypeError):
        return 0
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(main())
