#!/usr/bin/env python3
"""PreToolUse hook: block shell patterns that already broke a job, each a prose rule first.

- `pkill -f` / `killall`: a pattern kill hits other agents' processes (10-16 an
  implementer's `pkill -f swarm-check` killed another agent's gate). Kill the pid you
  started (`$!`).
- a `pgrep -f` wait loop: the pattern matches the loop's own command line, so it
  never ends. Wait on a pid: `while kill -0 <pid>`.
- an unquoted heredoc whose body holds a backtick: the shell runs it (twice: once
  ran ImageMagick's `import`). Quote the delimiter: `<<'EOF'`.
Exit 2 blocks the call and shows stderr to the agent. Unparseable input fails open.
"""

import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bd_notes_guard import _segments  # noqa: E402

_UNQUOTED_HEREDOC = re.compile(r"(?<!<)<<-?[ \t]*([A-Za-z_][\w.-]*)")


def _words(seg: str) -> list[str]:
    try:
        words = shlex.split(seg)
    except ValueError:
        return []
    while words and words[0] in {"do", "then", "else", "{", "!", "(", "sudo"}:
        words = words[1:]
    return words


def _has_f_flag(args: list[str]) -> bool:
    return any(w.startswith("-") and not w.startswith("--") and "f" in w for w in args)


def _pattern_kill(words: list[str]) -> bool:
    if not words:
        return False
    if words[0] == "killall":
        return True
    return words[0] == "pkill" and _has_f_flag(words[1:])


def _heredoc_backtick(command: str) -> bool:
    for m in _UNQUOTED_HEREDOC.finditer(command):
        start = command.find("\n", m.end())
        if start < 0:
            continue
        body: list[str] = []
        for line in command[start + 1 :].split("\n"):
            if line.strip() == m.group(1):
                break
            body.append(line)
        if "`" in "\n".join(body):
            return True
    return False


def _pgrep_loop(seg: str) -> bool:
    try:
        words = shlex.split(seg)
    except ValueError:
        return False
    if not words or words[0] not in {"while", "until"}:
        return False
    words = [w for w in words[1:] if w != "!"]
    return bool(words) and words[0] == "pgrep" and _has_f_flag(words[1:])


def reason(command: str) -> str | None:
    segs = _segments(command)  # heredoc bodies and quoted text are not commands
    for seg in segs:
        if _pattern_kill(_words(seg)):
            return "pkill -f / killall hits other agents' processes; kill the pid you started ($!)."
    if any(_pgrep_loop(seg) for seg in segs):
        return "a pgrep -f loop matches its own command line and never ends; wait with `while kill -0 <pid>`."
    if _heredoc_backtick(command):
        return "unquoted heredoc body contains a backtick, which the shell would run; quote the delimiter (<<'EOF')."
    return None


def main() -> int:
    try:
        command = json.load(sys.stdin)["tool_input"]["command"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return 0
    if not isinstance(command, str):
        return 0
    why = reason(command)
    if why:
        print(f"shell_guard: {why}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
