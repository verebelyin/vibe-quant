#!/usr/bin/env python3
"""PreToolUse hook: block shell patterns that already broke a job, each a prose rule first.

- `pkill -f` / `killall`: a pattern kill hits other agents' processes (10-16 an
  implementer's `pkill -f swarm-check` killed another agent's gate). Kill the pid you
  started (`$!`).
- a `pgrep -f` wait loop: the pattern matches the loop's own command line, so it
  never ends. Wait on a pid: `while kill -0 <pid>`.
- an unquoted heredoc whose body holds a backtick: the shell runs it (twice: once
  ran ImageMagick's `import`). Quote the delimiter: `<<'EOF'`.
- GNU `sed -i 's…'`: macOS (BSD) sed takes the word after `-i` as the backup suffix
  and fails (97 times 10-15..11-05). Use `sed -i '' …` or the Edit tool.
- a command stored in a variable (`AB="agent-browser --session x"; $AB open …`): zsh does not
  word-split, so it runs a program named "agent-browser --session x" (exit 127, 3 times
  2026-11-06..11-27). Write the command out, or use a function/array.
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
_HEREDOC_BODY = re.compile(
    r"<<-?[ \t]*(['\"]?)([A-Za-z_][\w.-]*)\1[^\n]*\n.*?^[ \t]*\2[ \t]*$", re.S | re.M
)
_SPACED_ASSIGN = re.compile(r"(?:^|[\s;&|(])([A-Za-z_]\w*)=([\"'])[^\"'\n]*\s[^\"'\n]*\2")


def _variable_command(command: str) -> str | None:
    """Name of a variable assigned a spaced string and then run in command position."""
    if re.search(r"\b(?:ba)?sh\s+-c\b", command):
        return None  # bash word-splits; the inner script is not zsh's business
    command = _HEREDOC_BODY.sub("", command)  # a heredoc body is data (a script file, Python)
    for name in dict.fromkeys(m.group(1) for m in _SPACED_ASSIGN.finditer(command)):
        if re.search(rf"(?:^|[;&|(]|&&|\|\||\bdo|\bthen)\s*\$\{{?{name}\}}?\s", command, re.M):
            return name
    return None


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


def _gnu_sed_in_place(words: list[str]) -> bool:
    """`sed ... -i <word>` where <word> is not '' or a `.suffix` (BSD takes it as the suffix)."""
    for at, word in enumerate(words):
        if word != "sed" and not word.endswith("/sed"):
            continue
        args = words[at + 1 :]
        if "-i" in args:
            nxt = args[args.index("-i") + 1 : args.index("-i") + 2]
            return bool(nxt) and nxt[0] != "" and not nxt[0].startswith(".")
    return False


def reason(command: str) -> str | None:
    segs = _segments(command)  # heredoc bodies and quoted text are not commands
    for seg in segs:
        if _pattern_kill(_words(seg)):
            return "pkill -f / killall hits other agents' processes; kill the pid you started ($!)."
        if _gnu_sed_in_place(_words(seg)):
            return (
                "macOS sed reads the word after -i as the backup suffix; "
                "use sed -i '' 's/…/…/' file, or the Edit tool."
            )
    if any(_pgrep_loop(seg) for seg in segs):
        return "a pgrep -f loop matches its own command line and never ends; wait with `while kill -0 <pid>`."
    name = _variable_command(command)
    if name:
        return (
            f"${name} runs as one program name in zsh (no word splitting, exit 127); "
            "write the command out, or use a function or an array."
        )
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
