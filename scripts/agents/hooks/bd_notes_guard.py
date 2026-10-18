"""PreToolUse hook: block `bd update ... --notes` (it replaces the whole notes field).

Agents lost bead notes this way twice (6osqd in job 20261017-grind, megq1 in
20261017-power-paper) despite the warning in docs/claude/beads.md. Use
`--append-notes`. Exit 2 blocks the call and shows stderr to the agent.

The command is split into simple commands at unquoted `;` `&` `|` and newlines
(heredoc bodies and quoted text are literal, never commands), then each is
tokenized with shlex. Handled: leading VAR=val assignments, an `env` wrapper, a
leading `(`; `env` options with a separate operand (-u -C). NOT handled
(bypasses): `$(...)`, backticks, `bash -c '...'`, `env -S '...'` (operand is a
command string, not inspected), escaped or concatenated heredoc
delimiters (<<\\EOF, <<E"OF"), unmatched quotes inside # comments, an
--append-notes whose separate value is exactly "--notes". This is a convenience
guard, not a security boundary.
Anything unparseable (unbalanced quote, non-string command) fails open (exit 0)
unless an earlier segment was already a confirmed destructive update.
"""

import json
import re
import shlex
import sys

_ASSIGN = re.compile(r"[A-Za-z_]\w*=")
_ENV_OPERAND_OPTS = {"-u", "-C", "--unset", "--chdir"}
_HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([^'\n]*)'|\"([^\"\n]*)\"|([\w.-]+))")


def _segments(command: str) -> list[str]:
    """Split at unquoted separators; drop heredoc bodies."""
    text = command.replace("\\\n", " ")  # shell line continuations
    segs: list[str] = []
    cur: list[str] = []
    pending: list[tuple[bool, str]] = []  # (strip tabs, terminator)
    quote = ""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote:
            cur.append(c)
            if c == quote:
                quote = ""
            elif c == "\\" and quote == '"' and i + 1 < n:
                cur.append(text[i + 1])
                i += 1
        elif c == "\\" and i + 1 < n:
            cur.append(c + text[i + 1])
            i += 1
        elif c in "'\"":
            quote = c
            cur.append(c)
        elif text.startswith("<<<", i):
            cur.append("<<<")  # here-string, not a heredoc
            i += 2
        elif c == "<" and text.startswith("<<", i):
            m = _HEREDOC.match(text, i)
            if m:
                pending.append((m.group(1) == "-", next(g for g in m.groups()[1:] if g is not None)))
                cur.append(" ")
                i = m.end() - 1
            else:
                cur.append(c)
        elif c == "&" and ((cur and cur[-1] in "<>") or text.startswith("&>", i)):
            cur.append(c)  # redirection like 2>&1 or &>file
        elif c in ";&|\n":
            segs.append("".join(cur))
            cur = []
            if c == "\n":
                while pending:
                    strip, term = pending.pop(0)
                    while i + 1 < n:
                        end = text.find("\n", i + 1)
                        end = n if end < 0 else end
                        line = text[i + 1 : end]
                        i = end
                        if (line.lstrip("\t") if strip else line) == term:
                            break
        else:
            cur.append(c)
        i += 1
    segs.append("".join(cur))
    return segs


def _replaces_notes(words: list[str]) -> bool:
    if words and words[0].startswith("("):
        words = [words[0].lstrip("("), *words[1:]]
    while words and (_ASSIGN.match(words[0]) or words[0] == "env"):
        wrapper = words[0] == "env"
        words = words[1:]
        while wrapper and words and (words[0].startswith("-") or _ASSIGN.match(words[0])):
            skip = 2 if words[0] in _ENV_OPERAND_OPTS else 1
            words = words[skip:]
    if words[:2] != ["bd", "update"]:
        return False
    for w in words[2:]:
        if w == "--":
            return False
        if w == "--notes" or w.startswith("--notes="):
            return True
    return False


def blocks(command: str) -> bool:
    for seg in _segments(command):
        try:
            words = shlex.split(seg)
        except ValueError:
            continue  # fail open
        if _replaces_notes(words):
            return True
    return False


def main() -> int:
    try:
        command = json.load(sys.stdin)["tool_input"]["command"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return 0
    if isinstance(command, str) and blocks(command):
        print("bd update --notes replaces all existing notes; use --append-notes.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
