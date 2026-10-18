"""PreToolUse hook: block `bd update ... --notes` (it replaces the whole notes field).

Agents lost bead notes this way twice (6osqd in job 20261017-grind, megq1 in
20261017-power-paper) despite the warning in docs/claude/beads.md. Use
`--append-notes`. Exit 2 blocks the call and shows stderr to the agent.
"""

import json
import re
import sys

_REPLACE_NOTES = re.compile(r"^\s*bd\s+update\b.*?(?<![\w-])--notes\b")


def blocks(command: str) -> bool:
    joined = command.replace("\\\n", " ")  # shell line continuations
    return any(_REPLACE_NOTES.search(part) for part in re.split(r"&&|\|\||;|\n", joined))


def main() -> int:
    try:
        command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    except (json.JSONDecodeError, AttributeError):
        return 0
    if blocks(command):
        print("bd update --notes replaces all existing notes; use --append-notes.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
