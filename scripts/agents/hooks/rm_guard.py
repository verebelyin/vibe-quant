#!/usr/bin/env python3
"""PreToolUse hook: `rm` runs freely inside temp dirs, asks everywhere else.

Replaces the global `ask: Bash(rm:*)` rule. An ask rule can't carve out /tmp
(ask beats allow, and a hook can't override an ask rule), and it broke
subagents, which cannot answer a prompt ("Claude V2 adapter has no active turn").
Subagents (hook input carries `agent_id`) therefore get "deny" with a reason
instead of "ask": the agent reads it and rewrites the command. 68 subagent rm
calls died on that error 10-16..10-19, mostly `D=/tmp/x; rm -rf $D`, so variables
assigned earlier in the same command and `$(mktemp -d)` resolve statically.

Decision per Bash command:
  - no `rm` subcommand                        -> no output (normal permission flow)
  - an `rm` whose target is outside a temp dir,
    unresolvable ($VAR, $(..), glob root, xargs) -> "ask"
  - only temp-dir rm (plus cd/mkdir/true/:)    -> "allow"
  - temp-dir rm mixed with other commands      -> no output (normal flow judges the rest)
Temp dirs: /tmp, /private/tmp, $TMPDIR, Claude's /private/tmp/claude-* scratch (all realpath'd),
plus a swarm job's scratch dir (data/swarm/<job>/..., gitignored; `_goldens`, `_gate-logs` and other
`_` dirs stay protected), plus an untracked, non-ignored FILE inside a linked worktree of this repo
(reviewers' probe tests in tests/unit; 39 denials 2026-11-06..11-27). Tracked files, directories,
ignored paths (data/catalog links) and the main checkout still ask. The target must be strictly inside (`rm -rf /tmp` itself asks).
A brace group holding `/` or `..` is unknowable (`/tmp/{a,../../etc}` expands outside /tmp).
Unattended sessions (subagents with `agent_id`, T3 delegated children whose T3CODE_THREAD_ID starts
`thread:delegated-task:`) get "deny" instead of "ask": nobody is there to answer, and the ask
otherwise pops up on the user's screen.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

SEPARATORS = {"&&", "||", ";", "|", "&", "|&", ";;"}
WRAPPERS = {"command", "builtin", "nice", "time", "nohup"}
KEYWORDS = {
    "do",
    "then",
    "else",
    "elif",
    "if",
    "while",
    "until",
    "{",
    "!",
}  # `do rm $d` is still rm
HARMLESS = {"cd", "mkdir", "true", ":", "echo", "ls"}
GLOB_CHARS = set("*?[")


def temp_roots() -> list[str]:
    roots = {"/tmp", "/private/tmp"}
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        roots.add(tmpdir)
    return sorted({os.path.realpath(r) for r in roots})


_REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_BRACE = re.compile(r"\{([^{}]*,[^{}]*)\}")


def inside_job_scratch(path: str) -> bool:
    rel = path.removeprefix(os.path.join(_REPO, "data", "swarm") + "/")
    if rel == path:
        return False
    job, _, rest = rel.partition("/")
    return bool(job) and not job.startswith((".", "_")) and bool(rest)


def inside_temp(path: str, roots: list[str]) -> bool:
    return any(path.startswith(root.rstrip("/") + "/") for root in roots) or inside_job_scratch(
        path
    )


def untracked_in_linked_worktree(path: str) -> bool:
    """An existing regular file that `git status` lists as untracked (`??`, not ignored) in a
    linked worktree of _REPO; never the main checkout."""
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    folder = os.path.dirname(path)

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-C", folder, *args], capture_output=True, text=True, timeout=5
        )
        return done.stdout.strip() if done.returncode == 0 else ""

    common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    top = git("rev-parse", "--show-toplevel")
    if not common or not top or os.path.realpath(top) == _REPO:
        return False
    if os.path.realpath(common) != os.path.realpath(os.path.join(_REPO, ".git")):
        return False
    return git("status", "--porcelain", "--untracked-files=all", "--", path).startswith("?? ")


def split_subcommands(command: str) -> list[list[str]] | None:
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    subs: list[list[str]] = [[]]
    for tok in tokens:
        if tok in SEPARATORS:
            subs.append([])
        else:
            subs[-1].append(tok)
    return [s for s in subs if s]


def strip_prefix(argv: list[str]) -> list[str]:
    i = 0
    while i < len(argv) and (
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[i]) or argv[i] in WRAPPERS or argv[i] in KEYWORDS
    ):
        i += 1
    return argv[i:]


_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)
# `$(mktemp -d)` / `$(mktemp -d -t name)` always lands in $TMPDIR (a temp root).
_MKTEMP = re.compile(r"\$\(mktemp(?:\s+-[dqtu]+)*(?:\s+[\w.-]+)?\s*\)")


def expand_local(text: str, local: dict[str, str | None]) -> str | None:
    """Substitute variables assigned earlier in this command; None if one is unknowable."""
    unknown = False

    def sub(m: re.Match[str]) -> str:
        nonlocal unknown
        name = m.group(1)
        if name not in local:
            return m.group(0)
        value = local[name]
        if value is None:
            unknown = True
            return ""
        return value

    out = _VAR.sub(sub, text)
    return None if unknown else out


def resolve_target(target: str, cwd: str, local: dict[str, str | None] | None = None) -> str | None:
    """Real path the rm would hit, or None when it can't be known statically."""
    if local:
        expanded_local = expand_local(target, local)
        if expanded_local is None:
            return None
        target = expanded_local
    if "$" in target or "`" in target:
        expanded = os.path.expandvars(target)
        if "$" in expanded:
            return None
        target = expanded
    target = os.path.expanduser(target)
    if any("/" in g or ".." in g for g in _BRACE.findall(target)):
        return None
    target = _BRACE.sub("x", target)
    if any(c in GLOB_CHARS for c in target):
        # Judge the directory the glob expands in; a glob in that directory part is unknowable.
        head = target[: min(target.index(c) for c in GLOB_CHARS if c in target)]
        target = head if head.endswith("/") else os.path.join(os.path.dirname(head), "x")
        if any(c in GLOB_CHARS for c in target):
            return None
    return os.path.realpath(os.path.join(cwd, target))


_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_SHELLS = re.compile(r"(^|[\s;&|(/])(bash|sh|zsh|ksh|dash|source|eval|xargs|\.)(\s|$)")


def strip_heredocs(command: str) -> str:
    """Drop heredoc bodies fed to a non-shell (python3 - <<'PY', cat > f <<EOF): they are data.

    A body fed to a shell (bash <<EOF, sh -s <<EOF) runs as commands, so it stays and is judged.
    """
    lines = command.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        tags = [(m.group(1), m.group(3)) for m in _HEREDOC.finditer(line)]
        if not tags or _SHELLS.search(line):
            continue
        for dash, tag in tags:
            while i < len(lines) and (lines[i].lstrip("\t") if dash else lines[i]) != tag:
                i += 1
            i += 1  # the terminator line
    return "\n".join(out)


def decide(command: str, cwd: str) -> tuple[str | None, str]:
    command = strip_heredocs(command)
    if not re.search(r"(^|[\s;&|(`/])rm(\s|$)", command):
        return None, ""
    command = _MKTEMP.sub(os.path.join(temp_roots()[0], "mktemp.XXXXXX"), command)
    if re.search(r"\$\(|`|<\(|\beval\b|\bxargs\b[^;&|]*\brm\b", command):
        return "ask", "rm inside a substitution/eval/xargs: target unknown"
    subs = split_subcommands(command)
    if subs is None:
        return "ask", "rm in a command that could not be parsed"
    roots = temp_roots()
    all_harmless = True
    saw_rm = False
    local: dict[str, str | None] = {}
    for raw in subs:
        words = raw[1:] if raw and raw[0] == "export" else raw
        if words and all(_ASSIGN.match(w) for w in words):
            for w in words:
                m = _ASSIGN.match(w)
                assert m
                value = expand_local(m.group(2), local)
                if value is not None and "`" not in value:
                    value = os.path.expandvars(value)
                local[m.group(1)] = None if value is None or "$" in value or "`" in value else value
            continue
        argv = strip_prefix(raw)
        if not argv:
            continue
        prog = os.path.basename(argv[0])
        if prog == "sudo":
            return "ask", "sudo"
        if prog == "cd":
            if len(argv) > 1:
                nxt = resolve_target(argv[1], cwd, local)
                if nxt is None:
                    return "ask", "cd to an unresolvable dir before rm"
                cwd = nxt
            continue
        if prog != "rm":
            if prog not in HARMLESS:
                all_harmless = False
            continue
        saw_rm = True
        targets: list[str] = []
        end_opts = False
        for arg in argv[1:]:
            if not end_opts and arg == "--":
                end_opts = True
            elif not end_opts and arg.startswith("-"):
                continue
            else:
                targets.append(arg)
        if not targets:
            return "ask", "rm without a target"
        for t in targets:
            real = resolve_target(t, cwd, local)
            if real is None:
                return "ask", f"rm target not statically known: {t}"
            if not inside_temp(real, roots) and not untracked_in_linked_worktree(real):
                return "ask", f"rm outside temp dirs: {t} -> {real}"
    if saw_rm and all_harmless:
        return "allow", "rm only inside temp dirs"
    return None, ""


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    if data.get("tool_name") != "Bash":
        return 0
    command = (data.get("tool_input") or {}).get("command") or ""
    cwd = data.get("cwd") or os.getcwd()
    decision, reason = decide(command, cwd)
    if decision is None:
        return 0
    unattended = os.environ.get("T3CODE_THREAD_ID", "").startswith("thread:delegated-task:")
    if decision == "ask" and (data.get("agent_id") or unattended):
        decision = "deny"
        reason += (
            " — subagents cannot answer a permission prompt. Use a literal /tmp path or a variable"
            " assigned to one earlier in the same command (D=/tmp/x; rm -rf $D), a data/swarm/<job>/ path, or leave the"
            " file for the chief."
        )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": f"rm_guard: {reason}",
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
