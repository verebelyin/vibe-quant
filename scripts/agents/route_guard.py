#!/usr/bin/env python3
"""PreToolUse hook: refuse a dispatch whose model differs from the route mkbrief recorded.

Why (doctor B4): the chief overrode the routed model 5 times (Opus instead of the route)
although ``docs/orchestration/models.json`` is the only routing source; prose did not stop
it. So the route ``mkbrief.py --route <role>`` recorded next to the brief
(``<job>/prompts/<id>.delegate.json``) is checked against the call that actually goes out.

Reads the hook payload as JSON on stdin (``tool_name``, ``tool_input``). Exit 2 blocks the
call and stderr is shown to the model; exit 0 allows. Allowed without a word: any other
tool, a task/prompt with no ``Read your brief at <path>`` line (except the unbriefed
lookups below), a brief whose route record
is missing or unreadable, and malformed stdin — the guard never blocks what it cannot judge.

``mcp__t3-code__delegate_task``: the call's ``target.providerInstanceId`` and
``target.model`` must match the record's, else

    route_guard: <id> routed to <model> by models.py; this call uses <model>.
    Re-render with mkbrief --route, or run models.py set (user decision) first.

``Agent``: the call's ``model`` must match the recorded model. A call that omits ``model``
inherits the chief's model, which is an override as much as a wrong name, so it is refused
too (the dispatch must name its model); the message says ``(inherited)``.

An ``Agent`` call with no brief line whose ``subagent_type`` is ``general-purpose`` (or
absent) or ``Explore`` and that names no ``model`` is refused too (retro 2026-11-05: 17
such lookups inherited Opus): the message names the ``lookup`` route's model. Agent types
with their own frontmatter model, and forks (which cannot take one), are allowed.

``SWARM_ROUTE_OVERRIDE=<reason>`` in the environment allows the call and appends one JSON
line ``{ts, id, routed, used, reason}`` to ``<job>/route-overrides.jsonl``, so every
override leaves a trace. Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

_DELEGATE = "mcp__t3-code__delegate_task"
_AGENT = "Agent"
# The dispatch instruction is the text's FIRST line (a quoted example later is not a dispatch).
_CLAUDE_T3 = "claudeAgent"  # T3 provider of Claude children (background dispatch)
_BRIEF_RE = re.compile(r"\A\s*Read your brief at (\S+?)(?:\s|$)")
_INHERITED = "(inherited)"
_OVERRIDE_ENV = "SWARM_ROUTE_OVERRIDE"
# Built-in Agent types whose model is the caller's unless the call names one.
_LOOKUP_TYPES = ("general-purpose", "Explore")
_MODELS = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "models.json"


class Verdict(NamedTuple):
    """(exit code, stderr message, (log file, JSON line) for an overridden call)."""

    code: int
    message: str = ""
    log: tuple[Path, dict[str, str]] | None = None


def route_id(text: str) -> tuple[Path, str] | None:
    """(job dir, brief id) named by a dispatch text's ``Read your brief at`` line, or None."""
    match = _BRIEF_RE.search(text)
    if match is None:
        return None
    brief = Path(match.group(1))
    if brief.suffix != ".md" or brief.parent.name != "prompts":
        return None
    return brief.parent.parent, brief.stem


def record_path(job: Path, ident: str) -> Path:
    return job / "prompts" / f"{ident}.delegate.json"


def recorded(path: Path) -> tuple[str, str] | None:
    """(model, provider) ``mkbrief --route`` recorded, or None when absent/unreadable.

    A T3 record carries the whole ``target`` (provider + model); a Claude record only the
    model, so its provider is "".
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    target = data.get("target")
    if isinstance(target, dict):
        model, provider = target.get("model"), target.get("providerInstanceId")
    else:
        model, provider = data.get("model"), ""
    if not isinstance(model, str) or not model:
        return None
    return model, provider if isinstance(provider, str) else ""


def used_target(tool_input: dict[str, Any]) -> tuple[str, str]:
    """(model, provider) the call itself dispatches to; an Agent call may inherit."""
    target = tool_input.get("target")
    if isinstance(target, dict):
        model, provider = target.get("model"), target.get("providerInstanceId")
        return (model if isinstance(model, str) else ""), (
            provider if isinstance(provider, str) else ""
        )
    model = tool_input.get("model")
    return (model if isinstance(model, str) and model else _INHERITED), ""


def lookup_model() -> str:
    """The model ``models.py route lookup`` names (first entry; "sonnet" if unreadable)."""
    try:
        data = json.loads(_MODELS.read_text())
        return str(data["models"][data["routes"]["lookup"][0]]["target"]["model"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return "sonnet"


def _unbriefed(tool: Any, tool_input: dict[str, Any]) -> Verdict:
    """An Agent call with no brief: a lookup type must name its model (it would inherit)."""
    kind = tool_input.get("subagent_type") or "general-purpose"
    if tool != _AGENT or kind not in _LOOKUP_TYPES or tool_input.get("model"):
        return Verdict(0)
    if os.environ.get(_OVERRIDE_ENV, ""):
        return Verdict(0)
    return Verdict(
        2,
        f"route_guard: a {kind} Agent call without model runs on the chief's model. "
        f"Pass model={lookup_model()!r} (models.py route lookup), "
        f"or set {_OVERRIDE_ENV}=<reason>.",
    )


def decision(payload: Any) -> Verdict:
    """The guard's answer for one hook payload (see the module docstring)."""
    if not isinstance(payload, dict):
        return Verdict(0)
    tool = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if tool not in (_DELEGATE, _AGENT) or not isinstance(tool_input, dict):
        return Verdict(0)
    text = tool_input.get("task" if tool == _DELEGATE else "prompt")
    found = route_id(text) if isinstance(text, str) else None
    if found is None:
        return _unbriefed(tool, tool_input)
    job, ident = found
    if tool == _DELEGATE:
        request = tool_input.get("clientRequestId")
        if isinstance(request, str) and request:
            ident = request
    stored = recorded(record_path(job, ident))
    if stored is None:
        return Verdict(0)
    routed, routed_provider = stored
    used, used_provider = used_target(tool_input)
    if used == routed and (not routed_provider or used_provider == routed_provider):
        return Verdict(0)
    if not routed_provider and used_provider == _CLAUDE_T3 and used.startswith(f"claude-{routed}-"):
        return Verdict(0)  # a Claude route dispatched as a background T3 child, same family
    reason = os.environ.get(_OVERRIDE_ENV, "")
    entry = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "id": ident,
        "routed": routed,
        "used": used,
        "reason": reason,
    }
    if reason:
        return Verdict(0, "", (job / "route-overrides.jsonl", entry))
    return Verdict(
        2,
        f"route_guard: {ident} routed to {routed} by models.py; this call uses {used}. "
        "Re-render with mkbrief --route, or run models.py set (user decision) first.",
    )


def append_log(path: Path, entry: dict[str, str]) -> str:
    """One JSON line per override; "" when written, else why not (the override is refused:
    an override without its logged reason is a silent route change)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as exc:
        return f"route_guard: cannot log the override to {path}: {exc}"
    return ""


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except (OSError, ValueError):
        return 0
    verdict = decision(payload)
    if verdict.log is not None:
        failure = append_log(*verdict.log)
        if failure:
            print(failure, file=sys.stderr)
            return 2
    if verdict.message:
        print(verdict.message, file=sys.stderr)
    return verdict.code


if __name__ == "__main__":
    raise SystemExit(main())
