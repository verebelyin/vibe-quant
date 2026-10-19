#!/usr/bin/env python3
"""Model routing CLI: one machine-readable file, one command.

    models.py [--file PATH] route ROLE
    models.py [--file PATH] set KEY --off [--until ISO] [--note TEXT]
    models.py [--file PATH] set KEY --on [--note TEXT]
    models.py [--file PATH] show

The file (default ``docs/orchestration/models.json``) holds ``models`` (each with
``runtime``, ``target``, ``enabled``, ``until``, ``note``) and ``routes`` (role ->
ordered model keys). ``route ROLE`` prints the first available model as

    ROLE -> <key> <runtime> <target as JSON>

then one ``  skipped <key>: <reason>`` line per model passed over. A model is
available when ``enabled`` is true, or when it is disabled with an ``until`` in
the past (ISO 8601 with offset; a naive value is local time) — so a temporary
switch-off re-enables itself.

``set`` rewrites the file (atomic: temp file + ``os.replace``) and stamps a
top-level ``"changed"`` timestamp. ``--until`` must be a valid ISO 8601
timestamp; anything else exits 2 without writing (a stored typo would otherwise
read as "off forever"). Unknown role or key exits 2 listing the valid ones.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_DEFAULT_FILE = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "models.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _is_past(value: str) -> bool:
    """True when ``value`` is a parseable ISO 8601 instant at or before now.

    A naive value is compared as local time; an unparseable value is treated as
    the future (still unavailable), never as "expired".
    """
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return False
    now = datetime.now(when.tzinfo) if when.tzinfo else datetime.now()
    return when <= now


def availability(model: Mapping[str, object]) -> tuple[bool, str]:
    """(available now, skip reason). See the module docstring."""
    if model.get("enabled"):
        return True, ""
    until = model.get("until")
    if isinstance(until, str) and until:
        if _is_past(until):
            return True, ""
        return False, f"off until {until}"
    return False, "off"


def load(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text()))


def save(path: Path, data: Mapping[str, object]) -> None:
    """Atomic write (temp file in the same dir + os.replace), pretty-printed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _models(data: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = data.get("models")
    return raw if isinstance(raw, dict) else {}


def _routes(data: Mapping[str, object]) -> dict[str, list[str]]:
    raw = data.get("routes")
    return raw if isinstance(raw, dict) else {}


def resolve(
    data: Mapping[str, object], role: str
) -> tuple[tuple[str, dict[str, object]] | None, list[tuple[str, str]]]:
    """(first available (key, model) on the role's route or None, skipped (key, reason))."""
    models = _models(data)
    skipped: list[tuple[str, str]] = []
    for key in _routes(data).get(role, []):
        model = models.get(key)
        if model is None:
            skipped.append((key, "unknown model"))
            continue
        available, reason = availability(model)
        if available:
            return (key, model), skipped
        skipped.append((key, reason))
    return None, skipped


def cmd_route(data: Mapping[str, object], role: str) -> int:
    routes = _routes(data)
    if role not in routes:
        print(f"unknown role {role!r}; valid roles: {', '.join(routes)}", file=sys.stderr)
        return 2
    chosen, skipped = resolve(data, role)
    if chosen is None:
        print(f"{role} -> none available")
        for key, reason in skipped:
            print(f"  skipped {key}: {reason}")
        return 1
    key, model = chosen
    print(f"{role} -> {key} {model.get('runtime')} {json.dumps(model.get('target'))}")
    for skip_key, reason in skipped:
        print(f"  skipped {skip_key}: {reason}")
    return 0


def cmd_set(
    data: dict[str, object],
    key: str,
    *,
    off: bool,
    until: str | None,
    note: str | None,
) -> int:
    models = _models(data)
    if key not in models:
        print(f"unknown model key {key!r}; valid keys: {', '.join(models)}", file=sys.stderr)
        return 2
    if until is not None:
        try:
            datetime.fromisoformat(until)
        except ValueError:
            print(f"invalid --until {until!r}: not an ISO 8601 timestamp", file=sys.stderr)
            return 2
    model = models[key]
    if off:
        model["enabled"] = False
        model["until"] = until
    else:
        model["enabled"] = True
        model["until"] = None
    if note is not None:
        model["note"] = note
    data["changed"] = _now()
    return 0


def cmd_show(data: Mapping[str, object]) -> int:
    for key, model in _models(data).items():
        if model.get("enabled"):
            state = "ON"
        elif model.get("until"):
            state = f"OFF(until {model['until']})"
        else:
            state = "OFF"
        note = model.get("note")
        print(f"{key} {state}" + (f" {note}" if note else ""))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="models.py", description="Model routing.")
    parser.add_argument("--file", default=str(_DEFAULT_FILE), help="models JSON path")
    subs = parser.add_subparsers(dest="command", required=True)

    route = subs.add_parser("route", help="print the model serving a role")
    route.add_argument("role")

    setter = subs.add_parser("set", help="enable/disable a model")
    setter.add_argument("key")
    toggle = setter.add_mutually_exclusive_group(required=True)
    toggle.add_argument("--off", action="store_true")
    toggle.add_argument("--on", action="store_true")
    setter.add_argument("--until", default=None, help="ISO 8601 auto-re-enable time (with --off)")
    setter.add_argument("--note", default=None)

    subs.add_parser("show", help="one line per model")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.file)

    if args.command == "route":
        return cmd_route(load(path), args.role)

    data = load(path)
    if args.command == "set":
        rc = cmd_set(data, args.key, off=args.off, until=args.until, note=args.note)
        if rc == 0:
            save(path, data)
        return rc
    return cmd_show(data)


if __name__ == "__main__":
    raise SystemExit(main())
