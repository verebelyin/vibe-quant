#!/usr/bin/env python3
"""Compact, read-only T3 history for retros: one scratch DB instead of re-deriving queries.

    scripts/agents/t3_history.py extract --since ISO [--until ISO] --out scratch.sqlite [--db SRC]
    scripts/agents/t3_history.py signatures --db scratch.sqlite

The source (`~/.t3/userdata/statev2.sqlite`, ~10 GB) is opened with `mode=ro` and always
filtered by thread id and time. `extract` copies the top-level threads created in
[--since, --until) — a thread is top-level when no subagent row names it as its child —
plus ALL their descendants (children, grandchildren, ...; whatever their start time), their user messages, and their
command_execution items: every command's text and status, with the captured output kept
only for non-completed items (the failures `signatures` reads; this is what keeps the
scratch DB small). Token usage is copied from a child's payload (`usage`), when T3 stored it.

`signatures` reads only the scratch DB and prints, for the retros that used to rebuild these
queries in /tmp (28 runs): error families, human correction candidates, and runs per model
family with the stored tokens. A family is one normalised diagnostic (`<program>:<error
class>`, `hook:<guard>:<reason>`, `pytest:failed`, ...); a failed command that shows several
diagnostics is counted once in each family, so the family counts overlap and summing the
`sed:` families gives exactly the failed commands whose output holds a `sed: ` line. `until`
is exclusive, so `--since 2026-10-15 --until 2026-11-06` is 2026-10-15..2026-11-05.
--since/--until are instants: an offset (`+02:00`) is honoured, a bare date or time is UTC. The
scratch DB is never written under `~/.t3` (the live T3 state) nor over the source DB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

T3_DB = Path.home() / ".t3" / "userdata" / "statev2.sqlite"

SCRATCH_SCHEMA = """
create table meta (since text not null, until text, source text not null, generated_at text not null);
create table threads (
    thread_id text primary key, title text, project_id text, provider text, model text,
    created_at text, updated_at text
);
create table children (
    subagent_id text primary key, parent_thread_id text not null, child_thread_id text,
    provider text, model text, status text, started_at text, completed_at text,
    usage_json text, title text
);
create table commands (
    item_id text primary key, thread_id text not null, status text not null, ordinal integer,
    updated_at text, first_token text, command text, output text, exit_code integer
);
create table messages (
    item_id text primary key, thread_id text not null, ordinal integer, updated_at text,
    created_by text, text text
);
create index commands_thread on commands(thread_id);
create index commands_failed on commands(status) where status != 'completed';
create index messages_thread on messages(thread_id);
"""

CORRECTION_MARKERS = re.compile(r"(?i)(\bno,|\bwrong\b|explain simply|\bstatus\?|\beta\b)")
NO_ERROR = ("completed",)
MAX_REASON_WORDS = 3

LINE_RE = re.compile(r"^(?P<prog>[a-z][a-z0-9_-]*)(?::(?P<line>\d+))?: (?P<msg>\S.*)$")
BLOCKED_RE = re.compile(r"(?m)^(?:Error: )?Blocked: (?P<reason>.+)$")
HOOK_RE = re.compile(
    r"(?m)^(?:Error: )?PreToolUse:[^\n]*hook error: "
    r"(?:\[(?P<script>[^\]]+)\]: |(?P<guard>\w+): )(?P<reason>.+)"
)
ERROR_LINE_RE = re.compile(r"(?m)^Error: (?!Exit code)(?P<msg>.+)$")
EXIT_RE = re.compile(r"Error: Exit code (?P<code>\d+)")
PYTEST_RE = re.compile(r"(?m)^FAILED \S")
MUTANTS_RE = re.compile(r"(?m)^SURVIVED |survived=[1-9]")
TYPE_RE = re.compile(r"(?m)^\S+:\d+: error: ")
SEGMENT_RE = re.compile(r"&&|\|\||[|;(){}\n\"'`]|\$\(")

ERROR_CLASSES: list[tuple[str, str]] = [
    (r"invalid command code", "invalid command code"),
    (r"undefined label", "undefined label"),
    (r"extra characters at the end of .*command", "extra characters at end of command"),
    (r"expects \\ followed by text", "expects backslash-text"),
    (r"unescaped newline", "unescaped newline"),
    (r"may not be used with stdin", "cannot use -i with stdin"),
    (r"No such file or directory", "No such file or directory"),
    (r"no matches found", "no matches found"),
    (r"command not found", "command not found"),
    (r"Permission denied", "Permission denied"),
    (r"unbound variable", "unbound variable"),
    (r"ambiguous redirect", "ambiguous redirect"),
    (r"unterminated", "unterminated string"),
    (r"not a git repository", "not a git repository"),
    (r"(?:unknown|invalid|unrecognized) option|illegal option", "bad option"),
    (r"EOF", "unexpected EOF"),
    (r"no active turn", "no active turn"),
    (r"no such tool", "no such tool"),
    (r"cannot find module|ModuleNotFoundError", "module not found"),
    (r"fails handoff lint", "handoff lint"),
]

REASON_CLASSES: list[tuple[str, str]] = [
    (r"unquoted heredoc", "unquoted heredoc"),
    (r"pkill|killall", "pkill/killall"),
    (r"bd update --notes", "bd update --notes"),
    (r"`?sleep\b|^sleep\b", "foreground sleep"),
    (r"rm inside a substitution|target unknown", "unknown rm target"),
    (r"cd to an unresolvable dir", "rm after unresolvable cd"),
    (r"can't open file|No such file", "guard script missing"),
]

MODEL_FAMILIES: list[tuple[str, str]] = [
    ("opus", "Opus"),
    ("sonnet", "Sonnet"),
    ("haiku", "Haiku"),
    ("gpt", "GPT"),
    ("codex", "GPT"),
    ("deepseek", "DeepSeek"),
]


@dataclass(frozen=True)
class Command:
    item_id: str
    thread_id: str
    status: str
    first_token: str | None
    command: str | None
    output: str | None
    exit_code: int | None


@dataclass(frozen=True)
class Message:
    item_id: str
    thread_id: str
    created_by: str | None
    text: str


@dataclass(frozen=True)
class Child:
    child_thread_id: str
    provider: str | None
    model: str | None
    usage: dict[str, object]


@dataclass(frozen=True)
class History:
    """One extracted window: the only input `signatures` needs."""

    since: str
    until: str | None
    threads: list[tuple[str, str | None]]  # (thread_id, model)
    children: list[Child]
    commands: list[Command]
    messages: list[Message]


@dataclass(frozen=True)
class Family:
    signature: str
    occurrences: int
    threads: int


@dataclass(frozen=True)
class ModelRuns:
    family: str
    runs: int
    tokens: int  # sum of the tokens T3 stored for this family's runs
    approximate: int  # runs whose token count is a T3 approximation (total only)
    unknown: int  # runs with no stored token count


@dataclass(frozen=True)
class Report:
    history: History
    failed: int
    families: list[Family]
    corrections: list[tuple[str, str]]  # (thread_id, message text)
    models: list[ModelRuns]


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _json(text: object) -> dict[str, object]:
    try:
        loaded = json.loads(str(text))
    except ValueError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _model_of(payload_json: object) -> str | None:
    payload = _json(payload_json)
    selection = payload.get("modelSelection")
    model = selection.get("model") if isinstance(selection, dict) else payload.get("model")
    return str(model) if model else None


def first_token(command: str | None) -> str | None:
    """The command's first word, skipping leading `VAR=value` assignments."""
    for token in (command or "").split():
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", token):
            continue
        return token.lstrip("(").strip("'\"") or None
    return None


def exit_code(output: str | None) -> int | None:
    match = EXIT_RE.search(output or "")
    return int(match.group("code")) if match else None


def _mask(message: str) -> str:
    """Drop sed's line prefix and any quoted path, then collapse whitespace."""
    text = re.sub(r"^\d+:\s*", "", message.strip())
    text = re.sub(r"\"[^\"]*\"|'[^']*'", "", text)
    return " ".join(text.split()).lstrip(": ").strip()


def _classify(text: str, classes: list[tuple[str, str]]) -> str | None:
    for pattern, name in classes:
        if re.search(pattern, text, re.I):
            return name
    return None


def _words(text: str, count: int = MAX_REASON_WORDS) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " _-." else " " for ch in _mask(text))
    words = " ".join(cleaned.lower().split()[:count])
    return words or "unclassified"


def _hook(output: str) -> str | None:
    blocked = BLOCKED_RE.search(output)
    if blocked:
        reason = blocked.group("reason")
        return f"hook:blocked:{_classify(reason, REASON_CLASSES) or _words(reason)}"
    hook = HOOK_RE.search(output)
    if hook:
        script = hook.group("script")
        name = Path(script.split()[-1]).stem if script else hook.group("guard")
        reason = hook.group("reason")
        return f"hook:{name}:{_classify(reason, REASON_CLASSES) or _words(reason)}"
    if "User rejected tool use" in output:
        return "hook:user-rejected"
    return None


def _programs(command: str | None) -> set[str]:
    """Words in a command position (start of the line or after a shell separator)."""
    found: set[str] = set()
    for segment in SEGMENT_RE.split(command or ""):
        token = first_token(segment)
        if token:
            found.add(Path(token).name)
    return found


def families_of(command: str | None, output: str | None) -> list[str]:
    """Every error family one failed tool call shows: a command can hide several."""
    if output is None:
        return ["no-output"]
    hook = _hook(output)
    if hook:
        return [hook]  # a guard refusal is the whole story
    found: list[str] = []
    for pattern, name in ((PYTEST_RE, "pytest:failed"), (MUTANTS_RE, "mutate:survived"),
                          (TYPE_RE, "mypy:error")):
        if pattern.search(output) and name not in found:
            found.append(name)
    if "Traceback (most recent call last)" in output:
        found.append("python:traceback")
    programs = _programs(command)
    for line in output.splitlines():
        match = LINE_RE.match(line)
        if match is None:
            continue
        program, message = match.group("prog"), match.group("msg")
        class_name = _classify(message, ERROR_CLASSES)
        if class_name is None and program not in programs:
            continue
        if class_name is None:
            script = re.sub(r"^\d+:\s*", "", message).lstrip()
            class_name = "malformed script" if script[:1] in ('"', "'", ",") else _words(message)
        name = f"{program}:{class_name}"
        if name not in found:
            found.append(name)
    for other in ERROR_LINE_RE.finditer(output):
        message = other.group("msg")
        name = f"error:{_classify(message, ERROR_CLASSES) or _words(message)}"
        if name not in found:
            found.append(name)
    if found:
        return found
    code = exit_code(output)
    return [f"exit:{code}" if code is not None else "unparsed"]


def signature(command: str | None, output: str | None) -> str:
    """The primary family of one failed tool call, e.g. `sed:invalid command code`."""
    return families_of(command, output)[0]


def _source_instant(text: str) -> str:
    """`text` as a UTC `YYYY-MM-DDTHH:MM:SS.mmmZ` string, the format T3 stores, rounded UP to the
    millisecond so string comparison keeps `>= since` and `< until` exact for sub-ms input."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    rounded = parsed.replace(microsecond=0) + timedelta(milliseconds=-(-parsed.microsecond // 1000))
    return rounded.strftime("%Y-%m-%dT%H:%M:%S.") + f"{rounded.microsecond // 1000:03d}Z"


def _refuse_protected_output(out: Path, source: Path) -> None:
    resolved = out.resolve()
    if resolved == source.resolve():
        raise ValueError(f"output {out} is the source DB; refusing to overwrite it")
    t3_dir = (Path.home() / ".t3").resolve()
    if resolved == t3_dir or t3_dir in resolved.parents:
        raise ValueError(f"output {out} is inside {t3_dir}; the live T3 state is never written")


def extract(since: str, until: str | None, out: Path, source: Path = T3_DB) -> dict[str, int]:
    """Copy the window's threads, their descendants and their turn items into `out`."""
    first = _source_instant(since)
    last = None if until is None else _source_instant(until)
    _refuse_protected_output(out, source)
    if out.exists():
        out.unlink()
    src = _connect_ro(source)
    try:
        params: tuple[str, ...] = (first,) if last is None else (first, last)
        window = "t.created_at >= ?" + ("" if last is None else " and t.created_at < ?")
        parents = src.execute(
            "select t.thread_id, t.title, t.project_id, t.default_provider, t.created_at, "
            "t.updated_at, t.payload_json from orchestration_v2_projection_threads t "
            f"where {window} and t.thread_id not in "
            "(select child_thread_id from orchestration_v2_projection_subagents "
            "where child_thread_id is not null) order by t.created_at",
            params,
        ).fetchall()
        ids = [str(row[0]) for row in parents]
        children: list[tuple[object, ...]] = []
        items: list[tuple[object, ...]] = []
        if ids:
            seen = set(ids)
            frontier = ids
            while frontier:  # every level of descendants; `seen` also ends any cycle
                marks = ",".join("?" * len(frontier))
                level = src.execute(
                    "select subagent_id, thread_id, child_thread_id, provider, status, started_at, "
                    f"completed_at, payload_json from orchestration_v2_projection_subagents "
                    f"where thread_id in ({marks})",
                    frontier,
                ).fetchall()
                children += level
                frontier = []
                for row in level:
                    if row[2] and str(row[2]) not in seen:
                        seen.add(str(row[2]))
                        frontier.append(str(row[2]))
            children.sort(key=lambda row: (str(row[5] or ""), str(row[0])))
            scope = sorted(seen)
            marks = ",".join("?" * len(scope))
            items = src.execute(
                "select turn_item_id, thread_id, type, status, ordinal, updated_at, payload_json "
                f"from orchestration_v2_projection_turn_items where thread_id in ({marks}) "
                "and type in ('command_execution', 'user_message') order by thread_id, ordinal",
                scope,
            ).fetchall()
    finally:
        src.close()

    scratch = sqlite3.connect(out)
    try:
        scratch.executescript(SCRATCH_SCHEMA)
        scratch.executemany(
            "insert into threads values (?,?,?,?,?,?,?)",
            [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), _model_of(r[6]), str(r[4]), str(r[5]))
             for r in parents],
        )
        scratch.executemany(
            "insert into children values (?,?,?,?,?,?,?,?,?,?)",
            [(str(r[0]), str(r[1]), None if r[2] is None else str(r[2]), str(r[3]), _model_of(r[7]),
              str(r[4]), None if r[5] is None else str(r[5]), None if r[6] is None else str(r[6]),
              json.dumps(_json(r[7]).get("usage")), str(_json(r[7]).get("title") or ""))
             for r in children],
        )
        commands: list[tuple[object, ...]] = []
        messages: list[tuple[object, ...]] = []
        for row in items:
            item_id, thread_id, kind, status = str(row[0]), str(row[1]), str(row[2]), str(row[3])
            ordinal, updated, payload = row[4], str(row[5]), _json(row[6])
            if kind == "command_execution":
                command = payload.get("input")
                output = payload.get("output")
                text = str(command) if command is not None else None
                kept = None if output is None or status in NO_ERROR else str(output)
                commands.append((item_id, thread_id, status, ordinal, updated, first_token(text),
                                 text, kept, exit_code(kept)))
            else:
                messages.append((item_id, thread_id, ordinal, updated,
                                 str(payload.get("createdBy") or "") or None,
                                 str(payload.get("text") or "")))
        scratch.executemany("insert into commands values (?,?,?,?,?,?,?,?,?)", commands)
        scratch.executemany("insert into messages values (?,?,?,?,?,?)", messages)
        scratch.execute(
            "insert into meta values (?,?,?,?)",
            (since, until, str(source), datetime.now(UTC).isoformat(timespec="seconds")),
        )
        scratch.commit()
    finally:
        scratch.close()
    return {"threads": len(parents), "children": len(children),
            "commands": len(commands), "messages": len(messages)}


def load(db: Path) -> History:
    con = _connect_ro(db)
    try:
        meta = con.execute("select since, until from meta").fetchone()
        threads = [(str(r[0]), None if r[1] is None else str(r[1]))
                   for r in con.execute("select thread_id, model from threads")]
        children = [
            Child(str(r[0]), None if r[1] is None else str(r[1]),
                  None if r[2] is None else str(r[2]), _json(r[3]))
            for r in con.execute("select child_thread_id, provider, model, usage_json from children")
        ]
        commands = [
            Command(str(r[0]), str(r[1]), str(r[2]),
                    None if r[3] is None else str(r[3]), None if r[4] is None else str(r[4]),
                    None if r[5] is None else str(r[5]), None if r[6] is None else int(r[6]))
            for r in con.execute(
                "select item_id, thread_id, status, first_token, command, output, exit_code "
                "from commands order by thread_id, ordinal")
        ]
        messages = [
            Message(str(r[0]), str(r[1]), None if r[2] is None else str(r[2]), str(r[3]))
            for r in con.execute(
                "select item_id, thread_id, created_by, text from messages order by thread_id, ordinal")
        ]
    finally:
        con.close()
    return History(since=str(meta[0]), until=None if meta[1] is None else str(meta[1]),
                   threads=threads, children=children, commands=commands, messages=messages)


def families(history: History) -> list[Family]:
    counts: Counter[str] = Counter()
    threads: dict[str, set[str]] = defaultdict(set)
    for command in history.commands:
        if command.status in NO_ERROR:
            continue
        for name in families_of(command.command, command.output):
            counts[name] += 1
            threads[name].add(command.thread_id)
    return [Family(name, count, len(threads[name])) for name, count in counts.most_common()]


def corrections(history: History) -> list[tuple[str, str]]:
    return [
        (message.thread_id, message.text)
        for message in history.messages
        if message.created_by == "user" and CORRECTION_MARKERS.search(message.text)
    ]


def _family(model: str | None) -> str:
    lowered = (model or "").lower()
    for needle, name in MODEL_FAMILIES:
        if needle in lowered:
            return name
    return model or "unknown"


def _int(value: object) -> int:
    return int(value) if isinstance(value, (int, float, str)) and not isinstance(value, bool) else 0


def _usage(usage: dict[str, object]) -> tuple[int, bool]:
    """(tokens, approximated): T3 stores input+output for some runs, only a total for others."""
    tokens_in = usage.get("inputTokens")
    if tokens_in is not None:
        return _int(tokens_in) + _int(usage.get("outputTokens")), False
    return _int(usage.get("totalTokens")), True


def models(history: History) -> list[ModelRuns]:
    runs: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    approximate: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    entries: list[tuple[str | None, dict[str, object] | None]] = [
        (model, None) for _, model in history.threads
    ]
    entries += [(child.model, child.usage) for child in history.children]
    for model, usage in entries:
        family = _family(model)
        runs[family] += 1
        if not usage or (usage.get("inputTokens") is None and usage.get("totalTokens") is None):
            unknown[family] += 1
            continue
        count, approx = _usage(usage)
        tokens[family] += count
        approximate[family] += int(approx)
    return [
        ModelRuns(family, count, tokens[family], approximate[family], unknown[family])
        for family, count in sorted(runs.items(), key=lambda item: (-item[1], item[0]))
    ]


def signatures(db: Path) -> Report:
    history = load(db)
    failed = sum(1 for command in history.commands if command.status not in NO_ERROR)
    return Report(history, failed, families(history), corrections(history), models(history))


def _tokens(entry: ModelRuns) -> str:
    if entry.tokens == 0 and entry.unknown == entry.runs:
        return "tokens unknown"
    text = f"tokens={entry.tokens:,}"
    if entry.approximate:
        text += " (approximate)"
    if entry.unknown:
        text += f", {entry.unknown} without usage"
    return text


def render(report: Report) -> str:
    window = f"{report.history.since}..{report.history.until or 'now'}"
    lines = [
        f"== window {window}  threads={len(report.history.threads)} "
        f"children={len(report.history.children)} commands={len(report.history.commands)} "
        f"failed={report.failed} messages={len(report.history.messages)}",
        "error families (failed commands, threads; one command can show several):",
    ]
    lines += [f"  {f.occurrences:4d} {f.threads:4d}  {f.signature}" for f in report.families]
    lines.append(f"user-correction candidates (human-authored): {len(report.corrections)}")
    for thread_id, text in report.corrections:
        snippet = " ".join(text.split())[:110]
        lines.append(f"  {thread_id}  {snippet}")
    lines.append("runs by model:")
    lines += [f"  {m.family} {m.runs} run(s) {_tokens(m)}" for m in report.models]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract", help="copy a window of T3 history into a scratch DB")
    ext.add_argument("--since", required=True, help="ISO timestamp, inclusive (offset honoured; bare = UTC)")
    ext.add_argument("--until", default=None, help="ISO timestamp, exclusive (default: now; offset honoured; bare = UTC)")
    ext.add_argument("--out", type=Path, required=True, help="scratch sqlite path (overwritten)")
    ext.add_argument("--db", type=Path, default=T3_DB, help="source statev2.sqlite (read-only)")
    sig = sub.add_parser("signatures", help="error families, corrections and runs by model")
    sig.add_argument("--db", type=Path, required=True, help="scratch DB written by extract")
    args = ap.parse_args(argv)
    if args.command == "extract":
        try:
            counts = extract(args.since, args.until, args.out, args.db)
        except ValueError as error:
            ap.error(str(error))
        print(f"wrote {args.out}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return 0
    print(render(signatures(args.db)))
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(main())
