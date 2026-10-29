#!/usr/bin/env python3
"""Per-job memory consolidation: a cheap model proposes add / supersede / retire ops on the board.

    consolidate_memory.py propose --job <job> [--top 40] [--out <job dir>/memory-ops.json]
    consolidate_memory.py apply   --ops <job dir>/memory-ops.json [--as chief]

Run at the orchestrate Land step. ``propose`` shows the model (route ``memory`` in
docs/orchestration/models.json) the job's bus messages and handoffs plus the SHOWN set: the
job's own posts on the persistent board and the ``--top`` posts ranked for the files the job
names. The model returns JSON ops; every op is validated and the file is written only when
all pass. The chief reads it, edits or deletes ops, then ``apply`` re-validates and writes
through bus.py. An op may only supersede or retire an id in the shown set: the model never
removes a fact it was not shown. ``apply`` is idempotent (client ids per op; retiring a
hidden post is a no-op), so a re-run after a crash finishes the batch.
Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

HERE = Path(__file__).resolve().parent
TOPICS = ("gotchas", "findings", "decisions")
MAX_BODY = 1200
MATERIAL_CHARS = 40_000
_FILE_TOKEN = re.compile(r"[\w./-]+\.(?:py|md|ts|tsx|json|yaml|yml|sh|toml)\b")

PROMPT = """You maintain the long-term memory board of a coding-agent swarm (crypto backtesting repo).
A job just finished. Turn what it learned into board changes:
- add: a reusable trap or fact the NEXT agent on a DIFFERENT task needs ("RULE: ... — WHEN: ... — EVIDENCE: ...").
- supersede: an existing post (by id) is wrong or outdated; give the corrected full text.
- retire: an existing post (by id) is stale, a duplicate, or job chatter with no reuse value.
Only ids listed under EXISTING POSTS may be superseded or retired. Prefer few, high-value ops; [] is fine.
Never add progress reports, verdicts, or thanks. Keep each body under 1200 characters.
importance: 3 = money/look-ahead/data-loss trap, 2 = normal trap or decision, 1 = minor.
Reply with ONLY a JSON object, no prose, no code fence:
{"ops": [{"op": "add", "topic": "gotchas|findings|decisions", "importance": 1-3, "body": "..."},
         {"op": "supersede", "id": "<id>", "topic": "...", "importance": 1-3, "body": "..."},
         {"op": "retire", "id": "<id>", "reason": "..."}]}

EXISTING POSTS (id, topic, importance, age):
{shown}

JOB {job} MATERIAL (bus messages, then handoffs):
{material}
"""


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bus = _load("bus")


def job_dir(job: str) -> Path:
    raw = os.environ.get("SWARM_JOBS")
    return (Path(raw) if raw else bus._repo_root() / "data" / "swarm") / job


def job_material(jdir: Path, limit: int = MATERIAL_CHARS) -> str:
    """The job's bus messages (one line each, bodies cut to 600 chars) then its handoff files, cut to ``limit``."""
    lines = [bus._one_line(m, 600) for m in bus._all(jdir / "bus")]
    for f in sorted((jdir / "handoffs").glob("*.md")):
        lines.append(f"--- handoff {f.name}\n{f.read_text(encoding='utf-8', errors='replace')}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n[... cut]"


def shown_posts(job: str, material: str, top: int, now: float) -> list[dict[str, object]]:
    """Visible global posts the model may change: the job's own posts first, then the ``top`` ranked for its files."""
    msgs = bus._all(bus.global_board_dir())
    hidden = bus.hidden_posts(msgs)
    own = [
        m for m in msgs
        if m.get("to") == "board" and m.get("job") == job and m.get("topic") != "needs-user" and str(m["id"]) not in hidden
    ]
    files = sorted(set(_FILE_TOKEN.findall(material)))
    ranked = [m for _, m in bus.ranked_posts(msgs, top, now, files, set(), bus.comparability_breaks())]
    seen: set[str] = set()
    out = []
    for m in own + ranked:
        if str(m["id"]) not in seen:
            seen.add(str(m["id"]))
            out.append(m)
    return out


def render_shown(posts: list[dict[str, object]]) -> str:
    return "\n".join(
        f"[{m['id']}] #{m.get('topic')} i{bus.importance_of(m)} {bus._age_of(m)}: {bus._first_chars(m['body'], 400)}"
        for m in posts
    ) or "(none)"


def validate(ops: object, shown: set[str]) -> list[str]:
    """Every problem with ``ops`` (a list of op dicts); [] when all may be applied."""
    if not isinstance(ops, list):
        return ["ops is not a list"]
    errors: list[str] = []
    touched: set[str] = set()
    for i, op in enumerate(ops):
        where = f"op {i}"
        if not isinstance(op, dict):
            errors.append(f"{where}: not an object")
            continue
        kind = op.get("op")
        if kind not in ("add", "supersede", "retire"):
            errors.append(f"{where}: op must be add|supersede|retire, got {kind!r}")
            continue
        if kind in ("supersede", "retire"):
            ref = op.get("id")
            if not isinstance(ref, str) or ref not in shown:
                errors.append(f"{where}: {kind} id {ref!r} was not shown to the model")
            elif ref in touched:
                errors.append(f"{where}: id {ref} changed twice")
            else:
                touched.add(ref)
        if kind == "retire":
            if not isinstance(op.get("reason"), str) or not str(op["reason"]).strip():
                errors.append(f"{where}: retire needs a reason")
            continue
        if op.get("topic") not in TOPICS:
            errors.append(f"{where}: topic must be one of {TOPICS}, got {op.get('topic')!r}")
        imp = op.get("importance")
        if not isinstance(imp, int) or isinstance(imp, bool) or not 1 <= imp <= 3:
            errors.append(f"{where}: importance must be 1-3, got {imp!r}")
        body = op.get("body")
        if not isinstance(body, str) or not body.strip():
            errors.append(f"{where}: empty body")
        elif len(body) > MAX_BODY:
            errors.append(f"{where}: body has {len(body)} chars > {MAX_BODY}")
    return errors


def parse_ops(text: str) -> object:
    """The ``ops`` list of the model's JSON reply (a code fence around it is tolerated); ValueError otherwise."""
    raw = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.S)
    data = json.loads(fence.group(1) if fence else raw)
    if not isinstance(data, dict) or "ops" not in data:
        raise ValueError('reply is not an object with "ops"')
    return data["ops"]


def route() -> tuple[str, str | None]:
    """(model, effort) of the first available claude-runtime model on route ``memory``."""
    models = _load("models")
    data = models.load(models._DEFAULT_FILE)
    for key in models._routes(data).get("memory", []):
        spec = models._models(data).get(key)
        if spec is None or spec.get("runtime") != "claude" or not models.availability(spec)[0]:
            continue
        target = spec.get("target")
        if isinstance(target, dict) and target.get("model"):
            effort = target.get("effort")
            return str(target["model"]), str(effort) if effort else None
    raise SystemExit("consolidate_memory: no available claude model on route 'memory' (models.py route memory)")


def call_claude(prompt: str) -> str:
    """``claude -p`` on the routed model, from a neutral cwd (no CLAUDE.md in its context)."""
    model, effort = route()
    cmd = ["claude", "-p", "--model", model, "--output-format", "text"]
    if effort:
        cmd += ["--effort", effort]
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd=tempfile.gettempdir(), check=False
    )
    if proc.returncode != 0:
        raise SystemExit(f"consolidate_memory: claude -p exited {proc.returncode}: {proc.stderr.strip()[:500]}")
    return proc.stdout


def propose(job: str, top: int, out: Path | None, ask: Callable[[str], str] = call_claude) -> int:
    jdir = job_dir(job)
    if not (jdir / "bus" / "messages.jsonl").exists():
        print(f"consolidate_memory: no job bus at {jdir / 'bus'}", file=sys.stderr)
        return 2
    material = job_material(jdir)
    shown = shown_posts(job, material, top, time.time())
    reply = ask(PROMPT.replace("{shown}", render_shown(shown)).replace("{job}", job).replace("{material}", material))
    try:
        ops = parse_ops(reply)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        print(f"consolidate_memory: unusable model reply ({exc}); nothing written:\n{reply[:2000]}", file=sys.stderr)
        return 1
    errors = validate(ops, {str(m["id"]) for m in shown})
    if errors:
        print("consolidate_memory: rejected, nothing written:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    assert isinstance(ops, list)
    target = out or jdir / "memory-ops.json"
    doc = {"job": job, "shown": [str(m["id"]) for m in shown], "ops": ops}
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for op in ops:
        print(f"{op['op']:9} {op.get('id', '')} {op.get('reason') or bus._first_chars(op.get('body', ''), 100)}")
    print(f"{len(ops)} op(s) from {len(shown)} shown post(s) -> {target}; review it, then: consolidate_memory.py apply --ops {target}")
    return 0


def apply(ops_file: Path, me: str) -> int:
    doc = json.loads(ops_file.read_text(encoding="utf-8"))
    job, ops, shown = doc.get("job"), doc.get("ops"), doc.get("shown")
    if not isinstance(job, str) or not isinstance(shown, list):
        print(f"consolidate_memory: {ops_file} lacks job/shown; nothing written", file=sys.stderr)
        return 1
    errors = validate(ops, {str(s) for s in shown})
    if errors:
        print("consolidate_memory: rejected, nothing written:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    assert isinstance(ops, list)
    gdir = bus.global_board_dir()
    for op in ops:
        # keyed by the op's content, not its index: a job landing in phases applies more than once
        digest = hashlib.sha1(json.dumps(op, sort_keys=True).encode()).hexdigest()[:12]
        cid = f"consolidate-{job}-{digest}"
        extra: dict[str, object] = {"importance": op.get("importance"), "job": job}
        if op["op"] == "add":
            msg = bus.post(gdir, me, "board", "info", op["body"], topic=op["topic"], extra=extra, client_id=cid)
        elif op["op"] == "supersede":
            msg = bus.supersede(gdir, me, op["id"], op["topic"], op["body"], client_id=cid, extra=extra)
        else:
            msg = bus.retire(gdir, me, op["id"], op["reason"])
        print(bus.fmt(msg) if msg else f"(already hidden: {op['id']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="consolidate_memory.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--job", required=True)
    p.add_argument("--top", type=int, default=40, help="ranked global posts shown besides the job's own")
    p.add_argument("--out", type=Path, help="default <job dir>/memory-ops.json")
    a = sub.add_parser("apply")
    a.add_argument("--ops", type=Path, required=True)
    a.add_argument("--as", dest="as_", default="chief")
    args = ap.parse_args(argv)
    if args.cmd == "propose":
        return propose(args.job, args.top, args.out)
    return apply(args.ops, args.as_)


if __name__ == "__main__":
    raise SystemExit(main())
