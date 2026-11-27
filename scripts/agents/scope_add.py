#!/usr/bin/env python3
"""Record a chief-approved ``SCOPE += <glob>`` for one task agent (vibe-quant-81mv9).

    scripts/agents/scope_add.py --job <job> --agent <agent id> <glob> [<glob> ...]

Appends each glob to ``data/swarm/<job>/scope/<agent>.txt``; the agent's next
``swarm-check.sh ... --bus <job>/bus --bus-agent <agent>`` run adds them to its brief's
``--scope``, so the brief's literal gate command never goes stale. A file outside the brief
scope and this list still FAILs the gate's scope check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", required=True, help="job directory or bare job name")
    parser.add_argument("--agent", required=True, help="the task agent id (brief id)")
    parser.add_argument("globs", nargs="+")
    args = parser.parse_args(argv)
    job = Path(args.job) if "/" in args.job else _ROOT / "data" / "swarm" / args.job
    if not job.is_dir():
        parser.error(f"no job directory {job}")
    path = job / "scope" / f"{args.agent}.txt"
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as fh:
        for glob in args.globs:
            fh.write(f"{glob.strip()}\n")
    print(
        f"SCOPE += {','.join(args.globs)} for {args.agent} ({path}); its next gate run includes it"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
