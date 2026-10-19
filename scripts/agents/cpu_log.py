#!/usr/bin/env python3
"""Shared CPU/RAM sampler for parallel agent work (perf numbers need the load they ran under).

    cpu_log.py sample --out data/swarm/<job>/cpu.jsonl [--every 15]   # run in background
    cpu_log.py summary --log data/swarm/<job>/cpu.jsonl --since HH:MM [--until HH:MM]

``sample`` appends one JSON line per tick: time, 1-minute load average, free RAM (GB) and the
five busiest processes (pid, %cpu, rss MB, short command). ``summary`` prints the mean / max
load, the minimum free RAM and how often each command label was in the top five over a
window, so a timing can be reported as "MEASURED at load mean X max Y".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def _top(n: int = 5) -> list[dict[str, object]]:
    out = subprocess.run(
        ["ps", "-Ao", "pid=,pcpu=,rss=,command="], capture_output=True, text=True, check=False
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, cpu, rss, cmd = parts
        try:
            rows.append({"pid": int(pid), "cpu": float(cpu.replace(",", ".")), "rss_mb": int(rss) // 1024,
                         "cmd": label(cmd)})
        except ValueError:
            continue
    rows.sort(key=lambda r: float(r["cpu"]), reverse=True)  # type: ignore[arg-type]
    return rows[:n]


def label(cmd: str) -> str:
    """Short, stable label: the script/module a python process runs, else the executable."""
    toks = cmd.split()
    for i, t in enumerate(toks):
        if t == "-m" and i + 1 < len(toks):
            return f"py -m {toks[i + 1]}"
        if t.endswith(".py"):
            return f"py {Path(t).name}"
    return Path(toks[0]).name if toks else "?"


def _free_gb() -> float | None:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    pages = 0
    for line in out.splitlines():
        if line.startswith(("Pages free", "Pages inactive", "Pages speculative")):
            pages += int(line.split(":")[1].strip().rstrip("."))
    return round(pages * 16384 / 2**30, 2) if pages else None


def sample(out: Path, every: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        rec = {"t": time.strftime("%H:%M:%S"), "load1": round(os.getloadavg()[0], 2),
               "free_gb": _free_gb(), "top": _top()}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(every)


def summarize(lines: list[dict[str, object]], since: str, until: str | None) -> dict[str, object]:
    """Mean/max load, min free RAM and top-five frequency over [since, until] (HH:MM[:SS])."""
    sel = [r for r in lines if str(r["t"]) >= since and (until is None or str(r["t"]) <= until)]
    if not sel:
        return {"samples": 0}
    loads = [float(r["load1"]) for r in sel]  # type: ignore[arg-type]
    frees = [float(r["free_gb"]) for r in sel if r.get("free_gb") is not None]  # type: ignore[arg-type]
    seen = Counter(str(p["cmd"]) for r in sel for p in r["top"])  # type: ignore[attr-defined]
    return {"samples": len(sel), "load_mean": round(sum(loads) / len(loads), 2),
            "load_max": max(loads), "free_gb_min": min(frees) if frees else None,
            "top_cmds": seen.most_common(5)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--every", type=float, default=15.0)
    m = sub.add_parser("summary")
    m.add_argument("--log", type=Path, required=True)
    m.add_argument("--since", required=True)
    m.add_argument("--until", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        sample(a.out, a.every)
        return 0
    lines = [json.loads(x) for x in a.log.read_text().splitlines() if x.strip()]
    print(json.dumps(summarize(lines, a.since, a.until)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
