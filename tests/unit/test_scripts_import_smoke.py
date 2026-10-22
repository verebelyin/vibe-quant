"""Every scripts/**/*.py that imports vibe_quant still imports cleanly.

scripts/paper_reconciliation.py went stale once with no test touching it. One
subprocess loads every such script with run_name != "__main__" (module body only,
no main()), so a removed or renamed vibe_quant name fails here, not at the next use.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_IMPORTS_VQ = re.compile(r"^\s*(from|import) vibe_quant\b", re.MULTILINE)

_LOADER = """
import json, runpy, sys, traceback
failed = {}
for path in sys.argv[1:]:
    try:
        runpy.run_path(path, run_name="__script_smoke__")
    except BaseException:
        failed[path] = traceback.format_exc(limit=-3)
print(json.dumps(failed))
"""


def _scripts() -> list[str]:
    return sorted(
        str(path.relative_to(_ROOT))
        for path in (_ROOT / "scripts").rglob("*.py")
        if _IMPORTS_VQ.search(path.read_text())
    )


def test_every_script_importing_vibe_quant_loads() -> None:
    scripts = _scripts()
    assert "scripts/paper_reconciliation.py" in scripts  # the selector still finds the known case

    proc = subprocess.run(
        [sys.executable, "-c", _LOADER, *scripts],
        cwd=_ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},  # keeps conftest's temp VIBE_QUANT_DB
    )  # fmt: skip

    assert proc.returncode == 0, proc.stderr[-2000:]
    failed = json.loads(proc.stdout.strip().splitlines()[-1])
    assert failed == {}
