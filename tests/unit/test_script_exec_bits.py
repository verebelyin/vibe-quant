"""Every tracked script with a shebang is executable in git (mode 100755).

Workers call `scripts/agents/<x>.py` directly; mutate.py and review_pack.py at mode
100644 failed with "Permission denied" (exit 126) in 6 worker threads (10-17..10-18).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_every_tracked_shebang_script_is_executable() -> None:
    out = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-s", "scripts"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    missing = []
    for line in out.splitlines():
        meta, rel = line.split("\t", 1)
        if meta.split()[0] != "100644":
            continue
        with (_ROOT / rel).open("rb") as fh:
            if fh.read(2) == b"#!":
                missing.append(rel)
    assert missing == [], f"git update-index --chmod=+x {' '.join(missing)}"
