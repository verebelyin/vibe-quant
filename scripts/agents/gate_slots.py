"""Gate slots: concurrent swarm-check gates share the CPU.

A gate registers a pid-named file in a shared dir while its pytest step runs; every gate
sizes its xdist workers from the number of live slots at that moment (``xdist_share``), so
N gates started together each take cpus // N instead of all taking ``-n auto``. A slot whose
pid is dead (a gate killed -9), or whose file is older than any gate's test step can run
(STALE_AFTER: the pid may have been reused by an unrelated process), is ignored and removed.
SIGTERM raises SystemExit inside the slot so ``finally`` removes it. Slots only size xdist
workers (speed): a wrong count makes a gate slower, never wrong. Stdlib only.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from types import FrameType

SLOTS_SUBDIR = "data/swarm/_gate-slots"
MIN_WORKERS = 2
STALE_AFTER = 1800  # seconds; = swarm_check.TEST_TIMEOUT (a unit test pins the equality)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def live_slots(slots_dir: Path, stale_after: float = STALE_AFTER) -> int:
    """Gates with a live slot; removes the slot files of dead pids and of files older than
    ``stale_after`` seconds (whatever their pid: it may belong to an unrelated process now)."""
    live = 0
    now = time.time()
    for slot in slots_dir.glob("*"):
        if not slot.name.isdigit():
            continue
        try:
            fresh = now - slot.stat().st_mtime <= stale_after
        except OSError:  # removed by a peer meanwhile
            continue
        if fresh and pid_alive(int(slot.name)):
            live += 1
        else:
            with contextlib.suppress(OSError):
                slot.unlink()
    return live


def _raise_exit(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def gate_slot(slots_dir: Path, stale_after: float = STALE_AFTER) -> Iterator[int]:
    """Hold a slot for the ``with`` body; yields the live gates now, this one included.
    Registers BEFORE counting, so gates starting together mostly see each other. The count is
    lock-free: two gates entering at the same instant may briefly oversubscribe (today's
    behaviour). SIGTERM raises SystemExit in the body so the slot is removed."""
    slot = slots_dir / str(os.getpid())
    previous: Callable[[int, FrameType | None], object] | int | None = None
    installed = False
    try:
        previous = signal.signal(signal.SIGTERM, _raise_exit)
        installed = True
    except ValueError:  # not the main thread: no handler, finally still covers exceptions
        pass
    try:
        slots_dir.mkdir(parents=True, exist_ok=True)
        slot.write_text("", encoding="utf-8")
        yield max(live_slots(slots_dir, stale_after), 1)
    finally:
        with contextlib.suppress(OSError):
            slot.unlink()
        if installed and previous is not None:
            signal.signal(signal.SIGTERM, previous)


def xdist_share(cpus: int, running: int) -> int | None:
    """Workers per gate: an even split of the cores (floor MIN_WORKERS); None for a lone gate."""
    if running <= 1:
        return None
    return max(MIN_WORKERS, cpus // running)
