"""One active paper session per account and per trader id: OS advisory locks.

The gate is ``fcntl.flock`` on ``<lock dir>/<account id>.lock`` held for the whole
process life. The lock dir is per machine (the OS user's home from the passwd
database, not ``$HOME``), so a main checkout, a worktree and a ui-check stack
contend on the same file. The kernel drops it when the holder dies (``SIGKILL`` included), so
there is no stale-lock file to time out or clean up.

The CLI takes a second flock per trader id (``<lock dir>/trader_<id>.lock``): a
second process for the same id (another key, or testnet beside live) would share
the one command queue and the one event log of the running session.

The API's probe (:func:`account_lock_holder`) is advisory only: between the probe
and the child process taking the lock another start can win (TOCTOU). The CLI
lock taken in ``run_with_config`` is the real gate.
"""

from __future__ import annotations

import fcntl
import logging
import os
import pwd
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.paper.config import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def default_lock_dir() -> Path:
    """``<OS user's home>/.vibe-quant/paper_locks``: absolute, machine-wide, at call time.

    The home comes from the passwd database (``pwd.getpwuid``), never ``$HOME``: a
    session started with a rewritten ``HOME`` must still share the machine's lock
    file. Not anchored at the checkout either: sessions from main, a worktree or a
    ui-check stack share one real account, so they must share one lock file.

    Raises:
        ConfigurationError: The OS user has no passwd entry or no home dir. Fail
            closed: there is no ``Path.home()`` fallback.
    """
    try:
        home = pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        raise ConfigurationError(
            "cannot resolve the OS user's home dir for the paper lock"
        ) from None
    if not home:
        raise ConfigurationError("cannot resolve the OS user's home dir for the paper lock")
    return Path(home) / ".vibe-quant" / "paper_locks"


#: Test override; ``None`` = :func:`default_lock_dir`. Both helpers resolve it at
#: call time, so a monkeypatch reaches the CLI and the API probe alike.
PAPER_LOCK_DIR: Path | None = None


class AccountBusyError(ConfigurationError):
    """Another session holds this account's paper lock."""


class TraderBusyError(ConfigurationError):
    """Another session already runs for this trader id."""


def _lock_dir(lock_dir: Path | None) -> Path:
    if lock_dir is not None:
        return lock_dir
    return default_lock_dir() if PAPER_LOCK_DIR is None else PAPER_LOCK_DIR


def _lock_path(account: str, lock_dir: Path | None) -> Path:
    """``binance:testnet:<hex>`` -> ``<dir>/binance_testnet_<hex>.lock`` (no ``:``)."""
    return _lock_dir(lock_dir) / f"{account.replace(':', '_')}.lock"


def _trader_lock_path(trader_id: str, lock_dir: Path | None) -> Path:
    """``PAPER-001`` -> ``<dir>/trader_PAPER-001.lock``."""
    return _lock_dir(lock_dir) / f"trader_{trader_id}.lock"


def _read_body(fd: int) -> str:
    """The holder's ``"<pid> <trader_id>"`` line, without moving the fd offset.

    Best-effort: the caller already knows the lock is held, so an unreadable body
    reads as empty (-> ``unknown``) instead of masking the refusal. Class only in
    the log: this runs on the money path.
    """
    try:
        return os.pread(fd, 4096, 0).decode("utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning("account lock holder unreadable (%s)", type(exc).__name__)
        return ""


def _holder_fields(body: str) -> tuple[str, str]:
    parts = body.split(None, 1)
    if not parts:
        return "unknown", "unknown"
    return parts[0], parts[1] if len(parts) > 1 else "unknown"


class AccountLock:
    """A held account lock; :meth:`release` unlocks and closes (idempotent)."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def release(self) -> None:
        fd = self._fd
        if fd < 0:
            return
        self._fd = -1
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> AccountLock:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def _acquire(path: Path, body: str, busy: Callable[[str, str], ConfigurationError]) -> AccountLock:
    """Take ``path``'s flock and write ``body``; ``busy(pid, holder)`` builds the refusal.

    One flock body for both locks; only ``path``, the body and the refusal text
    differ. ``busy`` receives the foreign holder's body fields (``unknown`` when
    the body cannot be read).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid, holder = _holder_fields(_read_body(fd))
            raise busy(pid, holder) from None

        # Only after winning the lock: truncating a foreign holder's body would erase
        # the pid/identity the next refusal prints.
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, body.encode())
    except BaseException:
        os.close(fd)  # every failure path: never leak the fd, never keep the lock
        raise
    return AccountLock(fd)


def acquire_account_lock(account: str, trader_id: str, lock_dir: Path | None = None) -> AccountLock:
    """Take the account's paper lock or raise :class:`AccountBusyError`.

    Raises:
        AccountBusyError: Another process holds the lock; its pid/trader_id (read
            best-effort from the lock body) are named, never a credential.
    """

    def _busy(pid: str, trader: str) -> AccountBusyError:
        return AccountBusyError(
            f"Paper session already running on account {account} "
            f"(pid {pid}, trader {trader}); stop it first"
        )

    return _acquire(_lock_path(account, lock_dir), f"{os.getpid()} {trader_id}\n", _busy)


def acquire_trader_lock(trader_id: str, account: str, lock_dir: Path | None = None) -> AccountLock:
    """Take this trader id's paper lock or raise :class:`TraderBusyError`.

    A trader id runs one session at a time on any key and on either venue: a
    second process (another key, or testnet beside live) would share the running
    session's command queue and event log.

    Raises:
        TraderBusyError: Another process holds the lock; the holder's pid and
            account (read best-effort from the lock body) are named, never a
            credential.
    """

    def _busy(pid: str, holder: str) -> TraderBusyError:
        return TraderBusyError(
            f"Paper session already running for trader {trader_id} "
            f"(pid {pid}, account {holder}); stop it first"
        )

    return _acquire(_trader_lock_path(trader_id, lock_dir), f"{os.getpid()} {account}\n", _busy)


def _holder_of(path: Path) -> str | None:
    """The body text of whoever holds the lock at ``path`` (or ``"unknown"``); ``None`` if free.

    Creates nothing and never leaves the lock held: a free lock is unlocked and
    closed before returning.
    """
    if not path.exists():
        return None
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_body(fd) or "unknown"
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def account_lock_holder(account: str, lock_dir: Path | None = None) -> str | None:
    """Return the account lock holder's body text (or ``"unknown"``), or ``None`` if free."""
    return _holder_of(_lock_path(account, lock_dir))


def trader_lock_holder(trader_id: str, lock_dir: Path | None = None) -> str | None:
    """Return the trader lock holder's body text (or ``"unknown"``), or ``None`` if free."""
    return _holder_of(_trader_lock_path(trader_id, lock_dir))
