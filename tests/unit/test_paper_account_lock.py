"""vibe-quant-rw1g8: one active paper session per account, via an OS file lock.

The lock is an advisory ``flock`` on ``<lock dir>/<account id>.lock``. The kernel
drops it when the holding process dies (including SIGKILL), which is why there is
no stale-lock recovery code to test: closing the holder's fd is the whole of it.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import types
from typing import TYPE_CHECKING

import pytest

from vibe_quant.paper.account_lock import (
    AccountBusyError,
    AccountLock,
    account_lock_holder,
    acquire_account_lock,
    default_lock_dir,
)
from vibe_quant.paper.config import BinanceTestnetConfig, ConfigurationError, account_id

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_API_KEY = "KEY-A"
_API_SECRET = "SECRET-XYZ"
_EIO_MARKER = "EIO-TEXT-MARKER-91"


def _account(api_key: str = _API_KEY, *, testnet: bool = True) -> str:
    return account_id(
        BinanceTestnetConfig(api_key=api_key, api_secret=_API_SECRET, testnet=testnet)
    )


def _lock_path(lock_dir: Path, account: str) -> Path:
    return lock_dir / f"{account.replace(':', '_')}.lock"


def _hold(lock_dir: Path, account: str, body: bytes = b"4242 PAPER-OTHER\n") -> int:
    """An independent holder: its own open file description, as a child process has."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(_lock_path(lock_dir, account), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if body:
        os.write(fd, body)
    return fd


class TestAcquireAccountLock:
    def test_default_lock_dir_uses_the_passwd_home_not_home_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """6ub67: the lock dir comes from the passwd database, so a rewritten $HOME
        cannot split one machine's sessions onto two lock files."""
        pw_home = tmp_path / "pw_home"
        monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
        monkeypatch.setattr(
            "vibe_quant.paper.account_lock.pwd.getpwuid",
            lambda uid: types.SimpleNamespace(pw_dir=str(pw_home)),
        )
        assert default_lock_dir() == pw_home / ".vibe-quant" / "paper_locks"
        # A different $HOME is ignored.
        monkeypatch.setenv("HOME", str(tmp_path / "other_home"))
        assert default_lock_dir() == pw_home / ".vibe-quant" / "paper_locks"
        assert default_lock_dir().is_absolute()

    def test_default_lock_dir_fails_closed_without_a_passwd_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No passwd row (KeyError) is a refusal, never a $HOME fallback."""

        def _missing(_uid: int) -> None:
            raise KeyError("no such uid")

        monkeypatch.setattr("vibe_quant.paper.account_lock.pwd.getpwuid", _missing)
        with pytest.raises(ConfigurationError, match="home"):
            default_lock_dir()

    def test_default_lock_dir_fails_closed_on_an_empty_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "vibe_quant.paper.account_lock.pwd.getpwuid",
            lambda uid: types.SimpleNamespace(pw_dir=""),
        )
        with pytest.raises(ConfigurationError, match="home"):
            default_lock_dir()

    def test_without_an_override_the_lock_file_lands_under_the_passwd_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("vibe_quant.paper.account_lock.PAPER_LOCK_DIR", None)
        fake_home = tmp_path / "home"
        pw_home = tmp_path / "pw_home"
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "vibe_quant.paper.account_lock.pwd.getpwuid",
            lambda uid: types.SimpleNamespace(pw_dir=str(pw_home)),
        )
        account = _account()

        lock = acquire_account_lock(account, "PAPER-001")
        try:
            expected = pw_home / ".vibe-quant" / "paper_locks"
            assert _lock_path(expected, account).read_text() == f"{os.getpid()} PAPER-001\n"
            assert account_lock_holder(account) == f"{os.getpid()} PAPER-001"
            # $HOME was never used: the code must not have created its lock dir.
            assert not fake_home.exists()
        finally:
            lock.release()

    def test_writes_pid_and_trader_and_refuses_a_second_holder(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()

        lock = acquire_account_lock(account, "PAPER-001", lock_dir)
        try:
            assert _lock_path(lock_dir, account).read_text() == f"{os.getpid()} PAPER-001\n"

            with pytest.raises(AccountBusyError) as excinfo:
                acquire_account_lock(account, "PAPER-002", lock_dir)

            message = str(excinfo.value)
            assert f"Paper session already running on account {account}" in message
            assert f"pid {os.getpid()}" in message
            assert "trader PAPER-001" in message
            # The identity is a hash: neither credential may reach the message.
            assert _API_KEY not in message
            assert _API_SECRET not in message
        finally:
            lock.release()

    def test_message_names_the_actual_holder(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        fd = _hold(lock_dir, account, b"4242 PAPER-OTHER\n")
        try:
            with pytest.raises(AccountBusyError, match=r"pid 4242, trader PAPER-OTHER"):
                acquire_account_lock(account, "PAPER-001", lock_dir)
        finally:
            os.close(fd)

    def test_unreadable_body_reads_unknown(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        fd = _hold(lock_dir, account, b"")
        try:
            with pytest.raises(AccountBusyError, match=r"pid unknown, trader unknown"):
                acquire_account_lock(account, "PAPER-001", lock_dir)
        finally:
            os.close(fd)

    def test_release_reacquires_and_is_idempotent(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()

        first = acquire_account_lock(account, "PAPER-001", lock_dir)
        first.release()
        first.release()  # releasing twice must not close a reused fd

        second = acquire_account_lock(account, "PAPER-001", lock_dir)
        second.release()

    def test_context_manager_releases(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()

        with acquire_account_lock(account, "PAPER-001", lock_dir) as lock:
            assert isinstance(lock, AccountLock)
            with pytest.raises(AccountBusyError):
                acquire_account_lock(account, "PAPER-002", lock_dir)

        again = acquire_account_lock(account, "PAPER-002", lock_dir)
        again.release()

    def test_died_holder_leaves_no_stale_lock(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()

        fd = _hold(lock_dir, account)
        with pytest.raises(AccountBusyError):
            acquire_account_lock(account, "PAPER-001", lock_dir)

        os.close(fd)  # the kernel drops the flock with the last descriptor
        lock = acquire_account_lock(account, "PAPER-001", lock_dir)
        lock.release()

    def test_accounts_are_independent(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        testnet_key_a = _account(_API_KEY, testnet=True)
        fd = _hold(lock_dir, testnet_key_a)
        try:
            # Same key on the other venue is a different account (testnet vs live).
            live_key_a = acquire_account_lock(_account(_API_KEY, testnet=False), "LIVE-1", lock_dir)
            live_key_a.release()
            # A different key on the same venue too.
            other_key = acquire_account_lock(_account("KEY-B", testnet=True), "PAPER-2", lock_dir)
            other_key.release()
        finally:
            os.close(fd)


class TestAccountLockHolder:
    def test_absent_file_returns_none_and_creates_nothing(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "never_created"

        assert account_lock_holder(_account(), lock_dir) is None
        assert not lock_dir.exists()

    def test_reports_a_held_lock_without_taking_or_releasing_it(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        fd = _hold(lock_dir, account, b"4242 PAPER-OTHER\n")
        try:
            assert account_lock_holder(account, lock_dir) == "4242 PAPER-OTHER"
            # Still held by the foreign fd: the probe must not unlock anything.
            with pytest.raises(AccountBusyError):
                acquire_account_lock(account, "PAPER-001", lock_dir)
        finally:
            os.close(fd)

    def test_reports_unreadable_body_as_unknown(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        fd = _hold(lock_dir, account, b"")
        try:
            assert account_lock_holder(account, lock_dir) == "unknown"
        finally:
            os.close(fd)

    def test_returns_none_for_a_lock_we_hold_and_keeps_it_held(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        lock = acquire_account_lock(account, "PAPER-001", lock_dir)
        try:
            assert account_lock_holder(account, lock_dir) == f"{os.getpid()} PAPER-001"
            with pytest.raises(AccountBusyError):
                acquire_account_lock(account, "PAPER-002", lock_dir)
        finally:
            lock.release()

        assert account_lock_holder(account, lock_dir) is None


def _eio_pread(_fd: int, _length: int, _offset: int) -> bytes:
    raise OSError(errno.EIO, _EIO_MARKER)


def _spy_opened_fds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every fd ``os.open`` hands out while the test runs."""
    real_open: Callable[..., int] = os.open
    opened: list[int] = []

    def _open(*args: object, **kwargs: object) -> int:
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", _open)
    return opened


def _assert_closed(fds: list[int]) -> None:
    assert fds
    for fd in fds:
        with pytest.raises(OSError):  # EBADF: the descriptor was closed
            os.fstat(fd)


class TestUnreadableHolderMetadata:
    """F1: a held lock whose body cannot be read is still busy, holder ``unknown``."""

    def test_acquire_still_refuses_names_unknown_and_closes_the_fd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        fd = _hold(lock_dir, account, b"4242 PAPER-OTHER\n")
        monkeypatch.setattr(os, "pread", _eio_pread)
        opened = _spy_opened_fds(monkeypatch)
        try:
            with caplog.at_level(logging.DEBUG), pytest.raises(AccountBusyError) as excinfo:
                acquire_account_lock(account, "PAPER-001", lock_dir)
        finally:
            os.close(fd)

        assert "pid unknown, trader unknown" in str(excinfo.value)
        assert "PAPER-OTHER" not in str(excinfo.value)  # nothing was read: do not guess
        assert _EIO_MARKER not in caplog.text  # class only, never the exception text
        _assert_closed(opened)

    def test_holder_probe_still_reports_busy_as_unknown_and_closes_the_fd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        fd = _hold(lock_dir, account, b"4242 PAPER-OTHER\n")
        monkeypatch.setattr(os, "pread", _eio_pread)
        opened = _spy_opened_fds(monkeypatch)
        try:
            with caplog.at_level(logging.DEBUG):
                holder = account_lock_holder(account, lock_dir)
        finally:
            os.close(fd)

        assert holder == "unknown"
        assert _EIO_MARKER not in caplog.text
        _assert_closed(opened)

    def test_a_failed_body_write_after_winning_closes_the_fd_and_frees_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock_dir = tmp_path / "locks"
        account = _account()
        opened = _spy_opened_fds(monkeypatch)

        def _boom(_fd: int, _length: int) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        monkeypatch.setattr(os, "ftruncate", _boom)
        with pytest.raises(OSError, match="disk full"):
            acquire_account_lock(account, "PAPER-001", lock_dir)
        _assert_closed(opened)
        monkeypatch.undo()

        lock = acquire_account_lock(account, "PAPER-001", lock_dir)  # not left held
        lock.release()
