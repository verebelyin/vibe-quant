"""The local Vercel Board foundation: pinned install, `boardctl.py up|down`, the `vb` launcher.

Every outcome is checked through the real `vb` binary against a throwaway `vb serve` on a
tmp dir + free port (``board_server`` in conftest, or a test's own server where the test is
about starting and stopping one). Nothing here can reach the real board: BOARD_URL is
pinned per test and every subprocess gets its own XDG_CONFIG_HOME.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.unit.conftest import UNROUTABLE_BOARD_URL, board_env, free_port

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from tests.unit.conftest import BoardServer

_ROOT = Path(__file__).resolve().parents[2]
_AGENTS = _ROOT / "scripts" / "agents"
_LAUNCHER = _AGENTS / "bin" / "vb"
_SECRET = "API_KEY=abcdefghijklmnop1234"
_ENV_DUMP = "\n".join(f"VAR_{i}=value{i}" for i in range(5))


@pytest.fixture(scope="module")
def boardctl() -> ModuleType:
    spec = importlib.util.spec_from_file_location("boardctl", _AGENTS / "boardctl.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctl(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_AGENTS / "boardctl.py"), *args],
        env=env, capture_output=True, text=True, timeout=600, check=False,
    )


def _launch(
    *args: str, env: dict[str, str], stdin: str | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_LAUNCHER), *args], input=stdin, env=env, cwd=cwd,
        capture_output=True, text=True, timeout=60, check=False,
    )


class _OwnBoard:
    """A board this test starts and stops itself (not the shared session one)."""

    def __init__(self, root: Path) -> None:
        self.port = free_port()
        self.dir = root / "data"
        self.url = f"http://127.0.0.1:{self.port}/api/board"
        self.env = board_env(root / "xdg", self.url)

    def up(self) -> subprocess.CompletedProcess[str]:
        return _ctl("up", "--dir", str(self.dir), "--port", str(self.port), "--no-inspect", env=self.env)

    def down(self) -> subprocess.CompletedProcess[str]:
        return _ctl("down", "--dir", str(self.dir), env=self.env)

    def pid(self) -> int:
        return int((self.dir / "serve.pid").read_text().strip())


@pytest.fixture
def own_board(tmp_path: Path) -> Iterator[_OwnBoard]:
    board = _OwnBoard(tmp_path)
    try:
        yield board
    finally:
        board.down()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _serve_pids(data: Path) -> list[int]:
    """Every live `vb serve` process on ``data``."""
    rows = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=True).stdout
    return [int(r.split()[0]) for r in rows.splitlines() if " serve " in r and f"--dir {data} " in r]


def _text_of(board: BoardServer, msg_id: str) -> str:
    got = board.vb("read", msg_id, "--json")
    assert got.returncode == 0, got.stderr
    return str(json.loads(got.stdout)["message"]["text"])


def _assert_refused(got: subprocess.CompletedProcess[str], kind: str) -> None:
    """Exit 2, nothing on stdout, one stderr line naming the argument kind and never the value."""
    assert (got.returncode, got.stdout) == (2, ""), got.stderr
    assert f"the {kind} looks like a secret" in got.stderr and "refused" in got.stderr
    assert len(got.stderr.strip().splitlines()) == 1 and _SECRET not in got.stderr


def _channel_missing(board: BoardServer, channel: str) -> bool:
    got = board.vb("read", channel)
    return got.returncode == 1 and "Couldn't find channel" in got.stderr


# ---------------------------------------------------------------- up / down


class TestUpDown:
    def test_up_starts_a_server_and_a_second_up_reuses_it(self, own_board: _OwnBoard) -> None:
        first = own_board.up()
        assert first.returncode == 0, first.stdout + first.stderr
        assert f"board up at {own_board.url}" in first.stdout
        pid = own_board.pid()
        assert _alive(pid)

        who = _launch("whoami", env={**own_board.env, "BOARD_PROFILE": "x"})
        assert who.returncode == 0, who.stderr
        assert who.stdout.startswith("x via ")

        second = own_board.up()
        assert second.returncode == 0, second.stdout + second.stderr
        assert f"board already up at {own_board.url}" in second.stdout
        assert own_board.pid() == pid and _alive(pid)

    @pytest.mark.slow(reason="two real `vb serve` starts and one stop: ~2.5 s alone")
    def test_restart_keeps_the_data(self, own_board: _OwnBoard) -> None:
        assert own_board.up().returncode == 0
        posted = _launch("post", "chat", "hello restart", env={**own_board.env, "BOARD_PROFILE": "x"})
        assert posted.returncode == 0, posted.stderr
        old_pid = own_board.pid()

        down = own_board.down()
        assert down.returncode == 0, down.stdout + down.stderr
        assert not _alive(old_pid)
        assert not (own_board.dir / "serve.pid").exists()
        assert _launch("whoami", env=own_board.env).returncode != 0

        assert own_board.up().returncode == 0
        read = _launch("read", "chat", env=own_board.env)
        assert read.returncode == 0, read.stderr
        assert "hello restart" in read.stdout

    @pytest.mark.slow(reason="holds the lock 1 s, then a real `vb serve` start and stop: ~1.5 s alone, >2 s under the gate")
    def test_up_waits_for_the_dir_lock_before_starting(self, own_board: _OwnBoard) -> None:
        own_board.dir.mkdir(parents=True)
        with open(own_board.dir / "boardctl.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            proc = subprocess.Popen(
                [sys.executable, str(_AGENTS / "boardctl.py"), "up", "--dir", str(own_board.dir),
                 "--port", str(own_board.port), "--no-inspect"],
                env=own_board.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )  # fmt: skip
            try:
                with pytest.raises(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
                assert not (own_board.dir / "serve.pid").exists()
                assert not (own_board.dir / "serve.log").exists()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, out + err
        assert _alive(own_board.pid())

    @pytest.mark.slow(reason="two racing `boardctl up` + a real `vb serve` start and stop: ~1 s alone, >2 s under the gate")
    def test_two_racing_ups_start_one_server_and_down_stops_it(self, own_board: _OwnBoard) -> None:
        with ThreadPoolExecutor(2) as pool:
            ups = list(pool.map(lambda _: own_board.up(), range(2)))
        assert [u.returncode for u in ups] == [0, 0], [u.stdout + u.stderr for u in ups]
        assert sorted(u.stdout.split(" at ")[0] for u in ups) == ["board already up", "board up"]
        pid = own_board.pid()
        assert _serve_pids(own_board.dir) == [pid]
        assert own_board.down().returncode == 0
        assert _serve_pids(own_board.dir) == []
        assert not (own_board.dir / "serve.pid").exists()

    def test_port_is_free_while_old_connections_sit_in_time_wait(self, boardctl: ModuleType) -> None:
        """`up` right after `down`: the server closed its connections first, so they sit in TIME_WAIT."""
        with socket.socket() as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = srv.accept()
                accepted.close()  # server side closes first -> TIME_WAIT on ``port``
                client.recv(1)
            assert boardctl._port_free(port) is False  # still listening
        assert boardctl._port_free(port) is True

    def test_down_with_nothing_running_exits_zero(self, own_board: _OwnBoard) -> None:
        assert own_board.down().returncode == 0

    def test_up_names_a_port_another_program_holds(
        self, boardctl: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # In-process so the whoami probe's bound (3 s; the squatter never answers HTTP) can shrink.
        monkeypatch.setattr(boardctl, "_WHOAMI_TIMEOUT_S", 0.5)
        port = free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
            squatter.bind(("127.0.0.1", port))
            squatter.listen(1)
            assert boardctl.up(tmp_path / "data", port, inspect=False) != 0
        out = capsys.readouterr()
        assert f"port {port} is in use" in out.err
        assert not (tmp_path / "data" / "serve.pid").exists()

    def test_up_refuses_node_older_than_22_18(
        self, boardctl: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(boardctl, "node_version", lambda: (22, 17, 9))
        port = free_port()
        assert boardctl.up(tmp_path / "data", port, inspect=False) != 0
        out = capsys.readouterr()
        assert "22.18" in out.out + out.err and "22.17.9" in out.out + out.err
        assert not (tmp_path / "data" / "serve.pid").exists()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))  # nothing started on it

    @pytest.mark.parametrize(
        ("version", "ok"),
        [(None, False), ((22, 17, 9), False), ((22, 18, 0), True), ((23, 0, 0), True), ((21, 99, 0), False)],
    )
    def test_node_version_edges(
        self, boardctl: ModuleType, version: tuple[int, int, int] | None, ok: bool
    ) -> None:
        assert (boardctl.node_problem(version) is None) is ok


# ---------------------------------------------------------------- launcher


class TestLauncher:
    def test_stdin_text_is_posted_unchanged(self, board_server: BoardServer) -> None:
        # No trailing newline: the real `vb` drops one from stdin text (measured on 0.53.0).
        text = "first line\n  second line, kept exactly  \tend  "
        got = board_server.vb("post", "launch-stdin", "-", profile="x", stdin=text)
        assert got.returncode == 0, got.stderr
        msg_id = got.stdout.strip()
        assert msg_id.isdigit()
        assert _text_of(board_server, msg_id) == text

    def test_a_failing_command_keeps_its_code_and_stderr(self, board_server: BoardServer) -> None:
        got = board_server.vb("read", "no-such-channel")
        assert got.returncode == 1
        assert "Couldn't find channel no-such-channel" in got.stderr

    def test_works_from_any_directory(self, board_server: BoardServer, tmp_path: Path) -> None:
        got = _launch("whoami", env={**board_server.env, "BOARD_PROFILE": "x"}, cwd=tmp_path)
        assert got.returncode == 0, got.stderr
        assert got.stdout.startswith("x via ")


def _send(board: BoardServer, how: str, channel: str, text: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    if how == "arg":
        return board.vb("post", channel, text, profile="x")
    if how == "file":
        f = tmp_path / "note.md"
        f.write_text(text)
        return board.vb("post", channel, "--file", str(f), profile="x")
    return board.vb("post", channel, "-", profile="x", stdin=text)


_HOWS = ["arg", "file", "stdin"]


class TestSecretScan:
    @pytest.mark.parametrize("how", _HOWS)
    @pytest.mark.parametrize("secret", [_SECRET, _ENV_DUMP], ids=["assign", "env-dump"])
    def test_secret_text_is_refused_and_nothing_sent(
        self, board_server: BoardServer, tmp_path: Path, how: str, secret: str,
        request: pytest.FixtureRequest,
    ) -> None:
        channel = f"sec-{request.node.callspec.id}".replace("[", "").replace("]", "")
        got = _send(board_server, how, channel, secret, tmp_path)
        assert got.returncode == 2
        assert got.stdout == ""
        assert "secret" in got.stderr and len(got.stderr.strip().splitlines()) == 1
        assert _channel_missing(board_server, channel)

    @pytest.mark.parametrize("how", _HOWS)
    def test_ordinary_text_mentioning_token_is_posted(
        self, board_server: BoardServer, tmp_path: Path, how: str
    ) -> None:
        text = f"rotate the token tomorrow ({how})"
        got = _send(board_server, how, f"plain-{how}", text, tmp_path)
        assert got.returncode == 0, got.stderr
        assert _text_of(board_server, got.stdout.strip()) == text

    def test_reply_and_edit_are_scanned(self, board_server: BoardServer) -> None:
        root = board_server.vb("post", "sec-reply", "root", profile="x")
        assert root.returncode == 0, root.stderr
        msg_id = root.stdout.strip()
        assert board_server.vb("reply", msg_id, _SECRET, profile="x").returncode == 2
        assert board_server.vb("edit", msg_id, _SECRET, profile="x").returncode == 2
        assert board_server.vb("--as", "x", "edit", msg_id, "-", stdin=_SECRET).returncode == 2
        assert _text_of(board_server, msg_id) == "root"

    def test_read_and_whoami_are_never_scanned(self, board_server: BoardServer) -> None:
        read = board_server.vb("read", _SECRET)
        assert read.returncode == 1 and "Couldn't find channel" in read.stderr

    @pytest.mark.parametrize("form", ["label", "label-flag", "label-flag-eq"])
    def test_a_secret_label_value_is_refused_and_nothing_written(
        self, board_server: BoardServer, form: str
    ) -> None:
        root = board_server.vb("post", f"sec-lbl-{form}", "labelled", "--label", "keep=yes", profile="x")
        assert root.returncode == 0, root.stderr
        msg_id = root.stdout.strip()
        args = {
            "label": ("label", msg_id, "ok=fine", f"note={_SECRET}"),
            "label-flag": ("reply", msg_id, "plain reply", "--label", f"note={_SECRET}"),
            "label-flag-eq": ("reply", msg_id, "plain reply", f"--label=note={_SECRET}"),
        }[form]
        got = board_server.vb(*args, profile="x")
        _assert_refused(got, "label value")
        thread = json.loads(board_server.vb("read", msg_id, "--json").stdout)
        assert thread["message"]["labels"] == {"keep": "yes"}
        assert thread["replies"] == []

    def test_a_plain_label_is_written(self, board_server: BoardServer) -> None:
        msg_id = board_server.vb("post", "sec-lbl-ok", "labelled", profile="x").stdout.strip()
        assert board_server.vb("label", msg_id, "note=rotate the token", profile="x").returncode == 0
        got = json.loads(board_server.vb("read", msg_id, "--json").stdout)
        assert got["message"]["labels"] == {"note": "rotate the token"}

    @pytest.mark.parametrize(
        ("args", "kind"),
        [
            (("post", "{ch}", "plain", "--idempotency-key", _SECRET), "idempotency key"),
            (("--avatar", "https://example.invalid/" + _SECRET, "post", "{ch}", "plain"), "avatar"),
            (("tasks", "create", "{ch}", _SECRET), "text"),
        ],
        ids=["idempotency-key", "avatar", "tasks-create"],
    )
    def test_other_secret_values_are_refused(
        self, board_server: BoardServer, args: tuple[str, ...], kind: str, request: pytest.FixtureRequest
    ) -> None:
        channel = "sec-" + request.node.callspec.id
        got = board_server.vb(*(a.format(ch=channel) for a in args), profile="x")
        _assert_refused(got, kind)
        assert _channel_missing(board_server, channel)

    def test_a_secret_text_attachment_is_refused(self, board_server: BoardServer, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text(_SECRET)
        got = board_server.vb("post", "sec-attach", "see file", "--attach", str(tmp_path / "a.txt"), profile="x")
        _assert_refused(got, "attachment")
        assert _channel_missing(board_server, "sec-attach")

    @pytest.mark.parametrize("content", [b"\x89PNG\0\0binary", b"a" * ((1 << 20) + 1)], ids=["binary", "over-1mb"])
    def test_an_unscannable_attachment_is_refused(
        self, board_server: BoardServer, tmp_path: Path, content: bytes, request: pytest.FixtureRequest
    ) -> None:
        (tmp_path / "blob").write_bytes(content)
        channel = "sec-blob-" + request.node.callspec.id
        got = board_server.vb("post", channel, "see file", "--attach", str(tmp_path / "blob"), profile="x")
        assert (got.returncode, got.stdout) == (2, "")
        assert "binary or over 1 MB" in got.stderr and len(got.stderr.strip().splitlines()) == 1
        assert _channel_missing(board_server, channel)

    def test_a_plain_text_attachment_is_posted_unchanged(self, board_server: BoardServer, tmp_path: Path) -> None:
        body = "rotate the token tomorrow\nline two\n"
        (tmp_path / "ok.txt").write_text(body)
        got = board_server.vb("post", "plain-attach", "see file", "--attach", str(tmp_path / "ok.txt"), "--json", profile="x")
        assert got.returncode == 0, got.stderr
        att = json.loads(got.stdout)["message"]["attachments"][0]
        assert board_server.vb("download", str(att["id"]), "-").stdout == body

    def test_run_vb_scans_for_python_callers(
        self, boardctl: ModuleType, board_server: BoardServer
    ) -> None:
        refused = boardctl.run_vb(["post", "sec-run-vb", "-"], stdin=_SECRET, env=board_server.env)
        assert refused.returncode == 2 and refused.stdout == ""
        assert _channel_missing(board_server, "sec-run-vb")
        _assert_refused(boardctl.run_vb(["post", "sec-run-vb", "x", "--label", f"k={_SECRET}"], env=board_server.env), "label value")
        assert _channel_missing(board_server, "sec-run-vb")
        sent = boardctl.run_vb(
            ["--as", "x", "post", "sec-run-vb", "-"], stdin="ok text", env=board_server.env
        )
        assert sent.returncode == 0, sent.stderr
        assert _text_of(board_server, sent.stdout.strip()) == "ok text"


class TestUrl:
    """`boardctl.py url` prints the board URL a client defaults to (cmd-task.sh exports it)."""

    @pytest.mark.parametrize(
        ("args", "url"),
        [((), "http://127.0.0.1:4100/api/board"), (("--port", "4196"), "http://127.0.0.1:4196/api/board")],
    )
    def test_prints_the_default_url_and_starts_nothing(self, tmp_path: Path, args: tuple[str, ...], url: str) -> None:
        got = _ctl("url", *args, env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path)})
        assert (got.returncode, got.stdout, got.stderr) == (0, url + "\n", "")
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- install + isolation


class TestBoardHome:
    def test_honours_vq_board_home(
        self, boardctl: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VQ_BOARD_HOME", str(tmp_path / "home"))
        assert boardctl.board_home() == tmp_path / "home"

    def test_changes_with_the_lock_file_bytes(
        self, boardctl: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VQ_BOARD_HOME", raising=False)
        pkg = tmp_path / "board"
        pkg.mkdir()
        monkeypatch.setattr(boardctl, "BOARD_PKG_DIR", pkg)
        (pkg / "package-lock.json").write_bytes(b'{"a": 1}\n')
        first = boardctl.board_home()
        (pkg / "package-lock.json").write_bytes(b'{"a": 2}\n')
        second = boardctl.board_home()
        assert first != second
        assert first.parent == second.parent == Path.home() / ".cache" / "vibe-quant" / "board"
        assert len(first.name) == 12 and int(first.name, 16) >= 0

    def test_the_pin_is_exact(self) -> None:
        pkg = json.loads((_AGENTS / "board" / "package.json").read_text())
        assert pkg["dependencies"] == {"@vercel/board": "0.53.0"}
        assert pkg["private"] is True


class TestBoardEnvGuard:
    def test_no_board_env_in_a_unit_test_body(self) -> None:
        assert os.environ.get("BOARD_URL") == UNROUTABLE_BOARD_URL
        for name in ("BOARD_PROFILE", "BOARD_TOKEN", "BOARD_NAME"):
            assert name not in os.environ

    @pytest.mark.slow(reason="child pytest: interpreter + unit conftest import, ~1.5 s alone, >2 s under the gate")
    def test_guard_holds_in_a_child_pytest_with_board_env_exported(self) -> None:
        env = {
            **os.environ,
            "BOARD_URL": "http://127.0.0.1:4100/api/board",
            "BOARD_PROFILE": "leak",
            "BOARD_TOKEN": "leak",
            "PYTHONPATH": str(_ROOT),
            "PYTEST_ADDOPTS": "",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(Path(__file__)), "-k", "no_board_env_in_a_unit_test_body",
             "-v", "-p", "no:randomly", "-o", "addopts="],
            cwd=str(_ROOT), env=env, capture_output=True, text=True, timeout=300, check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "test_no_board_env_in_a_unit_test_body PASSED" in proc.stdout
