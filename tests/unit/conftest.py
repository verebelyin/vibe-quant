"""Pytest fixtures for the vibe_quant unit suite."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


# ------------------------------------------------------- inherited swarm env guard
#
# The gate runs ``pytest tests/unit`` as a child of the worker's shell and ``real_run``
# merges ``os.environ`` into every tool subprocess (swarm_check.py:218), so the unit suite
# inherits the SWARM_* vars that scripts/agents/cmd-task.sh exports. ``swarm_check.main``
# resolves ``cfg.bus`` from ``$SWARM_BUS`` (swarm_check.py:1072) and the gate then
# posted ``<worktree>: GATE PASS`` into that directory (it no longer posts), so every gate test calling
# ``main()`` without ``--bus`` posted DMs into the LIVE job bus: one worker's gate run
# collected ~30 of them. An inherited ``SWARM_AGENT`` alone is a bus-inbox FAIL, which
# made TestIntegration::test_script_gate_pass/_fail red too. A unit test must never read
# or write a real swarm bus, board or telemetry ledger.
#
# Tests that need one of these variables set it themselves *after* this fixture (a
# function-scoped autouse fixture runs before the test body): test_swarm_bus.py's
# _fast_poll (SWARM_BOARD) and test_telemetry.py's _isolated_ledger (SWARM_TELEMETRY)
# both re-point theirs at tmp dirs anyway, and the SWARM_CHECK_* stub tests in
# test_swarm_check.py set theirs in the test body. monkeypatch restores the caller's env
# at teardown.
#
# Read somewhere under scripts/agents: SWARM_BUS, SWARM_AGENT, SWARM_BOARD, SWARM_JOBS
# (bus.py); SWARM_CHECK_RUFF / SWARM_CHECK_MYPY / SWARM_CHECK_PYTEST /
# SWARM_CHECK_ALLOW_STUBS (swarm_check.py); SWARM_TELEMETRY, SWARM_BUS_ANNOUNCE,
# SWARM_HOOKS. The prefix sweep is deliberate -- a new SWARM_* var must not need this
# fixture edited before the suite stays hermetic.
#
# It holds its OWN MonkeyPatch instance rather than the shared ``monkeypatch`` fixture:
# TestPassCache::test_stamp_write_is_atomic calls ``monkeypatch.undo()`` mid-test, which
# undoes everything that instance recorded -- the strip included -- and the gate runs after
# the undo then posted to the caller's bus again (measured with a timestamped -v run:
# the last live-bus DM of a worker's gate run came from exactly that test).


@pytest.fixture(autouse=True)
def _forbid_inherited_swarm_env() -> Iterator[None]:
    """Strip the caller's SWARM_* vars so no test can reach a real swarm bus/board/ledger."""
    guard = pytest.MonkeyPatch()
    for name in [key for key in os.environ if key.startswith("SWARM_")]:
        guard.delenv(name, raising=False)
    # bus.py's product default is the board (DEFAULT_STORE): the suite stays on JSONL unless a
    # board test sets SWARM_STORE=board itself (test_board_store.py's ``board`` fixture)
    guard.setenv("SWARM_STORE", "jsonl")
    yield
    guard.undo()


# ------------------------------------------------------------------ board isolation
#
# The swarm's Vercel Board (`vb`, scripts/agents/boardctl.py) is found through BOARD_URL
# and posts as BOARD_PROFILE; BOARD_TOKEN / BOARD_NAME pick a token or a team board. A
# worker shell exports them, so a test (or a child it spawns) would read or post to the
# real board. BOARD_URL is pinned to an unroutable port (a `vb` call fails instead of
# falling back to a Vercel team board); the rest are removed. Own MonkeyPatch instance,
# for the reason given above. A test that needs a board uses ``board_server``.

#: Where a unit test's BOARD_URL points: discard port, nothing listens there.
UNROUTABLE_BOARD_URL = "http://127.0.0.1:9/api/board"
_BOARD_ENV_REMOVED = ("BOARD_PROFILE", "BOARD_TOKEN", "BOARD_NAME")


@pytest.fixture(autouse=True)
def _forbid_inherited_board_env() -> Iterator[None]:
    """Pin BOARD_URL to an unroutable URL and drop BOARD_PROFILE/TOKEN/NAME."""
    guard = pytest.MonkeyPatch()
    guard.setenv("BOARD_URL", UNROUTABLE_BOARD_URL)
    # `vb read kb/` appends to a retro log in the main checkout: never from a test
    guard.setenv("VQ_KB_READS_LOG", str(Path(tempfile.gettempdir()) / f"vq-kb-reads-{os.getpid()}.jsonl"))
    for name in _BOARD_ENV_REMOVED:
        guard.delenv(name, raising=False)
    yield
    guard.undo()


_AGENTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "agents"


@dataclass(frozen=True)
class BoardServer:
    """A throwaway `vb serve` for tests: pass ``env`` to every subprocess that talks to it."""

    url: str
    dir: Path
    port: int
    env: dict[str, str]

    def vb(
        self, *args: str, profile: str | None = None, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run ``scripts/agents/bin/vb`` (the launcher, secret scan included) against this board."""
        env = dict(self.env)
        if profile is not None:
            env["BOARD_PROFILE"] = profile
        return subprocess.run(
            [str(_AGENTS_DIR / "bin" / "vb"), *args],
            input=stdin, env=env, capture_output=True, text=True, timeout=60, check=False,
        )


def free_port() -> int:
    """A loopback port nothing listens on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def board_env(xdg: Path, url: str) -> dict[str, str]:
    """The caller's env minus every BOARD_* var, pointed at ``url`` with its own token dir.

    Built from scratch rather than from the guarded env: a session fixture is set up
    before the function-scoped guard runs, so os.environ may still hold the caller's board."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOARD_")}
    env["BOARD_URL"] = url
    env["XDG_CONFIG_HOME"] = str(xdg)
    env["VQ_KB_READS_LOG"] = str(xdg / "kb-reads.jsonl")
    return env


#: One file per running fixture server, ``<server pid>.json`` = {server, owner, dir}: what the
#: next session reaps when the pytest that owned it was SIGKILLed (no cleanup could run).
BOARD_REGISTRY = Path(tempfile.gettempdir()) / "vq-board-fixture"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_leaked_boards() -> None:
    """Kill each registered `vb serve` whose owning pytest is gone (only a `vb serve` that still
    runs on its recorded dir: a recycled pid is left alone); drop the stale entries."""
    for entry in BOARD_REGISTRY.glob("*.json"):
        try:
            rec = json.loads(entry.read_text())
            server, owner, data = int(rec["server"]), int(rec["owner"]), str(rec["dir"])
        except (OSError, ValueError, KeyError, TypeError):
            entry.unlink(missing_ok=True)
            continue
        if _pid_alive(owner):
            continue  # another session's live board
        cmd = subprocess.run(["ps", "-p", str(server), "-o", "command="], capture_output=True, text=True, check=False).stdout
        if " serve " in f" {cmd} " and data in cmd:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(server, signal.SIGKILL)  # its own session (boardctl starts it so)
        entry.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def board_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BoardServer]:
    """One `vb serve` per xdist worker on a tmp dir + free port; stopped at teardown.

    The server runs in its own process group (boardctl). It is stopped in a ``finally`` and,
    should teardown never run, by an ``atexit`` hook; its pid is registered in
    ``BOARD_REGISTRY``, so a session whose pytest was SIGKILLed leaks it only until the next
    session's fixture reaps it. A missing Node or a failed install fails the test with the
    reason (never a skip)."""
    reap_leaked_boards()
    root = tmp_path_factory.mktemp("board")
    port = free_port()
    url = f"http://127.0.0.1:{port}/api/board"
    env = board_env(root / "xdg", url)
    ctl = [sys.executable, str(_AGENTS_DIR / "boardctl.py")]
    data = root / "data"
    up = subprocess.run(
        [*ctl, "up", "--dir", str(data), "--port", str(port), "--no-inspect"],
        env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    if up.returncode != 0:
        pytest.fail(f"board_server: boardctl up failed ({up.returncode}): {up.stdout}{up.stderr}")
    pid = int((data / "serve.pid").read_text().strip())
    BOARD_REGISTRY.mkdir(exist_ok=True)
    entry = BOARD_REGISTRY / f"{pid}.json"
    entry.write_text(json.dumps({"server": pid, "owner": os.getpid(), "dir": str(data)}))

    def stop() -> None:
        subprocess.run([*ctl, "down", "--dir", str(data)], env=env, capture_output=True, text=True, timeout=60, check=False)
        entry.unlink(missing_ok=True)

    atexit.register(stop)
    try:
        yield BoardServer(url=url, dir=data, port=port, env=env)
    finally:
        stop()
        atexit.unregister(stop)


@pytest.fixture(autouse=True)
def _no_telegram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete ``TELEGRAM_BOT_TOKEN``/``TELEGRAM_CHAT_ID`` so the start gate is inert.

    A dev or backend shell with Telegram configured would otherwise make every
    node test reach the probe (and the network guard) and refuse to start with
    ``TelegramUnavailableError``. Tests that want Telegram set
    it in the test body, after this fixture has run.

    regression: vibe-quant-u2r9r
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


@pytest.fixture(autouse=True)
def _isolate_paper_account_locks(tmp_path: Any) -> Iterator[None]:
    """Point the per-account paper lock dir at this test's tmp_path (never the real ``~``).

    Every CLI/API test reuses the same fake api key, so one real lock file would
    couple parallel xdist workers. ``acquire_account_lock`` and
    ``account_lock_holder`` resolve ``PAPER_LOCK_DIR`` at call time, so patching the
    module attribute reaches ``paper/cli.py`` and the API router alike.

    Own ``MonkeyPatch`` instance, like ``_forbid_inherited_swarm_env``: a test that
    calls ``monkeypatch.undo()`` must not restore the repo path mid-test.

    regression: vibe-quant-rw1g8
    """
    from vibe_quant.paper import account_lock

    guard = pytest.MonkeyPatch()
    guard.setattr(account_lock, "PAPER_LOCK_DIR", tmp_path / "paper_locks")
    yield
    guard.undo()


@pytest.fixture(autouse=True)
def _restore_aux_data_paths() -> Iterator[None]:
    """Undo ``aux_data.configure`` leaks: its archive/catalog paths are module globals.

    ``NTScreeningRunner`` re-configures them on every run -- with the run's tmp
    catalog in tests -- and never resets, so the last test's tmp path leaks into
    the next test on the same xdist worker.

    regression: vibe-quant-ca3v8
    """
    aux = sys.modules.get("vibe_quant.dsl.aux_data")
    saved = (aux._archive_path, aux._catalog_path) if aux is not None else None

    yield

    aux = sys.modules.get("vibe_quant.dsl.aux_data")
    if aux is None:
        return
    archive, catalog = saved if saved is not None else (None, None)
    aux.configure(archive, catalog)


@pytest.fixture(autouse=True)
def _restore_ga_indicator_pool() -> Iterator[None]:
    """Undo ``_apply_indicator_pool_filter`` leaks: it deletes module-global keys.

    ``DiscoveryPipeline._apply_indicator_pool_filter`` filters
    ``operators.INDICATOR_POOL`` (and the derived ``_INDICATOR_NAMES`` list) IN
    PLACE -- correct in production, where discovery runs as a subprocess, but an
    in-process test that runs the pipeline leaves every later test on the same
    xdist worker with a filtered pool. The default pool drops the needs_context
    specs (BTC_TREND, BTC_ROC, OWN_TREND, FUNDING, FUNDING_Z), so a later launch
    test validating a context pool dies with "indicator_pool contains names not
    available for GA discovery: ['BTC_TREND']" instead of its own assertion.
    ``_ensure_pool`` only rebuilds an EMPTY dict, so a filtered pool persists
    (the ordered pair
    ``test_discovery_pipeline.py test_launch_data_precheck.py`` fails 1/87;
    the parallel gate missed it because xdist split the two files apart).

    Snapshot-and-restore rather than rebuild from the registry: the first test of
    a worker starts from the real pool, so restoring the pre-test value is what
    makes a single-file run and a full-suite run see identical state. The import
    is deferred to fixture time and skipped entirely when the discovery package
    is not loaded yet -- importing it at conftest module scope would pull
    ``vibe_quant.discovery`` into every unit collection (see
    ``_NT_EAGER_IMPORT_TEST_MODULES`` in tests/conftest.py for why that matters).

    ``_apply_indicator_pool_filter`` also filters ``genome.MA_POOL`` and the
    derived ``operators._MA_NAMES`` draw list in place, so
    both are snapshot-and-restored here too.

    ``THRESHOLD_RANGES`` / ``_INT_PARAMS`` / ``_PARAM_DEFAULTS`` are built from
    the genome, not from the pools, so they need no restore here.

    The body is ``restore_ga_indicator_pool`` so one test can drive the exact restore
    without relying on test order (an ordered pair broke under xdist worksteal).

    regression: vibe-quant-1xhri
    """
    with restore_ga_indicator_pool():
        yield


@contextlib.contextmanager
def restore_ga_indicator_pool() -> Iterator[None]:
    """Snapshot the GA pools on entry, restore them on exit (see the fixture above)."""
    ops = sys.modules.get("vibe_quant.discovery.operators")
    if ops is None:
        yield
        return

    genome = sys.modules.get("vibe_quant.discovery.genome")
    saved_pool = dict(ops.INDICATOR_POOL)
    saved_names = list(ops._INDICATOR_NAMES)
    saved_ma_pool = dict(genome.MA_POOL) if genome is not None else None
    saved_ma_names = list(ops._MA_NAMES)

    try:
        yield
    finally:  # a context manager sees the test's exception at `yield`; restore anyway
        ops.INDICATOR_POOL.clear()
        ops.INDICATOR_POOL.update(saved_pool)
        ops._INDICATOR_NAMES.clear()
        ops._INDICATOR_NAMES.extend(saved_names)
        ops._MA_NAMES.clear()
        ops._MA_NAMES.extend(saved_ma_names)
        genome = genome if genome is not None else sys.modules.get("vibe_quant.discovery.genome")
        if genome is not None:
            genome.MA_POOL.clear()
            if saved_ma_pool is not None:
                genome.MA_POOL.update(saved_ma_pool)
            else:
                # First import happened inside the test: nothing had filtered it yet.
                genome.MA_POOL.update(genome.build_ma_pool())


def _noop_data_precheck(*args: object, **kwargs: object) -> None:
    """Stand-in for a launch data precheck (see ``_stub_launch_data_precheck``)."""


@pytest.fixture(autouse=True)
def _stub_launch_data_precheck(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neutralise the launch data prechecks (fill ticks, context-indicator aux data).

    ``vibe_quant.api.data_precheck`` reads the catalog (and the funding archive /
    BTC daily bars for context indicators), so a launch test with fake
    symbols/dates would 422 on machine data -- and worktrees symlink the real
    2 GB ``data/archive``. The tests that pin its behaviour opt out with
    ``@pytest.mark.real_data_precheck`` and point it at a tmp catalog.

    regression: vibe-quant-b5xfj
    """
    if request.node.get_closest_marker("real_data_precheck") is not None:
        return
    from vibe_quant.api import data_precheck

    monkeypatch.setattr(data_precheck, "precheck_discovery_data", _noop_data_precheck)
    monkeypatch.setattr(data_precheck, "precheck_screening_data", _noop_data_precheck)
