"""Pytest fixtures for the vibe_quant unit suite."""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
import sys
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


# ------------------------------------------------------------------ network guard
#
# A unit test must never open an outbound connection. risk-sb-t2a-1 drove
# ``PaperTradingNode.start()`` with a fake trading node and the liquidation watcher
# made a real Binance call; the guard below turns that into a loud RuntimeError.
# It is a Python-level guard: NautilusTrader's pyo3 HttpClient (Rust ``reqwest``)
# bypasses the ``socket`` module entirely and must be stubbed in the test instead.

#: The real functions, captured before any fixture patches them.
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_GETADDRINFO = socket.getaddrinfo

#: Host names that mean "this machine" without an IP literal.
_LOOPBACK_HOST_NAMES = frozenset({"localhost", "ip6-localhost", "localhost.localdomain"})


def _is_loopback_host(host: object) -> bool:
    """True for ``None``/empty, a loopback name, or a loopback IP literal."""
    if host is None:
        return True
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    if host == "" or host in _LOOPBACK_HOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guard_sockaddr(family: int, sockaddr: object) -> None:
    """Raise unless ``sockaddr`` is loopback or an ``AF_UNIX`` path."""
    if family == socket.AF_UNIX:
        return
    if not isinstance(sockaddr, tuple) or not sockaddr:
        return
    host = sockaddr[0]
    if _is_loopback_host(host):
        return
    port = sockaddr[1] if len(sockaddr) > 1 else None
    raise RuntimeError(f"unit tests must not use the network: {host}:{port}")


@pytest.fixture(autouse=True)
def _forbid_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1/N3/N6: refuse every non-loopback connect, connect_ex, getaddrinfo and
    NautilusTrader Binance request.

    ``socket.create_connection`` resolves through ``getaddrinfo`` and then calls the
    patched ``socket.socket.connect``, so it is covered without its own patch.
    NautilusTrader's ``BinanceHttpClient`` sends through a pyo3 (Rust ``reqwest``)
    ``HttpClient`` that never touches Python's ``socket`` module, so its async
    ``send_request`` (which ``sign_request`` calls) is patched too -- that is the
    exact path risk-sb-t2a-1 reached. A test marked ``allow_network`` keeps the real
    functions (N2).
    """
    if request.node.get_closest_marker("allow_network") is not None:
        return

    def guarded_connect(
        sock: socket.socket, address: Any, *args: Any, **kwargs: Any
    ) -> Any:
        _guard_sockaddr(sock.family, address)
        return _ORIGINAL_CONNECT(sock, address, *args, **kwargs)

    def guarded_connect_ex(
        sock: socket.socket, address: Any, *args: Any, **kwargs: Any
    ) -> Any:
        _guard_sockaddr(sock.family, address)
        return _ORIGINAL_CONNECT_EX(sock, address, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_loopback_host(host):
            raise RuntimeError(f"unit tests must not use the network: {host}:{port}")
        return _ORIGINAL_GETADDRINFO(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    # N6 (chief ruling): the pyo3 HttpClient bypasses `socket`, so block the Binance
    # request itself. `sign_request` routes through `send_request`, so both are cut.
    from nautilus_trader.adapters.binance.http.client import BinanceHttpClient

    async def guarded_send_request(
        self: BinanceHttpClient,
        http_method: Any,
        url_path: str,
        payload: Any = None,
        ratelimiter_keys: Any = None,
    ) -> Any:
        raise RuntimeError(
            f"unit tests must not use the network: BinanceHttpClient {http_method} {url_path}"
        )

    monkeypatch.setattr(BinanceHttpClient, "send_request", guarded_send_request)


# ------------------------------------------------------- inherited swarm env guard
#
# The gate runs ``pytest tests/unit`` as a child of the worker's shell and ``real_run``
# merges ``os.environ`` into every tool subprocess (swarm_check.py:218), so the unit suite
# inherits the SWARM_* vars that scripts/agents/cmd-task.sh exports. ``swarm_check.main``
# resolves ``cfg.bus`` from ``$SWARM_BUS`` (swarm_check.py:1072) and ``bus_post`` then
# writes ``<worktree>: GATE PASS`` into that directory, so every gate test calling
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
    yield
    guard.undo()


@pytest.fixture(autouse=True)
def _no_telegram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete ``TELEGRAM_BOT_TOKEN``/``TELEGRAM_CHAT_ID`` so the start gate is inert.

    A dev or backend shell with Telegram configured would otherwise make every
    node test reach the probe (and the network guard) and refuse to start with
    ``TelegramUnavailableError`` (vibe-quant-u2r9r). Tests that want Telegram set
    it in the test body, after this fixture has run.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


@pytest.fixture(autouse=True)
def _isolate_paper_account_locks(tmp_path: Any) -> Iterator[None]:
    """Point the per-account paper lock dir at this test's tmp_path (never the real ``~``).

    Every CLI/API test reuses the same fake api key, so one real lock file would
    couple parallel xdist workers (vibe-quant-rw1g8). ``acquire_account_lock`` and
    ``account_lock_holder`` resolve ``PAPER_LOCK_DIR`` at call time, so patching the
    module attribute reaches ``paper/cli.py`` and the API router alike.

    Own ``MonkeyPatch`` instance, like ``_forbid_inherited_swarm_env``: a test that
    calls ``monkeypatch.undo()`` must not restore the repo path mid-test.
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
    the next test on the same xdist worker (vibe-quant-ca3v8).
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
    (vibe-quant-73khn: the ordered pair
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
    derived ``operators._MA_NAMES`` draw list in place (vibe-quant-nn7i7), so
    both are snapshot-and-restored here too.

    ``THRESHOLD_RANGES`` / ``_INT_PARAMS`` / ``_PARAM_DEFAULTS`` are built from
    the genome, not from the pools, so they need no restore here.

    The body is ``restore_ga_indicator_pool`` so one test can drive the exact restore
    without relying on test order (an ordered pair broke under xdist worksteal, 1xhri).
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
    ``@pytest.mark.real_data_precheck`` and point it at a tmp catalog
    (vibe-quant-b5xfj).
    """
    if request.node.get_closest_marker("real_data_precheck") is not None:
        return
    from vibe_quant.api import data_precheck

    monkeypatch.setattr(data_precheck, "precheck_discovery_data", _noop_data_precheck)
    monkeypatch.setattr(data_precheck, "precheck_screening_data", _noop_data_precheck)
