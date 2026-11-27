"""Pytest fixtures for vibe-quant tests."""

from __future__ import annotations

import ipaddress
import os
import socket
import tempfile

# Must run before any vibe_quant import: DEFAULT_DB_PATH is resolved at import
# time, and opening a DB runs schema migrations. A test that forgets to pass a
# tmp DB must never touch (or migrate) the real data/state/vibe_quant.db.
os.environ.setdefault(
    "VIBE_QUANT_DB", os.path.join(tempfile.mkdtemp(prefix="vq_test_db_"), "default.db")
)

# Hard guard: vibe_quant.db.connection refuses the real state DB while this is set
# (inherited by job subprocesses a test launches).
os.environ["VIBE_QUANT_PYTEST"] = "1"

# Job log paths resolve through vibe_quant.utils.log_dir(); pin them to a temp
# dir so a launched job (or a test calling a launch endpoint) never writes into
# the repo's logs/ tree.
os.environ.setdefault(
    "VIBE_QUANT_LOG_DIR", os.path.join(tempfile.mkdtemp(prefix="vq_test_logs_"), "logs")
)

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.known_results import KnownResult, load_known_results

if TYPE_CHECKING:
    from collections.abc import Generator

    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.db import StateManager


# ------------------------------------------------------------------ network guard
#
# A test must never open an outbound connection. risk-sb-t2a-1 drove
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
def _forbid_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def guarded_connect(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        _guard_sockaddr(sock.family, address)
        return _ORIGINAL_CONNECT(sock, address, *args, **kwargs)

    def guarded_connect_ex(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
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


# ------------------------------------------------------------- paper warm-up REST seed
#
# A paper start fetches 8,000 closed REST bars per bar type before the node exists
# (``vibe_quant/paper/warmup.py``). A test never reaches Binance: the klines endpoint
# is answered here, at the httpx transport, with a deterministic synthetic series (contiguous,
# aligned on the interval, 50-point range). The fixture names no vibe_quant module on purpose:
# a conftest that reaches production code makes every diff plan the full suite. A test that
# wants other rows patches ``vibe_quant.paper.warmup.fetch_closed_klines`` itself.

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def synthetic_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
    """Closed klines of ``interval`` in [start_ms, end_ms): ``(open_time, o, h, l, c, v, close_time)``."""
    step = _INTERVAL_MS[interval]
    first = start_ms + (-start_ms) % step
    rows = []
    for open_ms in range(first, end_ms, step):
        if open_ms + step - 1 >= end_ms:
            break
        close = 40_000.0 + 100.0 * ((open_ms // step) % 7 - 3)
        rows.append(
            (
                open_ms,
                f"{close - 10.0:.1f}",
                f"{close + 25.0:.1f}",
                f"{close - 25.0:.1f}",
                f"{close:.1f}",
                "10.000",
                open_ms + step - 1,
            )
        )
    return rows


def _klines_page(params: dict[str, str]) -> list[list[Any]]:
    """One ``/fapi/v1/klines`` page like Binance serves it: open_time in [startTime, endTime]
    (so the still-forming bar is included), at most ``limit`` rows, 12 columns per row."""
    step = _INTERVAL_MS[params["interval"]]
    rows = synthetic_klines(
        params["symbol"],
        params["interval"],
        int(params["startTime"]),
        int(params["endTime"]) + step,
    )
    return [[*r, "1.0", 1, "1.0", "1.0", "0"] for r in rows[: int(params["limit"])]]


@pytest.fixture(autouse=True)
def _fake_paper_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    real = httpx.HTTPTransport.handle_request

    def handle_request(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        if request.url.host == "fapi.binance.com" and request.url.path == "/fapi/v1/klines":
            page = _klines_page(dict(request.url.params))
            return httpx.Response(200, json=page, request=request)
        return real(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)


_NT_EAGER_IMPORT_TEST_MODULES = frozenset(
    {
        "test_data_catalog.py",
        "test_ethereal_instruments.py",
        "test_ethereal_venue.py",
        "test_fill_timing.py",
        "test_validation_venue.py",
    }
)
_NT_RUNTIME_TEST_PREFIXES = ("test_validation_",)


@lru_cache(maxsize=1)
def _nt_runtime_preflight() -> tuple[bool, str]:
    """Check that NautilusTrader can be imported and basic C-extension types instantiate."""
    try:
        from nautilus_trader.model.objects import Price, Quantity

        Price.from_str("1.0")
        Quantity.from_str("1")
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _requires_nt_runtime(path: Path) -> bool:
    """Return True for tests that should be skipped when NT runtime is incompatible."""
    filename = path.name
    return filename in _NT_EAGER_IMPORT_TEST_MODULES or filename.startswith(
        _NT_RUNTIME_TEST_PREFIXES
    )


def _nt_skip_reason() -> str:
    _, reason = _nt_runtime_preflight()
    return f"Skipping NT-dependent tests: NautilusTrader runtime preflight failed ({reason})"


def pytest_report_header(config: pytest.Config) -> str:  # pragma: no cover - pytest hook
    """Print NT compatibility status once per test session."""
    compatible, reason = _nt_runtime_preflight()
    if compatible:
        return "NT runtime preflight: OK"
    return f"NT runtime preflight: FAILED ({reason})"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Avoid importing modules with top-level NT imports when runtime is incompatible."""
    compatible, _ = _nt_runtime_preflight()
    if compatible or collection_path.suffix != ".py":
        return False
    return collection_path.name in _NT_EAGER_IMPORT_TEST_MODULES


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark NT-dependent tests as skipped when preflight fails."""
    compatible, _ = _nt_runtime_preflight()
    if compatible:
        return

    marker = pytest.mark.skip(reason=_nt_skip_reason())
    for item in items:
        item_path = Path(str(getattr(item, "path", item.fspath)))
        if _requires_nt_runtime(item_path):
            item.add_marker(marker)


# Re-export for convenience
__all__ = ["KnownResult", "load_known_results"]


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def tmp_archive(tmp_path: Path) -> Generator[RawDataArchive]:
    """Create temporary RawDataArchive for testing."""
    from vibe_quant.data.archive import RawDataArchive

    archive_path = tmp_path / "archive.db"
    archive = RawDataArchive(archive_path)
    yield archive
    archive.close()


@pytest.fixture
def tmp_state_manager(tmp_path: Path) -> Generator[StateManager]:
    """Create temporary StateManager for testing."""
    from vibe_quant.db import StateManager

    db_path = tmp_path / "state.db"
    manager = StateManager(db_path)
    yield manager
    manager.close()


@pytest.fixture
def ohlc_aggregation_cases() -> list[KnownResult]:
    """Load OHLC aggregation known results."""
    return load_known_results("ohlc_aggregation")


@pytest.fixture
def sample_klines() -> list[tuple[int, float, float, float, float, float, int]]:
    """Sample kline data for testing.

    Returns list of (open_time, open, high, low, close, volume, close_time).
    """
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
    return [
        (base_time, 42000.0, 42100.0, 41900.0, 42050.0, 100.0, base_time + 59999),
        (base_time + 60000, 42050.0, 42200.0, 42000.0, 42150.0, 120.0, base_time + 119999),
        (base_time + 120000, 42150.0, 42300.0, 42100.0, 42250.0, 80.0, base_time + 179999),
        (base_time + 180000, 42250.0, 42350.0, 42200.0, 42300.0, 150.0, base_time + 239999),
        (base_time + 240000, 42300.0, 42400.0, 42250.0, 42350.0, 90.0, base_time + 299999),
    ]


@pytest.fixture
def sample_dsl_config() -> dict[str, object]:
    """Sample DSL configuration for testing."""
    return {
        "name": "test_strategy",
        "description": "Test strategy",
        "version": 1,
        "timeframe": "5m",
        "indicators": {
            "rsi": {"type": "RSI", "period": 14, "source": "close"},
        },
        "entry_conditions": {
            "long": ["rsi < 30"],
            "short": ["rsi > 70"],
        },
        "exit_conditions": {
            "long": ["rsi > 50"],
            "short": ["rsi < 50"],
        },
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }
