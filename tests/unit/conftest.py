"""Pytest fixtures for the vibe_quant unit suite."""

from __future__ import annotations

import ipaddress
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
