"""Venue hold at the Binance wire (opspt P1): the last check before a write leaves.

Every order write of NT's Binance futures execution client (new, cancel, cancel-all,
batch, modify, algo cancels) and its connect-time config writes (leverage, margin type)
reach ``BinanceHttpClient.send_request`` -> ``self._client.request(...)``. The paper node
wraps that transport (``_http_client._client``) with :class:`WriteGate` right after
``node.build()``: while ``hold()`` names a reason, every non-GET request except the
user-data ``listenKey`` keep-alive is refused before it reaches Binance. The refusal is a
``BinanceClientError`` raised in the coroutine that would have sent, so NT emits its own
per-order ``OrderRejected`` / ``OrderCancelRejected`` / ``OrderModifyRejected``; nothing is
queued or replayed.

Rate limits (opspt P1-R3). NT's pyo3 ``HttpClient`` waits for its rate-limit quota
INSIDE ``request``, after the caller's last Python check: a write that passed a check
before the hold could wait there and leave after it. So the gate owns the wait: its
:class:`QuotaLimiter` mirrors NT's factory quotas for the account
(``nautilus_trader/adapters/binance/factories.py:130-139``) and every request of
the gated client (reads AND writes: Binance's IP weight is shared) waits there first.
Only then the hold is checked, and the request goes to the native client with
``keys=None``, which never waits (measured: no keys = no limiter wait). There is no
``await`` between that final check and the native call.

Contract for the operator:

* **Not sent**: any write whose final check ran after the hold began, including writes
  queued for rate limit when it began.
* **Already sent before the hold, may still execute**: a write that passed its final
  check before the hold began left in the same loop turn; Binance may still accept,
  fill or cancel it after the hold.
* **Reads continue**: GETs pass (data feed, reconciliation). The node's own signed reads
  use separate NT clients (``venue_config.py`` and ``force_orders.py`` build theirs with
  their own clock, so ``get_cached_binance_http_client`` returns another instance): the
  periodic venue check stops at the latch, the report-only liquidation watch keeps
  reading until shutdown.

Before agreement (opspt P1-R8, user Q3). Until the node's first venue check agrees,
``unchecked()`` names a reason and every write is refused the same way, WITHOUT a latch
(the first agreeing check opens the gate): orders, cancels, modifies, order lists,
batch and algo writes from strategies (generated exits and ``on_stop`` at SIGTERM),
the guard and NT. Only the connect-time account-config writes NT sends before any
order, which the venue check itself verifies (``POST /fapi/v1/leverage``,
``POST /fapi/v1/marginType``), and the user-data ``listenKey`` keep-alive pass.

A Binance 429/418 still surfaces as NT's ``BinanceClientError`` exactly as before: the
gate never retries. The install fails closed: an execution client that is not a Binance
futures one, or whose transport moved (an NT bump renaming ``_client``), refuses the
start.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from nautilus_trader.adapters.binance.common.constants import (
    BINANCE_FUTURES_ORDER_COUNT_1M_KEY,
    BINANCE_FUTURES_ORDER_COUNT_10S_KEY,
)
from nautilus_trader.adapters.binance.futures.execution import BinanceFuturesExecutionClient
from nautilus_trader.adapters.binance.http.error import BinanceClientError
from nautilus_trader.core.nautilus_pyo3 import HttpMethod

from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.guard import WIRE_HOLD_PREFIX, WIRE_UNCHECKED_PREFIX, checked_hold

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

_ALWAYS_SENT_SUFFIX = "/listenKey"
#: P1-R8: account-config writes NT sends at connect, allowed before the first agreeing
#: venue check (the check verifies their result); refused after a latch like any write.
UNCHECKED_ALLOWED_PATHS = frozenset({"/fapi/v1/leverage", "/fapi/v1/marginType"})

#: (max burst, period seconds): NT's ``Quota.rate_per_minute(n)`` = ``(n, 60.0)``.
QuotaSpec = tuple[int, float]

#: NT 1.231 ``get_cached_binance_http_client`` futures quotas, key for key
#: (factories.py:130-139); a test pins this table against the factory.
BINANCE_FUTURES_QUOTAS: dict[str, QuotaSpec] = {
    "binance:global": (2400, 60.0),
    BINANCE_FUTURES_ORDER_COUNT_10S_KEY: (30, 1.0),
    BINANCE_FUTURES_ORDER_COUNT_1M_KEY: (1200, 60.0),
    "binance:fapi/v1/order": (1200, 60.0),
    "binance:fapi/v1/allOrders": (60, 60.0),
    "binance:fapi/v1/commissionRate": (120, 60.0),
    "binance:fapi/v1/klines": (600, 60.0),
}
#: The factory's default quota: a key without its own quota gets one of these, per key.
BINANCE_FUTURES_DEFAULT_QUOTA: QuotaSpec = (2400, 60.0)


class QuotaLimiter:
    """NT's native rate limiter, in Python (GCRA per key, measured on NT 1.231).

    A quota ``(n, period)`` admits a burst of ``n`` and then one request per
    ``period / n``. A key without its own quota uses ``default`` (each key its own
    state); with no default it is unlimited. A request waits until EVERY key it names
    is ready, then takes one slot of each, so each key's spacing is at least NT's.
    """

    def __init__(
        self,
        quotas: Mapping[str, QuotaSpec],
        default: QuotaSpec | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._quotas = dict(quotas)
        self._default = default
        self._clock = clock
        #: Theoretical arrival time per key (GCRA state).
        self._tat: dict[str, float] = {}

    def _limits(self, keys: Iterable[str]) -> list[tuple[str, float, float]]:
        """``(key, spacing, tolerance)`` for every limited key (tolerance = burst - 1 slots)."""
        out = []
        for key in keys:
            quota = self._quotas.get(key, self._default)
            if quota is not None:
                burst, period = quota
                out.append((key, period / burst, period - period / burst))
        return out

    async def acquire(self, keys: Iterable[str] | None) -> None:
        """Wait until every key of ``keys`` admits one more request, then take it."""
        limits = self._limits(keys or ())
        while limits:
            now = self._clock()
            wait = max(self._tat.get(key, now) - tolerance - now for key, _, tolerance in limits)
            if wait <= 0:
                for key, spacing, _ in limits:
                    self._tat[key] = max(self._tat.get(key, now), now) + spacing
                return
            await asyncio.sleep(wait)


class WriteGate:
    """Stands in for the pyo3 ``HttpClient``: owns its rate-limit wait, then refuses
    writes while held and forwards the rest with no native wait left."""

    def __init__(
        self,
        inner: Any,
        hold: Callable[[], str | None],
        quotas: Mapping[str, QuotaSpec],
        default_quota: QuotaSpec | None = None,
        unchecked: Callable[[], str | None] | None = None,
    ) -> None:
        self._inner = inner
        self._hold = hold
        self._unchecked = unchecked
        self._limiter = QuotaLimiter(quotas, default_quota)

    def _refuse_if_held(self, path: str) -> None:
        reason = checked_hold(self._hold)
        if reason is not None:
            raise BinanceClientError(400, f"{WIRE_HOLD_PREFIX}{reason}; not sent", {})  # type: ignore[no-untyped-call]
        if self._unchecked is None or path in UNCHECKED_ALLOWED_PATHS:
            return
        reason = checked_hold(self._unchecked)
        if reason is not None:
            raise BinanceClientError(400, f"{WIRE_UNCHECKED_PREFIX}{reason}; not sent", {})  # type: ignore[no-untyped-call]

    async def request(
        self,
        method: Any,
        url: str,
        params: Any = None,
        headers: Any = None,
        body: Any = None,
        keys: Any = None,
        timeout_secs: Any = None,
    ) -> Any:
        path = urlparse(url).path
        write = method != HttpMethod.GET and not path.endswith(_ALWAYS_SENT_SUFFIX)
        if write:
            self._refuse_if_held(path)  # held already: refuse without waiting for quota
        await self._limiter.acquire(keys)
        if write:
            # The dispatch boundary: no await between this check and the native call,
            # and keys=None so the native client sends without a limiter wait.
            self._refuse_if_held(path)
        return await self._inner.request(
            method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            keys=None,
            timeout_secs=timeout_secs,
        )


def install_write_gate(
    exec_client: Any,
    hold: Callable[[], str | None],
    unchecked: Callable[[], str | None] | None = None,
) -> WriteGate:
    """Wrap ``exec_client``'s Binance transport with a :class:`WriteGate` asking ``hold``
    (terminal) and ``unchecked`` (before the first agreeing venue check, P1-R8).

    Re-installing on the same (cached, shared) HTTP client replaces the previous gate
    instead of stacking gates. Raises ConfigurationError when the client is not a
    Binance futures execution client or its transport has no ``request``.
    """
    http = getattr(exec_client, "_http_client", None)
    inner = getattr(http, "_client", None)
    if isinstance(inner, WriteGate):
        inner = inner._inner
    if not isinstance(exec_client, BinanceFuturesExecutionClient) or not callable(
        getattr(inner, "request", None)
    ):
        raise ConfigurationError(
            f"Binance write gate not installable on {type(exec_client).__name__}: "
            "paper refuses to start without its venue hold check"
        )
    gate = WriteGate(inner, hold, BINANCE_FUTURES_QUOTAS, BINANCE_FUTURES_DEFAULT_QUOTA, unchecked)
    http._client = gate  # type: ignore[union-attr]
    return gate


__all__ = [
    "BINANCE_FUTURES_DEFAULT_QUOTA",
    "BINANCE_FUTURES_QUOTAS",
    "WIRE_HOLD_PREFIX",
    "UNCHECKED_ALLOWED_PATHS",
    "QuotaLimiter",
    "WriteGate",
    "install_write_gate",
]
