"""What the paper bot owns on Binance: its own orders and the net they filled (opspt P2).

An order is the bot's own iff the guard's gate admitted it in this process, or the
restored checkpoint names it in ``own_orders``. ``owned_net[symbol]`` moves only by the
fills of own orders, so a position or order the user opened by hand on a bot coin never
counts as the bot's: the venue check then sees Binance differ from the bot's record and
holds (it is never closed, cancelled or adopted).

Per own order the ledger keeps the filled quantity it already counted, and moves the net
by ``order.filled_qty - counted``. Syncing is idempotent, so a restart whose own order
filled while the bot was down (NT reconciles that fill before any actor listens) counts
it exactly once, and an NT re-created partly filled order is not counted twice.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

#: The restored checkpoint carried no usable ownership record (absent, or junk): the
#: bot owns nothing, so any non-zero Binance position on a bot coin holds (user Q8=A).
OWNERSHIP_UNKNOWN = "ownership_unknown"

#: Checkpoint ``node_status`` keys.
OWNED_NET_KEY = "owned_net"
OWN_ORDERS_KEY = "own_orders"


#: A saved quantity: a plain ASCII decimal, at most 12 + 12 digits (no exponent, NaN or
#: Inf; sums of such values stay exact in Decimal's default 28-digit context).
_QUANTITY: Final = re.compile(r"-?[0-9]{1,12}(\.[0-9]{1,12})?")


def _quantity(raw: object, *, signed: bool) -> Decimal | None:
    """The decimal a saved quantity string holds, else None (junk: a float, an exponent,
    a negative zero, a negative fill, a value Decimal would round)."""
    if not isinstance(raw, str) or _QUANTITY.fullmatch(raw) is None:
        return None
    value = Decimal(raw)
    if raw.startswith("-") and (not signed or value.is_zero()):
        return None
    return value


def _plain_key(key: object) -> bool:
    """A non-empty printable ASCII string with no surrounding blanks."""
    return (
        isinstance(key, str)
        and bool(key)
        and key.isascii()
        and key.isprintable()
        and key.strip() == key
    )


def _client_order_id(key: object) -> bool:
    """``key`` is a client order id NT accepts. ASCII is checked first: NT's Rust
    constructor aborts the process on a non-ASCII id (uncatchable); a ValueError of
    NT's own check reaches :meth:`OwnershipLedger.restore`, which logs it (unknown)."""
    if not _plain_key(key):
        return False
    from nautilus_trader.model.identifiers import ClientOrderId

    ClientOrderId(cast("str", key))
    return True


def _quantity_map(
    raw: object, valid_key: Callable[[object], bool], *, signed: bool
) -> dict[str, Decimal] | None:
    """``{key: quantity string}`` parsed whole, or None when any key or value is junk."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, Decimal] = {}
    for key, value in raw.items():
        amount = _quantity(value, signed=signed)
        if amount is None or not valid_key(key):
            return None
        out[key] = amount
    return out


class OwnershipLedger:
    """Own client order ids and the net position their fills add up to, per symbol."""

    def __init__(
        self,
        net: Mapping[str, Decimal] | None = None,
        counted: Mapping[str, Decimal] | None = None,
        note: str | None = OWNERSHIP_UNKNOWN,
    ) -> None:
        self._net: dict[str, Decimal] = dict(net or {})
        #: Own client order id -> filled quantity already in ``_net``.
        self._counted: dict[str, Decimal] = dict(counted or {})
        #: ``OWNERSHIP_UNKNOWN`` when the restored record was missing or junk.
        self.note = note

    @classmethod
    def restore(cls, node_status: object) -> OwnershipLedger:
        """The ledger a restored checkpoint's ``node_status`` seeds (R45).

        Any junk in either key (missing, null, an invalid order id or symbol, a junk
        quantity) makes the WHOLE record unknown: 0 for every coin, no own order,
        ``OWNERSHIP_UNKNOWN`` (never a partial record). Never raises.
        """
        status = node_status if isinstance(node_status, dict) else {}
        try:
            net = _quantity_map(status.get(OWNED_NET_KEY), _plain_key, signed=True)
            counted = _quantity_map(status.get(OWN_ORDERS_KEY), _client_order_id, signed=False)
        except Exception as exc:
            logger.error("ownership record restore failed (%s)", type(exc).__name__)
            net = counted = None
        if net is None or counted is None:
            # Key names only: the values are checkpoint data, not something to log.
            logger.warning(
                "ownership record missing or unreadable: the bot owns nothing on Binance"
            )
            return cls(None, None, OWNERSHIP_UNKNOWN)
        return cls(net, counted, None)

    def admit(self, client_order_id: str) -> None:
        """The gate let this order through: it is the bot's own from now on."""
        self._counted.setdefault(client_order_id, Decimal(0))

    def is_own(self, client_order_id: str) -> bool:
        return client_order_id in self._counted

    def sync(self, client_order_id: str, symbol: str, sign: int, filled: Decimal) -> bool:
        """Count an own order's total ``filled`` quantity (``sign`` +1 buy, -1 sell) once.

        Returns True when the net moved. A foreign id is ignored.
        """
        counted = self._counted.get(client_order_id)
        if counted is None or filled == counted:
            return False
        self._counted[client_order_id] = filled
        self._net[symbol] = self.owned(symbol) + sign * (filled - counted)
        return True

    def owned(self, symbol: str) -> Decimal:
        return self._net.get(symbol, Decimal(0))

    def clear(self, symbol: str) -> None:
        """Forget the bot's record for ``symbol`` (Binance is flat at a restart)."""
        self._net[symbol] = Decimal(0)

    def status(self, symbols: Iterable[str], open_ids: Iterable[str]) -> dict[str, object]:
        """The checkpoint keys: the net per session symbol and every own order still
        open (``open_ids`` filters which own ids to keep) with its counted fill."""
        return {
            OWNED_NET_KEY: {s: format(self.owned(s), "f") for s in symbols},
            OWN_ORDERS_KEY: {
                cid: format(self._counted[cid], "f") for cid in open_ids if cid in self._counted
            },
        }

    def own_ids(self) -> list[str]:
        return list(self._counted)
