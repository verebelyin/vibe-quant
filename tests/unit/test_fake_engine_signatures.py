"""The shared fake NT engine (tests/unit/fake_engine.py) must keep NT's call shapes.

A fake that accepts a call NT rejects hides a production bug: the fake ``cache.bars()``
took no argument for months while NT 1.231 requires a bar type, so the slippage path that
called it argument-less stayed green. Each fake method must require exactly the parameters
the real NT method requires and name no parameter NT does not have; each fake attribute
must exist on the NT class, as a property where NT has a property.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.cache.cache import Cache
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.position import Position
from nautilus_trader.trading.trader import Trader

from tests.unit.fake_engine import FakeFill, FakePosition, fake_engine

if TYPE_CHECKING:
    from collections.abc import Callable

# FakePosition fields that only feed a fake method (not NT attributes).
FAKE_ONLY_FIELDS = {"commission_list"}


def _params(fn: Callable[..., Any]) -> tuple[set[str], set[str]]:
    """(required, all) parameter names, without ``self``."""
    params = [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]
    required = {p.name for p in params if p.default is inspect.Parameter.empty}
    return required, {p.name for p in params}


def _assert_same_shape(fake: Callable[..., Any], real: Callable[..., Any], name: str) -> None:
    fake_req, fake_all = _params(fake)
    real_req, real_all = _params(real)
    assert fake_req == real_req, f"{name}: fake requires {fake_req}, NT requires {real_req}"
    assert fake_all <= real_all, f"{name}: fake names {fake_all - real_all}, NT has none"


def _engine() -> Any:
    return fake_engine([], bars_by_type={}, strategies=[])


@pytest.mark.parametrize("name", ["positions", "position_snapshots", "bar_types", "bars", "orders"])
def test_fake_cache_method_matches_nt_cache(name: str) -> None:
    _assert_same_shape(getattr(_engine().kernel.cache, name), getattr(Cache, name), f"Cache.{name}")


def test_fake_trader_strategies_matches_nt_trader() -> None:
    _assert_same_shape(_engine().trader.strategies, Trader.strategies, "Trader.strategies")


def test_fake_position_attributes_exist_on_nt_position() -> None:
    names = {f for f in FakePosition.__dataclass_fields__ if f not in FAKE_ONLY_FIELDS}
    names |= {"is_open"}
    missing = {n for n in names if not hasattr(Position, n)}
    assert not missing, f"FakePosition invents {missing}"
    # NT exposes these as properties: the fake must not make them callables.
    for name in names:
        real = inspect.getattr_static(Position, name)
        fake = inspect.getattr_static(FakePosition, name, None)
        if not callable(real):
            assert not callable(fake) or isinstance(fake, property), f"Position.{name} is a property"


def test_fake_position_commissions_matches_nt_position() -> None:
    _assert_same_shape(FakePosition.commissions, Position.commissions, "Position.commissions")


def test_fake_fill_attributes_exist_on_nt_order_filled() -> None:
    missing = {n for n in FakeFill.__dataclass_fields__ if not hasattr(OrderFilled, n)}
    assert not missing, f"FakeFill invents {missing}"
