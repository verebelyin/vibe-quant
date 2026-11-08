"""``signal_level`` stops and targets on the real runners (design-v2 section 3.2, R9, R10).

Toy catalog (``_write_catalog``): BTCUSDT rises +0.01 %/min from 10 000; 15m bar ``k`` covers minutes
``15k .. 15k+14`` with low ``10000 * (1 + 1e-4 * (15k - 1))`` and high ``10000 * (1 + 1e-4 * (15k + 15))``.
The strategy is always long (``close > 0``) with DONCHIAN(40) on 15m: the stop is ``dc.lower`` (the low of
the bar 39 bars back), so the first signal is bar 39 (stop 9 999.0, target 10 600.0) and each later bar's
stop is 15 higher. Account 1 000 USDT, risk 2 % = 20 USDT per trade, size cap 50 % of the account.
Everything is read from the ENGINE CACHE, never from serialized results.

The size proves the stop distance: 20 / (close - 9 999) = 0.033 BTC, where the generic 20 % stop would
size 0.009 and a 2 % fallback 0.047.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from tests.unit.test_screening_fill_mode import _START, _write_catalog
from vibe_quant.db.state_manager import StateManager
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.sizing_settings import SizingSettings
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from pathlib import Path

_END = "2024-01-02"


class Ord(NamedTuple):
    type: str
    side: str
    reduce_only: bool
    qty: float
    trigger: float
    limit_px: float
    status: str
    ts: int


def _orders(engine: Any) -> list[Ord]:
    out = []
    for o in engine.kernel.cache.orders():
        trigger = getattr(o, "trigger_price", None)
        limit_px = getattr(o, "price", None)
        out.append(
            Ord(
                type=o.order_type.name,
                side=o.side.name,
                reduce_only=bool(o.is_reduce_only),
                qty=float(o.quantity),
                trigger=float(trigger) if trigger is not None else 0.0,
                limit_px=float(limit_px) if limit_px is not None else 0.0,
                status=o.status.name,
                ts=int(o.ts_init),
            )
        )
    return sorted(out, key=lambda r: r.ts)


def dsl(name: str, *, target: bool = True, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "15m",
        "indicators": {"dc": {"type": "DONCHIAN", "period": 40}},
        "entry_conditions": {"long": ["close > 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 20.0},
        "take_profit": {"type": "fixed_pct", "percent": 90.0},
        "stop_loss_long": {"type": "signal_level", "indicator": "dc.lower"},
    }
    if target:
        d["take_profit_long"] = {"type": "signal_level", "indicator": "dc.upper"}
    d.update(over)
    return d


def run_screening(catalog: Path, d: dict[str, Any]) -> list[Ord]:
    seen: list[list[Ord]] = []
    orig = NTScreeningRunner._extract_metrics

    def spy(self: NTScreeningRunner, params: Any, bt_result: Any, engine: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(_orders(engine))
        return orig(self, params, bt_result, engine, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(NTScreeningRunner, "_extract_metrics", spy)
        NTScreeningRunner(
            d,
            ["BTCUSDT"],
            _START,
            _END,
            catalog_path=str(catalog),
            launch=LaunchSettings(sizing=SizingSettings(max_margin_pct=None)),
        )({})
    return seen[0]


def run_validation(tmp_path: Path, catalog: Path, d: dict[str, Any]) -> list[Ord]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "state.db"
    state = StateManager(db)
    strategy_id = state.create_strategy(name=str(d["name"]), dsl_config=d)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe=str(d["timeframe"]),
        start_date=_START,
        end_date=_END,
        parameters={},
    )
    state.close()

    seen: list[list[Ord]] = []
    orig = ValidationRunner._extract_results

    def spy(self: ValidationRunner, **kwargs: Any) -> Any:
        seen.append(_orders(kwargs["engine"]))
        return orig(self, **kwargs)

    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ValidationRunner, "_extract_results", spy)
        try:
            runner.run(run_id=run_id)
        finally:
            runner.close()
    return seen[0]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_catalog(tmp_path_factory.mktemp("catalog"), ("BTCUSDT",), timeframes=("1m", "15m", "1h", "4h"))


def first_pair(orders: list[Ord]) -> tuple[Ord, Ord, Ord]:
    entry = next(o for o in orders if not o.reduce_only)
    stop = next(o for o in orders if o.type == "STOP_MARKET")
    target = next(o for o in orders if o.type == "LIMIT" and o.reduce_only)
    return entry, stop, target


def test_screening_snapshots_the_signal_bar_levels_and_sizes_on_the_stop_distance(catalog: Path) -> None:
    entry, stop, target = first_pair(run_screening(catalog, dsl("slr_screening")))

    assert (entry.side, entry.qty) == ("BUY", 0.033)
    assert (stop.side, stop.trigger, stop.qty) == ("SELL", 9999.0, 0.033)
    assert (target.side, target.limit_px, target.qty) == ("SELL", 10600.0, 0.033)


def test_validation_snapshots_the_signal_bar_levels_and_sizes_on_the_stop_distance(
    catalog: Path, tmp_path: Path
) -> None:
    entry, stop, target = first_pair(run_validation(tmp_path, catalog, dsl("slr_validation")))

    assert (entry.side, entry.qty) == ("BUY", 0.033)
    assert (stop.side, stop.trigger, stop.qty) == ("SELL", 9999.0, 0.033)
    assert (target.side, target.limit_px, target.qty) == ("SELL", 10600.0, 0.033)


def _stop_triggers(orders: list[Ord]) -> list[float]:
    return [o.trigger for o in orders if o.type == "STOP_MARKET"]


def test_screening_refresh_moves_the_stop_up_with_every_signal(catalog: Path) -> None:
    """R10: the target stays the generic 90 %, so the long is held while every later bar signals again:
    each signal replaces the stop with that bar's lower (15 higher each time) and the old one is cancelled."""
    orders = run_screening(catalog, dsl("slr_refresh", target=False, refresh_on_signal=True))

    assert _stop_triggers(orders)[:4] == [9999.0, 10014.0, 10029.0, 10044.0]
    live = [o for o in orders if o.type == "STOP_MARKET" and o.status not in ("CANCELED", "FILLED")]
    assert len(live) == 1  # one live stop at the end of the data
    assert all(o.status == "CANCELED" for o in orders if o.type == "STOP_MARKET" and o is not live[0])


def test_validation_refresh_moves_the_stop_up_with_every_signal(catalog: Path, tmp_path: Path) -> None:
    orders = run_validation(tmp_path, catalog, dsl("slr_refresh_val", target=False, refresh_on_signal=True))

    assert _stop_triggers(orders)[:4] == [9999.0, 10014.0, 10029.0, 10044.0]


def test_without_refresh_the_entry_snapshot_stays(catalog: Path) -> None:
    orders = run_screening(catalog, dsl("slr_norefresh", target=False))

    assert _stop_triggers(orders) == [9999.0]
