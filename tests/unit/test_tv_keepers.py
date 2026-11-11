"""TradingView-port keeper bar: window slicing of a continuous run and the pass/fail edges."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TV_DIR = Path(__file__).resolve().parents[2] / "scripts" / "research" / "tv_ports"
_spec = importlib.util.spec_from_file_location("tv_keepers", TV_DIR / "keepers.py")
assert _spec is not None and _spec.loader is not None
kp = importlib.util.module_from_spec(_spec)
sys.modules["tv_keepers"] = kp
_spec.loader.exec_module(kp)

DAY = kp.oa.DAY_NS
D0 = kp.oa.day_of("2026-01-01")


def _pos(day: int, side: int, px_open: float, px_close: float | None) -> dict[str, object]:
    return {"side": side, "ts_open": (D0 + day) * DAY, "ts_close": None, "px_open": px_open, "px_close": px_close}


def test_window_metrics_slices_days_and_counts_trades_entered_in_window() -> None:
    row = {
        "first_day": D0,
        "daily": [0.5, -0.5, 0.25, 0.0],
        "positions": [
            _pos(0, 1, 128.0, 256.0),  # entered before the window: excluded
            _pos(1, 1, 128.0, 160.0),  # +0.25
            _pos(1, -1, 128.0, 160.0),  # -0.25
            _pos(2, 1, 128.0, 192.0),  # +0.5
            _pos(2, 1, 128.0, None),  # still open: a trade, not in PF
            _pos(3, 1, 128.0, 64.0),  # entered on the end day (exclusive): excluded
        ],
    }
    m = kp.window_metrics(row, "2026-01-02", "2026-01-04", 0.0)
    assert m["days"] == 2
    assert m["ret"] == 0.5 * 1.25 - 1.0  # -0.375
    assert m["dd"] == 0.5  # 1 -> 0.5 -> 0.625
    assert m["trades"] == 4
    assert m["pf"] == 3.0  # (0.25 + 0.5) / 0.25


def test_unseen_bar_edges() -> None:
    ok = {"ret": 0.01, "trades": 30, "pf": 1.2, "dd": 0.5}
    assert kp.unseen_ok(ok)
    assert not kp.unseen_ok(ok | {"ret": 0.0})
    assert not kp.unseen_ok(ok | {"trades": 29})
    assert not kp.unseen_ok(ok | {"pf": 1.19})
    assert not kp.unseen_ok(ok | {"dd": 0.51})


def test_search_bar_edges() -> None:
    assert kp.search_ok({"ret": 0.01, "trades": 30})
    assert not kp.search_ok({"ret": -0.01, "trades": 300})
    assert not kp.search_ok({"ret": 5.0, "trades": 29})
