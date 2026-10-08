"""NTBacktestFn failure dict carries the exception marker (vibe-quant-4h9l8)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vibe_quant.discovery.backtest_fn import NTBacktestFn


def test_failure_dict_contains_error(monkeypatch: Any) -> None:
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01")

    def boom(*_a: Any, **_k: Any) -> dict[str, float | int]:
        raise OSError("catalog missing")

    monkeypatch.setattr(fn, "_run_single", boom)
    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]
    assert str(out["error"]) == "OSError: catalog missing"
    assert out["total_trades"] == 0
    assert out["sharpe_ratio"] == -1.0
