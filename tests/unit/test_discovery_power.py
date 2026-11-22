"""Discovery launch power preflight: the DSR Sharpe bars a run can certify.

The bar is the smallest annual Sharpe (sqrt-252, the DSR unit) whose DSR p-value is
below 0.05 at N trials over T daily observations (skew 0, kurtosis 3). The pinned
points are the journal's 2026-10-18 P-1 S50 table (T = 1700 days).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vibe_quant.discovery import power
from vibe_quant.errors import DataUnavailableError
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio, deannualize_sharpe

if TYPE_CHECKING:
    from pathlib import Path

_WARNING_2_01 = "this run can only certify a Sharpe above 2.01; a realistic edge is 0.5-1.5"


def _p(annual_sharpe: float, n: int, days: int) -> float:
    return (
        DeflatedSharpeRatio()
        .calculate(deannualize_sharpe(annual_sharpe), num_trials=n, num_observations=days)
        .p_value
    )


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, 0.63), (10, 1.24), (100, 1.61), (700, 1.85), (22000, 2.20)],
)
def test_bar_matches_the_journal_points_and_is_the_dsr_boundary(n: int, expected: float) -> None:
    """Two decimals match the journal; the bar passes DSR and 1e-9 below it does not."""
    bar = power.dsr_sharpe_bar(n, 1700)
    assert round(bar, 2) == expected
    assert _p(bar, n, 1700) < 0.05
    assert _p(bar - 1e-9, n, 1700) >= 0.05


@pytest.mark.simple_target(
    reason="the exact 2.0 edge is unreachable through power_preflight inputs (a bisected bar)"
)
def test_warning_only_above_two() -> None:
    assert power.power_warning(2.0) is None
    assert power.power_warning(1.5) is None
    assert power.power_warning(2.01) == _WARNING_2_01


def test_preflight_long_train_low_prior() -> None:
    pre = power.power_preflight(
        train_start="2020-01-01",
        train_end="2024-08-27",
        holdout_start="2024-08-27",
        holdout_end="2025-08-27",
        population=50,
        generations=20,
        prior_n=0,
        require_dsr=True,
    )
    assert pre["sharpe_basis"] == "sqrt252"
    assert pre["train_days"] == 1700
    assert pre["holdout_days"] == 365
    assert pre["holdout_t_sharpe_1"] == pytest.approx(1.2035001862952488, rel=1e-12)
    assert pre["budget"] == 1000
    assert pre["prior_n"] == 0
    assert pre["bar_a"] == {"n_trials": 1, "sharpe": pytest.approx(0.6337293232746865, rel=1e-9)}
    assert pre["bar_b"] == {
        "n_trials": 1000,
        "sharpe": pytest.approx(1.8893504330511675, rel=1e-9),
    }
    assert pre["warning"] is None
    assert pre["line"] == (
        "Power: DSR bar 0.63 at N=1, 1.89 at N=1000 (252-day basis); "
        "holdout 365 days, t 1.20 at Sharpe 1"
    )


def test_preflight_short_train_heavy_prior_warns() -> None:
    pre = power.power_preflight(
        train_start="2023-06-01",
        train_end="2025-06-01",
        holdout_start="2025-06-01",
        holdout_end="2026-01-01",
        population=20,
        generations=15,
        prior_n=22120,
        require_dsr=True,
    )
    assert pre["train_days"] == 731
    assert pre["holdout_days"] == 214
    assert pre["holdout_t_sharpe_1"] == pytest.approx(0.9215239276363633, rel=1e-12)
    assert pre["budget"] == 300
    assert pre["prior_n"] == 22120
    assert pre["bar_a"] == {
        "n_trials": 22121,
        "sharpe": pytest.approx(3.357053974550383, rel=1e-9),
    }
    assert pre["bar_b"] == {
        "n_trials": 22420,
        "sharpe": pytest.approx(3.358912080885554, rel=1e-9),
    }
    assert pre["warning"] == (
        "this run can only certify a Sharpe above 3.36; a realistic edge is 0.5-1.5"
    )


def test_preflight_unknown_prior_uses_run_only_n_and_no_holdout() -> None:
    pre = power.power_preflight(
        train_start="2023-06-01",
        train_end="2025-06-01",
        holdout_start=None,
        holdout_end=None,
        population=20,
        generations=15,
        prior_n=None,
        require_dsr=True,
    )
    assert pre["prior_n"] == "unknown"
    assert pre["bar_a"] == {"n_trials": 1, "sharpe": pytest.approx(0.9673171075340505, rel=1e-9)}
    assert pre["bar_b"] == {"n_trials": 300, "sharpe": pytest.approx(2.674482506492491, rel=1e-9)}
    assert pre["holdout_days"] == 0
    assert pre["holdout_t_sharpe_1"] == 0.0
    assert pre["warning"] is None
    assert pre["line"] == (
        "Power: DSR bar 0.97 at N=1 (prior trials unknown), 2.67 at N=300 "
        "(252-day basis); no holdout"
    )


def test_preflight_dsr_off() -> None:
    pre = power.power_preflight(
        train_start="2023-06-01",
        train_end="2025-06-01",
        holdout_start="2025-06-01",
        holdout_end="2026-01-01",
        population=20,
        generations=15,
        prior_n=5,
        require_dsr=False,
    )
    assert pre["bar_a"] == "DSR off"
    assert pre["bar_b"] == "DSR off"
    assert pre["warning"] is None
    assert pre["line"] == "Power: DSR off; holdout 214 days, t 0.92 at Sharpe 1"


# --- full history: the latest first full UTC day among the symbols ----------


@pytest.fixture
def _ranges(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[datetime, datetime]]:
    """Fake catalog 1m ranges (ts_event = bar CLOSE, inside the open minute)."""
    from vibe_quant.data.catalog import CatalogManager

    ranges: dict[str, tuple[datetime, datetime]] = {
        "BTCUSDT": (
            datetime(2020, 1, 1, 0, 0, 59, 999000, tzinfo=UTC),
            datetime(2026, 11, 10, tzinfo=UTC),
        ),
        "HYPEUSDT": (
            datetime(2025, 5, 30, 10, 30, 59, 999000, tzinfo=UTC),
            datetime(2026, 11, 10, tzinfo=UTC),
        ),
    }

    def _range(self: object, symbol: str, interval: str) -> tuple[datetime, datetime] | None:
        assert interval == "1m"
        return ranges.get(symbol)

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", _range)
    return ranges


def _fake_mark(monkeypatch: pytest.MonkeyPatch, opens: dict[str, datetime]) -> None:
    from vibe_quant.data import mark_bars

    class _Series:
        def __init__(self, first: datetime) -> None:
            self.ts = np.array([int(first.timestamp()) * 1_000_000_000], dtype=np.int64)

    def _load(catalog_path: object, symbol: str) -> object:
        return _Series(opens[symbol]) if symbol in opens else None

    monkeypatch.setattr(mark_bars, "load_mark_series", _load)


def test_full_history_start_is_the_latest_first_full_day(
    tmp_path: Path, _ranges: dict[str, tuple[datetime, datetime]]
) -> None:
    assert power.full_history_start(["BTCUSDT"], tmp_path, mark=False) == "2020-01-01"
    assert power.full_history_start(["BTCUSDT", "HYPEUSDT"], tmp_path, mark=False) == "2025-05-31"


def test_full_history_start_clamps_to_mark_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _ranges: dict[str, tuple[datetime, datetime]],
) -> None:
    _fake_mark(monkeypatch, {"BTCUSDT": datetime(2020, 3, 4, 0, 0, tzinfo=UTC)})
    assert power.full_history_start(["BTCUSDT"], tmp_path, mark=True) == "2020-03-04"
    assert power.full_history_start(["BTCUSDT"], tmp_path, mark=False) == "2020-01-01"
    _fake_mark(monkeypatch, {"BTCUSDT": datetime(2020, 3, 4, 0, 1, tzinfo=UTC)})
    assert power.full_history_start(["BTCUSDT"], tmp_path, mark=True) == "2020-03-05"


def test_full_history_start_names_a_symbol_without_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _ranges: dict[str, tuple[datetime, datetime]],
) -> None:
    with pytest.raises(DataUnavailableError, match="NOPEUSDT"):
        power.full_history_start(["BTCUSDT", "NOPEUSDT"], tmp_path, mark=False)
    _fake_mark(monkeypatch, {})
    with pytest.raises(DataUnavailableError, match="HYPEUSDT: no MARK"):
        power.full_history_start(["HYPEUSDT"], tmp_path, mark=True)
