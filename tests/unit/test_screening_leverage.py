"""One leverage, every mode: the screening venue and the liquidation scan follow ``leverage``.

Contract rows C1-C5 and C13 (job 20261014-honest-lev, bead vibe-quant-6wq6w). Today the
screening venue ignores the value unless liquidation is on; after T1 the venue runs at it in
EVERY mode and mode off never imports the liquidation package.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_overfitting_liquidation import _NoTicks, _seed_run
from tests.unit.test_screening_liquidation import CALM, extract, write_marks
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    LiquidationConfigError,
    LiquidationSettings,
)
from vibe_quant.screening.nt_runner import NTScreeningRunner

if TYPE_CHECKING:
    from pathlib import Path

# ONE bad value: the leverage domain itself is tabled at its owner
# (tests/unit/test_liquidation_settings.py); what this file pins is that the runner
# builds the settings — and so refuses — in EVERY mode.
BAD_LEVERAGE: object = float("nan")


def _runner(
    *, mode: str = "off", leverage: object = DEFAULT_LEVERAGE, catalog: Path | None = None
) -> NTScreeningRunner:
    runner = NTScreeningRunner(
        {},
        ["BTCUSDT"],
        "2025-01-01",
        "2025-01-04",
        catalog_path=str(catalog) if catalog else None,
        funding_archive_path="/nonexistent/funding.db",
        launch=LaunchSettings(
            liquidation=LiquidationSettings(mode=mode, leverage=leverage),  # type: ignore[arg-type]
        ),
    )
    runner._all_timeframes = {"1m"}
    runner._resolved_catalog_path = catalog  # type: ignore[assignment]
    return runner


# ------------------------------------------------------------------ C1: default


def test_default_venue_is_ten() -> None:
    """C1: no leverage key -> the venue runs at the default 10x."""
    runner = _runner()
    assert runner._liquidation_mode == "off"
    assert runner._leverage == 10.0
    assert runner._venue_config().default_leverage == Decimal("10")
    assert runner.last_liquidation_notes is None


# ------------------------------------------------------------- C2: off follows


def test_off_leverage_reaches_the_venue() -> None:
    """C2: leverage 3 with the mode off -> the venue runs at 3, not the fixed 10."""
    runner = _runner(mode="off", leverage=3)
    assert runner._liquidation_mode == "off"
    assert runner._leverage == 3.0
    assert runner._venue_config().default_leverage == Decimal("3")
    assert runner.last_liquidation_notes is None


def test_off_leverage_never_imports_the_liquidation_package() -> None:
    """C2: a mode-off run at 3x still never imports ``vibe_quant.liquidation.*``."""
    code = (
        "import sys, time\n"
        "from types import SimpleNamespace as NS\n"
        "from decimal import Decimal\n"
        "from vibe_quant.launch_settings import LaunchSettings\n"
        "from vibe_quant.liquidation_settings import LiquidationSettings\n"
        "from vibe_quant.screening.nt_runner import NTScreeningRunner\n"
        "r = NTScreeningRunner({}, ['BTCUSDT'], '2025-01-01', '2025-01-02', "
        "launch=LaunchSettings(liquidation=LiquidationSettings(leverage=3)))\n"
        "assert r._venue_config().default_leverage == Decimal('3')\n"
        "cache = NS(positions=list, position_snapshots=list, bars=list)\n"
        "engine = NS(kernel=NS(cache=cache))\n"
        "bt = NS(total_positions=0, elapsed_time=0.0, stats_pnls={}, stats_returns={})\n"
        "r._extract_metrics({}, bt, engine, time.time(), 1000.0)\n"
        "bad = sorted(m for m in sys.modules if m == 'vibe_quant.liquidation' "
        "or m.startswith('vibe_quant.liquidation.'))\n"
        "assert not bad, bad\n"
        "assert r.last_liquidation_notes is None\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


# ------------------------------------------------------- C3: the scan sees it


def test_isolated_scan_uses_the_leverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C3: leverage 20 with the mode isolated -> venue 20 AND the scan gets leverage 20."""
    import vibe_quant.liquidation.apply as apply_module

    catalog = write_marks(tmp_path, CALM)
    captured: list[Any] = []
    real = apply_module.apply_liquidation

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["leverage"])
        return real(*args, **kwargs)

    monkeypatch.setattr(apply_module, "apply_liquidation", spy)
    runner = _runner(mode="isolated", leverage=20, catalog=catalog)
    assert runner._venue_config().default_leverage == Decimal("20")
    extract(runner, [], CALM, n_positions=0)
    assert captured == [20.0]


# ------------------------------------------------------------- C4: lower bound


def test_leverage_one_off_is_accepted() -> None:
    """C4: the [1, 125] lower bound is a valid venue leverage with the mode off."""
    runner = _runner(mode="off", leverage=1)
    assert runner._leverage == 1.0
    assert runner._venue_config().default_leverage == Decimal("1")


# ------------------------------------------------------------ C5: bad leverage


@pytest.mark.parametrize("mode", ["off", "isolated", "cross"])
def test_bad_leverage_is_rejected_in_every_mode(mode: str) -> None:
    """C5: a bad leverage fails before any engine, in EVERY mode (one wiring value)."""
    with pytest.raises(LiquidationConfigError, match=r"\[1, 125\]"):
        _runner(mode=mode, leverage=BAD_LEVERAGE)


# ------------------------------------------------- C13: overfitting reruns


def test_wfa_rerun_uses_the_stored_off_leverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C13: a WFA rerun of an off/3 screening run builds NTScreeningRunner at leverage 3."""
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    captured: list[dict[str, object]] = []

    class _Cap:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

        def __call__(self, _params: dict[str, object]) -> object:
            return type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.1})()

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Cap)
    monkeypatch.setattr(sp, "ParentFillTicks", lambda *a, **k: _NoTicks())
    db = _seed_run(tmp_path, parameters={"leverage": 3})
    NTWFARunner(run_id=42, db_path=db).backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})

    assert captured[0]["launch"].liquidation.leverage == 3.0  # type: ignore[union-attr]
    assert captured[0]["launch"].fee_schedule.is_default  # type: ignore[union-attr]


def test_cv_rerun_uses_the_stored_off_leverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C13: the CV rerun reads the same stored off/3 leverage."""
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    captured: list[dict[str, object]] = []

    class _Cap:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

        def __call__(self, _params: dict[str, object]) -> object:
            return type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.1})()

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Cap)
    monkeypatch.setattr(sp, "ParentFillTicks", lambda *a, **k: _NoTicks())
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    db = _seed_run(tmp_path, parameters={"leverage": 3})
    runner = NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=tmp_path / "catalog")
    runner._backtest("2024-02-01", "2024-03-01")  # noqa: SLF001

    assert captured[0]["launch"].liquidation.leverage == 3.0  # type: ignore[union-attr]
    assert captured[0]["launch"].fee_schedule.is_default  # type: ignore[union-attr]
