"""NC1 trend-null calibration driver (arch-c9-2 §5, rulings R-c9t5-1/2).

``scripts/research/trend_null_calibration.py`` evaluates the pre-registered long_flat
arm once per NC1 lag through ``research.evaluate`` and counts X = #{null_percentile
>= 0.95}. The heavy load and evaluate are faked here; the tests pin what the driver
forwards (spec, lags, draws, seed, family) and what it writes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.portfolio import research
from vibe_quant.portfolio.spec import CarrySpec
from vibe_quant.portfolio.ticks import CarryInputs

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/trend_null_calibration.py"

_NC1_SPEC = CarrySpec(
    symbols=("BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"),
    start_ns=1685577600000000000,  # 2023-06-01
    end_ns=1790553600000000000,  # 2026-09-28
    initial_balance=100_000.0,
    gross=1.0,
    band=0.10,
    weighting="time_series",
    vol_lookback_days=30,
    signal="trend",
    schedule="daily",
    fill_granularity="1m",
    direction="long_flat",
    vol_target=0.40,
    spread=0.0001,
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trend_null_calibration", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(spec: CarrySpec, closes: dict[str, dict[int, float]]) -> CarryInputs:
    return CarryInputs(
        spec=spec,
        decisions=(),
        instruments={},
        specs={},
        ticks={},
        settlements={},
        marks={},
        calc=None,  # type: ignore[arg-type]  # never read by the driver
        boundary_prices={},
        trend_closes=closes,
    )


def _driver(
    monkeypatch: pytest.MonkeyPatch, percentiles: dict[int, float | None]
) -> tuple[ModuleType, list[dict[str, Any]]]:
    module = _load_script()
    calls: list[dict[str, Any]] = []
    inputs = SimpleNamespace(spec=_NC1_SPEC)

    def evaluate(given: object, **kwargs: Any) -> dict[str, Any]:
        assert given is inputs
        calls.append(kwargs)
        lag = kwargs["signal_lag_days"]
        return {"null_percentile": percentiles[lag], "sharpe": lag / 1000, "lag_mask_mismatches": 0}

    monkeypatch.setattr(module, "load_nc1_inputs", lambda catalog, archive: inputs)
    monkeypatch.setattr(module.research, "evaluate", evaluate)
    return module, calls


@pytest.mark.parametrize(
    ("flagged_value", "rest", "expected"),
    [(0.95, 0.9499, "FAIL"), (None, 0.0, "FAIL"), (0.95, 0.9499, "PASS")],
)
def test_driver_forwards_every_lag_and_counts_x(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flagged_value: float | None,
    rest: float,
    expected: str,
) -> None:
    """Six flagged lags (0.95 or an undefined percentile) -> FAIL; five -> PASS. Each of
    the 50 lags is evaluated once with draws 100, seed 0, the NC1 spec's own arm as the
    family, and its percentile lands in nc1.json and the stdout verdict line."""
    count = 5 if expected == "PASS" else 6
    lags = research.NC1_LAGS_DAYS
    percentiles = {lag: (flagged_value if i < count else rest) for i, lag in enumerate(lags)}
    module, calls = _driver(monkeypatch, percentiles)

    assert module.main(["--workers", "1", "--out", str(tmp_path)]) == 0

    assert [call["signal_lag_days"] for call in calls] == list(lags)
    assert {(call["draws"], call["seed"], call["criteria"]) for call in calls} == {(100, 0, None)}
    assert {call["family"].arms for call in calls} == {(_NC1_SPEC.arm_hash(),)}
    assert all(call["trials_path"].parent == tmp_path for call in calls)
    summary = json.loads((tmp_path / "nc1.json").read_text())
    assert summary["flagged"] == count
    assert summary["verdict"] == expected
    assert summary["label"] == "calibration"
    assert summary["spec"] == _NC1_SPEC.to_json()
    assert [row["lag_days"] for row in summary["lags_detail"]] == list(lags)
    assert [row["null_percentile"] for row in summary["lags_detail"]] == [
        percentiles[lag] for lag in lags
    ]
    assert f"NC1: X={count}/50 fail_at=6 verdict={expected}" in capsys.readouterr().out


def test_driver_reuses_finished_lags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rerun skips every lag whose result file matches spec, draws and seed (resume)."""
    lags = research.NC1_LAGS_DAYS
    module, calls = _driver(monkeypatch, dict.fromkeys(lags, 0.5))
    assert module.main(["--workers", "1", "--out", str(tmp_path)]) == 0
    calls.clear()
    stale = tmp_path / f"lag_{lags[3]}.json"
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "draws": 99}))

    assert module.main(["--workers", "1", "--out", str(tmp_path)]) == 0

    assert [call["signal_lag_days"] for call in calls] == [lags[3]]


def test_loader_reads_closes_back_to_the_longest_lag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run is the NC1 spec; the closes come from the same spec loaded 708 days earlier."""
    module = _load_script()
    loads: list[CarrySpec] = []
    base = {"BTCUSDT": {10: 1.0}}
    longer = {"BTCUSDT": {5: 0.5, 10: 1.0}}

    def load(spec: CarrySpec, catalog: Path, archive: Path) -> CarryInputs:
        loads.append(spec)
        return _inputs(spec, base if len(loads) == 1 else longer)

    monkeypatch.setattr(module, "load_carry_inputs", load)

    inputs = module.load_nc1_inputs(Path("c"), Path("a"))

    assert loads == [_NC1_SPEC, research.lag_spec(_NC1_SPEC, 708)]
    assert inputs.spec == _NC1_SPEC
    assert inputs.trend_closes == longer


@pytest.mark.parametrize("workers", ["0", "5"])
def test_workers_capped_at_four(workers: str) -> None:
    """The CPU is shared with another research job: 1..4 workers, else exit 2."""
    module = _load_script()
    with pytest.raises(SystemExit) as raised:
        module.main(["--workers", workers])
    assert raised.value.code == 2
