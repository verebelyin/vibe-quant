"""R1 parity and R20 per-bar cost of warm paper against the validation replay.

Bead ``vibe-quant-3g81l`` acceptance test (design R2.4 rows R1/R20, R2.5 T5): a paper
session that seeds 8,000 catalog bars through ``_warm_from`` and then takes live bars must
agree with a validation replay of the SAME compiled strategy that has run since
2024-01-01. Parity is the user's Q-A rule: every output within ``1e-9`` RELATIVE to
``max(|validation|, median |output| over the seed)``, plus identical ``dir`` and
``sign(close - output)``; and the per-bar cost of the warm session must stay within 10 ms
(p50) of a cold session that already holds 8,000 bars.

Two seams are exercised, both production ones:

* the reference is the validation path (``nt_runner.install_pta_precompute``): the seed
  columns are replayed by ``pta_replay`` once and fed bar by bar. Without it a 15m
  full-history PMAX+SUPERTREND replay is O(n^2) over the whole history (hours), and the
  replay is the runtime stream's exact value for a causal ``compute_fn``.
* the paper side calls ``_warm_from`` (T2) and lets every live bar take the live
  ``_feed_pta_buffer`` recompute path over the 8,000-row buffer.

The cold-start control runs the same live bars with no seed and must FAIL the same
assertion, so the test has teeth.
"""

from __future__ import annotations

import math
import statistics
import time
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
import pytest

from tests.unit.test_fill_timing import _INSTRUMENT_ID, _new_engine
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.pta_replay import PtaPrecomputed, strategy_pta_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nautilus_trader.model.data import Bar

#: Q-A tolerance: relative, scaled by max(|validation|, median |output| over the seed).
_TOL = 1e-9
#: Design Q2: the paper seed is 8,000 bars (5.6 d at 1m, 83 d at 15m).
_SEED_BARS = 8000
_LIVE_BARS = 500
#: R1: T = 2025-07-10 plus two starts spread over 2025-2026 (the catalog is contiguous).
_T_STARTS = ("2025-07-10", "2025-12-25", "2026-07-15")
_T_P2 = "2025-07-10"
#: R1/R20 reference: the validation replay starts here (design R2.4 R1).
_VALIDATION_START = datetime(2024, 1, 1, tzinfo=UTC)
_OHLCV = ("open", "high", "low", "close", "volume")

_IND_15M: dict[str, dict[str, Any]] = {
    "pmax": {"type": "PMAX"},
    "st": {"type": "SUPERTREND"},
    "ema": {"type": "EMA", "period": 200},
}
_IND_1M: dict[str, dict[str, Any]] = {"pmax": {"type": "PMAX"}, "st": {"type": "SUPERTREND"}}
_LONG = ["pmax_dir > 0", "st_dir > 0"]
_SHORT = ["pmax_dir < 0", "st_dir < 0"]
_EXIT_LONG = ["pmax_dir < 0"]


class Row(NamedTuple):
    """One bar as the strategy left it, plus the signal inputs it was evaluated on."""

    #: ``ts_event`` of the bar (the live bar's close).
    ts: int
    #: ``_prev_values`` BEFORE this bar's ``_update_prev_values``: what ``_check_*`` read.
    prev: dict[str, float]
    #: ``_pta_values`` after the bar (every output of the compute_fn indicators).
    values: dict[str, float]
    #: The one NT indicator (EMA(200)) after the bar.
    ema: float
    close: float
    #: Wall time of ``on_bar``; 0.0 unless the probe is a timed one (P2).
    seconds: float = 0.0


def _ns(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()) * 1_000_000_000


@cache
def _compile(timeframe: str, indicators_key: str) -> tuple[type, type]:
    indicators = _IND_1M if indicators_key == "1m" else _IND_15M
    dsl = validate_strategy_dict(
        {
            "name": "warm_parity",
            "timeframe": timeframe,
            "indicators": dict(indicators),
            "entry_conditions": {"long": _LONG, "short": _SHORT},
            "exit_conditions": {"long": _EXIT_LONG},
            "stop_loss": {"type": "fixed_pct", "percent": 50.0},
            "take_profit": {"type": "fixed_pct", "percent": 50.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    return getattr(module, f"{camel}Strategy"), getattr(module, f"{camel}Config")


def _probe(base: type, *, timed: bool = False) -> type:
    """``base`` recording the per-bar state, and (``timed``) the ``on_bar`` wall time."""

    class Probe(base):  # type: ignore[valid-type, misc]
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.rows: list[Row] = []

        def on_bar(self, bar: Bar) -> None:
            started = time.perf_counter()
            prev = dict(self._prev_values)
            super().on_bar(bar)
            seconds = time.perf_counter() - started if timed else 0.0
            ema = getattr(self, "ind_ema", None)
            self.rows.append(
                Row(
                    ts=int(bar.ts_event),
                    prev=prev,
                    values=dict(self._pta_values),
                    ema=float(ema.value) if ema is not None else math.nan,
                    close=float(bar.close),
                    seconds=seconds,
                )
            )

    return Probe


def _run(strategy: Any, bars: Sequence[Bar], tf: str, pre: PtaPrecomputed | None = None) -> None:
    if pre is not None:
        strategy._pta_pre = {tf: pre}
    engine = _new_engine()
    engine.add_data(list(bars))
    engine.add_strategy(strategy)
    try:
        engine.run()
    finally:
        engine.reset()
        engine.dispose()


def _precompute(strategy: Any, bars: Sequence[Bar], tf: str) -> PtaPrecomputed:
    """The validation runner's seam (``nt_runner.install_pta_precompute``) for ``bars``."""
    frame = pd.DataFrame({field: [float(getattr(b, field)) for b in bars] for field in _OHLCV})
    columns = strategy_pta_columns(
        strategy._pta_params, type(strategy)._PTA_META, strategy._pta_buffer_cap, tf, frame
    )
    return PtaPrecomputed(
        [int(b.ts_init) for b in bars],
        [float(b.close) for b in bars],
        [(key, column.tolist()) for key, column in columns],
    )


@cache
def _bars(tf: str) -> list[Bar]:
    if tf == "15m":
        return CatalogManager().get_bars("BTCUSDT", tf, start=_VALIDATION_START)
    start = datetime.fromisoformat(_T_P2).replace(tzinfo=UTC) - timedelta(minutes=_SEED_BARS + 60)
    end = datetime.fromisoformat(_T_P2).replace(tzinfo=UTC) + timedelta(minutes=_LIVE_BARS + 60)
    return CatalogManager().get_bars("BTCUSDT", tf, start=start, end=end)


def _index_at(bars: Sequence[Bar], day: str) -> int:
    """Index of the last bar closing at or before ``day`` (a T start)."""
    return max(i for i, bar in enumerate(bars) if bar.ts_event <= _ns(day))


def _window(tf: str, day: str) -> tuple[list[Bar], list[Bar]]:
    """``(seed, live)``: the 8,000 bars ending at T and the 500 bars after them."""
    bars = _bars(tf)
    idx = _index_at(bars, day)
    seed = bars[idx - _SEED_BARS + 1 : idx + 1]
    live = bars[idx + 1 : idx + 1 + _LIVE_BARS]
    assert len(seed) == _SEED_BARS, f"seed holds {len(seed)} bars at T={day}"
    assert len(live) == _LIVE_BARS, f"live window holds {len(live)} bars after T={day}"
    step = int(seed[1].ts_event - seed[0].ts_event)
    previous = seed[0]
    for bar in (*seed[1:], *live):
        assert int(bar.ts_event - previous.ts_event) == step, (
            f"catalog gap after {previous.ts_event}"
        )
        previous = bar
    return seed, live


def _seed_of(bars: Sequence[Bar]) -> dict[str, list[Bar]]:
    return {str(bars[0].bar_type): list(bars)}


@cache
def _reference() -> dict[int, Row]:
    """The validation replay from 2024-01-01 (PTA precompute installed, warm seam off)."""
    bars = _bars("15m")
    last = max(_index_at(bars, day) for day in _T_STARTS) + _LIVE_BARS
    replay = bars[: last + 1]
    base, config_cls = _compile("15m", "15m")
    strategy = _probe(base)(config=config_cls(**_params()))
    pre = _precompute(strategy, replay, "15m")
    _run(strategy, replay, "15m", pre)
    assert pre.fed == pre.n, f"validation precompute fed {pre.fed} of {pre.n} bars"
    assert len(strategy.rows) == len(replay)
    return {row.ts: row for row in strategy.rows}


def _params() -> dict[str, Any]:
    return {"instrument_id": _INSTRUMENT_ID, "execution_delay_probability": 0.0}


def _outputs(row: Row) -> dict[str, float]:
    """Every output the signals see on this bar: ``_pta_values`` plus the NT indicator."""
    return {**row.values, "ema": row.ema}


def _scales(reference: dict[int, Row], seed: Sequence[Bar]) -> dict[str, float]:
    """Q-A scale per key: the median |output| over the seed's bars."""
    samples: dict[str, list[float]] = {}
    for bar in seed:
        for key, value in _outputs(reference[int(bar.ts_event)]).items():
            if not math.isnan(value):
                samples.setdefault(key, []).append(abs(value))
    return {key: statistics.median(values) for key, values in samples.items()}


def _rel(paper: float, validation: float, scale: float) -> float:
    paper_nan = math.isnan(paper)
    validation_nan = math.isnan(validation)
    if paper_nan or validation_nan:
        return 0.0 if paper_nan and validation_nan else math.inf
    denom = max(abs(validation), scale)
    if denom == 0.0:
        return 0.0 if paper == validation else math.inf
    return abs(paper - validation) / denom


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _compare(
    label: str,
    paper: dict[str, float],
    validation: dict[str, float],
    scales: dict[str, float],
    problems: list[str],
) -> None:
    if paper.keys() != validation.keys():
        problems.append(f"{label}: keys {sorted(paper)} != {sorted(validation)}")
    for key in paper.keys() & validation.keys():
        rel = _rel(paper[key], validation[key], scales.get(key, 0.0))
        if rel > _TOL:
            problems.append(f"{label} {key}: {paper[key]!r} vs {validation[key]!r} rel {rel:.3e}")


def _assert_parity(
    paper: Sequence[Row], reference: dict[int, Row], scales: dict[str, float]
) -> None:
    """R1: every output within the Q-A tolerance, identical ``dir`` and ``sign(close - out)``."""
    problems: list[str] = []
    for row in paper:
        validation = reference[row.ts]
        _compare(f"bar {row.ts} out", _outputs(row), _outputs(validation), scales, problems)
        _compare(f"bar {row.ts} signal-in", row.prev, validation.prev, scales, problems)
        for key, value in row.values.items():
            if key.endswith("_dir"):
                if value != validation.values[key]:
                    problems.append(
                        f"bar {row.ts} dir {key}: {value!r} vs {validation.values[key]!r}"
                    )
                continue
            if _sign(row.close - value) != _sign(validation.close - validation.values[key]):
                problems.append(
                    f"bar {row.ts} sign {key}: close {row.close!r} vs "
                    f"{value!r} / {validation.close!r} vs {validation.values[key]!r}"
                )
    assert not problems, f"{len(problems)} parity problem(s): " + " | ".join(problems[:5])


# ---------------------------------------------------------------------------
# R1 -- warm paper == validation replay, at 3 starts
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="real NT engine: 8,000-bar seed replay + 500 live 15m bars per start")
@pytest.mark.parametrize("start", _T_STARTS)
def test_warm_paper_matches_validation_after_seed(start: str) -> None:
    """R1: seed 8,000 catalog bars ending at T, then live bars T+1..T+500."""
    seed, live = _window("15m", start)
    base, config_cls = _compile("15m", "15m")
    paper = _probe(base)(config=config_cls(**_params()))
    paper._warm_from(_seed_of(seed))
    assert paper._warm_wm == {seed[0].bar_type: int(seed[-1].ts_event)}
    _run(paper, live, "15m")

    assert len(paper.rows) == _LIVE_BARS
    assert paper.ind_ema.count == _SEED_BARS + _LIVE_BARS
    assert (paper._warm_dropped, paper._warm_late_gap_bars, len(paper._warm_pending)) == (0, 0, 0)
    _assert_parity(paper.rows, _reference(), _scales(_reference(), seed))


@pytest.mark.slow(reason="real NT engine over the catalog: the validation replay reference")
def test_cold_paper_at_start_diffs_from_validation() -> None:
    """Control: without the seed the SAME assertion fails (the parity test has teeth)."""
    seed, live = _window("15m", _T_STARTS[0])
    base, config_cls = _compile("15m", "15m")
    cold = _probe(base)(config=config_cls(**_params()))
    _run(cold, live, "15m")

    assert cold._warm_wm is None
    assert cold.ind_ema.count == _LIVE_BARS
    reference = _reference()
    scales = _scales(reference, seed)
    with pytest.raises(AssertionError) as raised:
        _assert_parity(cold.rows, reference, scales)
    assert f"bar {live[0].ts_event}" in str(raised.value)
    # a value-level divergence too, not only the missing (NaN-hold) early keys
    assert _rel(cold.rows[-1].ema, reference[cold.rows[-1].ts].ema, scales["ema"]) > _TOL


# ---------------------------------------------------------------------------
# R20 / P2 -- warm per-bar cost vs cold after 8,000 bars
# ---------------------------------------------------------------------------


@pytest.mark.timing
@pytest.mark.slow(reason="8,000-bar 1m cold history + 500 live bars on each side")
def test_warm_per_bar_cost_within_10ms_of_cold() -> None:
    """R20/P2: on_bar p50 of a warm 1m session is within 10 ms of cold-after-8,000.

    PREDICTION (design R2.4 R20, written before measuring): delta <= 0.1 ms -- warm mode
    feeds the same compute_fn buffers by hand and adds one dict lookup per bar.
    """
    seed, live = _window("1m", _T_P2)
    base, config_cls = _compile("1m", "1m")
    probe = _probe(base, timed=True)

    warm = probe(config=config_cls(**_params()))
    warm._warm_from(_seed_of(seed))
    _run(warm, live, "1m")
    assert not hasattr(warm, "ind_ema")  # no NT indicator in this strategy
    assert len(warm._pta_bufs["1m"]) == _SEED_BARS + _LIVE_BARS

    cold = probe(config=config_cls(**_params()))
    # The cold session's first 8,000 bars go through the validation precompute seam so the
    # measured window starts on the same buffer state instead of an O(n^2) replay; the
    # seam stops at bar 8,000 and every measured bar takes the live recompute path.
    pre = _precompute(cold, seed, "1m")
    _run(cold, [*seed, *live], "1m", pre)
    assert (pre.n, pre.fed, pre.failed_at) == (_SEED_BARS, _SEED_BARS, _SEED_BARS)
    assert len(cold._pta_bufs["1m"]) == _SEED_BARS + _LIVE_BARS

    warm_p50 = statistics.median(row.seconds for row in warm.rows) * 1000.0
    cold_p50 = statistics.median(row.seconds for row in cold.rows[-_LIVE_BARS:]) * 1000.0
    print(
        f"\nP2 per-bar on_bar p50: warm {warm_p50:.3f} ms, cold-after-8000 {cold_p50:.3f} ms, "
        f"delta {abs(warm_p50 - cold_p50):.3f} ms (limit 10 ms)"
    )
    assert abs(warm_p50 - cold_p50) <= 10.0, (warm_p50, cold_p50)
