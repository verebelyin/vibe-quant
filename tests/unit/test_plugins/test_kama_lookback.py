"""KAMA must not be computed before it has the bars pandas_ta needs.

``pandas_ta_classic.kama`` rejects any series shorter than
``max(fast, slow, period)`` with ``slow = 30`` ("Series has N rows but indicator
requires at least 30. Returning None."). The spec's lookback used to be
``3 * period``, so periods 5..9 (gate 15..27 bars) called the indicator up to 15
bars too early: one such warning per bar at the start of every evaluation of a
genome with a short KAMA (discovery runs 863/867/877/878, bd vibe-quant-yul7u.23).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pta_stream_harness import (  # noqa: E402
    compile_indicators,
    random_walk_bars,
    run_stream,
)

from vibe_quant.dsl.indicators import pta_lookback  # noqa: E402
from vibe_quant.dsl.plugins import kama as kama_mod  # noqa: E402
from vibe_quant.dsl.plugins.kama import compute_kama  # noqa: E402
from vibe_quant.dsl.prefix_memo import MEMO  # noqa: E402

SHORT_PERIODS = [5, 6, 7, 8, 9]  # 3 * period < 30 bars
SLOW = 30


@pytest.fixture(autouse=True)
def _clean_memo() -> object:
    MEMO.clear()
    yield
    MEMO.clear()


def _stream(period: int, n_bars: int = 80) -> list[dict[str, float]]:
    spec = {"kama": ("KAMA", {"period": period}, "1h")}
    _, cls = compile_indicators({"kama": {"type": "KAMA", "period": period}})
    return cast("list[dict[str, float]]", run_stream(cls, random_walk_bars(n_bars), False, specs=spec))


@pytest.mark.parametrize("period", SHORT_PERIODS)
@pytest.mark.parametrize("port_ok", [True, False])
def test_short_kama_stream_emits_no_pandas_ta_warning(
    period: int, port_ok: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Real generated feed path; ``port_ok=False`` is the pandas fallback that warns."""
    monkeypatch.setattr(kama_mod, "_kama_port_ok", lambda: port_ok)
    with caplog.at_level(logging.WARNING):
        _stream(period)
    short = [r.getMessage() for r in caplog.records if "indicator requires" in r.getMessage()]
    assert short == []


@pytest.mark.parametrize("period", [*SHORT_PERIODS, 10, 14, 50])
def test_kama_lookback_covers_pandas_ta_minimum(period: int) -> None:
    assert pta_lookback("KAMA", {"period": period}) == max(3 * period, SLOW)


@pytest.mark.parametrize("period", SHORT_PERIODS)
def test_raising_the_gate_changes_no_value(period: int) -> None:
    """Below 30 bars KAMA is all-NaN, so the later gate skips only NaN bars:
    the first stored value lands on bar 30 and every value equals an ungated compute."""
    stream = _stream(period)
    first = next(i for i, snap in enumerate(stream) if "kama" in snap)
    assert first == SLOW - 1
    bars = random_walk_bars(80)
    close = np.array([b[4] for b in bars])
    for i in range(len(bars)):
        got = compute_kama(pd.DataFrame({"close": close[: i + 1]}), {"period": period}).iloc[-1]
        if i < SLOW - 1:
            assert np.isnan(got)
        else:
            assert stream[i]["kama"] == got
