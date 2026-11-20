"""Daily 1m quote volume from the raw archive (carry liquidity filter, x0axf.14 T5).

Contract row R34 (day keys are UTC day starts in ns) plus the archive half of
R30 (1m filter, malformed rows). Every test uses a tmp archive only
(``gotcha:real-archive-in-tests``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.data.archive import RawDataArchive

if TYPE_CHECKING:
    from pathlib import Path

_DAY_MS = 86_400_000


def _archive(tmp_path: Path, rows: dict[str, list[tuple[int, object]]]) -> RawDataArchive:
    """A tmp archive with ``{interval: [(open_time_ms, quote_volume), ...]}`` BTCUSDT rows."""
    archive = RawDataArchive(tmp_path / "raw_data.db")
    for interval, items in rows.items():
        archive.insert_klines(
            "BTCUSDT",
            interval,
            [(t, 1.0, 1.0, 1.0, 1.0, 1.0, t + 59_999, qv) for t, qv in items],
            "test",
        )
    return archive


def test_daily_quote_volume_keys(tmp_path: Path) -> None:
    """R34: open_time 86 400 000 ms -> key 86 400 000 000 000 ns; sums and counts per day."""
    archive = _archive(
        tmp_path,
        {
            "1m": [
                (_DAY_MS, 2.5),
                (_DAY_MS + 60_000, 4.0),
                (2 * _DAY_MS - 60_000, 1.5),  # last minute of the same UTC day
                (2 * _DAY_MS, 10.0),  # first minute of the next day
                (5 * _DAY_MS + 7, 3),  # an INTEGER-typed volume is numeric
            ],
        },
    )

    days = archive.get_daily_quote_volume("BTCUSDT", 0, 10 * _DAY_MS)

    assert days == {
        86_400_000_000_000: (8.0, 3),
        172_800_000_000_000: (10.0, 1),
        432_000_000_000_000: (3.0, 1),
    }
    # the window is [start, end) on open_time
    assert archive.get_daily_quote_volume("BTCUSDT", _DAY_MS + 1, 2 * _DAY_MS) == {
        86_400_000_000_000: (5.5, 2)
    }
    assert archive.get_daily_quote_volume("BTCUSDT", 0, _DAY_MS) == {}
    assert archive.get_daily_quote_volume("ETHUSDT", 0, 10 * _DAY_MS) == {}
    archive.close()


def test_daily_quote_volume_ignores_other_intervals(tmp_path: Path) -> None:
    """A same-minute 1d row adds neither volume nor coverage (R30)."""
    archive = _archive(
        tmp_path,
        {"1m": [(_DAY_MS, 2.0)], "1d": [(_DAY_MS, 1e12), (3 * _DAY_MS, 1e12)]},
    )
    assert archive.get_daily_quote_volume("BTCUSDT", 0, 10 * _DAY_MS) == {
        86_400_000_000_000: (2.0, 1)
    }
    archive.close()


@pytest.mark.parametrize(
    ("bad_rows", "message"),
    [
        ([(_DAY_MS + 60_000, None)], "1 malformed 1m quote_volume row"),
        ([(_DAY_MS + 60_000, "abc")], "1 malformed 1m quote_volume row"),
        # a negative row offset by positives still fails (sum stays positive)
        ([(_DAY_MS + 60_000, -1.0), (_DAY_MS + 120_000, 5.0)], "1 malformed 1m quote_volume row"),
        ([(_DAY_MS + 60_000, 1.7e308), (_DAY_MS + 120_000, 1.7e308)], "non-finite 1m quote_volume"),
    ],
)
def test_daily_quote_volume_malformed(
    tmp_path: Path, bad_rows: list[tuple[int, object]], message: str
) -> None:
    """NULL / text / negative rows and a non-finite day sum raise, naming symbol + UTC day."""
    archive = _archive(tmp_path, {"1m": [(_DAY_MS, 1.0), *bad_rows], "1d": [(0, 1.0)]})
    with pytest.raises(ValueError, match=rf"^BTCUSDT: {message}.* on 1970-01-02 UTC$"):
        archive.get_daily_quote_volume("BTCUSDT", 0, 10 * _DAY_MS)
    # a zero-volume row is valid data, not malformed
    zero = RawDataArchive(tmp_path / "zero.db")
    zero.insert_klines("BTCUSDT", "1m", [(_DAY_MS, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 0.0)], "test")
    assert zero.get_daily_quote_volume("BTCUSDT", 0, 10 * _DAY_MS) == {
        86_400_000_000_000: (0.0, 1)
    }
    archive.close()
    zero.close()
