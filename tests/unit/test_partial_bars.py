"""Guard: the catalog never holds an in-progress aggregated bar (vibe-quant-iz66w)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.data import ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.ingest import update_symbol

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from nautilus_trader.model.data import Bar

_DAY0 = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
_MIN = 60_000
_SYM = "BTCUSDT"


def _klines(start_min: int, end_min: int, skip: range | None = None) -> list[tuple[Any, ...]]:
    """1m klines for minute offsets [start_min, end_min) from _DAY0."""
    out = []
    for m in range(start_min, end_min):
        if skip is not None and m in skip:
            continue
        o = _DAY0 + m * _MIN
        out.append((o, 100.0 + m, 101.0 + m, 99.0, 100.5, 1.0, o + _MIN - 1))
    return out


def _bars_1m(klines: list[tuple[Any, ...]]) -> list[Bar]:
    inst = create_instrument(_SYM)
    return klines_to_bars(
        [
            dict(zip(
                ("open_time", "open", "high", "low", "close", "volume", "close_time"), k, strict=True
            ))
            for k in klines
        ],
        inst.id,
        get_bar_type(_SYM, "1m"),
        inst.size_precision,
        inst.price_precision,
    )


def _agg(klines: list[tuple[Any, ...]], interval: str, minutes: int) -> list[Bar]:
    inst = create_instrument(_SYM)
    return aggregate_bars(
        _bars_1m(klines), get_bar_type(_SYM, interval), minutes, inst.size_precision, inst.price_precision
    )


def test_1d_drops_trailing_partial_day() -> None:
    bars = _agg(_klines(0, 2 * 1440 + 600), "1d", 1440)
    assert len(bars) == 2


def test_4h_drops_trailing_partial() -> None:
    # 58h of data: 14 complete 4h bars + a 2h in-progress one
    bars = _agg(_klines(0, 58 * 60), "4h", 240)
    assert len(bars) == 14
    assert bars[-1].ts_init == _DAY0 * 1_000_000 + 56 * 3_600_000_000_000 - 1_000_000


def test_full_three_days_keeps_all() -> None:
    assert len(_agg(_klines(0, 3 * 1440), "1d", 1440)) == 3


def test_interior_hole_keeps_day() -> None:
    bars = _agg(_klines(0, 3 * 1440, skip=range(300, 330)), "1d", 1440)
    assert len(bars) == 3


def test_rest_in_progress_1m_kline_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    forming_open = now_ms - 10_000
    closed_open = now_ms - 5 * _MIN
    archive = RawDataArchive(tmp_path / "a.db")
    try:
        seed = now_ms - 10 * _MIN
        archive.insert_klines(_SYM, "1m", [(seed, 1.0, 2.0, 0.5, 1.0, 1.0, seed + 59_999)], "seed")
        monkeypatch.setattr(
            ingest,
            "download_recent_klines",
            lambda *a, **k: [
                (closed_open, 1.0, 2.0, 0.5, 1.0, 1.0, closed_open + 59_999),
                (forming_open, 1.0, 2.0, 0.5, 9.0, 1.0, forming_open + 59_999),
            ],
        )
        monkeypatch.setattr(ingest, "download_funding_rates", lambda *a, **k: [])
        update_symbol(_SYM, archive=archive, catalog=_Catalog(), verbose=False)
        opens = {r["open_time"] for r in archive.get_klines(_SYM, "1m")}
        assert closed_open in opens
        assert forming_open not in opens
    finally:
        archive.close()


class _Catalog:
    def __init__(self) -> None:
        self.written: list[Bar] = []

    def write_instrument(self, *a: object, **k: object) -> None: ...
    def clear_bar_data(self, *a: object, **k: object) -> None: ...

    def write_bars(self, bars: list[Bar], *a: object, **k: object) -> None:
        self.written.extend(bars)


def test_update_then_complete_writes_completed_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second update after the day completes must produce the day-1 1d bar."""
    monkeypatch.setattr(ingest, "download_funding_rates", lambda *a, **k: [])
    day_ms = 1440 * _MIN
    # Fixed 'now' far in the future so every kline counts as closed.
    archive = RawDataArchive(tmp_path / "b.db")
    try:
        archive.insert_klines(_SYM, "1m", _klines(0, 1440 + 600), "seed")
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: _klines(1440 + 600, 1440 + 601))
        cat1 = _Catalog()
        update_symbol(_SYM, archive=archive, catalog=cat1, verbose=False)
        d1 = [b for b in cat1.written if "1-DAY" in str(b.bar_type)]
        assert len(d1) == 1  # day 0 only; day 1 in progress

        monkeypatch.setattr(
            ingest, "download_recent_klines", lambda *a, **k: _klines(1440 + 601, 2 * 1440)
        )
        cat2 = _Catalog()
        update_symbol(_SYM, archive=archive, catalog=cat2, verbose=False)
        d2 = [b for b in cat2.written if "1-DAY" in str(b.bar_type)]
        assert len(d2) == 2
        assert d2[-1].ts_event == (_DAY0 + day_ms) * 1_000_000
    finally:
        archive.close()
