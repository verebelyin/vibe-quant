"""Fill-tick sets: 2 ticks per strategy-bar boundary at 1m closes (vibe-quant-yul7u.8)."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from nautilus_trader.model.data import Bar, TradeTick
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.data.fill_ticks import (
    FILL_TICK_SIZE,
    FILL_TICK_VERSION,
    build_fill_ticks,
    resolve_fill_ticks,
)
from vibe_quant.screening.nt_runner import MissingBarDataError

MIN = 60_000_000_000
HOUR = 60 * MIN
DAY0 = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z
INIT = 59_999_000_000  # 1m bar ts_init = open + 59.999s


def _series(opens: list[int], closes: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        ts=np.array([o + INIT for o in opens], dtype=np.int64),
        close=np.array(closes, dtype=np.float64),
    )


def _write_catalog(
    root: Path, opens: list[int], instrument: Any = None, symbol: str = "BTCUSDT"
) -> Path:
    """1m bars (close = 100 + minute index) + instrument in a catalog at ``root``."""
    inst = instrument or create_instrument(symbol)
    bt = get_bar_type(symbol, "1m")
    bars = []
    for n, o in enumerate(opens):
        px = Price(100.0 + n, inst.price_precision)
        bars.append(Bar(bt, px, px, px, px, Quantity(1, 0), o, o + INIT))
    cat = ParquetDataCatalog(str(root))
    cat.write_data([inst])
    cat.write_data(bars)
    return root


def _minutes(start: int, n: int) -> list[int]:
    return [start + i * MIN for i in range(n)]


def test_build_fill_ticks_two_per_boundary_at_minute_closes() -> None:
    step = 4 * HOUR
    # boundaries at DAY0 and DAY0+4h, a few unrelated minutes in between
    opens = [
        DAY0,
        DAY0 + MIN,
        DAY0 + 2 * MIN,
        DAY0 + step,
        DAY0 + step + MIN,
        DAY0 + step + 2 * MIN,
    ]
    ser = _series(opens, [10.1, 10.2, 10.3, 20.1, 20.2, 20.3])
    ticks, gaps = build_fill_ticks(ser, "4h", create_instrument("BTCUSDT"))
    assert gaps == 0
    assert [t.ts_init for t in ticks] == [
        DAY0 + INIT,
        DAY0 + MIN + INIT,
        DAY0 + step + INIT,
        DAY0 + step + MIN + INIT,
    ]
    assert [float(t.price) for t in ticks] == [10.1, 10.2, 20.1, 20.2]
    assert all(float(t.size) == FILL_TICK_SIZE for t in ticks)
    assert all(isinstance(t, TradeTick) and t.ts_event == t.ts_init for t in ticks)


def test_missing_minute0_bar_keeps_minute1_tick_gap_only_when_both_missing() -> None:
    step = 4 * HOUR
    # boundary 2: minute-0 absent, minute-1 present -> released on the T+1m tick (no gap)
    # boundary 3: both absent -> gap (a later bar keeps boundary 4 in range)
    opens = [DAY0, DAY0 + MIN, DAY0 + step + MIN, DAY0 + 3 * step + MIN]
    ser = _series(opens, [1, 2, 3, 4])
    ticks, gaps = build_fill_ticks(ser, "4h", create_instrument("BTCUSDT"))
    assert [float(t.price) for t in ticks] == [1.0, 2.0, 3.0, 4.0]
    assert gaps == 1  # boundary at 2*step has no tick
    ticks, gaps = build_fill_ticks(
        _series([DAY0, DAY0 + MIN, DAY0 + 2 * step], [1, 2, 3]), "4h", create_instrument("BTCUSDT")
    )
    assert gaps == 1  # boundary at step: no bars at all


def test_1m_timeframe_raises() -> None:
    with pytest.raises(ValueError):
        build_fill_ticks(_series([DAY0], [1.0]), "1m", create_instrument("BTCUSDT"))
    with pytest.raises(ValueError):
        resolve_fill_ticks(["BTCUSDT"], "1m", DAY0, DAY0 + HOUR, "/nonexistent/catalog")


def test_1h_and_1d_boundaries_are_utc_aligned() -> None:
    # 01:00 and 00:30 etc. are not boundaries for 1d; 1h boundary every full hour
    opens = [DAY0 + 30 * MIN, DAY0 + 60 * MIN, DAY0 + 61 * MIN, DAY0 + 120 * MIN, DAY0 + 121 * MIN]
    ser = _series(opens, [1, 2, 3, 4, 5])
    h_ticks, _ = build_fill_ticks(ser, "1h", create_instrument("BTCUSDT"))
    assert [float(t.price) for t in h_ticks] == [2.0, 3.0, 4.0, 5.0]
    d_ticks, _ = build_fill_ticks(ser, "1d", create_instrument("BTCUSDT"))
    assert d_ticks == []
    day = 24 * HOUR
    ser2 = _series([DAY0 + day - MIN, DAY0 + day, DAY0 + day + MIN], [1, 2, 3])
    d2, _ = build_fill_ticks(ser2, "1d", create_instrument("BTCUSDT"))
    assert [float(t.price) for t in d2] == [2.0, 3.0]


def _full_day_catalog(tmp_path: Path) -> Path:
    cat = tmp_path / "catalog"
    _write_catalog(cat, _minutes(DAY0, 24 * 60))
    return cat


def _only_dir(tmp_path: Path) -> Path:
    base = tmp_path / "fill_ticks" / FILL_TICK_VERSION / "1h" / "BTCUSDT"
    dirs = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert len(dirs) == 1
    return dirs[0]


def test_unchanged_fingerprint_no_rebuild_changed_gives_new_dir(tmp_path: Path) -> None:
    cat = _full_day_catalog(tmp_path)
    end = DAY0 + 23 * HOUR + 2 * MIN
    s1 = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    d1 = Path(s1.paths["BTCUSDT"])
    assert d1.parent == tmp_path / "fill_ticks" / FILL_TICK_VERSION / "1h" / "BTCUSDT"
    m1 = d1.stat().st_mtime_ns
    meta1 = (d1 / "meta.json").stat().st_mtime_ns
    s2 = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    assert s2.paths == s1.paths
    assert d1.stat().st_mtime_ns == m1
    assert (d1 / "meta.json").stat().st_mtime_ns == meta1
    # touching a 1m parquet changes its mtime -> new fingerprint -> new dir, old untouched
    pq_file = next((cat / "data" / "bar").rglob("*.parquet"))
    st = pq_file.stat()
    os.utime(pq_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    s3 = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    assert s3.paths["BTCUSDT"] != s1.paths["BTCUSDT"]
    assert d1.stat().st_mtime_ns == m1
    assert (d1 / "meta.json").is_file()


def test_instrument_only_change_gives_new_dir(tmp_path: Path) -> None:
    """The tick dir carries a copy of the instrument: a changed instrument must rebuild."""
    cat = _full_day_catalog(tmp_path)
    end = DAY0 + 23 * HOUR + 2 * MIN
    d1 = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat).paths["BTCUSDT"]
    inst_file = next((cat / "data" / "crypto_perpetual").rglob("*.parquet"))
    st = inst_file.stat()
    os.utime(inst_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    d2 = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat).paths["BTCUSDT"]
    assert d2 != d1
    assert Path(d1, "meta.json").is_file()


def test_resolved_set_records_its_window(tmp_path: Path) -> None:
    cat = _full_day_catalog(tmp_path)
    end = DAY0 + 23 * HOUR + 2 * MIN
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    assert (s.start_ns, s.end_ns) == (DAY0, end)
    s2 = resolve_fill_ticks(["BTCUSDT"], "1h", "2024-01-01", "2024-01-01T23:02", cat)
    assert (s2.start_ns, s2.end_ns) == (DAY0, end)


def _worker(args: tuple[str, int, int]) -> str:
    cat, start, end = args
    return resolve_fill_ticks(["BTCUSDT"], "1h", start, end, cat).paths["BTCUSDT"]


def test_concurrent_processes_one_valid_dir(tmp_path: Path) -> None:
    cat = tmp_path / "catalog"
    _write_catalog(cat, _minutes(DAY0, 7 * 24 * 60))  # enough work that unlocked builds overlap
    end = DAY0 + 7 * 24 * HOUR - 2 * MIN
    with mp.get_context("spawn").Pool(8) as pool:
        out = pool.map(_worker, [(str(cat), DAY0, end)] * 8, chunksize=1)
    assert len(set(out)) == 1
    d = _only_dir(tmp_path)
    assert str(d) == out[0]
    assert not [p for p in d.parent.iterdir() if p.name.startswith(".tmp")]
    ticks = ParquetDataCatalog(str(d)).trade_ticks()
    assert len(ticks) == 2 * 7 * 24


def test_absent_or_partial_coverage_raises(tmp_path: Path) -> None:
    cat = _full_day_catalog(tmp_path)
    # absent symbol
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["ETHUSDT"], "1h", DAY0, DAY0 + HOUR, cat)
    # window starts before the data
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", DAY0 - 5 * HOUR, DAY0 + HOUR, cat)
    # window runs past the data
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 30 * HOUR, cat)
    # one good symbol does not hide a bad one
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT", "ETHUSDT"], "1h", DAY0, DAY0 + 2 * HOUR, cat)


def test_gap_in_middle_is_counted_not_fatal(tmp_path: Path) -> None:
    opens = [o for o in _minutes(DAY0, 6 * 60) if o not in (DAY0 + 3 * HOUR, DAY0 + 3 * HOUR + MIN)]
    cat = _write_catalog(tmp_path / "catalog", opens)
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 5 * HOUR, cat)
    assert s.missing_boundaries == {"BTCUSDT": 1}
    assert len(ParquetDataCatalog(s.paths["BTCUSDT"]).trade_ticks()) == 2 * 5


def test_price_precision_from_catalog_instrument(tmp_path: Path) -> None:
    d = create_instrument("BTCUSDT").to_dict(create_instrument("BTCUSDT"))
    d["price_precision"] = 3
    d["price_increment"] = "0.001"
    from nautilus_trader.model.instruments import CryptoPerpetual

    inst = CryptoPerpetual.from_dict(d)
    assert inst.price_precision != create_instrument("BTCUSDT").price_precision
    cat = _write_catalog(tmp_path / "catalog", _minutes(DAY0, 3 * 60), instrument=inst)
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 2 * HOUR, cat)
    ticks = ParquetDataCatalog(s.paths["BTCUSDT"]).trade_ticks()
    assert ticks and all(t.price.precision == 3 for t in ticks)
    inst_out = ParquetDataCatalog(s.paths["BTCUSDT"]).instruments()[0]
    assert inst_out.price_precision == 3


REAL = Path(__file__).resolve().parents[2] / "data" / "catalog"


@pytest.mark.skipif(
    not (REAL / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir(),
    reason="real 1m catalog not available",
)
def test_real_btc_4h_tick_count_matches_boundaries(tmp_path: Path) -> None:
    import pandas as pd

    real = REAL
    (tmp_path / "catalog").symlink_to(real.resolve(), target_is_directory=True)
    start, end = "2024-01-01", "2024-03-01"
    s = resolve_fill_ticks(["BTCUSDT"], "4h", start, end, tmp_path / "catalog")
    # built under the tmp parent, never next to the real catalog
    assert Path(s.paths["BTCUSDT"]).is_relative_to(tmp_path)
    ticks = ParquetDataCatalog(s.paths["BTCUSDT"]).trade_ticks(
        start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC")
    )
    boundaries = 60 * 6
    assert abs(len(ticks) - 2 * boundaries) <= 2 * s.missing_boundaries["BTCUSDT"] + 2
    assert len(ticks) >= 2 * boundaries - 2


def _sym_dir(tmp_path: Path, tf: str = "1h") -> Path:
    return tmp_path / "fill_ticks" / FILL_TICK_VERSION / tf / "BTCUSDT"


def _first_last(path: str) -> tuple[int, int]:
    ts = sorted(t.ts_init for t in ParquetDataCatalog(path).trade_ticks())
    return ts[0], ts[-1]


def test_coverage_boundaries_are_exact(tmp_path: Path) -> None:
    from vibe_quant.screening.nt_runner import MissingBarDataError

    cat = _full_day_catalog(tmp_path)
    s0 = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + HOUR, cat)
    first, last = _first_last(s0.paths["BTCUSDT"])
    assert first == DAY0 + INIT and last == DAY0 + 23 * HOUR + MIN + INIT
    # start: first tick - tf passes, one ns earlier raises
    resolve_fill_ticks(["BTCUSDT"], "1h", first - HOUR, first + HOUR, cat)
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", first - HOUR - 1, first + HOUR, cat)
    # end: last tick + tf passes, one ns later raises
    resolve_fill_ticks(["BTCUSDT"], "1h", last - HOUR, last + HOUR, cat)
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", last - HOUR, last + HOUR + 1, cat)


def test_data_ending_on_minute0_is_covered(tmp_path: Path) -> None:
    # last 1h boundary has only its minute-0 bar (1 tick)
    cat = _write_catalog(tmp_path / "catalog", _minutes(DAY0, 3 * 60 + 1))
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 3 * HOUR, cat)
    assert len(ParquetDataCatalog(s.paths["BTCUSDT"]).trade_ticks()) == 2 * 3 + 1
    assert s.missing_boundaries == {"BTCUSDT": 0}


def test_deleted_tick_parquet_raises(tmp_path: Path) -> None:
    from vibe_quant.screening.nt_runner import MissingBarDataError

    cat = _full_day_catalog(tmp_path)
    end = DAY0 + 23 * HOUR + 2 * MIN
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    files = list((Path(s.paths["BTCUSDT"]) / "data" / "trade_tick").rglob("*.parquet"))
    assert files
    for f in files:
        f.unlink()
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)


def test_tick_count_disagreeing_with_build_meta_raises(tmp_path: Path) -> None:
    import json

    from vibe_quant.screening.nt_runner import MissingBarDataError

    cat = _full_day_catalog(tmp_path)
    end = DAY0 + 23 * HOUR + 2 * MIN
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    meta_path = Path(s.paths["BTCUSDT"]) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["n_ticks"] += 1  # as if a tick file were truncated after the build
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)


def test_window_inside_a_data_hole_raises_and_gaps_are_per_window(tmp_path: Path) -> None:
    from vibe_quant.screening.nt_runner import MissingBarDataError

    # 1m data hole of hours 6..17 (12 boundaries) inside a day
    opens = [o for o in _minutes(DAY0, 24 * 60) if not DAY0 + 6 * HOUR <= o < DAY0 + 18 * HOUR]
    cat = _write_catalog(tmp_path / "catalog", opens)
    with pytest.raises(MissingBarDataError):
        resolve_fill_ticks(["BTCUSDT"], "1h", DAY0 + 8 * HOUR, DAY0 + 15 * HOUR, cat)
    # window before the hole: no gaps counted from outside it
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 5 * HOUR, cat)
    assert s.missing_boundaries == {"BTCUSDT": 0}
    # window spanning the hole counts its boundaries
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 20 * HOUR, cat)
    assert s.missing_boundaries == {"BTCUSDT": 12}


def test_orphan_tmp_dirs_swept_and_failed_build_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    cat = _full_day_catalog(tmp_path)
    end = DAY0 + 23 * HOUR + 2 * MIN
    orphan = _sym_dir(tmp_path) / ".tmp-deadbeef"
    orphan.mkdir(parents=True)
    (orphan / "junk").write_text("x")
    resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    assert not orphan.exists()

    # a build that dies after writing leaves no .tmp-* behind
    os.utime(next((cat / "data" / "bar").rglob("*.parquet")), ns=(0, 1_700_000_000_000_000_000))

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(os, "rename", boom)
    with pytest.raises(RuntimeError):
        resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, end, cat)
    assert not [p for p in _sym_dir(tmp_path).iterdir() if p.name.startswith(".tmp-")]


def test_dir_signature_detects_change(tmp_path: Path) -> None:
    from vibe_quant.data.fill_ticks import fill_tick_dir_signature

    cat = _full_day_catalog(tmp_path)
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0, DAY0 + 23 * HOUR + 2 * MIN, cat)
    d = Path(s.paths["BTCUSDT"])
    sig = fill_tick_dir_signature(d)
    assert sig == fill_tick_dir_signature(d)
    (d / "data" / "epoch.parquet").parent.mkdir(exist_ok=True)
    (d / "data" / "epoch.parquet").write_bytes(b"x")
    assert fill_tick_dir_signature(d) != sig


def test_unaligned_window_gaps_ignore_boundaries_outside_it(tmp_path: Path) -> None:
    # hour-2 boundary has no bars; start is 30s past the hour-0 boundary, so hour-0's
    # tick lies inside [start, end] but its boundary is outside the window's range
    opens = [o for o in _minutes(DAY0, 6 * 60) if not DAY0 + 2 * HOUR <= o < DAY0 + 3 * HOUR]
    cat = _write_catalog(tmp_path / "catalog", opens)
    s = resolve_fill_ticks(["BTCUSDT"], "1h", DAY0 + 30_000_000_000, DAY0 + 4 * HOUR, cat)
    assert s.missing_boundaries == {"BTCUSDT": 1}


def test_data_starting_mid_boundary_is_not_a_gap() -> None:
    ser = _series([DAY0 + 30 * MIN, DAY0 + 60 * MIN, DAY0 + 61 * MIN], [1, 2, 3])
    _, gaps = build_fill_ticks(ser, "1h", create_instrument("BTCUSDT"))
    assert gaps == 0


def test_fill_tick_version_bumped_for_instrument_fingerprint() -> None:
    """v3 = instrument parquet in the fingerprint; v2 dirs (built without it) must be orphaned."""
    assert FILL_TICK_VERSION == "v3"
