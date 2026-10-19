"""Process caches keyed by ABSOLUTE roots (bd vibe-quant-n8fbi).

``extraction._BAR_CACHE`` and ``aux_data._REF``/``_REF_DERIVED`` are process-wide.
A relative catalog path ("catalog", or the default "data/catalog") spells the
same key in two different roots, so a second root used from another working
directory used to be served the first root's bars even when both roots carry
files with identical names, sizes and mtimes. Keys are absolute now:
``os.path.abspath`` -- never ``resolve()`` -- so a symlinked catalog keeps
working and keeps its own key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vibe_quant.dsl import aux_data
from vibe_quant.validation import extraction
from vibe_quant.validation.extraction import load_catalog_bars

if TYPE_CHECKING:
    from collections.abc import Iterator

DAY_BAR_TYPE = "BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL"
DAY_NS = 86_400 * 1_000_000_000
DAY0 = 1_704_067_200 * 1_000_000_000  # 2024-01-01T00:00:00Z
_PRICE_SCALE = 10**16
# One fixed mtime for every fixture file: two roots then differ only in content.
_FIXED_MTIME_NS = 1_700_000_000_000_000_000


def _encode_price(price: float) -> bytes:
    """NT fixed-point price: 128-bit little-endian int, low 8 bytes first."""
    raw = int(round(price * _PRICE_SCALE))
    return (raw & (2**64 - 1)).to_bytes(8, "little") + ((raw >> 64) & (2**64 - 1)).to_bytes(
        8, "little"
    )


def _write_day_bars(catalog: Path, closes: list[float]) -> Path:
    """Write ``closes`` as daily LAST bars (``ts_init`` 23:59:59.999) to ``catalog``."""
    rows = DAY0 + (np.arange(len(closes), dtype=np.int64) + 1) * DAY_NS - 1_000_000
    table = pa.table(
        {
            "open": pa.array([_encode_price(c) for c in closes], type=pa.binary(16)),
            "high": pa.array([_encode_price(c + 2.0) for c in closes], type=pa.binary(16)),
            "low": pa.array([_encode_price(c - 2.0) for c in closes], type=pa.binary(16)),
            "close": pa.array([_encode_price(c) for c in closes], type=pa.binary(16)),
            "volume": pa.array([_encode_price(1.0) for _ in closes], type=pa.binary(16)),
            "ts_event": pa.array(rows.tolist(), type=pa.uint64()),
            "ts_init": pa.array(rows.tolist(), type=pa.uint64()),
        }
    )
    bar_dir = catalog / "data" / "bar" / DAY_BAR_TYPE
    bar_dir.mkdir(parents=True, exist_ok=True)
    path = bar_dir / "part.parquet"
    pq.write_table(table, path)
    os.utime(path, ns=(_FIXED_MTIME_NS, _FIXED_MTIME_NS))
    return path


def _signature(catalog: Path) -> tuple[tuple[str, int, int], ...]:
    """The extraction cache's own file signature of ``catalog``'s daily bar dir."""
    bar_dir = catalog / "data" / "bar" / DAY_BAR_TYPE
    files = sorted(bar_dir.glob("*.parquet"))
    return tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)


@pytest.fixture(autouse=True)
def _clean_caches() -> Iterator[None]:
    extraction._BAR_CACHE.clear()
    aux_data._REF.clear()
    aux_data._REF_DERIVED.clear()
    yield
    extraction._BAR_CACHE.clear()
    aux_data._REF.clear()
    aux_data._REF_DERIVED.clear()
    aux_data.configure(None, None)


def test_bar_cache_serves_each_root_its_own_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two roots, equal signatures, one relative path: each cwd gets its own bars."""
    _write_day_bars(tmp_path / "root-a" / "catalog", [100.0, 110.0])
    _write_day_bars(tmp_path / "root-b" / "catalog", [200.0, 220.0])
    assert _signature(tmp_path / "root-a" / "catalog") == _signature(tmp_path / "root-b" / "catalog")

    monkeypatch.chdir(tmp_path / "root-a")
    first = load_catalog_bars("catalog", DAY_BAR_TYPE)
    assert first is not None
    assert first.close.tolist() == [100.0, 110.0]

    monkeypatch.chdir(tmp_path / "root-b")
    second = load_catalog_bars("catalog", DAY_BAR_TYPE)
    assert second is not None
    assert second.close.tolist() == [200.0, 220.0]


def test_bar_cache_keys_a_symlinked_root_by_its_own_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlinked catalog still loads, and is keyed by the symlink (no resolve())."""
    real = tmp_path / "real" / "catalog"
    _write_day_bars(real, [42.0])
    link = tmp_path / "linked-catalog"
    link.symlink_to(real, target_is_directory=True)

    monkeypatch.chdir(tmp_path)
    series = load_catalog_bars(link, DAY_BAR_TYPE)
    assert series is not None
    assert series.close.tolist() == [42.0]

    bar_dir = Path(os.path.abspath(link)) / "data" / "bar" / DAY_BAR_TYPE
    assert str(bar_dir) in extraction._BAR_CACHE
    assert str(real) not in extraction._BAR_CACHE


def test_aux_ref_cache_serves_each_root_its_own_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same collision on ``_REF``/``_REF_DERIVED``: close and trend follow the cwd."""
    _write_day_bars(tmp_path / "root-a" / "catalog", [100.0, 110.0])
    _write_day_bars(tmp_path / "root-b" / "catalog", [200.0, 180.0])

    monkeypatch.chdir(tmp_path / "root-a")
    aux_data.configure(None, "catalog")
    _, close_a = aux_data.ref_close_series("BTCUSDT", "1d")
    assert close_a.tolist() == [100.0, 110.0]
    trend_a = aux_data.ref_trend_asof(
        "BTCUSDT", np.array([DAY0 + 2 * DAY_NS], dtype=np.int64), 2
    )
    assert trend_a[0] == pytest.approx(110.0 / ((100.0 + 2 * 110.0) / 3) - 1.0)

    # No configure() on the second root: only the working directory changed, so
    # the cache must not hand root A's bars to root B.
    monkeypatch.chdir(tmp_path / "root-b")
    _, close_b = aux_data.ref_close_series("BTCUSDT", "1d")
    assert close_b.tolist() == [200.0, 180.0]
    trend_b = aux_data.ref_trend_asof(
        "BTCUSDT", np.array([DAY0 + 2 * DAY_NS], dtype=np.int64), 2
    )
    assert trend_b[0] == pytest.approx(180.0 / ((200.0 + 2 * 180.0) / 3) - 1.0)


def test_aux_ref_cache_serves_each_root_its_own_default_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default "data/catalog" is absolute too: no configure() needed to tell roots apart."""
    _write_day_bars(tmp_path / "root-a" / "data" / "catalog", [100.0])
    _write_day_bars(tmp_path / "root-b" / "data" / "catalog", [200.0])

    monkeypatch.chdir(tmp_path / "root-a")
    aux_data.configure(None, None)
    assert aux_data.ref_close_series("BTCUSDT", "1d")[1].tolist() == [100.0]

    monkeypatch.chdir(tmp_path / "root-b")
    assert aux_data.ref_close_series("BTCUSDT", "1d")[1].tolist() == [200.0]
