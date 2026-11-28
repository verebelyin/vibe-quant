"""Catalog instrument reads raise or retry a descriptor shortage, never read it as absent."""

from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.instrument_read import read_instruments

if TYPE_CHECKING:
    from pathlib import Path


def fd_exhausted_instrument_dir(
    monkeypatch: pytest.MonkeyPatch, cat: Path, failures: int | None
) -> list[int]:
    """Make directory reads of the catalog's instrument dir fail like EMFILE.

    ``failures`` None = always. Returns a one-item counter of injected failures.
    """
    target = str(cat.absolute() / "data" / "crypto_perpetual")
    injected = [0]

    def wrap(real: Any) -> Any:
        def guarded(path: Any = ".", *a: Any, **kw: Any) -> Any:
            if str(path).startswith(target) and (failures is None or injected[0] < failures):
                injected[0] += 1
                raise OSError(errno.EMFILE, "Too many open files", str(path))
            return real(path, *a, **kw)

        return guarded

    monkeypatch.setattr(os, "scandir", wrap(os.scandir))
    monkeypatch.setattr(os, "listdir", wrap(os.listdir))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return injected


def _catalog(tmp_path: Path, symbols: list[str]) -> Path:
    cat = tmp_path / "catalog"
    ParquetDataCatalog(str(cat)).write_data([create_instrument(s) for s in symbols])
    return cat


def _dicts(instruments: list[Any]) -> list[dict[str, Any]]:
    return [CryptoPerpetual.to_dict(i) for i in instruments]


def test_reads_equal_nt_instruments_per_id_and_all(tmp_path: Path) -> None:
    cat = _catalog(tmp_path, ["SOLUSDT", "BTCUSDT", "ETHUSDT"])
    nt = ParquetDataCatalog(str(cat))
    assert _dicts(read_instruments(cat)) == _dicts(nt.instruments())
    assert [str(i.id) for i in read_instruments(cat)] == [
        "BTCUSDT-PERP.BINANCE",
        "ETHUSDT-PERP.BINANCE",
        "SOLUSDT-PERP.BINANCE",
    ]
    one = read_instruments(str(cat), "ETHUSDT-PERP.BINANCE")
    assert _dicts(one) == _dicts(nt.instruments(instrument_ids=["ETHUSDT-PERP.BINANCE"]))
    assert [str(i.id) for i in one] == ["ETHUSDT-PERP.BINANCE"]
    assert read_instruments(cat, "XRPUSDT-PERP.BINANCE") == []
    assert read_instruments(tmp_path / "empty") == []


def test_fast_tier_instrument_survives_a_transient_fd_shortage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe_quant.fast_tier.signals import load_instrument

    cat = _catalog(tmp_path, ["BTCUSDT", "ETHUSDT"])
    injected = fd_exhausted_instrument_dir(monkeypatch, cat, failures=1)
    instrument = load_instrument(str(cat), "ETHUSDT-PERP.BINANCE")
    assert injected[0] == 1
    assert str(instrument.id) == "ETHUSDT-PERP.BINANCE"


def test_carry_catalog_instruments_raise_a_persistent_fd_shortage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe_quant.portfolio import ticks

    cat = _catalog(tmp_path, ["BTCUSDT", "ETHUSDT"])
    assert sorted(ticks._catalog_instruments(cat)) == ["BTCUSDT", "ETHUSDT"]
    fd_exhausted_instrument_dir(monkeypatch, cat, failures=None)
    with pytest.raises(OSError) as exc:
        ticks._catalog_instruments(cat)
    assert exc.value.errno == errno.EMFILE
