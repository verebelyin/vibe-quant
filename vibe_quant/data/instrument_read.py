"""Catalog instrument reads that raise on a listing or read error.

NT's ``catalog.instruments()`` lists ``data/<class>/`` through fsspec, whose walk
turns ANY OSError into an empty listing, so a transient EMFILE/ENFILE read as "no
instrument" (gate flake vibe-quant-28p7p). Here the listing and the read raise, and
a descriptor shortage is retried after dropping unreachable file objects.
"""

from __future__ import annotations

import errno
import gc
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import CryptoPerpetual

_FD_ERRNOS = (errno.EMFILE, errno.ENFILE)
_ATTEMPTS = 4


def _parquet_files(perp_dir: Path, instrument_id: str | None) -> list[str]:
    """Absolute instrument parquet paths, sorted like NT's glob (``[]`` when none)."""
    from nautilus_trader.persistence.funcs import urisafe_identifier

    if not perp_dir.is_dir():
        return []
    names = (
        sorted(os.listdir(perp_dir))
        if instrument_id is None
        else [urisafe_identifier(instrument_id)]
    )
    files: list[str] = []
    for name in names:
        sub = perp_dir / name
        if sub.is_dir():
            files.extend(str(sub / f) for f in os.listdir(sub) if f.endswith(".parquet"))
    return sorted(files)


def read_instruments(
    catalog_path: Path | str, instrument_id: str | None = None
) -> list[CryptoPerpetual]:
    """The catalog's perpetuals (one id, or all), as ``instruments()`` returns them.

    ``[]`` only when no parquet exists; a listing or read OSError raises.
    """
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    perp_dir = Path(catalog_path).absolute() / "data" / "crypto_perpetual"
    for attempt in range(_ATTEMPTS):
        try:
            files = _parquet_files(perp_dir, instrument_id)
            if not files:
                return []
            return ParquetDataCatalog(str(catalog_path)).query(CryptoPerpetual, files=files)
        except OSError as exc:
            if exc.errno not in _FD_ERRNOS or attempt == _ATTEMPTS - 1:
                raise
            gc.collect()
            time.sleep(0.25 * (attempt + 1))
    return []  # unreachable: the last attempt returns or raises
