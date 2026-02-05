"""Data management tab for vibe-quant Streamlit dashboard.

Provides UI for:
- Data coverage display (symbols, dates, bar counts)
- Data ingestion/update with progress tracking
- Raw archive status and catalog rebuild
- Storage usage display
- Data quality verification
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from typing import Any

# Default paths (must match data modules)
DEFAULT_ARCHIVE_PATH = Path("data/archive/raw_data.db")
DEFAULT_CATALOG_PATH = Path("data/catalog")


def _format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def _get_storage_usage() -> dict[str, int]:
    """Get storage usage for archive and catalog."""
    usage: dict[str, int] = {"archive": 0, "catalog": 0}

    if DEFAULT_ARCHIVE_PATH.exists():
        usage["archive"] = DEFAULT_ARCHIVE_PATH.stat().st_size

    if DEFAULT_CATALOG_PATH.exists():
        usage["catalog"] = sum(
            f.stat().st_size for f in DEFAULT_CATALOG_PATH.rglob("*") if f.is_file()
        )

    return usage


def _get_data_status() -> dict[str, Any]:
    """Get status from archive and catalog."""
    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.data.catalog import CatalogManager

    status: dict[str, Any] = {"symbols": {}}

    try:
        archive = RawDataArchive(DEFAULT_ARCHIVE_PATH)
        catalog = CatalogManager(DEFAULT_CATALOG_PATH)

        for symbol in archive.get_symbols():
            sym_status: dict[str, Any] = {}

            # Archive info
            date_range = archive.get_date_range(symbol, "1m")
            if date_range:
                start_dt = datetime.fromtimestamp(date_range[0] / 1000, tz=UTC)
                end_dt = datetime.fromtimestamp(date_range[1] / 1000, tz=UTC)
                sym_status["archive"] = {
                    "klines_1m": archive.get_kline_count(symbol, "1m"),
                    "start": start_dt,
                    "end": end_dt,
                }

            # Catalog info
            for interval in ["1m", "5m", "15m", "1h", "4h"]:
                bar_count = catalog.get_bar_count(symbol, interval)
                if bar_count > 0:
                    if "catalog" not in sym_status:
                        sym_status["catalog"] = {}
                    sym_status["catalog"][f"bars_{interval}"] = bar_count

            status["symbols"][symbol] = sym_status

        archive.close()
    except Exception as e:
        status["error"] = str(e)

    return status


def _verify_data(symbol: str) -> dict[str, Any]:
    """Run data verification for a symbol.

    Returns dict with keys: gaps, ohlc_errors, kline_count
    """
    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.data.verify import verify_symbol

    archive = RawDataArchive(DEFAULT_ARCHIVE_PATH)
    result = verify_symbol(archive, symbol)
    archive.close()
    # Cast to dict[str, Any] for streamlit compatibility
    return {
        "gaps": result["gaps"],
        "ohlc_errors": result["ohlc_errors"],
        "kline_count": result["kline_count"],
    }


def render_storage_metrics() -> None:
    """Render storage usage metrics."""
    st.subheader("Storage Usage")

    usage = _get_storage_usage()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SQLite Archive", _format_bytes(usage["archive"]))
    with col2:
        st.metric("Parquet Catalog", _format_bytes(usage["catalog"]))
    with col3:
        st.metric("Total", _format_bytes(usage["archive"] + usage["catalog"]))


def render_data_coverage() -> None:
    """Render data coverage table."""
    st.subheader("Data Coverage")

    status = _get_data_status()

    if "error" in status:
        st.error(f"Error loading status: {status['error']}")
        return

    if not status["symbols"]:
        st.info("No data available. Use 'Ingest Data' to download.")
        return

    # Build coverage table
    rows = []
    for symbol, info in sorted(status["symbols"].items()):
        row = {"Symbol": symbol}

        if "archive" in info:
            row["Start Date"] = info["archive"]["start"].strftime("%Y-%m-%d")
            row["End Date"] = info["archive"]["end"].strftime("%Y-%m-%d")
            row["1m Klines"] = info["archive"]["klines_1m"]
        else:
            row["Start Date"] = "-"
            row["End Date"] = "-"
            row["1m Klines"] = 0

        if "catalog" in info:
            row["5m Bars"] = info["catalog"].get("bars_5m", 0)
            row["1h Bars"] = info["catalog"].get("bars_1h", 0)
            row["4h Bars"] = info["catalog"].get("bars_4h", 0)
        else:
            row["5m Bars"] = 0
            row["1h Bars"] = 0
            row["4h Bars"] = 0

        rows.append(row)

    st.dataframe(rows, use_container_width=True)


def render_symbol_management() -> None:
    """Render symbol list management."""
    from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

    st.subheader("Supported Symbols")
    st.write(", ".join(SUPPORTED_SYMBOLS))


def render_data_actions() -> None:
    """Render data ingestion and update actions."""
    from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

    st.subheader("Data Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Ingest Historical Data**")
        st.caption("Download historical data from Binance Vision")

        selected_symbols = st.multiselect(
            "Symbols to ingest",
            SUPPORTED_SYMBOLS,
            default=SUPPORTED_SYMBOLS,
            key="ingest_symbols",
        )

        years = st.slider(
            "Years of history",
            min_value=1,
            max_value=5,
            value=2,
            key="ingest_years",
        )

        if st.button("Ingest Data", type="primary", key="btn_ingest"):
            if selected_symbols:
                _run_ingest(selected_symbols, years)
            else:
                st.warning("Select at least one symbol")

    with col2:
        st.write("**Update Recent Data**")
        st.caption("Fetch recent candles via REST API")

        update_symbols = st.multiselect(
            "Symbols to update",
            SUPPORTED_SYMBOLS,
            default=SUPPORTED_SYMBOLS,
            key="update_symbols",
        )

        if st.button("Update Data", key="btn_update"):
            if update_symbols:
                _run_update(update_symbols)
            else:
                st.warning("Select at least one symbol")

        st.write("**Rebuild Catalog**")
        st.caption("Recreate Parquet catalog from SQLite archive")

        if st.button("Rebuild Catalog", key="btn_rebuild"):
            _run_rebuild()


def _run_ingest(symbols: list[str], years: int) -> None:
    """Run data ingestion in subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "vibe_quant.data",
        "ingest",
        "--symbols",
        ",".join(symbols),
        "--years",
        str(years),
    ]

    with st.spinner(f"Ingesting data for {', '.join(symbols)} ({years} years)..."):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
            )
            if result.returncode == 0:
                st.success("Data ingestion completed successfully!")
                if result.stdout:
                    with st.expander("Output"):
                        st.code(result.stdout)
            else:
                st.error("Data ingestion failed")
                if result.stderr:
                    st.code(result.stderr)
        except subprocess.TimeoutExpired:
            st.error("Data ingestion timed out (1 hour limit)")
        except Exception as e:
            st.error(f"Error: {e}")


def _run_update(symbols: list[str]) -> None:
    """Run data update in subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "vibe_quant.data",
        "update",
        "--symbols",
        ",".join(symbols),
    ]

    with st.spinner(f"Updating data for {', '.join(symbols)}..."):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
                check=False,
            )
            if result.returncode == 0:
                st.success("Data update completed successfully!")
                if result.stdout:
                    with st.expander("Output"):
                        st.code(result.stdout)
            else:
                st.error("Data update failed")
                if result.stderr:
                    st.code(result.stderr)
        except subprocess.TimeoutExpired:
            st.error("Data update timed out")
        except Exception as e:
            st.error(f"Error: {e}")


def _run_rebuild() -> None:
    """Run catalog rebuild in subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "vibe_quant.data",
        "rebuild",
        "--from-archive",
    ]

    with st.spinner("Rebuilding catalog from archive..."):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
                check=False,
            )
            if result.returncode == 0:
                st.success("Catalog rebuild completed successfully!")
                if result.stdout:
                    with st.expander("Output"):
                        st.code(result.stdout)
            else:
                st.error("Catalog rebuild failed")
                if result.stderr:
                    st.code(result.stderr)
        except subprocess.TimeoutExpired:
            st.error("Catalog rebuild timed out")
        except Exception as e:
            st.error(f"Error: {e}")


def render_data_quality() -> None:
    """Render data quality verification section."""
    st.subheader("Data Quality")

    status = _get_data_status()
    symbols = list(status.get("symbols", {}).keys())

    if not symbols:
        st.info("No data to verify")
        return

    selected = st.selectbox("Select symbol to verify", symbols, key="verify_symbol")

    if st.button("Run Verification", key="btn_verify") and selected:
            with st.spinner(f"Verifying {selected}..."):
                result = _verify_data(selected)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Kline Count", result["kline_count"])
            with col2:
                gap_count = len(result["gaps"])
                st.metric(
                    "Gaps Detected",
                    gap_count,
                    delta="OK" if gap_count == 0 else None,
                    delta_color="off" if gap_count == 0 else "inverse",
                )
            with col3:
                error_count = len(result["ohlc_errors"])
                st.metric(
                    "OHLC Errors",
                    error_count,
                    delta="OK" if error_count == 0 else None,
                    delta_color="off" if error_count == 0 else "inverse",
                )

            if result["gaps"]:
                with st.expander(f"Gap Details ({len(result['gaps'])} gaps)"):
                    gap_rows = []
                    for start_ts, _end_ts, gap_min in result["gaps"][:20]:
                        start_dt = datetime.fromtimestamp(start_ts / 1000, tz=UTC)
                        gap_rows.append(
                            {
                                "Start": start_dt.strftime("%Y-%m-%d %H:%M"),
                                "Gap (min)": gap_min,
                            }
                        )
                    st.dataframe(gap_rows, use_container_width=True)
                    if len(result["gaps"]) > 20:
                        st.caption(f"Showing 20 of {len(result['gaps'])} gaps")

            if result["ohlc_errors"]:
                with st.expander(f"OHLC Errors ({len(result['ohlc_errors'])} errors)"):
                    err_rows = []
                    for ts, msg in result["ohlc_errors"][:20]:
                        err_dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
                        err_rows.append(
                            {
                                "Time": err_dt.strftime("%Y-%m-%d %H:%M"),
                                "Error": msg,
                            }
                        )
                    st.dataframe(err_rows, use_container_width=True)
                    if len(result["ohlc_errors"]) > 20:
                        st.caption(f"Showing 20 of {len(result['ohlc_errors'])} errors")


def render() -> None:
    """Render the data management page."""
    st.title("Data Management")

    # Storage metrics at top
    render_storage_metrics()

    st.divider()

    # Data coverage table
    render_data_coverage()

    st.divider()

    # Actions and quality in columns
    col1, col2 = st.columns([2, 1])

    with col1:
        render_data_actions()

    with col2:
        render_symbol_management()

    st.divider()

    # Data quality verification
    render_data_quality()


# Top-level call for st.navigation API
render()
