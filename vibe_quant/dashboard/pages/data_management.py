"""Data management tab for vibe-quant Streamlit dashboard.

Provides UI for:
- Data coverage display (symbols, dates, bar counts)
- Data ingestion/update with date range selection + download preview
- Download progress with real-time streaming output
- OHLCV data table browser with candlestick chart
- Download audit log (session history)
- Raw archive status and catalog rebuild
- Storage usage display
- Data quality verification
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st
from lightweight_charts.widgets import StreamlitChart

from vibe_quant.dashboard.utils import format_bytes

if TYPE_CHECKING:
    from typing import Any

# Default paths (must match data modules)
DEFAULT_ARCHIVE_PATH = Path("data/archive/raw_data.db")
DEFAULT_CATALOG_PATH = Path("data/catalog")

INTERVALS = ["1m", "5m", "15m", "1h", "4h"]


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

            # Funding rate count
            funding = archive.get_funding_rates(symbol)
            sym_status["funding_rates"] = len(funding)

            # Catalog info
            for interval in INTERVALS:
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


def _get_download_sessions() -> list[dict[str, Any]]:
    """Get download audit log."""
    from vibe_quant.data.archive import RawDataArchive

    try:
        archive = RawDataArchive(DEFAULT_ARCHIVE_PATH)
        sessions = archive.get_download_sessions(limit=20)
        archive.close()
        return [dict(s) for s in sessions]
    except Exception:
        return []


def _verify_data(symbol: str) -> dict[str, Any]:
    """Run data verification for a symbol."""
    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.data.verify import verify_symbol

    archive = RawDataArchive(DEFAULT_ARCHIVE_PATH)
    result = verify_symbol(archive, symbol)
    archive.close()
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
        st.metric("SQLite Archive", format_bytes(usage["archive"]))
    with col2:
        st.metric("Parquet Catalog", format_bytes(usage["catalog"]))
    with col3:
        st.metric("Total", format_bytes(usage["archive"] + usage["catalog"]))


def render_data_coverage() -> None:
    """Render data coverage table."""
    st.subheader("Data Coverage")

    status = _get_data_status()

    if "error" in status:
        st.error(f"Error loading status: {status['error']}")
        return

    if not status["symbols"]:
        st.info("No data available. Use 'Ingest Data' below to download.")
        return

    # Build coverage table
    rows = []
    for symbol, info in sorted(status["symbols"].items()):
        row: dict[str, Any] = {"Symbol": symbol}

        if "archive" in info:
            row["Start Date"] = info["archive"]["start"].strftime("%Y-%m-%d")
            row["End Date"] = info["archive"]["end"].strftime("%Y-%m-%d")
            row["1m Klines"] = info["archive"]["klines_1m"]
        else:
            row["Start Date"] = "-"
            row["End Date"] = "-"
            row["1m Klines"] = 0

        row["Funding Rates"] = info.get("funding_rates", 0)

        if "catalog" in info:
            row["5m Bars"] = info["catalog"].get("bars_5m", 0)
            row["15m Bars"] = info["catalog"].get("bars_15m", 0)
            row["1h Bars"] = info["catalog"].get("bars_1h", 0)
            row["4h Bars"] = info["catalog"].get("bars_4h", 0)
        else:
            row["5m Bars"] = 0
            row["15m Bars"] = 0
            row["1h Bars"] = 0
            row["4h Bars"] = 0

        rows.append(row)

    st.dataframe(rows, use_container_width=True)


def render_download_history() -> None:
    """Render download audit log."""
    st.subheader("Download History")

    sessions = _get_download_sessions()

    if not sessions:
        st.info("No download sessions recorded yet.")
        return

    rows = []
    for s in sessions:
        row: dict[str, Any] = {
            "Started": s.get("started_at", "-"),
            "Completed": s.get("completed_at", "-") or "-",
            "Symbols": s.get("symbols", "-"),
            "Date Range": f"{s.get('start_date', '?')} to {s.get('end_date', '?')}",
            "Source": s.get("source", "-"),
            "Klines Fetched": s.get("klines_fetched", 0),
            "Inserted": s.get("klines_inserted", 0),
            "Funding": s.get("funding_rates_fetched", 0),
            "Status": s.get("status", "-"),
        }
        if s.get("error_message"):
            row["Error"] = s["error_message"]
        rows.append(row)

    st.dataframe(rows, use_container_width=True)


def render_data_actions() -> None:
    """Render data ingestion and update actions with download preview."""
    from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

    st.subheader("Data Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Ingest Historical Data**")
        st.caption("Download from Binance Vision + REST API")

        selected_symbols = st.multiselect(
            "Symbols to ingest",
            SUPPORTED_SYMBOLS,
            default=SUPPORTED_SYMBOLS,
            key="ingest_symbols",
        )

        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start = st.date_input(
                "Start date",
                value=date.today() - timedelta(days=365),
                key="ingest_start",
            )
        with date_col2:
            end = st.date_input(
                "End date",
                value=date.today(),
                key="ingest_end",
            )

        # Download preview
        if (
            selected_symbols
            and start
            and end
            and st.button("Preview Download", key="btn_preview")
        ):
            _render_download_preview(selected_symbols, start, end)

        if st.button("Ingest Data", type="primary", key="btn_ingest"):
            if selected_symbols:
                _run_ingest(selected_symbols, start, end)
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


def _render_download_preview(
    symbols: list[str], start: date, end: date,
) -> None:
    """Show preview of what will be downloaded vs skipped."""
    from vibe_quant.data.ingest import get_download_preview

    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC)

    with st.spinner("Checking archive coverage..."):
        preview = get_download_preview(symbols, start_dt, end_dt)

    if not preview:
        st.info("No months in selected range")
        return

    # Summary counts
    archived = sum(1 for p in preview if p["Status"] == "Archived")
    to_download = len(preview) - archived
    st.info(
        f"{len(preview)} total months: "
        f"**{archived}** already archived, "
        f"**{to_download}** to download"
    )

    st.dataframe(preview, use_container_width=True)


def _run_ingest(symbols: list[str], start: date, end: date) -> None:
    """Run data ingestion with streaming progress."""
    cmd = [
        sys.executable, "-u",  # unbuffered output
        "-m", "vibe_quant.data",
        "ingest",
        "--symbols", ",".join(symbols),
        "--start", start.isoformat(),
        "--end", end.isoformat(),
    ]

    status_container = st.status(
        f"Ingesting {', '.join(symbols)} from {start} to {end}...",
        expanded=True,
    )
    log_area = status_container.empty()
    output_lines: list[str] = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):  # type: ignore[union-attr]
            output_lines.append(line.rstrip())
            # Show last 30 lines to keep UI responsive
            log_area.code("\n".join(output_lines[-30:]))

        process.wait()

        if process.returncode == 0:
            status_container.update(label="Data ingestion completed!", state="complete")
        else:
            status_container.update(label="Data ingestion failed", state="error")
    except Exception as e:
        status_container.update(label=f"Error: {e}", state="error")


def _run_update(symbols: list[str]) -> None:
    """Run data update with streaming progress."""
    cmd = [
        sys.executable, "-u",
        "-m", "vibe_quant.data",
        "update",
        "--symbols", ",".join(symbols),
    ]

    status_container = st.status(
        f"Updating data for {', '.join(symbols)}...",
        expanded=True,
    )
    log_area = status_container.empty()
    output_lines: list[str] = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):  # type: ignore[union-attr]
            output_lines.append(line.rstrip())
            log_area.code("\n".join(output_lines[-30:]))

        process.wait()

        if process.returncode == 0:
            status_container.update(label="Data update completed!", state="complete")
        else:
            status_container.update(label="Data update failed", state="error")
    except Exception as e:
        status_container.update(label=f"Error: {e}", state="error")


def _run_rebuild() -> None:
    """Run catalog rebuild with streaming progress."""
    cmd = [
        sys.executable, "-u",
        "-m", "vibe_quant.data",
        "rebuild",
        "--from-archive",
    ]

    status_container = st.status("Rebuilding catalog from archive...", expanded=True)
    log_area = status_container.empty()
    output_lines: list[str] = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):  # type: ignore[union-attr]
            output_lines.append(line.rstrip())
            log_area.code("\n".join(output_lines[-30:]))

        process.wait()

        if process.returncode == 0:
            status_container.update(
                label="Catalog rebuild completed!", state="complete",
            )
        else:
            status_container.update(label="Catalog rebuild failed", state="error")
    except Exception as e:
        status_container.update(label=f"Error: {e}", state="error")


def render_data_browser() -> None:
    """Render OHLCV data table browser and candlestick chart."""
    from vibe_quant.data.catalog import CatalogManager
    from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

    st.subheader("Data Browser")

    # Symbol + date selectors
    sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 2])

    with sel_col1:
        symbol = st.selectbox("Symbol", SUPPORTED_SYMBOLS, key="browser_symbol")
    with sel_col2:
        browse_start = st.date_input(
            "Start",
            value=date.today() - timedelta(days=7),
            key="browser_start",
        )
    with sel_col3:
        browse_end = st.date_input(
            "End",
            value=date.today(),
            key="browser_end",
        )

    # Timeframe selector (pill-style)
    interval = st.segmented_control(
        "Timeframe", INTERVALS, default="1h", key="browser_interval",
    )

    if not symbol or not browse_start or not browse_end or not interval:
        return

    # Load data from catalog
    try:
        catalog = CatalogManager(DEFAULT_CATALOG_PATH)
        start_dt = datetime(
            browse_start.year, browse_start.month, browse_start.day, tzinfo=UTC,
        )
        end_dt = datetime(
            browse_end.year, browse_end.month, browse_end.day,
            23, 59, 59, tzinfo=UTC,
        )
        bars = catalog.get_bars(symbol, interval, start=start_dt, end=end_dt)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if not bars:
        st.info("No data for selected symbol/interval/range")
        return

    # Build DataFrame from bars
    df = pd.DataFrame([
        {
            "Time": datetime.fromtimestamp(b.ts_event / 1e9, tz=UTC),
            "Open": float(b.open),
            "High": float(b.high),
            "Low": float(b.low),
            "Close": float(b.close),
            "Volume": float(b.volume),
        }
        for b in bars
    ])

    st.caption(
        f"{len(df)} bars | "
        f"{df['Time'].min().strftime('%Y-%m-%d %H:%M')} to "
        f"{df['Time'].max().strftime('%Y-%m-%d %H:%M')}"
    )

    # Tabbed view: Chart | Table
    tab_chart, tab_table = st.tabs(["Chart", "Table"])

    with tab_chart:
        _render_candlestick_chart(df, symbol, interval)

    with tab_table:
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "Volume": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def _render_candlestick_chart(
    df: pd.DataFrame, symbol: str, interval: str,
) -> None:
    """Render TradingView-style candlestick chart with volume."""
    max_candles = 10_000
    if len(df) > max_candles:
        st.warning(
            f"Showing last {max_candles:,} of {len(df):,} candles. "
            f"Narrow date range for full data."
        )
        df = df.tail(max_candles)

    # lightweight-charts expects lowercase columns + 'time' as str
    chart_df = df.rename(columns={
        "Time": "time", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    }).copy()
    chart_df["time"] = chart_df["time"].dt.strftime("%Y-%m-%d %H:%M")

    chart = StreamlitChart(height=600)

    chart.layout(
        background_color="#131722",
        text_color="#d1d4dc",
        font_size=12,
        font_family="Trebuchet MS",
    )

    chart.candle_style(
        up_color="#26a69a",
        down_color="#ef5350",
        wick_up_color="#26a69a",
        wick_down_color="#ef5350",
        border_up_color="#26a69a",
        border_down_color="#ef5350",
    )

    chart.volume_config(
        up_color="rgba(38,166,154,0.5)",
        down_color="rgba(239,83,80,0.5)",
    )

    chart.crosshair(mode="normal")

    chart.watermark(symbol, color="rgba(180, 180, 200, 0.15)")

    chart.legend(visible=True, font_size=14)

    chart.set(chart_df)
    chart.load()


def render_symbol_management() -> None:
    """Render symbol list management."""
    from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

    st.subheader("Supported Symbols")
    st.write(", ".join(SUPPORTED_SYMBOLS))


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

    # Download actions with date pickers + preview
    render_data_actions()

    st.divider()

    # OHLCV data browser (table + candlestick chart)
    render_data_browser()

    st.divider()

    # Download audit history
    render_download_history()

    st.divider()

    # Symbols and quality
    col1, col2 = st.columns([1, 2])

    with col1:
        render_symbol_management()

    with col2:
        render_data_quality()


# Top-level call for st.navigation API
render()
