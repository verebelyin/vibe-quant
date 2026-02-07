"""Settings tab for vibe-quant Streamlit dashboard.

Provides CRUD for sizing configs, risk configs, latency presets,
and system info display.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from vibe_quant.db import StateManager
from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.validation.latency import (
    LATENCY_PRESETS,
    LatencyPreset,
)

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import JsonDict

# Sizing method choices
SIZING_METHODS = ["fixed_fractional", "kelly", "atr"]

# Default parameter schemas per sizing method
SIZING_PARAM_SCHEMAS: dict[str, dict[str, dict[str, object]]] = {
    "fixed_fractional": {
        "max_leverage": {"type": "decimal", "default": "20", "min": "1", "max": "125"},
        "max_position_pct": {"type": "decimal", "default": "0.5", "min": "0.01", "max": "1"},
        "risk_per_trade": {"type": "decimal", "default": "0.02", "min": "0.001", "max": "0.2"},
    },
    "kelly": {
        "max_leverage": {"type": "decimal", "default": "20", "min": "1", "max": "125"},
        "max_position_pct": {"type": "decimal", "default": "0.5", "min": "0.01", "max": "1"},
        "win_rate": {"type": "decimal", "default": "0.55", "min": "0.01", "max": "0.99"},
        "avg_win": {"type": "decimal", "default": "1.5", "min": "0.01", "max": "100"},
        "avg_loss": {"type": "decimal", "default": "1.0", "min": "0.01", "max": "100"},
        "kelly_fraction": {"type": "decimal", "default": "0.5", "min": "0.1", "max": "1"},
    },
    "atr": {
        "max_leverage": {"type": "decimal", "default": "20", "min": "1", "max": "125"},
        "max_position_pct": {"type": "decimal", "default": "0.5", "min": "0.01", "max": "1"},
        "risk_per_trade": {"type": "decimal", "default": "0.02", "min": "0.001", "max": "0.2"},
        "atr_multiplier": {"type": "decimal", "default": "2.0", "min": "0.5", "max": "10"},
    },
}


def _get_state_manager() -> StateManager:
    """Get or create StateManager from session state."""
    if "state_manager" not in st.session_state:
        db_path = st.session_state.get("db_path", DEFAULT_DB_PATH)
        st.session_state["state_manager"] = StateManager(Path(db_path))
    manager: StateManager = st.session_state["state_manager"]
    return manager


def _parse_decimal(value: str, field_name: str) -> Decimal:
    """Parse string to Decimal with validation."""
    try:
        return Decimal(value.strip())
    except InvalidOperation as e:
        raise ValueError(f"Invalid decimal for {field_name}: {value}") from e


def _render_sizing_config_form(
    key_prefix: str,
    existing_config: JsonDict | None = None,
) -> tuple[str, str, dict[str, str]] | None:
    """Render form for creating/editing sizing config.

    Returns (name, method, params) on submit, None otherwise.
    """
    default_name = existing_config["name"] if existing_config else ""
    default_method = existing_config["method"] if existing_config else SIZING_METHODS[0]
    existing_params = existing_config.get("config", {}) if existing_config else {}

    with st.form(key=f"{key_prefix}_sizing_form"):
        name = st.text_input("Config Name", value=default_name, key=f"{key_prefix}_name")

        method_idx = SIZING_METHODS.index(default_method) if default_method in SIZING_METHODS else 0
        method = st.selectbox(
            "Sizing Method",
            options=SIZING_METHODS,
            index=method_idx,
            key=f"{key_prefix}_method",
        )

        st.markdown("**Parameters**")

        # Get schema for selected method
        param_schema = SIZING_PARAM_SCHEMAS.get(method, {})
        params: dict[str, str] = {}

        for param_name, schema in param_schema.items():
            default_val = str(existing_params.get(param_name, schema["default"]))
            params[param_name] = st.text_input(
                param_name.replace("_", " ").title(),
                value=default_val,
                key=f"{key_prefix}_{param_name}",
                help=f"Range: {schema['min']} - {schema['max']}",
            )

        submitted = st.form_submit_button("Save Sizing Config")

        if submitted:
            if not name.strip():
                st.error("Config name is required")
                return None
            return (name.strip(), method, params)

    return None


def _render_risk_config_form(
    key_prefix: str,
    existing_config: JsonDict | None = None,
) -> tuple[str, dict[str, str], dict[str, str]] | None:
    """Render form for creating/editing risk config.

    Returns (name, strategy_level, portfolio_level) on submit, None otherwise.
    """
    default_name = existing_config["name"] if existing_config else ""
    existing_strategy = existing_config.get("strategy_level", {}) if existing_config else {}
    existing_portfolio = existing_config.get("portfolio_level", {}) if existing_config else {}

    with st.form(key=f"{key_prefix}_risk_form"):
        name = st.text_input("Config Name", value=default_name, key=f"{key_prefix}_risk_name")

        st.markdown("**Strategy-Level Limits**")
        strategy_level: dict[str, str] = {}

        strategy_level["max_drawdown_pct"] = st.text_input(
            "Max Drawdown %",
            value=str(existing_strategy.get("max_drawdown_pct", "0.15")),
            key=f"{key_prefix}_max_dd",
            help="Halt strategy at this drawdown (0.15 = 15%)",
        )
        strategy_level["max_daily_loss_pct"] = st.text_input(
            "Max Daily Loss %",
            value=str(existing_strategy.get("max_daily_loss_pct", "0.02")),
            key=f"{key_prefix}_daily_loss",
            help="Halt for day at this loss (0.02 = 2%)",
        )
        strategy_level["max_consecutive_losses"] = st.text_input(
            "Max Consecutive Losses",
            value=str(existing_strategy.get("max_consecutive_losses", "10")),
            key=f"{key_prefix}_consec_loss",
        )
        strategy_level["max_position_count"] = st.text_input(
            "Max Position Count",
            value=str(existing_strategy.get("max_position_count", "5")),
            key=f"{key_prefix}_max_pos",
        )

        st.markdown("**Portfolio-Level Limits**")
        portfolio_level: dict[str, str] = {}

        portfolio_level["max_portfolio_drawdown_pct"] = st.text_input(
            "Max Portfolio Drawdown %",
            value=str(existing_portfolio.get("max_portfolio_drawdown_pct", "0.20")),
            key=f"{key_prefix}_portfolio_dd",
            help="Halt all trading at this drawdown (0.20 = 20%)",
        )
        portfolio_level["max_total_exposure_pct"] = st.text_input(
            "Max Total Exposure %",
            value=str(existing_portfolio.get("max_total_exposure_pct", "0.50")),
            key=f"{key_prefix}_exposure",
            help="Max total position value / equity (0.50 = 50%)",
        )
        portfolio_level["max_single_instrument_pct"] = st.text_input(
            "Max Single Instrument %",
            value=str(existing_portfolio.get("max_single_instrument_pct", "0.30")),
            key=f"{key_prefix}_single_inst",
            help="Max concentration in one instrument (0.30 = 30%)",
        )

        submitted = st.form_submit_button("Save Risk Config")

        if submitted:
            if not name.strip():
                st.error("Config name is required")
                return None
            return (name.strip(), strategy_level, portfolio_level)

    return None


def _validate_sizing_params(method: str, params: dict[str, str]) -> dict[str, Decimal] | None:
    """Validate and convert sizing params to Decimal.

    Returns validated dict or None on error (error displayed via st.error).
    """
    schema = SIZING_PARAM_SCHEMAS.get(method, {})
    validated: dict[str, Decimal] = {}

    for name, value in params.items():
        if name not in schema:
            continue
        try:
            dec_val = _parse_decimal(value, name)
            min_val = Decimal(str(schema[name]["min"]))
            max_val = Decimal(str(schema[name]["max"]))

            if dec_val < min_val or dec_val > max_val:
                st.error(f"{name}: value {dec_val} outside range [{min_val}, {max_val}]")
                return None
            validated[name] = dec_val
        except ValueError as e:
            st.error(str(e))
            return None

    return validated


def _validate_risk_params(
    strategy_level: dict[str, str],
    portfolio_level: dict[str, str],
) -> tuple[dict[str, Decimal | int], dict[str, Decimal]] | None:
    """Validate and convert risk params.

    Returns (strategy_dict, portfolio_dict) or None on error.
    """
    try:
        strategy: dict[str, Decimal | int] = {
            "max_drawdown_pct": _parse_decimal(strategy_level["max_drawdown_pct"], "max_drawdown_pct"),
            "max_daily_loss_pct": _parse_decimal(strategy_level["max_daily_loss_pct"], "max_daily_loss_pct"),
            "max_consecutive_losses": int(strategy_level["max_consecutive_losses"]),
            "max_position_count": int(strategy_level["max_position_count"]),
        }

        # Validate ranges
        if not (Decimal(0) < strategy["max_drawdown_pct"] <= Decimal(1)):
            st.error("max_drawdown_pct must be in (0, 1]")
            return None
        if not (Decimal(0) < strategy["max_daily_loss_pct"] <= Decimal(1)):
            st.error("max_daily_loss_pct must be in (0, 1]")
            return None
        if strategy["max_consecutive_losses"] < 1:
            st.error("max_consecutive_losses must be >= 1")
            return None
        if strategy["max_position_count"] < 1:
            st.error("max_position_count must be >= 1")
            return None

        portfolio: dict[str, Decimal] = {
            "max_portfolio_drawdown_pct": _parse_decimal(
                portfolio_level["max_portfolio_drawdown_pct"], "max_portfolio_drawdown_pct"
            ),
            "max_total_exposure_pct": _parse_decimal(
                portfolio_level["max_total_exposure_pct"], "max_total_exposure_pct"
            ),
            "max_single_instrument_pct": _parse_decimal(
                portfolio_level["max_single_instrument_pct"], "max_single_instrument_pct"
            ),
        }

        # Validate portfolio ranges
        if not (Decimal(0) < portfolio["max_portfolio_drawdown_pct"] <= Decimal(1)):
            st.error("max_portfolio_drawdown_pct must be in (0, 1]")
            return None
        if portfolio["max_total_exposure_pct"] <= Decimal(0):
            st.error("max_total_exposure_pct must be positive")
            return None
        if not (Decimal(0) < portfolio["max_single_instrument_pct"] <= Decimal(1)):
            st.error("max_single_instrument_pct must be in (0, 1]")
            return None

        return (strategy, portfolio)
    except (ValueError, KeyError) as e:
        st.error(f"Validation error: {e}")
        return None


def render_sizing_section() -> None:
    """Render position sizing configuration section."""
    st.subheader("Position Sizing Configs")

    manager = _get_state_manager()
    configs = manager.list_sizing_configs()

    # List existing configs
    if configs:
        st.markdown("**Existing Configs:**")
        for cfg in configs:
            with st.expander(f"{cfg['name']} ({cfg['method']})"):
                st.json(cfg["config"])
    else:
        st.info("No sizing configs yet. Create one below.")

    # Create new config
    st.markdown("---")
    st.markdown("**Create New Sizing Config**")

    result = _render_sizing_config_form("new")
    if result:
        name, method, params = result
        validated = _validate_sizing_params(method, params)
        if validated:
            try:
                # Convert Decimals to strings for JSON storage
                config_json = {k: str(v) for k, v in validated.items()}
                manager.create_sizing_config(name, method, config_json)
                st.success(f"Created sizing config: {name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")


def render_risk_section() -> None:
    """Render risk management configuration section."""
    st.subheader("Risk Management Configs")

    manager = _get_state_manager()
    configs = manager.list_risk_configs()

    # List existing configs
    if configs:
        st.markdown("**Existing Configs:**")
        for cfg in configs:
            with st.expander(cfg["name"]):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("*Strategy Level*")
                    st.json(cfg["strategy_level"])
                with col2:
                    st.markdown("*Portfolio Level*")
                    st.json(cfg["portfolio_level"])
    else:
        st.info("No risk configs yet. Create one below.")

    # Create new config
    st.markdown("---")
    st.markdown("**Create New Risk Config**")

    result = _render_risk_config_form("new")
    if result:
        name, strategy_level, portfolio_level = result
        validated = _validate_risk_params(strategy_level, portfolio_level)
        if validated:
            strategy_dict, portfolio_dict = validated
            try:
                # Convert Decimals to strings for JSON storage
                strategy_json = {
                    k: str(v) if isinstance(v, Decimal) else v
                    for k, v in strategy_dict.items()
                }
                portfolio_json = {k: str(v) for k, v in portfolio_dict.items()}
                manager.create_risk_config(name, strategy_json, portfolio_json)
                st.success(f"Created risk config: {name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")


def render_latency_section() -> None:
    """Render latency preset configuration section."""
    st.subheader("Latency Presets")

    st.markdown("**Built-in Presets:**")

    # Display built-in presets
    preset_data = []
    for preset, values in LATENCY_PRESETS.items():
        preset_data.append({
            "Preset": preset.value,
            "Base (ms)": values.base_ms,
            "Insert (ms)": values.insert_ms,
            "Update (ms)": values.update_ms,
            "Cancel (ms)": values.cancel_ms,
        })

    st.dataframe(preset_data, width="stretch")

    st.markdown("---")
    st.markdown("**Custom Latency Configuration**")
    st.info(
        "Custom latency values can be set per-backtest in the Backtest Launch tab. "
        "Use the preset selector or enter custom millisecond values."
    )

    # Show latency selector preview
    selected = st.selectbox(
        "Preview Preset",
        options=[p.value for p in LatencyPreset],
        key="latency_preview",
    )

    if selected:
        values = LATENCY_PRESETS[LatencyPreset(selected)]
        st.metric("Total Insert Latency", f"{values.base_ms + values.insert_ms} ms")
        st.caption(f"Base: {values.base_ms}ms + Insert: {values.insert_ms}ms")


def render_database_section() -> None:
    """Render database path configuration section."""
    st.subheader("Database Configuration")

    current_path = st.session_state.get("db_path", str(DEFAULT_DB_PATH))

    with st.form("db_path_form"):
        new_path = st.text_input(
            "Database Path",
            value=current_path,
            help="Path to SQLite state database. Relative to project root.",
        )

        if st.form_submit_button("Update Database Path"):
            path = Path(new_path)
            if path.suffix != ".db":
                st.error("Database path must end with .db")
            else:
                # Close existing manager if any
                if "state_manager" in st.session_state:
                    st.session_state["state_manager"].close()
                    del st.session_state["state_manager"]

                st.session_state["db_path"] = new_path
                st.success(f"Database path updated to: {new_path}")
                st.rerun()

    # Show current DB stats
    try:
        manager = _get_state_manager()
        cursor = manager.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        st.markdown("**Database Tables:**")
        st.write(", ".join(tables) if tables else "No tables")

        # Row counts for key tables
        if tables:
            st.markdown("**Row Counts:**")
            counts = {}
            for table in ["strategies", "sizing_configs", "risk_configs", "backtest_runs"]:
                if table in tables:
                    cursor = manager.conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                    counts[table] = cursor.fetchone()[0]
            st.json(counts)
    except Exception as e:
        st.error(f"Could not read database: {e}")


def render_system_info() -> None:
    """Render system information section."""
    st.subheader("System Information")

    # NautilusTrader version
    try:
        import nautilus_trader
        nt_version = getattr(nautilus_trader, "__version__", "unknown")
    except ImportError:
        nt_version = "not installed"

    # Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("NautilusTrader", nt_version)
        st.metric("Python", py_version)

    with col2:
        # Catalog size (if exists)
        catalog_path = Path("data/catalog")
        if catalog_path.exists():
            total_size = sum(f.stat().st_size for f in catalog_path.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            st.metric("Catalog Size", f"{size_mb:.1f} MB")
        else:
            st.metric("Catalog Size", "N/A")

        # SQLite DB size
        db_path = Path(st.session_state.get("db_path", DEFAULT_DB_PATH))
        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            st.metric("Database Size", f"{size_kb:.1f} KB")
        else:
            st.metric("Database Size", "N/A")


def render_settings_tab() -> None:
    """Render the complete settings tab."""
    st.title("Settings")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sizing",
        "Risk",
        "Latency",
        "Database",
        "System",
    ])

    with tab1:
        render_sizing_section()

    with tab2:
        render_risk_section()

    with tab3:
        render_latency_section()

    with tab4:
        render_database_section()

    with tab5:
        render_system_info()


# Top-level call for st.navigation API
render_settings_tab()
