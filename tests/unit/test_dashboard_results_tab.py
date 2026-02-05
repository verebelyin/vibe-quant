"""Tests for dashboard results analysis tab."""

from __future__ import annotations

import pandas as pd


class TestResultsAnalysisImports:
    """Test module imports without Streamlit runtime."""

    def test_import_module(self) -> None:
        """Module should import without errors."""
        from vibe_quant.dashboard.pages import results_analysis  # noqa: F401

    def test_import_helper_functions(self) -> None:
        """Helper functions should be importable."""
        from vibe_quant.dashboard.pages.results_analysis import (
            build_equity_curve,
            build_sweep_dataframe,
            build_trades_dataframe,
            compute_drawdown_series,
            export_to_csv,
            format_number,
            format_percent,
        )

        assert callable(format_percent)
        assert callable(format_number)
        assert callable(build_sweep_dataframe)
        assert callable(build_trades_dataframe)
        assert callable(build_equity_curve)
        assert callable(compute_drawdown_series)
        assert callable(export_to_csv)


class TestFormatHelpers:
    """Test formatting helper functions."""

    def test_format_percent_positive(self) -> None:
        """Should format positive percentage."""
        from vibe_quant.dashboard.pages.results_analysis import format_percent

        assert format_percent(0.1234) == "12.34%"

    def test_format_percent_negative(self) -> None:
        """Should format negative percentage."""
        from vibe_quant.dashboard.pages.results_analysis import format_percent

        assert format_percent(-0.05) == "-5.00%"

    def test_format_percent_none(self) -> None:
        """Should handle None value."""
        from vibe_quant.dashboard.pages.results_analysis import format_percent

        assert format_percent(None) == "N/A"

    def test_format_number_default_decimals(self) -> None:
        """Should format number with default decimals."""
        from vibe_quant.dashboard.pages.results_analysis import format_number

        assert format_number(1.5678) == "1.57"

    def test_format_number_custom_decimals(self) -> None:
        """Should format number with custom decimals."""
        from vibe_quant.dashboard.pages.results_analysis import format_number

        assert format_number(1.5678, 0) == "2"
        assert format_number(1.5678, 1) == "1.6"
        assert format_number(1.5678, 4) == "1.5678"

    def test_format_number_none(self) -> None:
        """Should handle None value."""
        from vibe_quant.dashboard.pages.results_analysis import format_number

        assert format_number(None) == "N/A"


class TestBuildSweepDataframe:
    """Test sweep results DataFrame builder."""

    def test_empty_results(self) -> None:
        """Should return empty DataFrame for empty input."""
        from vibe_quant.dashboard.pages.results_analysis import build_sweep_dataframe

        df = build_sweep_dataframe([])
        assert df.empty

    def test_basic_results(self) -> None:
        """Should convert sweep results to DataFrame."""
        from vibe_quant.dashboard.pages.results_analysis import build_sweep_dataframe

        results = [
            {
                "id": 1,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "max_drawdown": 0.1,
                "total_return": 0.25,
                "profit_factor": 1.8,
                "win_rate": 0.55,
                "total_trades": 100,
                "total_fees": 50.0,
                "total_funding": 10.0,
                "is_pareto_optimal": 1,
                "passed_deflated_sharpe": True,
                "passed_walk_forward": True,
                "passed_purged_kfold": False,
                "parameters": {"rsi_period": 14, "threshold": 30},
            }
        ]
        df = build_sweep_dataframe(results)

        assert len(df) == 1
        assert df.iloc[0]["Sharpe"] == 1.5
        assert df.iloc[0]["param_rsi_period"] == 14
        assert df.iloc[0]["DSR Pass"] == True  # noqa: E712

    def test_multiple_results(self) -> None:
        """Should handle multiple results."""
        from vibe_quant.dashboard.pages.results_analysis import build_sweep_dataframe

        results = [
            {"id": 1, "sharpe_ratio": 1.0, "parameters": {}},
            {"id": 2, "sharpe_ratio": 2.0, "parameters": {}},
        ]
        df = build_sweep_dataframe(results)
        assert len(df) == 2


class TestBuildTradesDataframe:
    """Test trades DataFrame builder."""

    def test_empty_trades(self) -> None:
        """Should return empty DataFrame for empty input."""
        from vibe_quant.dashboard.pages.results_analysis import build_trades_dataframe

        df = build_trades_dataframe([])
        assert df.empty

    def test_basic_trades(self) -> None:
        """Should convert trades to DataFrame."""
        from vibe_quant.dashboard.pages.results_analysis import build_trades_dataframe

        trades = [
            {
                "symbol": "BTCUSDT-PERP",
                "direction": "LONG",
                "leverage": 10,
                "entry_time": "2024-01-01T10:00:00",
                "exit_time": "2024-01-01T12:00:00",
                "entry_price": 42000.0,
                "exit_price": 42500.0,
                "quantity": 0.1,
                "gross_pnl": 50.0,
                "net_pnl": 45.0,
                "entry_fee": 2.5,
                "exit_fee": 2.5,
                "exit_reason": "take_profit",
            }
        ]
        df = build_trades_dataframe(trades)

        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "BTCUSDT-PERP"
        assert df.iloc[0]["direction"] == "LONG"


class TestBuildEquityCurve:
    """Test equity curve builder."""

    def test_empty_trades(self) -> None:
        """Should return empty DataFrame for empty input."""
        from vibe_quant.dashboard.pages.results_analysis import build_equity_curve

        df = build_equity_curve([])
        assert df.empty

    def test_no_closed_trades(self) -> None:
        """Should return empty for trades without exit times."""
        from vibe_quant.dashboard.pages.results_analysis import build_equity_curve

        trades = [{"entry_time": "2024-01-01", "net_pnl": 100.0}]  # No exit_time
        df = build_equity_curve(trades)
        assert df.empty

    def test_basic_equity_curve(self) -> None:
        """Should build equity curve from trades."""
        from vibe_quant.dashboard.pages.results_analysis import build_equity_curve

        trades = [
            {"entry_time": "2024-01-01T10:00:00", "exit_time": "2024-01-01T11:00:00", "net_pnl": 100.0},
            {"entry_time": "2024-01-01T12:00:00", "exit_time": "2024-01-01T13:00:00", "net_pnl": -50.0},
            {"entry_time": "2024-01-01T14:00:00", "exit_time": "2024-01-01T15:00:00", "net_pnl": 75.0},
        ]
        df = build_equity_curve(trades)

        assert len(df) == 4  # Initial + 3 trades
        assert df.iloc[0]["equity"] == 0.0
        assert df.iloc[1]["equity"] == 100.0
        assert df.iloc[2]["equity"] == 50.0
        assert df.iloc[3]["equity"] == 125.0


class TestComputeDrawdownSeries:
    """Test drawdown series computation."""

    def test_empty_equity(self) -> None:
        """Should return empty for empty equity curve."""
        from vibe_quant.dashboard.pages.results_analysis import compute_drawdown_series

        df = compute_drawdown_series(pd.DataFrame())
        assert df.empty

    def test_basic_drawdown(self) -> None:
        """Should compute drawdown series."""
        from vibe_quant.dashboard.pages.results_analysis import compute_drawdown_series

        equity_df = pd.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "equity": [100.0, 150.0, 120.0, 180.0],
            }
        )
        dd_df = compute_drawdown_series(equity_df)

        assert len(dd_df) == 4
        assert dd_df.iloc[0]["drawdown"] == 0.0  # No drawdown at start
        assert dd_df.iloc[1]["drawdown"] == 0.0  # New peak
        assert dd_df.iloc[2]["drawdown"] == 0.2  # 30/150 = 20% DD
        assert dd_df.iloc[3]["drawdown"] == 0.0  # New peak


class TestExportToCsv:
    """Test CSV export functionality."""

    def test_export_dataframe(self) -> None:
        """Should export DataFrame to CSV bytes."""
        from vibe_quant.dashboard.pages.results_analysis import export_to_csv

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        csv_bytes = export_to_csv(df, "test.csv")

        assert isinstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")
        assert "col1,col2" in csv_str
        assert "1,a" in csv_str
