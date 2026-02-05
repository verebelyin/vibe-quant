"""Pytest fixtures for vibe-quant tests."""

import pytest


@pytest.fixture
def sample_dsl_config() -> dict:
    """Sample DSL configuration for testing."""
    return {
        "name": "test_strategy",
        "description": "Test strategy",
        "version": 1,
        "timeframe": "5m",
        "indicators": {
            "rsi": {"type": "RSI", "period": 14, "source": "close"},
        },
        "entry_conditions": {
            "long": ["rsi < 30"],
            "short": ["rsi > 70"],
        },
        "exit_conditions": {
            "long": ["rsi > 50"],
            "short": ["rsi < 50"],
        },
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }
