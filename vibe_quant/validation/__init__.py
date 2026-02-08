"""Full-fidelity validation backtesting with NautilusTrader.

This module provides:
- Venue configuration with realistic latency and slippage modeling
- Custom fill models for volume-based market impact
- Latency presets for different execution environments
- ValidationRunner for full-fidelity backtesting
"""

from vibe_quant.validation.extraction import (
    compute_extended_metrics,
    estimate_market_stats,
    extract_results,
    extract_stats,
    extract_trades,
)
from vibe_quant.validation.fill_model import (
    ScreeningFillModelConfig,
    SlippageEstimator,
    VolumeSlippageFillModel,
    VolumeSlippageFillModelConfig,
    create_screening_fill_model,
    create_validation_fill_model,
)
from vibe_quant.validation.latency import (
    LATENCY_PRESETS,
    LatencyPreset,
    LatencyValues,
    create_custom_latency_model,
    get_latency_model,
)
from vibe_quant.validation.results import TradeRecord, ValidationResult
from vibe_quant.validation.runner import (
    ValidationRunner,
    ValidationRunnerError,
    list_validation_runs,
)
from vibe_quant.validation.venue import (
    BINANCE_MAKER_FEE,
    BINANCE_TAKER_FEE,
    VenueConfig,
    create_backtest_venue_config,
    create_venue_config_for_screening,
    create_venue_config_for_validation,
)

__all__ = [
    # Latency
    "LatencyPreset",
    "LatencyValues",
    "LATENCY_PRESETS",
    "get_latency_model",
    "create_custom_latency_model",
    # Fill models
    "VolumeSlippageFillModel",
    "VolumeSlippageFillModelConfig",
    "SlippageEstimator",
    "ScreeningFillModelConfig",
    "create_screening_fill_model",
    "create_validation_fill_model",
    # Venue
    "VenueConfig",
    "BINANCE_MAKER_FEE",
    "BINANCE_TAKER_FEE",
    "create_venue_config_for_screening",
    "create_venue_config_for_validation",
    "create_backtest_venue_config",
    # Extraction
    "extract_results",
    "extract_stats",
    "extract_trades",
    "estimate_market_stats",
    "compute_extended_metrics",
    # Runner
    "ValidationRunner",
    "ValidationRunnerError",
    "ValidationResult",
    "TradeRecord",
    "list_validation_runs",
]
