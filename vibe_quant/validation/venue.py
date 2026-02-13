"""Validation venue configuration for NautilusTrader backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from nautilus_trader.backtest.config import (
    ImportableFeeModelConfig,
    ImportableFillModelConfig,
    ImportableLatencyModelConfig,
)
from nautilus_trader.config import (
    BacktestVenueConfig,
    LatencyModelConfig,
)

from vibe_quant.validation.fill_model import (
    ScreeningFillModelConfig,
    VolumeSlippageFillModelConfig,
)
from vibe_quant.validation.latency import (
    LATENCY_PRESETS,
    LatencyPreset,
    LatencyValues,
    get_latency_model,
)

# Default Binance fee rates (percentage)
# Note: Binance has VIP tiers; these are default non-VIP rates
BINANCE_MAKER_FEE = Decimal("0.0002")  # 0.02%
BINANCE_TAKER_FEE = Decimal("0.0004")  # 0.04%


@dataclass
class VenueConfig:
    """Configuration for a simulated trading venue.

    Attributes:
        name: Venue identifier (e.g., "BINANCE").
        starting_balance_usdt: Starting account balance in USDT.
        default_leverage: Default leverage for all instruments.
        leverages: Per-instrument leverage overrides.
        latency_preset: Latency preset or None for no latency.
        latency_config: Custom latency config (overrides preset).
        use_volume_slippage: Use volume-based slippage (validation mode).
        fill_config: Fill model configuration.
        maker_fee: Maker fee rate as decimal.
        taker_fee: Taker fee rate as decimal.
    """

    name: str = "BINANCE"
    starting_balance_usdt: int = 100_000
    default_leverage: Decimal = Decimal("10")
    leverages: dict[str, Decimal] = field(default_factory=dict)

    # Latency configuration
    latency_preset: LatencyPreset | str | None = None
    latency_config: LatencyModelConfig | None = None

    # Fill model configuration
    use_volume_slippage: bool = False
    fill_config: VolumeSlippageFillModelConfig | ScreeningFillModelConfig | None = None

    # Fee configuration (uses instrument fees via MakerTakerFeeModel)
    # These are for reference; actual fees come from instrument definition
    maker_fee: Decimal = BINANCE_MAKER_FEE
    taker_fee: Decimal = BINANCE_TAKER_FEE


def create_venue_config_for_screening(
    starting_balance_usdt: int = 100_000,
    default_leverage: Decimal = Decimal("10"),
    leverages: dict[str, Decimal] | None = None,
) -> VenueConfig:
    """Create VenueConfig optimized for screening mode.

    Screening mode uses:
    - No latency (fast execution)
    - Simple probabilistic fill model
    - Standard fees from instrument definition

    Args:
        starting_balance_usdt: Starting balance.
        default_leverage: Default leverage.
        leverages: Per-instrument leverage overrides.

    Returns:
        VenueConfig for screening.
    """
    return VenueConfig(
        name="BINANCE",
        starting_balance_usdt=starting_balance_usdt,
        default_leverage=default_leverage,
        leverages=leverages or {},
        latency_preset=None,
        use_volume_slippage=False,
        fill_config=ScreeningFillModelConfig(
            prob_fill_on_limit=0.8,
            prob_slippage=0.5,
        ),
    )


def create_venue_config_for_validation(
    starting_balance_usdt: int = 100_000,
    default_leverage: Decimal = Decimal("10"),
    leverages: dict[str, Decimal] | None = None,
    latency_preset: LatencyPreset | str = LatencyPreset.RETAIL,
    impact_coefficient: float = 0.1,
) -> VenueConfig:
    """Create VenueConfig optimized for validation mode.

    Validation mode uses:
    - Realistic latency simulation
    - Post-fill SPEC slippage estimation (engine tick slippage disabled)
    - Standard fees from instrument definition

    Args:
        starting_balance_usdt: Starting balance.
        default_leverage: Default leverage.
        leverages: Per-instrument leverage overrides.
        latency_preset: Latency preset for execution delays.
        impact_coefficient: Market impact coefficient for slippage.

    Returns:
        VenueConfig for validation.
    """
    return VenueConfig(
        name="BINANCE",
        starting_balance_usdt=starting_balance_usdt,
        default_leverage=default_leverage,
        leverages=leverages or {},
        latency_preset=latency_preset,
        use_volume_slippage=True,
        fill_config=VolumeSlippageFillModelConfig(
            impact_coefficient=impact_coefficient,
            prob_fill_on_limit=0.8,
            # Keep engine slippage disabled to prevent dual slippage models:
            # costs are estimated post-fill via SlippageEstimator.
            prob_slippage=0.0,
        ),
    )


def create_backtest_venue_config(config: VenueConfig) -> BacktestVenueConfig:
    """Create NautilusTrader BacktestVenueConfig from VenueConfig.

    Args:
        config: Venue configuration.

    Returns:
        Configured BacktestVenueConfig.
    """
    # Build starting balances list
    starting_balances = [f"{config.starting_balance_usdt} USDT"]

    # Build leverages dict with proper string keys
    leverages: dict[str, float] | None = None
    if config.leverages:
        leverages = {k: float(v) for k, v in config.leverages.items()}

    # Create latency model config if specified
    latency_model: ImportableLatencyModelConfig | None = None
    if config.latency_config is not None:
        latency_model = _create_importable_latency_model_config(config.latency_config)
    elif config.latency_preset is not None:
        latency_config = get_latency_model(config.latency_preset)
        latency_model = _create_importable_latency_model_config(latency_config)

    # Create fill model config
    fill_model = _create_importable_fill_model_config(config)

    # Create fee model config (uses instrument's maker/taker fees)
    fee_model = _create_importable_fee_model_config()

    return BacktestVenueConfig(
        name=config.name,
        oms_type="NETTING",
        account_type="MARGIN",
        starting_balances=starting_balances,
        default_leverage=float(config.default_leverage),
        leverages=leverages,
        fill_model=fill_model,
        latency_model=latency_model,
        fee_model=fee_model,
        bar_execution=True,
        reject_stop_orders=False,
        support_gtd_orders=True,
        support_contingent_orders=True,
        use_position_ids=True,
        use_reduce_only=True,
    )


def _create_importable_fill_model_config(
    config: VenueConfig,
) -> ImportableFillModelConfig:
    """Create ImportableFillModelConfig for BacktestVenueConfig.

    Note: BacktestVenueConfig expects ImportableFillModelConfig which has
    fill_model_path, config_path, and config fields. For simpler cases,
    we use FillModelConfig directly through the config parameter.
    """

    if config.use_volume_slippage:
        # Use our custom VolumeSlippageFillModel
        fill_cfg = config.fill_config
        if fill_cfg is None:
            fill_cfg = VolumeSlippageFillModelConfig()

        if not isinstance(fill_cfg, VolumeSlippageFillModelConfig):
            fill_cfg = VolumeSlippageFillModelConfig()

        return ImportableFillModelConfig(
            fill_model_path="vibe_quant.validation.fill_model:VolumeSlippageFillModel",
            config_path="vibe_quant.validation.fill_model:VolumeSlippageFillModelConfig",
            config={
                "impact_coefficient": fill_cfg.impact_coefficient,
                "prob_fill_on_limit": fill_cfg.prob_fill_on_limit,
                "prob_slippage": fill_cfg.prob_slippage,
            },
        )
    else:
        # Use standard FillModel for screening
        fill_cfg = config.fill_config
        if fill_cfg is None:
            fill_cfg = ScreeningFillModelConfig()

        if not isinstance(fill_cfg, ScreeningFillModelConfig):
            fill_cfg = ScreeningFillModelConfig()

        # For standard FillModel, use the built-in FillModelConfig
        # BacktestVenueConfig will handle this internally
        return ImportableFillModelConfig(
            fill_model_path="nautilus_trader.backtest.models:FillModel",
            config_path="nautilus_trader.config:FillModelConfig",
            config={
                "prob_fill_on_limit": fill_cfg.prob_fill_on_limit,
                "prob_slippage": fill_cfg.prob_slippage,
            },
        )


def _create_importable_latency_model_config(
    latency_config: LatencyModelConfig,
) -> ImportableLatencyModelConfig:
    """Create ImportableLatencyModelConfig for BacktestVenueConfig."""
    return ImportableLatencyModelConfig(
        latency_model_path="nautilus_trader.backtest.models:LatencyModel",
        config_path="nautilus_trader.config:LatencyModelConfig",
        config={
            "base_latency_nanos": latency_config.base_latency_nanos,
            "insert_latency_nanos": latency_config.insert_latency_nanos,
            "update_latency_nanos": latency_config.update_latency_nanos,
            "cancel_latency_nanos": latency_config.cancel_latency_nanos,
        },
    )


def _create_importable_fee_model_config() -> ImportableFeeModelConfig:
    """Create ImportableFeeModelConfig for BacktestVenueConfig.

    Uses MakerTakerFeeModel which reads fees from instrument definition.
    """
    return ImportableFeeModelConfig(
        fee_model_path="nautilus_trader.backtest.models:MakerTakerFeeModel",
        config_path="nautilus_trader.config:MakerTakerFeeModelConfig",
        config={},
    )


# Re-export latency presets for convenience
__all__ = [
    "VenueConfig",
    "BINANCE_MAKER_FEE",
    "BINANCE_TAKER_FEE",
    "LATENCY_PRESETS",
    "LatencyPreset",
    "LatencyValues",
    "create_venue_config_for_screening",
    "create_venue_config_for_validation",
    "create_backtest_venue_config",
]
