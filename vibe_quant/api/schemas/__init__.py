"""API request/response schemas."""

from __future__ import annotations

from vibe_quant.api.schemas.backtest import (
    BacktestLaunchRequest,
    BacktestRunResponse,
    CoverageCheckRequest,
    CoverageCheckResponse,
    HeartbeatRequest,
    JobStatusResponse,
    ParetoMarkRequest,
    SweepResultsBatchRequest,
    TradesBatchRequest,
)
from vibe_quant.api.schemas.data import (
    BrowseDataResponse,
    DataCoverageItem,
    DataCoverageResponse,
    DataQualityResponse,
    DataStatusResponse,
    IngestPreviewResponse,
    IngestRequest,
)
from vibe_quant.api.schemas.discovery import (
    DiscoveryJobResponse,
    DiscoveryLaunchRequest,
    DiscoveryResultResponse,
)
from vibe_quant.api.schemas.paper_trading import (
    CheckpointResponse,
    PaperPositionResponse,
    PaperStartRequest,
    PaperStatusResponse,
)
from vibe_quant.api.schemas.result import (
    BacktestResultResponse,
    ComparisonResponse,
    DrawdownPoint,
    EquityCurvePoint,
    MonthlyReturn,
    NotesUpdateRequest,
    RunListResponse,
    SweepResultResponse,
    TradeResponse,
)
from vibe_quant.api.schemas.settings import (
    DatabaseInfoResponse,
    DatabaseSwitchRequest,
    LatencyPreset,
    RiskConfigCreate,
    RiskConfigResponse,
    RiskConfigUpdate,
    SizingConfigCreate,
    SizingConfigResponse,
    SizingConfigUpdate,
    SystemInfoResponse,
)
from vibe_quant.api.schemas.strategy import (
    StrategyCreate,
    StrategyListResponse,
    StrategyResponse,
    StrategyUpdate,
    ValidationResult,
)

__all__ = [
    # Strategy
    "StrategyCreate",
    "StrategyUpdate",
    "StrategyResponse",
    "StrategyListResponse",
    "ValidationResult",
    # Backtest
    "BacktestLaunchRequest",
    "BacktestRunResponse",
    "JobStatusResponse",
    "HeartbeatRequest",
    "TradesBatchRequest",
    "SweepResultsBatchRequest",
    "ParetoMarkRequest",
    "CoverageCheckRequest",
    "CoverageCheckResponse",
    # Result
    "BacktestResultResponse",
    "TradeResponse",
    "SweepResultResponse",
    "RunListResponse",
    "EquityCurvePoint",
    "DrawdownPoint",
    "MonthlyReturn",
    "ComparisonResponse",
    "NotesUpdateRequest",
    # Discovery
    "DiscoveryLaunchRequest",
    "DiscoveryJobResponse",
    "DiscoveryResultResponse",
    # Paper trading
    "PaperStartRequest",
    "PaperStatusResponse",
    "PaperPositionResponse",
    "CheckpointResponse",
    # Data
    "DataStatusResponse",
    "DataCoverageItem",
    "DataCoverageResponse",
    "IngestRequest",
    "IngestPreviewResponse",
    "BrowseDataResponse",
    "DataQualityResponse",
    # Settings
    "SizingConfigCreate",
    "SizingConfigUpdate",
    "SizingConfigResponse",
    "RiskConfigCreate",
    "RiskConfigUpdate",
    "RiskConfigResponse",
    "LatencyPreset",
    "SystemInfoResponse",
    "DatabaseInfoResponse",
    "DatabaseSwitchRequest",
]
