"""Data domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DataStatusResponse(BaseModel):
    archive_size_bytes: int
    catalog_size_bytes: int
    total_size_bytes: int


class DataCoverageItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    start_date: str
    end_date: str
    kline_count: int
    bar_count: int
    funding_rate_count: int


class DataCoverageResponse(BaseModel):
    coverage: list[DataCoverageItem]


class IngestRequest(BaseModel):
    symbols: list[str]
    start_date: str
    end_date: str


class IngestPreviewResponse(BaseModel):
    total_months: int
    archived_months: int
    new_months: int


class BrowseDataResponse(BaseModel):
    symbol: str
    interval: str
    data: list[dict[str, object]]


class IndicatorSeriesPoint(BaseModel):
    time: int  # open_time ms (matches browse_data format)
    value: float | None  # None during warmup


class IndicatorSeries(BaseModel):
    name: str  # "ema_20", "bbands_20"
    output_name: str  # "value", "upper", "middle", "lower", "macd", "signal", "histogram"
    indicator_type: str  # "EMA", "BBANDS", "RSI"
    display_label: str  # "EMA(20)", "BB Upper(20, 2.0)"
    pane: str  # "overlay" | "oscillator"
    params: dict[str, object]
    data: list[IndicatorSeriesPoint]


class IndicatorsResponse(BaseModel):
    symbol: str
    interval: str
    series: list[IndicatorSeries]


class OhlcError(BaseModel):
    timestamp: str
    error_type: str  # 'high_lt_low', 'zero_close', 'negative_volume', 'zero_open'
    values: dict[str, object]


class DataQualityResponse(BaseModel):
    symbol: str
    gaps: list[dict[str, object]]
    quality_score: float | None
    ohlc_errors: list[OhlcError] = []
    ohlc_error_count: int = 0
    error: str | None = None
