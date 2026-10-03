"""Data domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from vibe_quant.api.schemas._validators import check_date_range, clean_symbols


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
    interval: str = "1m"

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        return clean_symbols(v)

    @model_validator(mode="after")
    def _ranges(self) -> IngestRequest:
        check_date_range(self.start_date, self.end_date, allow_equal=True)
        return self


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
    # [{start, end, missing_bars}] — any break in exact 1-minute continuity (capped at 500)
    gaps: list[dict[str, object]]
    gap_count: int = 0
    missing_bars: int = 0
    # [{start, end, bars}] — flat zero-volume filler candles (exchange outages)
    zero_volume_runs: list[dict[str, object]] = []
    zero_volume_bars: int = 0
    # Fraction of bars that are clean (missing/filler/bad-OHLC bars count against it)
    quality_score: float | None
    ohlc_errors: list[OhlcError] = []
    ohlc_error_count: int = 0
    kline_count: int = 0
    error: str | None = None
