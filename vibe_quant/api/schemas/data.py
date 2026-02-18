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


class DataQualityResponse(BaseModel):
    symbol: str
    gaps: list[dict[str, object]]
    quality_score: float
