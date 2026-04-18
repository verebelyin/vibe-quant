"""Paper↔validation reconciliation schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ReconciliationTrade(BaseModel):
    position_id: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    net_pnl: float
    exit_reason: str


class PairedTrade(BaseModel):
    paper: ReconciliationTrade
    validation: ReconciliationTrade
    entry_slippage: float
    pnl_delta: float
    side_agrees: bool


class DivergenceSummary(BaseModel):
    matched: int
    paper_only: int
    validation_only: int
    parity_rate: float
    side_disagreements: int
    mean_entry_slippage: float
    mean_pnl_delta: float


class ReconciliationResponse(BaseModel):
    paper_run: str
    validation_run: str
    tolerance_seconds: int
    paired_trades: list[PairedTrade]
    unpaired_paper: list[ReconciliationTrade]
    unpaired_validation: list[ReconciliationTrade]
    divergence_summary: DivergenceSummary
