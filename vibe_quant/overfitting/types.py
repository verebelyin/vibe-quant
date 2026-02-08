"""Type definitions for overfitting pipeline.

Dataclasses used across the overfitting module for filter configuration
and pipeline results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.overfitting.dsr import DSRResult
    from vibe_quant.overfitting.purged_kfold import CVConfig, CVResult
    from vibe_quant.overfitting.wfa import WFAConfig, WFAResult


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Configuration for overfitting filter chain.

    Attributes:
        enable_dsr: Enable Deflated Sharpe Ratio filter.
        enable_wfa: Enable Walk-Forward Analysis filter.
        enable_purged_kfold: Enable Purged K-Fold CV filter.
        dsr_significance: DSR significance level (default 0.05).
        dsr_confidence_threshold: Confidence threshold for pass (default 0.95).
        wfa_config: WFA configuration. Uses default if None.
        cv_config: Purged K-Fold configuration. Uses default if None.
        cv_robustness_threshold: Threshold for CV robustness (default 0.5).
    """

    enable_dsr: bool = True
    enable_wfa: bool = True
    enable_purged_kfold: bool = True
    dsr_significance: float = 0.05
    dsr_confidence_threshold: float = 0.95
    wfa_config: WFAConfig | None = None
    cv_config: CVConfig | None = None
    cv_robustness_threshold: float = 0.5

    @classmethod
    def default(cls) -> FilterConfig:
        """Return default configuration with all filters enabled."""
        return cls()

    @classmethod
    def dsr_only(cls) -> FilterConfig:
        """Return configuration with only DSR enabled."""
        return cls(enable_dsr=True, enable_wfa=False, enable_purged_kfold=False)

    @classmethod
    def wfa_only(cls) -> FilterConfig:
        """Return configuration with only WFA enabled."""
        return cls(enable_dsr=False, enable_wfa=True, enable_purged_kfold=False)

    @classmethod
    def cv_only(cls) -> FilterConfig:
        """Return configuration with only Purged K-Fold enabled."""
        return cls(enable_dsr=False, enable_wfa=False, enable_purged_kfold=True)


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Result of overfitting filter chain for a single candidate.

    Attributes:
        sweep_result_id: ID in sweep_results table.
        run_id: Associated backtest run ID.
        strategy_name: Strategy name.
        parameters: Strategy parameters JSON.
        sharpe_ratio: Observed Sharpe ratio.
        total_return: Total return percentage.
        passed_dsr: Whether passed DSR filter (None if disabled).
        passed_wfa: Whether passed WFA filter (None if disabled).
        passed_cv: Whether passed Purged K-Fold filter (None if disabled).
        passed_all: Whether passed all enabled filters.
        dsr_result: Full DSR result (None if disabled).
        wfa_result: Full WFA result (None if disabled).
        cv_result: Full CV result (None if disabled).
    """

    sweep_result_id: int
    run_id: int
    strategy_name: str
    parameters: str
    sharpe_ratio: float
    total_return: float
    passed_dsr: bool | None
    passed_wfa: bool | None
    passed_cv: bool | None
    passed_all: bool
    dsr_result: DSRResult | None = None
    wfa_result: WFAResult | None = None
    cv_result: CVResult | None = None


@dataclass
class PipelineResult:
    """Aggregated result from overfitting pipeline.

    Attributes:
        config: Filter configuration used.
        total_candidates: Total candidates evaluated.
        passed_dsr: Number passing DSR (0 if disabled).
        passed_wfa: Number passing WFA (0 if disabled).
        passed_cv: Number passing Purged K-Fold (0 if disabled).
        passed_all: Number passing all enabled filters.
        candidates: List of all candidate results.
        filtered_candidates: Candidates that passed all enabled filters.
    """

    config: FilterConfig
    total_candidates: int
    passed_dsr: int
    passed_wfa: int
    passed_cv: int
    passed_all: int
    candidates: list[CandidateResult] = field(default_factory=list)

    @property
    def filtered_candidates(self) -> list[CandidateResult]:
        """Get candidates that passed all enabled filters."""
        return [c for c in self.candidates if c.passed_all]
