"""Overfitting prevention pipeline.

Provides toggleable filters for detecting overfit strategies:
- Deflated Sharpe Ratio (DSR): multiple testing correction
- Walk-Forward Analysis (WFA): sliding train/test windows
- Purged K-Fold CV: cross-validation with embargo
- Pipeline Orchestrator: toggleable filter chain
"""

from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    DSRResult,
    calculate_dsr,
)
from vibe_quant.overfitting.pipeline import (
    CandidateResult,
    FilterConfig,
    OverfittingPipeline,
    PipelineResult,
    run_overfitting_pipeline,
)
from vibe_quant.overfitting.purged_kfold import (
    CVConfig,
    CVResult,
    FoldResult,
    PurgedKFold,
    PurgedKFoldCV,
)
from vibe_quant.overfitting.wfa import (
    WalkForwardAnalysis,
    WFAConfig,
    WFAResult,
    WFAWindow,
)

__all__ = [
    # Deflated Sharpe Ratio
    "DSRResult",
    "DeflatedSharpeRatio",
    "calculate_dsr",
    # Pipeline Orchestrator
    "CandidateResult",
    "FilterConfig",
    "OverfittingPipeline",
    "PipelineResult",
    "run_overfitting_pipeline",
    # Purged K-Fold CV
    "CVConfig",
    "CVResult",
    "FoldResult",
    "PurgedKFold",
    "PurgedKFoldCV",
    # Walk-Forward Analysis
    "WalkForwardAnalysis",
    "WFAConfig",
    "WFAResult",
    "WFAWindow",
]
