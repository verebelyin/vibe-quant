"""Overfitting prevention pipeline orchestrator.

Toggleable filter chain that reads sweep_results, applies DSR/WFA/PurgedKFold
filters, tags pass/fail per filter, and outputs filtered candidates.

Each filter is independent and can be enabled/disabled. Results are stored
back in sweep_results with passed_* flags for each filter.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio, DSRResult
from vibe_quant.overfitting.mock_runner import MockBacktestRunner
from vibe_quant.overfitting.purged_kfold import CVConfig, CVResult, PurgedKFoldCV
from vibe_quant.overfitting.types import (
    CandidateResult,
    FilterConfig,
    PipelineResult,
)
from vibe_quant.overfitting.wfa import WalkForwardAnalysis, WFAConfig, WFAResult

logger = logging.getLogger(__name__)


class OverfittingPipeline:
    """Overfitting prevention filter chain.

    Reads candidates from sweep_results, applies enabled filters,
    updates database with pass/fail flags, and returns filtered candidates.

    Example:
        pipeline = OverfittingPipeline()
        result = pipeline.run(run_id=1, config=FilterConfig.default())
        for candidate in result.filtered_candidates:
            print(f"{candidate.strategy_name}: passed all filters")
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        wfa_runner: Any = None,
        cv_runner: Any = None,
    ) -> None:
        """Initialize overfitting pipeline.

        Args:
            db_path: Path to SQLite database. Defaults to DEFAULT_DB_PATH.
            wfa_runner: Optional backtest runner for WFA. Uses mock if None.
            cv_runner: Optional backtest runner for Purged K-Fold. Uses mock if None.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._wfa_runner = wfa_runner
        self._cv_runner = cv_runner

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def run(
        self,
        run_id: int,
        config: FilterConfig | None = None,
        num_observations: int = 252,
        data_start: date | None = None,
        data_end: date | None = None,
        n_samples: int = 1000,
    ) -> PipelineResult:
        """Run overfitting filter chain on sweep results.

        Args:
            run_id: Backtest run ID to filter candidates for.
            config: Filter configuration. Uses default if None.
            num_observations: Number of observations for DSR (default 252 = 1 year).
            data_start: Start date for WFA windows (optional).
            data_end: End date for WFA windows (optional).
            n_samples: Number of samples for Purged K-Fold (default 1000).

        Returns:
            PipelineResult with all candidate results and filter counts.
        """
        config = config or FilterConfig.default()

        # Load candidates from database
        candidates = self._load_candidates(run_id)
        if not candidates:
            logger.warning("No candidates found for run_id=%d", run_id)
            return PipelineResult(
                config=config,
                total_candidates=0,
                passed_dsr=0,
                passed_wfa=0,
                passed_cv=0,
                passed_all=0,
            )

        logger.info(
            "Running overfitting pipeline on %d candidates (DSR=%s, WFA=%s, CV=%s)",
            len(candidates),
            config.enable_dsr,
            config.enable_wfa,
            config.enable_purged_kfold,
        )

        # Count number of trials for DSR
        num_trials = len(candidates)

        # Initialize filters
        dsr = DeflatedSharpeRatio(significance_level=config.dsr_significance)

        wfa_config = config.wfa_config or WFAConfig.default()
        wfa = WalkForwardAnalysis(config=wfa_config)
        if self._wfa_runner:
            wfa.runner = self._wfa_runner
        else:
            logger.warning(
                "No WFA backtest runner provided - using MockBacktestRunner. "
                "Results will be synthetic. Pass wfa_runner= to OverfittingPipeline "
                "for real walk-forward analysis."
            )
            wfa.runner = MockBacktestRunner()

        cv_config = config.cv_config or CVConfig()
        cv = PurgedKFoldCV(config=cv_config, robustness_threshold=config.cv_robustness_threshold)

        # Process each candidate
        results: list[CandidateResult] = []
        passed_dsr_count = 0
        passed_wfa_count = 0
        passed_cv_count = 0

        for candidate in candidates:
            # Apply DSR filter
            dsr_result: DSRResult | None = None
            passed_dsr: bool | None = None

            if config.enable_dsr:
                # Use actual return distribution moments if available in
                # sweep_results, otherwise fall back to normal distribution
                # assumption (skewness=0, kurtosis=3). For accurate DSR,
                # the screening pipeline should store these in sweep_results.
                skewness = candidate.get("skewness", 0.0)
                kurtosis = candidate.get("kurtosis", 3.0)
                dsr_result = dsr.calculate(
                    observed_sharpe=candidate["sharpe_ratio"],
                    num_trials=num_trials,
                    num_observations=num_observations,
                    skewness=skewness,
                    kurtosis=kurtosis,
                )
                passed_dsr = dsr.passes_threshold(dsr_result, config.dsr_confidence_threshold)
                if passed_dsr:
                    passed_dsr_count += 1

            # Apply WFA filter
            wfa_result: WFAResult | None = None
            passed_wfa: bool | None = None

            if config.enable_wfa:
                # Use provided dates or defaults
                start = data_start or date(2024, 1, 1)
                end = data_end or date(2025, 12, 31)

                # Parse parameters for param_grid
                params = json.loads(candidate["parameters"])
                param_grid = {k: [v] for k, v in params.items()}

                try:
                    wfa_result = wfa.run(
                        strategy_id=str(candidate["id"]),
                        data_start=start,
                        data_end=end,
                        param_grid=param_grid,
                    )
                    passed_wfa = wfa_result.is_robust
                except ValueError as e:
                    logger.warning("WFA failed for candidate %d: %s", candidate["id"], e)
                    passed_wfa = False

                if passed_wfa:
                    passed_wfa_count += 1

            # Apply Purged K-Fold filter
            cv_result: CVResult | None = None
            passed_cv: bool | None = None

            if config.enable_purged_kfold:
                if self._cv_runner:
                    runner = self._cv_runner
                else:
                    if candidate is candidates[0]:  # Log once
                        logger.warning(
                            "No CV backtest runner provided - using MockBacktestRunner. "
                            "Results will be synthetic. Pass cv_runner= to OverfittingPipeline "
                            "for real purged k-fold analysis."
                        )
                    runner = MockBacktestRunner(
                        oos_sharpe=candidate["sharpe_ratio"],
                        oos_return=candidate["total_return"],
                    )
                cv_result = cv.run(n_samples=n_samples, runner=runner)
                passed_cv = cv_result.is_robust
                if passed_cv:
                    passed_cv_count += 1

            # Determine if passed all enabled filters
            passed_all = True
            if config.enable_dsr and not passed_dsr:
                passed_all = False
            if config.enable_wfa and not passed_wfa:
                passed_all = False
            if config.enable_purged_kfold and not passed_cv:
                passed_all = False

            result = CandidateResult(
                sweep_result_id=candidate["id"],
                run_id=candidate["run_id"],
                strategy_name=candidate.get("strategy_name", f"run_{candidate['run_id']}"),
                parameters=candidate["parameters"],
                sharpe_ratio=candidate["sharpe_ratio"],
                total_return=candidate["total_return"],
                passed_dsr=passed_dsr,
                passed_wfa=passed_wfa,
                passed_cv=passed_cv,
                passed_all=passed_all,
                dsr_result=dsr_result,
                wfa_result=wfa_result,
                cv_result=cv_result,
            )
            results.append(result)

            # Update database
            self._update_candidate(candidate["id"], passed_dsr, passed_wfa, passed_cv)

        passed_all_count = sum(1 for r in results if r.passed_all)

        logger.info(
            "Pipeline complete: %d/%d passed all filters (DSR=%d, WFA=%d, CV=%d)",
            passed_all_count,
            len(candidates),
            passed_dsr_count,
            passed_wfa_count,
            passed_cv_count,
        )

        return PipelineResult(
            config=config,
            total_candidates=len(candidates),
            passed_dsr=passed_dsr_count,
            passed_wfa=passed_wfa_count,
            passed_cv=passed_cv_count,
            passed_all=passed_all_count,
            candidates=results,
        )

    def _load_candidates(self, run_id: int) -> list[dict[str, Any]]:
        """Load sweep result candidates from database.

        Args:
            run_id: Backtest run ID.

        Returns:
            List of candidate dictionaries.
        """
        cursor = self.conn.execute(
            """
            SELECT sr.id, sr.run_id, sr.parameters, sr.sharpe_ratio, sr.total_return,
                   sr.sortino_ratio, sr.max_drawdown, sr.profit_factor, sr.win_rate,
                   sr.is_pareto_optimal, br.strategy_id
            FROM sweep_results sr
            LEFT JOIN backtest_runs br ON sr.run_id = br.id
            WHERE sr.run_id = ?
            ORDER BY sr.sharpe_ratio DESC
            """,
            (run_id,),
        )

        candidates: list[dict[str, Any]] = []
        for row in cursor:
            candidates.append(dict(row))

        return candidates

    def _update_candidate(
        self,
        sweep_result_id: int,
        passed_dsr: bool | None,
        passed_wfa: bool | None,
        passed_cv: bool | None,
    ) -> None:
        """Update sweep_result with filter pass/fail flags.

        Args:
            sweep_result_id: ID in sweep_results table.
            passed_dsr: DSR filter result (None if not run).
            passed_wfa: WFA filter result (None if not run).
            passed_cv: CV filter result (None if not run).
        """
        self.conn.execute(
            """
            UPDATE sweep_results
            SET passed_deflated_sharpe = ?,
                passed_walk_forward = ?,
                passed_purged_kfold = ?
            WHERE id = ?
            """,
            (
                1 if passed_dsr else (0 if passed_dsr is not None else None),
                1 if passed_wfa else (0 if passed_wfa is not None else None),
                1 if passed_cv else (0 if passed_cv is not None else None),
                sweep_result_id,
            ),
        )
        self.conn.commit()

    def get_filtered_candidates(
        self,
        run_id: int,
        require_all: bool = True,
    ) -> list[dict[str, Any]]:
        """Get candidates that passed filters from database.

        Args:
            run_id: Backtest run ID.
            require_all: If True, require all filters to pass. If False, any filter.

        Returns:
            List of candidate dictionaries that passed filters.
        """
        if require_all:
            # All enabled filters must pass (non-NULL and = 1)
            query = """
                SELECT * FROM sweep_results
                WHERE run_id = ?
                AND (passed_deflated_sharpe IS NULL OR passed_deflated_sharpe = 1)
                AND (passed_walk_forward IS NULL OR passed_walk_forward = 1)
                AND (passed_purged_kfold IS NULL OR passed_purged_kfold = 1)
                ORDER BY sharpe_ratio DESC
            """
        else:
            # Any filter passes
            query = """
                SELECT * FROM sweep_results
                WHERE run_id = ?
                AND (passed_deflated_sharpe = 1
                     OR passed_walk_forward = 1
                     OR passed_purged_kfold = 1)
                ORDER BY sharpe_ratio DESC
            """

        cursor = self.conn.execute(query, (run_id,))
        return [dict(row) for row in cursor]

    def generate_report(self, result: PipelineResult) -> str:
        """Generate text report from pipeline result.

        Args:
            result: Pipeline result to report.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 70,
            "OVERFITTING PREVENTION PIPELINE REPORT",
            "=" * 70,
            "",
            f"Total candidates: {result.total_candidates}",
            "",
            "-" * 70,
            "FILTER RESULTS",
            "-" * 70,
        ]

        if result.config.enable_dsr:
            pct = (result.passed_dsr / result.total_candidates * 100) if result.total_candidates else 0
            lines.append(f"  DSR (Deflated Sharpe):   {result.passed_dsr}/{result.total_candidates} ({pct:.1f}%)")
        else:
            lines.append("  DSR (Deflated Sharpe):   DISABLED")

        if result.config.enable_wfa:
            pct = (result.passed_wfa / result.total_candidates * 100) if result.total_candidates else 0
            lines.append(f"  WFA (Walk-Forward):      {result.passed_wfa}/{result.total_candidates} ({pct:.1f}%)")
        else:
            lines.append("  WFA (Walk-Forward):      DISABLED")

        if result.config.enable_purged_kfold:
            pct = (result.passed_cv / result.total_candidates * 100) if result.total_candidates else 0
            lines.append(f"  Purged K-Fold CV:        {result.passed_cv}/{result.total_candidates} ({pct:.1f}%)")
        else:
            lines.append("  Purged K-Fold CV:        DISABLED")

        lines.append("")
        pct = (result.passed_all / result.total_candidates * 100) if result.total_candidates else 0
        lines.append(f"  PASSED ALL FILTERS:      {result.passed_all}/{result.total_candidates} ({pct:.1f}%)")
        lines.append("")

        if result.filtered_candidates:
            lines.append("-" * 70)
            lines.append("FILTERED CANDIDATES (top 10)")
            lines.append("-" * 70)

            for i, c in enumerate(result.filtered_candidates[:10]):
                lines.append(f"\n  [{i+1}] {c.strategy_name}")
                lines.append(f"      Sharpe: {c.sharpe_ratio:.3f}  Return: {c.total_return:.2f}%")
                lines.append(f"      DSR: {'PASS' if c.passed_dsr else ('FAIL' if c.passed_dsr is False else 'N/A')}")
                lines.append(f"      WFA: {'PASS' if c.passed_wfa else ('FAIL' if c.passed_wfa is False else 'N/A')}")
                lines.append(f"      CV:  {'PASS' if c.passed_cv else ('FAIL' if c.passed_cv is False else 'N/A')}")

        else:
            lines.append("-" * 70)
            lines.append("NO CANDIDATES PASSED ALL FILTERS")
            lines.append("-" * 70)

        return "\n".join(lines)


def run_overfitting_pipeline(
    run_id: int,
    db_path: str | Path | None = None,
    config: FilterConfig | None = None,
    num_observations: int = 252,
) -> PipelineResult:
    """Convenience function to run overfitting pipeline.

    Args:
        run_id: Backtest run ID to filter candidates for.
        db_path: Optional database path.
        config: Filter configuration. Uses default if None.
        num_observations: Number of observations for DSR.

    Returns:
        PipelineResult with all candidate results.
    """
    pipeline = OverfittingPipeline(db_path)
    try:
        return pipeline.run(run_id=run_id, config=config, num_observations=num_observations)
    finally:
        pipeline.close()
