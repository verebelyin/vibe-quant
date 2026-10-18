"""Overfitting prevention pipeline orchestrator.

Toggleable filter chain that reads sweep_results, applies DSR/WFA/PurgedKFold
filters, tags pass/fail per filter, and outputs filtered candidates.

Each filter is independent and can be enabled/disabled. Results are stored
back in sweep_results with passed_* flags for each filter.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.db.schema import migrate_add_columns
from vibe_quant.errors import DataUnavailableError
from vibe_quant.overfitting.dsr import (
    TRADING_DAYS_PER_YEAR,
    DeflatedSharpeRatio,
    DSRResult,
    mtm_dsr_calculate,
)
from vibe_quant.overfitting.mock_runner import MockBacktestRunner
from vibe_quant.overfitting.purged_kfold import CVConfig, CVResult, PurgedKFoldCV
from vibe_quant.overfitting.types import (
    CandidateResult,
    FilterConfig,
    PipelineResult,
)
from vibe_quant.overfitting.wfa import WalkForwardAnalysis, WFAConfig, WFAResult

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel: leave a sweep_results flag column untouched."""


_UNSET = _Unset()

#: ``input`` of a discovery DSR record that was computed on the pooled
#: mark-to-market daily series (``apply_discovery_dsr_returns``). Older records
#: ("pooled_daily_returns") tested the realized-balance series and are not
#: trusted for promotion. Duplicated here to avoid importing discovery.
_MTM_DSR_INPUT = "pooled_mtm_daily_returns"

#: sweep_results columns the DSR recheck reads; the connection must have them all.
_MTM_SWEEP_COLUMNS = ("mtm_sharpe_daily", "mtm_skewness", "mtm_kurtosis", "mtm_days")


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
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA foreign_keys=ON")
                # This connection bypasses StateManager/init_schema, so a DB from
                # before a column migration (e.g. sweep_results.mtm_*) would raise
                # on promotion instead of failing DSR closed. The shared helper
                # now propagates an error from a needed ALTER (a concurrent writer
                # -> SQLITE_BUSY reaches this except), and the column check below
                # still fails closed if the helper ever no-ops instead.
                migrate_add_columns(conn)
                present = {r[1] for r in conn.execute("PRAGMA table_info(sweep_results)")}
                missing = [c for c in _MTM_SWEEP_COLUMNS if present and c not in present]
                if missing:
                    msg = (
                        "sweep_results migration incomplete (database busy?): "
                        f"missing {', '.join(missing)}; retry"
                    )
                    raise RuntimeError(msg)
            except BaseException:
                conn.close()
                raise
            self._conn = conn
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
        data_start: date | None = None,
        data_end: date | None = None,
        n_samples: int = 1000,
        allow_mock: bool = False,
        total_trials: int | None = None,
        trials_sharpe_variance: float | None = None,
    ) -> PipelineResult:
        """Run overfitting filter chain on sweep results.

        Args:
            run_id: Backtest run ID to filter candidates for.
            config: Filter configuration. Uses default if None.
            data_start: Start date for WFA windows. ``None`` = run start date.
            data_end: End date for WFA windows. ``None`` = run end date.
            n_samples: Number of samples for Purged K-Fold (default 1000).
            allow_mock: If True, allow MockBacktestRunner fallback for WFA/CV.
                If False (default), raise ValueError when no real runner is
                injected. Mock-derived verdicts are NEVER persisted to the DB.
            total_trials: Total number of strategies evaluated (DSR N). When
                ``None``: a discovery candidate uses its run's ``evaluated``
                count (every GA evaluation is a trial), else len(candidates).
            trials_sharpe_variance: Empirical variance of Sharpe ratios across
                all evaluated trials (for DSR).

        Returns:
            PipelineResult with all candidate results and filter counts.

        Raises:
            ValueError: If WFA or CV is enabled, no runner is injected, and
                allow_mock is False; or WFA dates can't be resolved.
        """
        config = config or FilterConfig.default()
        run_dates = self._run_dates(run_id)

        # Load candidates from database
        candidates = self._load_candidates(run_id)
        if not candidates:
            logger.error("No candidates found for run_id=%d", run_id)
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

        # Initialize filters
        dsr = DeflatedSharpeRatio(significance_level=config.dsr_significance)

        wfa_config = config.wfa_config or WFAConfig.default()
        wfa = WalkForwardAnalysis(config=wfa_config)
        wfa_is_mock = False
        cv_is_mock = config.enable_purged_kfold and self._cv_runner is None
        if self._wfa_runner:
            wfa.runner = self._wfa_runner
        elif config.enable_wfa:
            wfa_is_mock = True
            if not allow_mock:
                raise ValueError(
                    "WFA enabled but no backtest runner injected. "
                    "Pass wfa_runner= to OverfittingPipeline, or use --allow-mock / allow_mock=True "
                    "to fall back to MockBacktestRunner (synthetic results)."
                )
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
            dsr_fail_reason: str | None = None

            if config.enable_dsr:
                # Tests the MARK-TO-MARKET daily statistics (bd vibe-quant-68630),
                # never the realized Sharpe: the realized series books a multi-day
                # trade's whole PnL on its exit day. Missing stats fail closed.
                num_trials = self._resolve_num_trials(candidate, total_trials, len(candidates))
                dsr_result, dsr_fail_reason = mtm_dsr_calculate(
                    dsr,
                    candidate.get("mtm_sharpe_daily"),
                    candidate.get("mtm_skewness"),
                    candidate.get("mtm_kurtosis"),
                    candidate.get("mtm_days"),
                    num_trials,
                    trials_sharpe_variance=(
                        trials_sharpe_variance / TRADING_DAYS_PER_YEAR
                        if trials_sharpe_variance is not None
                        else None
                    ),
                )
                if dsr_result is None:
                    passed_dsr = False
                else:
                    passed_dsr = dsr.passes_threshold(dsr_result, config.dsr_confidence_threshold)
                    if not passed_dsr:
                        dsr_fail_reason = (
                            f"DSR not significant: p={dsr_result.p_value:.4f} "
                            f"(T={candidate.get('mtm_days')}, N={num_trials})"
                        )
                    floor_reasons = self._symbol_floor_reasons(candidate)
                    if floor_reasons:
                        passed_dsr = False
                        dsr_fail_reason = "; ".join(
                            ([dsr_fail_reason] if dsr_fail_reason else []) + floor_reasons
                        )
                if dsr_fail_reason:
                    logger.info(
                        "DSR fail for sweep_result %s: %s", candidate.get("id"), dsr_fail_reason
                    )
                if passed_dsr:
                    passed_dsr_count += 1

            # Apply WFA filter
            wfa_result: WFAResult | None = None
            passed_wfa: bool | None = None

            if config.enable_wfa:
                # The run's own window -- never a hardcoded calendar range
                start = data_start or (run_dates[0] if run_dates else None)
                end = data_end or (run_dates[1] if run_dates else None)
                if start is None or end is None:
                    msg = (
                        f"WFA needs a date range: run {run_id} has no start/end dates "
                        "and none were passed (--start-date/--end-date)"
                    )
                    raise ValueError(msg)

                # WALK-FORWARD STABILITY TEST: the candidate's own parameter
                # combo is held fixed across windows (single-combo grid), so
                # "optimize" on IS is just a backtest of that combo. This tests
                # whether one fixed parameter set keeps working out of sample;
                # it is NOT a re-optimizing walk-forward of the search procedure.
                param_grid = {k: [v] for k, v in self._candidate_params(candidate).items()}

                try:
                    wfa_result = wfa.run(
                        strategy_id=str(candidate["id"]),
                        data_start=start,
                        data_end=end,
                        param_grid=param_grid,
                    )
                    passed_wfa = wfa_result.is_robust
                except DataUnavailableError:
                    # Missing data (a data hole, or the mode-on liquidation
                    # path -- LiquidationIntegrityError is a subclass): fail the
                    # run closed (I11) instead of masking it as passed_wfa=False,
                    # which would make a data hole look like "not robust". The
                    # swallow below now only covers non-data ValueErrors.
                    raise
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
                    # Each candidate is CV'd with ITS params (the runner used
                    # to run the DSL defaults for every candidate -> identical
                    # verdicts across the whole sweep).
                    bind = getattr(self._cv_runner, "bind_params", None)
                    runner = (
                        bind(self._candidate_params(candidate))
                        if callable(bind)
                        else self._cv_runner
                    )
                elif not allow_mock:
                    raise ValueError(
                        "Purged K-Fold CV enabled but no backtest runner injected. "
                        "Pass cv_runner= to OverfittingPipeline, or use --allow-mock / allow_mock=True "
                        "to fall back to MockBacktestRunner (synthetic results)."
                    )
                else:
                    if candidate == candidates[0]:  # Log once
                        logger.warning(
                            "No CV backtest runner provided - using MockBacktestRunner. "
                            "Results will be synthetic. Pass cv_runner= to OverfittingPipeline "
                            "for real purged k-fold analysis."
                        )
                    runner = MockBacktestRunner(
                        oos_sharpe=candidate.get("sharpe_ratio") or 0.0,
                        oos_return=candidate.get("total_return") or 0.0,
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

            # Normalize types for CandidateResult (expects str parameters, float sharpe/return)
            raw_params_val = candidate.get("parameters", "{}")
            norm_params = (
                json.dumps(raw_params_val)
                if isinstance(raw_params_val, dict)
                else str(raw_params_val or "{}")
            )
            result = CandidateResult(
                sweep_result_id=candidate["id"],
                run_id=candidate["run_id"],
                strategy_name=candidate.get("strategy_name", f"run_{candidate['run_id']}"),
                parameters=norm_params,
                sharpe_ratio=float(candidate.get("sharpe_ratio") or 0.0),
                total_return=float(candidate.get("total_return") or 0.0),
                passed_dsr=passed_dsr,
                passed_wfa=passed_wfa,
                passed_cv=passed_cv,
                passed_all=passed_all,
                dsr_result=dsr_result,
                wfa_result=wfa_result,
                cv_result=cv_result,
                dsr_fail_reason=dsr_fail_reason,
            )
            results.append(result)

            # Update database -- only verdicts from REAL backtests. Synthetic
            # (MockBacktestRunner) verdicts are reported but never persisted:
            # sweep_results has no provenance column, so a stored passed_* flag
            # must always mean "passed on real data".
            self._update_candidate(
                candidate["id"],
                passed_dsr,
                _UNSET if wfa_is_mock else passed_wfa,
                _UNSET if cv_is_mock else passed_cv,
            )

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

        Queries sweep_results first. If empty and the run is a discovery run,
        falls back to backtest_results and promotes the entry into sweep_results
        so downstream update/filter queries work unchanged.

        Args:
            run_id: Backtest run ID.

        Returns:
            List of candidate dictionaries.
        """
        cursor = self.conn.execute(
            """
            SELECT sr.*, br.strategy_id, s.name AS strategy_name
            FROM sweep_results sr
            LEFT JOIN backtest_runs br ON sr.run_id = br.id
            LEFT JOIN strategies s ON br.strategy_id = s.id
            WHERE sr.run_id = ?
            ORDER BY sr.sharpe_ratio DESC
            """,
            (run_id,),
        )

        candidates: list[dict[str, Any]] = []
        for row in cursor:
            candidates.append(dict(row))

        if not candidates:
            candidates = self._promote_discovery_results(run_id)

        return candidates

    def _promote_discovery_results(self, run_id: int) -> list[dict[str, Any]]:
        """Copy backtest_results into sweep_results for discovery runs.

        Discovery-mode runs store results in backtest_results, not sweep_results.
        This copies them so the rest of the pipeline works unchanged.

        Returns:
            List of promoted candidate dicts (empty if not a discovery run or
            no backtest_results exist).
        """
        row = self.conn.execute(
            "SELECT run_mode FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row or row["run_mode"] != "discovery":
            return []

        br_row = self.conn.execute(
            """
            SELECT br.run_id, br.sharpe_ratio, br.sortino_ratio, br.max_drawdown,
                   br.total_return, br.profit_factor, br.win_rate, br.total_trades,
                   br.total_fees, br.total_funding, br.execution_time_seconds,
                   br.skewness, br.kurtosis,
                   br.notes, r.strategy_id, s.name AS strategy_name
            FROM backtest_results br
            LEFT JOIN backtest_runs r ON br.run_id = r.id
            LEFT JOIN strategies s ON r.strategy_id = s.id
            WHERE br.run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if not br_row:
            return []

        d = dict(br_row)
        parameters = d.pop("notes", None) or "{}"
        # MTM DSR inputs come from the champion's persisted DSR record, never
        # from the headline replay's realized Sharpe (bd vibe-quant-68630).
        mtm = self._mtm_columns_from_notes(parameters)
        d.pop("strategy_name", None)
        d.pop("strategy_id", None)
        d.pop("total_trades", None)

        cursor = self.conn.execute(
            """
            INSERT INTO sweep_results
                (run_id, parameters, sharpe_ratio, sortino_ratio, max_drawdown,
                 total_return, profit_factor, win_rate, total_fees, total_funding,
                 execution_time_seconds, skewness, kurtosis,
                 mtm_sharpe_daily, mtm_skewness, mtm_kurtosis, mtm_days, is_pareto_optimal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                run_id,
                parameters,
                d.get("sharpe_ratio"),
                d.get("sortino_ratio"),
                d.get("max_drawdown"),
                d.get("total_return"),
                d.get("profit_factor"),
                d.get("win_rate"),
                d.get("total_fees"),
                d.get("total_funding"),
                d.get("execution_time_seconds"),
                d.get("skewness"),
                d.get("kurtosis"),
                mtm["mtm_sharpe_daily"],
                mtm["mtm_skewness"],
                mtm["mtm_kurtosis"],
                mtm["mtm_days"],
            ),
        )
        self.conn.commit()

        sweep_id = cursor.lastrowid
        logger.info(
            "Promoted discovery backtest_result to sweep_results id=%d for run_id=%d",
            sweep_id,
            run_id,
        )

        return [
            {
                "id": sweep_id,
                "run_id": run_id,
                "parameters": parameters,
                "sharpe_ratio": d.get("sharpe_ratio"),
                "total_return": d.get("total_return"),
                "sortino_ratio": d.get("sortino_ratio"),
                "max_drawdown": d.get("max_drawdown"),
                "profit_factor": d.get("profit_factor"),
                "win_rate": d.get("win_rate"),
                "skewness": d.get("skewness"),
                "kurtosis": d.get("kurtosis"),
                **mtm,
                "is_pareto_optimal": 1,
                "strategy_id": br_row["strategy_id"],
                "strategy_name": br_row["strategy_name"],
            }
        ]

    def _update_candidate(
        self,
        sweep_result_id: int,
        passed_dsr: bool | None | _Unset,
        passed_wfa: bool | None | _Unset,
        passed_cv: bool | None | _Unset,
    ) -> None:
        """Update sweep_result with filter pass/fail flags.

        Args:
            sweep_result_id: ID in sweep_results table.
            passed_dsr: DSR filter result (None if not run -> NULL).
            passed_wfa: WFA filter result (None -> NULL; ``_UNSET`` -> column
                left untouched, used for synthetic/mock verdicts).
            passed_cv: CV filter result (same convention).
        """

        def _flag(value: bool | None) -> int | None:
            return 1 if value else (0 if value is not None else None)

        columns = (
            ("passed_deflated_sharpe", passed_dsr),
            ("passed_walk_forward", passed_wfa),
            ("passed_purged_kfold", passed_cv),
        )
        sets: list[str] = []
        values: list[int | None] = []
        for column, value in columns:
            if isinstance(value, _Unset):
                continue
            sets.append(f"{column} = ?")
            values.append(_flag(value))
        if not sets:
            return
        # Column names come from the fixed tuple above, values are bound
        self.conn.execute(
            f"UPDATE sweep_results SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
            (*values, sweep_result_id),
        )
        self.conn.commit()

    def _run_dates(self, run_id: int) -> tuple[date, date] | None:
        """(start_date, end_date) of the run, or None when unset/unparseable."""
        row = self.conn.execute(
            "SELECT start_date, end_date FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None or not row["start_date"] or not row["end_date"]:
            return None
        try:
            return (
                date.fromisoformat(str(row["start_date"])[:10]),
                date.fromisoformat(str(row["end_date"])[:10]),
            )
        except ValueError:
            return None

    @staticmethod
    def _discovery_payload(candidate: dict[str, Any]) -> dict[str, Any] | None:
        """Discovery notes when the candidate was promoted from a discovery run."""
        raw = candidate.get("parameters")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if isinstance(raw, dict) and raw.get("type") == "discovery":
            return raw
        return None

    def _candidate_params(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Strategy parameter overrides of a candidate ({} for discovery runs).

        A discovery candidate's ``parameters`` column holds the run's notes
        JSON, not strategy params -- its params are baked into the DSL.
        """
        if self._discovery_payload(candidate) is not None:
            return {}
        raw = candidate.get("parameters", "{}")
        try:
            params = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in parameters for candidate %s", candidate.get("id"))
            return {}
        return params if isinstance(params, dict) else {}

    def _resolve_num_trials(
        self,
        candidate: dict[str, Any],
        total_trials: int | None,
        num_candidates: int,
    ) -> int:
        """DSR trial count N for one candidate.

        Explicit ``total_trials`` wins. A discovery champion was picked from
        ``evaluated`` GA trials (6000 evals != 1 trial); a sweep candidate from
        the full grid (every sweep row is loaded as a candidate).
        """
        if total_trials is not None:
            return max(1, total_trials)
        payload = self._discovery_payload(candidate)
        if payload is not None:
            gate_n = self._effective_rule_gate_n(payload)
            if gate_n is not None:
                return max(gate_n, num_candidates)
            evaluated = payload.get("evaluated")
            if isinstance(evaluated, int) and not isinstance(evaluated, bool) and evaluated > 0:
                return max(evaluated, num_candidates)
            logger.warning(
                "Discovery candidate %s has no 'evaluated' trial count — DSR N falls "
                "back to %d (understates the search)", candidate.get("id"), num_candidates,
            )
        return max(1, num_candidates)

    @staticmethod
    def _effective_rule_gate_n(payload: dict[str, Any]) -> int | None:
        """``notes.dsr_trials.gate_n`` when the run recorded the effective-N rule.

        Design D3 (megq1): discovery judged the champion at its EV-matched
        effective trial count, so the promote-time re-check must use the same N —
        at raw ``evaluated`` a champion discovery passed can be dropped. The
        candidate floor is kept (review N2: ``N = max(gate_n, num_candidates)``).
        Anything unrecognised returns None and keeps the raw fallback, so legacy
        payloads are byte-identical.

        The constant is imported here, not at module level: ``neff`` imports
        ``overfitting.dsr``, so a top-level import cycles through this package's
        ``__init__`` when ``neff`` is imported first (same reason
        ``discovery/pipeline.py`` imports it inside the function).
        """
        from vibe_quant.discovery.neff import N_RULE_EFFECTIVE

        trials = payload.get("dsr_trials")
        if not isinstance(trials, dict) or trials.get("n_rule") != N_RULE_EFFECTIVE:
            return None
        gate_n = trials.get("gate_n")
        if isinstance(gate_n, int) and not isinstance(gate_n, bool) and gate_n >= 1:
            return gate_n
        return None

    @staticmethod
    def _champion_dsr_record(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """DSR record of the run's best champion (``top_strategies[0]``).

        ``top_strategies[0]`` is the champion whose headline is the run's
        ``backtest_results`` row, i.e. the one promoted to ``sweep_results``.
        """
        if payload is None:
            return None
        top = payload.get("top_strategies")
        if not isinstance(top, list) or not top or not isinstance(top[0], dict):
            return None
        record = top[0].get("dsr")
        return record if isinstance(record, dict) else None

    @staticmethod
    def _finite(value: object) -> float | None:
        """``value`` as a finite float, else None (bools and strings rejected)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    def _mtm_columns_from_notes(self, notes_json: str | None) -> dict[str, float | int | None]:
        """Pooled MTM DSR inputs a discovery run persisted for its champion.

        They describe the train-range series the run's own DSR gate tested, so
        the recheck at a different N uses the same Sharpe, moments and T. A
        record from before the MTM switch (``input`` is not the MTM series), a
        missing record or non-numeric fields give all-None -> DSR fails closed.
        """
        none: dict[str, float | int | None] = {
            "mtm_sharpe_daily": None,
            "mtm_skewness": None,
            "mtm_kurtosis": None,
            "mtm_days": None,
        }
        try:
            payload = json.loads(notes_json) if notes_json else None
        except json.JSONDecodeError:
            return none
        record = self._champion_dsr_record(payload if isinstance(payload, dict) else None)
        if record is None or record.get("input") != _MTM_DSR_INPUT or record.get("missing"):
            return none
        sharpe = self._finite(record.get("sharpe_daily"))
        skew = self._finite(record.get("skewness"))
        kurt = self._finite(record.get("kurtosis"))
        days = self._finite(record.get("observations"))
        if sharpe is None or skew is None or kurt is None or days is None:
            return none
        return {
            "mtm_sharpe_daily": sharpe,
            "mtm_skewness": skew,
            "mtm_kurtosis": kurt,
            "mtm_days": int(days),
        }

    def _symbol_floor_reasons(self, candidate: dict[str, Any]) -> list[str]:
        """Worst-mode per-symbol PSR floor of a discovery champion (empty = ok).

        The floor is undeflated PSR(SR > 0) per symbol (N = 1), so it does not
        move with the trial count of the recheck: the persisted PSRs are
        re-applied as they are. Fails closed when the run was worst-mode over
        2+ symbols and the persisted record lacks a symbol's PSR.
        """
        payload = self._discovery_payload(candidate)
        if payload is None or payload.get("symbol_agg") != "worst":
            return []
        syms = payload.get("symbols")
        if not isinstance(syms, list) or len(syms) < 2:
            return []
        record = self._champion_dsr_record(payload) or {}
        per_symbol = record.get("symbols")
        per_symbol = per_symbol if isinstance(per_symbol, dict) else {}
        from vibe_quant.discovery.guardrails import SYMBOL_PSR_FLOOR

        floor = self._finite(record.get("symbol_psr_floor"))
        floor = SYMBOL_PSR_FLOOR if floor is None else floor
        reasons: list[str] = []
        for sym in sorted(str(x) for x in syms):
            entry = per_symbol.get(sym)
            psr = self._finite(entry.get("psr")) if isinstance(entry, dict) else None
            if psr is None:
                reasons.append(f"DSR input missing: no per-symbol PSR for {sym}")
            elif not psr >= floor:
                reasons.append(f"DSR per-symbol floor: {sym} PSR(SR>0)={psr:.3f} < {floor:.2f}")
        return reasons

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
            pct = (
                (result.passed_dsr / result.total_candidates * 100)
                if result.total_candidates
                else 0
            )
            lines.append(
                f"  DSR (Deflated Sharpe):   {result.passed_dsr}/{result.total_candidates} ({pct:.1f}%)"
            )
        else:
            lines.append("  DSR (Deflated Sharpe):   DISABLED")

        if result.config.enable_wfa:
            pct = (
                (result.passed_wfa / result.total_candidates * 100)
                if result.total_candidates
                else 0
            )
            lines.append(
                f"  WFA (Walk-Forward):      {result.passed_wfa}/{result.total_candidates} ({pct:.1f}%)"
            )
            lines.append(
                "    (stability test: params held fixed per window, no re-optimization)"
            )
        else:
            lines.append("  WFA (Walk-Forward):      DISABLED")

        if result.config.enable_purged_kfold:
            pct = (
                (result.passed_cv / result.total_candidates * 100) if result.total_candidates else 0
            )
            lines.append(
                f"  Purged K-Fold CV:        {result.passed_cv}/{result.total_candidates} ({pct:.1f}%)"
            )
        else:
            lines.append("  Purged K-Fold CV:        DISABLED")

        lines.append("")
        pct = (result.passed_all / result.total_candidates * 100) if result.total_candidates else 0
        lines.append(
            f"  PASSED ALL FILTERS:      {result.passed_all}/{result.total_candidates} ({pct:.1f}%)"
        )
        lines.append("")

        if result.filtered_candidates:
            lines.append("-" * 70)
            lines.append("FILTERED CANDIDATES (top 10)")
            lines.append("-" * 70)

            for i, c in enumerate(result.filtered_candidates[:10]):
                lines.append(f"\n  [{i + 1}] {c.strategy_name}")
                lines.append(
                    f"      Sharpe: {c.sharpe_ratio:.3f}  Return: {c.total_return * 100:.2f}%"
                )
                lines.append(
                    f"      DSR: {'PASS' if c.passed_dsr else ('FAIL' if c.passed_dsr is False else 'N/A')}"
                )
                lines.append(
                    f"      WFA: {'PASS' if c.passed_wfa else ('FAIL' if c.passed_wfa is False else 'N/A')}"
                )
                lines.append(
                    f"      CV:  {'PASS' if c.passed_cv else ('FAIL' if c.passed_cv is False else 'N/A')}"
                )

        else:
            lines.append("-" * 70)
            lines.append("NO CANDIDATES PASSED ALL FILTERS")
            lines.append("-" * 70)

        return "\n".join(lines)


def run_overfitting_pipeline(
    run_id: int,
    db_path: str | Path | None = None,
    config: FilterConfig | None = None,
) -> PipelineResult:
    """Convenience function to run overfitting pipeline.

    Args:
        run_id: Backtest run ID to filter candidates for.
        db_path: Optional database path.
        config: Filter configuration. Uses default if None.

    Returns:
        PipelineResult with all candidate results.
    """
    pipeline = OverfittingPipeline(db_path)
    try:
        return pipeline.run(run_id=run_id, config=config)
    finally:
        pipeline.close()
