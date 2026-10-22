"""Real NT-backed runner for overfitting filters (bd-yfbg).

Thin adapter that wraps :class:`vibe_quant.screening.nt_runner.NTScreeningRunner`
into the :class:`vibe_quant.overfitting.wfa.BacktestRunner` protocol so the
overfitting CLI can run WFA against real NT backtests instead of the
:class:`MockBacktestRunner` synthetic fallback.

Scope: WFA only for now. The purged-k-fold runner protocol works on bar
indices (not dates), so a corresponding adapter needs a different shape
— tracked separately.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.fees import FeeSchedule
from vibe_quant.launch_settings import _run_params, parent_effective_balance
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    from datetime import date

    from vibe_quant.screening.pipeline import ParentFillTicks

logger = logging.getLogger(__name__)


class NTWFARunner:
    """WFA BacktestRunner backed by NTScreeningRunner.

    Resolves the strategy DSL + symbols from the overfitting pipeline's
    ``run_id`` once at construction, then spins up a fresh
    ``NTScreeningRunner`` per WFA window. Conforms to the ``BacktestRunner``
    Protocol defined in :mod:`vibe_quant.overfitting.wfa`.

    The WFA caller passes a per-candidate ``strategy_id`` (actually the
    sweep_result row id) into ``optimize`` / ``backtest``; we ignore it
    and always use the DSL we resolved up front. WFA's single-combo
    param_grid (pipeline.py line 224) makes ``optimize`` a trivial
    backtest of that combo.
    """

    def __init__(self, run_id: int, db_path: str | Path) -> None:
        self._run_id = run_id
        self._db_path = Path(db_path)
        self._dsl_dict: dict[str, object] = {}
        self._symbols: list[str] = []
        self._liquidation: LiquidationSettings = LiquidationSettings()
        # The parent run's own recorded account size (R11); never its parameters.
        self._starting_balance: int = 1000
        # The parent screening run's own fee schedule (VIP tier + BNB discount);
        # default VIP0/no-BNB keeps a params-less run byte-identical (E2).
        self._fee_schedule: FeeSchedule = FeeSchedule()
        # The parent run's own sizing settings (bead vibe-quant-2gyk1.2), read from the
        # same parameters row: a corrupt stored value raises HERE, before the pipeline
        # builds any runner, and the default adds no key (byte-identical, S19).
        self._sizing: SizingSettings = SizingSettings()
        self._ticks: ParentFillTicks | None = None
        self._resolve()

    def _resolve(self) -> None:
        """Look up strategy DSL + symbols for the given run id."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            run = conn.execute(
                "SELECT strategy_id, symbols, parameters FROM backtest_runs WHERE id=?",
                (self._run_id,),
            ).fetchone()
            if run is None:
                msg = f"backtest_runs.id={self._run_id} not found"
                raise ValueError(msg)
            # The parent's RECORDED balance, never its parameters (R11/R11b): a legacy
            # parent whose params say 10000 but whose sweep rows are NULL reruns at 1000.
            self._starting_balance = parent_effective_balance(conn, self._run_id)
            self._liquidation = LiquidationSettings.from_params(_run_params(run["parameters"]))
            # Same params row: the parent's own sizing settings, checked against the
            # leverage the rerun will use (S13). A corrupt or unmarginable stored one
            # raises from this constructor, before the overfitting pipeline exists.
            self._sizing = SizingSettings.from_params(_run_params(run["parameters"]))
            self._sizing.check_leverage(self._liquidation.leverage)
            # Same params row: fail closed on an unknown tier rather than
            # silently backtesting a discounted run at VIP0 (E2).
            self._fee_schedule = FeeSchedule.from_params(_run_params(run["parameters"]))
            strat = conn.execute(
                "SELECT dsl_config FROM strategies WHERE id=?",
                (run["strategy_id"],),
            ).fetchone()
            if strat is None:
                msg = f"strategies.id={run['strategy_id']} not found"
                raise ValueError(msg)
            self._dsl_dict = json.loads(strat["dsl_config"])
            self._symbols = json.loads(run["symbols"])
            from vibe_quant.screening.pipeline import ParentFillTicks

            self._ticks = ParentFillTicks(self._dsl_dict, self._symbols)
        finally:
            conn.close()

    def optimize(
        self,
        strategy_id: str,  # noqa: ARG002 — resolved via run_id at init
        start_date: date,
        end_date: date,
        param_grid: dict[str, list[object]],
    ) -> tuple[dict[str, object], float, float]:
        """WFA optimize() step.

        The overfitting pipeline passes a single-combo param_grid
        (pipeline.py builds ``{k: [v] for k, v in params.items()}``), so
        there is nothing to search over — collapse to the single combo
        and run one backtest.
        """
        params = {k: v[0] for k, v in param_grid.items() if v}
        sharpe, total_return = self.backtest(strategy_id, start_date, end_date, params)
        return params, sharpe, total_return

    def backtest(
        self,
        strategy_id: str,  # noqa: ARG002 — resolved via run_id at init
        start_date: date,
        end_date: date,
        params: dict[str, object],
    ) -> tuple[float, float]:
        """Run a single NT screening backtest and return (sharpe, total_return)."""
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        assert self._ticks is not None
        runner = NTScreeningRunner(
            dsl_dict=self._dsl_dict,
            symbols=self._symbols,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            # the parent run's own fee schedule (E2); VIP0/no-BNB by default
            fee_schedule=self._fee_schedule,
            # resolved by this (parent) process, once per covered span
            fill_ticks=self._ticks.get(start_date.isoformat(), end_date.isoformat()),
            # the screening run's own liquidation settings (E14); off by default
            liquidation_mode=self._liquidation.mode,
            leverage=self._liquidation.leverage,
            # the parent's own recorded account size (R11)
            starting_balance=self._starting_balance,
            # the parent run's own sizing settings (S20)
            sizing=self._sizing,
        )
        numeric_params: dict[str, float | int] = {
            k: v
            for k, v in params.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        metrics = runner(numeric_params)
        sharpe = float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0)
        total_return = float(getattr(metrics, "total_return", 0.0) or 0.0)
        return sharpe, total_return
