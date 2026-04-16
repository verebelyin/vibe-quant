"""Real NT-backed runner for purged k-fold CV (bd-xrli).

PurgedKFold yields bar *indices* (0..n_samples-1), but NTScreeningRunner
needs *date strings*. This adapter resolves that gap by loading the bar
timestamps for the run's primary timeframe from the parquet catalog at
construction, then mapping each fold's index range back to a date span
the screening runner can consume.

Train indices in PurgedKFold are non-contiguous (everything except the
test fold, with purge/embargo gaps). NT can't run a single backtest over
a non-contiguous range, so we use min(train_idx)→max(train_idx) as the
train span. Train metrics are diagnostic only — ``CVResult.is_robust``
only consults ``test_sharpe``/``test_return`` (see
``PurgedKFoldCV._aggregate_results``), so the train span overlap with
the test fold is acceptable for the robustness check.
"""

from __future__ import annotations

import glob
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.overfitting.purged_kfold import FoldResult

logger = logging.getLogger(__name__)

# Mirror INTERVAL_TO_AGGREGATION naming used by the parquet catalog
# (data/catalog/data/bar/<symbol>-PERP.BINANCE-<STEP>-<AGG>-LAST-EXTERNAL).
_TIMEFRAME_TO_DIR_SUFFIX: dict[str, str] = {
    "1m": "1-MINUTE",
    "5m": "5-MINUTE",
    "15m": "15-MINUTE",
    "1h": "1-HOUR",
    "4h": "4-HOUR",
    "1d": "1-DAY",
}


class NTPurgedKFoldRunner:
    """Purged-k-fold BacktestRunner backed by NTScreeningRunner.

    Conforms to the ``BacktestRunner`` Protocol in
    :mod:`vibe_quant.overfitting.purged_kfold`. Resolves the strategy DSL,
    symbols, and primary timeframe from the overfitting pipeline's
    ``run_id`` once at construction, loads bar timestamps for the first
    symbol from the parquet catalog, and maps fold indices to date spans
    on each ``run`` call.

    Each ``run`` triggers two NT screening backtests (one per fold for
    train + test). With the default ``n_splits=5`` that's 10 backtests
    per candidate — slow but correct.
    """

    def __init__(
        self,
        run_id: int,
        db_path: str | Path,
        catalog_path: str | Path | None = None,
    ) -> None:
        self._run_id = run_id
        self._db_path = Path(db_path)
        self._catalog_path = Path(catalog_path) if catalog_path else None
        self._dsl_dict: dict[str, object] = {}
        self._symbols: list[str] = []
        self._timeframe: str = ""
        self._bar_ts_ns: list[int] = []
        self._resolve()
        self._load_bar_timestamps()

    @property
    def n_samples(self) -> int:
        """Number of bars available for splitting."""
        return len(self._bar_ts_ns)

    def _resolve(self) -> None:
        """Look up strategy DSL, symbols, and timeframe for the given run."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            run = conn.execute(
                "SELECT strategy_id, symbols, timeframe FROM backtest_runs WHERE id=?",
                (self._run_id,),
            ).fetchone()
            if run is None:
                msg = f"backtest_runs.id={self._run_id} not found"
                raise ValueError(msg)
            strat = conn.execute(
                "SELECT dsl_config FROM strategies WHERE id=?",
                (run["strategy_id"],),
            ).fetchone()
            if strat is None:
                msg = f"strategies.id={run['strategy_id']} not found"
                raise ValueError(msg)
            self._dsl_dict = json.loads(strat["dsl_config"])
            self._symbols = json.loads(run["symbols"])
            # backtest_runs.timeframe is the source of truth for the run's
            # primary timeframe; fall back to the DSL's timeframe field.
            self._timeframe = (
                run["timeframe"]
                or str(self._dsl_dict.get("timeframe", "")).strip()
            )
            if not self._timeframe:
                msg = f"run_id={self._run_id} has no timeframe (DB or DSL)"
                raise ValueError(msg)
        finally:
            conn.close()

    def _load_bar_timestamps(self) -> None:
        """Load ts_event for the primary timeframe of the first symbol.

        We use the first symbol as representative — multi-symbol runs are
        already approximated elsewhere by intersection of date ranges.
        """
        import pandas as pd

        suffix = _TIMEFRAME_TO_DIR_SUFFIX.get(self._timeframe)
        if suffix is None:
            msg = (
                f"Unsupported timeframe '{self._timeframe}' for catalog lookup. "
                f"Supported: {sorted(_TIMEFRAME_TO_DIR_SUFFIX)}"
            )
            raise ValueError(msg)

        catalog_root = self._catalog_path or Path("data/catalog")
        first_symbol = self._symbols[0]
        bar_dir = (
            catalog_root
            / "data"
            / "bar"
            / f"{first_symbol}-PERP.BINANCE-{suffix}-LAST-EXTERNAL"
        )
        files = sorted(glob.glob(str(bar_dir / "*.parquet")))
        if not files:
            msg = f"No parquet files in {bar_dir}"
            raise ValueError(msg)

        timestamps: list[int] = []
        for f in files:
            df = pd.read_parquet(f, columns=["ts_event"])
            timestamps.extend(int(t) for t in df["ts_event"].tolist())
        timestamps.sort()
        self._bar_ts_ns = timestamps
        logger.info(
            "NTPurgedKFoldRunner: loaded %d bars (%s/%s) for run_id=%d",
            len(timestamps),
            first_symbol,
            self._timeframe,
            self._run_id,
        )

    def _index_range_to_dates(self, indices: list[int]) -> tuple[str, str]:
        """Convert an index list to (start_date, end_date) strings.

        Uses min/max so non-contiguous train spans collapse to a single
        date range NT can backtest. Trims to date-only (NT's screening
        runner accepts ``YYYY-MM-DD``).
        """
        if not indices:
            msg = "Cannot convert empty index list to date range"
            raise ValueError(msg)
        lo, hi = indices[0], indices[-1]
        # PurgedKFold yields lists, not necessarily sorted across train
        # ranges. Recompute defensively.
        for i in indices:
            if i < lo:
                lo = i
            if i > hi:
                hi = i
        start_ns = self._bar_ts_ns[lo]
        end_ns = self._bar_ts_ns[hi]
        start = datetime.fromtimestamp(start_ns / 1e9, tz=UTC).date().isoformat()
        end = datetime.fromtimestamp(end_ns / 1e9, tz=UTC).date().isoformat()
        return start, end

    def _backtest(self, start_date: str, end_date: str) -> tuple[float, float]:
        """Run one NT screening backtest, return (sharpe, total_return)."""
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        runner = NTScreeningRunner(
            dsl_dict=self._dsl_dict,
            symbols=self._symbols,
            start_date=start_date,
            end_date=end_date,
            catalog_path=str(self._catalog_path) if self._catalog_path else None,
        )
        # Empty params: use the DSL's defaults (the candidate's params are
        # already baked into the persisted DSL for discovery runs; for
        # screening sweeps the per-candidate params would need plumbing
        # through the candidate row — out of scope for this initial
        # purged-k-fold wiring).
        metrics = runner({})
        sharpe = float(getattr(metrics, "sharpe_ratio", 0.0) or 0.0)
        total_return = float(getattr(metrics, "total_return", 0.0) or 0.0)
        return sharpe, total_return

    def run(self, train_indices: list[int], test_indices: list[int]) -> FoldResult:
        """Execute one purged-k-fold split via two NT screening backtests."""
        from vibe_quant.overfitting.purged_kfold import FoldResult

        train_start, train_end = self._index_range_to_dates(train_indices)
        test_start, test_end = self._index_range_to_dates(test_indices)

        train_sharpe, train_return = self._backtest(train_start, train_end)
        test_sharpe, test_return = self._backtest(test_start, test_end)

        return FoldResult(
            fold_index=0,  # Pipeline overrides via _aggregate_results
            train_size=len(train_indices),
            test_size=len(test_indices),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_return=train_return,
            test_return=test_return,
        )
