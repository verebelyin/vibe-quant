"""Batch validation orchestration for scenario checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vibe_quant.data.ingest import ingest_all
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class ValidationBatchResult:
    """Summary of one validation run created by the batch helper."""

    strategy_id: int
    strategy_name: str
    run_id: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float


def parse_strategy_ids(raw_value: str) -> list[int]:
    """Parse a comma-separated strategy ID string."""
    strategy_ids: list[int] = []
    seen: set[int] = set()

    for chunk in raw_value.split(","):
        value = chunk.strip()
        if not value:
            continue
        strategy_id = int(value)
        if strategy_id not in seen:
            seen.add(strategy_id)
            strategy_ids.append(strategy_id)

    if not strategy_ids:
        raise ValueError("At least one strategy ID is required")

    return strategy_ids


def _parse_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD date into UTC."""
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def validate_archive_window(
    symbols: Sequence[str],
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    """Return symbols whose raw 1m archive does not cover the requested window."""
    from vibe_quant.data.archive import RawDataArchive

    start = _parse_date(start_date).date()
    end = _parse_date(end_date).date()

    archive = RawDataArchive()
    try:
        missing: list[str] = []
        for symbol in symbols:
            date_range = archive.get_date_range(symbol, "1m")
            if date_range is None:
                missing.append(symbol)
                continue

            available_start = datetime.fromtimestamp(date_range[0] / 1000, tz=UTC).date()
            available_end = datetime.fromtimestamp(date_range[1] / 1000, tz=UTC).date()
            if available_start > start or available_end < end:
                missing.append(symbol)

        return missing
    finally:
        archive.close()


def ensure_data_window(
    symbols: Sequence[str],
    *,
    start_date: str,
    end_date: str,
    ingest_missing: bool,
    verbose: bool,
) -> None:
    """Verify data coverage, optionally ingesting missing history first."""
    missing_symbols = validate_archive_window(symbols, start_date=start_date, end_date=end_date)
    if not missing_symbols:
        return

    if not ingest_missing:
        missing_list = ", ".join(missing_symbols)
        raise ValueError(
            "Raw 1m archive does not cover the requested window for: "
            f"{missing_list}. Re-run with --ensure-data to download it."
        )

    ingest_all(
        symbols=list(missing_symbols),
        start_date=_parse_date(start_date),
        end_date=_parse_date(end_date),
        verbose=verbose,
    )

    still_missing = validate_archive_window(symbols, start_date=start_date, end_date=end_date)
    if still_missing:
        missing_list = ", ".join(still_missing)
        raise ValueError(f"Raw 1m archive still missing requested coverage for: {missing_list}")


def run_validation_batch(
    strategy_ids: Sequence[int],
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    db_path: Path | None = None,
    latency_preset: str | None = None,
    ensure_data: bool = False,
    verbose: bool = True,
) -> list[ValidationBatchResult]:
    """Create and execute validation runs for multiple strategies on one scenario window."""
    ensure_data_window(
        [symbol],
        start_date=start_date,
        end_date=end_date,
        ingest_missing=ensure_data,
        verbose=verbose,
    )

    state = StateManager(db_path)
    runner = ValidationRunner(db_path=db_path)
    try:
        results: list[ValidationBatchResult] = []
        for strategy_id in strategy_ids:
            strategy = state.get_strategy(strategy_id)
            if strategy is None:
                raise ValueError(f"Strategy {strategy_id} not found")

            strategy_name = str(strategy["name"])
            run_id = state.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="validation",
                symbols=[symbol],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters={},
                latency_preset=latency_preset,
            )

            if verbose:
                print(
                    f"Running validation for sid={strategy_id} ({strategy_name}) "
                    f"on {symbol} {timeframe} {start_date} -> {end_date}"
                )

            result = runner.run(run_id, latency_preset=latency_preset)
            results.append(
                ValidationBatchResult(
                    strategy_id=strategy_id,
                    strategy_name=result.strategy_name,
                    run_id=run_id,
                    total_return=result.total_return,
                    sharpe_ratio=result.sharpe_ratio,
                    max_drawdown=result.max_drawdown,
                    total_trades=result.total_trades,
                    win_rate=result.win_rate,
                    profit_factor=result.profit_factor,
                )
            )

        return results
    finally:
        runner.close()
        state.close()


def format_batch_results_markdown(results: Sequence[ValidationBatchResult]) -> str:
    """Render batch validation results as a markdown table."""
    lines = [
        "| Strategy ID | Strategy | Run ID | Sharpe | Return | Max DD | Trades | Win Rate | PF |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.strategy_id} | "
            f"{result.strategy_name} | "
            f"{result.run_id} | "
            f"{result.sharpe_ratio:.2f} | "
            f"{result.total_return * 100:.2f}% | "
            f"{result.max_drawdown * 100:.2f}% | "
            f"{result.total_trades} | "
            f"{result.win_rate * 100:.1f}% | "
            f"{result.profit_factor:.2f} |"
        )

    return "\n".join(lines)
