"""Discovery launch power preflight (bead vibe-quant-clmmc.1). Reporting only.

Before a run starts, say which Sharpe it could certify at all. Evidence grows as
``t = Sharpe x sqrt(years)``, so a short train window or a large trial count can
make a champion impossible before the first backtest. Nothing here feeds a
verdict, fitness or gate: the dict is printed, shown on the launch form and stored
in ``notes.power_preflight``.

Units: Sharpe in sqrt-252 annual units, the DSR gate's own unit
(``overfitting.dsr.TRADING_DAYS_PER_YEAR``); T is the train window's calendar-day
count, exactly as the gate counts it (``utils.compute_day_count``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.errors import DataUnavailableError
from vibe_quant.overfitting.dsr import TRADING_DAYS_PER_YEAR, calculate_dsr, deannualize_sharpe
from vibe_quant.utils import compute_day_count

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The DSR significance the gate tests at.
SIGNIFICANCE = 0.05
#: Bar (a) above this gets the warning (chief pick 2026-11-22; never a refusal).
WARN_ABOVE = 2.0
DSR_OFF = "DSR off"
PRIOR_UNKNOWN = "unknown"
#: ``status`` of a preflight that could not be computed (stored with a ``reason``).
UNAVAILABLE = "unavailable"

_HI_LIMIT = 1e4  # no annual Sharpe below this passes: the bar is unavailable


def dsr_sharpe_bar(num_trials: int, days: int) -> float:
    """Smallest annual Sharpe (sqrt-252) whose DSR p-value is below 0.05.

    Bisection on ``calculate_dsr`` at ``num_trials`` trials over ``days`` daily
    observations, skew 0 and kurtosis 3 (normal returns: the lower bound on the
    bar, fat tails only raise it) and the theoretical null variance (the emax
    rule; the q95 rule's spread is known only after the run). p falls
    monotonically as Sharpe rises, so ``lo`` always fails and ``hi`` passes.

    Raises:
        ValueError: fewer than 2 days (``calculate_dsr``), or no annual Sharpe up to
            ``_HI_LIMIT`` passes (the bar is unbounded; never stored as infinity).
    """

    def passes(annual: float) -> bool:
        p = calculate_dsr(deannualize_sharpe(annual), num_trials, days).p_value
        return p < SIGNIFICANCE

    lo, hi = 0.0, 1.0
    while not passes(hi):
        lo, hi = hi, hi * 2
        if hi > _HI_LIMIT:
            msg = f"no annual Sharpe up to {_HI_LIMIT:g} passes DSR at N={num_trials} over {days} days"
            raise ValueError(msg)
    while hi - lo > 1e-13 * hi:
        mid = (lo + hi) / 2
        if passes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def power_warning(bar_a: float) -> str | None:
    """The plain warning when bar (a) is above ``WARN_ABOVE``, else None."""
    if bar_a > WARN_ABOVE:
        return f"this run can only certify a Sharpe above {bar_a:.2f}; a realistic edge is 0.5-1.5"
    return None


def holdout_t(days: int) -> float:
    """t of an annual Sharpe of 1 over ``days`` daily observations: sqrt(days / 252)."""
    return math.sqrt(days / TRADING_DAYS_PER_YEAR)


def power_preflight(
    *,
    train_start: str,
    train_end: str,
    holdout_start: str | None,
    holdout_end: str | None,
    population: int,
    generations: int,
    prior_n: int | None,
    require_dsr: bool,
) -> dict[str, object]:
    """The power preflight: the two DSR bars, the holdout's length and its t.

    Bar (a) is at ``N = prior_n + 1`` (one genome, the best case); bar (b) at
    ``N = prior_n + population x generations`` (the raw budget; the gate uses the
    effective count, which is at most this). ``prior_n`` None means the prior-trial
    lookup failed: reported as "unknown" and counted as 0, like the gate's own
    fail-open. ``require_dsr`` False replaces both bars with "DSR off".
    """
    train_days = compute_day_count(train_start, train_end) or 0
    holdout_days = (
        (compute_day_count(holdout_start, holdout_end) or 0) if holdout_start and holdout_end else 0
    )
    budget = population * generations
    prior = prior_n or 0
    bar_a: dict[str, object] | str = DSR_OFF
    bar_b: dict[str, object] | str = DSR_OFF
    warning = None
    if require_dsr:
        sharpe_a = dsr_sharpe_bar(prior + 1, train_days)
        bar_a = {"n_trials": prior + 1, "sharpe": sharpe_a}
        bar_b = {"n_trials": prior + budget, "sharpe": dsr_sharpe_bar(prior + budget, train_days)}
        warning = power_warning(sharpe_a)
    pre: dict[str, object] = {
        "sharpe_basis": "sqrt252",
        "train_days": train_days,
        "holdout_days": holdout_days,
        "holdout_t_sharpe_1": holdout_t(holdout_days),
        "budget": budget,
        "prior_n": PRIOR_UNKNOWN if prior_n is None else prior_n,
        "bar_a": bar_a,
        "bar_b": bar_b,
        "warning": warning,
    }
    pre["line"] = power_line(pre)
    return pre


def power_line(pre: dict[str, object]) -> str:
    """The one-line summary the CLI prints and the launch form shows."""
    bar_a, bar_b = pre["bar_a"], pre["bar_b"]
    if isinstance(bar_a, dict) and isinstance(bar_b, dict):
        unknown = " (prior trials unknown)" if pre["prior_n"] == PRIOR_UNKNOWN else ""
        bars = (
            f"DSR bar {bar_a['sharpe']:.2f} at N={bar_a['n_trials']}{unknown}, "
            f"{bar_b['sharpe']:.2f} at N={bar_b['n_trials']} (252-day basis)"
        )
    else:
        bars = DSR_OFF
    days = pre["holdout_days"]
    holdout = (
        f"holdout {days} days, t {pre['holdout_t_sharpe_1']:.2f} at Sharpe 1"
        if days
        else "no holdout"
    )
    return f"Power: {bars}; {holdout}"


def _first_full_day(first_open: datetime) -> str:
    """The first UTC day that starts at or after ``first_open``."""
    day = first_open.replace(hour=0, minute=0, second=0, microsecond=0)
    return (day if day == first_open else day + timedelta(days=1)).strftime("%Y-%m-%d")


def full_history_start(symbols: Sequence[str], catalog_path: Path | str, *, mark: bool) -> str:
    """The latest first full UTC day of 1m bars among ``symbols`` ("Use full history").

    The source is the catalog's 1m bars (the set the launch precheck builds fill
    ticks from), read from parquet metadata; a bar's ``ts_event`` is its close,
    inside its open minute. ``mark`` (liquidation on) also clamps to each symbol's
    first MARK 1m bar (``ts`` holds bar opens).

    Raises:
        DataUnavailableError: a symbol has no 1m bars (or no MARK bars with ``mark``).
    """
    from vibe_quant.data import mark_bars
    from vibe_quant.data.catalog import CatalogManager

    catalog = CatalogManager(Path(catalog_path))
    starts: list[str] = []
    for symbol in symbols:
        bar_range = catalog.get_bar_date_range(symbol, "1m")
        if bar_range is None:
            raise DataUnavailableError(f"{symbol}: no 1m bars in the catalog")
        starts.append(_first_full_day(bar_range[0].replace(second=0, microsecond=0)))
        if mark:
            series = mark_bars.load_mark_series(catalog_path, symbol)
            if series is None or len(series.ts) == 0:
                raise DataUnavailableError(f"{symbol}: no MARK 1m bars in the catalog")
            starts.append(_first_full_day(datetime.fromtimestamp(int(series.ts[0]) / 1e9, UTC)))
    return max(starts)
