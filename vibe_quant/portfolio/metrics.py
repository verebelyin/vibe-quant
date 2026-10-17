"""Funding-carry metrics: equity path, headline ratios and P&L decomposition.

Design §4 / §8-T4 of the funding-carry engine (epic vibe-quant-x0axf): one
:class:`~vibe_quant.portfolio.ledger.CarryRun` plus its
:class:`~vibe_quant.portfolio.ticks.CarryInputs` become the
:class:`CarryReport` the CLI (T6a), the exactness proof (T6b) and the research
evaluator (T8) consume.

Three rules shape the file, all from §4:

* **Equity is the cash form** -- :func:`~vibe_quant.portfolio.ledger.equity_at`,
  so every sum here is ``math.fsum`` and only events with ``ts < t`` count; a
  leg held at a boundary is marked at ``float(Decimal(steps) * step)`` (R47).
* **Per-year price P&L is mark-to-market** (R27b, the B1 fix): a fill's price P&L
  is booked in its own year and the two Jan-1 boundary marks make the years
  telescope to the run total (R27).
* **Turnover is grouped per decision** -- ``fsum(abs(q*px) for the decision's
  fills) / E_D`` -- so the R25 pin ``0.9948494453248812`` uses one division. The
  per-year buckets (R27c) hold the same per-fill terms but key on the FILL's
  timestamp: a 23:59 decision fills one tick later, in the next year.

``E(B) <= 0`` fails closed (R22) and a held symbol without a boundary close
raises :class:`CarryDataError` (through :func:`equity_at`).
"""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from vibe_quant.portfolio.ledger import equity_at, steps_held
from vibe_quant.portfolio.spec import (
    DAY_NS,
    CarryDataError,
    CarryEquityTooSmall,
    CarryExecutionError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.ticks import CarryInputs

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NS_PER_S = 10**9
_DAYS_PER_YEAR = 365.0
_WEEKS_PER_YEAR = 52.0
_MONDAY = 4  # day 0 (1970-01-01) was a Thursday, so Monday is day % 7 == 4
# R50: a return series this flat carries no information. Equal returns computed through
# the equity path differ by last-bit noise (stdev ~1e-16), so `stdev == 0.0` is not enough.
_STDEV_FLOOR = 1e-12
_MIN_PERIODS = 3  # R50b: the same minimum for the weekly, daily and per-year Sharpe
# R3a/R24: the run's P&L decomposition must equal E(end) - E0 within this relative
# tolerance (design §4: the per-year identity uses ``1e-9 * E0``).
_COST_IDENTITY_TOL = 1e-9


@dataclass(frozen=True)
class CarryReport:
    """The metrics of one carry run (§8-T4)."""

    summary: dict[str, Any]
    daily: list[tuple[int, float, float]]  # (B_{i+1} ns, E(B_{i+1}), r_i)
    weekly: list[tuple[int, float]]  # (M_{j+1} ns, w_j)
    legs: list[dict[str, Any]]  # symbol, side, price_pnl, funding_paid, fees, turnover
    by_year: list[dict[str, Any]]  # year, y0_ns, y1_ns, equity_*, price_pnl, ..., dispersion


def _jan1_ns(year: int) -> int:
    """00:00 UTC of Jan 1 of ``year``, in epoch ns."""
    delta = datetime(year, 1, 1, tzinfo=UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * _NS_PER_S


def _year_of(ts_ns: int) -> int:
    return datetime.fromtimestamp(ts_ns // _NS_PER_S, UTC).year


def _is_monday(ts_ns: int) -> bool:
    return (ts_ns // DAY_NS) % 7 == _MONDAY


def _mean_stdev(values: Sequence[float]) -> tuple[float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _annualised_sharpe(returns: Sequence[float], periods: float) -> float | None:
    """``mean/stdev(ddof=1) * sqrt(periods)``; null below ``_MIN_PERIODS`` or a flat series.

    R50/R50b: the same rule serves the weekly, daily and per-year weekly Sharpe -- fewer
    than three returns, or a stdev at/below ``1e-12 * max(1, |mean|)``, is no information.
    """
    if len(returns) < _MIN_PERIODS:
        return None
    mean, stdev = _mean_stdev(returns)
    if stdev <= _STDEV_FLOOR * max(1.0, abs(mean)):
        return None
    return mean / stdev * math.sqrt(periods)


def _weekly_sharpe(returns: Sequence[float]) -> float | None:
    """R23 headline: annualised weekly Sharpe over Monday-to-Monday returns."""
    return _annualised_sharpe(returns, _WEEKS_PER_YEAR)


def _lo_se(sharpe: float | None, weeks: int) -> float | None:
    """R23: ``sqrt((1 + SR^2/2) / years)`` with years = weeks/52."""
    if sharpe is None:
        return None
    return math.sqrt((1.0 + sharpe * sharpe / 2.0) / (weeks / _WEEKS_PER_YEAR))


# CF1/T8: the draw's ruin scan (R22) must cover every daily boundary but must not pay the
# exact `equity_at` walk on all of them. A naive float64 prefix sum of n terms is off by at
# most 2 * n * eps * gross (gross = the sum of the terms' magnitudes), so `_RUIN_ERROR *
# gross` leaves ~200x headroom for n <= 1e4; only boundaries inside `_ruin_band` of 0 are
# recomputed exactly and judged by the same ``not (E > 0.0)`` rule.
_RUIN_BAND = 1e-6  # relative to max(1, |E0|), six orders above that error
_RUIN_ERROR = 1e-9  # gross-relative slack covering the naive prefix error


def _ruin_band(initial_balance: float, gross: float) -> float:
    """The half-width around 0 the approximate path refuses to decide on its own (CF1)."""
    return _RUIN_BAND * max(1.0, abs(initial_balance)) + _RUIN_ERROR * gross


def _running_sums(pairs: Sequence[tuple[int, float]]) -> tuple[list[int], list[float]]:
    """``(times, prefix)`` where ``prefix[i]`` sums every term with ``ts <= times[i]``."""
    ordered = sorted(pairs)
    return [ts for ts, _ in ordered], list(itertools.accumulate(term for _, term in ordered))


def _summed_before(times: Sequence[int], prefix: Sequence[float], at_ns: int) -> float:
    """The running sum over every term strictly before ``at_ns`` (``0.0`` when there is none)."""
    index = bisect.bisect_left(times, at_ns)
    return prefix[index - 1] if index else 0.0


def _approx_equity_path(
    run: CarryRun, inputs: CarryInputs, boundaries: Sequence[int]
) -> tuple[list[float], float]:
    """A naive float64 equity path at every boundary, plus the ``gross`` that sizes the band.

    A stand-in for :func:`_boundary_equity` used only to *filter* the R22 ruin scan: the
    cash and funding prefixes are plain running sums and the marks use the same
    ``float(Decimal(steps) * step) * price``, so the value is within ``2 * n * eps *
    gross`` of the exact one. A held symbol without a close makes its boundary ``-inf`` so
    the exact call decides it (and raises :class:`CarryDataError`), and fills that imply
    different size increments raise :class:`CarryExecutionError`, as :func:`equity_at` does.
    """
    cash = [(fill.ts, -fill.qty * fill.price - fill.fee) for fill in run.fills]
    funding = [(payment.ts, payment.amount) for payment in run.payments]
    cash_times, cash_prefix = _running_sums(cash)
    funding_times, funding_prefix = _running_sums(funding)
    gross = math.fsum(abs(term) for _, term in itertools.chain(cash, funding))
    positions: list[tuple[list[int], list[int], Decimal | None, Mapping[int, float]]] = []
    for symbol in run.spec.symbols:
        fills = sorted(
            (fill for fill in run.fills if fill.symbol == symbol), key=lambda fill: fill.ts
        )
        step: Decimal | None = None
        for fill in fills:
            if not fill.steps:
                continue
            increment = Decimal(repr(fill.qty)) / Decimal(fill.steps)
            if step is not None and step != increment:
                raise CarryExecutionError(
                    f"inconsistent size increment for {symbol}: {step} vs {increment}"
                )
            step = increment
        positions.append(
            (
                [fill.ts for fill in fills],
                list(itertools.accumulate(fill.steps for fill in fills)),
                step,
                inputs.boundary_prices.get(symbol, {}),
            )
        )
    values: list[float] = []
    mark_peak = 0.0
    for boundary in boundaries:
        equity = (
            run.spec.initial_balance
            + _summed_before(cash_times, cash_prefix, boundary)
            - _summed_before(funding_times, funding_prefix, boundary)
        )
        marked = 0.0
        for times, counts, step, prices in positions:
            index = bisect.bisect_left(times, boundary)
            held = counts[index - 1] if index else 0
            if held == 0:
                continue
            price = prices.get(boundary)
            if price is None or step is None:
                equity = float("-inf")
                break
            mark = float(Decimal(held) * step) * price
            marked = math.fsum([marked, mark])
            equity += mark
        mark_peak = max(mark_peak, abs(marked))
        values.append(equity)
    return values, gross + mark_peak


def _require_daily_ruin_scan(
    run: CarryRun, inputs: CarryInputs, boundaries: Sequence[int]
) -> None:
    """R22/CF1: raise unless every daily boundary has ``E(B) > 0``, as ``carry_metrics`` does.

    Runs :func:`_approx_equity_path` once, keeps the days it clears by more than
    :func:`_ruin_band`, and recomputes the exact equity for the rest -- the same
    :func:`_boundary_equity` and the same ``not (equity > 0.0)`` rule as the daily walk.
    """
    values, gross = _approx_equity_path(run, inputs, boundaries)
    band = _ruin_band(run.spec.initial_balance, gross)
    for boundary, approx in zip(boundaries, values, strict=True):
        if approx > band:
            continue
        equity = _boundary_equity(run, inputs, boundary)
        if not (equity > 0.0):
            raise CarryEquityTooSmall("<book>", boundary, equity, 0.0)


def weekly_sharpe_only(run: CarryRun, inputs: CarryInputs) -> float | None:
    """``carry_metrics(run, inputs).summary["weekly_sharpe"]``, Monday boundaries only.

    A research draw (§6/T8) needs this one number, not the daily equity path, the
    drawdown or the P&L decomposition. The weekly returns come from the Monday boundaries
    alone -- ``days // 7 + 1`` exact :func:`_boundary_equity` calls (226 for the pinned
    4-coin book) instead of ``days + 1`` (1,581) -- combined exactly as
    :func:`carry_metrics` combines them (``E(M_{j+1}) / E(M_j) - 1.0``, then
    :func:`_weekly_sharpe`), so the value equals the report's headline bit for bit.

    Ruin (R22) still covers EVERY daily boundary (CF1): :func:`_require_daily_ruin_scan`
    clears the calm days with its float64 path and recomputes the exact equity wherever a
    boundary could be ``<= 0``, raising the same :class:`CarryEquityTooSmall` -- so a ruined
    draw stays ``-inf`` and counted exactly as before (R36).

    Raises:
        CarryEquityTooSmall: if ``E(B) <= 0`` at any daily boundary (R22).
        CarryDataError: if a held symbol has no close in ``(B-60 min, B]`` (R22).
        CarryExecutionError: if one symbol's fills imply different size increments -- raised
            up front by the ruin scan's :func:`_approx_equity_path`, and unreachable from
            ``simulate_reference``, which never emits a mixed step for one symbol.
    """
    spec = run.spec
    days = (spec.end_ns - spec.start_ns) // DAY_NS
    boundaries = [spec.start_ns + day * DAY_NS for day in range(days + 1)]
    _require_daily_ruin_scan(run, inputs, boundaries)
    values = [
        _boundary_equity(run, inputs, boundary) for boundary in boundaries if _is_monday(boundary)
    ]
    returns = [values[index + 1] / values[index] - 1.0 for index in range(len(values) - 1)]
    return _weekly_sharpe(returns)


def _open_mark(run: CarryRun, inputs: CarryInputs, at_ns: int) -> float:
    """``fsum_s q_s(t) * p_s(t)`` for the legs held at ``at_ns`` (R47, fail closed)."""
    terms: list[float] = []
    for symbol in run.spec.symbols:
        steps = steps_held(run.fills, symbol, at_ns)
        if steps == 0:
            continue
        price = inputs.boundary_prices.get(symbol, {}).get(at_ns)
        if price is None:
            raise CarryDataError(f"no price for held {symbol} at {at_ns}")
        terms.append(float(Decimal(steps) * inputs.specs[symbol].step) * price)
    return math.fsum(terms)


def _boundary_equity(run: CarryRun, inputs: CarryInputs, at_ns: int) -> float:
    """Cash-form equity at a daily boundary; a held symbol without a close fails closed."""
    prices = {
        symbol: inputs.boundary_prices[symbol][at_ns]
        for symbol in run.spec.symbols
        if at_ns in inputs.boundary_prices.get(symbol, {})
    }
    return equity_at(run.spec.initial_balance, run.fills, run.payments, prices, at_ns)


def _side(steps: int, prior_steps: int) -> str:
    """Side a fill belongs to: the position it reduces, else the one it builds."""
    if prior_steps > 0 and steps < 0:
        return "long"
    if prior_steps < 0 and steps > 0:
        return "short"
    return "long" if steps > 0 else "short"


def _decision_index(times: Sequence[int], ts_ns: int, symbol: str) -> int:
    """Index of the decision that released a fill, fail closed when none precedes it."""
    index = bisect.bisect_right(times, ts_ns) - 1
    if index < 0:
        raise CarryExecutionError(f"no decision precedes the fill of {symbol} at {ts_ns}")
    return index


def _legs(run: CarryRun, inputs: CarryInputs) -> list[dict[str, Any]]:
    """Per ``(symbol, side)`` price/funding/fee/turnover over the whole run (§4)."""
    times = [decision.ts for decision in run.decisions]
    equities = {decision.ts: decision.equity for decision in run.decisions}
    price: dict[tuple[str, str], list[float]] = {}
    fees: dict[tuple[str, str], list[float]] = {}
    funding: dict[tuple[str, str], list[float]] = {}
    turnover: dict[tuple[str, str], list[float]] = {}
    steps: dict[str, int] = {}
    for index in sorted(range(len(run.fills)), key=lambda position: run.fills[position].ts):
        fill = run.fills[index]
        prior = steps.get(fill.symbol, 0)
        key = (fill.symbol, _side(fill.steps, prior))
        price.setdefault(key, []).append(-fill.qty * fill.price)
        fees.setdefault(key, []).append(fill.fee)
        turnover.setdefault(key, []).append(
            abs(fill.qty * fill.price) / equities[times[_decision_index(times, fill.ts, fill.symbol)]]
        )
        steps[fill.symbol] = prior + fill.steps
    for payment in run.payments:
        prior = steps_held(run.fills, payment.symbol, payment.ts)
        if prior == 0:
            # accrue_book only charges a held leg; a zero-position settlement means the
            # run is corrupt and dropping it would hide money from the decomposition.
            raise CarryExecutionError(
                f"funding settlement of {payment.symbol} at {payment.ts} with no position"
            )
        funding.setdefault((payment.symbol, "long" if prior > 0 else "short"), []).append(
            payment.amount
        )
    for symbol in run.spec.symbols:
        held = steps_held(run.fills, symbol, run.spec.end_ns)
        if held == 0:
            continue
        mark = inputs.boundary_prices.get(symbol, {}).get(run.spec.end_ns)
        if mark is None:
            raise CarryDataError(f"no price for held {symbol} at {run.spec.end_ns}")
        key = (symbol, "long" if held > 0 else "short")
        price.setdefault(key, []).append(
            float(Decimal(held) * inputs.specs[symbol].step) * mark
        )
    keys = sorted(set(price) | set(fees) | set(funding) | set(turnover))
    return [
        {
            "symbol": symbol,
            "side": side,
            "price_pnl": math.fsum(price.get((symbol, side), [])),
            "funding_paid": math.fsum(funding.get((symbol, side), [])),
            "fees": math.fsum(fees.get((symbol, side), [])),
            "turnover": math.fsum(turnover.get((symbol, side), [])),
        }
        for symbol, side in keys
    ]


def _turnover_by_decision(run: CarryRun) -> dict[int, float]:
    """``fsum(abs(q*px) of a decision's fills) / E_D`` per decision (§4).

    ``turnover_total`` sums this grouping -- ``(4998 + 5046) / 10096`` -- so the R25
    pin ``0.9948494453248812`` uses ONE division; dividing per fill gives ``...811``.
    """
    times = [decision.ts for decision in run.decisions]
    equities = {decision.ts: decision.equity for decision in run.decisions}
    notional: dict[int, list[float]] = {}
    for fill in run.fills:
        ts = times[_decision_index(times, fill.ts, fill.symbol)]
        notional.setdefault(ts, []).append(abs(fill.qty * fill.price))
    return {ts: math.fsum(values) / equities[ts] for ts, values in notional.items()}


def _turnover_by_fill(run: CarryRun) -> list[tuple[int, float]]:
    """``(fill.ts, abs(q*px) / E_D)`` per fill, normalised by its decision's equity (§4).

    A decision is released one tick before its fills, so a 23:59 decision can fill
    after the year boundary: the per-year buckets (R27c) key on the fill, and a
    decision's fills split across the boundary keep their shared ``E_D``.
    """
    times = [decision.ts for decision in run.decisions]
    equities = {decision.ts: decision.equity for decision in run.decisions}
    return [
        (
            fill.ts,
            abs(fill.qty * fill.price)
            / equities[times[_decision_index(times, fill.ts, fill.symbol)]],
        )
        for fill in run.fills
    ]


def _by_year(
    run: CarryRun,
    inputs: CarryInputs,
    equities: dict[int, float],
    weekly: Sequence[tuple[int, int, float]],
    turnover: Sequence[tuple[int, float]],
) -> list[dict[str, Any]]:
    """Per calendar year (R27b, R27c): MTM price P&L, funding, fees, Sharpe, dispersion."""
    rows: list[dict[str, Any]] = []
    for year in range(_year_of(run.spec.start_ns), _year_of(run.spec.end_ns) + 1):
        y0 = max(_jan1_ns(year), run.spec.start_ns)
        y1 = min(_jan1_ns(year + 1), run.spec.end_ns)
        if y1 <= y0:
            continue
        fills = [fill for fill in run.fills if y0 <= fill.ts < y1]
        fees = math.fsum(fill.fee for fill in fills)
        funding = math.fsum(payment.amount for payment in run.payments if y0 <= payment.ts < y1)
        price = (
            math.fsum(-fill.qty * fill.price for fill in fills)
            + _open_mark(run, inputs, y1)
            - _open_mark(run, inputs, y0)
        )
        returns = [move for start, _, move in weekly if y0 <= start < y1]
        spreads = [
            max(values) - min(values)
            for decision in run.decisions
            if y0 <= decision.ts < y1
            for values in [[value for value in decision.signals.values() if value is not None]]
            if len(values) >= 2
        ]
        rows.append(
            {
                "year": year,
                "y0_ns": y0,
                "y1_ns": y1,
                "equity_start": equities[y0],
                "equity_end": equities[y1],
                "price_pnl": price,
                "funding_paid": funding,
                "fees": fees,
                "weekly_sharpe": _weekly_sharpe(returns),
                "turnover": math.fsum(
                    value for ts, value in turnover if y0 <= ts < y1
                ),
                "fees_pct_equity": fees / equities[y0],
                "funding_dispersion_ann": (
                    _WEEKS_PER_YEAR * math.fsum(spreads) / len(spreads) if spreads else None
                ),
            }
        )
    return rows


def _require_cost_identity(
    run: CarryRun, initial_balance: float, summary: dict[str, Any], equity_end: float
) -> None:
    """R3a/R24: fail closed unless ``price - funding - fees == E(end) - E0``.

    The decomposition (``_legs``) and the equity path (``equity_at``) are computed
    independently, so nothing else ties them together: a silent accounting bug
    would leave the headline numbers internally inconsistent. The tolerance is
    relative to E0, as the per-year identity of design §4 uses (``1e-9 * E0``).
    """
    identity = math.fsum(
        (summary["price_pnl_total"], -summary["funding_paid_total"], -summary["fees_total"])
    )
    equity_delta = equity_end - initial_balance
    tolerance = _COST_IDENTITY_TOL * max(1.0, abs(initial_balance))
    if abs(identity - equity_delta) > tolerance:
        raise CarryExecutionError(
            f"cost identity violated for the {run.engine} run: "
            f"price-fees-funding {identity!r} != E(end) - E0 {equity_delta!r}"
        )


def carry_metrics(run: CarryRun, inputs: CarryInputs) -> CarryReport:
    """Daily/weekly equity, headline ratios and the P&L decomposition of one run.

    Raises:
        CarryEquityTooSmall: if ``E(B) <= 0`` at any daily boundary (R22).
        CarryDataError: if a held symbol has no close in ``(B-60 min, B]`` (R22).
        CarryExecutionError: if a fill precedes every decision (no ``E_D``).
    """
    spec = run.spec
    days = (spec.end_ns - spec.start_ns) // DAY_NS
    boundaries = [spec.start_ns + day * DAY_NS for day in range(days + 1)]
    values = [_boundary_equity(run, inputs, boundary) for boundary in boundaries]
    for boundary, equity in zip(boundaries, values, strict=True):
        if not (equity > 0.0):
            raise CarryEquityTooSmall("<book>", boundary, equity, 0.0)
    equities = dict(zip(boundaries, values, strict=True))
    daily = [
        (boundaries[i + 1], values[i + 1], values[i + 1] / values[i] - 1.0) for i in range(days)
    ]
    monday = [i for i, boundary in enumerate(boundaries) if _is_monday(boundary)]
    weekly: list[tuple[int, float]] = []
    weekly_detail: list[tuple[int, int, float]] = []
    for left, right in pairwise(monday):
        move = values[right] / values[left] - 1.0
        weekly.append((boundaries[right], move))
        weekly_detail.append((boundaries[left], boundaries[right], move))
    returns = [move for _, move in weekly]
    weekly_sharpe = _weekly_sharpe(returns)
    peak = values[0]
    drawdown = 0.0
    for equity in values:
        peak = max(peak, equity)
        drawdown = max(drawdown, 1.0 - equity / peak)
    legs = _legs(run, inputs)
    turnover = _turnover_by_decision(run)
    turnover_total = math.fsum(turnover.values())
    fallback = [payment for payment in run.payments if payment.fallback]
    drifts = [decision.rounding_drift for decision in run.decisions]
    summary: dict[str, Any] = {
        "weekly_sharpe": weekly_sharpe,
        "weekly_sharpe_lo_se": _lo_se(weekly_sharpe, len(returns)),
        "weeks": len(weekly),
        "daily_sharpe_365": _annualised_sharpe([row[2] for row in daily], _DAYS_PER_YEAR),
        "total_return": values[-1] / spec.initial_balance - 1.0,
        "max_drawdown_daily": drawdown,
        "n_fills": len(run.fills),
        "n_decisions": len(run.decisions),
        "flat_decisions": sum(
            1 for decision in run.decisions if all(w == 0.0 for w in decision.weights.values())
        ),
        "fees_total": math.fsum(fill.fee for fill in run.fills),
        "funding_paid_total": math.fsum(payment.amount for payment in run.payments),
        "funding_fallback_settlements": len(fallback),
        "funding_estimated": bool(fallback),
        "funding_fallback_amount": math.fsum(payment.amount for payment in fallback),
        "price_pnl_total": math.fsum(leg["price_pnl"] for leg in legs),
        "turnover_total": turnover_total,
        "turnover_annual": turnover_total / ((spec.end_ns - spec.start_ns) / (_DAYS_PER_YEAR * DAY_NS)),
        "rounding_drift_mean": math.fsum(drifts) / len(drifts) if drifts else None,
        "rounding_drift_max": max(drifts) if drifts else None,
        "band_skips": sum(decision.band_skips for decision in run.decisions),
        "min_notional_skips": sum(decision.min_notional_skips for decision in run.decisions),
        "liquidation": "not_modelled",
    }
    _require_cost_identity(run, spec.initial_balance, summary, values[-1])
    return CarryReport(
        summary=summary,
        daily=daily,
        weekly=weekly,
        legs=legs,
        by_year=_by_year(run, inputs, equities, weekly_detail, _turnover_by_fill(run)),
    )
