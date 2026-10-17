"""Research evaluator for the funding-carry book (design §6, §8-T8).

Runs the no-NT reference over one :class:`~vibe_quant.portfolio.ticks.CarryInputs`
and reports the robustness numbers an edge claim needs:

* **Persistence-matched random-rank null** (§6, SF4): the strategy's own pick
  persistence ``p_keep`` drives a seeded null whose weekly Sharpe gives the
  p-value ``(1 + #{null Sharpe >= strategy Sharpe}) / (1 + draws)`` -- post-cost
  and pre-cost (``fee_multiplier=0.0`` for the strategy AND every draw). A draw
  that raises :class:`~vibe_quant.portfolio.spec.CarryEquityTooSmall` is
  *ruined*: Sharpe ``-inf``, counted in ``ruined_draws``, never dropped (R36).
* **Placebo rank** (R59): the same ranker hook, fed the trailing 7-day price
  return. ``rank_weights`` longs the LOWEST signal, so this is a one-sided
  *reversal* placebo (long the weekly losers, short the winners), reported as
  ``placebo_kind="reversal"``; it is not a momentum placebo.
* **Beta / alpha** (R56): two separate single-factor OLS regressions of the
  weekly returns -- on the BTC weekly return and on the equal-weight universe
  basket return -- with plain OLS standard errors (``n - 2`` dof).
* **Stress**: 2x fees (``fee_multiplier=2.0``), funding capped to +-0.1% per
  settlement through :func:`capped_inputs` (signal AND paid funding, R37), and
  the drop-top-3-weeks Sharpe.
* **DSR** (R38/R41/R57): the deflated Sharpe with ``N = N_prior + N_variants``,
  ``N_prior = 3`` (the H-11 / H-11-66 / H-11F4 journal trials) and
  ``N_variants`` = distinct ``spec_hash`` values in the trial ledger plus the
  spec being evaluated. The headline ``dsr`` / ``dsr_p_value`` go through the
  discovery gate's own helper (``mtm_dsr_calculate``: weekly per-period Sharpe,
  skew clamped <= 0, kurtosis clamped >= 3, ``T`` = weeks) and fail closed to
  ``None`` + ``dsr_error``; ``dsr_raw_p_value`` is the same call on the raw
  moments. ``evaluate`` NEVER appends to the ledger.

This module re-implements no money rule: it drives ``reference.simulate_reference``
(the hooks default to a byte-identical run, R35) and ``metrics.carry_metrics``,
so its numbers are only as wrong as those two are.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    calculate_dsr,
    daily_sharpe_inputs,
    mtm_dsr_calculate,
)
from vibe_quant.portfolio.metrics import (
    _WEEKS_PER_YEAR,
    _annualised_sharpe,
    carry_metrics,
    weekly_sharpe_only,
)
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.signal import Settlements, carry_hours
from vibe_quant.portfolio.spec import DAY_NS, CarryEquityTooSmall
from vibe_quant.portfolio.ticks import sizing_price
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from vibe_quant.portfolio.metrics import CarryReport
    from vibe_quant.portfolio.sizing import Decision
    from vibe_quant.portfolio.ticks import CarryInputs, PriceTick

# The three journal trials of `docs/discovery-journal.md` that already spent
# DSR budget on this idea (R38): RESULT H-11 (189 coins) and its 66-coin
# secondary (2026-10-11), and PRE-REGISTRATION H-11F4 (4 coins, 2026-10-12).
# H-11F (29 coins) was withdrawn before any evaluation -> not a trial.
N_PRIOR_TRIALS = 3
# ±0.1% per settlement (design §6): Binance's hard funding-rate clamp.
DEFAULT_FUNDING_CAP = 0.001
_WEEK_NS = 7 * DAY_NS
PLACEBO_KIND = "reversal"  # R59: long the lowest trailing return


class Ranker(Protocol):
    """Replacement signal values for one decision (§6).

    Returns a signal value per symbol (rank positions as floats when hunting a
    null), so :func:`~vibe_quant.portfolio.signal.rank_weights` stays the one
    ranking implementation. ``None`` keeps a symbol excluded.
    """

    def __call__(
        self, decision_ns: int, signals: Mapping[str, float | None]
    ) -> dict[str, float | None]: ...


class PersistenceMatchedRanker:
    """A seeded random-rank null that repeats a pick with probability ``p_keep`` (§6).

    At each decision the rankable set (non-``None`` signals) is sorted and, with
    probability ``p_keep`` when the same set was ranked before, the previous
    ordering is kept; otherwise it is shuffled. Symbols are handed back as rank
    positions (distinct floats), so the ranking itself stays in ``rank_weights``.
    """

    def __init__(self, p_keep: float, seed: int) -> None:
        self.p_keep = p_keep
        self._random = random.Random(seed)
        self._previous: dict[str, float] | None = None
        self._previous_symbols: frozenset[str] | None = None

    def __call__(
        self, decision_ns: int, signals: Mapping[str, float | None]
    ) -> dict[str, float | None]:
        rankable = sorted(symbol for symbol, value in signals.items() if value is not None)
        result: dict[str, float | None] = dict(signals)
        if not rankable:
            self._previous = None
            self._previous_symbols = None
            return result
        symbols = frozenset(rankable)
        if (
            self._previous is not None
            and self._previous_symbols == symbols
            and self._random.random() < self.p_keep
        ):
            ordering = dict(self._previous)
        else:
            shuffled = list(rankable)
            self._random.shuffle(shuffled)
            ordering = {symbol: float(position) for position, symbol in enumerate(shuffled)}
        self._previous = ordering
        self._previous_symbols = symbols
        result.update(ordering)
        return result


class _PlaceboRanker:
    """Signal = trailing price return between consecutive sizing ticks (§6, R59).

    The book goes long the lowest signal and short the highest, so this ranks as a
    one-sided REVERSAL: long last week's losers, short its winners.
    """

    def __init__(self, ticks: Mapping[str, Sequence[PriceTick]]) -> None:
        self._ticks = ticks
        self._previous: dict[str, float] | None = None

    def __call__(
        self, decision_ns: int, signals: Mapping[str, float | None]
    ) -> dict[str, float | None]:
        current: dict[str, float] = {}
        for symbol in signals:
            tick = sizing_price(self._ticks[symbol], decision_ns)
            if tick is not None:
                current[symbol] = tick.price
        previous = self._previous
        result: dict[str, float | None] = {}
        for symbol, value in signals.items():
            before = None if previous is None else previous.get(symbol)
            if value is None or symbol not in current or before is None or before == 0.0:
                result[symbol] = None
            else:
                result[symbol] = current[symbol] / before - 1.0
        self._previous = current
        return result


class CappedFundingCalculator(FundingCalculator):
    """A funding calculator whose rates are clipped to +-``cap`` per settlement (R37).

    ``_load_rates`` returns CLIPPED COPIES of the base calculator's series, so
    the process-level ``_RATE_CACHE`` keeps the true rates and a robustness run
    never contaminates the next one.
    """

    def __init__(self, base: FundingCalculator, cap: float) -> None:
        super().__init__(base._archive_path, base.default_rate)
        self._base = base
        self._cap = cap

    def _load_rates(self, symbol: str) -> tuple[list[int], list[float]]:
        times, rates = self._base._load_rates(symbol)
        return list(times), [min(max(rate, -self._cap), self._cap) for rate in rates]

    def _load_intervals(self, symbol: str) -> list[int]:
        """The base's intervals: they are not rates, so the cap cannot touch them.

        Delegating (rather than re-resolving from the copied times) keeps the
        base calculator's identity-keyed interval cache hit.
        """
        return self._base._load_intervals(symbol)


def capped_inputs(inputs: CarryInputs, cap: float) -> CarryInputs:
    """``inputs`` with a capped funding calculator AND capped signal settlements (R37).

    Replacing ``settlements`` too is what makes the cap bite the SIGNAL as well as
    the paid funding (``FundingAccumulator`` reads ``calc``); the module rate cache
    is never written.
    """
    capped = CappedFundingCalculator(inputs.calc, cap)
    return replace(
        inputs,
        calc=capped,
        settlements={
            symbol: Settlements(
                *capped.settlements(symbol),
                hours=carry_hours(capped.settlement_hours(symbol)),
            )
            for symbol in inputs.spec.symbols
        },
    )


def _rankable(decision: Decision) -> int:
    return sum(1 for value in decision.signals.values() if value is not None)


def _pick(decision: Decision) -> tuple[frozenset[str], frozenset[str]]:
    """The decision's ``(long set, short set)``, read off its weights."""
    return (
        frozenset(symbol for symbol, weight in decision.weights.items() if weight > 0.0),
        frozenset(symbol for symbol, weight in decision.weights.items() if weight < 0.0),
    )


def _persistence(decisions: Sequence[Decision]) -> float:
    """``p_keep``: fraction of consecutive n>=2 decision pairs with an unchanged pick (§6)."""
    total = 0
    repeats = 0
    for previous, current in zip(decisions, decisions[1:], strict=False):
        if _rankable(previous) < 2 or _rankable(current) < 2:
            continue
        total += 1
        if _pick(previous) == _pick(current):
            repeats += 1
    return repeats / total if total else 0.0


def _p_value(strategy_sharpe: float | None, null_sharpes: Sequence[float]) -> float | None:
    """``(1 + #{null Sharpe >= strategy Sharpe}) / (1 + draws)`` (§6)."""
    if strategy_sharpe is None:
        return None
    at_least = sum(1 for sharpe in null_sharpes if sharpe >= strategy_sharpe)
    return (1 + at_least) / (1 + len(null_sharpes))


def _draw_sharpes(
    inputs: CarryInputs, p_keep: float, seed: int, draws: int, fee_multiplier: float
) -> tuple[list[float], int]:
    """Weekly Sharpes of ``draws`` persistence-matched null runs; ruined -> ``-inf`` (R36).

    Each draw reads only the Monday boundaries its weekly Sharpe needs
    (:func:`~vibe_quant.portfolio.metrics.weekly_sharpe_only`) instead of the full
    :func:`~vibe_quant.portfolio.metrics.carry_metrics` report -- 226 boundary
    equities instead of 1,581 on the pinned 4-coin book -- and gets the report's
    ``weekly_sharpe`` bit for bit. A null draw therefore never builds the §4 report:
    :func:`~vibe_quant.portfolio.metrics._legs`,
    :func:`~vibe_quant.portfolio.metrics._turnover_by_decision` and
    :func:`~vibe_quant.portfolio.metrics._require_cost_identity` do not run for it.
    """
    sharpes: list[float] = []
    ruined = 0
    for index in range(draws):
        ranker = PersistenceMatchedRanker(p_keep, seed + index)
        try:
            sharpe = weekly_sharpe_only(
                simulate_reference(inputs, ranker=ranker, fee_multiplier=fee_multiplier), inputs
            )
        except CarryEquityTooSmall:
            ruined += 1
            sharpes.append(float("-inf"))
            continue
        sharpes.append(float("-inf") if sharpe is None else sharpe)
    return sharpes, ruined


def _market_returns(
    inputs: CarryInputs, symbol: str, weekly: Sequence[tuple[int, float]]
) -> list[float | None]:
    """The symbol's weekly returns at each report boundary (``None`` where a price is missing)."""
    prices = inputs.boundary_prices.get(symbol)
    returns: list[float | None] = []
    for ts, _ in weekly:
        before = ts - _WEEK_NS
        if prices is None or before not in prices or ts not in prices or prices[before] == 0.0:
            returns.append(None)
        else:
            returns.append(prices[ts] / prices[before] - 1.0)
    return returns


def _ew_returns(
    inputs: CarryInputs, weekly: Sequence[tuple[int, float]]
) -> list[float | None]:
    """The equal-weight universe basket's weekly returns (mean of the per-symbol returns)."""
    columns = [_market_returns(inputs, symbol, weekly) for symbol in inputs.spec.symbols]
    basket: list[float | None] = []
    for index in range(len(weekly)):
        row: list[float] = []
        for column in columns:
            value = column[index]
            if value is not None:
                row.append(value)
        basket.append(math.fsum(row) / len(row) if row else None)
    return basket


def _ols(returns: Sequence[float], factor: Sequence[float]) -> tuple[float | None, float | None]:
    """Single-factor OLS: ``(beta, alpha t-stat)``; ``None`` with < 3 points or no spread (R56)."""
    n = len(returns)
    if n < 3 or n != len(factor):
        return None, None
    mean_return = math.fsum(returns) / n
    mean_factor = math.fsum(factor) / n
    sxx = math.fsum((value - mean_factor) ** 2 for value in factor)
    if sxx <= 0.0:
        return None, None
    sxy = math.fsum(
        (value - mean_factor) * (move - mean_return)
        for value, move in zip(factor, returns, strict=True)
    )
    beta = sxy / sxx
    alpha = mean_return - beta * mean_factor
    residuals = [
        move - alpha - beta * value for value, move in zip(factor, returns, strict=True)
    ]
    rss = math.fsum(residual * residual for residual in residuals)
    residual_variance = rss / (n - 2)
    if residual_variance <= 0.0:
        return beta, None  # an exact line has no residual spread -> no t-stat
    se_alpha = math.sqrt(residual_variance * (1.0 / n + mean_factor * mean_factor / sxx))
    return beta, alpha / se_alpha if se_alpha > 0.0 else None


def _ols_pairs(
    returns: Sequence[float], factor: Sequence[float | None]
) -> tuple[float | None, float | None]:
    """``_ols`` over the aligned pairs where the factor is present."""
    pairs = [
        (move, value)
        for move, value in zip(returns, factor, strict=True)
        if value is not None
    ]
    if len(pairs) < 3:
        return None, None
    return _ols([move for move, _ in pairs], [value for _, value in pairs])


def _drop_top3(returns: Sequence[float]) -> float | None:
    """The weekly Sharpe with the three best weeks removed (§6 stress)."""
    if len(returns) <= 3:
        return None
    return _annualised_sharpe(sorted(returns)[:-3], _WEEKS_PER_YEAR)


def _trial_counts(trials_path: Path, spec_hash: str) -> tuple[int, float | None]:
    """``(N, trials_sharpe_variance)`` from the ledger and the evaluated spec (R38/R41).

    ``N = N_prior + |recorded spec hashes u {spec_hash}|``; the variance is over
    the recorded variants' weekly Sharpes in PER-PERIOD units, and is ``None``
    with fewer than two recorded variants.
    """
    recorded: dict[str, Any] = {}
    if trials_path.is_file():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            recorded[str(record["spec_hash"])] = record.get("weekly_sharpe")
    sharpes = [
        float(value) / math.sqrt(_WEEKS_PER_YEAR)
        for value in recorded.values()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    variance = statistics.variance(sharpes) if len(sharpes) >= 2 else None
    return N_PRIOR_TRIALS + len(set(recorded) | {spec_hash}), variance


def _dsr(
    weekly_returns: Sequence[float],
    weekly_sharpe: float | None,
    num_trials: int,
    trials_sharpe_variance: float | None,
) -> tuple[float | None, float | None, str | None, float | None]:
    """``(dsr, p-value, error, raw p-value)`` of the weekly Sharpe in per-period units (R57).

    The headline goes through the gate's ``mtm_dsr_calculate`` (moments clamped
    skew <= 0 / kurtosis >= 3; ``(None, reason)`` fails closed); the raw p-value is
    ``calculate_dsr`` on the unclamped moments, for the record.
    """
    weeks = len(weekly_returns)
    _, skewness, kurtosis = daily_sharpe_inputs(weekly_returns)
    per_period = None if weekly_sharpe is None else weekly_sharpe / math.sqrt(_WEEKS_PER_YEAR)
    result, reason = mtm_dsr_calculate(
        DeflatedSharpeRatio(),
        per_period,
        skewness,
        kurtosis,
        weeks,
        num_trials,
        trials_sharpe_variance,
    )
    if result is None or per_period is None:
        return None, None, reason, None
    raw = calculate_dsr(
        observed_sharpe=per_period,
        num_trials=num_trials,
        num_observations=weeks,
        skewness=skewness,
        kurtosis=kurtosis,
        trials_sharpe_variance=trials_sharpe_variance,
    )
    return result.deflated_sharpe, result.p_value, None, raw.p_value


def _betas(
    inputs: CarryInputs, report: CarryReport
) -> tuple[float | None, float | None, float | None, float | None]:
    """``(beta_btc, alpha_t_btc, beta_ew, alpha_t)`` -- two single-factor OLS (R56)."""
    weekly = report.weekly
    returns = [move for _, move in weekly]
    beta_btc, alpha_t_btc = _ols_pairs(returns, _market_returns(inputs, "BTCUSDT", weekly))
    beta_ew, alpha_t = _ols_pairs(returns, _ew_returns(inputs, weekly))
    return beta_btc, alpha_t_btc, beta_ew, alpha_t


def evaluate(
    inputs: CarryInputs, *, draws: int, seed: int, trials_path: Path
) -> dict[str, Any]:
    """The §6/§8-T8 robustness report over one preflighted ``CarryInputs``.

    Raises:
        CarryEquityTooSmall: if the strategy's own run is ruined (the caller exits 2).
    """
    run = simulate_reference(inputs)
    report = carry_metrics(run, inputs)
    strategy_sharpe = report.summary["weekly_sharpe"]
    p_keep = _persistence(run.decisions)

    null_sharpes, ruined = _draw_sharpes(inputs, p_keep, seed, draws, 1.0)
    p_value = _p_value(strategy_sharpe, null_sharpes)

    # Pre-cost: fee_multiplier=0.0 for the strategy AND every draw (§6).
    precost_report = carry_metrics(simulate_reference(inputs, fee_multiplier=0.0), inputs)
    precost_sharpes, _ = _draw_sharpes(inputs, p_keep, seed, draws, 0.0)
    p_value_pre_cost = _p_value(precost_report.summary["weekly_sharpe"], precost_sharpes)

    placebo = carry_metrics(simulate_reference(inputs, ranker=_PlaceboRanker(inputs.ticks)), inputs)
    fee2x = carry_metrics(simulate_reference(inputs, fee_multiplier=2.0), inputs)
    capped = capped_inputs(inputs, DEFAULT_FUNDING_CAP)
    capped_report = carry_metrics(simulate_reference(capped), capped)

    beta_btc, alpha_t_btc, beta_ew, alpha_t = _betas(inputs, report)
    weekly_returns = [move for _, move in report.weekly]
    num_trials, trials_sharpe_variance = _trial_counts(trials_path, inputs.spec.spec_hash())
    dsr, dsr_p_value, dsr_error, dsr_raw_p_value = _dsr(
        weekly_returns, strategy_sharpe, num_trials, trials_sharpe_variance
    )

    return {
        "p_value": p_value,
        "p_value_pre_cost": p_value_pre_cost,
        "ruined_draws": ruined,
        "draws": draws,
        "p_keep": p_keep,
        "placebo_weekly_sharpe": placebo.summary["weekly_sharpe"],
        "beta_btc": beta_btc,
        "alpha_t_btc": alpha_t_btc,
        "beta_ew": beta_ew,
        "alpha_t": alpha_t,
        "fee2x_weekly_sharpe": fee2x.summary["weekly_sharpe"],
        "capped_weekly_sharpe": capped_report.summary["weekly_sharpe"],
        "drop_top3_weekly_sharpe": _drop_top3(weekly_returns),
        "placebo_kind": PLACEBO_KIND,
        "dsr": dsr,
        "dsr_p_value": dsr_p_value,
        "dsr_error": dsr_error,
        "dsr_raw_p_value": dsr_raw_p_value,
        "dsr_num_trials": num_trials,
        "trials_sharpe_variance": trials_sharpe_variance,
        "by_year": report.by_year,
    }
