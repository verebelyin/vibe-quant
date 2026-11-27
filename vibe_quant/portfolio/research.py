"""Research evaluator for the funding-carry book (design §6, §8-T8).

Runs the no-NT reference over one :class:`~vibe_quant.portfolio.ticks.CarryInputs`
and reports the robustness numbers an edge claim needs:

* **Persistence-matched random-rank null** (§6, SF4): the strategy's own pick
  persistence ``p_keep`` drives a seeded null whose weekly Sharpe gives the
  p-value ``(1 + #{null Sharpe >= strategy Sharpe}) / (1 + draws)`` -- post-cost
  and pre-cost (``fee_multiplier=0.0`` for the strategy AND every draw). A draw
  that raises :class:`~vibe_quant.portfolio.spec.CarryEquityTooSmall` (a cross
  ``CarryLiquidated`` bust included) is *ruined*: Sharpe ``-inf``, counted in
  ``ruined_draws``, never dropped (R36, R43).
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
  moments. The ledger and its reducer live in :mod:`vibe_quant.portfolio.trials`:
  ``evaluate`` registers its OWN spec's completion right after the own report
  (once per hash; an own run ruined after its first decision is recorded as a
  ``ruined`` trial before the error propagates), then reads one locked snapshot
  for ``N``. No auxiliary run (pre-cost, placebo, 2x fees, capped, null draws)
  registers a trial, nor does a lagged (NC1 calibration) trend evaluate.

A **trend** spec (arch-c9-2, :func:`evaluate_trend`) reports every Sharpe as
``daily_sharpe_365`` and replaces the funding pieces: N from a frozen family file
(``n_prior + len(arms)``, theoretical variance, so the DSR ignores run order, R55/R56);
a joint rotation null that shifts each common-eligibility segment's signal and regime
rows together (each coin stays inside its own mask, co-movement kept; C9-6, R57); a
cross-wired placebo with the true regime (R60); pre-cost = fees x0 and spread 0 (R59);
daily OLS betas; the executed exposure of the own run vs the null mean; the run's
evidence block (C9-8); an optional pre-registered criteria verdict (R62); and a
signal-only lag for the NC1 calibration (C9-17, :func:`nc1_verdict`). Null draws and the
placebo run the filter-free spec with :class:`TableRanker` zeroing the book while the
(rotated) regime is off -- the native ``filter_off`` book bit for bit (R-c9t5-1).

This module re-implements no money rule: it drives ``reference.simulate_reference``
(the hooks default to a byte-identical run, R35) and ``metrics.carry_metrics``,
so its numbers are only as wrong as those two are.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import random
import re
from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol

from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    calculate_dsr,
    daily_sharpe_inputs,
    mtm_dsr_calculate,
)
from vibe_quant.portfolio.metrics import (
    _DAYS_PER_YEAR,
    _WEEKS_PER_YEAR,
    _annualised_sharpe,
    _turnover_by_decision,
    carry_metrics,
    daily_sharpe_only,
    provenance,
    weekly_sharpe_only,
)
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.signal import Settlements, carry_hours, decision_signals, trend_signal
from vibe_quant.portfolio.spec import DAY_NS, CarryEquityTooSmall, family_num_trials
from vibe_quant.portfolio.ticks import sizing_price
from vibe_quant.portfolio.trials import (
    completion_line,
    failure_line,
    failure_outcome,
    git_sha,
    read_trials,
    record_completion,
    record_failure,
    trial_counts,
)
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.metrics import CarryReport
    from vibe_quant.portfolio.sizing import Decision
    from vibe_quant.portfolio.spec import CarrySpec
    from vibe_quant.portfolio.ticks import CarryInputs, PriceTick

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
    inputs: CarryInputs,
    symbol: str,
    weekly: Sequence[tuple[Any, ...]],
    period_ns: int = _WEEK_NS,
) -> list[float | None]:
    """The symbol's returns over ``period_ns`` ending at each report row's boundary
    (row[0]); ``None`` where a price is missing. Weekly by default, daily on trend."""
    prices = inputs.boundary_prices.get(symbol)
    returns: list[float | None] = []
    for row in weekly:
        ts = row[0]
        before = ts - period_ns
        if prices is None or before not in prices or ts not in prices or prices[before] == 0.0:
            returns.append(None)
        else:
            returns.append(prices[ts] / prices[before] - 1.0)
    return returns


def _ew_returns(
    inputs: CarryInputs, weekly: Sequence[tuple[Any, ...]], period_ns: int = _WEEK_NS
) -> list[float | None]:
    """The equal-weight universe basket's returns (mean of the per-symbol returns)."""
    columns = [_market_returns(inputs, symbol, weekly, period_ns) for symbol in inputs.spec.symbols]
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
    residuals = [move - alpha - beta * value for value, move in zip(factor, returns, strict=True)]
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
        (move, value) for move, value in zip(returns, factor, strict=True) if value is not None
    ]
    if len(pairs) < 3:
        return None, None
    return _ols([move for move, _ in pairs], [value for _, value in pairs])


def _drop_top3(returns: Sequence[float]) -> float | None:
    """The weekly Sharpe with the three best weeks removed (§6 stress)."""
    if len(returns) <= 3:
        return None
    return _annualised_sharpe(sorted(returns)[:-3], _WEEKS_PER_YEAR)


def _dsr(
    weekly_returns: Sequence[float],
    weekly_sharpe: float | None,
    num_trials: int,
    trials_sharpe_variance: float | None,
    periods: float = _WEEKS_PER_YEAR,
) -> tuple[float | None, float | None, str | None, float | None]:
    """``(dsr, p-value, error, raw p-value)`` of the annualised Sharpe in per-period units
    (R57): weekly by default, ``periods=365`` for the trend's daily statistic (T = days).

    The headline goes through the gate's ``mtm_dsr_calculate`` (moments clamped
    skew <= 0 / kurtosis >= 3; ``(None, reason)`` fails closed); the raw p-value is
    ``calculate_dsr`` on the unclamped moments, for the record.
    """
    weeks = len(weekly_returns)
    _, skewness, kurtosis = daily_sharpe_inputs(weekly_returns)
    per_period = None if weekly_sharpe is None else weekly_sharpe / math.sqrt(periods)
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


def _own_report(
    inputs: CarryInputs, trials_path: Path, simulate: Callable[[], CarryRun]
) -> tuple[CarryRun, CarryReport]:
    """The strategy's own run and report, registered as its trial (once per hash).

    A ruin after the first decision is recorded as a failure trial before it propagates.
    """
    spec = inputs.spec
    sha = git_sha()
    try:
        run = simulate()
        report = carry_metrics(run, inputs)
    except CarryEquityTooSmall as error:
        outcome = failure_outcome(error, inputs.decisions)
        if outcome is not None and error.decision_ns is not None:
            record_failure(
                trials_path, failure_line("reference", spec, outcome, error.decision_ns, sha)
            )
        raise
    record_completion(
        trials_path, completion_line("reference", spec, report.summary, sha), history=False
    )
    return run, report


def evaluate(
    inputs: CarryInputs,
    *,
    draws: int,
    seed: int,
    trials_path: Path,
    family: Family | None = None,
    criteria: Criteria | None = None,
    signal_lag_days: int = 0,
    spread_source: str = "unspecified",
    universe: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The §6/§8-T8 robustness report over one preflighted ``CarryInputs``.

    A trend spec goes to :func:`evaluate_trend` (the frozen family, the rotation null,
    the criteria verdict); ``family``, ``criteria`` and ``signal_lag_days`` are trend
    only. The funding report below is unchanged (R63).

    The own completion is registered in ``trials_path`` right after the own
    report (``record_completion(history=False)``: once per spec hash), so a later
    failure keeps it and a rerun finds it; ``N`` comes from one locked snapshot.

    Raises:
        CarryEquityTooSmall: if the strategy's own run is ruined (the caller exits 2);
            a ruin after the first decision is first recorded as a trial.
        CarryLedgerError: if the ledger cannot be read or appended to.
        ValueError: a trend option on a funding spec, or a trend spec without its family.
    """
    spec = inputs.spec
    if spec.signal == "trend":
        return evaluate_trend(
            inputs,
            draws=draws,
            seed=seed,
            trials_path=trials_path,
            family=family,
            criteria=criteria,
            signal_lag_days=signal_lag_days,
            spread_source=spread_source,
            universe=universe,
        )
    if family is not None or criteria is not None or signal_lag_days:
        raise ValueError("--family, --criteria and --signal-lag-days need --signal trend")
    run, report = _own_report(inputs, trials_path, lambda: simulate_reference(inputs))
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
    num_trials, trials_sharpe_variance = trial_counts(
        read_trials(trials_path), spec.spec_hash(), signal=spec.signal
    )
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


# --- the trend evaluator (arch-c9-2 §2 "Evaluator", R55-R63; dreview-b2-2 C9-6/7/8/17) ---

TREND_STATISTIC = "daily_sharpe_365"
TREND_PLACEBO_KIND = "cross_wired"  # R60: coin signals cross-wired, the true regime kept
DSR_VARIANCE = "theoretical"  # R56: variance None, so N and the DSR ignore run order
# NC1 (R-c9-8): the lagged-signal null set and its pre-registered cutoff
NC1_LAGS_DAYS = tuple(range(365, 709, 7))
NC1_PERCENTILE = 0.95
NC1_FAIL_AT = 6
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}
CRITERIA_METRICS = (
    "sharpe",
    "total_return",
    "max_drawdown_daily",
    "p_value",
    "null_percentile",
    "p_value_pre_cost",
    "beta_btc",
    "alpha_t_btc",
    "beta_ew",
    "alpha_t",
    "fee2x_sharpe",
    "capped_sharpe",
    "placebo_sharpe",
    "drop_top3_weeks_sharpe",
    "dsr",
    "dsr_p_value",
    "coins_with_data",
)


def _json_file(path: Path, what: str) -> tuple[Any, dict[str, str]]:
    """The parsed JSON of a frozen input file and its ``{path, sha256}`` identity."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{what} file {path} unreadable: {error}") from error
    try:
        payload = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"{what} file {path} is not JSON: {error}") from error
    return payload, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True)
class Family:
    """The frozen trial family (R55, C9-7): ``N = n_prior + len(arms)``, fixed before any run.

    ``arms`` are :meth:`CarrySpec.arm_hash` values, so a re-run of an arm on other
    coins, dates or tier adds no trial.
    """

    path: str
    sha256: str
    n_prior: int
    arms: tuple[str, ...]

    @property
    def num_trials(self) -> int:
        return family_num_trials(self.n_prior, len(self.arms))

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_prior": self.n_prior,
            "arms": list(self.arms),
        }


def load_family(path: Path) -> Family:
    """``{"n_prior": int >= 0, "arms": [sha256 hex, ...]}`` or ``ValueError`` (R55).

    Arms are distinct and non-empty; ``N`` must stay in the DSR quantile domain
    (:func:`~vibe_quant.portfolio.spec.family_num_trials`).
    """
    payload, identity = _json_file(path, "family")
    where = f"family file {path}"
    if not isinstance(payload, dict) or set(payload) != {"n_prior", "arms"}:
        raise ValueError(f"{where}: must be an object with exactly n_prior and arms")
    n_prior, arms = payload["n_prior"], payload["arms"]
    if isinstance(n_prior, bool) or not isinstance(n_prior, int) or n_prior < 0:
        raise ValueError(f"{where}: n_prior must be an int >= 0, got {n_prior!r}")
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{where}: arms must be a non-empty list of arm hashes")
    for arm in arms:
        if not isinstance(arm, str) or _SHA256_HEX.fullmatch(arm) is None:
            raise ValueError(f"{where}: arm {arm!r} is not a sha256 hex arm hash")
    if len(set(arms)) != len(arms):
        raise ValueError(f"{where}: duplicate arm")
    try:
        family_num_trials(n_prior, len(arms))
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    return Family(identity["path"], identity["sha256"], n_prior, tuple(arms))


def require_arm(family: Family, spec: CarrySpec) -> None:
    """Refuse a spec whose arm hash is not in the frozen family (R55)."""
    arm = spec.arm_hash()
    if arm not in family.arms:
        raise ValueError(f"arm {arm} not in frozen family {family.path}")


@dataclass(frozen=True)
class Criteria:
    """A pre-registered criteria file: ``(metric, op, bar)`` rules and its identity (R62)."""

    path: str
    sha256: str
    rules: tuple[tuple[str, str, float], ...]


def load_criteria(path: Path) -> Criteria:
    """``{metric: {"op": ">=|>|<=|<", "bar": finite number}}``, non-empty, or ``ValueError``."""
    payload, identity = _json_file(path, "criteria")
    where = f"criteria file {path}"
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{where}: must be a non-empty {{metric: {{op, bar}}}} object")
    rules: list[tuple[str, str, float]] = []
    for metric, rule in payload.items():
        if metric not in CRITERIA_METRICS:
            raise ValueError(f"{where}: unknown metric {metric!r}")
        if not isinstance(rule, dict) or set(rule) != {"op", "bar"}:
            raise ValueError(f"{where}: {metric} must be {{op, bar}}")
        op, bar = rule["op"], rule["bar"]
        if op not in _OPS:
            raise ValueError(f"{where}: unknown op {op!r} for {metric}")
        bad_bar = f"{where}: bar for {metric} must be a finite number, got {bar!r}"
        if isinstance(bar, bool) or not isinstance(bar, (int, float)):
            raise ValueError(bad_bar)
        try:
            number = float(bar)
        except OverflowError as error:
            raise ValueError(bad_bar) from error
        if not math.isfinite(number):
            raise ValueError(bad_bar)
        rules.append((metric, op, number))
    return Criteria(identity["path"], identity["sha256"], tuple(rules))


def judge(
    criteria: Criteria | None,
    values: Mapping[str, Any],
    fill_granularity: str,
    *,
    calibration: bool = False,
) -> dict[str, Any]:
    """Per criterion ``{value, op, bar, pass}`` and the verdict (R62).

    An undefined value (None, nan, inf) fails its criterion. ``PASS`` needs every
    criterion and the 1m tier; at 1h the verdict is ``research_only`` (the pass state
    stays in ``criteria_all_pass``). Without criteria nothing can PASS: ``no_criteria``.
    A ``calibration`` (lagged NC1) run is never a verdict on the arm: ``calibration``.
    """
    hour = fill_granularity == "1h"
    if criteria is None:
        return {
            "criteria": None,
            "criteria_all_pass": None,
            "verdict": "calibration"
            if calibration
            else ("research_only" if hour else "no_criteria"),
        }
    results: dict[str, dict[str, Any]] = {}
    for metric, op, bar in criteria.rules:
        value = values.get(metric)
        number = (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else math.nan
        )
        results[metric] = {
            "value": value,
            "op": op,
            "bar": bar,
            "pass": math.isfinite(number) and _OPS[op](number, bar),
        }
    all_pass = all(row["pass"] for row in results.values())
    verdict = "research_only" if hour else ("PASS" if all_pass else "FAIL")
    if calibration:
        verdict = "calibration"
    return {"criteria": results, "criteria_all_pass": all_pass, "verdict": verdict}


@dataclass(frozen=True)
class SignalTable:
    """The strategy's signal per decision and symbol (None = ineligible) and its regime."""

    decisions: tuple[int, ...]
    signals: tuple[dict[str, float | None], ...]
    regime: tuple[float | None, ...]


def lag_spec(spec: CarrySpec, lag_days: int) -> CarrySpec:
    """``spec`` starting ``lag_days`` earlier: the load whose closes a lagged signal reads.

    Its trend closes reach back to ``start - lag - 160 d`` (C9-17: 2021-01-14 for the
    708-day NC1 arm on a 2023-06-01 window).
    """
    if isinstance(lag_days, bool) or not isinstance(lag_days, int) or lag_days < 0:
        raise ValueError(f"--signal-lag-days must be >= 0, got {lag_days!r}")
    return replace(spec, start_ns=spec.start_ns - lag_days * DAY_NS)


def extend_closes(
    base: Mapping[str, Mapping[int, float]], longer: Mapping[str, Mapping[int, float]]
) -> dict[str, dict[int, float]]:
    """``longer``'s closes for ``base``'s symbols; every day ``base`` holds must match.

    The lag reads earlier days only: the prices and book-vol closes of the run itself
    stay the same values (C9-17), else ``ValueError`` naming the symbol and day.
    """
    for symbol, closes in base.items():
        other = longer.get(symbol, {})
        for boundary, close in closes.items():
            if other.get(boundary) != close:
                raise ValueError(
                    f"{symbol}: the longer load's close at {boundary} is "
                    f"{other.get(boundary)!r}, not {close!r}"
                )
    return {symbol: dict(longer.get(symbol, {})) for symbol in base}


def signal_table(inputs: CarryInputs, lag_days: int = 0) -> tuple[SignalTable, int]:
    """The own signals of every decision (the shared eligibility rule) and the mask
    mismatches of a lag (C9-17).

    With ``lag_days = K`` an eligible coin's signal is ``trend_signal(closes, D - K d)``
    (the regime too); eligibility, prices and book vol stay those of D. A lagged
    window without history makes that coin None and counts its decision as a mismatch.
    """
    spec = inputs.spec
    closes = inputs.trend_closes or {}
    lag_ns = lag_days * DAY_NS
    rows: list[dict[str, float | None]] = []
    regimes: list[float | None] = []
    mismatches = 0
    for decision_ns in inputs.decisions:
        sizing = {s: sizing_price(inputs.ticks[s], decision_ns) for s in spec.symbols}
        eligible = decision_signals(
            spec,
            inputs.settlements,
            sizing,
            decision_ns,
            trend_closes=inputs.trend_closes,
            listings=inputs.listings,
        )
        if not lag_days:
            rows.append(dict(eligible.signals))
            regimes.append(eligible.regime)
            continue
        row = {
            symbol: None
            if value is None
            else trend_signal(closes.get(symbol, {}), decision_ns - lag_ns)
            for symbol, value in eligible.signals.items()
        }
        mismatches += any(
            eligible.signals[symbol] is not None and value is None for symbol, value in row.items()
        )
        rows.append(row)
        regimes.append(
            None
            if spec.trend_filter is None
            else trend_signal(closes.get(spec.trend_filter, {}), decision_ns - lag_ns)
        )
    return SignalTable(tuple(inputs.decisions), tuple(rows), tuple(regimes)), mismatches


def rotate[T](values: Sequence[T], u: float) -> list[T]:
    """Circular shift by ``k = 1 + floor(u (n - 1))`` (never the identity); n <= 1 unchanged."""
    n = len(values)
    if n <= 1:
        return list(values)
    k = 1 + math.floor(u * (n - 1))
    return [*values[k:], *values[:k]]


def _segments(table: SignalTable) -> list[range]:
    """Runs of consecutive decisions with the same eligible set and regime presence."""
    keys = [
        (frozenset(s for s, value in row.items() if value is not None), regime is not None)
        for row, regime in zip(table.signals, table.regime, strict=True)
    ]
    segments: list[range] = []
    start = 0
    for index in range(1, len(keys) + 1):
        if index == len(keys) or keys[index] != keys[start]:
            segments.append(range(start, index))
            start = index
    return segments


def rotate_table(
    table: SignalTable, u: float, segments: Sequence[range] | None = None
) -> tuple[SignalTable, int]:
    """The joint rotation null (C9-6, R-c9t5-1): each segment's rows shift together.

    Every coin's values stay on its own eligible decisions (a segment lies inside each
    eligible coin's mask), None stays None, and coins (and the regime) keep their
    co-movement on the overlap. Returns the rotated table and the one-decision
    (identity) segments. ``segments`` (from :func:`_segments`) saves recomputing them
    per draw.
    """
    order = list(range(len(table.decisions)))
    degenerate = 0
    for segment in _segments(table) if segments is None else segments:
        if len(segment) == 1:
            degenerate += 1
        order[segment.start : segment.stop] = rotate(list(segment), u)
    return (
        SignalTable(
            table.decisions,
            tuple(dict(table.signals[index]) for index in order),
            tuple(table.regime[index] for index in order),
        ),
        degenerate,
    )


def draw_fraction(seed: int, index: int) -> float:
    """Null draw ``index``'s rotation fraction ``u = Random(seed + index).random()`` (R57)."""
    return random.Random(seed + index).random()


def cross_wire(table: SignalTable) -> SignalTable:
    """The placebo (R60): each sorted eligible coin takes the next one's signal (cyclic)."""
    rows: list[dict[str, float | None]] = []
    for row in table.signals:
        eligible = sorted(symbol for symbol, value in row.items() if value is not None)
        wired = dict(row)
        for index, symbol in enumerate(eligible):
            wired[symbol] = row[eligible[(index + 1) % len(eligible)]]
        rows.append(wired)
    return SignalTable(table.decisions, tuple(rows), table.regime)


class TableRanker:
    """Feeds a :class:`SignalTable` to ``simulate_reference`` (R-c9t5-1 Q2).

    With ``gate`` (the spec has a trend filter) every eligible signal becomes 0.0 while
    the table's regime is None or <= 0: on a filter-free spec that is the native
    ``filter_off`` book (all weights 0, same eligible set for the size check).
    """

    def __init__(self, table: SignalTable, gate: bool) -> None:
        self._rows = dict(
            zip(table.decisions, zip(table.signals, table.regime, strict=True), strict=True)
        )
        self._gate = gate

    def __call__(
        self, decision_ns: int, signals: Mapping[str, float | None]
    ) -> dict[str, float | None]:
        row, regime = self._rows[decision_ns]
        for symbol, value in row.items():
            if value is not None and signals.get(symbol) is None:
                raise ValueError(
                    f"signal table has {symbol} eligible at {decision_ns}; the run does not"
                )
        result = {symbol: row.get(symbol) for symbol in signals}
        if self._gate and not (regime is not None and regime > 0.0):
            return {symbol: None if value is None else 0.0 for symbol, value in result.items()}
        return result


def table_run(inputs: CarryInputs, table: SignalTable, *, fee_multiplier: float = 1.0) -> CarryRun:
    """The reference run of ``table`` on the filter-free spec, the filter applied by
    :class:`TableRanker` (null draws, placebo, a lagged own run)."""
    spec = inputs.spec
    plain = (
        inputs
        if spec.trend_filter is None
        else replace(inputs, spec=replace(spec, trend_filter=None))
    )
    ranker = TableRanker(table, gate=spec.trend_filter is not None)
    return simulate_reference(plain, ranker=ranker, fee_multiplier=fee_multiplier)


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _signal_repeat(table: SignalTable) -> float | None:
    """Pooled share of consecutive eligible (coin, decision) pairs with an unchanged signal."""
    pairs = [
        (before[symbol], after.get(symbol))
        for before, after in pairwise(table.signals)
        for symbol in before
        if before[symbol] is not None and after.get(symbol) is not None
    ]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def exposure_stats(run: CarryRun, inputs: CarryInputs, table: SignalTable) -> dict[str, Any]:
    """The executed book's exposure (C9-6), the same function for the own run and a draw.

    Per decision, after its orders: gross and net notional / E (sizing prices) and the
    active legs; the regime's duty cycle (None without a filter); annual turnover (the
    report's formula) and the table's signal repeat rate.
    """
    spec = inputs.spec
    held = dict.fromkeys(spec.symbols, 0)
    gross: list[float] = []
    net: list[float] = []
    active: list[float] = []
    for decision in run.decisions:
        for order in decision.orders:
            held[order.symbol] += order.steps
        values = [
            float(Decimal(steps) * inputs.specs[symbol].step) * decision.prices[symbol]
            for symbol, steps in held.items()
            if steps
        ]
        gross.append(math.fsum(abs(value) for value in values) / decision.equity)
        net.append(math.fsum(values) / decision.equity)
        active.append(len(values))
    duty = None
    if spec.trend_filter is not None and table.regime:
        duty = sum(1 for value in table.regime if value is not None and value > 0.0) / len(
            table.regime
        )
    turnover = math.fsum(_turnover_by_decision(run).values())
    return {
        "mean_gross": _mean(gross),
        "mean_net": _mean(net),
        "mean_active": _mean(active),
        "filter_duty": duty,
        "turnover_annual": turnover / ((spec.end_ns - spec.start_ns) / (_DAYS_PER_YEAR * DAY_NS)),
        "signal_repeat": _signal_repeat(table),
    }


def _mean_exposure(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None] | None:
    """Per key, the mean over the draws that ran (None where undefined)."""
    if not rows:
        return None
    return {key: _mean([row[key] for row in rows if row[key] is not None]) for key in rows[0]}


def null_percentile(own: float | None, nulls: Sequence[float]) -> float | None:
    """``#{null < own} / draws`` (R58); None without an own Sharpe or draws."""
    if own is None or not nulls:
        return None
    return sum(1 for value in nulls if value < own) / len(nulls)


def nc1_verdict(percentiles: Sequence[float | None]) -> dict[str, Any]:
    """NC1 (R-c9-8, C9-17): FAIL iff ``#{percentile >= 0.95} >= 6`` over the 50 lags.

    An undefined percentile counts as flagged, so a broken null never certifies.
    """
    if len(percentiles) != len(NC1_LAGS_DAYS):
        raise ValueError(f"NC1 needs {len(NC1_LAGS_DAYS)} lag percentiles, got {len(percentiles)}")
    flagged = sum(1 for value in percentiles if value is None or not value < NC1_PERCENTILE)
    return {
        "lags": len(NC1_LAGS_DAYS),
        "flagged": flagged,
        "fail_at": NC1_FAIL_AT,
        "verdict": "FAIL" if flagged >= NC1_FAIL_AT else "PASS",
    }


def _trend_draws(
    inputs: CarryInputs,
    table: SignalTable,
    segments: Sequence[range],
    seed: int,
    draws: int,
    fee_multiplier: float,
) -> tuple[list[float], int, int, list[dict[str, Any]]]:
    """Daily Sharpes of the rotation null and the exposure block of every draw that
    completed. A ruined draw is -inf (its path really did worse than any survivor; R58,
    R-c9t5-3); an undefined (None) Sharpe is unknown, so it counts as >= own: +inf
    (conservative, R-c9t5-2). Both are counted, never dropped."""
    sharpes: list[float] = []
    ruined = 0
    undefined = 0
    exposures: list[dict[str, Any]] = []
    for index in range(draws):
        rotated, _ = rotate_table(table, draw_fraction(seed, index), segments)
        try:
            run = table_run(inputs, rotated, fee_multiplier=fee_multiplier)
            sharpe = daily_sharpe_only(run, inputs)
        except CarryEquityTooSmall:
            ruined += 1
            sharpes.append(float("-inf"))
            continue
        if sharpe is None:
            undefined += 1
        sharpes.append(math.inf if sharpe is None else sharpe)
        exposures.append(exposure_stats(run, inputs, rotated))
    return sharpes, ruined, undefined, exposures


def _drop_top3_daily(
    daily: Sequence[tuple[int, float, float]], weekly: Sequence[tuple[int, float]]
) -> float | None:
    """The daily Sharpe without the days of the three best Monday weeks (R61)."""
    if len(weekly) <= 3:
        return None
    best = sorted(weekly, key=lambda row: row[1])[-3:]
    windows = [(end - _WEEK_NS, end) for end, _ in best]
    kept = [move for ts, _, move in daily if not any(low < ts <= high for low, high in windows)]
    return _annualised_sharpe(kept, _DAYS_PER_YEAR)


def evaluate_trend(
    inputs: CarryInputs,
    *,
    draws: int,
    seed: int,
    trials_path: Path,
    family: Family | None,
    criteria: Criteria | None,
    signal_lag_days: int,
    spread_source: str,
    universe: Mapping[str, str] | None,
) -> dict[str, Any]:
    """The trend research report (arch-c9-2 R55-R62); every Sharpe is ``daily_sharpe_365``.

    N comes from the frozen ``family`` with the theoretical variance (order-free, R56);
    the null is :func:`rotate_table` (C9-6); pre-cost = fees x0 and spread 0 for the own
    run and every draw (R59); the placebo is :func:`cross_wire` (R60); the run's
    evidence block (:func:`~vibe_quant.portfolio.metrics.provenance`) is merged in
    (C9-8). ``signal_lag_days`` > 0 runs the own book on the lagged table (NC1, C9-17;
    ``inputs`` must already hold the earlier closes, :func:`extend_closes`): a calibration
    run, verdict ``calibration``, that never writes the trial ledger (the lag is not in
    the spec hash).
    """
    if family is None:
        raise ValueError("signal='trend' needs a frozen family (--family)")
    spec = inputs.spec
    require_arm(family, spec)
    lag_spec(spec, signal_lag_days)  # validates the lag
    table, mismatches = signal_table(inputs, signal_lag_days)

    def own(given: CarryInputs, fee_multiplier: float = 1.0) -> CarryRun:
        if signal_lag_days:
            return table_run(given, table, fee_multiplier=fee_multiplier)
        return simulate_reference(given, fee_multiplier=fee_multiplier)

    if signal_lag_days:
        # B1: the lag is not in the spec hash, so a calibration run never registers a trial
        run = own(inputs)
        report = carry_metrics(run, inputs)
    else:
        run, report = _own_report(inputs, trials_path, lambda: own(inputs))
    sharpe = report.summary["daily_sharpe_365"]
    segments = _segments(table)
    rotatable = any(len(segment) > 1 for segment in segments)
    degenerate = sum(1 for segment in segments if len(segment) == 1)

    null_sharpes, ruined, undefined, null_exposures = _trend_draws(
        inputs, table, segments, seed, draws, 1.0
    )
    precost = replace(inputs, spec=replace(spec, spread=0.0))
    precost_sharpe = daily_sharpe_only(own(precost, 0.0), precost)
    precost_sharpes, _, _, _ = _trend_draws(precost, table, segments, seed, draws, 0.0)

    placebo = daily_sharpe_only(table_run(inputs, cross_wire(table)), inputs)
    fee2x = daily_sharpe_only(own(inputs, 2.0), inputs)
    capped = replace(inputs, calc=CappedFundingCalculator(inputs.calc, DEFAULT_FUNDING_CAP))
    capped_sharpe = daily_sharpe_only(own(capped), capped)

    daily_returns = [move for _, _, move in report.daily]
    beta_btc, alpha_t_btc = _ols_pairs(
        daily_returns, _market_returns(inputs, "BTCUSDT", report.daily, DAY_NS)
    )
    beta_ew, alpha_t = _ols_pairs(daily_returns, _ew_returns(inputs, report.daily, DAY_NS))
    dsr, dsr_p_value, dsr_error, dsr_raw_p_value = _dsr(
        daily_returns, sharpe, family.num_trials, None, periods=_DAYS_PER_YEAR
    )
    ledger_num_trials, _ = trial_counts(
        read_trials(trials_path), spec.spec_hash(), signal=spec.signal
    )
    evidence = provenance(run, inputs, spread_source=spread_source, universe=universe)
    result: dict[str, Any] = {
        "statistic": TREND_STATISTIC,
        "sharpe": sharpe,
        "total_return": report.summary["total_return"],
        "max_drawdown_daily": report.summary["max_drawdown_daily"],
        "p_value": _p_value(sharpe, null_sharpes) if rotatable else None,
        "null_percentile": null_percentile(sharpe, null_sharpes) if rotatable else None,
        "p_value_pre_cost": _p_value(precost_sharpe, precost_sharpes) if rotatable else None,
        "draws": draws,
        "seed": seed,
        "ruined_draws": ruined,
        "undefined_draws": undefined,
        "degenerate_segments": degenerate,
        "exposure": {
            "own": exposure_stats(run, inputs, table),
            "null_mean": _mean_exposure(null_exposures),
        },
        "beta_btc": beta_btc,
        "alpha_t_btc": alpha_t_btc,
        "beta_ew": beta_ew,
        "alpha_t": alpha_t,
        "fee2x_sharpe": fee2x,
        "capped_sharpe": capped_sharpe,
        "placebo_sharpe": placebo,
        "placebo_kind": TREND_PLACEBO_KIND,
        "drop_top3_weeks_sharpe": _drop_top3_daily(report.daily, report.weekly),
        "dsr": dsr,
        "dsr_p_value": dsr_p_value,
        "dsr_error": dsr_error,
        "dsr_raw_p_value": dsr_raw_p_value,
        "dsr_num_trials": family.num_trials,
        "dsr_variance": DSR_VARIANCE,
        "family": family.to_json(),
        "ledger_num_trials": ledger_num_trials,
        "signal_lag_days": signal_lag_days,
        "lag_mask_mismatches": mismatches,
        "criteria_file": None
        if criteria is None
        else {"path": criteria.path, "sha256": criteria.sha256},
        "by_year": report.by_year,
        **evidence,
    }
    values = {**result, "coins_with_data": evidence["coverage"]["coins_with_data"]}
    result.update(judge(criteria, values, spec.fill_granularity, calibration=signal_lag_days > 0))
    return result
