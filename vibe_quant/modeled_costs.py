"""The modeled costs NT never books, as a ledger a strategy can size against (vibe-quant-q3eq5.2).

NT's backtest account books commissions only: archived funding and validation's post-fill SPEC
slippage are reconstructed AFTER the run (`trade_ledger.cost_positions`), so a generated strategy
sizes on an equity that never paid them -- a funded run's printed balance and its costed balance
drift apart, and a costed balance can go non-positive while NT keeps trading.

This module is the ONE rule both sides read: :class:`CostLedger` finalizes a position's costs
(funding to the SAME refined exit the extraction charges, plus the entry and exit slippage legs)
when NT closes it, answers ``paid_through(now)`` for the positions still open, and
:func:`reconcile` proves the sizing ledger and the post-hoc trade ledger agree to the float.

Nothing installs a ledger in a run yet (design-costs.md § 2 "Install", T2): the module imports
without the validation/liquidation packages, which import it back; those imports are lazy, as in
:mod:`vibe_quant.trade_ledger` (design Risk 3).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from vibe_quant.trade_ledger import PositionCosts
    from vibe_quant.validation.fill_model import SlippageEstimator
    from vibe_quant.validation.funding import FundingCalculator

__all__ = [
    "CostLedger",
    "CostModel",
    "MarketStatsSeries",
    "ModeledCostIntegrityError",
    "SlippageLegModel",
    "reconcile",
]

#: Conservative bar stats of an instrument with no series: the same fallbacks
#: ``validation.extraction`` uses today, inlined (importing that module here would close the
#: modeled_costs -> validation -> extraction -> modeled_costs cycle T3 opens).
DEFAULT_AVG_BAR_VOLUME = 1000.0
DEFAULT_BAR_VOLATILITY = 0.02
#: The SPEC half-spread input of every modeled slippage leg (as ``extraction._SlippageModel``).
DEFAULT_SPREAD = 0.0001
#: The prefix-sum variance is trusted while it keeps more than this share of the raw second
#: moment (rounding error ~1e-16 * moment / variance <= 1e-12); below it ``before`` recomputes.
CANCELLATION_RATIO = 1e-4


#: NT NETTING snapshots a closed position on re-entry under ``<id>-<uuid4>``.
_SNAPSHOT_SUFFIX = re.compile(
    r"-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def cost_key(pos: Any) -> str:
    """``<position id>@<ts_opened>`` with a trailing NT snapshot ``-<uuid4>`` stripped.

    The live close event carries ``X-000``; the same trade read back from
    ``cache.position_snapshots`` is ``X-000-<uuid4>``. Both must land on one key.
    """
    return f"{_SNAPSHOT_SUFFIX.sub('', str(pos.id))}@{int(pos.ts_opened)}"


class ModeledCostIntegrityError(DataUnavailableError):
    """The sizing ledger and the trade ledger disagree about a trade's modeled costs.

    A :class:`~vibe_quant.errors.DataUnavailableError`, so a run fails closed instead of scoring
    a balance the post-hoc ledger cannot reproduce. Raised by :func:`reconcile`; transported by
    the screening process pool, so it takes ONE positional message and pickles unchanged.
    """


@dataclass(frozen=True)
class CostModel:
    """Everything a :class:`CostLedger` needs to price a run's positions.

    Screening: no ``slippage`` (the fill model books it inside the prices) and ``refine`` =
    :func:`~vibe_quant.trade_ledger.funding_refine_instruments`. Validation: 1m execution refines
    nothing (empty ``refine``) and the SPEC slippage model is set. ``execution_bar_ns`` is the
    execution bar's length in ns (0 when unknown); ``catalog_path`` is where the refinement loads
    its 1m LAST windows from (``None`` -> no series -> ``MissingExitBars``).
    """

    funding: FundingCalculator
    refine: frozenset[str] = frozenset()
    execution_tf: str | None = None
    execution_bar_ns: int = 0
    catalog_path: str | None = None
    slippage: SlippageLegModel | None = None


class CostLedger:
    """The modeled costs of one account: finalized closes plus the open positions' funding.

    One ledger per shared NT account. ``finalized`` is insertion-ordered and updated once per NEW
    key (``on_position_closed`` is idempotent, so a repeated close event changes no number); the
    running total is kept alongside it, never recomputed from the dict on every read.
    """

    def __init__(self, model: CostModel, cache: Any) -> None:
        self.model = model
        #: The engine cache: only ``order(client_order_id)`` is read (the exit refinement).
        self.cache = cache
        #: ``position key -> modeled costs paid`` (positive = paid), in close order.
        self.finalized: dict[str, float] = {}
        #: When the run's positions were force-closed at the end of data (NT ``on_stop``).
        self.stopped_ns: int | None = None
        self._total = 0.0

    def on_position_closed(self, pos: Any) -> None:
        """Finalize ``pos``: funding to its exit, then the entry and the exit slippage leg.

        The funding exit is the post-hoc one: for a ``refine`` instrument the CLOSING fill's
        refined 1m instant (:func:`~vibe_quant.screening.liquidation_metrics.funding_exit_ts`
        over ONE 1m window per boundary exit), else NT's ``ts_closed``. A failure of the
        refinement propagates unchanged (``MissingExitBars``) or as the R10b
        ``DataUnavailableError``; nothing is finalized for that position.
        """
        key = cost_key(pos)
        if key in self.finalized:
            return
        total = self._costs(pos)
        self.finalized[key] = total
        self._total += total

    def paid_through(self, now_ns: int, open_positions: Iterable[Any]) -> float:
        """Modeled costs paid up to ``now_ns``: finalized closes plus the open positions' funding.

        An open position is funded from its entry through ``now_ns`` inclusive (the accrual's
        ``(entry, now]`` rule) on its entry notional; positive = paid, 0.0 when there is nothing.
        """
        from vibe_quant.trade_ledger import accrue_position_funding

        total = self._total
        for pos in open_positions:
            total += accrue_position_funding(self.model.funding, pos, exit_ns=now_ns).total
        return total

    def stopped(self, now_ns: int) -> None:
        """Record that NT force-closed the run at ``now_ns`` (closes at/after it are not ours)."""
        self.stopped_ns = int(now_ns)

    def _costs(self, pos: Any) -> float:
        """One closed position's funding plus its two slippage legs (the post-hoc sum)."""
        from vibe_quant.trade_ledger import _funding_close_ns, accrue_position_funding

        instrument_id = str(pos.instrument_id)
        exit_ns = int(pos.ts_closed)
        if self.model.refine and instrument_id in self.model.refine:
            from vibe_quant.screening.liquidation_metrics import funding_exit_ts, scan_fill_ts_map

            timing = funding_exit_ts(
                self.cache,
                [pos],
                self.model.execution_tf,
                self.model.catalog_path,
                self.model.refine,
            )
            exit_ns = _funding_close_ns(pos, scan_fill_ts_map(timing), self.model.refine)
        total = accrue_position_funding(self.model.funding, pos, exit_ns=exit_ns).total
        slippage = self.model.slippage
        if slippage is not None:
            # Legs first, then onto funding: the post-hoc `funding + (entry + exit)` float order.
            legs = slippage.leg(pos, float(pos.avg_px_open), int(pos.ts_opened))
            legs += slippage.leg(pos, float(pos.avg_px_close), int(pos.ts_closed))
            total += legs
        return total


def reconcile(ledger: CostLedger, costed: Sequence[PositionCosts]) -> None:
    """Prove the sizing ledger and the post-hoc trade ledger agree, trade by trade.

    Closed positions only (amendment 5: an open position's end-of-data exit fee and slippage are
    a documented addition, not part of the sizing path). For every position the ledger finalized,
    ``accrual.total + slippage_cost`` must equal the finalized value EXACTLY -- the same
    :func:`~vibe_quant.trade_ledger.accrue_position_funding` call and the same float sums. A
    closed position the ledger never finalized is allowed only at or after ``stopped_ns`` (NT's
    ``on_stop`` closes); before it, or with no stop at all, the ledger missed a real close. The
    other direction too: every finalized key needs a closed costed item, else a cost charged to
    sizing has no post-hoc trade (an empty ``costed`` after a cache-read error included).

    Raises:
        ModeledCostIntegrityError: the values disagree or a close is missing from the ledger.
    """
    seen: set[str] = set()
    for item in costed:
        if item.mark is not None:
            continue
        pos = item.pos
        key = cost_key(pos)
        seen.add(key)
        if key in ledger.finalized:
            sizing = ledger.finalized[key]
            trade = (0.0 if item.accrual is None else item.accrual.total) + item.slippage_cost
            if sizing != trade:
                raise ModeledCostIntegrityError(
                    f"modeled costs: sizing ledger and trade ledger disagree for {key}: "
                    f"sizing={sizing!r} ledger={trade!r}"
                )
            continue
        ts_closed = int(pos.ts_closed)
        if ledger.stopped_ns is None or ts_closed < ledger.stopped_ns:
            raise ModeledCostIntegrityError(
                f"modeled costs: the sizing ledger never closed {key} "
                f"(ts_closed={ts_closed}, stopped_ns={ledger.stopped_ns})"
            )
    for key in ledger.finalized:
        if key not in seen:
            raise ModeledCostIntegrityError(
                f"modeled costs: the trade ledger has no closed trade for the finalized {key}"
            )


class MarketStatsSeries:
    """Causal bar stats of ONE instrument: prefix sums over its bars, keyed by ``ts_init``.

    ``before(ts_ns)`` answers the (average bar volume, per-bar volatility) pair from the bars
    with ``ts_init < ts_ns`` -- strictly before, so a bar at the fill instant (or later) cannot
    move the slippage that fill is charged (design Risk 6 / D1). The pair is today's
    ``extraction.estimate_market_stats`` formula over that prefix: mean volume of the bars with
    volume > 0 (1000.0 when none), and the sample std of the log returns of consecutive positive
    closes (0.02 when fewer than two returns or a zero variance).
    """

    __slots__ = ("_ret_n", "_ret_sum", "_ret_sum2", "_returns", "_ts", "_vol_n", "_vol_sum")

    def __init__(
        self,
        ts: np.ndarray,
        vol_sum: np.ndarray,
        vol_n: np.ndarray,
        ret_n: np.ndarray,
        ret_sum: np.ndarray,
        ret_sum2: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        self._ts = ts
        self._vol_sum = vol_sum
        self._vol_n = vol_n
        self._ret_n = ret_n
        self._ret_sum = ret_sum
        self._ret_sum2 = ret_sum2
        #: The valid log returns in order: ``_returns[:ret_n]`` is a prefix's return list.
        self._returns = returns

    @classmethod
    def from_bars(
        cls, ts_init: Any, close: Any, volume: Any
    ) -> MarketStatsSeries:
        """Prefix sums of an instrument's bars, sorted by ``ts_init`` (stable on ties)."""
        ts = np.asarray(ts_init, dtype=np.int64)
        closes = np.asarray(close, dtype=np.float64)
        volumes = np.asarray(volume, dtype=np.float64)
        order = np.argsort(ts, kind="stable")
        ts, closes, volumes = ts[order], closes[order], volumes[order]

        positive = volumes > 0
        vol_sum = np.cumsum(np.where(positive, volumes, 0.0))
        vol_n = np.cumsum(positive.astype(np.int64))

        # A log return pairs two CONSECUTIVE positive closes (a non-positive close is dropped and
        # the next positive one pairs with the last positive before it, as in the group formula).
        returns = np.zeros(len(ts), dtype=np.float64)
        has_return = np.zeros(len(ts), dtype=bool)
        previous: float | None = None
        for i, value in enumerate(closes):
            if value <= 0:
                continue
            if previous is not None:
                returns[i] = math.log(value / previous)
                has_return[i] = True
            previous = value
        return cls(
            ts=ts,
            vol_sum=vol_sum,
            vol_n=vol_n,
            ret_n=np.cumsum(has_return.astype(np.int64)),
            ret_sum=np.cumsum(returns),
            ret_sum2=np.cumsum(returns * returns),
            returns=returns[has_return],
        )

    def before(self, ts_ns: int) -> tuple[float, float]:
        """``(avg_bar_volume, bar_volatility)`` of the bars with ``ts_init < ts_ns``."""
        end = int(np.searchsorted(self._ts, int(ts_ns), side="left"))
        if end <= 0:
            return DEFAULT_AVG_BAR_VOLUME, DEFAULT_BAR_VOLATILITY
        count = int(self._vol_n[end - 1])
        avg_volume = (
            float(self._vol_sum[end - 1]) / count if count else DEFAULT_AVG_BAR_VOLUME
        )
        n = int(self._ret_n[end - 1])
        volatility = DEFAULT_BAR_VOLATILITY
        if n >= 2:
            total = float(self._ret_sum[end - 1])
            squares = float(self._ret_sum2[end - 1])
            centered = squares - total * total / n
            if centered <= squares * CANCELLATION_RATIO:
                # The prefix moments cancel (near-constant returns): recompute the reference
                # two-pass formula over the prefix, as ``estimate_market_stats`` does.
                returns = self._returns[:n].tolist()
                mean = sum(returns) / n
                centered = sum((r - mean) ** 2 for r in returns)
            variance = centered / (n - 1)
            if variance > 0:
                volatility = math.sqrt(variance)
        return avg_volume, volatility


@dataclass(frozen=True)
class SlippageLegModel:
    """One fill's SPEC slippage cost, from the instrument's causal bar stats.

    ``estimator=None`` is the "off" shape: the engine already models slippage (``prob_slippage``
    > 0) and a leg charges nothing, exactly as ``extraction._SlippageModel`` today. The leg is
    ``estimate_cost(price, peak_qty, avg_volume, volatility, 0.0001)``.
    """

    estimator: SlippageEstimator | None = None
    stats: Mapping[str, MarketStatsSeries] = field(default_factory=dict)

    def leg(self, pos: Any, price: float, ts_ns: int) -> float:
        """Slippage cost of one fill of ``pos`` at ``price`` (``peak_qty`` units) at ``ts_ns``."""
        if self.estimator is None:
            return 0.0
        series = self.stats.get(str(pos.instrument_id))
        if series is None:
            avg_volume, volatility = DEFAULT_AVG_BAR_VOLUME, DEFAULT_BAR_VOLATILITY
        else:
            avg_volume, volatility = series.before(int(ts_ns))
        return self.estimator.estimate_cost(
            entry_price=price,
            order_size=float(pos.peak_qty),
            avg_volume=avg_volume,
            volatility=volatility,
            spread=DEFAULT_SPREAD,
        )
