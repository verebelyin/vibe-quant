"""Static paper warm-up classes for registry specs.

A paper start replays an 8,000-bar seed fetched from Binance REST through the
same per-bar path validation uses, then hands over to the live websocket feed.
Parity with a validation run therefore requires every indicator to reach its
full-history value inside the seed: an indicator whose value still moves with
the bar it started on cannot be warmed, and a strategy holding one is refused.

Each registry spec declares ``paper_warmup: Callable[[dict, int], int] | None``
(``vibe_quant/dsl/indicators.py``): the effective params and the bar length in
seconds go in, the number of bars the indicator needs to reach scaled parity
goes out, and a :class:`WarmupUnsupported` refusal comes out for a kind paper
refuses at any seed length.

The classes are envelopes derived from MEASURED convergence (architect probe
``out_bounds.txt``: 1e-9 horizon of a cold start vs the full history, max over
BTC/ETH/SOL 15m, five starts), not proofs — see design.md R2.1/R2.6.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

WarmupFn = Callable[[dict[str, object], int], int]
PeriodFn = Callable[[dict[str, object]], int]

# The validation-side start-phase drift the refused kinds share: a capped
# Wilder/adaptive recursion whose splice error is above the 1e-9 parity target.
D42M5_REASON: str = "start-phase drift above 1e-9 in validation itself (vibe-quant-d42m5)"

# Backtests compute VIDYA, ADAPTIVE_RSI and ADX(period > 16) from the bar type's first
# catalog bar; a paper seed starts elsewhere, so its value cannot be shown to match.
SEED_PARITY_REASON: str = (
    "paper's seed cannot be shown to reach the backtest's catalog-origin value "
    "(vibe-quant-d42m5; self-check vibe-quant-jriky)"
)

# A paper start holding this kind is refused whatever the seed length.
CUMULATIVE_REASON: str = "cumulative level never converges"

# A ``needs_context`` kind reads cross-asset/daily archives and never reaches
# the generated compute_fn path in a live node.
CONTEXT_REASON: str = (
    "context indicator: paper refuses the START (node.py reject_context_indicators)"
)


class WarmupUnsupported(ValueError):
    """The indicator class cannot be warmed in paper at any seed length."""


def _periods(params: Mapping[str, object]) -> list[int]:
    """Positive int params that can act as a lookback (``offset`` is a lag, not a period)."""
    return [
        int(value)
        for key, value in params.items()
        if key != "offset" and isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]


def _longest(params: Mapping[str, object], period: PeriodFn | None) -> int:
    """Longest period driving convergence: the spec's own lookback or the largest int param."""
    if period is not None:
        return max(int(period(dict(params))), 1)
    return max(_periods(params), default=1)


def _offset(params: Mapping[str, object]) -> int:
    """Bars the value is lagged by (the ``offset`` param), when the spec declares one."""
    value = params.get("offset", 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _ratio(k: int, period: PeriodFn | None) -> WarmupFn:
    """``k x longest period + offset``: one family's measured convergence ratio."""

    def paper_warmup(params: dict[str, object], bar_seconds: int) -> int:
        return k * _longest(params, period) + _offset(params)

    return paper_warmup


def finite(k: int = 2, *, period: PeriodFn | None = None) -> WarmupFn:
    """Window/finite kinds: ``k x longest period + offset`` (default ``k=2``)."""
    return _ratio(k, period)


def ema(k: int = 15, *, period: PeriodFn | None = None) -> WarmupFn:
    """EMA-family recursions: ``k x longest period + offset`` (default ``k=15``)."""
    return _ratio(k, period)


def rma(k: int = 30, *, period: PeriodFn | None = None) -> WarmupFn:
    """Wilder/RMA recursions: ``k x longest period + offset`` (default ``k=30``)."""
    return _ratio(k, period)


def adaptive(k: int = 60, *, period: PeriodFn | None = None) -> WarmupFn:
    """Adaptive (CMO/volatility-weighted) recursions: default ``k=60``."""
    return _ratio(k, period)


def daily() -> WarmupFn:
    """Indicators that reset at UTC midnight (VWAP): one full day of bars."""

    def paper_warmup(params: dict[str, object], bar_seconds: int) -> int:
        if bar_seconds <= 0:
            msg = f"bar length must be positive, got {bar_seconds}"
            raise ValueError(msg)
        return 24 * 60 * 60 // bar_seconds + 1

    return paper_warmup


def windowed(
    lookback_fn: PeriodFn,
    cap: int | None = None,
    *,
    min_margin: int = 1,
    reason: str = D42M5_REASON,
) -> WarmupFn:
    """Capped compute_fn kinds: one buffer trim cycle plus the lookback.

    ``lookback_fn`` is the spec's OWN ``pta_lookback_fn`` (or a function with
    the same formula when the spec relies on the ``period`` fallback of
    ``pta_lookback``): the class must use the same number the buffer uses, so
    there is no generic period proxy to drift from it.

    The runtime hands ``compute_fn`` a rolling buffer capped at ``cap`` (the
    spec's own ``pta_buffer_cap`` rule when ``None``), so a cold start computes
    from a different window start until the trim schedule realigns. Measured
    drift inside one cycle is <= 1e-9 for the declared kinds (``capped_all.py``).

    ``min_margin`` guards a recursion whose per-period decay needs far more
    history than a ``10 x lookback`` cap provides: below ``min_margin x
    lookback`` the kind refuses, because validation itself drifts there. The
    value is measured per spec (ADX: Wilder x Wilder at 400/16 passes, 400/17
    fails).
    """

    def paper_warmup(params: dict[str, object], bar_seconds: int) -> int:
        # Imported at call time: ``indicators`` imports this module to classify
        # its own specs, so a module-level import would be circular.
        from vibe_quant.dsl.indicators import pta_buffer_cap

        lookback = max(int(lookback_fn(dict(params))), 1)
        effective = int(cap) if cap is not None else pta_buffer_cap([lookback], full_history=False)
        if effective < min_margin * lookback:
            msg = f"{reason}: cap {effective} < {min_margin} x lookback {lookback}"
            raise WarmupUnsupported(msg)
        return effective + effective // 4 + lookback

    return paper_warmup


def refuse(reason: str) -> WarmupFn:
    """A kind paper always refuses, whatever the seed length."""

    def paper_warmup(params: dict[str, object], bar_seconds: int) -> int:
        raise WarmupUnsupported(reason)

    return paper_warmup


def by_kind(
    classes: Mapping[str, WarmupFn],
    *,
    param: str = "kind",
    default: str = "",
) -> WarmupFn:
    """Dispatch on the string param that selects the recursion family.

    Moving-average plugins (MA, MA_MACD) and PMAX accept a ``kind`` param, and
    HULL a ``mode``, that swaps the smoothing kernel: the bound has to follow
    the effective kind, since one spec cannot be finite and adaptive at once.
    """

    def paper_warmup(params: dict[str, object], bar_seconds: int) -> int:
        value = params.get(param)
        key = value.lower() if isinstance(value, str) else default
        chosen = classes.get(key)
        if chosen is None:
            msg = f"unknown {param} {key!r}, cannot classify the paper warm-up"
            raise WarmupUnsupported(msg)
        return chosen(params, bar_seconds)

    return paper_warmup
