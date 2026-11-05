"""The ONE run-sizing rule: exposure cap and risk per trade (bead vibe-quant-2gyk1.2).

Stdlib only, and deliberately outside :mod:`vibe_quant.validation` and
:mod:`vibe_quant.liquidation`: the API schemas import it on every launch request, so
reading a run's sizing settings must pull none of the engine stack (import isolation).

``max_position_pct`` is an exposure CAP, not a target: the compiled template sizes
``min(equity * risk_per_trade / stop_distance, equity * max_position_pct / price)`` and a
position is skipped below the instrument's min qty / min notional, so the actual exposure is
the smaller of the two. Margin leverage is a separate setting
(:mod:`vibe_quant.liquidation_settings`) and must be at least the cap
(:meth:`SizingSettings.check_leverage`).

``max_margin_pct`` is the margin limit of :mod:`vibe_quant.margin_limit`, and it is OFF by
default: ``None`` means the run never asked for one, and it is absent from ``to_params()``
and from :meth:`SizingSettings.strategy_params` (which carries the run leverage along only
when the limit is on).

``sizing_mode`` picks the stop distance the risk budget divides by (vibe-quant-9sost):
``"risk"`` (the default) uses the strategy's own stop-loss distance; ``"atr"`` uses
``ATR(sizing_atr_period) * sizing_atr_multiplier`` on the strategy timeframe, so a position
shrinks when the market gets volatile. The cap, the instrument minimums and the margin limit
apply the same way in both modes. The two ATR fields exist only in ATR mode: setting either
with ``"risk"`` is an error, and a risk run stores nothing new. (They are not named
``atr_period``: that is the config field of a DSL indicator named ``atr``.)

Every field is validated in EVERY mode and stored as floats (``sizing_atr_period`` as an
int, ``sizing_mode`` as a string): an invalid stored value must fail the run rather than
quietly become a different number downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_MAX_POSITION_PCT: float = 0.5
"""The one default exposure cap: at most 50% of equity per position."""

DEFAULT_RISK_PER_TRADE: float = 0.02
"""The one default risk budget: 2% of equity risked per trade."""

MAX_MAX_POSITION_PCT: float = 125.0
MAX_RISK_PER_TRADE: float = 0.5

SIZING_MODES: tuple[str, ...] = ("risk", "atr")
DEFAULT_ATR_PERIOD: int = 14
DEFAULT_ATR_MULTIPLIER: float = 2.0
MIN_ATR_PERIOD: int = 2
MAX_ATR_PERIOD: int = 10_000
MAX_ATR_MULTIPLIER: float = 20.0

SizingParams = dict[str, float | int | str]
"""Stored sizing parameters (:meth:`SizingSettings.to_params`): floats, plus the ATR mode's
``sizing_mode`` string and ``sizing_atr_period`` int."""

SIZING_FIELDS: tuple[str, ...] = (
    "max_position_pct",
    "risk_per_trade",
    "max_margin_pct",
    "sizing_mode",
    "sizing_atr_period",
    "sizing_atr_multiplier",
)
"""The ``parameters`` keys a run's sizing settings live under (``parameters.<key>``)."""

_ABSENT_MARGIN: object = object()
"""Sentinel: ``max_margin_pct`` missing from the parameters (absent) vs present-but-null."""


class SizingConfigError(ValueError):
    """A run's sizing settings are invalid (they are never silently defaulted)."""


def _invalid_cap(value: object) -> SizingConfigError:
    return SizingConfigError(
        f"max_position_pct must be a number in (0, {MAX_MAX_POSITION_PCT:g}], got {value!r}"
    )


def _invalid_risk(value: object) -> SizingConfigError:
    return SizingConfigError(
        "risk_per_trade must be more than 0 and at most "
        f"{MAX_RISK_PER_TRADE:g} ({MAX_RISK_PER_TRADE:.0%} of equity per trade), got {value!r}"
    )


def _invalid_margin(value: object) -> SizingConfigError:
    return SizingConfigError(f"max_margin_pct must be a number in (0, 1], got {value!r}")


def _is_amount(value: object, upper: float) -> bool:
    """Whether ``value`` is a finite number in ``(0, upper]``.

    The type is checked BEFORE any conversion: comparing an int against the bounds is exact
    (``float(10**400)`` would raise OverflowError), and a bool — an int subclass — is not a
    number here. ``0 < value <= upper`` is also False for ``nan``, ``inf`` and ``-inf``, so
    one comparison covers the whole non-finite family.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 < value <= upper


def _is_margin(value: object) -> bool:
    """Whether ``value`` is a finite number in ``(0, 1]`` (the margin limit's domain).

    Same type-first rule as :func:`_is_amount`: a bool is not a number here, ``None`` is not
    a fraction, and one comparison rejects ``nan``/``inf``/``-inf``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 < value <= 1.0


def _invalid_mode(value: object) -> SizingConfigError:
    return SizingConfigError(f"sizing_mode must be 'risk' or 'atr', got {value!r}")


def _invalid_atr_period(value: object) -> SizingConfigError:
    return SizingConfigError(
        f"sizing_atr_period must be a whole number in [{MIN_ATR_PERIOD}, {MAX_ATR_PERIOD}],"
        f" got {value!r}"
    )


def _invalid_atr_multiplier(value: object) -> SizingConfigError:
    return SizingConfigError(
        f"sizing_atr_multiplier must be a number in (0, {MAX_ATR_MULTIPLIER:g}], got {value!r}"
    )


def _is_atr_period(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return MIN_ATR_PERIOD <= value <= MAX_ATR_PERIOD


def _margin_value(value: object) -> float:
    """A present ``max_margin_pct`` as a float, or the error naming the raw value."""
    if not _is_margin(value):
        raise _invalid_margin(value)
    return float(cast("float", value))


@dataclass(frozen=True)
class SizingSettings:
    """The validated sizing of one run: ``max_position_pct``, ``risk_per_trade`` and the
    optional margin limit ``max_margin_pct``.

    A missing key means its default; an explicit ``None`` is rejected for the two exposure
    fields, while ``max_margin_pct`` defaults to ``None`` (OFF) and accepts ``None`` in the
    constructor — only :meth:`from_params` can tell an absent key from a stored ``null``, and
    it rejects the ``null``. A boolean is not a number, and every field is validated in every
    mode and stored as a float.
    """

    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE
    max_margin_pct: float | None = None
    sizing_mode: str = "risk"
    sizing_atr_period: int | None = None
    sizing_atr_multiplier: float | None = None

    def __post_init__(self) -> None:
        cap = self.max_position_pct
        if not _is_amount(cap, MAX_MAX_POSITION_PCT):
            raise _invalid_cap(cap)
        risk = self.risk_per_trade
        if not _is_amount(risk, MAX_RISK_PER_TRADE):
            raise _invalid_risk(risk)
        margin = self.max_margin_pct
        if margin is not None and not _is_margin(margin):
            raise _invalid_margin(margin)
        object.__setattr__(self, "max_position_pct", float(cap))
        object.__setattr__(self, "risk_per_trade", float(risk))
        if margin is not None:
            object.__setattr__(self, "max_margin_pct", float(margin))
        self._check_atr()

    def _check_atr(self) -> None:
        """``sizing_mode`` is one of :data:`SIZING_MODES`; the ATR fields are set (to their
        defaults when absent) in ATR mode and must be absent in risk mode."""
        mode = self.sizing_mode
        if mode not in SIZING_MODES:
            raise _invalid_mode(mode)
        period, mult = self.sizing_atr_period, self.sizing_atr_multiplier
        if mode == "risk":
            for name, value in (("sizing_atr_period", period), ("sizing_atr_multiplier", mult)):
                if value is not None:
                    raise SizingConfigError(
                        f"{name} is only valid with sizing_mode 'atr', got {name}={value!r}"
                        " with sizing_mode 'risk'"
                    )
            return
        period = DEFAULT_ATR_PERIOD if period is None else period
        if not _is_atr_period(period):
            raise _invalid_atr_period(period)
        mult = DEFAULT_ATR_MULTIPLIER if mult is None else mult
        if not _is_amount(mult, MAX_ATR_MULTIPLIER):
            raise _invalid_atr_multiplier(mult)
        object.__setattr__(self, "sizing_atr_period", int(period))
        object.__setattr__(self, "sizing_atr_multiplier", float(mult))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> SizingSettings:
        """Read the sizing settings from a run's parameters.

        A missing key means its default; an explicit ``None`` is rejected everywhere. The
        margin limit is the only field whose default is ``None``, so absence is detected on
        the KEY (never with a falsy ``or`` that would turn a stored ``null`` or ``0`` into
        "off"). Every other key is ignored and the mapping is never modified.
        """
        raw_margin = params.get("max_margin_pct", _ABSENT_MARGIN)
        margin = None if raw_margin is _ABSENT_MARGIN else _margin_value(raw_margin)
        raw_period = params.get("sizing_atr_period", _ABSENT_MARGIN)
        if raw_period is None:
            raise _invalid_atr_period(raw_period)
        raw_mult = params.get("sizing_atr_multiplier", _ABSENT_MARGIN)
        if raw_mult is None:
            raise _invalid_atr_multiplier(raw_mult)
        return cls(
            max_position_pct=cast(
                "float", params.get("max_position_pct", DEFAULT_MAX_POSITION_PCT)
            ),
            risk_per_trade=cast("float", params.get("risk_per_trade", DEFAULT_RISK_PER_TRADE)),
            max_margin_pct=margin,
            sizing_mode=cast("str", params.get("sizing_mode", "risk")),
            sizing_atr_period=None if raw_period is _ABSENT_MARGIN else cast("int", raw_period),
            sizing_atr_multiplier=None if raw_mult is _ABSENT_MARGIN else cast("float", raw_mult),
        )

    def to_params(self) -> SizingParams:
        """The parameters to store: only the keys a run moved off the default.

        ``SizingSettings().to_params() == {}``, so an explicit default stores byte-identically
        to a default launch. ``max_margin_pct`` is stored whenever it is set (its default is
        "off", which is not a number). ATR mode stores all three of its fields, defaults
        included, so the record names exactly what ran.
        """
        params: SizingParams = {}
        if self.max_position_pct != DEFAULT_MAX_POSITION_PCT:
            params["max_position_pct"] = self.max_position_pct
        if self.risk_per_trade != DEFAULT_RISK_PER_TRADE:
            params["risk_per_trade"] = self.risk_per_trade
        if self.max_margin_pct is not None:
            params["max_margin_pct"] = self.max_margin_pct
        if self.sizing_mode == "atr":
            params["sizing_mode"] = "atr"
            params["sizing_atr_period"] = cast("int", self.sizing_atr_period)
            params["sizing_atr_multiplier"] = cast("float", self.sizing_atr_multiplier)
        return params

    def strategy_params(self, leverage: float) -> SizingParams:
        """The parameters a strategy CONFIG gets: :meth:`to_params` plus the run leverage.

        The margin limit needs the leverage to compute a margin, and a compiled strategy must
        not read it off the live account (that only exposes per-instrument overrides), so the
        run leverage rides along — but only when the limit is ON, leaving an off run's config
        byte-identical to today's.
        """
        params = self.to_params()
        if self.max_margin_pct is not None:
            params["margin_leverage"] = float(leverage)
        return params

    def check_leverage(self, leverage: float) -> None:
        """Reject a cap the run's margin leverage cannot cover (S13).

        Checked in EVERY liquidation mode: the cap is exposure, the leverage only changes how
        much margin it needs, and a cap above the leverage would be unmarginable. Equality is
        allowed — a cap exactly at the leverage is fully marginable.
        """
        if self.max_position_pct > float(leverage):
            raise SizingConfigError(
                f"max_position_pct ({self.max_position_pct!r}) must not exceed "
                f"leverage ({float(leverage)!r})"
            )


def recorded_sizing(params: object) -> SizingSettings:
    """The one rule for reading sizing settings recorded with a run.

    A legacy row whose parameters are absent or not a mapping (``None``, a list, a string,
    a number) reads as the defaults. A mapping goes through the same rule as a launch
    request, so a corrupt recorded value is an error rather than a silent default.
    """
    if not isinstance(params, dict):
        return SizingSettings()
    return SizingSettings.from_params(params)
