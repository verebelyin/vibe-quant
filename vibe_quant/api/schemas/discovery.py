"""Discovery domain schemas."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from vibe_quant.api.schemas.finite import reject_non_finite
from vibe_quant.bankroll_grid import (
    BankrollCell,
    BankrollGridError,
    build_cells,
    reject_run_level_with_grid,
)
from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    MISSING,
    LiquidationSettings,
    resolve_leverage,
)
from vibe_quant.sizing_settings import SizingSettings


class BankrollGridRequest(BaseModel):
    """A discovery bankroll grid: margin mode x leverage x margin cushion (bead vibe-quant-t8nrv).

    The rule (domain, ordering, the 12-cell cap, the per-cell exposure-cap check) is
    ``vibe_quant.bankroll_grid``; ``strict`` keeps a bool or a numeric string out of the lists.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    modes: list[str]
    leverages: list[float]
    max_margin_pcts: list[float]


class DiscoveryLaunchRequest(BaseModel):
    # Unknown fields are an error: a mistyped knob (``bnb_discounts``) used to be
    # dropped silently and launch a default-fee run (bd vibe-quant-2gyk1.4).
    model_config = ConfigDict(extra="forbid")

    population: int = Field(default=20, ge=2)
    generations: int = Field(default=15, ge=1)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    crossover_rate: float = Field(default=0.8, ge=0, le=1)
    elite_count: int = Field(default=2, ge=0)
    tournament_size: int = Field(default=3, ge=1)
    convergence_generations: int = Field(default=10, ge=1)
    symbols: list[str] = ["BTCUSDT"]
    timeframes: list[str] = ["4h"]
    indicator_pool: list[str] | None = None
    direction: str | None = None  # "long", "short", "both", or None (random)
    start_date: str | None = None
    end_date: str | None = None
    eval_windows: int = Field(default=3, ge=0)  # worst-of-N sub-windows; 1=single, 0=not sent (CLI default 3)
    # "worst": each symbol backtested alone, genome scored by its worst symbol
    symbol_agg: Literal["portfolio", "worst"] = "portfolio"
    # TRAIN fraction; default 0.8 = last 20% is a holdout used once as the final
    # pass/fail gate (vibe-quant-e70tl.5). 0 explicitly disables the holdout.
    train_test_split: float = Field(default=0.8, ge=0, lt=1)
    cross_window_months: list[int] | None = None  # e.g. [1, 2]: train sub-windows starting +1mo/+2mo
    cross_window_min_sharpe: float = 0.5  # min Sharpe on each shifted window (all must pass)
    num_seeds: int = Field(default=1, ge=1)  # >1 enables multi-seed ensemble
    wfa_oos_step_days: int = Field(default=0, ge=0)  # >0 enables rolling-window stability check over the TRAIN range
    wfa_min_consistency: float = Field(default=0.75, ge=0, le=1)  # min profitable fraction for WFA
    immigrant_fraction: float = 0.15  # fraction of pop replaced when entropy low; 0 disables
    entropy_threshold: float = 0.4  # population entropy below this triggers immigrant injection
    crowding_enabled: bool = True  # deterministic crowding selection (vs classic tournament)
    seed_run_id: int | None = None  # warm-start GA from top chromosomes of prior run
    no_bootstrap_ci: bool = False  # True disables bootstrap-CI hard gate (low-budget regimes)
    bootstrap_min_sharpe: float | None = None  # override timeframe-aware default (0.5 for 1m, 1.0 otherwise)
    seed: int | None = Field(default=None, ge=0, lt=2**32)  # RNG seed to replay a run; None = random
    max_workers: int = Field(default=4, ge=1, le=64)  # backtest processes for this run
    # Charge the BNB-discounted (x0.9) fee schedule on every screening backtest
    # of this discovery (bd vibe-quant-2gyk1.4). Recorded in run parameters and
    # forwarded to the CLI only when true; the default keeps VIP0.
    bnb_discount: StrictBool = False
    # Discovery screens with market entries only: post-only maker entries are
    # validation-only (screening maker-fill parity unproven, bead
    # vibe-quant-7eek0). Declared so an explicit non-market value fails loudly
    # instead of being ignored; never forwarded to the run.
    entry_order: str = "market"
    # Run-level liquidation mode for every screening backtest of this discovery
    # (vibe-quant-2gyk1.1.10). Forwarded to the CLI only when on; the default
    # keeps the mode off and the stored params unchanged (C6).
    liquidation_mode: Literal["off", "isolated", "cross"] = "off"
    # Run-level leverage for every screening backtest of this discovery, in
    # EVERY mode: a whole number in [1, 125] (vibe-quant-6wq6w). Forwarded and
    # stored only when the mode is on or the value is not the default, so a
    # default launch keeps its stored params and argv unchanged (C6).
    leverage: float | None = None
    # Deprecated alias for ``leverage``; the two must agree (A2).
    liquidation_leverage: float | None = None
    # Run-level sizing for every screening backtest of this discovery (bead
    # vibe-quant-2gyk1.2): the exposure cap (equity fractions, (0, 125]) and the
    # risk budget per trade ((0, 0.5]). Omitted = the run default (0.5 / 0.02);
    # an explicit null is a 422, never a silent default. Forwarded and stored
    # only when not the default, so a default launch keeps its stored params and
    # argv unchanged (S27).
    max_position_pct: float | None = None
    risk_per_trade: float | None = None
    # Run-level margin limit for every screening backtest of this discovery (bead
    # vibe-quant-t8nrv, T6): the margin in use across open positions plus pending
    # entries may not exceed this fraction of equity (the one rule of
    # ``vibe_quant.margin_limit``). Omitted = OFF, which is the limit's own default;
    # an explicit null is a 422, never a silent off. Forwarded and stored only when
    # set, so a default launch keeps its stored params and argv unchanged.
    max_margin_pct: float | None = None
    # Bankroll grid (bead vibe-quant-t8nrv, T8): discovery searches every mode x leverage x
    # cushion cell and each champion records its cell. Excludes the run-level fields above;
    # omitted = discovery runs exactly as without a grid.
    bankroll_grid: BankrollGridRequest | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_null_leverage(cls, data: object) -> object:
        """An explicitly sent ``null`` is a 422, never a silent default (R1)."""
        if isinstance(data, dict):
            for key in ("leverage", "liquidation_leverage"):
                if key in data and data[key] is None:
                    msg = f"{key} must be a whole number in [1, 125], got None"
                    raise ValueError(msg)
        return data

    # Declared AFTER every other before-validator on purpose: Pydantic runs
    # model before-validators in reverse declaration order (last declared first), so
    # this one runs first and a non-finite number is always named, never hidden by
    # another before-validator's error (review B1).
    @model_validator(mode="before")
    @classmethod
    def _reject_non_finite(cls, data: object) -> object:
        # JSON's bare NaN/Infinity/-Infinity are not JSON values, and a NaN
        # leverage used to make FastAPI's own 422 un-serialisable -> 500.
        reject_non_finite(data)
        return data

    @field_validator("leverage", "liquidation_leverage", mode="before")
    @classmethod
    def _validate_leverage(cls, value: Any) -> float:
        """One parse rule for the value: the shared ``LiquidationSettings`` check."""
        return LiquidationSettings(mode="off", leverage=value).leverage

    @field_validator("max_position_pct", "risk_per_trade", "max_margin_pct", mode="before")
    @classmethod
    def _validate_sizing(cls, value: Any, info: ValidationInfo) -> float:
        """One parse rule per sizing field: the shared ``SizingSettings`` check.

        A field validator (not a model before-validator) on purpose: an explicit ``null``
        is a 422 whose error sits on its OWN field, so an independently bad leverage is
        still reported alongside it (S27b). An omitted field never reaches here and keeps
        its ``None`` default, i.e. the run's default (S1).

        ``max_margin_pct`` is the one sizing field whose default IS ``None`` (off), so the
        constructor cannot tell an omitted key from a stored null; it is read through the
        shared ``from_params`` rule instead, which rejects the null.
        """
        if info.field_name == "max_position_pct":
            return SizingSettings(max_position_pct=value).max_position_pct
        if info.field_name == "risk_per_trade":
            return SizingSettings(risk_per_trade=value).risk_per_trade
        return cast("float", SizingSettings.from_params({"max_margin_pct": value}).max_margin_pct)

    @field_validator("entry_order")
    @classmethod
    def _market_entries_only(cls, value: str) -> str:
        if value != "market":
            msg = (
                f"entry_order={value!r} is validation-only "
                "(screening maker-fill parity unproven, bead vibe-quant-7eek0)"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _elite_below_population(self) -> DiscoveryLaunchRequest:
        if self.elite_count >= self.population:
            msg = "elite_count must be < population"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _resolve_leverage_alias(self) -> DiscoveryLaunchRequest:
        """Collapse ``leverage`` / ``liquidation_leverage`` through the shared rule.

        Both supplied and equal is fine; different values are a 422 naming both
        (A2). The resolved value is kept on ``leverage`` for the launch router.
        """
        if self.leverage is not None or self.liquidation_leverage is not None:
            resolved = resolve_leverage(
                MISSING if self.leverage is None else self.leverage,
                MISSING if self.liquidation_leverage is None else self.liquidation_leverage,
            )
            self.leverage = cast("float", resolved)
        return self

    # Declared AFTER ``_resolve_leverage_alias`` on purpose: model after-validators run in
    # declaration order, so the leverage alias is resolved before the cap is compared to it.
    @model_validator(mode="after")
    def _check_sizing_leverage(self) -> DiscoveryLaunchRequest:
        """S13/S27: a cap above the run's margin leverage would be unmarginable (422).

        A grid has no run-level leverage: each cell checks the cap at its own leverage.
        """
        if self.bankroll_grid is None:
            self.sizing_settings().check_leverage(self.liquidation_settings().leverage)
        return self

    @model_validator(mode="after")
    def _check_bankroll_grid(self) -> DiscoveryLaunchRequest:
        """R52/R53/R54: a grid excludes run-level settings and every cell must be valid (422)."""
        if self.bankroll_grid is not None:
            try:
                reject_run_level_with_grid(
                    liquidation_mode=self.liquidation_mode,
                    leverage=self.leverage,
                    liquidation_leverage=self.liquidation_leverage,
                    max_margin_pct=self.max_margin_pct,
                )
                self.bankroll_cells()
            except BankrollGridError as exc:
                raise ValueError(str(exc)) from exc
        return self

    def bankroll_cells(self) -> list[BankrollCell]:
        """The grid's validated cells (empty without a grid)."""
        grid = self.bankroll_grid
        if grid is None:
            return []
        return build_cells(
            grid.modes, grid.leverages, grid.max_margin_pcts, self.sizing_settings()
        )

    def liquidation_settings(self) -> LiquidationSettings:
        """The run's validated settings (the launch router's single source)."""
        leverage = self.leverage if self.leverage is not None else DEFAULT_LEVERAGE
        return LiquidationSettings(mode=self.liquidation_mode, leverage=leverage)

    def sizing_settings(self) -> SizingSettings:
        """The run's validated sizing (the launch router's single source).

        An omitted field means its default; only the fields explicitly supplied are
        read, so ``SizingSettings().to_params()`` stays empty for a default launch.
        """
        params: dict[str, float] = {}
        if self.max_position_pct is not None:
            params["max_position_pct"] = self.max_position_pct
        if self.risk_per_trade is not None:
            params["risk_per_trade"] = self.risk_per_trade
        if self.max_margin_pct is not None:
            params["max_margin_pct"] = self.max_margin_pct
        return SizingSettings.from_params(params)


class DiscoveryJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    status: str
    started_at: str | None
    completed_at: str | None = None
    progress: dict[str, object] | None = None
    symbols: list[str] | None = None
    timeframe: str | None = None
    generations: int | None = None
    population: int | None = None
    strategies_found: int | None = None
    error_message: str | None = None
    num_seeds: int | None = None
    wfa_oos_step_days: int | None = None
    cross_window_months: list[int] | None = None
    eval_windows: int | None = None
    direction: str | None = None
    best_sharpe: float | None = None
    best_return: float | None = None
    seed: int | None = None  # RNG seed from the run's result notes (None while running / old runs)


class DiscoveryResultResponse(BaseModel):
    strategies: list[dict[str, object]]
    # Why a completed run has no strategies (all champions guardrail-rejected)
    reason: str | None = None
    guardrail_rejections: list[dict[str, object]] | None = None


class PromoteResponse(BaseModel):
    strategy_id: int
    run_id: int
    name: str
    mode: str
    # Range the launched run covers: "holdout" (out-of-sample, validation
    # default when the discovery has a holdout) or "full" (discovery range).
    date_range: str = "full"
    start_date: str | None = None
    end_date: str | None = None


class ReplayResponse(BaseModel):
    replay_run_id: int
    original_run_id: int
    # Set when replay metrics are NOT expected to equal the stored champion
    # fitness (discovery scored worst-of-N eval windows; replay runs one
    # continuous full-range backtest).
    metrics_note: str | None = None
