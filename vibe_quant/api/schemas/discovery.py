"""Discovery domain schemas."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    computed_field,
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
from vibe_quant.metrics import SHARPE_DISPLAY_DESCRIPTION, sharpe_to_display
from vibe_quant.sizing_settings import SizingSettings
from vibe_quant.symbol_leverage import SymbolLeverageError, check_leverage


class BankrollGridRequest(BaseModel):
    """A discovery bankroll grid: margin mode x leverage x margin cushion (bead vibe-quant-t8nrv).

    The rule (domain, ordering, the 12-cell cap, the per-cell exposure-cap check) is
    ``vibe_quant.bankroll_grid``; ``strict`` keeps a bool or a numeric string out of the lists.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    modes: list[str]
    leverages: list[float]
    max_margin_pcts: list[float]


_INT64_LIMIT = 2**63


def _reject_big_ints(value: object, path: str) -> None:
    """Raise ``ValueError`` naming the first integer (not bool) outside int64 in ``value``."""
    if isinstance(value, int) and not isinstance(value, bool):
        if not -_INT64_LIMIT <= value < _INT64_LIMIT:
            msg = f"{path or 'body'}: integer out of range"
            raise ValueError(msg)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_big_ints(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_big_ints(item, f"{path}.{index}" if path else str(index))


# The API adapter's own defaults and validators live here; the run a request launches
# is ``DiscoveryRunSpec.from_request`` (``vibe_quant.discovery.run_spec``), which owns
# the parser defaults. A per-adapter default difference (neuvu.6) stays here, commented.
class DiscoveryLaunchRequest(BaseModel):
    # Unknown fields are an error: a mistyped knob (``bnb_discounts``) used to be
    # dropped silently and launch a default-fee run (bd vibe-quant-2gyk1.4).
    # ``allow_inf_nan=False``: a float parsed from a string ("NaN", "-Infinity") is a 422
    # too; the raw JSON literals are caught by ``_reject_non_finite`` (bead vibe-quant-clmmc.1).
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # 50 x 20 = the CLI default and the launch form's 4h budget (bead vibe-quant-clmmc.1).
    population: int = Field(default=50, ge=2)
    generations: int = Field(default=20, ge=1)
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
    eval_windows: int = Field(
        default=3, ge=0
    )  # worst-of-N sub-windows; 1=single, 0=not sent (CLI default 3)
    # "worst": each symbol backtested alone, genome scored by its worst symbol; "pooled"
    # (vibe-quant-clmmc.5): each symbol on its own account, scored as one equal-weight pool
    # of their daily MTM returns (needs 2+ symbols and evaluator "nt": 422 otherwise)
    symbol_agg: Literal["portfolio", "worst", "pooled"] = "portfolio"
    # TRAIN fraction; default 0.8 = last 20% is a holdout used once as the final
    # pass/fail gate (vibe-quant-e70tl.5). 0 explicitly disables the holdout.
    train_test_split: float = Field(default=0.8, ge=0, lt=1)
    cross_window_months: list[int] | None = (
        None  # e.g. [1, 2]: train sub-windows starting +1mo/+2mo
    )
    # Min Sharpe on each shifted window, typed in 365-day units like the on-screen Sharpe;
    # None = the stored default 0.5 (shown 0.60). Converted once in DiscoveryRunSpec.
    cross_window_min_sharpe_365: float | None = None
    num_seeds: int = Field(default=1, ge=1)  # >1 enables multi-seed ensemble
    wfa_oos_step_days: int = Field(
        default=0, ge=0
    )  # >0 enables rolling-window stability check over the TRAIN range
    wfa_min_consistency: float = Field(default=0.75, ge=0, le=1)  # min profitable fraction for WFA
    immigrant_fraction: float = 0.15  # fraction of pop replaced when entropy low; 0 disables
    entropy_threshold: float = 0.4  # population entropy below this triggers immigrant injection
    crowding_enabled: bool = True  # deterministic crowding selection (vs classic tournament)
    seed_run_id: int | None = None  # warm-start GA from top chromosomes of prior run
    no_bootstrap_ci: bool = False  # True disables bootstrap-CI hard gate (low-budget regimes)
    bootstrap_min_sharpe: float | None = (
        None  # None = the CLI's timeframe default (1m 0.5, 4h/1d 0.0, else 1.0)
    )
    seed: int | None = Field(
        default=None, ge=0, lt=2**32
    )  # RNG seed to replay a run; None = random
    max_workers: int = Field(default=4, ge=1, le=64)  # backtest processes for this run
    # GA fitness evaluator (bead vibe-quant-z8vqv.5): "fast" scores genomes with the
    # numpy fast tier and re-runs every gate input on NT before any gate (a mismatch
    # fails the run). Forwarded and stored only when "fast", so a default launch keeps
    # its stored params and argv unchanged.
    evaluator: Literal["nt", "fast"] = "nt"
    # Champion gate pool (vibe-quant-clmmc.2): "all" (default) screens every train-passing
    # genome and caps the survivors at top-K by DSR statistic. Forwarded on the argv only
    # as "top_k" (the CLI default is "all"); always stored in the params.
    gate_pool: Literal["top_k", "all"] = "all"
    # GA fitness mode (vibe-quant-clmmc.3): "v2" scores the shrunk MTM t-stat plus a
    # parameter-plateau step (neighbours count as DSR trials). Refused (422, before any run
    # row) with --evaluator fast, --null-gate and --port, whose comparers walk every fitness
    # field. Forwarded and stored only when "v2", so a default launch is unchanged.
    fitness: Literal["v1", "v2"] = "v1"
    # GA exit gene space (vibe-quant-clmmc.4): "extended" also evolves ATR / trailing / safety
    # stops, ATR / risk-reward targets and a time exit. Refused (422, before any run row) without
    # ATR sizing, and with liquidation off above 1x; forwarded and stored only when "extended",
    # so a default launch keeps its stored params and argv unchanged.
    exit_gene_space: Literal["fixed", "extended"] = "fixed"
    # Opt-in timing-null gate (vibe-quant-clmmc.14): champions must beat random re-timings
    # of their own trades. Refused (422) before any run row when the gate could not judge
    # the run; forwarded and stored only when true, so a default launch is unchanged.
    null_gate: StrictBool = False
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
    # ATR sizing (vibe-quant-9sost): omitted = "risk" (the strategy's stop-loss distance);
    # the two ATR fields are valid only with "atr" (checked by ``SizingSettings``, 422).
    # An explicit null is a 422. Forwarded and stored only in ATR mode.
    sizing_mode: Literal["risk", "atr"] | None = None
    sizing_atr_period: int | None = None
    sizing_atr_multiplier: float | None = None
    # Bankroll grid (bead vibe-quant-t8nrv, T8): discovery searches every mode x leverage x
    # cushion cell and each champion records its cell. Excludes the run-level fields above;
    # omitted = discovery runs exactly as without a grid.
    bankroll_grid: BankrollGridRequest | None = None

    @field_validator(
        "population", "generations", "elite_count", "tournament_size",
        "convergence_generations", "eval_windows", "num_seeds", "wfa_oos_step_days",
        "seed_run_id", "seed", "max_workers", "sizing_atr_period", mode="before",
    )  # fmt: skip
    @classmethod
    def _no_bool_counts(cls, value: Any, info: ValidationInfo) -> Any:
        """``true`` is not a count: lax int parsing would read it as 1 (bead vibe-quant-clmmc.1)."""
        if isinstance(value, bool):
            msg = f"{info.field_name} must be an integer, got {value}"
            raise ValueError(msg)
        return value

    @model_validator(mode="before")
    @classmethod
    def _reject_unrepresentable_ints(cls, data: object) -> object:
        """Any integer outside int64 is a 422 naming its field (bead vibe-quant-clmmc.1).

        JSON integers are unbounded in Python; past int64 they overflow SQLite, numpy and
        calendar arithmetic downstream instead of failing here.
        """
        _reject_big_ints(data, "")
        return data

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

    @field_validator("sizing_mode", "sizing_atr_period", "sizing_atr_multiplier", mode="before")
    @classmethod
    def _no_null_sizing_mode(cls, value: Any, info: ValidationInfo) -> Any:
        """An explicit null is an error on its own field, never a silent default."""
        if value is None:
            msg = f"{info.field_name} must not be null (omit it for the default)"
            raise ValueError(msg)
        return value

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

    # Declared after ``_check_bankroll_grid`` on purpose: the grid's own cells are
    # validated first, so a bad cell names its cell rather than the snapshot cap.
    @model_validator(mode="after")
    def _check_symbol_leverage(self) -> DiscoveryLaunchRequest:
        """L4 (vibe-quant-4oo1i): no launch at a leverage above a symbol's Binance maximum.

        A grid has no run-level leverage: every cell's leverage is checked. Without a
        grid the run's resolved leverage is checked (10.0 by default), so a symbol the
        snapshot does not cover fails closed even when no leverage was given.
        """
        grid = self.bankroll_grid
        if grid is not None:
            leverages = list(grid.leverages)
        else:
            leverages = [self.leverage if self.leverage is not None else DEFAULT_LEVERAGE]
        try:
            for leverage in leverages:
                check_leverage(self.symbols, leverage)
        except SymbolLeverageError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def bankroll_cells(self) -> list[BankrollCell]:
        """The grid's validated cells (empty without a grid)."""
        grid = self.bankroll_grid
        if grid is None:
            return []
        return build_cells(grid.modes, grid.leverages, grid.max_margin_pcts, self.sizing_settings())

    def liquidation_settings(self) -> LiquidationSettings:
        """The run's validated settings (the launch router's single source)."""
        leverage = self.leverage if self.leverage is not None else DEFAULT_LEVERAGE
        return LiquidationSettings(mode=self.liquidation_mode, leverage=leverage)

    def sizing_settings(self) -> SizingSettings:
        """The run's validated sizing (the launch router's single source).

        An omitted field means its default; only the fields explicitly supplied are
        read, so ``SizingSettings().to_params()`` stays empty for a default launch.
        """
        params: dict[str, float | int | str] = {}
        if self.max_position_pct is not None:
            params["max_position_pct"] = self.max_position_pct
        if self.risk_per_trade is not None:
            params["risk_per_trade"] = self.risk_per_trade
        if self.max_margin_pct is not None:
            params["max_margin_pct"] = self.max_margin_pct
        if self.sizing_mode is not None:
            params["sizing_mode"] = self.sizing_mode
        if self.sizing_atr_period is not None:
            params["sizing_atr_period"] = self.sizing_atr_period
        if self.sizing_atr_multiplier is not None:
            params["sizing_atr_multiplier"] = self.sizing_atr_multiplier
        return SizingSettings.from_params(params)


_MAX_PREFLIGHT_BUDGET = 10**9


class DiscoveryPreflightRequest(DiscoveryLaunchRequest):
    """The launch body plus the preflight-only ``no_dsr`` (bead vibe-quant-clmmc.1).

    Same fields and validators as a launch; the launch API itself has no ``no_dsr``
    (API launches always run the DSR gate; ``--no-dsr`` is CLI-only).
    """

    no_dsr: StrictBool = False

    @field_validator("symbols")
    @classmethod
    def _some_symbols(cls, value: list[str]) -> list[str]:
        if not value:
            msg = "symbols must name at least one symbol"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _budget_has_a_dsr_bar(self) -> DiscoveryPreflightRequest:
        # Beyond ~1e15 trials the DSR's E[max Z] (inverse normal CDF of 1 - 1/N) is
        # undefined in floats; 1e9 is far past any real run.
        if self.population * self.generations > _MAX_PREFLIGHT_BUDGET:
            msg = f"population x generations must be at most {_MAX_PREFLIGHT_BUDGET:,}"
            raise ValueError(msg)
        return self


class PowerBar(BaseModel):
    """One DSR Sharpe bar: the smallest annual Sharpe (sqrt-252) with DSR p < 0.05."""

    n_trials: int
    sharpe: float

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def sharpe_display(self) -> float | None:
        return sharpe_to_display(self.sharpe)


class DiscoveryPreflightResponse(BaseModel):
    """The power preflight of a launch (``vibe_quant.discovery.power``). Reporting only."""

    sharpe_basis: Literal["sqrt252"]
    train_days: int
    holdout_days: int
    holdout_t_sharpe_1: float
    budget: int
    prior_n: int | Literal["unknown"]
    bar_a: PowerBar | Literal["DSR off"]
    bar_b: PowerBar | Literal["DSR off"]
    warning: str | None
    line: str
    # "Use full history": the latest first full UTC day of 1m bars among the symbols,
    # clamped to MARK coverage when liquidation is on.
    full_history_start: str


class FullHistoryStartResponse(BaseModel):
    """The "Use full history" start for a symbol set (bead vibe-quant-clmmc.1)."""

    # The latest first full UTC day of 1m bars among the symbols, clamped to MARK
    # coverage when ``mark`` (liquidation on) was asked.
    full_history_start: str


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

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def best_sharpe_display(self) -> float | None:
        return sharpe_to_display(self.best_sharpe)


class RejectionCounts(BaseModel):
    """``--gate-pool all`` counts over EVERY rejected candidate (the rows keep the first 50).

    ``by_stage`` sums to ``total``; ``by_kind`` counts candidates per distinct reason kind,
    so it can sum above ``total``. Stage ``gate_pool_cap`` (kind ``Top-K cap``) = passed
    every gate, left out by the top_k cap.
    """

    total: int
    by_stage: dict[str, int]
    by_kind: dict[str, int]


class DiscoveryResultResponse(BaseModel):
    strategies: list[dict[str, object]]
    # Why a completed run has no strategies (all champions guardrail-rejected)
    reason: str | None = None
    guardrail_rejections: list[dict[str, object]] | None = None
    # --gate-pool all only (None for top_k and legacy runs)
    rejection_counts: RejectionCounts | None = None
    # --null-gate only: the run's notes.timing_null_gate (pass bar ``pct``, ``exposure_max``)
    # that each strategy's ``timing_null`` record is judged against; None when off;
    # ``{"unreadable": <detail>}`` when promote would refuse the policy as unreadable.
    timing_null_gate: dict[str, object] | None = None


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
