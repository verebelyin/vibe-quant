"""``vibe_quant.launch_settings``: the launch-parameter table and settings.

Bead vibe-quant-rabmo.3, step S2. Rows pinned here: R25 (defaults), R27
(the per-entry-point parse order; R27c's partial prefix), R28
(``from_parent_run``), R29 (stored ``parameters`` shapes) and R32 (import
isolation). The entry points adopt the library in S3/S5; this module pins the
table against today's per-entry-point orders. S5b (Q5=A) made them one canonical
field order (balance, liquidation, sizing, leverage_cap, entry_order, fee), so the
rows now differ only at the phase splits.

``PARSE_ORDER`` rows:

- screening ``__main__`` 124-160: balance, liquidation, sizing, fee (no cap check);
- ``create_screening_pipeline`` 579-613: balance, liquidation, sizing, cap check,
  entry refusal, then fee (real branch only, after ``resolve_parent_fill_ticks``);
- direct ``_create_venue_config(run_config, ...)`` 1118-1133: balance, liquidation, fee;
- validation ``run`` 388-440 / ``run_walk_forward`` 629-664: liquidation, sizing, cap
  check around detail/clamp/latency, then balance, fee;
- WFA/CV ``_resolve``: the recorded balance first, then liquidation, sizing, cap, fee.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vibe_quant.balance_settings import BalanceConfigError
from vibe_quant.db.state_manager import StateManager
from vibe_quant.fees import FeeSchedule
from vibe_quant.launch_settings import (
    PARSE_ORDER,
    LaunchSettings,
    _run_params,
    iter_fields,
    parent_effective_balance,
)
from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.sizing_settings import SizingConfigError, SizingSettings

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Params in which every launch key is invalid. ``liquidation_mode`` stays absent
#: (a bad mode would pre-empt the leverage error the table row names) and
#: ``entry_order`` cannot be invalid by design (``str()`` of anything parses).
_ALL_INVALID: dict[str, object] = {
    "initial_balance": 0,
    "leverage": 0,
    "max_position_pct": 0,
    "risk_per_trade": -1.0,
    "max_margin_pct": 2.0,
    "fee_tier": "VIP99",
}

#: The first field of a row -> the error class and exact message it raises.
_FIRST_FIELD_ERROR: dict[str, type[ValueError]] = {
    "balance": BalanceConfigError,
    "fee": ValueError,
    "liquidation": LiquidationConfigError,
    "sizing": SizingConfigError,
}

_FIRST_FIELD_MESSAGE: dict[str, str] = {
    "balance": "initial_balance must be a whole number in [1, 1000000000], got 0",
    "fee": "unknown fee tier 'VIP99'; known tiers: ['VIP0']",
    "liquidation": "leverage must be a whole number in [1, 125], got 0",
    "sizing": "max_margin_pct must be a number in (0, 1], got 2.0",
}

#: The first field of every row is one whose invalid value raises; ``entry_order``
#: accepts any string and ``leverage_cap`` is only reachable after its two inputs,
#: so neither can be a row's first field.
_FIRST_FIELD_IS_INVALIDATABLE = frozenset(_FIRST_FIELD_MESSAGE)


def _open(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed(
    tmp_path: Path,
    name: str,
    *,
    parameters: dict[str, object] | None = None,
    sweep: list[dict[str, object]] | None = None,
    result: dict[str, object] | None = None,
) -> tuple[Path, int]:
    """A state DB with one screening run plus the rows the case needs."""
    db = tmp_path / f"{name}.db"
    state = StateManager(db)
    try:
        strategy_id = state.create_strategy("stub", {"timeframe": "4h", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-03-01",
            parameters=parameters or {},
        )
        if sweep is not None:
            state.save_sweep_results_batch(run_id, sweep)
        if result is not None:
            state.save_backtest_result(run_id, result)
    finally:
        state.close()
    return db, run_id


# ------------------------------------------------- R27: the parse-order table


@pytest.mark.parametrize("entry_point", sorted(PARSE_ORDER))
def test_parse_order_table_matches_code(entry_point: str) -> None:
    """Each row starts with the field today's code parses first, same error."""
    order = PARSE_ORDER[entry_point]
    assert order[0] in _FIRST_FIELD_IS_INVALIDATABLE, (
        f"{entry_point}: the row starts with {order[0]!r}, which no invalid "
        "parameter can make raise, so the order cannot be pinned this way"
    )
    with pytest.raises(_FIRST_FIELD_ERROR[order[0]]) as excinfo:
        list(iter_fields(_ALL_INVALID, order))
    assert str(excinfo.value) == _FIRST_FIELD_MESSAGE[order[0]]


#: Params in which every launch key is valid, so a whole row parses.
_VALID: dict[str, object] = {
    "initial_balance": 25000,
    "liquidation_mode": "isolated",
    "leverage": 3,
    "max_position_pct": 1.0,
    "risk_per_trade": 0.01,
    "max_margin_pct": 0.3,
    "entry_order": "market",
    "fee_tier": "VIP0",
    "bnb_discount": True,
}


@pytest.mark.parametrize("entry_point", sorted(PARSE_ORDER))
def test_iter_fields_emits_each_row_in_order(entry_point: str) -> None:
    """The table is what ``iter_fields`` walks: one step per field, in order."""
    order = PARSE_ORDER[entry_point]
    assert [name for name, _ in iter_fields(_VALID, order)] == list(order)


def test_iter_fields_yields_the_parsed_component_values() -> None:
    """Each step yields its component's own value (``leverage_cap`` is a check)."""
    fields = dict(iter_fields(_VALID, PARSE_ORDER["screening_pipeline_pre_tick"]))
    assert fields["balance"] == 25000
    assert fields["liquidation"] == LiquidationSettings("isolated", 3.0)
    assert fields["sizing"] == SizingSettings(1.0, 0.01, 0.3)
    assert fields["entry_order"] == "market"
    assert fields["leverage_cap"] is None


def test_iter_fields_is_lazy_and_stops_where_the_bad_field_is() -> None:
    """R27c: balance + liquidation are parsed before the bad sizing raises.

    The screening CLI prints one line for the balance and two for the liquidation, and
    only then fails on the sizing: exactly 3 stdout lines (S5b's canonical order).
    """
    seen: list[str] = []
    with pytest.raises(SizingConfigError, match="^max_position_pct must be a number in "):
        for name, _ in iter_fields({"max_position_pct": 0}, PARSE_ORDER["screening_cli"]):
            seen.append(name)
    assert seen == ["balance", "liquidation"]


def test_iter_fields_parsed_carries_an_earlier_phase() -> None:
    """Validation's two steps: the pre-detail fields are visible to the post-detail one."""
    pre = dict(iter_fields({"initial_balance": 25000}, PARSE_ORDER["validation_pre_detail"]))
    assert set(pre) == {"liquidation", "sizing", "leverage_cap"}
    post = dict(
        iter_fields(
            {"initial_balance": 25000},
            PARSE_ORDER["validation_post_detail"],
            parsed=pre,
        )
    )
    assert post == {"balance": 25000, "fee": FeeSchedule()}


def test_leverage_cap_uses_a_carried_liquidation() -> None:
    """A carried liquidation reaches the ``leverage_cap`` step (R40's separate count)."""
    with pytest.raises(SizingConfigError) as excinfo:
        list(
            iter_fields(
                {"max_position_pct": 20.0},
                ("sizing", "leverage_cap"),
                parsed={"liquidation": LiquidationSettings(leverage=3)},
            )
        )
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (3.0)"


def test_leverage_cap_without_a_carried_liquidation_uses_the_default() -> None:
    """No carried liquidation and no liquidation step: the default 10x is checked."""
    assert dict(iter_fields({}, ("leverage_cap",))) == {"leverage_cap": None}


# ------------------------------------------------------ R25: the default settings


@pytest.mark.parametrize("entry_point", sorted(PARSE_ORDER))
def test_from_params_defaults_match_launchsettings(entry_point: str) -> None:
    """R25: ``from_params({}, any order) == LaunchSettings()`` with literal values."""
    launch = LaunchSettings.from_params({}, PARSE_ORDER[entry_point])
    assert launch == LaunchSettings()
    assert (launch.fee_schedule.tier, launch.fee_schedule.bnb_discount) == ("VIP0", False)
    assert (launch.liquidation.mode, launch.liquidation.leverage) == ("off", 10.0)
    assert (
        launch.sizing.max_position_pct,
        launch.sizing.risk_per_trade,
        launch.sizing.max_margin_pct,
    ) == (0.5, 0.02, None)
    assert launch.starting_balance == 1000
    assert launch.entry_order == "market"


def test_from_params_forwards_each_field() -> None:
    """Two unrelated parameter sets, both read through the public entry point."""
    first = LaunchSettings.from_params(_VALID, PARSE_ORDER["screening_pipeline_pre_tick"])
    assert first == LaunchSettings(
        FeeSchedule(),
        LiquidationSettings("isolated", 3.0),
        SizingSettings(1.0, 0.01, 0.3),
        25000,
        "market",
    )

    second = LaunchSettings.from_params(
        {
            "fee_tier": "VIP0",
            "bnb_discount": True,
            "initial_balance": 1,
            "max_position_pct": 125.0,
            "leverage": 125,
            "risk_per_trade": 0.5,
        },
        PARSE_ORDER["screening_cli"],
    )
    assert second == LaunchSettings(
        FeeSchedule("VIP0", True),
        LiquidationSettings("off", 125.0),
        SizingSettings(125.0, 0.5, None),
        1,
        "market",
    )


def test_from_fields_missing_fields_take_their_defaults() -> None:
    assert LaunchSettings.from_fields({}) == LaunchSettings()


def test_from_fields_forwards_the_values_it_is_given() -> None:
    fee = FeeSchedule("VIP0", True)
    liquidation = LiquidationSettings("cross", 5)
    sizing = SizingSettings(2.0, 0.1, 0.5)
    launch = LaunchSettings.from_fields(
        {
            "fee": fee,
            "liquidation": liquidation,
            "sizing": sizing,
            "balance": 25,
            "entry_order": "limit",
            "leverage_cap": None,
        }
    )
    assert launch == LaunchSettings(fee, liquidation, sizing, 25, "limit")


def test_post_init_validates_the_balance_and_normalises_it() -> None:
    with pytest.raises(BalanceConfigError) as excinfo:
        LaunchSettings(starting_balance=0)
    assert str(excinfo.value) == (
        "initial_balance must be a whole number in [1, 1000000000], got 0"
    )
    normalized = LaunchSettings(starting_balance=25000.0)
    assert normalized.starting_balance == 25000
    assert isinstance(normalized.starting_balance, int)


def test_post_init_checks_every_cap_against_the_leverage() -> None:
    """Equality is allowed (a cap exactly at the leverage is marginable)."""
    at_the_edge = LaunchSettings(
        liquidation=LiquidationSettings(leverage=125),
        sizing=SizingSettings(max_position_pct=125.0),
    )
    assert at_the_edge.sizing.max_position_pct == 125.0
    with pytest.raises(SizingConfigError) as excinfo:
        LaunchSettings(
            liquidation=LiquidationSettings(leverage=3),
            sizing=SizingSettings(max_position_pct=20),
        )
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (3.0)"


# --------------------------------------------------- R28: from_parent_run


def _sweep_row(starting_balance: object = None, sharpe: float = 1.0) -> dict[str, object]:
    row: dict[str, object] = {"parameters": {"rsi.period": 14}, "sharpe_ratio": sharpe}
    if starting_balance is not None:
        row["starting_balance"] = starting_balance
    return row


def test_from_parent_run_ignores_the_params_balance(tmp_path: Path) -> None:
    """R28: params say "abc", the sweep row says 25000 -> 25000, no error."""
    db, run_id = _seed(
        tmp_path,
        "parent_balance",
        parameters={"initial_balance": "abc"},
        sweep=[_sweep_row(25000.0)],
    )
    conn = _open(db)
    try:
        launch = LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()
    assert launch.starting_balance == 25000
    assert launch.entry_order == "market"


def test_from_parent_run_reads_the_balance_before_the_stored_settings(tmp_path: Path) -> None:
    """A bad stored leverage cannot pre-empt a missing recorded balance."""
    db, run_id = _seed(tmp_path, "parent_order", parameters={"leverage": 0})
    conn = _open(db)
    try:
        with pytest.raises(ValueError) as excinfo:
            LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()
    assert str(excinfo.value) == f"run {run_id} has no recorded starting balance"


def test_from_parent_run_without_any_row_raises(tmp_path: Path) -> None:
    db, _ = _seed(tmp_path, "parent_missing")
    conn = _open(db)
    try:
        with pytest.raises(ValueError) as excinfo:
            LaunchSettings.from_parent_run(conn, 999)
    finally:
        conn.close()
    assert str(excinfo.value) == "run 999 has no recorded starting balance"


def test_from_parent_run_null_result_means_the_default(tmp_path: Path) -> None:
    """R28: a NULL recorded balance (an older row) means 1000."""
    db, run_id = _seed(tmp_path, "parent_null", result={})
    conn = _open(db)
    try:
        launch = LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()
    assert launch.starting_balance == 1000


def test_from_parent_run_conflicting_sweep_values_raise(tmp_path: Path) -> None:
    db, run_id = _seed(
        tmp_path,
        "parent_conflict",
        sweep=[_sweep_row(10000.0), _sweep_row(5000.0)],
    )
    conn = _open(db)
    try:
        with pytest.raises(ValueError, match="conflicting sweep starting_balance"):
            LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()


def test_from_parent_run_parses_json_string_parameters(tmp_path: Path) -> None:
    """R29/R28: the stored parameters are a JSON string; the row balance wins."""
    db, run_id = _seed(tmp_path, "parent_json", result={"starting_balance": 4000.0})
    conn = _open(db)
    try:
        conn.execute(
            "UPDATE backtest_runs SET parameters = ? WHERE id = ?",
            (
                '{"liquidation_mode": "isolated", "leverage": 3, '
                '"max_position_pct": 1.0, "fee_tier": "VIP0", "bnb_discount": true, '
                '"entry_order": "limit", "initial_balance": 999}',
                run_id,
            ),
        )
        conn.commit()
        launch = LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()
    assert launch == LaunchSettings(
        FeeSchedule("VIP0", True),
        LiquidationSettings("isolated", 3.0),
        SizingSettings(1.0),
        4000,
        "market",
    )


# ------------------------------------------- R28: the WFA/CV runners resolve


def test_wfa_and_cv_runners_resolve_through_from_parent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R28: both runners hold exactly what ``from_parent_run`` resolves, once.

    The parent's params say ``initial_balance: "abc"`` (never parsed) and the sweep
    row says 25000, so the literal values pin the sweep-row balance, the ``"market"``
    entry and the stored leverage.
    """
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    db, run_id = _seed(
        tmp_path,
        "runner_launch",
        parameters={
            "initial_balance": "abc",
            "liquidation_mode": "isolated",
            "leverage": 3,
        },
        sweep=[_sweep_row(25000.0)],
    )
    conn = _open(db)
    try:
        expected = LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()

    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    wfa = NTWFARunner(run_id=run_id, db_path=db)
    cv = NTPurgedKFoldRunner(run_id=run_id, db_path=db)

    for runner in (wfa, cv):
        assert runner._launch == expected  # noqa: SLF001
        assert runner._launch.starting_balance == 25000  # noqa: SLF001
        assert runner._launch.entry_order == "market"  # noqa: SLF001
        assert runner._launch.liquidation == LiquidationSettings("isolated", 3.0)  # noqa: SLF001


@pytest.mark.parametrize("runner_name", ["wfa", "cv"])
def test_runners_read_the_balance_before_the_stored_settings(
    tmp_path: Path, runner_name: str
) -> None:
    """R28 precedence: a bad stored leverage cannot pre-empt a missing balance."""
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    build = NTWFARunner if runner_name == "wfa" else NTPurgedKFoldRunner
    db, run_id = _seed(tmp_path, f"runner_order_{runner_name}", parameters={"leverage": 0})
    with pytest.raises(ValueError) as excinfo:
        build(run_id=run_id, db_path=db)
    assert str(excinfo.value) == f"run {run_id} has no recorded starting balance"


# ------------------------------------------------------- R29: params shapes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, {}),
        ("null", {}),
        ("not json", {}),
        ("[1, 2]", {}),
        ("42", {}),
        ([1, 2], {}),
        (42, {}),
        ('{"initial_balance": 25000}', {"initial_balance": 25000}),
        ('{"max_position_pct": null}', {"max_position_pct": None}),
        ({"leverage": 3}, {"leverage": 3}),
    ],
)
def test_run_params_shapes(raw: object, expected: dict[str, object]) -> None:
    assert _run_params(raw) == expected


@pytest.mark.parametrize("raw", ["null", "not json", "[1, 2]", 42])
def test_from_parent_run_reads_unusable_parameters_as_defaults(tmp_path: Path, raw: object) -> None:
    """R29 for WFA/CV: JSON null, bad JSON, a list and a number all mean ``{}``.

    (The column is NOT NULL, so "absent" cannot be stored; :func:`_run_params`
    covers ``None`` above.)
    """
    db, run_id = _seed(tmp_path, "parent_shape", result={"starting_balance": 4000.0})
    conn = _open(db)
    try:
        conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", (raw, run_id))
        conn.commit()
        launch = LaunchSettings.from_parent_run(conn, run_id)
    finally:
        conn.close()
    assert launch == LaunchSettings(starting_balance=4000)


# ------------------------------------------------------- R32: import isolation


def test_import_isolation_in_a_fresh_process() -> None:
    """R32: no ``nautilus_trader`` and no ``vibe_quant.liquidation.*`` on import."""
    code = textwrap.dedent(
        """
        import sys

        import vibe_quant.launch_settings  # noqa: F401

        leaked = sorted(
            name
            for name in sys.modules
            if name == "nautilus_trader"
            or name.startswith("nautilus_trader.")
            or name == "vibe_quant.liquidation"
            or name.startswith("vibe_quant.liquidation.")
        )
        print(leaked)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout + completed.stderr


def test_both_overfitting_runners_use_the_shared_helpers() -> None:
    """The moved functions are one object: no copy survived in either runner.

    ``getattr(mod, name, shared)`` accepts "absent" (the runners import neither name
    any more) and still fails on a re-exported copy. The settings each runner forwards
    to ``NTScreeningRunner`` are pinned by
    ``test_wfa_and_cv_runners_resolve_through_from_parent_run`` above and by
    ``test_overfitting_liquidation.py``.
    """
    import vibe_quant.overfitting.nt_cv_runner as nt_cv
    import vibe_quant.overfitting.nt_runner as nt_wfa

    for module in (nt_wfa, nt_cv):
        assert (
            getattr(module, "parent_effective_balance", parent_effective_balance)
            is parent_effective_balance
        )
        assert getattr(module, "_run_params", _run_params) is _run_params
