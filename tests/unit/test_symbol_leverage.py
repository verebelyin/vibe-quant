"""The ONE rule for a symbol's Binance maximum leverage (bead vibe-quant-4oo1i).

Every launch refuses a leverage above the coin's maximum, in the bracket snapshot:
screening/validation (``parameters.leverage``), discovery (``leverage`` / the deprecated
``liquidation_leverage`` and every bankroll-grid leverage) and the discovery CLI
(``--leverage`` / ``--liquidation-leverage`` / ``--bankroll-leverages``). A symbol the
snapshot does not cover fails closed too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from vibe_quant.api.schemas.backtest import BacktestLaunchRequest
from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest
from vibe_quant.data.catalog import INSTRUMENT_CONFIGS
from vibe_quant.discovery import __main__ as cli
from vibe_quant.symbol_leverage import SymbolLeverageError, check_leverage, max_leverage

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DATE = "2026-10-12"
# The first bracket's max_leverage of each snapshot symbol = the coin maximum.
COIN_MAXIMA = {"BTCUSDT": 150, "ETHUSDT": 150, "SOLUSDT": 100, "BNBUSDT": 75}
MISSING_SYMBOL_MESSAGE = "no Binance leverage bracket for XRPUSDT (snapshot 2026-10-12)"


def _backtest_body(**overrides: object) -> dict[str, object]:
    return {
        "strategy_id": 1,
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": {"initial_balance": 1000},
        **overrides,
    }


# ---------------------------------------------------------------------------
# L1 / L2 -- the module rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("symbol", "expected"), sorted(COIN_MAXIMA.items()))
def test_max_leverage_is_the_coin_maximum(symbol: str, expected: int) -> None:
    assert max_leverage(symbol) == expected
    assert type(max_leverage(symbol)) is int


def test_max_leverage_of_an_unknown_symbol_fails_closed() -> None:
    with pytest.raises(SymbolLeverageError) as exc:
        max_leverage("XRPUSDT")
    assert str(exc.value) == MISSING_SYMBOL_MESSAGE


@pytest.mark.parametrize(("symbol", "leverage"), [("BNBUSDT", 75), ("SOLUSDT", 100)])
def test_check_leverage_at_the_maximum_is_allowed(symbol: str, leverage: int) -> None:
    """The equality edge: the maximum itself is launchable."""
    assert check_leverage([symbol], leverage) is None
    assert check_leverage([symbol], float(leverage)) is None


@pytest.mark.parametrize(
    ("symbol", "leverage", "message"),
    [
        ("BNBUSDT", 76, "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)"),
        ("SOLUSDT", 101, "leverage 101x exceeds the Binance maximum for SOLUSDT (100x)"),
    ],
)
def test_check_leverage_one_above_the_maximum_is_refused(
    symbol: str, leverage: int, message: str
) -> None:
    with pytest.raises(SymbolLeverageError) as exc:
        check_leverage([symbol], leverage)
    assert str(exc.value) == message


def test_check_leverage_names_each_offending_symbol_per_case() -> None:
    """A hard-coded symbol in the message would survive a single-symbol test."""
    with pytest.raises(SymbolLeverageError) as sol:
        check_leverage(["SOLUSDT"], 101)
    assert "SOLUSDT" in str(sol.value) and "BNBUSDT" not in str(sol.value)
    with pytest.raises(SymbolLeverageError) as bnb:
        check_leverage(["BNBUSDT"], 76)
    assert "BNBUSDT" in str(bnb.value) and "SOLUSDT" not in str(bnb.value)


def test_check_leverage_names_the_first_offending_symbol_in_order() -> None:
    with pytest.raises(SymbolLeverageError) as exc:
        check_leverage(["SOLUSDT", "BNBUSDT"], 76)
    assert str(exc.value) == "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)"
    with pytest.raises(SymbolLeverageError) as exc2:
        check_leverage(["BNBUSDT", "SOLUSDT"], 101)
    assert str(exc2.value) == "leverage 101x exceeds the Binance maximum for BNBUSDT (75x)"


def test_importing_the_rule_never_imports_the_liquidation_package() -> None:
    """Import isolation: a mode-off launch must not pull ``vibe_quant.liquidation``."""
    code = (
        "import sys\n"
        "from vibe_quant.api.schemas.backtest import BacktestLaunchRequest\n"
        "BacktestLaunchRequest(**{"
        "'strategy_id': 1, 'symbols': ['BTCUSDT'], 'timeframe': '4h',"
        "'start_date': '2025-01-01', 'end_date': '2025-02-01',"
        "'parameters': {'initial_balance': 1000}})\n"
        "from vibe_quant.symbol_leverage import max_leverage\n"
        "assert max_leverage('BNBUSDT') == 75\n"
        "leaked = [m for m in sys.modules"
        " if m == 'vibe_quant.liquidation' or m.startswith('vibe_quant.liquidation.')]\n"
        "assert not leaked, sorted(leaked)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_catalog_max_leverage_matches_the_snapshot() -> None:
    """One truth: the stale INSTRUMENT_CONFIGS values are corrected to the snapshot."""
    assert INSTRUMENT_CONFIGS
    for symbol, config in INSTRUMENT_CONFIGS.items():
        assert config["max_leverage"] == max_leverage(symbol), symbol


def test_catalog_covers_every_snapshot_symbol() -> None:
    assert set(INSTRUMENT_CONFIGS) == set(COIN_MAXIMA)


# ---------------------------------------------------------------------------
# L3 -- screening / validation launches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", sorted(COIN_MAXIMA))
def test_backtest_launch_default_leverage_passes_on_every_coin(symbol: str) -> None:
    request = BacktestLaunchRequest(**_backtest_body(symbols=[symbol]))
    assert request.symbols == [symbol]


def test_backtest_launch_refuses_leverage_above_the_coin_maximum() -> None:
    with pytest.raises(ValidationError) as exc:
        BacktestLaunchRequest(
            **_backtest_body(symbols=["BNBUSDT"], parameters={"leverage": 76})
        )
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in str(exc.value)


def test_backtest_launch_at_the_coin_maximum_is_allowed() -> None:
    request = BacktestLaunchRequest(
        **_backtest_body(symbols=["BNBUSDT"], parameters={"leverage": 75})
    )
    assert request.parameters["leverage"] == 75


def test_backtest_launch_symbol_missing_from_the_snapshot_is_refused() -> None:
    with pytest.raises(ValidationError) as exc:
        BacktestLaunchRequest(**_backtest_body(symbols=["XRPUSDT"]))
    assert MISSING_SYMBOL_MESSAGE in str(exc.value)


# ---------------------------------------------------------------------------
# L4 -- discovery launches
# ---------------------------------------------------------------------------


def test_discovery_launch_default_leverage_passes() -> None:
    request = DiscoveryLaunchRequest()
    assert request.symbols == ["BTCUSDT"]


def test_discovery_launch_refuses_leverage_above_the_coin_maximum() -> None:
    with pytest.raises(ValidationError) as exc:
        DiscoveryLaunchRequest(symbols=["SOLUSDT"], leverage=101)
    assert "leverage 101x exceeds the Binance maximum for SOLUSDT (100x)" in str(exc.value)
    with pytest.raises(ValidationError) as exc2:
        DiscoveryLaunchRequest(symbols=["SOLUSDT"], liquidation_leverage=101)
    assert "leverage 101x exceeds the Binance maximum for SOLUSDT (100x)" in str(exc2.value)


def test_discovery_launch_at_the_coin_maximum_is_allowed() -> None:
    assert DiscoveryLaunchRequest(symbols=["SOLUSDT"], leverage=100).leverage == 100.0


def test_discovery_launch_refuses_a_grid_leverage_above_the_coin_maximum() -> None:
    grid = {"modes": ["isolated"], "leverages": [76], "max_margin_pcts": [0.5]}
    with pytest.raises(ValidationError) as exc:
        DiscoveryLaunchRequest(symbols=["BNBUSDT"], bankroll_grid=grid)
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in str(exc.value)


def test_discovery_launch_refuses_a_later_leverage_of_a_multi_leverage_grid() -> None:
    """EVERY cell is checked: a grid whose first leverage is fine must refuse a later one.

    The single-value grid above lets a ``leverages[:1]`` in ``_check_symbol_leverage``
    survive (rev-gr-maxlev-1); this pins the whole list.
    """
    grid = {"modes": ["isolated"], "leverages": [5, 76], "max_margin_pcts": [0.5]}
    with pytest.raises(ValidationError) as exc:
        DiscoveryLaunchRequest(symbols=["BNBUSDT"], bankroll_grid=grid)
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in str(exc.value)

    long_grid = {"modes": ["isolated"], "leverages": [1, 5, 76], "max_margin_pcts": [0.5]}
    with pytest.raises(ValidationError) as exc2:
        DiscoveryLaunchRequest(symbols=["BNBUSDT"], bankroll_grid=long_grid)
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in str(exc2.value)


def test_discovery_launch_refuses_a_later_grid_leverage_on_the_second_symbol() -> None:
    """Both the later leverage AND the later symbol are reached, in the grid branch."""
    grid = {"modes": ["isolated"], "leverages": [5, 76], "max_margin_pcts": [0.5]}
    with pytest.raises(ValidationError) as exc:
        DiscoveryLaunchRequest(symbols=["BTCUSDT", "BNBUSDT"], bankroll_grid=grid)
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in str(exc.value)


def test_discovery_launch_accepts_a_grid_leverage_at_the_coin_maximum() -> None:
    grid = {"modes": ["isolated"], "leverages": [75], "max_margin_pcts": [0.5]}
    cells = DiscoveryLaunchRequest(symbols=["BNBUSDT"], bankroll_grid=grid).bankroll_cells()
    assert [cell.leverage for cell in cells] == [75.0]


def test_discovery_launch_symbol_missing_from_the_snapshot_is_refused() -> None:
    with pytest.raises(ValidationError) as exc:
        DiscoveryLaunchRequest(symbols=["XRPUSDT"])
    assert MISSING_SYMBOL_MESSAGE in str(exc.value)


# ---------------------------------------------------------------------------
# L5 / L6 -- the discovery CLI
# ---------------------------------------------------------------------------


def _cli_args(argv: list[str]) -> tuple[object, object]:
    parser = cli.build_parser()
    return parser, parser.parse_args(["--run-id", "1", *argv])


def test_cli_refuses_leverage_above_the_coin_maximum(capsys: pytest.CaptureFixture[str]) -> None:
    parser, args = _cli_args(["--symbols", "SOLUSDT", "--leverage", "101"])
    with pytest.raises(SystemExit) as exc:
        cli.check_launch_leverage(parser, args)
    assert exc.value.code == 2
    assert "leverage 101x exceeds the Binance maximum for SOLUSDT (100x)" in capsys.readouterr().err


def test_cli_refuses_the_deprecated_leverage_alias(capsys: pytest.CaptureFixture[str]) -> None:
    parser, args = _cli_args(["--symbols", "BNBUSDT", "--liquidation-leverage", "76"])
    with pytest.raises(SystemExit) as exc:
        cli.check_launch_leverage(parser, args)
    assert exc.value.code == 2
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in capsys.readouterr().err


def test_cli_refuses_a_bankroll_leverage_above_the_coin_maximum(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser, args = _cli_args(
        [
            "--symbols", "BNBUSDT",
            "--bankroll-modes", "isolated",
            "--bankroll-leverages", "1,76",
            "--bankroll-max-margin-pcts", "0.5",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        cli.check_launch_leverage(parser, args)
    assert exc.value.code == 2
    assert "leverage 76x exceeds the Binance maximum for BNBUSDT (75x)" in capsys.readouterr().err


def test_cli_symbol_missing_from_the_snapshot_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser, args = _cli_args(["--symbols", "XRPUSDT"])
    with pytest.raises(SystemExit) as exc:
        cli.check_launch_leverage(parser, args)
    assert exc.value.code == 2
    assert MISSING_SYMBOL_MESSAGE in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--symbols", "BTCUSDT", "--leverage", "200"],
        ["--symbols", "SOLUSDT", "--leverage", "3", "--liquidation-leverage", "101"],
        ["--symbols", "BNBUSDT", "--bankroll-modes", "isolated",
         "--bankroll-leverages", "200", "--bankroll-max-margin-pcts", "0.5"],
    ],
)
def test_cli_leaves_a_value_the_run_settings_refuse_to_them(argv: list[str]) -> None:
    """L7: an out-of-range value or disagreeing aliases stay a failed run (exit 1), not exit 2."""
    parser, args = _cli_args(argv)
    assert cli.check_launch_leverage(parser, args) is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--symbols", "BTCUSDT"],
        ["--symbols", "BTCUSDT,ETHUSDT", "--leverage", "125"],
        ["--symbols", "SOLUSDT", "--leverage", "100"],
        [
            "--symbols", "BNBUSDT",
            "--bankroll-modes", "isolated,cross",
            "--bankroll-leverages", "1,5",
            "--bankroll-max-margin-pcts", "0.25",
        ],
    ],
)
def test_cli_accepts_leverage_within_every_coin_maximum(argv: list[str]) -> None:
    parser, args = _cli_args(argv)
    assert cli.check_launch_leverage(parser, args) is None
