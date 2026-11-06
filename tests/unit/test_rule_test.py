"""scripts/agents/rule_test.py runs a DSL YAML through the shared engine path, nothing private."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("rule_test", ROOT / "scripts" / "agents" / "rule_test.py")
assert _spec is not None and _spec.loader is not None
rule_test = importlib.util.module_from_spec(_spec)
sys.modules["rule_test"] = rule_test
_spec.loader.exec_module(rule_test)

RULE = """
name: rsi_dip_test
description: "1m RSI(14) crosses below {thr} -> long, exit RSI crosses above 50"
version: 1
timeframe: 1m
additional_timeframes: []
indicators:
  rsi:
    type: RSI
    period: 14
    source: close
entry_conditions:
  long:
    - "rsi crosses_below {thr}"
exit_conditions:
  long:
    - "rsi crosses_above 50"
stop_loss:
  type: fixed_pct
  percent: 50.0
take_profit:
  type: fixed_pct
  percent: 100.0
{extra}
"""


def _write(tmp_path: Path, thr: int = 30, extra: str = "") -> Path:
    path = tmp_path / f"rule_{thr}.yaml"
    path.write_text(RULE.format(thr=thr, extra=extra))
    return path


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("VIBE_QUANT_DB", str(tmp_path / "state.db"))


def _direct_nt(yaml_path: Path, symbol: str, start: str, end: str, **launch: Any) -> Any:
    """The shared runner called by hand, with no rule_test code in the path."""
    import yaml

    from vibe_quant.launch_settings import LaunchSettings
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.sizing_settings import SizingSettings

    dsl = yaml.safe_load(yaml_path.read_text())
    settings = LaunchSettings(
        liquidation=LiquidationSettings(mode="off", leverage=launch.get("leverage", 10)),
        sizing=SizingSettings(
            max_position_pct=launch.get("cap", 0.5), risk_per_trade=launch.get("risk", 0.02)
        ),
    )
    return NTScreeningRunner(dsl, [symbol], start, end, launch=settings)({})


@pytest.mark.parametrize(
    ("symbol", "start", "end", "thr", "trades"),
    [("BTCUSDT", "2026-03-02", "2026-03-04", 30, 60), ("ETHUSDT", "2026-03-09", "2026-03-11", 25, 33)],
)
def test_nt_tier_equals_the_shared_runner(
    tmp_path: Path, symbol: str, start: str, end: str, thr: int, trades: int
) -> None:
    path = _write(tmp_path, thr)
    got = rule_test.run_rule(path, [symbol], start, end, tier="nt")
    direct = _direct_nt(path, symbol, start, end)
    assert got["total_trades"] == direct.total_trades == trades
    assert got["total_return"] == direct.total_return
    assert got["win_rate"] == direct.win_rate
    assert got["sharpe_ratio"] == direct.sharpe_ratio


def test_fast_tier_matches_nt_fill_for_fill_metrics(tmp_path: Path) -> None:
    path = _write(tmp_path, 30)
    nt = rule_test.run_rule(path, ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="nt")
    fast = rule_test.run_rule(path, ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="fast")
    for key in ("total_trades", "total_return", "sharpe_ratio", "profit_factor", "max_drawdown"):
        assert fast[key] == nt[key], key
    assert fast["tier"] == "fast" and nt["tier"] == "nt"


def test_sizing_flags_reach_the_engine(tmp_path: Path) -> None:
    path = _write(tmp_path, 30)
    base = rule_test.run_rule(path, ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="nt")
    big = rule_test.run_rule(
        path, ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="nt", max_position_pct=1.0, risk_per_trade=0.5
    )
    direct = _direct_nt(path, "BTCUSDT", "2026-03-02", "2026-03-04", cap=1.0, risk=0.5)
    assert big["total_return"] == direct.total_return
    assert big["total_return"] != base["total_return"]
    assert big["total_trades"] == base["total_trades"]


@pytest.mark.parametrize(("leverage", "cap", "ok"), [(1, 1.0, True), (1, 1.5, False), (2, 1.5, True), (2, 2.5, False)])
def test_leverage_reaches_the_launch_check(tmp_path: Path, leverage: int, cap: float, ok: bool) -> None:
    from vibe_quant.sizing_settings import SizingConfigError

    path = _write(tmp_path, 30)
    args = (path, ["BTCUSDT"], "2026-03-02", "2026-03-03")
    kwargs: dict[str, Any] = {"tier": "nt", "leverage": leverage, "max_position_pct": cap, "risk_per_trade": 0.5}
    if ok:
        assert rule_test.run_rule(*args, **kwargs)["total_trades"] >= 0
    else:
        with pytest.raises(SizingConfigError):
            rule_test.run_rule(*args, **kwargs)


def test_fast_tier_refuses_a_rule_it_cannot_replay(tmp_path: Path) -> None:
    path = _write(tmp_path, 30, extra="max_bars_in_trade: 60")
    with pytest.raises(rule_test.TierUnsupported, match="max_bars_in_trade"):
        rule_test.run_rule(path, ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="fast")
    # the same rule still runs on NT
    assert rule_test.run_rule(path, ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="nt")["total_trades"] >= 1


def test_cli_prints_json_and_never_uses_the_real_state_db(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert os.environ["VIBE_QUANT_DB"] == str(tmp_path / "state.db")  # a DB is already pinned: main must replace it
    path = _write(tmp_path, 30)
    code = rule_test.main(
        [str(path), "--symbols", "BTCUSDT", "--start", "2026-03-02", "--end", "2026-03-04", "--tier", "nt"]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["total_trades"] == 60
    assert os.environ["VIBE_QUANT_DB"].startswith("/tmp/rule-test-")


def test_cli_exit_2_when_fast_tier_unsupported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, 30, extra="max_bars_in_trade: 60")
    code = rule_test.main(
        [str(path), "--symbols", "BTCUSDT", "--start", "2026-03-02", "--end", "2026-03-04", "--tier", "fast"]
    )
    assert code == 2
    assert "use --tier nt" in capsys.readouterr().err


def _fail_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime failure inside a valid window: the shared runner masks it as a -inf result."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    def boom(self: Any, params: Any, start_time: float) -> Any:
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(NTScreeningRunner, "_run_backtest", boom)


def test_failed_engine_run_is_refused_not_reported_as_zero_trades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_the_engine(monkeypatch)
    with pytest.raises(rule_test.RuleRunFailed, match="engine blew up|failed"):
        rule_test.run_rule(_write(tmp_path, 30), ["BTCUSDT"], "2026-03-02", "2026-03-04", tier="nt")


def test_cli_exits_nonzero_without_metrics_when_the_engine_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_the_engine(monkeypatch)
    code = rule_test.main(
        [str(_write(tmp_path, 30)), "--symbols", "BTCUSDT", "--start", "2026-03-02", "--end", "2026-03-04"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.strip() == ""  # no metrics JSON a journal entry could copy
    assert "failed" in captured.err


@pytest.mark.parametrize("tier", ["nt", "fast"])
@pytest.mark.parametrize(("start", "end"), [("2026-03-04", "2026-03-02"), ("2026-03-04", "2026-03-04")])
def test_cli_exits_nonzero_for_reversed_or_empty_dates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tier: str, start: str, end: str
) -> None:
    code = rule_test.main(
        [str(_write(tmp_path, 30)), "--symbols", "BTCUSDT", "--start", start, "--end", end, "--tier", tier]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.strip() == ""
    assert "must be after" in captured.err


def test_a_valid_zero_trade_window_still_reports_metrics(tmp_path: Path) -> None:
    path = _write(tmp_path, 30).with_name("never.yaml")
    path.write_text(RULE.format(thr="0.001", extra=""))
    got = rule_test.run_rule(path, ["BTCUSDT"], "2026-03-02", "2026-03-03", tier="nt")
    assert got["total_trades"] == 0
    assert got["sharpe_ratio"] == 0.0
