"""GA exit genes T1: an extended genome's exit block renders to DSL the compiler runs.

Every exit gene must reach generated code (a field the validators accept but the
compiler never emits is a silent no-op), so each stop/target type is compiled through
the real ``StrategyCompiler`` and the generated config line is asserted literally.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from vibe_quant.discovery.exit_genes import (
    STOP_TYPES,
    TARGET_TYPES,
    ExitBlock,
    ExitRules,
    canonical_block,
    max_bars_range,
    rules_from_dict,
    rules_problems,
    rules_summary,
    rules_to_dict,
)
from vibe_quant.discovery.genome import (
    chromosome_to_dsl,
    chromosome_to_serializable,
    genome_summary,
    serializable_to_chromosome,
    validate_chromosome,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.schema import StrategyDSL


def _gene(period: float = 14.0) -> StrategyGene:
    return StrategyGene(
        indicator_type="RSI",
        parameters={"period": period},
        condition=ConditionType.LT,
        threshold=30.0,
    )


def _chrom(direction: Direction, rules: ExitRules | None) -> StrategyChromosome:
    both = direction == Direction.BOTH
    return StrategyChromosome(
        entry_genes=[_gene()],
        exit_genes=[_gene(10.0)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=direction,
        short_entry_genes=[_gene(9.0)] if both else [],
        short_exit_genes=[_gene(8.0)] if both else [],
        exit_rules=rules,
    )


def _block(
    stop: str, stop_v: float | None, target: str, target_v: float | None, atr: int | None
) -> ExitBlock:
    return canonical_block(
        ExitBlock(
            stop_type=stop,  # type: ignore[arg-type]
            stop_value=stop_v,
            target_type=target,  # type: ignore[arg-type]
            target_value=target_v,
            atr_period=atr,
        )
    )


def _compile(chrom: StrategyChromosome) -> tuple[dict[str, object], str]:
    dsl = chromosome_to_dsl(chrom)
    return dsl, StrategyCompiler().compile(StrategyDSL(**dsl))  # type: ignore[arg-type]


# Two unrelated value sets per type pair so a forwarded value is not matched by luck.
_VALUES = {
    "fixed_pct": (3.0, 7.5),
    "atr_fixed": (2.5, 3.25),
    "atr_trailing": (2.5, 3.25),
    "safety": (None, None),
}
_TARGET_VALUES = {
    "fixed_pct": (4.0, 12.5),
    "atr_fixed": (3.0, 6.5),
    "risk_reward": (2.0, 4.5),
    "none": (None, None),
}
_ATR_PERIODS = (14, 21)


def _expected_stop_lines(prefix: str, stop: str, value: float | None, atr_name: str) -> list[str]:
    """Literal config lines the compiler must emit for a stop (written from the DSL contract)."""
    if stop in ("fixed_pct", "safety"):
        pct = 50.0 if stop == "safety" else value
        return [f'    {prefix}_type: str = "fixed_pct"', f"    {prefix}_percent: float = {pct}"]
    return [
        f'    {prefix}_type: str = "{stop}"',
        f"    {prefix}_atr_multiplier: float = {value}",
        f'    {prefix}_indicator: str = "{atr_name}"',
    ]


def _expected_target_lines(
    prefix: str, target: str, value: float | None, atr_name: str
) -> list[str]:
    if target in ("fixed_pct", "none"):
        pct = 100.0 if target == "none" else value
        return [f'    {prefix}_type: str = "fixed_pct"', f"    {prefix}_percent: float = {pct}"]
    if target == "atr_fixed":
        return [
            f'    {prefix}_type: str = "atr_fixed"',
            f"    {prefix}_atr_multiplier: float = {value}",
            f'    {prefix}_indicator: str = "{atr_name}"',
        ]
    return [f'    {prefix}_type: str = "risk_reward"', f"    {prefix}_risk_reward: float = {value}"]


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize(("stop", "target"), list(itertools.product(STOP_TYPES, TARGET_TYPES)))
def test_every_gene_reaches_generated_code(stop: str, target: str, variant: int) -> None:
    """R3: each stop x target type compiles to its exact config line and runtime hook."""
    atr = _ATR_PERIODS[variant]
    block = _block(stop, _VALUES[stop][variant], target, _TARGET_VALUES[target][variant], atr)
    max_bars = (12, 37)[variant]
    dsl, src = _compile(
        _chrom(Direction.LONG, ExitRules(long=block, short=None, max_bars=max_bars))
    )

    # risk_reward under a safety stop is canonicalised to none (R4b): expect the none lines
    eff_target = "none" if (stop, target) == ("safety", "risk_reward") else target
    eff_target_v = None if eff_target == "none" else _TARGET_VALUES[target][variant]
    expected = _expected_stop_lines("stop_loss", stop, _VALUES[stop][variant], "exit_atr")
    expected += _expected_target_lines("take_profit", eff_target, eff_target_v, "exit_atr")
    needs_atr = stop in ("atr_fixed", "atr_trailing") or eff_target == "atr_fixed"
    if needs_atr:
        expected.append(f"    exit_atr_period: int = {atr}")
        assert dsl["indicators"]["exit_atr"] == {"type": "ATR", "period": atr}  # type: ignore[index]
    else:
        assert "exit_atr" not in dsl["indicators"]  # type: ignore[operator]
        assert "exit_atr_period" not in src
    for line in expected:
        assert line in src, line
    assert ("self._update_trailing_stop(bar)" in src) is (stop == "atr_trailing")
    assert f"if self._bars_in_trade >= {max_bars}:" in src


def test_no_time_exit_when_max_bars_none() -> None:
    """max_bars None renders no max_bars_in_trade and no bar counter branch."""
    block = _block("fixed_pct", 3.0, "fixed_pct", 4.0, None)
    dsl, src = _compile(_chrom(Direction.SHORT, ExitRules(long=block, short=None, max_bars=None)))
    assert "max_bars_in_trade" not in dsl
    assert "self._bars_in_trade >=" not in src
    # SHORT: the genome's own block is the generic stop/target; no directional override
    assert dsl["stop_loss"] == {"type": "fixed_pct", "percent": 3.0}
    assert "stop_loss_long" not in dsl
    assert "stop_loss_short" not in dsl


def test_both_sides_render_their_own_blocks() -> None:
    """R3 BOTH: long atr_trailing/risk_reward, short safety/none, 12-bar time exit."""
    long_b = _block("atr_trailing", 2.5, "risk_reward", 2.0, 14)
    short_b = _block("safety", None, "none", None, None)
    dsl, src = _compile(_chrom(Direction.BOTH, ExitRules(long=long_b, short=short_b, max_bars=12)))
    assert (
        dsl["stop_loss"]
        == dsl["stop_loss_long"]
        == {
            "type": "atr_trailing",
            "atr_multiplier": 2.5,
            "indicator": "exit_atr",
        }
    )
    assert (
        dsl["take_profit"]
        == dsl["take_profit_long"]
        == {
            "type": "risk_reward",
            "risk_reward_ratio": 2.0,
        }
    )
    assert dsl["stop_loss_short"] == {"type": "fixed_pct", "percent": 50.0}
    assert dsl["take_profit_short"] == {"type": "fixed_pct", "percent": 100.0}
    for line in (
        '    stop_loss_long_type: str = "atr_trailing"',
        "    stop_loss_long_atr_multiplier: float = 2.5",
        '    stop_loss_long_indicator: str = "exit_atr"',
        "    take_profit_long_risk_reward: float = 2.0",
        "    stop_loss_short_percent: float = 50.0",
        "    take_profit_short_percent: float = 100.0",
        "    exit_atr_period: int = 14",
        "self._update_trailing_stop(bar)",
        "if self._bars_in_trade >= 12:",
    ):
        assert line in src, line
    assert "exit_atr_short" not in src  # the short block needs no ATR


def test_both_short_block_gets_its_own_atr() -> None:
    """Long and short ATR periods stay separate indicators (exit_atr vs exit_atr_short)."""
    long_b = _block("atr_fixed", 2.0, "fixed_pct", 4.0, 14)
    short_b = _block("fixed_pct", 3.0, "atr_fixed", 2.0, 7)
    dsl, src = _compile(
        _chrom(Direction.BOTH, ExitRules(long=long_b, short=short_b, max_bars=None))
    )
    assert dsl["indicators"]["exit_atr"] == {"type": "ATR", "period": 14}  # type: ignore[index]
    assert dsl["indicators"]["exit_atr_short"] == {"type": "ATR", "period": 7}  # type: ignore[index]
    assert dsl["take_profit_short"] == {
        "type": "atr_fixed",
        "atr_multiplier": 2.0,
        "indicator": "exit_atr_short",
    }
    assert "    exit_atr_period: int = 14" in src
    assert "    exit_atr_short_period: int = 7" in src
    assert '    take_profit_short_indicator: str = "exit_atr_short"' in src


def test_safety_and_none_are_described_in_the_dsl() -> None:
    """safety/none render as fixed_pct, so the DSL description says what they mean."""
    plain = _block("fixed_pct", 3.0, "fixed_pct", 4.0, None)
    safety = _block("safety", None, "fixed_pct", 4.0, None)
    none = _block("fixed_pct", 3.0, "none", None, None)
    assert "description" not in chromosome_to_dsl(
        _chrom(Direction.LONG, ExitRules(long=plain, short=None, max_bars=None))
    )
    for block in (safety, none):
        desc = chromosome_to_dsl(
            _chrom(Direction.LONG, ExitRules(long=block, short=None, max_bars=None))
        )["description"]
        assert "safety = 50% stop net" in str(desc)
        assert "none = 100% target" in str(desc)


def test_legacy_genome_has_no_exit_rules_anywhere() -> None:
    """exit_rules None: no DSL change, no serialized key, no summary change (legacy bit-identical)."""
    chrom = _chrom(Direction.LONG, None)
    assert chrom.clone().exit_rules is None
    assert "exit_rules" not in repr(chrom)  # mock_backtest seeds its metrics from repr(chrom)
    dsl = chromosome_to_dsl(chrom)
    assert dsl["stop_loss"] == {"type": "fixed_pct", "percent": 2.0}
    assert dsl["take_profit"] == {"type": "fixed_pct", "percent": 3.0}
    assert not {"description", "max_bars_in_trade", "stop_loss_long"} & dsl.keys()
    assert not any(name.startswith("exit_atr") for name in dsl["indicators"])  # type: ignore[attr-defined]
    assert "exit_rules" not in chromosome_to_serializable(chrom)
    assert genome_summary(chrom).endswith("sl=2.0% tp=3.0%")


def test_canonical_pins_dead_params() -> None:
    """R4 / R4b: params a block does not use are pinned, and a TP behind a safety net is none."""
    atr = _block("atr_fixed", 2.0, "fixed_pct", 4.0, 14)
    assert atr.atr_period == 14
    dsl = chromosome_to_dsl(_chrom(Direction.LONG, ExitRules(long=atr, short=None, max_bars=None)))
    stop = dsl["stop_loss"]
    assert isinstance(stop, dict) and "percent" not in stop

    switched = canonical_block(dataclasses.replace(atr, stop_type="fixed_pct", stop_value=3.0))
    assert switched.atr_period is None
    dsl = chromosome_to_dsl(
        _chrom(Direction.LONG, ExitRules(long=switched, short=None, max_bars=None))
    )
    assert "exit_atr" not in dsl["indicators"]  # type: ignore[operator]

    # an ATR target keeps the period even when the stop is not ATR
    assert (
        canonical_block(
            dataclasses.replace(switched, target_type="atr_fixed", target_value=3.0, atr_period=14)
        ).atr_period
        == 14
    )

    rr = canonical_block(ExitBlock("safety", None, "risk_reward", 2.0, None))
    assert (rr.target_type, rr.target_value) == ("none", None)
    dsl = chromosome_to_dsl(_chrom(Direction.LONG, ExitRules(long=rr, short=None, max_bars=None)))
    assert dsl["take_profit"] == {"type": "fixed_pct", "percent": 100.0}
    # values are rounded to 2 dp; an unused value is dropped
    rounded = canonical_block(ExitBlock("atr_fixed", 2.0049, "fixed_pct", 4.006, 14))
    assert (rounded.stop_value, rounded.target_value) == (2.0, 4.01)
    assert canonical_block(ExitBlock("safety", 5.0, "none", 5.0, 14)) == ExitBlock(
        "safety", None, "none", None, None
    )


@pytest.mark.simple_target(
    reason="design contract row R5: the time-exit span per timeframe sets how long a trade may run"
)
@pytest.mark.parametrize(
    ("tf", "expected"),
    [
        ("1m", (720, 28800)),
        ("5m", (144, 5760)),
        ("15m", (48, 1920)),
        ("1h", (12, 480)),
        ("4h", (3, 120)),
        ("1d", (1, 20)),
    ],
)
def test_time_exit_range_per_timeframe(tf: str, expected: tuple[int, int]) -> None:
    """R5: 12 h .. 20 d expressed in bars of each timeframe."""
    assert max_bars_range(tf) == expected


@pytest.mark.simple_target(
    reason="design contract row R5: an unknown timeframe must fail closed, not default"
)
def test_time_exit_range_unknown_timeframe() -> None:
    with pytest.raises(ValueError, match="13m"):
        max_bars_range("13m")


def _all_r3_rules() -> list[tuple[Direction, ExitRules]]:
    out: list[tuple[Direction, ExitRules]] = []
    for stop, target in itertools.product(STOP_TYPES, TARGET_TYPES):
        block = _block(stop, _VALUES[stop][1], target, _TARGET_VALUES[target][1], 21)
        out.append((Direction.LONG, ExitRules(long=block, short=None, max_bars=37)))
    out.append(
        (
            Direction.BOTH,
            ExitRules(
                long=_block("atr_trailing", 2.5, "risk_reward", 2.0, 14),
                short=_block("safety", None, "none", None, None),
                max_bars=12,
            ),
        )
    )
    out.append(
        (
            Direction.BOTH,
            ExitRules(
                long=_block("atr_fixed", 2.0, "fixed_pct", 4.0, 14),
                short=_block("fixed_pct", 3.0, "atr_fixed", 2.0, 7),
                max_bars=None,
            ),
        )
    )
    return out


def test_round_trip_and_bad_shapes() -> None:
    """R6: every R3 genome survives serialize -> parse; malformed shapes name the field."""
    for direction, rules in _all_r3_rules():
        assert rules_from_dict(rules_to_dict(rules)) == rules
        assert rules_problems(rules, direction) == []
        chrom = _chrom(direction, rules)
        assert validate_chromosome(chrom) == []
        stored = chromosome_to_serializable(chrom)
        assert stored["exit_rules"] == rules_to_dict(rules)
        assert serializable_to_chromosome(stored).exit_rules == rules

    legacy = chromosome_to_serializable(_chrom(Direction.LONG, None))
    assert "exit_rules" not in legacy
    assert serializable_to_chromosome(legacy).exit_rules is None
    assert serializable_to_chromosome({**legacy, "exit_rules": None}).exit_rules is None

    good = rules_to_dict(
        ExitRules(long=_block("atr_fixed", 2.0, "fixed_pct", 4.0, 14), short=None, max_bars=12)
    )
    long_good = good["long"]
    assert isinstance(long_good, dict)

    def with_long(**changes: object) -> dict[str, object]:
        return {**good, "long": {**long_good, **changes}}

    bad: list[tuple[object, str]] = [
        ([], "exit_rules"),
        (with_long(stop_type="trail"), "stop_type"),
        (with_long(stop_value=0.4), "stop_value"),
        (with_long(stop_value=6.5), "stop_value"),
        (with_long(stop_value=True), "stop_value"),
        (with_long(atr_period=None), "atr_period"),
        (with_long(atr_period=6), "atr_period"),
        (with_long(atr_period=14.5), "atr_period"),
        (with_long(target_type="fixed_pct", target_value=25.0), "target_value"),
        ({**good, "max_bars": 0}, "max_bars"),
        ({**good, "max_bars": 100_001}, "max_bars"),
        ({**good, "max_bars": True}, "max_bars"),
        ({k: v for k, v in good.items() if k != "long"}, "long"),
        ({**good, "extra": 1}, "extra"),
    ]
    for raw, field_name in bad:
        with pytest.raises(ValueError, match=field_name):
            rules_from_dict(raw)
        if isinstance(raw, dict):
            with pytest.raises(ValueError, match=field_name):
                serializable_to_chromosome({**legacy, "exit_rules": raw})
    with pytest.raises(ValueError, match="exit_rules"):
        serializable_to_chromosome({**legacy, "exit_rules": []})

    # a short block on a LONG genome is refused at load
    with_short = {**good, "short": good["long"]}
    with pytest.raises(ValueError, match="short"):
        serializable_to_chromosome({**legacy, "exit_rules": with_short})


def test_edges_of_value_ranges_are_valid_and_just_outside_is_not() -> None:
    """Both ends of each type's range pass; one step outside fails (equality edges)."""
    cases = [
        ("fixed_pct", "stop_value", 0.5, 10.0),
        ("atr_fixed", "stop_value", 1.0, 6.0),
        ("atr_trailing", "stop_value", 1.0, 6.0),
    ]
    for stop, _field, lo, hi in cases:
        for v in (lo, hi):
            assert (
                rules_problems(
                    ExitRules(_block(stop, v, "none", None, 14), None, None), Direction.LONG
                )
                == []
            )
        for v in (round(lo - 0.01, 2), round(hi + 0.01, 2)):
            problems = rules_problems(
                ExitRules(_block(stop, v, "none", None, 14), None, None), Direction.LONG
            )
            assert any("stop_value" in p for p in problems), (stop, v)
    target_cases = [
        ("fixed_pct", 0.5, 20.0),
        ("atr_fixed", 1.0, 10.0),
        ("risk_reward", 1.0, 5.0),
    ]
    for target, lo, hi in target_cases:
        for v in (lo, hi):
            assert (
                rules_problems(
                    ExitRules(_block("fixed_pct", 3.0, target, v, 14), None, None), Direction.LONG
                )
                == []
            )
        for v in (round(lo - 0.01, 2), round(hi + 0.01, 2)):
            problems = rules_problems(
                ExitRules(_block("fixed_pct", 3.0, target, v, 14), None, None), Direction.LONG
            )
            assert any("target_value" in p for p in problems), (target, v)
    block = _block("atr_fixed", 2.0, "none", None, 14)
    for n in (7, 30):
        assert (
            rules_problems(
                ExitRules(dataclasses.replace(block, atr_period=n), None, None), Direction.LONG
            )
            == []
        )
    for n in (6, 31):
        problems = rules_problems(
            ExitRules(dataclasses.replace(block, atr_period=n), None, None), Direction.LONG
        )
        assert any("atr_period" in p for p in problems), n
    for n in (1, 100_000):
        assert rules_problems(ExitRules(block, None, n), Direction.LONG) == []
    for n in (0, 100_001):
        assert any(
            "max_bars" in p for p in rules_problems(ExitRules(block, None, n), Direction.LONG)
        )


def test_problems_name_direction_and_canonical_violations() -> None:
    """validate_chromosome reports an exit_rules problem; BOTH needs a short block, LONG forbids one."""
    block = _block("fixed_pct", 3.0, "fixed_pct", 4.0, None)
    assert rules_problems(ExitRules(block, None, None), Direction.BOTH) == [
        "exit_rules.short: required for direction=both"
    ]
    assert rules_problems(ExitRules(block, block, None), Direction.LONG) == [
        "exit_rules.short: only valid for direction=both"
    ]
    non_canonical = ExitBlock("safety", None, "risk_reward", 2.0, None)
    assert any(
        "not canonical" in p
        for p in rules_problems(ExitRules(non_canonical, None, None), Direction.LONG)
    )
    errors = validate_chromosome(_chrom(Direction.BOTH, ExitRules(block, None, None)))
    assert errors == ["exit_rules.short: required for direction=both"]


def test_summary_line_for_logs() -> None:
    rules = ExitRules(
        long=_block("atr_trailing", 2.5, "risk_reward", 2.0, 14),
        short=_block("safety", None, "none", None, None),
        max_bars=12,
    )
    assert rules_summary(rules) == (
        "exits=long:atr_trailing(2.5,atr14)/risk_reward(2.0) short:safety/none time=12"
    )
    single = ExitRules(
        long=_block("fixed_pct", 3.0, "atr_fixed", 2.0, 21), short=None, max_bars=None
    )
    assert rules_summary(single) == "exits=long:fixed_pct(3.0)/atr_fixed(2.0,atr21)"
    chrom = _chrom(Direction.LONG, single)
    assert genome_summary(chrom).endswith("exits=long:fixed_pct(3.0)/atr_fixed(2.0,atr21)")
    assert "sl=" not in genome_summary(chrom)
