"""vibe-quant-e70tl.1 (DSL part): generated module names must be content-addressed.

Before the fix ``compile_to_module`` registered every strategy under
``vibe_quant.dsl.generated.{dsl.name}`` and the screening compile cache returned
that name-keyed path for a DSL-content key. Two DSLs sharing a name (a GA elite
and its mutant keep the same ``genome_{uid}``) overwrote each other in
``sys.modules`` so re-evaluating A after B ran B's code.
"""

from __future__ import annotations

import copy
import sys
from typing import Any

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.nt_runner import NTScreeningRunner


def _dsl_dict(sl_pct: float) -> dict[str, Any]:
    return {
        "name": "genome_sameuid_collision",
        "timeframe": "4h",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "stop_loss": {"type": "fixed_pct", "percent": sl_pct},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def _sl_default(module_path: str) -> float:
    cfg_cls = sys.modules[module_path].GenomeSameuidCollisionConfig
    return float(cfg_cls(instrument_id="BTCUSDT-PERP.BINANCE").stop_loss_percent)


class TestContentAddressedModuleName:
    def test_same_name_different_dsl_get_distinct_modules(self) -> None:
        a = validate_strategy_dict(_dsl_dict(1.0))
        b = validate_strategy_dict(_dsl_dict(3.0))
        mod_a = StrategyCompiler().compile_to_module(a)
        mod_b = StrategyCompiler().compile_to_module(b)

        assert mod_a.__name__ != mod_b.__name__
        assert mod_a.__name__.startswith("vibe_quant.dsl.generated.genome_sameuid_collision_")
        # A's module is still A's code after B compiled.
        assert sys.modules[mod_a.__name__] is mod_a
        assert _sl_default(mod_a.__name__) == 1.0
        assert _sl_default(mod_b.__name__) == 3.0

    def test_same_dsl_is_deterministic(self) -> None:
        a1 = StrategyCompiler().compile_to_module(validate_strategy_dict(_dsl_dict(1.5)))
        a2 = StrategyCompiler().compile_to_module(validate_strategy_dict(_dsl_dict(1.5)))
        assert a1.__name__ == a2.__name__

    def test_runner_cache_returns_own_code_after_same_name_mutant(self) -> None:
        """A -> B (same name) -> A again: A's runner must resolve A's module."""
        dsl_a = _dsl_dict(1.0)
        dsl_b = copy.deepcopy(dsl_a)
        dsl_b["stop_loss"]["percent"] = 3.0

        def resolved_path(d: dict[str, Any]) -> str:
            r = NTScreeningRunner(d, ["BTCUSDT"], "2024-01-01", "2024-02-01")
            r._ensure_compiled()
            return r._module_path

        path_a1 = resolved_path(dsl_a)
        path_b = resolved_path(dsl_b)
        path_a2 = resolved_path(dsl_a)  # compile-cache hit

        assert path_a1 == path_a2
        assert path_a1 != path_b
        assert _sl_default(path_a2) == 1.0
        assert _sl_default(path_b) == 3.0

    def test_runner_recompiles_when_module_missing(self) -> None:
        """A pickled runner (``_compiled`` True) landing in a fresh worker process
        must recompile instead of KeyError-ing on ``sys.modules``."""
        r = NTScreeningRunner(_dsl_dict(2.0), ["BTCUSDT"], "2024-01-01", "2024-02-01")
        r._ensure_compiled()
        path = r._module_path
        del sys.modules[path]
        r._ensure_compiled()
        assert r._module_path == path
        assert path in sys.modules
