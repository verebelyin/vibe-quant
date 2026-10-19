"""P-2 production-budget study driver (``scripts/research/power/budget.py``).

Rows R1-R12 are the contract rows of ``design-budget.md`` § Contract, copied here each
as its own test; R13-R15 are the driver rows the brief adds. No test launches a real
discovery or backtest: every heavy step (``layer2.run_worker`` / ``layer2.run_discovery``
/ ``layer2.harvest_run`` / ``layer2._harvest_one``) is faked via monkeypatch, exactly as
``tests/unit/test_power_layer2.py`` does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import budget  # noqa: E402
import layer2  # noqa: E402


def _out(tmp_path: Path) -> Path:
    return tmp_path / "o" / "budget.json"


def _cli(tmp_path: Path) -> list[str]:
    return [
        "--out", str(_out(tmp_path)),
        "--data-root", str(tmp_path / "data"),
        "--real-db", str(tmp_path / "none.db"),
        "--archive", str(tmp_path / "src.db"),
    ]


def _root(tmp_path: Path, variant: str, k: int = 1) -> Path:
    root = tmp_path / "o" / "scratch" / f"{variant}-b{k}"
    (root / "data" / "catalog").mkdir(parents=True, exist_ok=True)
    return root


def _accept(out: Path, variant: str, tmp_path: Path, k: int = 1) -> Path:
    root = _root(tmp_path, variant, k)

    def seed(s: dict[str, Any]) -> None:
        s.setdefault("variants", {}).setdefault(variant, {})["accepted"] = {"root": str(root), "k": k}

    budget.update_state(out, seed)
    return root


def _run_rec(
    variant: str, rung: str, seed: int, *, status: str = "completed", m1: bool = False,
    m2: bool = False, champions: int = 0, probe: bool = False, reruns: int = 0, sha: str = "sha0",
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "variant": variant, "rung": rung, "seed": seed, "status": status, "probe": probe,
        "reruns": reruns, "sha": sha,
    }
    if status == "completed":
        rec["score"] = {"M1": m1, "M2": m2}
        rec["harvest"] = {"champions": champions, "candidates": []}
    return rec


def _seed_runs(out: Path, recs: list[dict[str, Any]]) -> None:
    def mutate(s: dict[str, Any]) -> None:
        runs = s.setdefault("runs", {})
        for r in recs:
            runs[budget.run_key(r["variant"], r["rung"], r["seed"], bool(r.get("probe")))] = r

    budget.update_state(out, mutate)


# --- the brief's named first test ----------------------------------------------------------


def test_m2_requires_champion_source() -> None:
    """Planted S 4.0, C_K 0.60, candidate corr 0.62 / S 2.1: M1 needs any source, M2 a champion."""
    rejected = [{"source": "rejected", "corr": 0.62, "train_S": 2.1}]
    champion = [{"source": "champion", "corr": 0.62, "train_S": 2.1}]
    assert budget.score_candidates(rejected, 4.0, 0.60) == {"M1": True, "M2": False}
    assert budget.score_candidates(champion, 4.0, 0.60) == {"M1": True, "M2": True}


# --- R1 build accept -----------------------------------------------------------------------


def test_r1_build_accept_edges() -> None:
    """R1: S3 train 3.29 accept, 3.31 reject, 2.70 accept; unconverged / ratio reject."""
    ok = {"converged": True, "price_ratio_gt_3": False, "train_S": 3.29}
    assert budget.accepts("4h-s3", ok) is True
    assert budget.accepts("4h-s3", {**ok, "train_S": 3.31}) is False
    assert budget.accepts("4h-s3", {**ok, "train_S": 2.70}) is True  # the [0.9S, 1.1S] edge
    assert budget.accepts("4h-s3", {**ok, "train_S": 2.6999999}) is False
    assert budget.accepts("4h-s3", {**ok, "converged": False}) is False
    assert budget.accepts("4h-s3", {**ok, "converged": 1}) is False  # a truthy is not True
    assert budget.accepts("4h-s3", {**ok, "price_ratio_gt_3": True}) is False
    assert budget.accepts("4h-s3", {**ok, "train_S": None}) is False
    # s0 is the identity plant: no target, accepted whenever the build converged
    assert budget.accepts("4h-s0", {"converged": True, "price_ratio_gt_3": False}) is True
    assert budget.accepts("4h-s0", {"converged": False, "price_ratio_gt_3": False}) is False


def _stub_builder(
    monkeypatch: pytest.MonkeyPatch, trains: list[float], ratio: float = 1.2
) -> list[float]:
    deltas: list[float] = []

    def fake_load(_archive: Path, *_a: Any, **_kw: Any) -> object:
        return object()

    def fake_build(
        variant: str, k: int, delta: float, _klines: object, _archive: Path, scratch: Path,
        _db: Path, profile: layer2.Profile = layer2.P4H,
    ) -> dict[str, Any]:
        deltas.append(delta)
        root = budget.build_root(scratch, variant, k)
        (root / "data" / "catalog").mkdir(parents=True, exist_ok=True)
        return {
            "k": k, "root": str(root), "delta": delta, "kappa": 0.0, "l_end": 0.0,
            "converged": True, "n_triggers": 9, "price_ratio": ratio,
            "price_ratio_gt_3": ratio > 3.0, "train_S": trains[k - 1], "full_S": 0.0,
        }

    monkeypatch.setattr(layer2, "build_one", fake_build)
    monkeypatch.setattr(budget.plant, "load_klines", fake_load)
    return deltas


def test_r1_build_cli_accept_not_calibrated_and_price_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deltas = _stub_builder(monkeypatch, trains=[3.29, 0.0, 0.0, 0.0])
    assert budget.main([*_cli(tmp_path), "build", "--variant", "4h-s3"]) == 0
    assert len(deltas) == 1 and deltas[0] > 0.0
    assert budget.load_state(_out(tmp_path))["variants"]["4h-s3"]["status"] == "accepted"

    # never accepted -> exactly MAX_BUILDS builds, then not_calibrated (exit 2)
    out2 = tmp_path / "o2" / "budget.json"
    base2 = ["--out", str(out2), "--data-root", str(tmp_path / "data"),
             "--real-db", str(tmp_path / "n.db"), "--archive", str(tmp_path / "src.db")]
    deltas2 = _stub_builder(monkeypatch, trains=[3.31] * 4)
    assert budget.main([*base2, "build", "--variant", "4h-s4"]) == 2
    assert len(deltas2) == budget.MAX_BUILDS
    assert budget.load_state(out2)["variants"]["4h-s4"]["status"] == "not_calibrated"

    # a price ratio over 3x fails closed on the FIRST build (exit 2)
    out3 = tmp_path / "o3" / "budget.json"
    base3 = ["--out", str(out3), "--data-root", str(tmp_path / "data"),
             "--real-db", str(tmp_path / "n.db"), "--archive", str(tmp_path / "src.db")]
    deltas3 = _stub_builder(monkeypatch, trains=[3.0] * 4, ratio=3.01)
    assert budget.main([*base3, "build", "--variant", "4h-s2"]) == 2
    assert len(deltas3) == 1
    assert budget.load_state(out3)["variants"]["4h-s2"]["status"] == "failed_price_ratio"


# --- R2 P4H byte-identical -----------------------------------------------------------------


def test_r2_p4h_argv_is_byte_identical() -> None:
    argv = budget.discovery_argv(layer2.P4H, 5, Path("/x/s.db"), 2, "B1")
    assert argv == layer2.discovery_command(5, Path("/x/s.db"), 2, pop=16, gens=6)
    assert argv[-2:] == ["--dsr-rule", "emax"]
    assert budget.discovery_argv(layer2.P4H, 5, Path("/x/s.db"), 2, "B3") == (
        layer2.discovery_command(5, Path("/x/s.db"), 2, pop=36, gens=14)
    )
    # the full-pool probe drops the --indicator-pool restriction only
    probe = budget.discovery_argv(layer2.P4H, 5, Path("/x/s.db"), 2, "B1", probe=True)
    assert "--indicator-pool" not in probe
    assert len(probe) == len(argv) - 2
    # design-dsr-q95 T3: an absent rule is byte-identical; q95 appends exactly one pair.
    assert argv.count("--dsr-rule") == 1
    q95 = budget.discovery_argv(layer2.P4H, 5, Path("/x/s.db"), 2, "B1", dsr_rule="q95")
    assert q95.count("--dsr-rule") == 1
    assert q95[-2:] == ["--dsr-rule", "q95"]
    assert q95[:-2] == argv[:-2]
    probe_q95 = budget.discovery_argv(
        layer2.P4H, 5, Path("/x/s.db"), 2, "B1", probe=True, dsr_rule="q95"
    )
    assert probe_q95.count("--dsr-rule") == 1
    assert probe_q95[-2:] == ["--dsr-rule", "q95"]
    assert probe_q95.count("--indicator-pool") == 0


def test_budget_argv_resolves_to_the_requested_dsr_rule_through_the_cli_parser() -> None:
    """Wave argv parsed by the discovery CLI: emax unless q95 is asked (the CLI default is q95)."""
    from vibe_quant.discovery.__main__ import build_parser

    def effective(argv: list[str]) -> str:
        return str(build_parser().parse_args(argv[argv.index("discovery") + 1 :]).dsr_rule)

    db = Path("/x/s.db")
    assert effective(budget.discovery_argv(layer2.P4H, 5, db, 2, "B1")) == "emax"
    assert effective(budget.discovery_argv(layer2.P4H, 5, db, 2, "B1", probe=True)) == "emax"
    assert effective(budget.discovery_argv(layer2.P1D, 5, db, 2, "B1")) == "emax"
    assert effective(budget.discovery_argv(layer2.P4H, 5, db, 2, "B1", dsr_rule="q95")) == "q95"


def test_dsr_rule_recorded_and_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3: ``discover --dsr-rule`` forwards one literal pair and records the rule; absent -> None.

    Two unrelated inputs: an untagged 4h seed 1 and a q95 1d seed 3.
    """
    out = _out(tmp_path)
    for variant in ("4h-s0", "1d-s0"):
        _accept(out, variant, tmp_path)
    argvs: dict[str, list[str]] = {}

    def fake_discovery(
        root: Path, run_dir: Path, seed: int, *, command_fn: Any, profile: layer2.Profile, **kw: Any
    ) -> dict[str, Any]:
        argvs[f"{profile.timeframe}|{seed}"] = command_fn(7, run_dir / "s.db", seed)
        return {"status": "completed", "run_id": 7, "db": str(run_dir / "s.db"), "wall_s": 0.1}

    monkeypatch.setattr(layer2, "run_discovery", fake_discovery)
    cases = [("4h-s0", 1, None), ("1d-s0", 3, "q95")]
    for variant, seed, rule in cases:
        extra = [] if rule is None else ["--dsr-rule", rule]
        assert budget.main([
            *_cli(tmp_path), "discover", "--variant", variant, "--rung", "B1",
            "--seed", str(seed), *extra,
        ]) == 0
        argv = argvs[f"{variant[:2]}|{seed}"]
        rec = budget.load_state(out)["runs"][budget.run_key(variant, "B1", seed)]
        if rule is None:
            assert argv[-2:] == ["--dsr-rule", "emax"]
            assert rec["dsr_rule"] is None
        else:
            assert argv[-2:] == ["--dsr-rule", "q95"]
            assert rec["dsr_rule"] == "q95"


# --- R3 1d btc0 fidelity -------------------------------------------------------------------


def test_r3_check_btc0_fidelity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _out(tmp_path)
    _accept(out, "1d-s0", tmp_path)
    nums = {"sharpe": 1.1, "trades": 12, "return": 0.2, "profit_factor": 1.3, "max_drawdown": 0.05}
    mode = {"drift": False}

    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        drift = 1e-12 if (mode["drift"] and r != tmp_path) else 0.0
        return [{"tag": "full", **{**nums, "sharpe": nums["sharpe"] + drift}}]

    monkeypatch.setattr(layer2, "run_worker", worker)
    assert budget.main([*_cli(tmp_path), "check-btc0", "--tf", "1d"]) == 0
    assert budget.load_state(out)["check_btc0"]["1d"]["passed"] is True

    mode["drift"] = True
    assert budget.main([*_cli(tmp_path), "--force", "check-btc0", "--tf", "1d"]) == 3
    rec = budget.load_state(out)["check_btc0"]["1d"]
    assert rec["passed"] is False and rec["diffs"] and "sharpe" in rec["diffs"][0]


# --- R4 LONG only --------------------------------------------------------------------------


def test_r4_long_only_argv_and_dsl() -> None:
    prof = budget.profile_for({}, "1d")
    assert prof.direction == "long"
    argv = budget.discovery_argv(prof, 5, Path("/x/s.db"), 1, "B1")
    assert argv.count("--direction") == 1
    assert argv[argv.index("--direction") + 1] == "long"

    assert budget.is_long_dsl(
        {"entry_conditions": {"long": ["a"]}, "exit_conditions": {"long": ["b"]}}
    ) is True
    assert budget.is_long_dsl(
        {"entry_conditions": {"long": ["a"], "short": ["b"]}, "exit_conditions": {"long": ["b"]}}
    ) is False
    assert budget.is_long_dsl({}) is False


def test_r4_non_long_harvested_candidate_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    _accept(out, "4h-s0", tmp_path)
    rec = _run_rec("4h-s0", "B1", 1, champions=1)
    rec.update(db=str(tmp_path / "r.db"), run_id=1)
    _seed_runs(out, [rec])
    short = {"entry_conditions": {"long": ["a"], "short": ["b"]}, "exit_conditions": {"long": ["b"]}}

    def fake_harvest(db: Path, rid: int) -> dict[str, Any]:
        return {"r31": {"ok": True}, "dsr_trials": {"n_rule": layer2.N_RULE_EFFECTIVE}, "candidates": [
            {"uid": "u", "source": "champion", "dsl": short}
        ], "champions": 1}

    monkeypatch.setattr(layer2, "harvest_run", fake_harvest)
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: True)
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "4h"]) == 3
    assert "non-LONG" in capsys.readouterr().out


# --- R5 isolation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trials",
    [
        {"ok": False, "prior_n": 1, "gate_n": 38, "run_n_eff": 37},
        {"ok": False, "n_rule": "legacy"},
        {"ok": False, "n_rule": None},
    ],
)
def test_r5_isolation_violation_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    trials: dict[str, Any],
) -> None:
    out = _out(tmp_path)
    _accept(out, "4h-s0", tmp_path)
    rec = _run_rec("4h-s0", "B1", 1, champions=0)
    rec.update(db=str(tmp_path / "r.db"), run_id=1)
    _seed_runs(out, [rec])
    monkeypatch.setattr(layer2, "harvest_run", lambda db, rid: {"r31": trials, "candidates": []})
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: True)
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "4h"]) == 3
    assert "R31" in capsys.readouterr().out


# --- R6 M1 edges ---------------------------------------------------------------------------


def test_r6_m1_edges() -> None:
    def cand(corr: float | None, s: float | None, src: str = "champion") -> list[dict[str, Any]]:
        return [{"source": src, "corr": corr, "train_S": s}]

    assert budget.score_candidates(cand(0.60, 2.0), 4.0, 0.60) == {"M1": True, "M2": True}
    assert budget.score_candidates(cand(0.60, 2.0, "rejected"), 4.0, 0.60)["M1"] is True
    assert budget.score_candidates(cand(0.5999, 2.0), 4.0, 0.60)["M1"] is False
    assert budget.score_candidates(cand(0.60, 1.9999), 4.0, 0.60)["M1"] is False  # S < 0.5*planted
    assert budget.score_candidates(cand(0.70, 2.0), 4.0, 0.60) == {"M1": True, "M2": True}
    assert budget.score_candidates(cand(None, 2.0), 4.0, 0.60)["M1"] is False
    assert budget.score_candidates([], 4.0, 0.60) == {"M1": False, "M2": False}


# --- R7 K ----------------------------------------------------------------------------------


def test_r7_k_requires_evaluated_and_fits_the_null_pool() -> None:
    assert budget.k_for_run({"harvest": {"evaluated": 85}}, 1200) == 85
    with pytest.raises(layer2.Layer2Error, match="evaluated"):
        budget.k_for_run({"harvest": {"evaluated": None}}, 1200)
    with pytest.raises(layer2.Layer2Error, match="null pool"):
        budget.k_for_run({"harvest": {"evaluated": 1201}}, 1200)


# --- R8 C_K --------------------------------------------------------------------------------


def test_r8_c_k_needs_same_tf_same_budget_s0() -> None:
    state: dict[str, Any] = {"runs": {
        budget.run_key("4h-s0", "B1", 1): {
            "variant": "4h-s0", "rung": "B1", "seed": 1, "status": "completed",
            "harvest": {"candidates": [{"corr": 0.3}]},
        },
        budget.run_key("1d-s0", "B1", 1): {
            "variant": "1d-s0", "rung": "B1", "seed": 1, "status": "completed",
            "harvest": {"candidates": [{"corr": 0.9}]},
        },
        budget.run_key("4h-s4", "B1", 1): {
            "variant": "4h-s4", "rung": "B1", "seed": 1, "status": "completed",
            "harvest": {"candidates": [{"corr": 0.99}]},
        },
    }}
    assert budget.s0_max_signed_corr(state, "4h", 16, 6) == 0.3  # other tf + non-s0 ignored
    assert layer2.corr_threshold(0.5, budget.s0_max_signed_corr(state, "4h", 16, 6)) == 0.5
    with pytest.raises(layer2.Layer2Error):
        budget.s0_max_signed_corr(state, "4h", 24, 10)  # no s0 at that budget


# --- R9 failed / timed-out runs ------------------------------------------------------------


def test_r9_failed_run_is_missing_then_incomplete(
    tmp_path: Path,
) -> None:
    out = _out(tmp_path)
    recs = [
        _run_rec("4h-s0", "B1", 1), _run_rec("4h-s0", "B1", 2), _run_rec("4h-s0", "B1", 3),
        _run_rec("4h-s3", "B1", 1, m1=True, m2=True),
        _run_rec("4h-s3", "B1", 3, m1=True, m2=True),
        _run_rec("4h-s3", "B1", 2, status="failed", reruns=0),
        _run_rec("4h-s5", "B1", 1, status="timeout", reruns=1),
    ]
    _seed_runs(out, recs)
    st = budget.load_state(out)
    budget.refresh_cells(st, "4h")
    # 2/3 M2 but one seed missing: the cell must NOT stop on a rung with a missing run
    assert st["cells"]["4h-s3"]["B_star"] is None and st["cells"]["4h-s3"]["status"] == "open"
    wave = budget.next_wave(st, "4h")
    assert ("4h-s3", "B1", 2, False) in wave  # the failed run is a re-runnable rerun
    # a seed given up after one rerun makes the whole cell incomplete, B* None, no more runs
    assert st["cells"]["4h-s5"]["status"] == "incomplete"
    assert st["cells"]["4h-s5"]["B_star"] is None
    assert all(v != "4h-s5" for v, _k, _s, _p in wave)


# --- R10 stop rule -------------------------------------------------------------------------


def test_r10_two_of_three_stop_and_absent_from_the_next_wave(tmp_path: Path) -> None:
    out = _out(tmp_path)
    recs: list[dict[str, Any]] = []
    for s in (1, 2, 3):
        recs += [_run_rec("4h-s0", "B1", s), _run_rec("4h-s0", "B2", s)]
        recs += [_run_rec("4h-s2", "B1", s), _run_rec("4h-s2", "B2", s)]
        recs += [_run_rec("4h-s3", "B1", s, m1=s == 1, m2=s == 1),
                 _run_rec("4h-s3", "B2", s, m1=s == 1, m2=s == 1)]
        recs += [_run_rec("4h-s4", "B1", s, m1=s == 1, m2=s == 1),
                 _run_rec("4h-s4", "B2", s, m1=s in (1, 2), m2=s in (1, 2))]
        recs += [_run_rec("4h-s5", "B1", s, m1=s in (1, 2), m2=s in (1, 2))]
    _seed_runs(out, recs)
    st = budget.load_state(out)
    budget.refresh_cells(st, "4h")
    assert st["cells"]["4h-s4"]["B_star"] == "B2"
    assert st["cells"]["4h-s5"]["B_star"] == "B1"
    assert st["cells"]["4h-s3"]["B_star"] is None and st["cells"]["4h-s3"]["status"] == "open"
    wave = budget.next_wave(st, "4h")
    assert all(v != "4h-s4" for v, _k, _s, _p in wave)
    assert all(v != "4h-s5" for v, _k, _s, _p in wave)
    assert ("4h-s3", "B3", 1, False) in wave  # the 1/3 cell continues to the next rung


def test_r10_two_seed_mode_needs_two_of_two(tmp_path: Path) -> None:
    out = _out(tmp_path)
    recs = []
    for s in (1, 2):
        recs.append(_run_rec("1d-s4", "B1", s, m1=True, m2=True))
        recs.append(_run_rec("1d-s3", "B1", s, m1=s == 1, m2=s == 1))
    _seed_runs(out, recs)
    budget.update_state(out, lambda s: s.setdefault("seeds", {}).__setitem__("1d", 2))
    st = budget.load_state(out)
    budget.refresh_cells(st, "1d")
    assert budget.tf_seeds(st, "1d") == 2
    assert st["cells"]["1d-s4"]["B_star"] == "B1"  # 2/2
    assert st["cells"]["1d-s3"]["B_star"] is None  # 1/2 is not enough


# --- R11 s0 champion -> GATE LEAK ----------------------------------------------------------


def test_r11_s0_champion_stops_the_tf_and_reports_gate_leak(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    _seed_runs(out, [
        _run_rec("4h-s0", "B1", 1, champions=1),
        _run_rec("4h-s4", "B1", 1),
    ])
    st = budget.load_state(out)
    assert budget.gate_leak(st, "4h") is True
    assert budget.next_wave(st, "4h") == []

    assert budget.main([*_cli(tmp_path), "next-wave", "--tf", "4h"]) == 0
    assert capsys.readouterr().out.strip() == ""  # prints nothing for the leaked tf

    assert budget.main([*_cli(tmp_path), "report"]) == 0
    assert "GATE LEAK" in capsys.readouterr().out


# --- R12 pick-rule -------------------------------------------------------------------------


def test_r12_pick_rule_trades_window_and_ga_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    _accept(out, "1d-s0", tmp_path)
    trades = {0: 79, 1: 120, 2: 251, 3: 80}
    seen: list[list[str]] = []

    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        seen.append([j["tag"] for j in payload["jobs"]])
        return [
            {"tag": j["tag"], "trades": trades[i], "sharpe": 0.0, "return": 0.0,
             "profit_factor": 0.0, "max_drawdown": 0.0, "mtm": None, "bh_daily_sharpe": None}
            for i, j in enumerate(payload["jobs"])
        ]

    monkeypatch.setattr(layer2, "run_worker", worker)
    assert budget.main([*_cli(tmp_path), "pick-rule", "--tf", "1d"]) == 0
    assert seen == [[r["name"] for r in budget.PICK_RULES]]
    st = budget.load_state(out)
    assert st["picked_rule"]["name"] == budget.PICK_RULES[1]["name"]  # trades 120 is in [80, 250]
    prof = budget.profile_for(st, "1d")
    assert prof.chromosome.entry_genes[0].parameters["period"] == 7.0
    assert prof.chromosome.entry_genes[0].threshold == 30.0
    assert prof.plant_kw["rsi_period"] == 7 and prof.plant_kw["rsi_level"] == 30.0
    assert prof.chromosome.direction is layer2.Direction.LONG

    # every parameter outside the GA ranges -> skipped (exit 2), even with a good trade count
    bad = {"name": "bad", "period": 100, "xb": 30.0, "xa": 60.0, "sl": 4.0, "tp": 8.0}
    assert budget.rule_in_ga_ranges(bad) is False
    assert budget.rule_in_ga_ranges(budget.PICK_RULES[0]) is True
    out2 = tmp_path / "o2" / "budget.json"
    _accept(out2, "1d-s0", tmp_path)
    monkeypatch.setattr(budget, "PICK_RULES", [bad])

    def worker2(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        return [{"tag": j["tag"], "trades": 120, "sharpe": 0.0, "return": 0.0,
                 "profit_factor": 0.0, "max_drawdown": 0.0, "mtm": None, "bh_daily_sharpe": None}
                for j in payload["jobs"]]

    monkeypatch.setattr(layer2, "run_worker", worker2)
    base2 = ["--out", str(out2), "--data-root", str(tmp_path / "data"),
             "--real-db", str(tmp_path / "n.db"), "--archive", str(tmp_path / "src.db")]
    assert budget.main([*base2, "pick-rule", "--tf", "1d"]) == 2
    assert "no rule" in capsys.readouterr().out


# --- R13 next-wave on an empty state -------------------------------------------------------


def test_r13_next_wave_empty_state_prints_15_b1_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert budget.main([*_cli(tmp_path), "next-wave", "--tf", "4h"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 15
    assert all("--rung B1" in ln and "discover --variant 4h-" in ln for ln in lines)
    variants = {ln.split("--variant ")[1].split()[0] for ln in lines}
    assert variants == {"4h-s0", "4h-s2", "4h-s3", "4h-s4", "4h-s5"}


# --- R14 full-pool probe -------------------------------------------------------------------


def test_r14_probe_runs_never_enter_bstar_or_c_k(tmp_path: Path) -> None:
    out = _out(tmp_path)
    recs = [_run_rec("4h-s4", "B1", s, m2=False) for s in (1, 2, 3)]
    recs.append(_run_rec("4h-s4", "B1", 1, m1=True, m2=True, probe=True))  # a probe M2
    _seed_runs(out, recs)
    st = budget.load_state(out)
    budget.refresh_cells(st, "4h")
    assert budget.m2_count(st, "4h-s4", "B1") == 0  # the probe's M2 is not counted
    assert st["cells"]["4h-s4"]["B_star"] is None
    # a probe candidate never becomes the s0 comparator either
    assert budget.is_probe_run(
        {"variant": "4h-s4", "probe": True}
    ) and not budget.is_probe_run({"variant": "4h-s4"})


# --- R15 report ----------------------------------------------------------------------------


def test_r15_report_prints_bstar_counts_s0_and_shas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    recs: list[dict[str, Any]] = []
    for s in (1, 2, 3):
        recs.append(_run_rec("4h-s4", "B1", s, m1=s == 1, m2=s == 1, sha="aaaa"))
        recs.append(_run_rec("4h-s4", "B2", s, m1=s in (1, 2), m2=s in (1, 2), sha="bbbb"))
    _seed_runs(out, recs)
    assert budget.main([*_cli(tmp_path), "report"]) == 0
    text = capsys.readouterr().out
    assert "4h-s4: B*=B2" in text
    assert "B1 M1=1/3 M2=1/3" in text
    assert "B2 M1=2/3 M2=2/3" in text
    assert "s0 champion count: 0" in text
    assert "aaaa" in text and "bbbb" in text


# --- C_K probe nit -------------------------------------------------------------------------


def test_c_k_ignores_probe_s0_records_exact_max() -> None:
    """Same-tf / same-budget s0 records with and without the probe tag: only the untagged count."""

    def s0(seed: int, corr: float, probe: bool) -> dict[str, Any]:
        return {
            "variant": "4h-s0", "rung": "B1", "seed": seed, "status": "completed", "probe": probe,
            "harvest": {"candidates": [{"corr": corr}]},
        }

    state: dict[str, Any] = {"runs": {
        budget.run_key("4h-s0", "B1", 1): s0(1, 0.31, False),
        budget.run_key("4h-s0", "B1", 2): s0(2, 0.44, False),
        budget.run_key("4h-s0", "B1", 3, True): s0(3, 0.95, True),
    }}
    assert budget.s0_max_signed_corr(state, "4h", 16, 6) == 0.44


# --- F1 LONG-only profiles for both timeframes ----------------------------------------------


def test_f1_both_timeframes_run_long_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert budget.profile_for({}, "4h").direction == "long"
    assert budget.profile_for({}, "1d").direction == "long"
    assert layer2.P4H.direction == "both"  # layer2 legacy default untouched
    out = _out(tmp_path)
    for variant in ("4h-s0", "1d-s0"):
        _accept(out, variant, tmp_path)
    argvs: dict[str, list[str]] = {}
    payloads: dict[str, dict[str, Any]] = {}

    def fake_discovery(
        root: Path, run_dir: Path, seed: int, *, command_fn: Any, profile: layer2.Profile, **kw: Any
    ) -> dict[str, Any]:
        argvs[profile.timeframe] = command_fn(7, run_dir / "s.db", seed)
        return {"status": "completed", "run_id": 7, "db": str(run_dir / "s.db"), "wall_s": 0.1}

    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        payloads[payload["profile"]["timeframe"]] = payload
        return [{"seed": 0}]

    monkeypatch.setattr(layer2, "run_discovery", fake_discovery)
    monkeypatch.setattr(layer2, "run_worker", worker)
    for variant in ("4h-s0", "1d-s0"):
        assert budget.main([*_cli(tmp_path), "discover", "--variant", variant, "--rung", "B1", "--seed", "1"]) == 0
        tf = variant[:2]
        assert argvs[tf][argvs[tf].index("--direction") + 1] == "long"
        assert budget.main([*_cli(tmp_path), "null-corr", "--tf", tf, "--m", "2", "--shards", "1"]) == 0
        assert payloads[tf]["profile"]["direction"] == "long"


# --- F2 printed retry command ---------------------------------------------------------------


def _printed_args(line: str) -> list[str]:
    import shlex

    toks = shlex.split(line)
    return toks[toks.index("scripts/research/power/budget.py") + 1 :]


def test_f2_printed_retry_command_reruns_then_goes_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    _accept(out, "4h-s3", tmp_path)
    failed = _run_rec("4h-s3", "B1", 2, status="failed", reruns=0)
    _seed_runs(out, [failed])
    launches: list[int] = []

    def fake_discovery(root: Path, run_dir: Path, seed: int, **kw: Any) -> dict[str, Any]:
        launches.append(seed)
        return {"status": "failed", "run_id": 1, "db": str(run_dir / "s.db"), "wall_s": 0.1, "error": "boom"}

    monkeypatch.setattr(layer2, "run_discovery", fake_discovery)

    def printed() -> dict[str, str]:
        capsys.readouterr()
        assert budget.main([*_cli(tmp_path), "next-wave", "--tf", "4h"]) == 0
        lines = capsys.readouterr().out.splitlines()
        return {ln.split("--variant ")[1].split()[0] + ln.split("--seed ")[1].split()[0]: ln for ln in lines if ln.strip()}

    cmds = printed()
    retry = cmds["4h-s32"]
    assert " --force discover " in retry
    assert all("--force" not in ln for key, ln in cmds.items() if key != "4h-s32")
    # the retry really ran and failed again: the exit code now says so (SEMANTICS BREAK, 4urlp)
    assert budget.main([*_cli(tmp_path), *_printed_args(retry)]) == budget.EXIT_RUN_NOT_COMPLETED
    assert launches == [2]  # the printed command really re-launched the run
    st = budget.load_state(out)
    assert st["runs"][budget.run_key("4h-s3", "B1", 2)]["reruns"] == 1
    budget.refresh_cells(st, "4h")
    assert st["cells"]["4h-s3"]["status"] == "incomplete"
    assert "4h-s32" not in printed()  # terminal: no third attempt is offered


# --- F3 harvest requires the effective-N marker through the real parser -----------------------


def _notes_db(path: Path, notes: dict[str, Any]) -> None:
    import json
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE backtest_results (run_id INTEGER, notes TEXT)")
    conn.execute("INSERT INTO backtest_results VALUES (1, ?)", (json.dumps(notes),))
    conn.commit()
    conn.close()


@pytest.mark.parametrize(
    "notes",
    [
        {},  # no dsr_trials at all
        {"dsr_trials": {}},
        {"dsr_trials": {"prior_n": 0, "gate_n": 5, "run_n_eff": 5}},  # marker absent
        {"dsr_trials": {"prior_n": 0, "gate_n": 5, "run_n_eff": 5, "n_rule": "legacy"}},
        {"dsr_trials": {"prior_n": 0, "gate_n": 5, "run_n_eff": 5, "n_rule": None}},
    ],
)
def test_f3_harvest_refuses_missing_or_invalid_effective_n_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    notes: dict[str, Any],
) -> None:
    out = _out(tmp_path)
    _accept(out, "4h-s0", tmp_path)
    db = tmp_path / "r.db"
    _notes_db(db, notes)
    rec = _run_rec("4h-s0", "B1", 1)
    rec.update(db=str(db), run_id=1)
    _seed_runs(out, [rec])
    called: list[int] = []
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: called.append(1) or True)
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "4h"]) == 3
    assert called == []  # refused before any screening
    assert "effective-N" in capsys.readouterr().out


def test_f3_harvest_accepts_the_effective_n_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _out(tmp_path)
    _accept(out, "4h-s0", tmp_path)
    db = tmp_path / "r.db"
    trials = {"prior_n": 0, "gate_n": 5, "run_n_eff": 5, "n_rule": layer2.N_RULE_EFFECTIVE, "dsr_rule": "emax"}
    _notes_db(db, {"dsr_trials": trials, "evaluated": 5})
    rec = _run_rec("4h-s0", "B1", 1)
    rec.update(db=str(db), run_id=1)
    _seed_runs(out, [rec])
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: True)
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "4h"]) == 0


# --- F4 build invalidation and build_id --------------------------------------------------------


def _seeded_study(tmp_path: Path) -> Path:
    """4h state: every variant accepted with build_id b0, runs scored, null/btc0/chain recorded."""
    out = _out(tmp_path)
    for variant in ("4h-s0", "4h-s4"):
        _accept(out, variant, tmp_path)
    recs = [_run_rec(v, "B1", s, m1=True, m2=True) for v in ("4h-s0", "4h-s4") for s in (1, 2, 3)]
    for r in recs:
        r["build_id"] = "b0"
    _seed_runs(out, recs)

    def seed(s: dict[str, Any]) -> None:
        for v in ("4h-s0", "4h-s4"):
            s["variants"][v]["build_id"] = "b0"
        s["direct_chain"] = {"4h-s4": {"failures": []}, "4h-s0": {"failures": []}}
        s["check_btc0"] = {"4h": {"passed": True}}
        s["null"] = {"4h": {"status": "completed"}, "1d": {"status": "completed"}}
        budget.refresh_cells(s, "4h")

    budget.update_state(out, seed)
    return out


def test_f4_forced_variant_build_drops_its_dependents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _seeded_study(tmp_path)
    _stub_builder(monkeypatch, trains=[4.0])
    assert budget.main([*_cli(tmp_path), "--force", "build", "--variant", "4h-s4"]) == 0
    st = budget.load_state(out)
    assert not [r for r in st["runs"].values() if r["variant"] == "4h-s4"]
    assert "4h-s4" not in st["direct_chain"] and "4h-s4" not in st["cells"]
    assert "4h-s0" in st["direct_chain"]  # other variants untouched
    assert len([r for r in st["runs"].values() if r["variant"] == "4h-s0"]) == 3
    assert st["variants"]["4h-s4"]["build_id"] != "b0"
    # the next wave offers the whole rebuilt cell again at B1 (old champions are gone)
    assert budget.main([*_cli(tmp_path), "next-wave", "--tf", "4h"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert sum("--variant 4h-s4 --rung B1" in ln for ln in lines) == 3
    assert not any("--force" in ln for ln in lines)


def test_f4_forced_s0_build_drops_tf_null_check_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _seeded_study(tmp_path)
    _stub_builder(monkeypatch, trains=[0.0])
    assert budget.main([*_cli(tmp_path), "--force", "build", "--variant", "4h-s0"]) == 0
    st = budget.load_state(out)
    assert "4h" not in st["null"] and "1d" in st["null"]  # only the s0's tf
    assert "4h" not in (st.get("check_btc0") or {})
    assert not [r for r in st["runs"].values() if r["variant"] == "4h-s0"]
    survivors = [r for r in st["runs"].values() if r["variant"] == "4h-s4"]
    assert len(survivors) == 3 and all("score" not in r for r in survivors)
    assert budget.m2_count(st, "4h-s4", "B1") == 0
    assert budget.refresh_cells(st, "4h") is None and st["cells"]["4h-s4"]["B_star"] is None


def test_f4_changed_pick_rule_drops_1d_null_check_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o" / "budget.json"
    _accept(out, "1d-s0", tmp_path)
    _accept(out, "1d-s4", tmp_path)
    recs = [_run_rec("1d-s4", "B1", s, m1=True, m2=True) for s in (1, 2, 3)]
    _seed_runs(out, recs)

    def seed(s: dict[str, Any]) -> None:
        s["picked_rule"] = {**budget.PICK_RULES[0], "trades": 100}
        s["null"] = {"1d": {"status": "completed"}, "4h": {"status": "completed"}}
        s["check_btc0"] = {"1d": {"passed": True}}

    budget.update_state(out, seed)

    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        # rule 0 now yields too few trades, rule 1 is picked
        return [
            {"tag": j["tag"], "trades": 10 if i == 0 else 120, "sharpe": 0.0, "return": 0.0,
             "profit_factor": 0.0, "max_drawdown": 0.0, "mtm": None, "bh_daily_sharpe": None}
            for i, j in enumerate(payload["jobs"])
        ]

    monkeypatch.setattr(layer2, "run_worker", worker)
    assert budget.main([*_cli(tmp_path), "--force", "pick-rule", "--tf", "1d"]) == 0
    st = budget.load_state(out)
    assert st["picked_rule"]["name"] == budget.PICK_RULES[1]["name"]
    assert "1d" not in st["null"] and "4h" in st["null"]
    assert "1d" not in (st.get("check_btc0") or {})
    assert all("score" not in r for r in st["runs"].values())


def test_f4_unchanged_pick_rule_keeps_the_null_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o" / "budget.json"
    _accept(out, "1d-s0", tmp_path)

    def seed(s: dict[str, Any]) -> None:
        s["picked_rule"] = {**budget.PICK_RULES[0], "trades": 100}
        s["null"] = {"1d": {"status": "completed"}}

    budget.update_state(out, seed)
    monkeypatch.setattr(
        layer2, "run_worker",
        lambda kind, r, payload, db, **kw: [
            {"tag": j["tag"], "trades": 120, "sharpe": 0.0, "return": 0.0,
             "profit_factor": 0.0, "max_drawdown": 0.0, "mtm": None, "bh_daily_sharpe": None}
            for j in payload["jobs"]
        ],
    )
    assert budget.main([*_cli(tmp_path), "--force", "pick-rule", "--tf", "1d"]) == 0
    assert "1d" in budget.load_state(out)["null"]


def test_f4_harvest_refuses_a_run_from_another_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _seeded_study(tmp_path)

    def stale(s: dict[str, Any]) -> None:
        s["variants"]["4h-s4"]["build_id"] = "b1"  # rebuilt since the runs were made

    budget.update_state(out, stale)
    budget.update_state(out, lambda s: s["runs"].update(
        {k: {**r, "db": str(tmp_path / "r.db"), "run_id": 1} for k, r in s["runs"].items()}
    ))
    called: list[int] = []
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: called.append(1) or True)
    monkeypatch.setattr(
        layer2, "harvest_run",
        lambda db, rid: {"r31": {"ok": True}, "dsr_trials": {"n_rule": layer2.N_RULE_EFFECTIVE},
                         "candidates": []},
    )
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "4h"]) == 3
    assert called == []
    assert "build" in capsys.readouterr().out


# --- R34 1d planted builds follow the picked rule ----------------------------------------------


def _rule_worker(trades: dict[int, int]) -> Any:
    def worker(kind: str, r: Path, payload: dict[str, Any], db: Path, **kw: Any) -> Any:
        return [
            {"tag": j["tag"], "trades": trades[i], "sharpe": 0.0, "return": 0.0,
             "profit_factor": 0.0, "max_drawdown": 0.0, "mtm": None, "bh_daily_sharpe": None}
            for i, j in enumerate(payload["jobs"])
        ]

    return worker


def _seed_1d(tmp_path: Path, rule_name: str | None) -> Path:
    out = _out(tmp_path)
    for variant in ("1d-s0", "1d-s4"):
        _accept(out, variant, tmp_path)
    _seed_runs(out, [_run_rec("1d-s4", "B1", s, m1=True, m2=True) for s in (1, 2, 3)])

    def seed(s: dict[str, Any]) -> None:
        s["picked_rule"] = {**budget.PICK_RULES[0], "trades": 100}
        for v in ("1d-s0", "1d-s4"):
            s["variants"][v]["rule"] = rule_name
            s["variants"][v]["build_id"] = "b0"
        s["direct_chain"] = {"1d-s4": {"failures": []}}
        budget.refresh_cells(s, "1d")

    budget.update_state(out, seed)
    return out


def test_r34_changed_pick_rule_invalidates_planted_1d_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _seed_1d(tmp_path, budget.PICK_RULES[0]["name"])
    monkeypatch.setattr(layer2, "run_worker", _rule_worker({0: 10, 1: 120, 2: 120, 3: 120}))
    assert budget.main([*_cli(tmp_path), "--force", "pick-rule", "--tf", "1d"]) == 0
    st = budget.load_state(out)
    v4 = st["variants"]["1d-s4"]
    assert "accepted" not in v4 and "build_id" not in v4 and v4["status"] == "building"
    assert not [r for r in st["runs"].values() if r["variant"] == "1d-s4"]
    assert "1d-s4" not in st["direct_chain"]
    assert st["cells"]["1d-s4"] == {"tf": "1d", "B_star": None, "rung": "B1", "status": "open"}
    assert "accepted" in st["variants"]["1d-s0"]  # the unplanted s0 build is rule-free


def test_r34_unchanged_pick_rule_drops_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _seed_1d(tmp_path, budget.PICK_RULES[0]["name"])
    monkeypatch.setattr(layer2, "run_worker", _rule_worker({0: 120, 1: 120, 2: 120, 3: 120}))
    assert budget.main([*_cli(tmp_path), "--force", "pick-rule", "--tf", "1d"]) == 0
    st = budget.load_state(out)
    assert "accepted" in st["variants"]["1d-s4"]
    assert len([r for r in st["runs"].values() if r["variant"] == "1d-s4"]) == 3
    assert "1d-s4" in st["direct_chain"]


def test_r34_build_records_the_picked_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _out(tmp_path)
    budget.update_state(out, lambda s: s.__setitem__("picked_rule", {**budget.PICK_RULES[2]}))
    _stub_builder(monkeypatch, trains=[4.0])
    assert budget.main([*_cli(tmp_path), "build", "--variant", "1d-s4"]) == 0
    assert budget.load_state(out)["variants"]["1d-s4"]["rule"] == budget.PICK_RULES[2]["name"]


def test_r34_discover_and_harvest_refuse_a_build_from_another_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _seed_1d(tmp_path, "some_old_rule")
    launched: list[int] = []
    monkeypatch.setattr(layer2, "run_discovery", lambda *a, **k: launched.append(1) or {})
    assert budget.main(
        [*_cli(tmp_path), "discover", "--variant", "1d-s4", "--rung", "B1", "--seed", "9"]
    ) == 2
    assert launched == [] and "rule" in capsys.readouterr().out

    db = tmp_path / "r.db"
    budget.update_state(out, lambda s: s["runs"].update(
        {k: {**r, "db": str(db), "run_id": 1, "build_id": "b0"} for k, r in s["runs"].items()}
    ))
    called: list[int] = []
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: called.append(1) or True)
    monkeypatch.setattr(
        layer2, "harvest_run",
        lambda d, rid: {"r31": {"ok": True}, "dsr_trials": {"n_rule": layer2.N_RULE_EFFECTIVE},
                        "candidates": []},
    )
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "1d"]) == 3
    assert called == [] and "rule" in capsys.readouterr().out


# --- R34b / R34c (round 3) -----------------------------------------------------------------------


def test_r34b_discover_of_an_existing_seed_still_checks_the_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_1d(tmp_path, "some_old_rule")  # 1d-s4 B1 seeds 1-3 already recorded
    launched: list[int] = []
    monkeypatch.setattr(layer2, "run_discovery", lambda *a, **k: launched.append(1) or {})
    assert budget.main(
        [*_cli(tmp_path), "discover", "--variant", "1d-s4", "--rung", "B1", "--seed", "1"]
    ) == 2
    assert launched == [] and "rule" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["accepted", "not_calibrated"])
def test_r34c_pick_rule_change_resets_calibration_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    out = _out(tmp_path)
    _accept(out, "1d-s0", tmp_path)
    _accept(out, "1d-s4", tmp_path, k=4)
    if status != "accepted":
        budget.update_state(out, lambda s: s["variants"]["1d-s4"].pop("accepted"))
    old = budget.PICK_RULES[0]["name"]

    def seed(s: dict[str, Any]) -> None:
        s["picked_rule"] = {**budget.PICK_RULES[0], "trades": 100}
        v = s["variants"]["1d-s4"]
        v["builds"] = [
            {"k": k, "delta": 1e-5 * k, "train_S": t, "converged": True}
            for k, t in enumerate([2.0, 3.0, 3.5, 4.0], 1)
        ]
        v.update(rule=old, build_id="b0", status=status)

    budget.update_state(out, seed)
    monkeypatch.setattr(layer2, "run_worker", _rule_worker({0: 10, 1: 120, 2: 120, 3: 120}))
    assert budget.main([*_cli(tmp_path), "--force", "pick-rule", "--tf", "1d"]) == 0
    assert budget.load_state(out)["picked_rule"]["name"] == budget.PICK_RULES[1]["name"]

    deltas = _stub_builder(monkeypatch, trains=[4.0])
    assert budget.main([*_cli(tmp_path), "build", "--variant", "1d-s4"]) == 0
    assert len(deltas) == 1  # build_one ran: the old four builds no longer count
    v = budget.load_state(out)["variants"]["1d-s4"]
    assert v["status"] == "accepted" and v["rule"] == budget.PICK_RULES[1]["name"]
    assert [b["k"] for b in v["builds"]] == [1]


def test_r34e_unaccepted_builds_record_their_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _out(tmp_path)
    budget.update_state(out, lambda s: s.__setitem__("picked_rule", {**budget.PICK_RULES[1]}))
    _stub_builder(monkeypatch, trains=[1.0] * 4)
    assert budget.main([*_cli(tmp_path), "build", "--variant", "1d-s4"]) == 2
    assert budget.load_state(out)["variants"]["1d-s4"]["rule"] == budget.PICK_RULES[1]["name"]


# --- R9b discover exit code ----------------------------------------------------------------


def _stub_discovery(status: str) -> Any:
    def fake_discovery(
        root: Path, run_dir: Path, seed: int, **kw: Any
    ) -> dict[str, Any]:
        return {"status": status, "run_id": 7, "db": str(run_dir / "s.db"), "wall_s": 0.1,
                "error": "killed at the timeout" if status != "completed" else ""}

    return fake_discovery


@pytest.mark.parametrize("status", ["timeout", "failed"])
def test_discover_exits_nonzero_on_a_non_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    """Bead 4urlp: a wave runner keyed off exit codes saw 0 while 5/6 runs were killed."""
    out = _out(tmp_path)
    _accept(out, "4h-s4", tmp_path)
    monkeypatch.setattr(layer2, "run_discovery", _stub_discovery(status))
    rc = budget.main([
        *_cli(tmp_path), "discover", "--variant", "4h-s4", "--rung", "B1", "--seed", "2",
    ])
    assert rc == budget.EXIT_RUN_NOT_COMPLETED == 4
    text = capsys.readouterr().out
    assert "RUN NOT COMPLETED" in text and status in text
    # the record is still written: the next wave offers a forced rerun, not a silent pass
    rec = budget.load_state(out)["runs"][budget.run_key("4h-s4", "B1", 2)]
    assert rec["status"] == status

    # the equality edge: a completed run keeps the old exit 0
    out2 = tmp_path / "o2" / "budget.json"
    _accept(out2, "4h-s4", tmp_path)
    monkeypatch.setattr(layer2, "run_discovery", _stub_discovery("completed"))
    assert budget.main([
        "--out", str(out2), "--data-root", str(tmp_path / "data"),
        "--real-db", str(tmp_path / "n.db"), "--archive", str(tmp_path / "src.db"),
        "discover", "--variant", "4h-s4", "--rung", "B1", "--seed", "2",
    ]) == 0


@pytest.mark.parametrize("status", ["timeout", "failed"])
def test_discover_of_an_already_recorded_non_completed_run_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    """The idempotent "already recorded" path must not report success for a failed run."""
    out = _out(tmp_path)
    _accept(out, "4h-s4", tmp_path)
    _seed_runs(out, [_run_rec("4h-s4", "B1", 2, status=status, reruns=1)])
    launched: list[int] = []
    monkeypatch.setattr(layer2, "run_discovery", lambda *a, **k: launched.append(1) or {})
    rc = budget.main([
        *_cli(tmp_path), "discover", "--variant", "4h-s4", "--rung", "B1", "--seed", "2",
    ])
    assert rc == budget.EXIT_RUN_NOT_COMPLETED
    assert launched == []  # nothing relaunched without --force
    assert "RUN NOT COMPLETED" in capsys.readouterr().out

    # a recorded completed run is still the idempotent exit 0
    _seed_runs(out, [_run_rec("4h-s4", "B1", 3, status="completed")])
    assert budget.main([
        *_cli(tmp_path), "discover", "--variant", "4h-s4", "--rung", "B1", "--seed", "3",
    ]) == 0
    assert launched == []


# --- R15b report: the prereg s0 check per run -----------------------------------------------


def _null_pool(tmp_path: Path, tf: str, corrs: list[float]) -> Path:
    """A null pool of len(corrs) entries; the highest score owns the largest |corr|, so every
    K = len(corrs) subsample picks it first and T (hence C95) is a known literal."""
    from vibe_quant.discovery.genome import chromosome_to_serializable

    chrom = chromosome_to_serializable(layer2.planted_chromosome())
    path = tmp_path / "o" / "scratch" / f"null-pool-{tf}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"seed": i, "score": float(len(corrs) - i), "corr": corr, "chromosome": chrom}
        for i, corr in enumerate(corrs)
    ]))
    return path


def test_report_prints_per_s0_run_max_corr_against_c95_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bead 4urlp: the prereg check "s0 max corr > C95(K)" was computed by hand for P-2."""
    out = _out(tmp_path)
    pool_4h = _null_pool(tmp_path, "4h", [0.70, 0.10])
    pool_1d = _null_pool(tmp_path, "1d", [0.70, 0.10])
    recs = []
    for seed, corr in ((1, 0.80), (2, 0.70)):  # above, then exactly equal (not above)
        rec = _run_rec("4h-s0", "B1", seed)
        rec["harvest"] = {"evaluated": 2, "candidates": [{"corr": corr}, {"corr": 0.20}]}
        rec["evaluated"] = 2
        recs.append(rec)
    rec = _run_rec("1d-s0", "B1", 1)
    rec["harvest"] = {"evaluated": 2, "candidates": [{"corr": 0.50}]}
    recs.append(rec)
    _seed_runs(out, recs)

    def seed(s: dict[str, Any]) -> None:
        s["null"] = {
            "4h": {"status": "completed", "m": 2, "reps": 5, "pool_path": str(pool_4h)},
            "1d": {"status": "completed", "m": 2, "reps": 5, "pool_path": str(pool_1d)},
        }

    budget.update_state(out, seed)
    assert budget.main([*_cli(tmp_path), "report"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert (
        "  s0 4h-s0 B1 seed 1: K=2 max corr 0.8000 vs C95 0.7000: ABOVE C95" in lines
    )
    assert (
        "  s0 4h-s0 B1 seed 2: K=2 max corr 0.7000 vs C95 0.7000: below C95" in lines
    )
    assert (
        "  s0 1d-s0 B1 seed 1: K=2 max corr 0.5000 vs C95 0.7000: below C95" in lines
    )


def test_report_s0_check_degrades_loudly_without_a_null_pool_or_evaluated_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    recs = []
    for seed, evaluated in ((1, None), (2, 2)):
        rec = _run_rec("4h-s0", "B1", seed)
        rec["harvest"] = {"evaluated": evaluated, "candidates": [{"corr": 0.80}]}
        recs.append(rec)
    _seed_runs(out, recs)
    assert budget.main([*_cli(tmp_path), "report"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(
        ln.startswith("  s0 4h-s0 B1 seed 1: max corr 0.8000")
        and "no evaluated count" in ln
        for ln in lines
    )
    assert any(
        ln.startswith("  s0 4h-s0 B1 seed 2: max corr 0.8000")
        and "no null pool for 4h" in ln
        for ln in lines
    )


def test_report_s0_check_ignores_probe_runs_and_non_completed_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(tmp_path)
    pool = _null_pool(tmp_path, "4h", [0.70, 0.10])
    probe = _run_rec("4h-s0", "B1", 3, probe=True)
    probe["harvest"] = {"evaluated": 2, "candidates": [{"corr": 0.99}]}
    failed = _run_rec("4h-s0", "B1", 4, status="timeout")
    _seed_runs(out, [probe, failed])
    budget.update_state(
        out, lambda s: s.__setitem__(
            "null", {"4h": {"status": "completed", "m": 2, "reps": 5, "pool_path": str(pool)}}
        ),
    )
    assert budget.main([*_cli(tmp_path), "report"]) == 0
    text = capsys.readouterr().out
    assert "s0 4h-s0" not in text  # no completed untagged s0 run: no line, no crash


def test_harvest_exit_3_when_a_run_used_another_dsr_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run launched without --dsr-rule is pinned to emax; a q95 gate means the CLI default
    leaked in (rev-dsr-t4-1, 2026-10-19), so the harvest refuses it."""
    out = _out(tmp_path)
    _accept(out, "4h-s0", tmp_path)
    rec = _run_rec("4h-s0", "B1", 1, champions=0)
    rec.update(db=str(tmp_path / "r.db"), run_id=1)
    _seed_runs(out, [rec])
    trials = {"n_rule": layer2.N_RULE_EFFECTIVE, "gate_n": 40, "dsr_rule": "q95"}
    monkeypatch.setattr(
        layer2,
        "harvest_run",
        lambda db, rid: {"r31": {"ok": True}, "dsr_trials": trials, "candidates": [], "champions": 0},
    )
    monkeypatch.setattr(layer2, "_harvest_one", lambda *a, **k: True)
    assert budget.main([*_cli(tmp_path), "harvest", "--tf", "4h"]) == 3
    assert "DSR rule 'q95' != requested 'emax'" in capsys.readouterr().out


def test_requested_config_follows_probe_and_dsr_rule() -> None:
    assert budget.requested_config({"probe": True}) == (None, "emax")
    assert budget.requested_config({"dsr_rule": "q95"}) == (layer2.NULL_INDICATOR_POOL, "q95")
