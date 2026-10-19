"""P-2 production-budget study driver: the smallest GA budget that recovers a planted LONG edge.

Job 20261018-grind-dir, task T3 (design-budget.md). The driver runs the P-2 study on top of
the profile-aware ``layer2`` functions (``P4H`` byte-identical, ``P1D`` = 1d / W6 / LONG):

- ``snapshot`` / ``verify-untouched``: R26 shared-data safety, reusing layer2's signatures.
- ``build --variant``: plant a LONG drift into a PRIVATE scratch catalog and calibrate TRAIN
  Sharpe into ``[0.9S, 1.1S]`` (<= 4 builds, ``|l_end| <= 0.005``, price ratio <= 3x).
- ``pick-rule --tf 1d``: the first rule of ``PICK_RULES`` whose 1d-s0 W6 train trades land in
  ``[80, 250]`` with every parameter inside the GA ranges; it becomes ``P1D``'s chromosome.
- ``check-btc0 --tf``: the s0 scratch catalog reproduces the real catalog bit for bit.
- ``direct-chain --variant``: the planted rule through the real gate chain (positive control).
- ``null-corr --tf``: the calibrated null pool of that timeframe (LONG draws for 1d).
- ``discover --variant --rung --seed [--probe full_pool] [--dsr-rule {emax,q95}]``: one
  isolated discovery run (``--dsr-rule`` defaults to an explicit emax pin).
- ``harvest [--tf]``: screen candidates, compute M1/M2 under ``C_K`` and advance the cells.
- ``next-wave --tf``: PRINT the exact shell commands of the next budget wave (runs nothing).
- ``report``: per-cell ``B*``, M1/M2 counts per rung, s0 champion count and the git-sha set.

State: ONE JSON (default ``data/swarm/20261018-grind-dir/power/budget.json``) written under a
flock; every subcommand is idempotent per its state key (``--force`` redoes). Exit codes mirror
layer2: 0 ok, 2 prerequisite missing, 3 machinery defect (isolation / LONG / btc0 fidelity).
It NEVER writes ``data/catalog``, ``data/archive`` or ``data/state``: it reuses layer2's
scratch guards (its own DB per run, cwd = scratch root).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layer2  # noqa: E402
import plant  # noqa: E402

from vibe_quant.discovery.genome import chromosome_to_dsl  # noqa: E402
from vibe_quant.discovery.operators import (  # noqa: E402
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path("data/swarm/20261018-grind-dir/power/budget.json")
SYMBOL = "BTCUSDT"

#: Rung -> (population, generations). The ascending budget ladder.
LADDER: dict[str, tuple[int, int]] = {
    "B1": (16, 6), "B2": (24, 10), "B3": (36, 14), "B4": (50, 20),
}
LADDER_KEYS: tuple[str, ...] = ("B1", "B2", "B3", "B4")
SEEDS: tuple[int, ...] = (1, 2, 3)

#: Variant -> planted TRAIN-Sharpe target (0.0 = the s0 control, no plant).
VARIANTS: dict[str, float] = {
    "4h-s0": 0.0, "4h-s2": 2.0, "4h-s3": 3.0, "4h-s4": 4.0, "4h-s5": 5.0,
    "1d-s0": 0.0, "1d-s3": 3.0, "1d-s4": 4.0, "1d-s5": 5.0,
}

ACCEPT_TOL = 0.1  # R1: TRAIN S within [0.9S, 1.1S]
DEFAULT_DELTA0 = 2e-5
MAX_BUILDS = 4
PRICE_RATIO_LIMIT = 3.0
#: per-timeframe discovery timeout (design-budget.md § Approach).
TIMEOUT_S: dict[str, float] = {"4h": 2400.0, "1d": 3600.0}

#: The one full-pool probe (user answer Q1): 4h S4 at B* and the next rung, 3 seeds.
PROBE_FULL_POOL = "full_pool"
PROBE_TF = "4h"
PROBE_VARIANT = "4h-s4"

#: Confidence fields the btc0 fidelity check bit-compares (R3).
CONF_FIELDS = ("sharpe", "trades", "return", "profit_factor", "max_drawdown")

#: A cell stops at the first rung with M2 in >= this many seeds (2 of 3, or 2 of 2 in two-seed mode).
MIN_M2 = 2

PICK_TRADES_MIN = 80
PICK_TRADES_MAX = 250
#: The 1d plant rules, tried in order by ``pick-rule`` (design-budget.md § 1d plant rule).
PICK_RULES: list[dict[str, Any]] = [
    {"name": "rsi14_xb30_xa60_sl4_tp8", "period": 14, "xb": 30.0, "xa": 60.0, "sl": 4.0, "tp": 8.0},
    {"name": "rsi7_xb30_xa60_sl4_tp8", "period": 7, "xb": 30.0, "xa": 60.0, "sl": 4.0, "tp": 8.0},
    {"name": "rsi7_xb35_xa60_sl4_tp8", "period": 7, "xb": 35.0, "xa": 60.0, "sl": 4.0, "tp": 8.0},
    {"name": "rsi5_xb35_xa60_sl4_tp8", "period": 5, "xb": 35.0, "xa": 60.0, "sl": 4.0, "tp": 8.0},
]

#: The study searches LONG only (R4) on both timeframes; layer2's legacy P4H stays direction=both.
P4H_LONG = replace(layer2.P4H, direction="long")

load_state = layer2.load_state
update_state = layer2.update_state
build_root = layer2.build_root


class IsolationViolation(RuntimeError):
    """R31/LONG machinery defect of a harvested run: exit 3, score nothing."""


# --- provenance -----------------------------------------------------------------------------


def git_sha() -> str:
    """``git rev-parse HEAD`` of the worktree the driver runs in ("" when git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


# --- variants / profiles --------------------------------------------------------------------


def tf_of(variant: str) -> str:
    return variant.split("-", 1)[0]


def variants_for(tf: str) -> list[str]:
    return [v for v in VARIANTS if tf_of(v) == tf]


def s0_variant(tf: str) -> str:
    return f"{tf}-s0"


def variant_target(variant: str) -> float:
    return VARIANTS[variant]


def rule_chromosome(rule: dict[str, Any]) -> StrategyChromosome:
    """The LONG chromosome of one 1d pick rule (RSI entry cross-below, exit cross-above)."""
    period = float(rule["period"])
    return StrategyChromosome(
        entry_genes=[
            StrategyGene("RSI", {"period": period}, ConditionType.CROSSES_BELOW, float(rule["xb"]), None)
        ],
        exit_genes=[
            StrategyGene("RSI", {"period": period}, ConditionType.CROSSES_ABOVE, float(rule["xa"]), None)
        ],
        stop_loss_pct=float(rule["sl"]),
        take_profit_pct=float(rule["tp"]),
        direction=Direction.LONG,
        uid=f"planted_{rule['name']}",
    )


def rule_dsl(rule: dict[str, Any], profile: layer2.Profile) -> dict[str, object]:
    dsl = chromosome_to_dsl(rule_chromosome(rule))
    dsl["timeframe"] = profile.timeframe
    return dsl


def rule_in_ga_ranges(rule: dict[str, Any]) -> bool:
    """R12: every parameter of the rule inside the GA ranges (RSI period/threshold, SL, TP)."""
    from vibe_quant.discovery import operators as op

    op._ensure_pool()
    plo, phi = op.INDICATOR_POOL["RSI"]["period"]
    tlo, thi = op.THRESHOLD_RANGES["RSI"]
    slo, shi = op.SL_RANGE
    tplo, tphi = op.TP_RANGE
    return bool(
        plo <= float(rule["period"]) <= phi
        and tlo <= float(rule["xb"]) <= thi
        and tlo <= float(rule["xa"]) <= thi
        and slo <= float(rule["sl"]) <= shi
        and tplo <= float(rule["tp"]) <= tphi
    )


def p1d_from_rule(rule: dict[str, Any]) -> layer2.Profile:
    """The 1d profile whose chromosome AND plant_kw come from the picked rule."""
    return replace(
        layer2.P1D,
        chromosome=rule_chromosome(rule),
        plant_kw={**layer2.P1D.plant_kw, "rsi_period": int(rule["period"]), "rsi_level": float(rule["xb"])},
    )


def profile_for(state: dict[str, Any], tf: str) -> layer2.Profile:
    """The study profile of a timeframe: P4H, or P1D once ``pick-rule`` chose the 1d rule."""
    if tf == "4h":
        return P4H_LONG
    picked = state.get("picked_rule")
    if isinstance(picked, dict):
        return p1d_from_rule(picked)
    return layer2.P1D


def is_long_dsl(dsl: dict[str, Any]) -> bool:
    """R4: a LONG candidate has a ``long`` leg and no ``short`` leg in either condition set."""
    entry = dsl.get("entry_conditions") or {}
    exit_ = dsl.get("exit_conditions") or {}
    if "short" in entry or "short" in exit_:
        return False
    return "long" in entry and "long" in exit_


# --- build acceptance / calibration ---------------------------------------------------------


def accepts(variant: str, rec: dict[str, Any]) -> bool:
    """R1: an s0 build accepts once converged; a planted build only in ``[0.9S, 1.1S]``."""
    if rec.get("converged") is not True or rec.get("price_ratio_gt_3"):
        return False
    target = variant_target(variant)
    if target == 0.0:
        return True
    train_s = rec.get("train_S")
    if train_s is None:
        return False
    return abs(float(train_s) - target) <= ACCEPT_TOL * target


def next_delta(target: float, builds: list[dict[str, Any]], delta0: float) -> float:
    """Secant on delta toward the TRAIN target; proportional with one point; clamped 0.25x..4x."""
    pts = [
        (float(b["delta"]), float(b["train_S"]))
        for b in builds
        if b.get("train_S") is not None and b["delta"] > 0
    ]
    if not pts:
        return delta0 if not builds else float(builds[-1]["delta"]) * 2.0
    d2, s2 = pts[-1]
    guess = d2 * target / s2 if s2 > 0 else d2 * 2.0
    if len(pts) >= 2:
        d1, s1 = pts[-2]
        if s2 != s1:
            sec = d2 - (s2 - target) * (d2 - d1) / (s2 - s1)
            if sec > 0.0:
                guess = sec
    return float(min(max(guess, d2 / 4.0), d2 * 4.0))


# --- scoring (M1/M2) ------------------------------------------------------------------------


def score_candidates(
    candidates: list[dict[str, Any]], planted_s: float | None, c_k: float
) -> dict[str, bool]:
    """M1 = any candidate clears ``C_K`` and ``0.5 * planted``; M2 = a CHAMPION does (primary)."""

    def hit(cand: dict[str, Any]) -> bool:
        return layer2.found_criterion(cand.get("corr"), cand.get("train_S"), planted_s, corr_min=c_k)

    m1 = any(hit(c) for c in candidates)
    m2 = any(hit(c) for c in candidates if c.get("source") == "champion")
    return {"M1": bool(m1), "M2": bool(m2)}


def k_for_run(run: dict[str, Any], m: int) -> int:
    """R7: the run's evaluated count, refused when absent or larger than the null pool."""
    evaluated = (run.get("harvest") or {}).get("evaluated")
    if evaluated is None:
        msg = f"run {run.get('variant')} {run.get('rung')} seed {run.get('seed')} has no evaluated count"
        raise layer2.Layer2Error(msg)
    k = int(evaluated)
    if k > m:
        msg = f"K {k} exceeds the null pool of {m}; draw a larger null pool"
        raise layer2.Layer2Error(msg)
    return k


def s0_max_signed_corr(state: dict[str, Any], tf: str, pop: int, gens: int) -> float:
    """R8: the max SIGNED corr over completed s0 runs of THIS tf at this (pop, gens). Absent -> exit 2."""
    corrs: list[float] = []
    for run in all_runs(state):
        if run.get("variant") != s0_variant(tf) or run.get("status") != "completed":
            continue
        if is_probe_run(run) or LADDER.get(str(run.get("rung"))) != (pop, gens):
            continue
        for cand in (run.get("harvest") or {}).get("candidates") or []:
            if cand.get("corr") is not None:
                corrs.append(float(cand["corr"]))
    if not corrs:
        msg = f"no harvested s0 run of {tf} at budget {pop}x{gens} for C_K; run s0 at that rung first"
        raise layer2.Layer2Error(msg)
    return max(corrs)


# --- state helpers --------------------------------------------------------------------------


def run_key(variant: str, rung: str, seed: int, probe: bool = False) -> str:
    return f"{variant}|{rung}|{seed}" + ("|probe" if probe else "")


def is_probe_run(run: dict[str, Any]) -> bool:
    return bool(run.get("probe"))


def all_runs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list((state.get("runs") or {}).values())


def tf_seeds(state: dict[str, Any], tf: str) -> int:
    return int((state.get("seeds") or {}).get(tf, len(SEEDS)))


def get_run(state: dict[str, Any], variant: str, rung: str, seed: int, probe: bool = False) -> dict[str, Any] | None:
    return (state.get("runs") or {}).get(run_key(variant, rung, seed, probe))


def _completed(state: dict[str, Any], variant: str, rung: str, seed: int) -> bool:
    run = get_run(state, variant, rung, seed)
    return run is not None and run.get("status") == "completed"


def _given_up(state: dict[str, Any], variant: str, rung: str, seed: int) -> bool:
    run = get_run(state, variant, rung, seed)
    return (
        run is not None
        and run.get("status") in ("failed", "timeout")
        and int(run.get("reruns", 0)) >= 1
    )


def _pending(state: dict[str, Any], variant: str, rung: str, seed: int, probe: bool = False) -> bool:
    run = get_run(state, variant, rung, seed, probe)
    if run is None:
        return True
    if run.get("status") == "completed":
        return False
    return int(run.get("reruns", 0)) < 1


def m1_count(state: dict[str, Any], variant: str, rung: str) -> int:
    return sum(
        1
        for s in range(1, tf_seeds(state, tf_of(variant)) + 1)
        if (r := get_run(state, variant, rung, s)) is not None
        and r.get("status") == "completed"
        and (r.get("score") or {}).get("M1") is True
    )


def m2_count(state: dict[str, Any], variant: str, rung: str) -> int:
    return sum(
        1
        for s in range(1, tf_seeds(state, tf_of(variant)) + 1)
        if (r := get_run(state, variant, rung, s)) is not None
        and r.get("status") == "completed"
        and (r.get("score") or {}).get("M2") is True
    )


def _b_star(state: dict[str, Any], variant: str) -> str | None:
    """The first rung where EVERY seed completed and M2 held in at least MIN_M2 of them."""
    seeds = range(1, tf_seeds(state, tf_of(variant)) + 1)
    for k in LADDER_KEYS:
        if all(_completed(state, variant, k, s) for s in seeds) and m2_count(state, variant, k) >= MIN_M2:
            return k
    return None


def cell_rung(state: dict[str, Any], variant: str) -> str | None:
    """The first rung not yet fully completed by every seed (None = every rung is done)."""
    seeds = range(1, tf_seeds(state, tf_of(variant)) + 1)
    for k in LADDER_KEYS:
        if any(not _completed(state, variant, k, s) for s in seeds):
            return k
    return None


def gate_leak(state: dict[str, Any], tf: str) -> bool:
    """R11: an s0 run that produced a champion is a machinery gate leak."""
    return any(
        r.get("status") == "completed"
        and int((r.get("harvest") or {}).get("champions", 0)) > 0
        for r in all_runs(state)
        if r.get("variant") == s0_variant(tf) and not is_probe_run(r)
    )


def refresh_cells(state: dict[str, Any], tf: str) -> None:
    """Recompute ``state['cells']`` for one timeframe from its runs."""
    cells = state.setdefault("cells", {})
    seeds = range(1, tf_seeds(state, tf) + 1)
    for variant in variants_for(tf):
        b_star = _b_star(state, variant)
        rung = cell_rung(state, variant)
        if b_star is not None:
            status = "stopped"
        elif rung is not None and any(_given_up(state, variant, rung, s) for s in seeds):
            status = "incomplete"
        else:
            status = "open"
        cells[variant] = {"tf": tf, "B_star": b_star, "rung": rung, "status": status}


def next_wave(state: dict[str, Any], tf: str) -> list[tuple[str, str, int, bool]]:
    """The still-pending runs of the next wave: the lowest rung with an open cell (+ that tf's s0),
    then the full-pool probe once the primary waves are done. Empty on a gate leak."""
    refresh_cells(state, tf)
    if gate_leak(state, tf):
        return []
    cells = state.get("cells") or {}
    seeds = range(1, tf_seeds(state, tf) + 1)
    for k in LADDER_KEYS:
        open_here = any(
            variant != s0_variant(tf)
            and (cells.get(variant) or {}).get("status") == "open"
            and (cells.get(variant) or {}).get("rung") == k
            for variant in variants_for(tf)
        )
        if not open_here:
            continue
        pending = [
            (variant, k, s, False)
            for variant in variants_for(tf)
            if (cells.get(variant) or {}).get("status") == "open"
            and (cells.get(variant) or {}).get("rung") == k
            for s in seeds
            if _pending(state, variant, k, s)
        ]
        if pending:
            return pending
    if tf == PROBE_TF:
        b_star = (cells.get(PROBE_VARIANT) or {}).get("B_star")
        if b_star:
            rungs = [b_star]
            i = LADDER_KEYS.index(b_star)
            if i + 1 < len(LADDER_KEYS):
                rungs.append(LADDER_KEYS[i + 1])
            probe = [
                (PROBE_VARIANT, k, s, True)
                for k in rungs
                for s in seeds
                if _pending(state, PROBE_VARIANT, k, s, probe=True)
            ]
            if probe:
                return probe
    return []


def next_wave_commands(state: dict[str, Any], tf: str, ctx: layer2.Ctx) -> list[str]:
    """R13: the exact shell commands of the next wave (prints them; runs nothing)."""
    cmds = []
    for variant, rung, seed, probe in next_wave(state, tf):
        tail = " --probe full_pool" if probe else ""
        # a recorded failed/timed-out run is skipped unless the retry is forced
        force = " --force" if get_run(state, variant, rung, seed, probe) is not None else ""
        cmds.append(
            f"PYTHONPATH=$PWD {sys.executable} scripts/research/power/budget.py"
            f" --out {ctx.out} --scratch {ctx.scratch}{force} discover"
            f" --variant {variant} --rung {rung} --seed {seed}{tail}"
        )
    return cmds


def report_lines(state: dict[str, Any]) -> list[str]:
    """R15: per-cell B*, M1/M2 counts per rung, s0 champion count and the git-sha set."""
    lines = ["budget report"]
    shas = sorted({str(r.get("sha")) for r in all_runs(state) if r.get("sha")})
    lines.append("git shas: " + (", ".join(shas) if shas else "(none)"))
    for tf in ("4h", "1d"):
        if not any(tf_of(r["variant"]) == tf for r in all_runs(state)):
            continue
        refresh_cells(state, tf)
        n = tf_seeds(state, tf)
        lines.append(f"tf {tf} seeds={n}")
        if gate_leak(state, tf):
            lines.append(f"  GATE LEAK: {tf} s0 has a champion")
        for variant in variants_for(tf):
            cell = (state.get("cells") or {}).get(variant, {})
            parts = " | ".join(
                f"{k} M1={m1_count(state, variant, k)}/{n} M2={m2_count(state, variant, k)}/{n}"
                for k in LADDER_KEYS
            )
            lines.append(f"  {variant}: B*={cell.get('B_star') or 'none'} | {parts}")
        champion = sum(
            int((r.get("harvest") or {}).get("champions", 0))
            for r in all_runs(state)
            if r["variant"] == s0_variant(tf) and r.get("status") == "completed"
        )
        lines.append(f"  s0 champion count: {champion}")
    return lines


def picked_rule_name(state: dict[str, Any]) -> str | None:
    picked = state.get("picked_rule")
    return str(picked["name"]) if isinstance(picked, dict) else None


def rule_mismatch(state: dict[str, Any], variant: str) -> bool:
    """R34: a planted 1d variant whose accepted build used another rule than the current pick."""
    if tf_of(variant) != "1d" or variant_target(variant) == 0.0:
        return False
    vrec = (state.get("variants") or {}).get(variant) or {}
    return bool(vrec.get("accepted")) and vrec.get("rule") != picked_rule_name(state)


# --- invalidation (a rebuilt catalog voids everything derived from the old one) -------------


def invalidate_tf_scores(s: dict[str, Any], tf: str) -> None:
    """C_K depends on the tf's s0 and the picked rule: drop the null pool, the btc0 check and
    every score of the tf (the next harvest re-scores), then refresh the cells."""
    (s.get("null") or {}).pop(tf, None)
    (s.get("check_btc0") or {}).pop(tf, None)
    for run in (s.get("runs") or {}).values():
        if tf_of(run["variant"]) == tf:
            run.pop("score", None)
    refresh_cells(s, tf)


def invalidate_variant(s: dict[str, Any], variant: str) -> None:
    """A forced build of ``variant`` deletes its runs, cell, direct chain and build record; a
    rebuilt s0 also voids the tf-level dependents."""
    runs = s.get("runs") or {}
    for key in [k for k, r in runs.items() if r["variant"] == variant]:
        del runs[key]
    (s.get("direct_chain") or {}).pop(variant, None)
    (s.get("cells") or {}).pop(variant, None)
    v = s.setdefault("variants", {}).setdefault(variant, {})
    for key in ("accepted", "build_id", "builds"):
        v.pop(key, None)
    v["status"] = "building"
    if variant == s0_variant(tf_of(variant)):
        invalidate_tf_scores(s, tf_of(variant))


# --- discovery argv -------------------------------------------------------------------------


def discovery_argv(
    profile: layer2.Profile,
    run_id: int,
    db: Path,
    seed: int,
    rung: str,
    probe: bool = False,
    dsr_rule: str | None = None,
) -> list[str]:
    """The discovery CLI argv for one wave run; ``probe`` drops the 5-indicator restriction and
    ``dsr_rule`` appends ``--dsr-rule <rule>`` (``None`` = explicit ``emax`` pin)."""
    pop, gens = LADDER[rung]
    cmd = layer2.discovery_command(
        run_id, db, seed, pop=pop, gens=gens, profile=profile, dsr_rule=dsr_rule
    )
    if probe:
        i = cmd.index("--indicator-pool")
        del cmd[i : i + 2]
    return cmd


# --- subcommands ----------------------------------------------------------------------------


def cmd_snapshot(ctx: layer2.Ctx, _args: argparse.Namespace) -> int:
    if "snapshot" in load_state(ctx.out) and not ctx.force:
        print("snapshot already recorded (kept; --force re-records)")
        return 0
    snap = layer2.take_snapshot(ctx.data_root, ctx.real_db, ctx.archive)
    update_state(ctx.out, lambda s: s.__setitem__("snapshot", snap))
    print(f"snapshot: {len(snap['files'])} files, discovery rows {snap['discovery_rows']}")
    return 0


def cmd_build(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    variant = args.variant
    state = load_state(ctx.out)
    vrec = state.get("variants", {}).get(variant, {})
    if vrec.get("accepted") and Path(vrec["accepted"]["root"]).exists() and not ctx.force:
        print(f"{variant}: accepted build {vrec['accepted']['k']} already recorded")
        return 0
    if vrec.get("status") == "failed_price_ratio" and not ctx.force:
        print(f"{variant}: FAILED closed (price ratio > 3x); --force rebuilds")
        return 2
    target = variant_target(variant)
    profile = profile_for(state, tf_of(variant))
    rule_name = picked_rule_name(state)  # recorded on the accepted build (R34)
    builds: list[dict[str, Any]] = [] if ctx.force else list(vrec.get("builds", []))

    def _invalidate(s: dict[str, Any]) -> None:
        invalidate_variant(s, variant)

    update_state(ctx.out, _invalidate)
    klines = plant.load_klines(ctx.archive, start=profile.kline_start, end=profile.plant_span[1])
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    limit = 1 if target == 0.0 else MAX_BUILDS
    accepted: dict[str, Any] | None = None
    while len(builds) < limit:
        k = len(builds) + 1
        delta0 = args.delta0 if args.delta0 is not None else DEFAULT_DELTA0
        delta = 0.0 if target == 0.0 else next_delta(target, builds, delta0)
        rec = layer2.build_one(
            variant, k, delta, klines, ctx.archive, ctx.scratch, ctx.scratch_db, profile
        )
        rec["accepted"] = accepts(variant, rec)
        builds.append(rec)

        def _save(s: dict[str, Any], r: dict[str, Any] = rec) -> None:
            v = s.setdefault("variants", {}).setdefault(variant, {})
            v["builds"] = list(builds)
            v["status"] = "accepted" if r["accepted"] else "calibrating"
            v["rule"] = rule_name if tf_of(variant) == "1d" else None
            if r["accepted"]:
                v["accepted"] = r
                v["build_id"] = uuid.uuid4().hex  # runs are stamped with it; harvest checks it

        update_state(ctx.out, _save)
        print(
            f"{variant} b{k}: delta={delta:.3e} l_end={rec['l_end']:.4f} converged={rec['converged']} "
            f"ratio={rec['price_ratio']:.2f} train_S={rec['train_S']} accepted={rec['accepted']}"
        )
        if rec["price_ratio_gt_3"]:
            for b in builds:
                shutil.rmtree(b["root"], ignore_errors=True)
            update_state(
                ctx.out, lambda s: s.setdefault("variants", {}).setdefault(variant, {}).__setitem__(
                    "status", "failed_price_ratio"
                )
            )
            print(f"{variant}: FAILED closed, price ratio {rec['price_ratio']:.2f} > 3x")
            return 2
        if rec["accepted"]:
            accepted = rec
            break
    if accepted is None:
        update_state(
            ctx.out, lambda s: s.setdefault("variants", {}).setdefault(variant, {}).__setitem__(
                "status", "not_calibrated"
            )
        )
        print(f"{variant}: NOT calibrated after {len(builds)} builds")
        return 2
    for b in builds:
        root = Path(b["root"])
        if b is not accepted and root.exists() and layer2._under(root, ctx.scratch):
            shutil.rmtree(root)
    return 0


def cmd_pick_rule(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    tf = args.tf
    if tf != "1d":
        msg = "pick-rule is defined for --tf 1d only"
        raise layer2.Layer2Error(msg)
    state = load_state(ctx.out)
    if isinstance(state.get("picked_rule"), dict) and not ctx.force:
        print(f"pick-rule: {state['picked_rule']['name']} already recorded")
        return 0
    profile = layer2.P1D
    root = ctx.accepted_root(state, s0_variant(tf))
    start, end = layer2.train_window(profile)
    jobs = [
        {"tag": r["name"], "dsl": rule_dsl(r, profile), "start": start, "end": end, "collect": False}
        for r in PICK_RULES
    ]
    shots = {r["tag"]: r for r in layer2.run_worker("screen", root, {"jobs": jobs}, ctx.scratch_db)}
    chosen: dict[str, Any] | None = None
    for rule in PICK_RULES:
        trades = int(shots[rule["name"]]["trades"])
        if PICK_TRADES_MIN <= trades <= PICK_TRADES_MAX and rule_in_ga_ranges(rule):
            chosen = {**rule, "trades": trades}
            break
    if chosen is None:
        print(
            f"pick-rule: no rule with 1d-s0 W6 train trades in "
            f"[{PICK_TRADES_MIN}, {PICK_TRADES_MAX}] and params in the GA ranges"
        )
        return 2
    rec = {**chosen, "profile": p1d_from_rule(chosen).to_dict(), "recorded_at": layer2._now()}
    changed = (state.get("picked_rule") or {}).get("name") != chosen["name"]

    def _save_rule(s: dict[str, Any]) -> None:
        s["picked_rule"] = rec
        if changed:
            for v in variants_for(tf):
                vrec = (s.get("variants") or {}).get(v)
                if vrec is None or variant_target(v) == 0.0 or vrec.get("rule") == chosen["name"]:
                    continue
                if vrec.get("accepted"):
                    invalidate_variant(s, v)
                else:
                    vrec.pop("builds", None)  # the 4-build cap is per rule (R34e)
            invalidate_tf_scores(s, tf)

    update_state(ctx.out, _save_rule)
    print(f"pick-rule: {chosen['name']} (trades={chosen['trades']})")
    return 0


def cmd_check_btc0(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    tf = args.tf
    state = load_state(ctx.out)
    existing = (state.get("check_btc0") or {}).get(tf)
    if isinstance(existing, dict) and existing.get("passed") is True and not ctx.force:
        print(f"check-btc0 {tf} already passed")
        return 0
    profile = profile_for(state, tf)
    root = ctx.accepted_root(state, s0_variant(tf))
    start, end = layer2.full_window(profile)
    jobs = [{"tag": "full", "dsl": layer2.planted_dsl(profile), "start": start, "end": end}]
    real = layer2.run_worker("screen", ctx.data_root.parent, {"jobs": jobs}, ctx.scratch_db)[0]
    scratch = layer2.run_worker("screen", root, {"jobs": jobs}, ctx.scratch_db)[0]
    want = {k: real[k] for k in CONF_FIELDS}
    diffs = layer2.compare_exact(scratch, want)
    passed = not diffs
    rec = {
        "passed": passed, "diffs": diffs, "got": {k: scratch[k] for k in CONF_FIELDS},
        "real": want, "checked_at": layer2._now(),
    }
    update_state(ctx.out, lambda s: s.setdefault("check_btc0", {}).__setitem__(tf, rec))
    print(f"R3 {tf} btc0 fidelity: {'IDENTICAL' if passed else 'DRIFT'}")
    for diff in diffs:
        print(f"  {diff}")
    return 0 if passed else 3


def cmd_direct_chain(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    variant = args.variant
    state = load_state(ctx.out)
    existing = (state.get("direct_chain") or {}).get(variant)
    if isinstance(existing, dict) and not ctx.force:
        print(f"{variant}: direct chain already recorded")
        return _chain_exit(existing, variant)
    root = ctx.accepted_root(state, variant)
    profile = profile_for(state, tf_of(variant))
    out = layer2.run_worker("chain", root, {"profile": profile.to_dict()}, ctx.scratch_db)
    result = json.loads(json.dumps(out["result"], default=layer2._json_default))
    failures = layer2.positive_control_failures(result, layer2.POSITIVE_CONTROL_NS)
    rec = {
        "fitness": out["fitness"], "holdout": out["holdout"], "result": result,
        "failures": failures, "checked_at": layer2._now(),
    }
    update_state(ctx.out, lambda s: s.setdefault("direct_chain", {}).__setitem__(variant, rec))
    print(f"{variant} direct chain: failures={failures}")
    return _chain_exit(rec, variant)


def _chain_exit(rec: dict[str, Any], variant: str) -> int:
    """Only the tf's s5 (the positive control) must pass; its failure is a machinery defect."""
    if variant == f"{tf_of(variant)}-s5" and rec.get("failures"):
        print(f"direct chain positive control FAILED: {', '.join(rec['failures'])}")
        return 3
    return 0


def cmd_null_corr(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    tf = args.tf
    state = load_state(ctx.out)
    existing = (state.get("null") or {}).get(tf)
    if isinstance(existing, dict) and existing.get("status") == "completed" and not ctx.force:
        print(f"null-corr {tf} already recorded (kept; --force redoes)")
        return 0
    profile = profile_for(state, tf)
    root = ctx.accepted_root(state, s0_variant(tf))
    m, shards = int(args.m), int(args.shards)
    chunk = -(-m // shards)

    def shard(i: int, count: int) -> list[dict[str, Any]]:
        return layer2.run_worker(
            "null", root,
            {"count": count, "seed_base": int(args.seed), "offset": i * chunk,
             "pool": list(layer2.NULL_INDICATOR_POOL), "profile": profile.to_dict()},
            ctx.scratch_db, timeout_s=args.timeout,
        )

    jobs = [(i, max(0, min(chunk, m - i * chunk))) for i in range(shards)]
    jobs = [(i, count) for i, count in jobs if count]
    pool: list[dict[str, Any]] = []
    if jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool_ex:
            futures = [pool_ex.submit(shard, i, count) for i, count in jobs]
            pool = [entry for fut in futures for entry in fut.result()]
    pool.sort(key=lambda entry: entry["seed"])
    pool_path = ctx.scratch / f"null-pool-{tf}.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(pool, default=layer2._json_default) + "\n")
    rec = {
        "status": "completed", "m": len(pool), "reps": int(args.reps), "seed": int(args.seed),
        "shards": shards, "root": str(root), "pool_path": str(pool_path), "tf": tf,
        "recorded_at": layer2._now(),
    }
    update_state(ctx.out, lambda s: s.setdefault("null", {}).__setitem__(tf, rec))
    print(f"null-corr {tf}: m={len(pool)} reps={rec['reps']} root={root} pool={pool_path}")
    return 0


def cmd_discover(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    variant, rung, seed = args.variant, args.rung, int(args.seed)
    probe = args.probe == PROBE_FULL_POOL
    if variant not in VARIANTS:
        msg = f"unknown variant {variant!r}"
        raise layer2.Layer2Error(msg)
    if rung not in LADDER:
        msg = f"unknown rung {rung!r}"
        raise layer2.Layer2Error(msg)
    tf = tf_of(variant)
    key = run_key(variant, rung, seed, probe)
    state = load_state(ctx.out)
    if rule_mismatch(state, variant):
        msg = f"{variant} was built with another pick rule than {picked_rule_name(state)!r}; rebuild it"
        raise layer2.Layer2Error(msg)
    existing = (state.get("runs") or {}).get(key)
    where = f"{variant} {rung} seed {seed}" + (" [probe]" if probe else "")
    if existing and not ctx.force:
        print(f"{where}: already recorded ({existing['status']})")
        return 0
    reruns = int(existing.get("reruns", 0)) + 1 if existing else 0
    profile = profile_for(state, tf)
    root = ctx.accepted_root(state, variant)
    pop, gens = LADDER[rung]
    run_dir = ctx.scratch / "runs" / (f"{variant}-{rung}-seed{seed}" + ("-probe" if probe else ""))
    if not layer2._under(run_dir, ctx.scratch):
        msg = f"refusing run dir {run_dir}: outside {ctx.scratch}"
        raise layer2.Layer2Error(msg)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def command(run_id: int, db: Path, sd: int) -> list[str]:
        return discovery_argv(profile, run_id, db, sd, rung, probe, args.dsr_rule)

    timeout = args.timeout if args.timeout is not None else TIMEOUT_S[tf]
    rec = layer2.run_discovery(
        root, run_dir, seed, timeout_s=timeout, command_fn=command, profile=profile
    )
    rec.update(
        variant=variant, rung=rung, seed=seed, pop=pop, gens=gens, probe=probe,
        profile=profile.to_dict(), sha=git_sha(), reruns=reruns, evaluated=None,
        build_id=(state.get("variants") or {}).get(variant, {}).get("build_id"),
        dsr_rule=args.dsr_rule,
    )
    update_state(ctx.out, lambda s: s.setdefault("runs", {}).__setitem__(key, rec))
    print(f"{where}: {rec['status']} run_id={rec['run_id']} {rec['wall_s']}s {rec.get('error', '')}".rstrip())
    return 0


def harvest_tf(ctx: layer2.Ctx, state: dict[str, Any], tf: str) -> int:
    """Harvest every completed run of a tf, score the non-s0/non-probe ones, refresh the cells."""
    runs = [r for r in all_runs(state) if tf_of(r["variant"]) == tf and r.get("status") == "completed"]
    for run in runs:
        h = layer2.harvest_run(Path(run["db"]), int(run["run_id"]))
        current = ((state.get("variants") or {}).get(run["variant"]) or {}).get("build_id")
        if run.get("build_id") != current:
            print(
                f"harvest {tf}: {run['variant']} {run['rung']} seed {run['seed']} was run on build "
                f"{run.get('build_id')!r} but the current build is {current!r}; re-run it"
            )
            return 3
        if rule_mismatch(state, run["variant"]):
            print(
                f"harvest {tf}: {run['variant']} was built with another pick rule than "
                f"{picked_rule_name(state)!r}; rebuild it"
            )
            return 3
        n_rule = (h.get("dsr_trials") or {}).get("n_rule")
        if not (h.get("r31") or {}).get("ok") or n_rule != layer2.N_RULE_EFFECTIVE:
            print(
                f"harvest {tf}: R31 isolation / effective-N marker violated in {run['variant']} "
                f"{run['rung']} seed {run['seed']}: {h.get('r31')}"
            )
            return 3
        bad = [c for c in h.get("candidates") or [] if not is_long_dsl(c.get("dsl") or {})]
        if bad:
            print(
                f"harvest {tf}: non-LONG candidate {bad[0].get('uid')!r} in {run['variant']} "
                f"{run['rung']} seed {run['seed']}"
            )
            return 3
    profile = profile_for(state, tf)
    for run in runs:
        layer2._harvest_one(ctx, ctx.accepted_root(state, run["variant"]), run, profile)
    null = (state.get("null") or {}).get(tf)
    pool: list[tuple[StrategyChromosome, float, float | None]] | None = None
    reps = 0
    c95_cache: dict[int, float] = {}
    for run in runs:
        run["evaluated"] = (run.get("harvest") or {}).get("evaluated")
        if run["variant"] == s0_variant(tf) or is_probe_run(run):
            continue
        if not isinstance(null, dict) or null.get("status") != "completed":
            msg = (
                f"harvest {tf} needs the calibrated null record (state['null']['{tf}']): "
                f"run `null-corr --tf {tf}` first"
            )
            raise layer2.Layer2Error(msg)
        if pool is None:
            pool = layer2.load_null_pool(Path(null["pool_path"]))
            reps = int(null["reps"])
        m = int(null["m"])
        k = k_for_run(run, m)
        if k not in c95_cache:
            c95_cache[k] = float(layer2.null_threshold(pool, k, reps)["c95"])
        pop, gens = LADDER[run["rung"]]
        s0_max = s0_max_signed_corr(state, tf, pop, gens)
        c_k = layer2.corr_threshold(c95_cache[k], s0_max)
        planted_s = ((run.get("harvest") or {}).get("found") or {}).get("planted_train_S")
        run["score"] = {
            **score_candidates(run["harvest"]["candidates"], planted_s, c_k),
            "K": k, "c95": c95_cache[k], "s0_max": s0_max, "C_K": c_k,
        }

    keys = [run_key(r["variant"], r["rung"], r["seed"], is_probe_run(r)) for r in runs]

    def _save(s: dict[str, Any]) -> None:
        for key in keys:
            s.setdefault("runs", {})[key] = state["runs"][key]
        refresh_cells(s, tf)

    update_state(ctx.out, _save)
    return 0


def cmd_harvest(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    state = load_state(ctx.out)
    tfs = [args.tf] if args.tf else sorted({tf_of(r["variant"]) for r in all_runs(state)})
    for tf in tfs:
        rc = harvest_tf(ctx, state, tf)
        if rc != 0:
            return rc
    return 0


def cmd_next_wave(ctx: layer2.Ctx, args: argparse.Namespace) -> int:
    tf = args.tf
    if args.seeds is not None:
        update_state(
            ctx.out, lambda s: s.setdefault("seeds", {}).__setitem__(tf, int(args.seeds))
        )
    state = load_state(ctx.out)
    for line in next_wave_commands(state, tf, ctx):
        print(line)
    return 0


def cmd_report(_ctx: layer2.Ctx, _args: argparse.Namespace) -> int:
    for line in report_lines(load_state(_ctx.out)):
        print(line)
    return 0


def cmd_verify_untouched(ctx: layer2.Ctx, _args: argparse.Namespace) -> int:
    state = load_state(ctx.out)
    snap = state.get("snapshot")
    if snap is None:
        print("no snapshot recorded; run `snapshot` first")
        return 2
    problems = layer2.compare_snapshot(snap, ctx.data_root, ctx.real_db, ctx.archive)
    for variant, vrec in state.get("variants", {}).items():
        acc = vrec.get("accepted")
        scratch_arch = Path(acc["root"]) / "data" / "archive" / "raw_data.db" if acc else None
        if scratch_arch is None or not scratch_arch.exists():
            continue
        count = layer2.funding_count(scratch_arch)
        if count != snap["funding_source_count"]:
            problems.append(f"{variant}: scratch funding rows {count} != source {snap['funding_source_count']}")
    rec = {"ok": not problems, "problems": problems, "checked_at": layer2._now()}
    update_state(ctx.out, lambda s: s.__setitem__("verify", rec))
    print("R26 shared data untouched" if not problems else "R26 shared data CHANGED")
    for problem in problems:
        print(f"  {problem}")
    return 0 if not problems else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="the one JSON all steps write")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--real-db", default=str(layer2.REAL_DB), help="read-only (R26 rows)")
    parser.add_argument("--archive", default=None, help="source raw_data.db (default data-root)")
    parser.add_argument("--scratch", default=None, help="scratch dir (default <out dir>/scratch)")
    parser.add_argument("--force", action="store_true", help="redo a recorded step")
    sub = parser.add_subparsers(dest="cmd", required=True)
    tfs = sorted({tf_of(v) for v in VARIANTS})

    sub.add_parser("snapshot", help="R26: record signatures before any build")
    p = sub.add_parser("pick-rule", help="the first 1d plant rule inside the trades/GA window")
    p.add_argument("--tf", choices=tfs, default="1d")
    p = sub.add_parser("build", help="R1: plant + calibrate TRAIN Sharpe (<= 4 fresh builds)")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--delta0", type=float, default=None)
    p = sub.add_parser("check-btc0", help="R3: the s0 scratch reproduces the real catalog")
    p.add_argument("--tf", choices=tfs, required=True)
    p = sub.add_parser("direct-chain", help="the planted rule through the real gate chain")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p = sub.add_parser("null-corr", help="the calibrated null pool of one timeframe")
    p.add_argument("--tf", choices=tfs, required=True)
    p.add_argument("--m", type=int, default=layer2.NULL_DEFAULT_M, help="draws in the null pool (>= K)")
    p.add_argument("--reps", type=int, default=layer2.NULL_DEFAULT_REPS, help="subsamples for C95")
    p.add_argument("--shards", type=int, default=layer2.NULL_DEFAULT_SHARDS, help="parallel worker count")
    p.add_argument("--seed", type=int, default=0, help="first chromosome seed")
    p.add_argument("--timeout", type=float, default=7200.0, help="per-shard worker timeout (s)")
    p = sub.add_parser("discover", help="one isolated discovery run")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--rung", choices=list(LADDER), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--probe", choices=[PROBE_FULL_POOL], default=None, help="Q1: full indicator pool")
    p.add_argument("--dsr-rule", choices=["emax", "q95"], default=None,
                   help="design-dsr-q95: opt-in DSR critical value (absent = legacy argv)")
    p.add_argument("--timeout", type=float, default=None)
    p = sub.add_parser("harvest", help="R5/R6: isolation, LONG, M1/M2 under C_K, cells")
    p.add_argument("--tf", choices=tfs, default=None)
    p = sub.add_parser("next-wave", help="R13: print the next wave's shell commands (runs nothing)")
    p.add_argument("--tf", choices=tfs, required=True)
    p.add_argument("--seeds", type=int, default=None, help="two-seed mode (e.g. --seeds 2)")
    sub.add_parser("report", help="R15: per-cell B*, M1/M2 counts, s0 champions, git shas")
    sub.add_parser("verify-untouched", help="R26: shared data identical to the snapshot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = layer2.Ctx(args)
    handlers: dict[str, Callable[[layer2.Ctx, argparse.Namespace], int]] = {
        "snapshot": cmd_snapshot,
        "pick-rule": cmd_pick_rule,
        "build": cmd_build,
        "check-btc0": cmd_check_btc0,
        "direct-chain": cmd_direct_chain,
        "null-corr": cmd_null_corr,
        "discover": cmd_discover,
        "harvest": cmd_harvest,
        "next-wave": cmd_next_wave,
        "report": cmd_report,
        "verify-untouched": cmd_verify_untouched,
    }
    try:
        return handlers[args.cmd](ctx, args)
    except IsolationViolation as exc:
        print(f"budget: {exc}")
        return 3
    except layer2.Layer2Error as exc:
        print(f"budget: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
