"""L2 end-to-end power layer: a known edge planted into PRIVATE copies of the BTCUSDT bars.

Job 20261017-power-paper, tasks T4a + T4b (design-power.md §3 "L2 end-to-end layer",
§5 V4/V5, §6 rows R25-R32, §8 risks).

``plant`` writes a scratch catalog per build; this module calibrates the drift ``delta`` of
the variants (``btc0`` control, ``btc2.0``, ``btc5.0``), checks the btc0 rebuild against the
real catalog bit for bit (R25), runs the positive-control direct chain through the real
gates (R28), runs isolated discovery runs on a scratch catalog and scratch state DB and
harvests them (R30-R32), and proves the shared data was never touched (R26).

Hard safety: the parent pins ``VIBE_QUANT_DB`` to a scratch DB BEFORE any ``vibe_quant``
import; every subprocess gets its own ``VIBE_QUANT_DB`` + ``--db`` and cwd = the scratch
root (catalog/archive paths are cwd-relative). Nothing here writes ``data/catalog``,
``data/archive`` or ``data/state/vibe_quant.db``.

Every subcommand records into ONE JSON (default
``data/swarm/20261017-power-paper/power/layer2.json``) under a file lock and is
idempotent per variant (``--force`` redoes a step). Exit codes: 0 ok, 2 prerequisite
missing / not calibrated, 3 machinery defect (R25, R26, R28, R31).

Not on the mypy baseline (``files = ["vibe_quant"]``); kept typed and ruff-clean.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_real_db() -> Path:
    """The main checkout's state DB (worktrees have no ``data/state``)."""
    here = Path(__file__).resolve().parents[3]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True, cwd=here,
        ).stdout.strip()
        return (Path(common).parent / "data" / "state" / "vibe_quant.db").resolve()
    except (OSError, subprocess.CalledProcessError):
        return (here / "data" / "state" / "vibe_quant.db").resolve()


REAL_DB = _find_real_db()


def _pin_scratch_db() -> str:
    """Point ``VIBE_QUANT_DB`` at a scratch DB unless it already names a non-real one (R26/R31).

    Runs BEFORE any ``vibe_quant`` import: ``DEFAULT_DB_PATH`` is read from the env at import.
    """
    current = os.environ.get("VIBE_QUANT_DB")
    if not current or Path(current).resolve() == REAL_DB:
        os.environ["VIBE_QUANT_DB"] = str(
            Path(tempfile.mkdtemp(prefix="layer2-parent-")) / "state.db"
        )
    return os.environ["VIBE_QUANT_DB"]


PARENT_DB = _pin_scratch_db()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import sqlite3  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

import gatechain  # noqa: E402
import layer1  # noqa: E402
import numpy as np  # noqa: E402
import plant  # noqa: E402

from vibe_quant.discovery.genome import (  # noqa: E402
    chromosome_to_dsl,
    chromosome_to_serializable,
    serializable_to_chromosome,
)
from vibe_quant.discovery.neff import N_RULE_EFFECTIVE  # noqa: E402
from vibe_quant.discovery.operators import (  # noqa: E402
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path("data/swarm/20261017-power-paper/power/layer2.json")

SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"
#: W3 plan window (gatechain._WINDOWS): full range, train = first 80% by plan_windows.
W3_FULL = ("2023-06-01", "2026-06-01")
#: Full range of each plan window (gatechain._WINDOWS; a unit test pins the match).
WINDOW_FULL: dict[str, tuple[str, str]] = {"W3": W3_FULL, "W6": ("2021-01-01", "2026-10-15")}
SQRT_365 = math.sqrt(365.0)

#: The L2 variants: calibration target and acceptance (R27). ``None`` target = no plant.
VARIANTS: dict[str, dict[str, Any]] = {
    "btc0": {"series": None, "target": None},
    "btc2.0": {"series": "full", "target": 2.0, "tol": 0.15, "delta0": 1.5e-5},
    "btc5.0": {"series": "train", "target": 5.0, "floor": 4.5, "delta0": 4e-5},
}
MAX_BUILDS = 4
KAPPA_TOL = 0.005  # R23: |L_end| of a converged kappa
PRICE_RATIO_LIMIT = 3.0  # design §8 risk 2
SEEDS = (1, 2)
RUN_TIMEOUT_S = 600.0  # R32: a run > 10 min is killed and marked timeout

#: Direct-chain N for the positive control (R28) and the reported marginals (T4a).
POSITIVE_CONTROL_NS = (1, 167, 22121)
REPORT_NS = (1, 7, 9, 11, 12, 167, 3901, 22121)
POSTHOC_NS = (1, 167, 3901, 22121)  # R29: candidates re-read at these N

#: R25 literals: strategy 239 screening (exactness_239.py) and the planted DSL on full W3.
EXPECTED_239 = {
    "sharpe": 0.7974716442710239,
    "trades": 67,
    "return": 0.159784359286041,
    "profit_factor": 1.6359612484989312,
    "max_drawdown": 0.05716872229767501,
}
EXPECTED_PLANTED_W3 = {
    "sharpe": -0.3729222589307,
    "trades": 142,
    "return": -0.1734565196672771,
}
WINDOW_239 = ("2024-01-01", "2026-03-17")

#: R30: a candidate "finds" the plant iff corr >= CORR_MIN and S >= RATIO_MIN * planted S.
CORR_MIN = 0.5
RATIO_MIN = 0.5

#: R40: the pre-registered default GA budget (12 x 4); the 50x20 follow-up runs pass pop/gens.
DEFAULT_POP = 12
DEFAULT_GENS = 4

#: State-JSON keys: `null` is T2's `null-corr` record (R47), the other two are written here.
NULL_RECORD_KEY = "null"
RESCREEN_RECORD_KEY = "rescreen"
FOLLOWUP_RUNS_KEY = "followup_runs"
LABEL_RE = re.compile(r"[A-Za-z0-9_-]+")

#: The three accepted builds a rescreen scores a candidate on (design-l2-followup.md §1(a)).
RESCREEN_VARIANTS = ("btc0", "btc2.0", "btc5.0")

#: R44: the discovery command's GA indicator pool (``discovery_command --indicator-pool``). The
#: null draws run AFTER the pipeline's own pool filter, so a gene can only use these indicators.
NULL_INDICATOR_POOL: tuple[str, ...] = ("ROC", "DONCHIAN", "RSI", "CCI", "NATR")
#: R47/P-1a: a 12x4 run calibrates at the pre-registered literal K; a 50x20 run uses its own
#: harvested evaluated count. The null pool must be at least as large as the largest K used.
NULL_K_12X4: int = 40
#: R45: the fixed subsample seed, so C95(K) is reproducible from the stored pool.
NULL_SUBSAMPLE_SEED: int = 20261018
NULL_DEFAULT_M: int = 1200  # >= the 50x20 run's evaluated count (chief ruling, P-1a)
NULL_DEFAULT_REPS: int = 2000
NULL_DEFAULT_SHARDS: int = 3
#: The null pool scratch file (per-chromosome seed + score + corr + chromosome) under ``scratch``.
NULL_POOL_FILE: str = "null-pool.json"


class Layer2Error(RuntimeError):
    """A prerequisite or machinery failure with a message for the operator."""


# --- planted rule ---------------------------------------------------------------------------


def planted_chromosome() -> StrategyChromosome:
    """Entry RSI(14) crosses_below 30, exit RSI(14) crosses_above 60, SL 4, TP 8, LONG."""
    return StrategyChromosome(
        entry_genes=[
            StrategyGene("RSI", {"period": 14.0}, ConditionType.CROSSES_BELOW, 30.0, None)
        ],
        exit_genes=[
            StrategyGene("RSI", {"period": 14.0}, ConditionType.CROSSES_ABOVE, 60.0, None)
        ],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.LONG,
        uid="planted_rsi",
    )


def planted_dsl(profile: Profile | None = None) -> dict[str, object]:
    """The planted rule as the DSL dict a discovery champion would carry (profile timeframe)."""
    prof = P4H if profile is None else profile
    dsl = chromosome_to_dsl(prof.chromosome)
    dsl["timeframe"] = prof.timeframe
    return dsl


def planted_both_dsl() -> dict[str, object]:
    """The planted rule forced to BOTH, as a warm-started champion becomes (operators.py:1232)."""
    dsl = chromosome_to_dsl(replace(planted_chromosome(), direction=Direction.BOTH))
    dsl["timeframe"] = TIMEFRAME
    return dsl


# --- Profile: the timeframe / window / plant of one study -------------------------------------


@dataclass(frozen=True)
class Profile:
    """Everything that differs between the 4h/W3 study and the 1d/W6 study.

    ``plant_span`` is the plant window (ISO dates, end day inclusive); ``kline_start`` the first
    archive day the klines are read from (warm-up before the plant); ``plant_kw`` is forwarded to
    ``plant.plant_drift`` / ``plant.solve_kappa``; ``direction`` is the discovery ``--direction``
    and the direction of null draws. ``chromosome`` is the planted rule (replaceable by the driver).
    """

    timeframe: str
    window: str
    plant_span: tuple[str, str]
    kline_start: str
    chromosome: StrategyChromosome
    plant_kw: dict[str, Any]
    direction: str = "both"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe, "window": self.window,
            "plant_span": list(self.plant_span), "kline_start": self.kline_start,
            "chromosome": chromosome_to_serializable(self.chromosome),
            "chromosome_uid": self.chromosome.uid,
            "plant_kw": dict(self.plant_kw), "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        start, end = data["plant_span"]
        return cls(
            timeframe=data["timeframe"], window=data["window"], plant_span=(start, end),
            kline_start=data["kline_start"],
            # the uid names the planted DSL (``genome_<uid>``): a round trip must keep it
            chromosome=replace(
                serializable_to_chromosome(data["chromosome"]), uid=data["chromosome_uid"]
            ),
            plant_kw=dict(data["plant_kw"]), direction=data["direction"],
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Profile:
        """The worker payload's profile; P4H when absent (old payloads)."""
        wire = payload.get("profile")
        return P4H if wire is None else cls.from_dict(wire)


P4H = Profile(
    timeframe=TIMEFRAME, window="W3", plant_span=(plant.PLANT_START, plant.PLANT_END),
    kline_start=plant.PLANT_START, chromosome=planted_chromosome(),
    plant_kw={
        "bar_min": plant.BAR_MIN, "drift_bars": plant.DRIFT_BARS,
        "rsi_period": plant.RSI_PERIOD, "rsi_level": plant.RSI_LEVEL,
    },
    direction="both",
)
#: The 1d study: W6, LONG. ``chromosome`` is the 4h rule until the driver picks the 1d rule.
P1D = Profile(
    timeframe="1d", window="W6", plant_span=("2021-01-01", "2026-10-15"),
    kline_start="2020-12-01", chromosome=planted_chromosome(),
    plant_kw={
        "bar_min": 1440, "drift_bars": 6, "rsi_period": plant.RSI_PERIOD,
        "rsi_level": plant.RSI_LEVEL,
    },
    direction="long",
)


def full_window(profile: Profile = P4H) -> tuple[str, str]:
    return WINDOW_FULL[profile.window]


# --- JSON state (one file, locked read-modify-write) ----------------------------------------


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def update_state(path: Path, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Locked read-modify-write; concurrent ``discover`` processes share one JSON."""
    with _locked(path):
        state = load_state(path)
        mutate(state)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=_json_default) + "\n")
        os.replace(tmp, path)
    return state


def _json_default(obj: object) -> object:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _variant(state: dict[str, Any], name: str) -> dict[str, Any]:
    out: dict[str, Any] = state.setdefault("variants", {}).setdefault(name, {})
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- R26: signatures of the shared data -----------------------------------------------------


def file_signature(path: Path, sha_deadline_s: float = 60.0) -> dict[str, Any]:
    """``size``, ``mtime_ns`` and ``sha256`` of a file (sha ``None`` past the deadline)."""
    st = path.stat()
    digest = hashlib.sha256()
    deadline = time.monotonic() + sha_deadline_s
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            digest.update(chunk)
            if time.monotonic() > deadline:  # a prefix digest would be wrong: drop it
                return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": None}
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest.hexdigest()}


def tracked_files(data_root: Path, symbol: str = SYMBOL) -> list[Path]:
    """Every file R26 pins: the symbol's bar + instrument catalog files and the raw archive."""
    cat = data_root / "catalog" / "data"
    files: list[Path] = []
    for bar_dir in sorted((cat / "bar").glob(f"{symbol}-PERP.BINANCE-*")):
        files.extend(p for p in sorted(bar_dir.rglob("*")) if p.is_file())
    inst = cat / "crypto_perpetual" / f"{symbol}-PERP.BINANCE"
    if inst.exists():
        files.extend(p for p in sorted(inst.rglob("*")) if p.is_file())
    archive = data_root / "archive" / "raw_data.db"
    if archive.exists():
        files.append(archive)
    return files


def discovery_rows(real_db: Path) -> dict[str, Any]:
    """``count`` and ``max_id`` of ``run_mode='discovery'`` rows (DB opened read-only)."""
    if not real_db.exists():
        return {"count": None, "max_id": None}
    conn = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
    try:
        count, max_id = conn.execute(
            "SELECT count(*), max(id) FROM backtest_runs WHERE run_mode = 'discovery'"
        ).fetchone()
    finally:
        conn.close()
    return {"count": count, "max_id": max_id}


def funding_count(archive: Path, symbol: str = SYMBOL) -> int:
    conn = sqlite3.connect(f"file:{archive}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT count(*) FROM raw_funding_rates WHERE symbol = ?", (symbol,)
        ).fetchone()
    finally:
        conn.close()
    return int(row[0])


def take_snapshot(data_root: Path, real_db: Path, archive: Path | None = None) -> dict[str, Any]:
    arch = archive if archive is not None else data_root / "archive" / "raw_data.db"
    return {
        "taken_at": _now(),
        "files": {
            str(p.relative_to(data_root)): file_signature(p) for p in tracked_files(data_root)
        },
        "discovery_rows": discovery_rows(real_db),
        "funding_source_count": funding_count(arch),
    }


def _sig_diff(want: dict[str, Any], got: dict[str, Any]) -> bool:
    for key in ("size", "mtime_ns"):
        if want[key] != got[key]:
            return True
    return want["sha256"] is not None and got["sha256"] is not None and want["sha256"] != got["sha256"]


def compare_snapshot(
    snap: dict[str, Any], data_root: Path, real_db: Path, archive: Path | None = None
) -> list[str]:
    """Mismatch descriptions between a snapshot and the data as it is now (empty = untouched)."""
    problems: list[str] = []
    now_files = {str(p.relative_to(data_root)): p for p in tracked_files(data_root)}
    for rel, want in snap["files"].items():
        path = now_files.get(rel)
        if path is None:
            problems.append(f"file missing: {rel}")
        elif _sig_diff(want, file_signature(path)):
            problems.append(f"file changed: {rel}")
    problems.extend(f"new file: {rel}" for rel in now_files if rel not in snap["files"])

    rows = discovery_rows(real_db)
    want_rows = snap["discovery_rows"]
    if rows != want_rows:
        problems.append(f"discovery rows {want_rows} -> {rows}")
        if real_db.exists():
            conn = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
            try:
                new = conn.execute(
                    "SELECT id, created_at FROM backtest_runs "
                    "WHERE run_mode = 'discovery' AND id > ? ORDER BY id",
                    (want_rows["max_id"] or 0,),
                ).fetchall()
            finally:
                conn.close()
            for row_id, created_at in new:
                problems.append(f"new discovery row: id={row_id} created_at={created_at}")
    arch = archive if archive is not None else data_root / "archive" / "raw_data.db"
    count_now = funding_count(arch)
    if count_now != snap["funding_source_count"]:
        problems.append(f"source funding rows {snap['funding_source_count']} -> {count_now}")
    return problems


# --- subprocess plumbing --------------------------------------------------------------------


def assert_not_real_db(db: Path) -> None:
    if db.resolve() == REAL_DB:
        msg = f"refusing to use the real state DB {REAL_DB}"
        raise Layer2Error(msg)


def child_env(db: Path, log_dir: Path | None = None) -> dict[str, str]:
    """Env of every subprocess: its OWN scratch ``VIBE_QUANT_DB`` (R26/R31), never the real one."""
    assert_not_real_db(db)
    env = dict(os.environ)
    env["VIBE_QUANT_DB"] = str(db.resolve())
    if log_dir is not None:
        env["VIBE_QUANT_LOG_DIR"] = str(log_dir.resolve())
    return env


def assert_scratch_root(root: Path, data_root: Path) -> None:
    """The scratch root must hold its OWN catalog (never a link into the shared one)."""
    cat = (root / "data" / "catalog").resolve()
    if not cat.exists():
        msg = f"scratch root {root} has no catalog"
        raise Layer2Error(msg)
    real = (data_root / "catalog").resolve()
    if cat == real or real in cat.parents or (root / "data" / "catalog").is_symlink():
        msg = f"scratch catalog {cat} is the shared catalog"
        raise Layer2Error(msg)


def run_worker(
    kind: str, root: Path, payload: dict[str, Any], db: Path, timeout_s: float = 3600.0
) -> Any:
    """Run ``layer2.py _worker`` with cwd = scratch ``root`` and return its JSON result."""
    root, db = root.resolve(), db.resolve()  # the child runs with cwd = root: no relative paths
    with tempfile.TemporaryDirectory(prefix="layer2-w-") as tmp:
        inp, out = Path(tmp) / "in.json", Path(tmp) / "out.json"
        inp.write_text(json.dumps({**payload, "root": str(root)}, default=_json_default))
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_worker", kind,
             "--in", str(inp), "--out", str(out)],
            cwd=root, env=child_env(db, root / "logs"),
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stderr or proc.stdout)[-1500:]
            msg = f"worker {kind} failed rc={proc.returncode}: {tail}"
            raise Layer2Error(msg)
        return json.loads(out.read_text())


# --- metrics --------------------------------------------------------------------------------


def ann_sharpe(mtm: dict[str, Any] | None) -> float | None:
    """S = daily mean/std of the dense MTM series (day 0 dropped) x sqrt(365)."""
    if not mtm:
        return None
    from vibe_quant.discovery.buyhold import daily_sharpe

    sharpe = daily_sharpe(list(mtm["values"])[1:])
    return None if sharpe is None else sharpe * SQRT_365


def signed_corr(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    """Pearson corr of two ``{first_day, values}`` series on their common days; None if undefined."""
    if not a or not b:
        return None
    fa, fb = int(a["first_day"]), int(b["first_day"])
    lo = max(fa, fb)
    hi = min(fa + len(a["values"]), fb + len(b["values"]))
    if hi - lo < 2:
        return None
    xa = np.asarray(a["values"][lo - fa : hi - fa], dtype=np.float64)
    xb = np.asarray(b["values"][lo - fb : hi - fb], dtype=np.float64)
    if float(np.std(xa)) == 0.0 or float(np.std(xb)) == 0.0:
        return None
    return float(np.corrcoef(xa, xb)[0, 1])


def found_criterion(
    corr: float | None,
    cand_s: float | None,
    planted_s: float | None,
    *,
    corr_min: float = CORR_MIN,
) -> bool:
    """R30/R46 for ONE candidate: signed corr >= ``corr_min`` AND train S >= 0.5 x planted S.

    ``corr_min`` defaults to ``CORR_MIN``; the calibrated null threshold C_K (T2) passes its own.
    """
    if corr is None or cand_s is None or planted_s is None:
        return False
    return corr >= corr_min and cand_s >= RATIO_MIN * planted_s


def assess_found(candidates: Sequence[dict[str, Any]], planted_s: float | None) -> dict[str, Any]:
    """R30 over a run: ``found`` = any candidate passes; ``max_abs_corr`` is INFO only."""
    flagged: list[dict[str, Any]] = []
    max_abs: float | None = None
    for cand in candidates:
        corr, cand_s = cand.get("corr"), cand.get("train_S")
        if corr is not None:
            max_abs = abs(corr) if max_abs is None else max(max_abs, abs(corr))
        if found_criterion(corr, cand_s, planted_s):
            flagged.append({"uid": cand.get("uid"), "corr": corr, "train_S": cand_s})
    return {
        "found": bool(flagged),
        "found_by": flagged,
        "max_abs_corr": max_abs,
        "planted_train_S": planted_s,
    }


# --- calibration (R27) ----------------------------------------------------------------------


def calib_value(variant: str, rec: dict[str, Any]) -> float | None:
    series = VARIANTS[variant]["series"]
    value = rec.get("full_S") if series == "full" else rec.get("train_S")
    return None if value is None else float(value)


def accepts(variant: str, rec: dict[str, Any]) -> bool:
    """R27 + R23: btc2.0 |full-W3 S - 2.0| <= 0.15; btc5.0 TRAIN S >= 4.5; btc0 always.

    A planted variant is also accepted only when its kappa build converged
    (``rec["converged"]``: ``|l_end| <= KAPPA_TOL``); a non-converged plant fails closed,
    whatever S says, because kappa never compensated the delta it was built for.
    """
    cfg = VARIANTS[variant]
    if cfg["series"] is None:
        return True
    if rec.get("converged") is not True:
        return False
    value = calib_value(variant, rec)
    if value is None:
        return False
    if variant == "btc5.0":
        return bool(value >= cfg["floor"])
    return bool(abs(value - cfg["target"]) <= cfg["tol"])


def next_delta(variant: str, builds: Sequence[dict[str, Any]], delta0: float) -> float:
    """Secant on delta toward the target; proportional with one point; clamped to 0.25x..4x."""
    cfg = VARIANTS[variant]
    target = float(cfg["target"])
    pts = [
        (float(b["delta"]), v)
        for b in builds
        if (v := calib_value(variant, b)) is not None and b["delta"] > 0
    ]
    if not pts:
        return delta0 if not builds else float(builds[-1]["delta"]) * 2.0
    d2, s2 = pts[-1]
    guess = d2 * target / s2 if s2 > 0 else d2 * 2.0
    if len(pts) >= 2:
        d1, s1 = pts[-2]
        if s2 != s1:
            sec = d2 - (s2 - target) * (d2 - d1) / (s2 - s1)
            if math.isfinite(sec) and sec > 0:
                guess = sec
    return float(min(max(guess, d2 / 4.0), d2 * 4.0))


# --- builds ---------------------------------------------------------------------------------


def plant_window_ms(profile: Profile = P4H) -> tuple[int, int]:
    """[start, end) of the plant window (P4H: 2022-06-01 .. end of 2026-06-30)."""
    start, end = profile.plant_span
    return plant._ms(start), plant._ms(end) + 86_400_000


def _under(path: Path, parent: Path) -> bool:
    return parent.resolve() in path.resolve().parents


def build_root(scratch: Path, variant: str, k: int) -> Path:
    return scratch / f"{variant}-b{k}"


def measure_jobs(profile: Profile = P4H) -> list[dict[str, Any]]:
    """Screening jobs that calibrate one build: planted rule on the full window and on TRAIN."""
    train_start, train_end = train_window(profile)
    full_start, full_end = full_window(profile)
    return [
        {"tag": "full", "dsl": planted_dsl(profile), "start": full_start, "end": full_end,
         "collect": True, "bh": False},
        {"tag": "train", "dsl": planted_dsl(profile), "start": train_start, "end": train_end,
         "collect": True, "bh": True},
    ]


def train_window(profile: Profile = P4H) -> tuple[str, str]:
    plan = gatechain.settings_for(profile.window).plan
    return plan.train_start, plan.train_end


def measure_build(root: Path, db: Path, profile: Profile = P4H) -> dict[str, Any]:
    """NT sharpe/trades + MTM S on the full window and on train, and the planted-catalog B&H bar."""
    res = {r["tag"]: r for r in run_worker("screen", root, {"jobs": measure_jobs(profile)}, db)}
    full, train = res["full"], res["train"]
    return {
        "nt_sharpe_full": full["sharpe"],
        "trades_full": full["trades"],
        "full_S": ann_sharpe(full["mtm"]),
        "nt_sharpe_train": train["sharpe"],
        "trades_train": train["trades"],
        "train_S": ann_sharpe(train["mtm"]),
        "bh_train_daily_sharpe": train["bh_daily_sharpe"],
    }


def build_one(
    variant: str,
    k: int,
    delta: float,
    klines: plant.Klines,
    archive: Path,
    scratch: Path,
    db: Path,
    profile: Profile = P4H,
) -> dict[str, Any]:
    """One build in a FRESH scratch root: plant (kappa-compensated), write, measure."""
    root = build_root(scratch, variant, k)
    if root.exists():
        if not _under(root, scratch):
            msg = f"refusing to remove {root}"
            raise Layer2Error(msg)
        shutil.rmtree(root)  # a crashed earlier attempt; builds are never reused
    start_ms, end_ms = plant_window_ms(profile)
    from vibe_quant.data.catalog import create_instrument

    decimals = int(create_instrument(SYMBOL).price_precision)
    kappa, passes, l_end = 0.0, 0, 0.0
    if delta != 0.0:
        # Refine kappa with the global search only when the fixed point misses KAPPA_TOL -- the
        # very threshold that marks a build non-converged below. A build that already met it
        # (btc2.0) keeps its phase-1 kappa, so old builds stay byte-identical.
        kappa, passes, l_end = plant.solve_kappa(
            klines, delta, plant_start_ms=start_ms, plant_end_ms=end_ms, price_decimals=decimals,
            refine_above=KAPPA_TOL, **profile.plant_kw,
        )
    result = plant.plant_drift(
        klines, delta=delta, kappa=kappa, plant_start_ms=start_ms, plant_end_ms=end_ms,
        price_decimals=decimals, **profile.plant_kw,
    )
    plant.write_scratch(root, result.klines, archive)
    rep = result.report
    ratio = math.exp(rep.l_max_abs)
    rec: dict[str, Any] = {
        "k": k, "root": str(root), "delta": delta, "kappa": kappa, "kappa_passes": passes,
        "l_end": rep.l_end, "converged": abs(rep.l_end) <= KAPPA_TOL,
        "n_triggers": rep.n_triggers, "drift_minutes": rep.drift_minutes,
        "l_max_abs": rep.l_max_abs, "price_ratio": ratio,
        # plant.py has no hold_bars knob: a ratio over 3x is reported, not worked around.
        "price_ratio_gt_3": ratio > PRICE_RATIO_LIMIT,
        "built_at": _now(),
    }
    rec.update(measure_build(root, db, profile))
    return rec


# --- R28 positive control -------------------------------------------------------------------


def positive_control_failures(
    chain: dict[str, Any], ns: Sequence[int] = POSITIVE_CONTROL_NS
) -> list[str]:
    """Every gate that did not pass in a direct-chain result (JSON, string N keys); fail closed."""
    fails: list[str] = []
    if chain.get("floor") is not True:
        fails.append("floor")
    marginals = chain.get("marginals") or {}
    if not marginals:
        fails.append("marginals missing")
    fails.extend(name for name, ok in marginals.items() if ok is not True)
    for n in ns:
        if (chain.get("dsr") or {}).get(str(n)) is not True:
            fails.append(f"dsr[{n}]")
        if (chain.get("chain_full") or {}).get(str(n)) is not True:
            fails.append(f"chain_full[{n}]")
    return fails


# --- R25 ------------------------------------------------------------------------------------


def compare_exact(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """``==`` per key (money numbers are bit-compared, never approximated)."""
    return [f"{k}: got {got[k]!r} want {v!r}" for k, v in want.items() if got[k] != v]


def read_strategy_239(real_db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT dsl_config FROM strategies WHERE id = 239").fetchone()
    finally:
        conn.close()
    if row is None:
        msg = "strategy 239 not found in the state DB"
        raise Layer2Error(msg)
    cfg: dict[str, Any] = json.loads(row[0])
    return cfg


# --- discovery runs (R31, R32) --------------------------------------------------------------


def discovery_command(
    run_id: int,
    db: Path,
    seed: int,
    *,
    pop: int = DEFAULT_POP,
    gens: int = DEFAULT_GENS,
    seed_run: int | None = None,
    profile: Profile = P4H,
) -> list[str]:
    """R40: the discovery CLI argv. The defaults reproduce the 12x4 command byte for byte;
    ``seed_run`` adds one ``--seed-from-run`` (R42 warm start)."""
    start, end = full_window(profile)
    cmd = [
        sys.executable, "-m", "vibe_quant", "discovery",
        "--run-id", str(run_id), "--db", str(db),
        "--symbols", SYMBOL,
        "--timeframe", profile.timeframe,
        "--start-date", start,
        "--end-date", end,
        "--indicator-pool", "ROC,DONCHIAN,RSI,CCI,NATR",
        "--population-size", str(pop),
        "--max-generations", str(gens),
        "--eval-windows", "2",
        "--train-test-split", "0.8",
        "--direction", profile.direction,
        "--max-workers", "3",
        "--seed", str(seed),
    ]
    if seed_run is not None:
        cmd += ["--seed-from-run", str(seed_run)]
    return cmd


def discover_command_fn(
    pop: int, gens: int, seed_run: int | None, profile: Profile = P4H
) -> Callable[[int, Path, int], list[str]]:
    """Bind ``--pop/--gens/--seed-from-run`` into the ``command_fn`` ``run_discovery`` calls.

    A value at its default is not passed, so the default 12x4 command stays a plain 3-arg call.
    """
    def command(run_id: int, db: Path, seed: int) -> list[str]:
        kwargs: dict[str, Any] = {}
        if pop != DEFAULT_POP:
            kwargs["pop"] = pop
        if gens != DEFAULT_GENS:
            kwargs["gens"] = gens
        if seed_run is not None:
            kwargs["seed_run"] = seed_run
        if profile != P4H:
            kwargs["profile"] = profile
        return discovery_command(run_id, db, seed, **kwargs)

    return command


def classify_run(
    row: tuple[str, str | None] | None, returncode: int | None, timed_out: bool, timeout_s: float
) -> dict[str, Any]:
    """R32: failed / killed / timed-out runs are ``failed`` / ``timeout`` records, never "0 champions"."""
    if timed_out:
        return {"status": "timeout", "error": f"killed after {timeout_s:g}s"}
    if row is None:
        return {"status": "failed", "error": f"run row missing (exit code {returncode})"}
    status, error = row
    if status == "completed" and returncode == 0:
        return {"status": "completed"}
    return {
        "status": "failed",
        "error": error or f"exit code {returncode}, run status {status!r}",
    }


def create_run(db: Path, profile: Profile = P4H) -> int:
    """A discovery ``backtest_runs`` row in the run's OWN scratch DB."""
    assert_not_real_db(db)
    start, end = full_window(profile)
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db)
    try:
        return int(
            state.create_backtest_run(
                strategy_id=None, run_mode="discovery", symbols=[SYMBOL],
                timeframe=profile.timeframe, start_date=start, end_date=end, parameters={},
            )
        )
    finally:
        state.close()


def create_seed_run(db: Path, profile: Profile = P4H) -> int:
    """R42: the warm-start seed row in the run's OWN DB, written BEFORE the discovery run.

    A completed discovery row whose only champion is the planted chromosome (as the
    serialized genome ``--seed-from-run`` reloads). It carries NO ``evaluated`` count, so
    ``_prior_discovery_trials`` skips it and the real run still sees ``prior_n == 0`` (R31).
    The planted gene becomes BOTH at load under ``--direction both`` (operators.py:1232).
    """
    assert_not_real_db(db)
    from vibe_quant.db.state_manager import StateManager

    start, end = full_window(profile)
    notes = {
        "top_strategies": [
            {"chromosome": chromosome_to_serializable(profile.chromosome),
             "dsl": planted_dsl(profile)}
        ]
    }
    state = StateManager(db)
    try:
        run_id = int(
            state.create_backtest_run(
                strategy_id=None, run_mode="discovery", symbols=[SYMBOL],
                timeframe=profile.timeframe, start_date=start, end_date=end, parameters={},
            )
        )
        state.save_backtest_result(
            run_id,
            {
                "total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "profit_factor": 0.0, "total_trades": 0, "skewness": 0.0, "kurtosis": 3.0,
                "execution_time_seconds": 0.0, "notes": json.dumps(notes),
            },
        )
        state.update_backtest_run_status(run_id, "completed")
        return run_id
    finally:
        state.close()


def run_discovery(
    root: Path,
    run_dir: Path,
    seed: int,
    *,
    timeout_s: float = RUN_TIMEOUT_S,
    command_fn: Callable[[int, Path, int], list[str]] = discovery_command,
    poll_s: float = 1.0,
    profile: Profile = P4H,
) -> dict[str, Any]:
    """Launch one isolated discovery run (cwd = scratch root, own DB) and poll it to a final status.

    ``profile`` sets the timeframe and dates persisted on the run row; ``command_fn`` must be bound
    to the same profile (``discover_command_fn(..., profile)``)."""
    root, run_dir = root.resolve(), run_dir.resolve()  # the child runs with cwd = root
    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / "state.db"
    log = run_dir / "run.log"
    run_id = create_run(db, profile)
    started = time.monotonic()
    timed_out = False
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            command_fn(run_id, db, seed), cwd=root, env=child_env(db, run_dir / "logs"),
            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
        while proc.poll() is None:
            if time.monotonic() - started > timeout_s:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
            time.sleep(poll_s)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    rec = classify_run(row, proc.returncode, timed_out, timeout_s)
    rec.update(
        run_id=run_id, seed=seed, db=str(db), log=str(log), root=str(root),
        wall_s=round(time.monotonic() - started, 1), finished_at=_now(),
    )
    return rec


def harvest_run(db: Path, run_id: int) -> dict[str, Any]:
    """Champions, per-stage rejections, DSR trial counts (R31) and candidate DSLs of a completed run."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        msg = f"run {run_id} has no result notes"
        raise Layer2Error(msg)
    notes: dict[str, Any] = json.loads(row[0])
    tops: list[dict[str, Any]] = notes.get("top_strategies") or []
    rejects: list[dict[str, Any]] = notes.get("guardrail_rejections") or []
    trials: dict[str, Any] = notes.get("dsr_trials") or {}
    candidates = [{**c, "source": "champion", "uid": c.get("dsl", {}).get("name")} for c in tops]
    candidates += [{**c, "source": "rejected"} for c in rejects]
    return {
        "champions": len(tops),
        "outcome": notes.get("outcome"),
        "evaluated": notes.get("evaluated"),
        "rejections_by_stage": dict(Counter(str(r.get("stage", "guardrails")) for r in rejects)),
        "dsr_trials": trials,
        "r31": check_isolation(trials),
        "candidates": candidates,
    }


def check_isolation(trials: dict[str, Any]) -> dict[str, Any]:
    """R31/R5: a run on its own DB has ``prior_n == 0``, ``gate_n == run_n_eff`` and recorded the
    effective-N rule (``n_rule == N_RULE_EFFECTIVE``; megq1: gate_n = run_n_eff + prior_n, with
    run_n_eff <= raw). A run with no marker, or another one, fails closed."""
    if not trials:
        return {"ok": True, "checked": False, "note": "no dsr_trials in notes (nothing reached DSR)"}
    prior_n, gate_n, raw = trials.get("prior_n"), trials.get("gate_n"), trials.get("raw")
    run_n_eff, n_rule = trials.get("run_n_eff"), trials.get("n_rule")
    ok = (
        prior_n == 0 and gate_n is not None and gate_n == run_n_eff and n_rule == N_RULE_EFFECTIVE
    )
    return {"ok": ok, "checked": True, "prior_n": prior_n, "gate_n": gate_n, "raw": raw,
            "run_n_eff": run_n_eff, "n_rule": n_rule}


def posthoc_p(candidates: Sequence[dict[str, Any]], ns: Sequence[int]) -> dict[str, list[float | None]]:
    """R29 per candidate: the DSR p-value of its stored record re-read at each N."""
    out: dict[str, list[float | None]] = {}
    for cand in candidates:
        rec = cand.get("dsr")
        key = str(cand.get("uid"))
        row: list[float | None] = []
        for n in ns:
            try:
                row.append(float(gatechain.posthoc_dsr_p(rec, n))) if rec else row.append(None)
            except (KeyError, TypeError, ValueError):
                row.append(None)
        out[key] = row
    return out


# --- verdict lines --------------------------------------------------------------------------


def verdict_lines(state: dict[str, Any]) -> list[str]:
    """V4 (positive control) and V5 (found) lines from the recorded JSON."""
    lines: list[str] = []
    variants = state.get("variants", {})
    for name in ("btc5.0", "btc2.0"):
        chain = variants.get(name, {}).get("direct_chain")
        if chain is None:
            lines.append(f"V4 {name} direct chain: not run")
            continue
        fails = chain.get("failures")
        if fails is None:
            fails = positive_control_failures(chain["result"])
        verdict = "PASS" if not fails else f"FAIL ({', '.join(fails)})"
        label = "positive control" if name == "btc5.0" else "report only"
        lines.append(f"V4 {name} direct chain [{label}]: {verdict}")
    for name in ("btc2.0", "btc5.0", "btc0"):
        runs = variants.get(name, {}).get("runs", {})
        parts = []
        for seed, run in sorted(runs.items()):
            if run.get("status") != "completed":
                parts.append(f"seed {seed}: {run.get('status')} ({run.get('error')})")
                continue
            h = run.get("harvest")
            if h is None:
                parts.append(f"seed {seed}: not harvested")
                continue
            parts.append(
                f"seed {seed}: champions={h['champions']} found={h['found']['found']} "
                f"max|corr|={h['found']['max_abs_corr']}"
            )
        if name == "btc0":
            lines.append("V5 btc0 (null, info): " + ("; ".join(parts) or "no runs"))
            continue
        any_found = any(
            r.get("harvest", {}).get("found", {}).get("found") for r in runs.values()
        )
        lines.append(
            f"V5 {name}: {'FOUND' if any_found else 'not found'} | " + ("; ".join(parts) or "no runs")
        )
    return lines


# --- R44/R45/R47: the calibrated null threshold (design-l2-followup.md §1(b)) ---------------


def _null_backtest(_chrom: StrategyChromosome) -> dict[str, float | int]:
    """The pipeline instance below only applies the pool filter; its backtest fn is never called."""
    return {}


def apply_ga_pool_filter(pool: Sequence[str] = NULL_INDICATOR_POOL) -> None:
    """Restrict ``operators.INDICATOR_POOL`` exactly the way the discovery pipeline does (R44).

    Discovery runs as a subprocess, where filtering the module-global pool in place is safe; the
    null draws run here for the same reason, so a gene can only use the ``--indicator-pool`` set.
    """
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    pipeline = DiscoveryPipeline(DiscoveryConfig(indicator_pool=list(pool)), _null_backtest)
    pipeline._apply_indicator_pool_filter()


def null_chromosome(
    seed: int, pool: Sequence[str] = NULL_INDICATOR_POOL, direction: Direction = Direction.BOTH
) -> StrategyChromosome:
    """R44: one null draw the GA's way -- pool filter, ``random.seed``, ``_random_chromosome``."""
    from vibe_quant.discovery.operators import _random_chromosome

    apply_ga_pool_filter(pool)
    random.seed(seed)
    return _random_chromosome(direction)


def null_threshold(
    pool: Sequence[tuple[StrategyChromosome, float, float | None]],
    k: int,
    reps: int,
    *,
    seed: int = NULL_SUBSAMPLE_SEED,
    top_k: int = layer1.SELECTION_TOP_K,
    min_distance: float = layer1.SELECTION_MIN_DISTANCE,
) -> dict[str, Any]:
    """R45: the null 95th percentile C95(K) over ``reps`` K-subsamples of the pool.

    Each rep draws K pool entries (fixed ``seed``), takes the production top-K pick of their float
    scores (``layer1.select_picks``: a non-positive score is never eligible) and records
    ``T = max|corr|`` of the picks (0.0 when nothing is picked or every corr is None). C95(K) is the
    95th percentile of T over the reps.
    """
    if k > len(pool):
        msg = f"null pool of {len(pool)} cannot supply a subsample of K={k}"
        raise Layer2Error(msg)
    rng = random.Random(seed)
    ts: list[float] = []
    for _ in range(reps):
        sample = rng.sample(list(pool), k)
        scored = sorted(
            ((chrom, score) for chrom, score, _corr in sample),
            key=lambda entry: entry[1],
            reverse=True,
        )
        picked = {
            id(chrom)
            for chrom, _fit in layer1.select_picks(scored, top_k=top_k, min_distance=min_distance)
        }
        vals = [corr for chrom, _score, corr in sample if id(chrom) in picked and corr is not None]
        ts.append(max((abs(v) for v in vals), default=0.0))
    return {"k": k, "reps": reps, "seed": seed, "c95": float(np.percentile(np.asarray(ts), 95.0))}


def corr_threshold(c95: float, btc0_max_corr: float | None) -> float:
    """R47/P-1a: C_K = max(C95(K), the same-budget btc0 GA max SIGNED corr)."""
    if btc0_max_corr is None:
        return float(c95)
    return float(max(float(c95), float(btc0_max_corr)))


def load_null_pool(path: Path) -> list[tuple[StrategyChromosome, float, float | None]]:
    """The recorded null pool as ``(chromosome, adjusted_score, corr)`` triples (R47 harvest)."""
    from vibe_quant.discovery.genome import serializable_to_chromosome

    raw: list[dict[str, Any]] = json.loads(Path(path).read_text())
    return [
        (serializable_to_chromosome(entry["chromosome"]), float(entry["score"]), entry.get("corr"))
        for entry in raw
    ]


# --- workers (run inside a subprocess with cwd = scratch root) ------------------------------


def _assert_worker_cwd(root: Path) -> None:
    if Path.cwd().resolve() != root.resolve():
        msg = f"worker cwd {Path.cwd()} is not the scratch root {root}"
        raise Layer2Error(msg)


def _series(dr: Any) -> dict[str, Any] | None:
    if dr is None:
        return None
    return {"first_day": int(dr.first_day), "values": [float(v) for v in dr.values]}


def worker_screen(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Screening jobs on the scratch catalog: metrics, optional MTM series and B&H daily Sharpe."""
    from vibe_quant.discovery.buyhold import buy_hold_daily, daily_sharpe, load_daily_closes
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict

    root = Path(payload["root"]).resolve()
    _assert_worker_cwd(root)
    catalog = root / "data" / "catalog"
    archive = root / "data" / "archive" / "raw_data.db"
    out: list[dict[str, Any]] = []
    for job in payload["jobs"]:
        dsl = (
            _dsl_to_dict(validate_strategy_dict(job["dsl_config"]))
            if "dsl_config" in job else job["dsl"]
        )
        runner = NTScreeningRunner(
            dsl, [SYMBOL], job["start"], job["end"],
            catalog_path=str(catalog), funding_archive_path=str(archive),
            collect_daily_returns=bool(job.get("collect")),
        )
        m = runner({})
        entry: dict[str, Any] = {
            "tag": job["tag"], "sharpe": m.sharpe_ratio, "trades": m.total_trades,
            "return": m.total_return, "profit_factor": m.profit_factor,
            "max_drawdown": m.max_drawdown, "mtm": None, "bh_daily_sharpe": None,
        }
        if job.get("collect"):
            entry["mtm"] = _series(runner.last_mtm_daily_returns)
        if job.get("bh") and entry["mtm"] is not None:
            vals = entry["mtm"]["values"]
            closes = load_daily_closes([SYMBOL], catalog)
            bh, _err = buy_hold_daily(closes, [SYMBOL], entry["mtm"]["first_day"] + 1, len(vals) - 1)
            entry["bh_daily_sharpe"] = None if bh is None else daily_sharpe(bh)
        out.append(entry)
    return out


def _cell_fns(profile: Profile = P4H) -> tuple[Any, gatechain.ChainSettings]:
    """The cell functions (train + holdout + cross-window factory) of the real chain, for the
    profile's plan window and timeframe.

    ONE place builds them so ``worker_chain`` (the planted rule) and ``worker_null`` (the random
    draws) score through the exact same train window and holdout (design-l2-followup.md §1(b)).
    """
    from vibe_quant.discovery.__main__ import _make_cell_fns
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    settings = gatechain.settings_for(profile.window)
    plan = settings.plan
    fns = _make_cell_fns(
        use_mock=False, symbols=[SYMBOL], timeframe=profile.timeframe, symbol_agg="portfolio",
        fee_schedule=FeeSchedule(), entry_order="market", liquidation=LiquidationSettings(),
        sizing=SizingSettings(), train_start=plan.train_start, train_end=plan.train_end,
        eval_windows=plan.eval_windows, min_trades=settings.config.min_trades,
        holdout=(str(plan.holdout_start), str(plan.holdout_end)), needs_factory=True,
    )
    fns.backtest_fn.collect_daily_returns = True  # type: ignore[attr-defined]
    return fns, settings


def _w3_cell_fns() -> tuple[Any, gatechain.ChainSettings]:
    """The W3/4h cell functions: ``_cell_fns`` of the default profile."""
    return _cell_fns(P4H)


def worker_chain(payload: dict[str, Any]) -> dict[str, Any]:
    """The planted rule through the REAL gate chain on the scratch catalog (fresh process, risk 3)."""
    from vibe_quant.discovery.buyhold import daily_sharpe, load_daily_closes
    from vibe_quant.discovery.fitness import MIN_TRADES, _evaluate_single

    root = Path(payload["root"]).resolve()
    _assert_worker_cwd(root)
    profile = Profile.from_payload(payload)
    fns, settings = _w3_cell_fns() if profile == P4H else _cell_fns(profile)
    chrom = profile.chromosome
    fit = _evaluate_single(chrom, fns.backtest_fn, None, MIN_TRADES, profile.timeframe)  # type: ignore[arg-type]
    holdout_bt = fns.holdout_backtest_fn(chrom)  # type: ignore[operator]
    closes = load_daily_closes([SYMBOL], root / "data" / "catalog")
    result = gatechain.evaluate_candidate(
        chrom, fit, holdout_bt, fns.backtest_fn_factory, closes, settings  # type: ignore[arg-type]
    )
    series = fit.mtm_daily_returns
    sharpe = daily_sharpe(list(series.values)) if series is not None else None
    return {
        "fitness": {
            "sharpe_ratio": fit.sharpe_ratio, "total_trades": fit.total_trades,
            "total_return": fit.total_return, "adjusted_score": fit.adjusted_score,
            "train_S": None if sharpe is None else sharpe * SQRT_365,
        },
        "holdout": {k: holdout_bt.get(k) for k in ("sharpe_ratio", "total_trades", "total_return")},
        "result": result,
    }


def worker_null(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """R44/R47: draw ``count`` random chromosomes the GA's way and score them on the train window.

    Fitness (``adjusted_score``, what ``select_picks`` needs) comes from the SHARED ``_cell_fns``
    cell functions; the corr comes from the harvest's own train-window screen of the chromosome's
    DSL against the planted rule's train series (design-l2-followup.md §1(b)).
    """
    from vibe_quant.discovery.fitness import MIN_TRADES, _evaluate_single
    from vibe_quant.discovery.genome import chromosome_to_dsl

    root = Path(payload["root"]).resolve()
    _assert_worker_cwd(root)
    pool = payload.get("pool") or NULL_INDICATOR_POOL
    profile = Profile.from_payload(payload)
    fns, _settings = _cell_fns(profile)
    direction = Direction(profile.direction)
    base = int(payload["seed_base"]) + int(payload.get("offset", 0))
    count = int(payload["count"])
    start, end = train_window(profile)
    jobs: list[dict[str, Any]] = [
        {"tag": "planted", "dsl": planted_dsl(profile), "start": start, "end": end,
         "collect": True}
    ]
    entries: list[dict[str, Any]] = []
    for j in range(count):
        seed = base + j
        chrom = null_chromosome(seed, pool, direction)
        fit = _evaluate_single(chrom, fns.backtest_fn, None, MIN_TRADES, profile.timeframe)  # type: ignore[arg-type]
        dsl = chromosome_to_dsl(chrom)
        dsl["timeframe"] = profile.timeframe
        jobs.append({"tag": f"c{j}", "dsl": dsl, "start": start, "end": end, "collect": True})
        entries.append({
            "seed": seed, "score": float(fit.adjusted_score),
            "chromosome": chromosome_to_serializable(chrom),
        })
    shots = {r["tag"]: r for r in worker_screen({"root": str(root), "jobs": jobs})}
    planted_series = shots["planted"]["mtm"]
    for j, entry in enumerate(entries):
        entry["corr"] = signed_corr(shots[f"c{j}"]["mtm"], planted_series)
    return entries


WORKERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "screen": worker_screen,
    "chain": worker_chain,
    "null": worker_null,
}


# --- subcommands ----------------------------------------------------------------------------


class Ctx:
    """Resolved paths of one invocation."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Absolute from the start: children run with cwd = scratch root, and paths are persisted.
        self.out = Path(args.out).resolve()
        self.data_root = Path(args.data_root).resolve()
        self.real_db = Path(args.real_db).resolve()
        self.archive = (
            Path(args.archive).resolve() if args.archive
            else self.data_root / "archive" / "raw_data.db"
        )
        self.scratch = Path(args.scratch).resolve() if args.scratch else self.out.parent / "scratch"
        self.force = bool(getattr(args, "force", False))
        self.scratch_db = self.scratch / "parent-state.db"

    def accepted_root(self, state: dict[str, Any], variant: str) -> Path:
        acc = state.get("variants", {}).get(variant, {}).get("accepted")
        if not acc:
            msg = f"variant {variant} has no accepted build; run `build --variant {variant}`"
            raise Layer2Error(msg)
        root = Path(acc["root"]).resolve()
        assert_scratch_root(root, self.data_root)
        return root


def cmd_snapshot(ctx: Ctx, _args: argparse.Namespace) -> int:
    if "snapshot" in load_state(ctx.out) and not ctx.force:
        print("snapshot already recorded (kept; --force re-records)")
        return 0
    snap = take_snapshot(ctx.data_root, ctx.real_db, ctx.archive)
    update_state(ctx.out, lambda s: s.__setitem__("snapshot", snap))
    rows = snap["discovery_rows"]
    print(f"snapshot: {len(snap['files'])} files, discovery rows {rows}, funding {snap['funding_source_count']}")
    return 0


def cmd_build(ctx: Ctx, args: argparse.Namespace) -> int:
    variant = args.variant
    state = load_state(ctx.out)
    vrec = state.get("variants", {}).get(variant, {})
    if vrec.get("accepted") and Path(vrec["accepted"]["root"]).exists() and not ctx.force:
        print(f"{variant}: accepted build {vrec['accepted']['k']} already recorded")
        return 0
    if vrec.get("status") == "failed_price_ratio" and not ctx.force:
        print(f"{variant}: FAILED closed (price ratio > 3x); --force rebuilds")
        return 2
    cfg = VARIANTS[variant]
    builds: list[dict[str, Any]] = [] if ctx.force else list(vrec.get("builds", []))

    def _invalidate(s: dict[str, Any]) -> None:
        # Every result below was measured on the OLD plant: drop it before a new one exists,
        # so a rebuild that fails or is interrupted cannot leave stale proofs behind.
        v = _variant(s, variant)
        for key in ("accepted", "check_btc0", "direct_chain", "runs"):
            v.pop(key, None)
        v["status"] = "building"

    update_state(ctx.out, _invalidate)
    klines = plant.load_klines(ctx.archive)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    limit = 1 if cfg["series"] is None else MAX_BUILDS
    accepted: dict[str, Any] | None = None
    while len(builds) < limit:
        k = len(builds) + 1
        delta = 0.0 if cfg["series"] is None else next_delta(
            variant, builds, args.delta0 if args.delta0 is not None else float(cfg["delta0"])
        )
        rec = build_one(variant, k, delta, klines, ctx.archive, ctx.scratch, ctx.scratch_db)
        rec["accepted"] = accepts(variant, rec) and not rec["price_ratio_gt_3"]
        builds.append(rec)

        def _save(s: dict[str, Any], rec: dict[str, Any] = rec) -> None:
            v = _variant(s, variant)
            v["builds"] = list(builds)
            v["status"] = "accepted" if rec["accepted"] else "calibrating"
            if rec["accepted"]:
                v["accepted"] = rec

        update_state(ctx.out, _save)
        print(
            f"{variant} b{k}: delta={delta:.3e} kappa={rec['kappa']:.3e} l_end={rec['l_end']:.4f} "
            f"converged={rec['converged']} triggers={rec['n_triggers']} ratio={rec['price_ratio']:.2f} "
            f"train_S={rec['train_S']} full_S={rec['full_S']} accepted={rec['accepted']}"
        )
        if rec["price_ratio_gt_3"]:
            # W-R6: a plant that moves the price > 3x is not a valid plant; never accepted.
            for b in builds:
                shutil.rmtree(b["root"], ignore_errors=True)
            update_state(ctx.out, lambda s: _variant(s, variant).__setitem__("status", "failed_price_ratio"))
            print(f"{variant}: FAILED closed, price ratio {rec['price_ratio']:.2f} > 3x (no downstream steps)")
            return 2
        if rec["accepted"]:
            accepted = rec
            break
    if accepted is None:
        update_state(ctx.out, lambda s: _variant(s, variant).__setitem__("status", "not_calibrated"))
        print(f"{variant}: NOT calibrated after {len(builds)} builds")
        return 2
    for b in builds:  # keep only the accepted root; the JSON keeps every build's numbers
        root = Path(b["root"])
        if b is not accepted and root.exists() and _under(root, ctx.scratch):
            shutil.rmtree(root)
    return 0


def cmd_check_btc0(ctx: Ctx, _args: argparse.Namespace) -> int:
    state = load_state(ctx.out)
    vrec = state.get("variants", {}).get("btc0", {})
    if vrec.get("check_btc0", {}).get("passed") is True and not ctx.force:
        print("btc0 check already passed")
        return 0
    root = ctx.accepted_root(state, "btc0")
    jobs = [
        {"tag": "s239", "dsl_config": read_strategy_239(ctx.real_db),
         "start": WINDOW_239[0], "end": WINDOW_239[1]},
        {"tag": "planted_w3", "dsl": planted_dsl(), "start": W3_FULL[0], "end": W3_FULL[1]},
    ]
    res = {r["tag"]: r for r in run_worker("screen", root, {"jobs": jobs}, ctx.scratch_db)}
    diffs = {
        "s239": compare_exact(res["s239"], EXPECTED_239),
        "planted_w3": compare_exact(res["planted_w3"], EXPECTED_PLANTED_W3),
    }
    passed = not any(diffs.values())
    rec = {
        "passed": passed, "diffs": diffs, "checked_at": _now(),
        "got": {t: {k: r[k] for k in ("sharpe", "trades", "return", "profit_factor", "max_drawdown")}
                for t, r in res.items()},
    }
    update_state(ctx.out, lambda s: _variant(s, "btc0").__setitem__("check_btc0", rec))
    print(f"R25 btc0 fidelity: {'IDENTICAL' if passed else 'DRIFT'}")
    for tag, items in diffs.items():
        for item in items:
            print(f"  {tag} {item}")
    return 0 if passed else 3


def cmd_direct_chain(ctx: Ctx, args: argparse.Namespace) -> int:
    variant = args.variant
    state = load_state(ctx.out)
    if state.get("variants", {}).get(variant, {}).get("direct_chain") and not ctx.force:
        print(f"{variant}: direct chain already recorded")
        return _chain_exit(state, variant)
    root = ctx.accepted_root(state, variant)
    out = run_worker("chain", root, {}, ctx.scratch_db)
    result = json.loads(json.dumps(out["result"], default=_json_default))
    ns = POSITIVE_CONTROL_NS
    failures = positive_control_failures(result, ns)
    rec = {
        "fitness": out["fitness"], "holdout": out["holdout"], "result": result,
        "failures": failures, "report_ns": list(REPORT_NS), "checked_at": _now(),
    }
    update_state(ctx.out, lambda s: _variant(s, variant).__setitem__("direct_chain", rec))
    print(f"{variant} direct chain: fitness={out['fitness']}")
    print(f"  marginals={result['marginals']} floor={result['floor']}")
    print(f"  dsr={ {n: result['dsr'].get(str(n)) for n in REPORT_NS} }")
    print(f"  chain_recent={ {n: result['chain_recent'].get(str(n)) for n in REPORT_NS} }")
    return _chain_exit(load_state(ctx.out), variant)


def _chain_exit(state: dict[str, Any], variant: str) -> int:
    """R28: only the btc5.0 positive control must pass; failure is a machinery defect (exit 3)."""
    fails = state["variants"][variant]["direct_chain"]["failures"]
    if variant == "btc5.0" and fails:
        print(f"R28 positive control FAILED: {', '.join(fails)} -- STOP, report to the chief")
        return 3
    return 0


def cmd_discover(ctx: Ctx, args: argparse.Namespace) -> int:
    variant, seed, label = args.variant, int(args.seed), args.label
    if label is not None and not LABEL_RE.fullmatch(label):
        msg = f"--label {label!r} must be one path component matching [A-Za-z0-9_-]+"
        raise Layer2Error(msg)
    state = load_state(ctx.out)
    if label is not None:
        existing = state.get(FOLLOWUP_RUNS_KEY, {}).get(label)
        where = f"{variant} label {label}"
        # 7vn2j: a label belongs to the variant that first used it — reusing it would either
        # silently skip this run or overwrite the other variant's comparator (harvest --followup
        # needs btc0's). Refused before any mkdir/rmtree/seed row/launch, with and without --force.
        if existing is not None:
            owner = existing.get("variant")
            if owner is None:
                msg = (f"label {label!r}: the recorded follow-up run has no variant; refusing "
                       f"to reuse it for {variant}")
                raise Layer2Error(msg)
            if owner != variant:
                msg = (f"label {label!r} is already recorded for variant {owner!r}; refusing "
                       f"to reuse it for {variant}")
                raise Layer2Error(msg)
    else:
        existing = state.get("variants", {}).get(variant, {}).get("runs", {}).get(str(seed))
        where = f"{variant} seed {seed}"
    if existing and not ctx.force:
        print(f"{where}: already recorded ({existing['status']})")
        return 0
    root = ctx.accepted_root(state, variant)
    run_dir = ctx.scratch / "runs" / (f"{variant}-{label}" if label is not None else f"{variant}-seed{seed}")
    if not _under(run_dir, ctx.scratch):  # before any rmtree/mkdir/seed row/launch
        msg = f"refusing run dir {run_dir}: outside {ctx.scratch}"
        raise Layer2Error(msg)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # R42: the seed row lives in the run's OWN db, written before run_discovery adds its row.
    seed_run = create_seed_run(run_dir / "state.db") if args.warm else None
    rec = run_discovery(
        root, run_dir, seed, timeout_s=args.timeout,
        command_fn=discover_command_fn(args.pop, args.gens, seed_run),
    )
    rec = {**rec, "pop": args.pop, "gens": args.gens}  # R47: the run's own GA budget
    if label is not None:
        rec = {**rec, "label": label, "variant": variant, "seed": seed}

    def _save(s: dict[str, Any], r: dict[str, Any] = rec) -> None:
        # R41: a labelled follow-up run goes to followup_runs, never into variants[*]["runs"].
        if label is not None:
            s.setdefault(FOLLOWUP_RUNS_KEY, {})[label] = r
        else:
            _variant(s, variant).setdefault("runs", {})[str(seed)] = r

    update_state(ctx.out, _save)
    detail = rec.get("error", "")
    print(f"{where}: {rec['status']} run_id={rec['run_id']} {rec['wall_s']}s {detail}")
    return 0


def cmd_null_corr(ctx: Ctx, args: argparse.Namespace) -> int:
    """R44/R47: draw the random-chromosome null pool on btc0 and record its scratch pool file.

    ``m`` draws are split across ``shards`` worker processes; the merged pool (per-chromosome seed,
    ``adjusted_score`` and planted-train corr) goes to ``scratch/null-pool.json`` and the path is
    recorded in ``state["null"]``. C95(K) is computed at harvest time, so a later 50x20 run's K is
    supported without re-running the pool.
    """
    state = load_state(ctx.out)
    existing = state.get(NULL_RECORD_KEY)
    if isinstance(existing, dict) and existing.get("status") == "completed" and not ctx.force:
        print("null-corr already recorded (kept; --force redoes)")
        return 0
    root = ctx.accepted_root(state, "btc0")
    m, shards = int(args.m), int(args.shards)
    chunk = -(-m // shards)

    def shard(i: int, count: int) -> list[dict[str, Any]]:
        # q2g9o/R6: the payload is the sequential one verbatim — each shard is a separate
        # ``_worker`` subprocess with its own tempdir, so the shards are independent and the
        # merged pool stays seed-deterministic (R7).
        return run_worker(
            "null", root,
            {"count": count, "seed_base": int(args.seed), "offset": i * chunk,
             "pool": list(NULL_INDICATOR_POOL)},
            ctx.scratch_db, timeout_s=args.timeout,
        )

    # R9: a count-0 shard draws nothing, exactly as the sequential loop skipped it.
    jobs = [(i, max(0, min(chunk, m - i * chunk))) for i in range(shards)]
    jobs = [(i, count) for i, count in jobs if count]
    pool: list[dict[str, Any]] = []
    if jobs:
        # R6: one worker thread per non-empty shard (the m=1200 pool was ~60 min sequential).
        # R8: the pool is assembled only after every shard returns, so a shard that raises (via
        # fut.result()) propagates with no pool file and no state["null"] — nothing half-written.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool_ex:
            futures = [pool_ex.submit(shard, i, count) for i, count in jobs]
            pool = [entry for fut in futures for entry in fut.result()]
    pool.sort(key=lambda entry: entry["seed"])
    pool_path = ctx.scratch / NULL_POOL_FILE
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(pool, default=_json_default) + "\n")
    rec = {
        "status": "completed", "m": len(pool), "reps": int(args.reps), "k": [], "thresholds": {},
        "seed": int(args.seed), "shards": shards, "root": str(root), "pool_path": str(pool_path),
        "recorded_at": _now(),
    }
    update_state(ctx.out, lambda s: s.__setitem__(NULL_RECORD_KEY, rec))
    print(f"null-corr: m={len(pool)} reps={rec['reps']} root={root} pool={pool_path}")
    return 0


# --- R43 rescreen / R47 follow-up harvest (design-l2-followup.md §1(a)/(d)) ------------------


def rescreen_candidate(
    state: dict[str, Any], run_variant: str, seed: int, uid: str
) -> dict[str, Any]:
    """R43: the harvested candidate ``uid`` of one completed run; an absent uid is refused."""
    run = (state.get("variants", {}).get(run_variant, {}).get("runs") or {}).get(str(seed))
    if not run or run.get("status") != "completed":
        msg = f"{run_variant} seed {seed} has no completed run to rescreen"
        raise Layer2Error(msg)
    candidates: list[dict[str, Any]] = (run.get("harvest") or {}).get("candidates") or []
    cand: dict[str, Any] | None = next((c for c in candidates if c.get("uid") == uid), None)
    if cand is None:
        msg = f"uid {uid!r} is not a harvested candidate of {run_variant} seed {seed}"
        raise Layer2Error(msg)
    # the harvest record keeps no DSL (uid/source/stage/trades/train_S/corr): re-read it from the
    # run's own DB, which works for state harvested before this lookup existed
    full = next(
        (c for c in harvest_run(Path(run["db"]), int(run["run_id"]))["candidates"]
         if c.get("uid") == uid and c.get("dsl")),
        None,
    )
    if full is None:
        msg = f"uid {uid!r} has no DSL in the run DB of {run_variant} seed {seed}"
        raise Layer2Error(msg)
    return {**cand, "dsl": full["dsl"]}


def rescreen_jobs(candidate_dsl: dict[str, Any], profile: Profile = P4H) -> list[dict[str, Any]]:
    """The three rules of §1(a), each on the TRAIN window: planted, planted-as-BOTH, candidate.

    A non-default profile is a LONG study: the BOTH diagnostic is omitted (``planted_both_dsl`` is
    the fixed 4h rule and would silently score another timeframe and chromosome)."""
    start, end = train_window(profile)
    jobs = [{"tag": "planted", "dsl": planted_dsl(profile), "start": start, "end": end,
             "collect": True}]
    if profile == P4H:
        jobs.append({"tag": "planted_both", "dsl": planted_both_dsl(), "start": start, "end": end,
                     "collect": True})
    jobs.append({"tag": "candidate", "dsl": candidate_dsl, "start": start, "end": end,
                 "collect": True})
    return jobs


def cmd_rescreen(ctx: Ctx, args: argparse.Namespace) -> int:
    state = load_state(ctx.out)
    seed = int(args.seed)
    cand = rescreen_candidate(state, args.run_variant, seed, args.uid)
    jobs = rescreen_jobs(cand["dsl"])
    variants: dict[str, Any] = {}
    for variant in RESCREEN_VARIANTS:
        root = ctx.accepted_root(state, variant)
        shots = {r["tag"]: r for r in run_worker("screen", root, {"jobs": jobs}, ctx.scratch_db)}
        planted_series = shots["planted"]["mtm"]
        record: dict[str, Any] = {"root": str(root)}
        for tag in ("planted", "planted_both", "candidate"):
            record[tag] = {
                "trades": shots[tag]["trades"],
                "train_S": ann_sharpe(shots[tag]["mtm"]),
                "corr": signed_corr(shots[tag]["mtm"], planted_series),
            }
        variants[variant] = record
    rec = {
        "uid": args.uid, "run_variant": args.run_variant, "seed": seed,
        "variants": variants, "checked_at": _now(),
    }

    def _save(s: dict[str, Any], r: dict[str, Any] = rec) -> None:
        s.setdefault(RESCREEN_RECORD_KEY, {})[args.uid] = r

    update_state(ctx.out, _save)
    print(f"rescreen {args.uid} ({args.run_variant} seed {seed}):")
    for variant, record in variants.items():
        print(f"  {variant}: candidate corr={record['candidate']['corr']} "
              f"train_S={record['candidate']['train_S']}")
    return 0


def require_null_record(state: dict[str, Any]) -> dict[str, Any]:
    """R47: the completed calibrated null record (T2's ``null-corr`` output) or a refusal.

    T2 fills in the record's calibrated threshold comparison; T1 pins the key and the refusal
    so ``harvest --followup`` cannot score against an uncalibrated (or missing) null.
    """
    record = state.get(NULL_RECORD_KEY)
    if not isinstance(record, dict) or record.get("status") != "completed":
        msg = (
            f"harvest --followup needs the calibrated null record (state[{NULL_RECORD_KEY!r}], "
            "status 'completed'): run `null-corr` first"
        )
        raise Layer2Error(msg)
    return record


def _harvest_one(ctx: Ctx, root: Path, run: dict[str, Any], profile: Profile = P4H) -> bool:
    """Harvest ONE completed run into ``run["harvest"]``; returns its R31 isolation flag."""
    h = harvest_run(Path(run["db"]), int(run["run_id"]))
    ok = bool(h["r31"]["ok"])
    cands = h.pop("candidates")
    start, end = train_window(profile)
    jobs = [{"tag": "planted", "dsl": planted_dsl(profile), "start": start, "end": end,
             "collect": True}]
    jobs += [
        {"tag": f"c{i}", "dsl": c["dsl"], "start": start, "end": end, "collect": True}
        for i, c in enumerate(cands)
    ]
    shots = {r["tag"]: r for r in run_worker("screen", root, {"jobs": jobs}, ctx.scratch_db)}
    planted_series = shots["planted"]["mtm"]
    planted_s = ann_sharpe(planted_series)
    rows = []
    for i, cand in enumerate(cands):
        shot = shots[f"c{i}"]
        rows.append({
            "uid": cand.get("uid"), "source": cand["source"], "stage": cand.get("stage"),
            "trades": shot["trades"], "train_S": ann_sharpe(shot["mtm"]),
            "corr": signed_corr(shot["mtm"], planted_series),
            "dsr": cand.get("dsr"),
        })
    h["found"] = assess_found(rows, planted_s)
    h["candidates"] = [{k: v for k, v in r.items() if k != "dsr"} for r in rows]
    h["posthoc_p"] = {"ns": list(POSTHOC_NS), "by_candidate": posthoc_p(rows, POSTHOC_NS)}
    run["harvest"] = h
    return ok


def harvest_variant(ctx: Ctx, state: dict[str, Any], variant: str) -> tuple[dict[str, Any], bool]:
    """Harvest every completed run of a variant; returns (runs, isolation_ok)."""
    vrec = state["variants"][variant]
    root = ctx.accepted_root(state, variant)
    runs: dict[str, Any] = vrec.get("runs", {})
    ok = True
    for run in runs.values():
        if run.get("status") != "completed":
            continue
        ok = _harvest_one(ctx, root, run) and ok
    return runs, ok


# --- R47: calibrated harvest (design-l2-followup.md §1(b), P-1a threshold) -------------------


def _run_budget(run: dict[str, Any]) -> tuple[int, int]:
    """The GA budget (pop, gens) a run was launched with; an absent budget is the 12x4 default."""
    return int(run.get("pop", DEFAULT_POP)), int(run.get("gens", DEFAULT_GENS))


def _run_k(run: dict[str, Any]) -> int:
    """R47: K = 40 for a 12x4 run (pre-registered literal); else the run's evaluated count."""
    pop, gens = _run_budget(run)
    if (pop, gens) == (DEFAULT_POP, DEFAULT_GENS):
        return NULL_K_12X4
    evaluated = (run.get("harvest") or {}).get("evaluated")
    if evaluated is None:
        msg = f"run at budget {pop}x{gens} has no harvest evaluated count for its K"
        raise Layer2Error(msg)
    return int(evaluated)


def _iter_runs(state: dict[str, Any]) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """Every recorded run as ``(variant, run)``: the V5 runs and the labelled follow-up runs."""
    for variant, vrec in state.get("variants", {}).items():
        for run in (vrec.get("runs") or {}).values():
            yield variant, run
    for run in (state.get(FOLLOWUP_RUNS_KEY) or {}).values():
        yield run.get("variant"), run


def _btc0_max_corr(state: dict[str, Any], pop: int, gens: int) -> float:
    """R47/P-1a: the max SIGNED corr over ALL btc0 GA candidates at the same (pop, gens) budget."""
    corrs: list[float] = []
    for variant, run in _iter_runs(state):
        if variant != "btc0" or run.get("status") != "completed" or _run_budget(run) != (pop, gens):
            continue
        for cand in (run.get("harvest") or {}).get("candidates") or []:
            if cand.get("corr") is not None:
                corrs.append(float(cand["corr"]))
    if not corrs:
        msg = f"harvest --followup needs btc0 GA candidates at budget {pop}x{gens}; none recorded"
        raise Layer2Error(msg)
    return max(corrs)


def _found_cal(harvest: dict[str, Any], c_k: float) -> dict[str, Any]:
    """R47: found_cal over a run's harvested candidates under the calibrated threshold C_K."""
    planted_s = (harvest.get("found") or {}).get("planted_train_S")
    flagged = [
        {"uid": cand.get("uid"), "corr": cand.get("corr"), "train_S": cand.get("train_S")}
        for cand in harvest.get("candidates") or []
        if found_criterion(cand.get("corr"), cand.get("train_S"), planted_s, corr_min=c_k)
    ]
    return {"found_cal": bool(flagged), "found_by_cal": flagged}


def harvest_followup(ctx: Ctx, state: dict[str, Any]) -> int:
    """R47: re-score every V5 and follow-up run against the calibrated null threshold C_K.

    C95(K) is computed from the stored null pool at a fixed subsample seed; the btc0 comparator is
    the max signed corr over ALL btc0 candidates at the run's own (pop, gens) budget; the
    thresholds actually used are written back into ``state["null"]``.
    """
    record = require_null_record(state)
    violations: list[str] = []
    for variant, run in list(_iter_runs(state)):
        if run.get("status") != "completed":
            continue
        if not run.get("harvest"):
            if variant is None:
                msg = "a completed run has no variant to harvest against"
                raise Layer2Error(msg)
            ok = _harvest_one(ctx, ctx.accepted_root(state, variant), run)
        else:  # cached: a missing r31 record fails closed
            ok = bool(((run["harvest"].get("r31")) or {}).get("ok"))
        if not ok:
            violations.append(f"{variant} run_id={run.get('run_id')}")
    if violations:  # R31: score nothing and persist nothing
        print(f"R31 isolation VIOLATED in {', '.join(violations)}: harvest --followup scores nothing")
        return 3

    scorable = [
        (variant, run)
        for variant, run in _iter_runs(state)
        if variant != "btc0" and run.get("status") == "completed" and run.get("harvest")
    ]
    if not scorable:
        print("harvest --followup: no completed V5/follow-up runs to score")
        return _write_followup(ctx, state, record)

    pool = load_null_pool(Path(record["pool_path"]))
    reps = int(record["reps"])
    c95_by_k = {k: null_threshold(pool, k, reps)["c95"] for k in {_run_k(r) for _v, r in scorable}}
    for variant, run in scorable:
        k = _run_k(run)
        pop, gens = _run_budget(run)
        btc0_max = _btc0_max_corr(state, pop, gens)
        c_k = corr_threshold(c95_by_k[k], btc0_max)
        calibration = {
            "k": k, "budget": [pop, gens], "c95": c95_by_k[k], "btc0_max_corr": btc0_max, "C_K": c_k,
        }
        calibration.update(_found_cal(run["harvest"], c_k))
        run["harvest"]["calibration"] = calibration
        print(f"harvest --followup {variant}: K={k} C95={c95_by_k[k]:.4f} btc0_max={btc0_max:.4f} "
              f"C_K={c_k:.4f} found_cal={calibration['found_cal']}")
    record["k"] = sorted(c95_by_k)
    record["thresholds"] = {str(k): v for k, v in sorted(c95_by_k.items())}
    return _write_followup(ctx, state, record)


def _write_followup(ctx: Ctx, state: dict[str, Any], record: dict[str, Any]) -> int:
    """Write every harvested run and the updated null thresholds back under one lock."""
    def _save(s: dict[str, Any]) -> None:
        for variant, vrec in state.get("variants", {}).items():
            _variant(s, variant)["runs"] = vrec.get("runs", {})
        s[FOLLOWUP_RUNS_KEY] = state.get(FOLLOWUP_RUNS_KEY, {})
        s[NULL_RECORD_KEY] = record

    update_state(ctx.out, _save)
    return 0


def cmd_harvest(ctx: Ctx, args: argparse.Namespace) -> int:
    state = load_state(ctx.out)
    if args.followup:
        return harvest_followup(ctx, state)
    names = [args.variant] if args.variant else [
        v for v, r in state.get("variants", {}).items() if r.get("runs")
    ]
    all_ok = True
    for variant in names:
        runs, ok = harvest_variant(ctx, state, variant)
        all_ok = all_ok and ok

        def _save(s: dict[str, Any], v: str = variant, r: dict[str, Any] = runs) -> None:
            _variant(s, v)["runs"] = r

        update_state(ctx.out, _save)
    for line in verdict_lines(load_state(ctx.out)):
        print(line)
    if not all_ok:
        print("R31 isolation VIOLATED: a run saw prior trials, gate_n != run_n_eff or no effective-N marker")
        return 3
    return 0


def cmd_verify_untouched(ctx: Ctx, _args: argparse.Namespace) -> int:
    state = load_state(ctx.out)
    snap = state.get("snapshot")
    if snap is None:
        print("no snapshot recorded; run `snapshot` first")
        return 2
    problems = compare_snapshot(snap, ctx.data_root, ctx.real_db, ctx.archive)
    for variant, vrec in state.get("variants", {}).items():
        acc = vrec.get("accepted")
        scratch_arch = Path(acc["root"]) / "data" / "archive" / "raw_data.db" if acc else None
        if scratch_arch is None or not scratch_arch.exists():
            continue
        count = funding_count(scratch_arch)
        if count != snap["funding_source_count"]:
            problems.append(f"{variant}: scratch funding rows {count} != source {snap['funding_source_count']}")
    rec = {"ok": not problems, "problems": problems, "checked_at": _now()}
    update_state(ctx.out, lambda s: s.__setitem__("verify", rec))
    print("R26 shared data untouched" if not problems else "R26 shared data CHANGED")
    for p in problems:
        print(f"  {p}")
    for line in verdict_lines(load_state(ctx.out)):
        print(line)
    return 0 if not problems else 3


def cmd_worker(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.inp).read_text())
    result = WORKERS[args.kind](payload)
    Path(args.outp).write_text(json.dumps(result, default=_json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="the one JSON all steps write")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--real-db", default=str(REAL_DB), help="read-only (R26 rows, strategy 239)")
    parser.add_argument("--archive", default=None, help="source raw_data.db (default data-root)")
    parser.add_argument("--scratch", default=None, help="scratch dir (default <out dir>/scratch)")
    parser.add_argument("--force", action="store_true", help="redo a recorded step")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="R26: record signatures before any build")
    p = sub.add_parser("build", help="R27: plant + calibrate (<= 4 fresh builds)")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--delta0", type=float, default=None)
    sub.add_parser("check-btc0", help="R25: btc0 scratch reproduces the real-catalog numbers")
    p = sub.add_parser("direct-chain", help="R28: planted rule through the real gate chain")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p = sub.add_parser("discover", help="one isolated discovery run")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timeout", type=float, default=RUN_TIMEOUT_S)
    p.add_argument("--pop", type=int, default=DEFAULT_POP, help="R40: GA population size")
    p.add_argument("--gens", type=int, default=DEFAULT_GENS, help="R40: GA max generations")
    p.add_argument("--label", default=None, help="R41: record under followup_runs[label]")
    p.add_argument("--warm", action="store_true", help="R42: warm-start from a planted seed row")
    p = sub.add_parser("null-corr", help="R44/R47: random-chromosome null pool on btc0")
    p.add_argument("--m", type=int, default=NULL_DEFAULT_M, help="draws in the null pool (>= K)")
    p.add_argument("--reps", type=int, default=NULL_DEFAULT_REPS, help="subsamples for C95")
    p.add_argument("--shards", type=int, default=NULL_DEFAULT_SHARDS, help="parallel worker count")
    p.add_argument("--seed", type=int, default=0, help="first chromosome seed")
    p.add_argument("--timeout", type=float, default=7200.0, help="per-shard worker timeout (s)")
    p = sub.add_parser("rescreen", help="R43: a harvested candidate on all three accepted roots")
    p.add_argument("--run-variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--uid", required=True)
    p = sub.add_parser("harvest", help="R30/R31: champions, rejections, found, post-hoc p")
    p.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    p.add_argument("--followup", action="store_true", help="R47: calibrated-threshold harvest")
    sub.add_parser("verify-untouched", help="R26: shared data identical to the snapshot")
    w = sub.add_parser("_worker")
    w.add_argument("kind", choices=sorted(WORKERS))
    w.add_argument("--in", dest="inp", required=True)
    w.add_argument("--out", dest="outp", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "_worker":
        return cmd_worker(args)
    ctx = Ctx(args)
    handlers: dict[str, Callable[[Ctx, argparse.Namespace], int]] = {
        "snapshot": cmd_snapshot,
        "build": cmd_build,
        "check-btc0": cmd_check_btc0,
        "direct-chain": cmd_direct_chain,
        "discover": cmd_discover,
        "null-corr": cmd_null_corr,
        "rescreen": cmd_rescreen,
        "harvest": cmd_harvest,
        "verify-untouched": cmd_verify_untouched,
    }
    try:
        return handlers[args.cmd](ctx, args)
    except Layer2Error as exc:
        print(f"layer2: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
