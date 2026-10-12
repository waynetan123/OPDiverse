"""The step-count rule (docs/OPDiverse_short.md, "Run length"): TRAIN_STEPS from the cost pilot's measured cost.

Per arm a, from its cost-pilot run (train.launch --out-root <pilot-root>, so <pilot-root>_<arm>_lr<lr>/):

    t_a   median seconds per step over pinned.SHORT_TIMED_STEPS (3..10), from train_log.jsonl's elapsed_s
    o_a   the rest of the run: load and checks plus training time, less steps_done x t_a
    E     pinned.SHORT_EVAL_HOURS per run (merge and test evaluation);  R  pinned.SHORT_RESERVE_HOURS;  H  hours left
    n     runs per arm (the five M2 configurations x the seeds);  k = ceil(n / SHORT_GPUS)

S is the largest multiple of SHORT_STEP_MULTIPLE, at most SHORT_MAX_STEPS, with both

    total work fits    n x sum_a (o_a + E) + n x S x sum_a t_a <= SHORT_GPUS x (H - R)
    busiest GPU fits   k x (o_a + S x t_a + E) <= H - R, for the arm with the longest run

The doc states the second for GRPO (ten runs on four GPUs, so one GPU runs three), the longest arm; taking the
longest arm gives the same condition and a defined one under the fallback. The fallback drops GRPO and recomputes
over the other arms if S < SHORT_MIN_STEPS or there is no GRPO cost run (it did not fit in colocate mode).

The result is written to step_count.json once; a recomputation that differs is refused, as for the sweep's scale.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from converters import masks
from etl import pinned
from etl.build import read_jsonl, write_json
from etl.manifest import file_sha256
from train.config import run_name as train_run_name

from .files import ShortFiles, lr

HOUR = 3600.0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def elapsed_by_step(records: list[dict]) -> dict[int, float]:
    """step -> elapsed_s of its training-loss record (train.run.log_record)."""
    out = {}
    for r in records:
        t = r.get("trainer")
        if isinstance(t, dict) and "loss" in t and isinstance(r.get("step"), int) and "elapsed_s" in r:
            out[r["step"]] = r["elapsed_s"]
    return out


def seconds_per_step(records: list[dict], timed: tuple[int, int] = pinned.SHORT_TIMED_STEPS) -> float:
    """The median duration of steps first..last: step k lasts elapsed(k) - elapsed(k - 1)."""
    first, last = timed
    at = elapsed_by_step(records)
    missing = [k for k in range(first - 1, last + 1) if k not in at]
    if missing:
        raise ValueError(f"no timed training record for steps {missing}")
    durations = [at[k] - at[k - 1] for k in range(first, last + 1)]
    if any(d <= 0 for d in durations):
        raise ValueError("elapsed_s does not increase step to step (a resumed run?); rerun the cost pilot")
    return statistics.median(durations)


def arm_cost(records: list[dict], meta: dict) -> dict:
    """{t: seconds per step, o: fixed seconds per run} from one cost-pilot run."""
    t = seconds_per_step(records)
    seconds = meta["seconds"]
    o = seconds["load_and_checks"] + seconds["train_this_session"] - meta["steps_done"] * t
    return {"t": t, "o": o, "steps_done": meta["steps_done"]}


def runs_per_arm() -> int:
    return len(masks.M2) * len(pinned.SHORT_SEEDS)


def run_seconds(cost: dict, steps: int) -> float:
    """One run's estimated time: its fixed part, its steps, and its merge plus test evaluation."""
    return cost["o"] + steps * cost["t"] + float(pinned.SHORT_EVAL_HOURS) * HOUR


def conditions(costs: dict[str, dict], steps: int, hours_left: float) -> dict:
    n, gpus = runs_per_arm(), pinned.SHORT_GPUS
    k = math.ceil(n / gpus)
    budget = (hours_left - pinned.SHORT_RESERVE_HOURS) * HOUR
    longest = max(costs, key=lambda a: (run_seconds(costs[a], steps), a))
    total = n * sum(run_seconds(c, steps) for c in costs.values())
    busiest = k * run_seconds(costs[longest], steps)
    return {
        "total": {"hours": total / HOUR, "limit_hours": gpus * budget / HOUR, "ok": total <= gpus * budget},
        "busiest": {"arm": longest, "runs": k, "hours": busiest / HOUR, "limit_hours": budget / HOUR,
                    "ok": busiest <= budget},
    }


def largest_steps(costs: dict[str, dict], hours_left: float) -> int:
    """The largest multiple of SHORT_STEP_MULTIPLE up to SHORT_MAX_STEPS meeting both conditions; 0 if none does.
    Both sides grow with S, so the first passing value from the top is the answer."""
    m = pinned.SHORT_STEP_MULTIPLE
    for s in range(pinned.SHORT_MAX_STEPS // m * m, 0, -m):
        c = conditions(costs, s, hours_left)
        if c["total"]["ok"] and c["busiest"]["ok"]:
            return s
    return 0


def choose(costs: dict[str, dict], hours_left: float) -> dict:
    """S over the arms with cost runs, or over the fallback arms if GRPO has none or S < SHORT_MIN_STEPS."""
    arms = [a for a in pinned.SHORT_ARMS if a in costs]
    if set(arms) != set(pinned.SHORT_ARMS):
        missing = sorted(set(pinned.SHORT_ARMS) - set(arms))
        if missing != ["grpo"]:
            raise ValueError(f"no cost run for {missing}; only GRPO may be missing (the fallback)")
    full = largest_steps({a: costs[a] for a in arms}, hours_left) if "grpo" in arms else None
    reason = None
    if full is None:
        reason = "no GRPO cost run (GRPO did not fit in colocate mode)"
    elif full < pinned.SHORT_MIN_STEPS:
        reason = f"S = {full} over four arms is below {pinned.SHORT_MIN_STEPS}"
    if reason:
        arms = list(pinned.SHORT_FALLBACK_ARMS)
    use = {a: costs[a] for a in arms}
    steps = largest_steps(use, hours_left) if reason else full
    return {
        "steps": steps,
        "fallback": reason is not None,
        "fallback_reason": reason,
        "steps_over_four_arms": full,
        "arms": arms,
        "fits": steps >= pinned.SHORT_MIN_STEPS,
        "conditions": conditions(use, steps, hours_left) if steps else None,
        "run_hours": {a: run_seconds(c, steps) / HOUR for a, c in use.items()},
    }


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def pilot_dir(root: Path, arm: str) -> Path:
    """train.launch --out-root's directory for a pilot run."""
    return Path(f"{root}_{train_run_name(arm, lr(arm))}")


def read_costs(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """(arm -> cost, arm -> sha256 of the files read) for every arm with a finished cost run."""
    costs, sources = {}, {}
    for arm in pinned.SHORT_ARMS:
        d = pilot_dir(root, arm)
        log, meta_path = d / "train_log.jsonl", d / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["config"]["arm"] != arm or not meta["pilot"]:
            raise SystemExit(f"{meta_path} is not a pilot run of {arm}")
        try:
            costs[arm] = arm_cost(read_jsonl(log), meta)
        except ValueError as e:
            raise SystemExit(f"{d}: {e}")
        sources[arm] = {"train_log_sha256": file_sha256(log), "run_meta_sha256": file_sha256(meta_path)}
    return costs, sources


def run(paths, pilot_root: Path | None, hours_left: float) -> dict:
    files = ShortFiles.of(paths)
    root = pilot_root or files.pilot / "cost"
    costs, sources = read_costs(root)
    try:
        out = choose(costs, hours_left)
    except ValueError as e:
        raise SystemExit(f"{root}: {e}")
    out = {**out, "hours_left": hours_left, "reserve_hours": pinned.SHORT_RESERVE_HOURS,
           "eval_hours": str(pinned.SHORT_EVAL_HOURS), "pilot_root": str(root),
           "costs": {a: {"seconds_per_step": c["t"], "overhead_seconds": c["o"], "pilot_steps": c["steps_done"]}
                     for a, c in costs.items()},
           "sources": sources}
    if files.step_count.exists():
        old = json.loads(files.step_count.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(out)):
            raise SystemExit("step_count.json is frozen and this computation differs (its steps "
                             f"{old['steps']}); a change needs a decision-record entry")
    write_json(files.step_count, out)
    return out
