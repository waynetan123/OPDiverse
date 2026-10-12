"""The shared run queue for the day: every M2 run not yet evaluated, longest estimated first.

Each run's estimate is step_count.json's run time for its arm (o_a + S x t_a + E). The order is longest first, ties
broken by GRPO first, then arm, configuration and seed; train.launch --plan gives the next job to whichever GPU frees
first, so GRPO starts first and the real run times decide the rest. The forecast simulates that on the estimates:
each job to the GPU that frees first (the least work so far; ties to the lowest index). It is a forecast only.

Runs whose test generations already exist are left out, so after a failure the same command queues what is left.
"""

from __future__ import annotations

import json

from converters import masks
from etl import pinned
from etl.build import write_json
from etl.paths import Paths

from .files import ShortFiles, lr


def job_spec(arm: str, config: str, seed: int) -> str:
    """train.launch's job spec: arm@lr@config@seed."""
    return f"{arm}@{lr(arm):.0e}@{config}@{seed}"


def order(jobs: list[tuple[str, str, int]], hours: dict[str, float]) -> list[tuple[str, str, int]]:
    """Longest estimated run first; ties: GRPO first, then arm, configuration and seed order."""
    return sorted(jobs, key=lambda j: (-hours[j[0]], j[0] != "grpo", pinned.SHORT_ARMS.index(j[0]),
                                       masks.M2.index(j[1]), j[2]))


def forecast(jobs: list[tuple[str, str, int]], hours: dict[str, float], gpus: list[str]) -> dict[str, dict]:
    """Each job, in order, to the GPU that frees first on the estimates (ties to the earlier GPU in `gpus`)."""
    out = {g: {"jobs": [], "hours": 0.0} for g in gpus}
    for job in jobs:
        g = min(gpus, key=lambda x: (out[x]["hours"], gpus.index(x)))
        out[g]["jobs"].append(job_spec(*job))
        out[g]["hours"] += hours[job[0]]
    return out


def build(paths: Paths, gpus: list[str]) -> dict:
    files = ShortFiles.of(paths)
    if not files.step_count.exists():
        raise SystemExit("step_count.json is missing: run `python -m short steps` on the cost pilot first")
    count = json.loads(files.step_count.read_text(encoding="utf-8"))
    if pinned.TRAIN_STEPS != count["steps"]:
        raise SystemExit(f"pinned.TRAIN_STEPS is {pinned.TRAIN_STEPS}, step_count.json says {count['steps']}: "
                         "pin it and commit before queueing")
    hours = count["run_hours"]
    every = [(a, c, s) for a in count["arms"] for c in masks.M2 for s in pinned.SHORT_SEEDS]
    done = [j for j in every if files.generations(*j).exists()]
    jobs = order([j for j in every if j not in done], hours)
    plan = forecast(jobs, hours, gpus)
    finish = max(p["hours"] for p in plan.values()) if jobs else 0.0
    limit = count["hours_left"] - pinned.SHORT_RESERVE_HOURS
    out = {
        "steps": count["steps"],
        "fallback": count["fallback"],
        "arms": count["arms"],
        "jobs": [job_spec(*j) for j in jobs],
        "already_evaluated": [job_spec(*j) for j in done],
        "costs_hours": hours,
        "gpus": gpus,
        "forecast": plan,
        "forecast_finish_hours": finish,
        "hours_left_at_step_count": count["hours_left"],
        "over_budget": finish > limit,
    }
    write_json(files.queues, out)
    return out


def launch_command(paths: Paths, gpus: list[str]) -> str:
    files = ShortFiles.of(paths)
    return (f"PYTHONPATH=src python -m train.launch --plan {files.queues} --gpus {' '.join(gpus)} --retries 1 "
            f"--runner short.job --out-dir {files.runs}")
