"""The one-day run's CLI (docs/OPDiverse_short.md). Training, merging and evaluation run on the GPU machine through
`python -m train.launch --plan ... --runner short.job`; these commands need no GPU.

    PYTHONPATH=src python -m short prepare                      # -> data/short/test_requests.jsonl
    PYTHONPATH=src python -m short steps --hours-left 22        # cost pilot -> step_count.json: S, the fallback (frozen)
    PYTHONPATH=src python -m short queue [--gpus 0 1 2 3]       # -> queues.json: the runs left, longest first
    PYTHONPATH=src python -m short score [--arms sft distill_self dpo]   # -> m2_scores.jsonl, score_summary.{json,md}

`steps` reads the cost pilot in data/short/pilot/cost_<arm>_lr<lr>/ (or --pilot-root); pin its S as
pinned.TRAIN_STEPS and commit before `queue`. `score --arms` defaults to step_count.json's arms (three if the
fallback fired), else pinned.SHORT_ARMS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from etl import pinned
from etl.paths import DEFAULT, Paths

from . import prepare, queue, score, steps
from .files import ShortFiles

COMMANDS = ("prepare", "steps", "queue", "score")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m short", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--pilot-root", type=Path, default=None, help="steps: the cost pilot's train.launch --out-root")
    ap.add_argument("--hours-left", type=float, default=None, help="steps: hours left in the GPU window")
    ap.add_argument("--gpus", nargs="+", default=[str(g) for g in range(pinned.SHORT_GPUS)], help="queue: GPU indices")
    ap.add_argument("--arms", nargs="+", default=None, help="score: the arms trained (default: step_count.json's)")
    args = ap.parse_args(argv)
    paths = Paths(args.data_dir.resolve())
    files = ShortFiles.of(paths)

    if args.command == "prepare":
        out = prepare.prepare(paths)
        print(f"{out['items']:,} test requests ({out['cves']} CVEs), longest prompt {out['max_prompt_tokens']:,} "
              f"tokens -> {files.test_requests}")
    elif args.command == "steps":
        if args.hours_left is None:
            raise SystemExit("steps needs --hours-left: the hours left in the GPU window once the pins are committed")
        out = steps.run(paths, args.pilot_root, args.hours_left)
        for a, c in out["costs"].items():
            print(f"{a}: {c['seconds_per_step']:.1f} s/step, {c['overhead_seconds'] / 60:.1f} min fixed per run")
        if out["fallback"]:
            print(f"FALLBACK: GRPO dropped ({out['fallback_reason']}); the test runs on {', '.join(out['arms'])}")
        c = out["conditions"]
        if c:
            print(f"total work {c['total']['hours']:.1f} h of {c['total']['limit_hours']:.1f} h; busiest GPU "
                  f"({c['busiest']['runs']} x {c['busiest']['arm']}) {c['busiest']['hours']:.1f} h of "
                  f"{c['busiest']['limit_hours']:.1f} h")
        print(f"S = {out['steps']}" + ("" if out["fits"] else f": below {pinned.SHORT_MIN_STEPS}, the day does not fit")
              + f"; pin TRAIN_STEPS = {out['steps']} and commit (written to {files.step_count})")
    elif args.command == "queue":
        out = queue.build(paths, args.gpus)
        for g, f in out["forecast"].items():
            print(f"GPU {g}: {len(f['jobs'])} runs, ~{f['hours']:.1f} h (forecast)")
        print(f"{len(out['jobs'])} runs queued, {len(out['already_evaluated'])} already evaluated; forecast finish "
              f"~{out['forecast_finish_hours']:.1f} h" + ("  !! past the window less the reserve" if out["over_budget"] else ""))
        print(queue.launch_command(paths, args.gpus))
    else:
        arms = tuple(args.arms) if args.arms else (
            tuple(json.loads(files.step_count.read_text(encoding="utf-8"))["arms"]) if files.step_count.exists()
            else pinned.SHORT_ARMS)
        out = score.score(paths, arms)
        print(f"{len(out['runs'])} runs scored, {len(out['missing'])} missing"
              + (f" -> {files.scores}" if out["scores_written"] else " (m2_scores.jsonl not written)")
              + f"; summary {files.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
