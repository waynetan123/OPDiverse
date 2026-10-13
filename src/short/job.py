"""One M2 run of the one-day run, start to finish, on the one GPU it is given. train.launch starts it per job:

    PYTHONPATH=src python -m train.launch --plan data/short/queues.json --gpus 0 1 2 3 --retries 1 \\
        --runner short.job --out-dir data/short/runs

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m short.job --arm grpo --lr 5e-5 --config m2-cvss --seed 1 \\
        --out data/short/runs/grpo_m2-cvss_s1

Runs, each as its own process so each starts with the GPU's memory free:

    1. train.run          skipped once run_meta.json records all pinned.TRAIN_STEPS steps; run with
                          PYTORCH_CUDA_ALLOC_CONF = pinned.SHORT_CUDA_ALLOC_CONF
    2. optimizer.pt is deleted from every trainer/checkpoint-* once training has finished (the adapters stay)
    3. train.merge        the last checkpoint (trainer/checkpoint-<TRAIN_STEPS>) -> the merged weights
    4. evaluate.run_vllm  data/short/test_requests.jsonl on the merged weights -> final/test_generations.jsonl
    5. the merged weights are deleted, keeping their merge_meta.json in final/ once the generations exist

The merged weights go to final/merged, or with --scratch DIR to DIR/<run dir name>/merged: on gpu17, /dev/shm
(memory), so the four GPUs' merges (~15 GB each) take no disk. They are deleted whether the evaluation succeeds or
fails, and when the launcher stops the job, so none is left behind.

Every stage picks up where it stopped (training from its latest checkpoint, evaluation from its partial file, a
finished evaluation is not redone), so a failed job is simply started again; a merge whose evaluation failed is
redone. A failing stage ends the job with that stage's exit code. Arguments other than these go to train.run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from etl import pinned
from etl.paths import DEFAULT
from generators.bank.files import generations_for
from train.config import train_steps

from .files import ShortFiles

TRAIN, MERGE, EVALUATE = "train.run", "train.merge", "evaluate.run_vllm"   # tests substitute stand-ins


def trained(out: Path, steps: int) -> bool:
    """The run finished training: its run_meta.json records every step, outside a pilot."""
    meta = out / "run_meta.json"
    if not meta.exists():
        return False
    m = json.loads(meta.read_text(encoding="utf-8"))
    return not m.get("pilot") and m.get("steps_done") == steps


def drop_optimizer_states(out: Path) -> list[Path]:
    """Delete optimizer.pt from every checkpoint of a finished run (owner, one-day run: gpu17's disk). The merge reads
    only the adapter, so every snapshot can still be evaluated; none can be trained further."""
    gone = sorted((out / "trainer").glob("checkpoint-*/optimizer.pt"))
    for p in gone:
        p.unlink()
    return gone


def stage(name: str, module: str, *args: str, env: dict | None = None) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"== {name}: {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, env=None if env is None else {**os.environ, **env}).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m short.job", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--requests", type=Path, default=ShortFiles.of(DEFAULT).test_requests)
    ap.add_argument("--scratch", type=Path, default=None,
                    help="where the merged weights go while evaluated, as <scratch>/<run dir name>/merged "
                         "(gpu17: /dev/shm/opdiverse); default <out>/final/merged")
    args, extra = ap.parse_known_args(argv)
    if "--pilot" in extra:
        raise SystemExit("short.job runs the day's runs; pilots go straight to train.run (train.launch --queue)")
    steps = train_steps()
    final = args.out / "final"
    merged = (args.scratch / args.out.name if args.scratch else final) / "merged"
    generations = final / generations_for(args.requests).name

    if trained(args.out, steps):
        print(f"== train: {args.out} already has all {steps} steps", flush=True)
    elif code := stage("train", TRAIN, "--arm", args.arm, "--lr", args.lr, "--config", args.config,
                       "--seed", args.seed, "--out", str(args.out), *extra,
                       env={"PYTORCH_CUDA_ALLOC_CONF": pinned.SHORT_CUDA_ALLOC_CONF}):
        return code
    if not trained(args.out, steps):
        raise SystemExit(f"training ended without run_meta.json recording all {steps} steps")
    if dropped := drop_optimizer_states(args.out):
        print(f"== cleanup: deleted optimizer.pt from {len(dropped)} finished checkpoints", flush=True)
    final.mkdir(parents=True, exist_ok=True)
    try:
        if not generations.exists():
            if code := stage("merge", MERGE, "--adapter", str(args.out / "trainer" / f"checkpoint-{steps}"),
                             "--out", str(merged)):
                return code
            if code := stage("eval", EVALUATE, "--requests", str(args.requests), "--model", str(merged),
                             "--out", str(final)):
                return code
        if not generations.exists():
            raise SystemExit(f"the evaluation ended without writing {generations}")
        if merged.exists():
            shutil.copy(merged / "merge_meta.json", final / "merge_meta.json")   # scoring checks the chain to the adapter
    finally:
        if merged.exists():   # after success or failure: a merge is redone rather than left on disk or in memory
            shutil.rmtree(merged)
            print(f"== cleanup: deleted {merged}", flush=True)
    print(f"== done: {generations}", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # train.launch stops jobs with SIGTERM: clean up first
    raise SystemExit(main())
