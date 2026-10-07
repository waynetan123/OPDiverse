"""Start training runs on several GPUs at once, one queue per GPU, with one live progress line per GPU.

    PYTHONPATH=src python -m train.launch \\
        --queue 1=grpo@5e-5 --queue 2=dpo@5e-5,distill_self@5e-5 --queue 3=base@5e-5,sft@5e-5 \\
        --out-root data/sweep/pilot/cost -- --pilot --max-steps 20 --gradient-checkpointing on

Each --queue is GPU=arm@lr[,arm@lr...]: that GPU runs those jobs one after another, so no GPU ever holds two. Everything
after `--` goes to every `python -m train.run`. With --out-root, each run writes to <out-root>_<arm>_lr<lr>/; without
it, train.run's default directory. Each run's output goes to logs/<run dir name>.log.

Before starting, every requested GPU must be idle (no process on it, by nvidia-smi); the launcher refuses otherwise.
While running, each GPU's line shows its current job: steps done (from the run's train_log.jsonl), time per step, time
left, and how long since the run last wrote anything, so a hang shows. A run that exits with an error is marked FAILED
with its error line, and the GPU moves on to its next job. At the end a summary lists every run, and the launcher exits
1 if any failed.

Run it inside tmux, so the runs survive a disconnect; Ctrl+C stops every run it started.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from etl import pinned
from generators.progress import clock

from .config import run_name

RUNNER = "train.run"      # the module each job runs (tests substitute a stand-in)
STALLED_AFTER = 20 * 60   # seconds without any output before a run is marked "no output for ..."
REFRESH = 2.0             # seconds between redraws
BAR = 24


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    gpu: str
    arm: str
    lr: float


def parse_queue(spec: str) -> list[Job]:
    """'2=dpo@5e-5,distill_self@5e-5' -> that GPU's jobs, in order."""
    gpu, sep, rest = spec.partition("=")
    if not sep or not gpu.strip().isdigit() or not rest:
        raise ValueError(f"--queue {spec!r}: expected GPU=arm@lr[,arm@lr...]")
    jobs = []
    for item in rest.split(","):
        arm, at, lr = item.strip().partition("@")
        if not at or arm not in pinned.SWEEP_ARMS:
            raise ValueError(f"--queue {spec!r}: {item!r} is not arm@lr with an arm from {pinned.SWEEP_ARMS}")
        jobs.append(Job(gpu.strip(), arm, float(lr)))
    return jobs


def check_queues(queues: list[list[Job]]) -> None:
    gpus = [q[0].gpu for q in queues]
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"a GPU appears in two queues ({gpus}); give each GPU one queue")
    runs = [(j.arm, j.lr) for q in queues for j in q]
    if len(set(runs)) != len(runs):
        raise ValueError("the same arm and LR is queued twice; two runs would share one output directory")


def total_steps(passthrough: list[str]) -> int | None:
    """--max-steps from train.run's arguments, else pinned.TRAIN_STEPS."""
    for i, a in enumerate(passthrough):
        if a == "--max-steps" and i + 1 < len(passthrough):
            return int(passthrough[i + 1])
        if a.startswith("--max-steps="):
            return int(a.split("=", 1)[1])
    return pinned.TRAIN_STEPS


def steps_done(log_lines: list[str]) -> int:
    """The highest optimizer step with a training-loss record in a train_log.jsonl."""
    best = 0
    for line in log_lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue   # a line being written right now
        t = r.get("trainer") if isinstance(r, dict) else None
        if isinstance(t, dict) and "loss" in t and isinstance(r.get("step"), int):
            best = max(best, r["step"])
    return best


NOISE = ("destroy_process_group", "ProcessGroupNCCL", "Backend impl 'liger_kernel.ops.backends._ascend")
ERROR = re.compile(r"(Error|Exception|SystemExit|Killed|Segmentation fault|out of memory)", re.IGNORECASE)


def last_line(text: str) -> str:
    """The last meaningful line of a run's output: progress bars are split on carriage returns, known noise skipped."""
    lines = [s.strip() for s in re.split(r"[\r\n]", text) if s.strip()]
    lines = [s for s in lines if not any(n in s for n in NOISE)]
    return lines[-1] if lines else ""


def error_line(text: str) -> str:
    """The line that says why a run failed: the last line naming an error, else the last meaningful line."""
    lines = [s.strip() for s in re.split(r"[\r\n]", text) if s.strip()]
    lines = [s for s in lines if not any(n in s for n in NOISE)]
    for s in reversed(lines):
        if ERROR.search(s):
            return re.sub(r"^\[rank\d+\]:\s*", "", s)
    return lines[-1] if lines else "(no output)"


def exit_reason(code: int) -> str:
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


def busy_gpus(query_gpus: str, query_apps: str) -> dict[str, list[str]]:
    """GPU index -> 'pid (MiB)' of each process on it, from nvidia-smi's CSV queries
    (--query-gpu=index,uuid and --query-compute-apps=gpu_uuid,pid,used_memory)."""
    index = {}
    for line in query_gpus.strip().splitlines():
        i, uuid = [x.strip() for x in line.split(",")[:2]]
        index[uuid] = i
    out: dict[str, list[str]] = {}
    for line in query_apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [x.strip() for x in line.split(",")[:3]]
        out.setdefault(index.get(uuid, uuid), []).append(f"pid {pid} ({mem})")
    return out


def bar(done: int, total: int | None, width: int = BAR) -> str:
    if not total:
        return "[" + "?" * width + "]"
    filled = min(width, width * done // total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(gpu: str, label: str, state: str, done: int, total: int | None, per_step: float | None,
           idle: float | None, note: str, width: int = 160, elapsed: float | None = None) -> str:
    """One GPU's status line."""
    head = f"GPU {gpu} {label:<30}"
    if state == "running" and done:
        left = f" ~{clock(per_step * (total - done))} left" if per_step and total else ""
        body = f"{bar(done, total)} {done}/{total or '?'}  {per_step:.0f}s/step{left}" if per_step else \
            f"{bar(done, total)} {done}/{total or '?'}"
    elif state == "running":
        body = f"starting ({clock(elapsed or 0)}): {note or 'launching'}"
    else:
        body = state if not note else f"{state}: {note}"
    if state == "running" and idle is not None and idle > STALLED_AFTER:
        body += f"  !! no output for {clock(idle)}"
    return (head + " " + body)[:width]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class Running:
    job: Job
    name: str
    out: Path
    log: Path
    proc: subprocess.Popen
    started: float
    first: tuple[float, int] | None = None   # (time, step) when training steps were first seen


@dataclass
class Result:
    job: Job
    name: str
    status: str
    detail: str
    log: Path
    out: Path


@dataclass
class Queue:
    gpu: str
    jobs: list[Job]
    current: Running | None = None
    results: list[Result] = field(default_factory=list)


def nvidia_smi_busy(gpus: list[str]) -> dict[str, list[str]] | None:
    try:
        q_gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout
        q_apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    busy = busy_gpus(q_gpus, q_apps)
    return {g: busy[g] for g in gpus if g in busy}


def read(path: Path, tail: int = 256 * 1024) -> str:
    """The last `tail` bytes of a file (whole lines only), or "" if it does not exist yet."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail))
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    return text.split("\n", 1)[1] if size > tail and "\n" in text else text


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="python -m train.launch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", action="append", required=True, help="GPU=arm@lr[,arm@lr...]; repeat per GPU")
    ap.add_argument("--out-root", default=None, help="each run writes to <out-root>_<arm>_lr<lr>/")
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--allow-busy", action="store_true", help="start even if a requested GPU has a process on it")
    args = ap.parse_args(own)
    if "--out" in passthrough:
        raise SystemExit("give --out-root to the launcher, not --out to train.run: each run needs its own directory")
    try:
        queues = [Queue(jobs[0].gpu, jobs) for jobs in map(parse_queue, args.queue)]
        check_queues([q.jobs for q in queues])
    except ValueError as e:
        raise SystemExit(str(e))
    total = total_steps(passthrough)
    gpus = [q.gpu for q in queues]
    busy = nvidia_smi_busy(gpus)
    if busy is None:
        print("warning: nvidia-smi is unavailable; cannot check that the GPUs are idle", file=sys.stderr)
    elif busy and not args.allow_busy:
        raise SystemExit("GPUs already in use: " + "; ".join(f"GPU {g}: {', '.join(p)}" for g, p in busy.items())
                         + ". Stop those processes (or pick other GPUs) first.")
    args.logs.mkdir(parents=True, exist_ok=True)
    env_base = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH") or "src"}
    tty = sys.stdout.isatty()

    def start(q: Queue) -> None:
        job = q.jobs.pop(0)
        name = run_name(job.arm, job.lr)
        if args.out_root:
            out = Path(f"{args.out_root}_{name}")
        else:
            out = Path("data/sweep") / ("pilot" if "--pilot" in passthrough else "runs") / name
        log = args.logs / f"{out.name}.log"
        cmd = [sys.executable, "-m", RUNNER, "--arm", job.arm, "--lr", repr(job.lr), "--out", str(out), *passthrough]
        f = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**env_base, "CUDA_VISIBLE_DEVICES": job.gpu},
                                start_new_session=True)
        f.close()
        q.current = Running(job, name, out, log, proc, time.monotonic())

    def finish(q: Queue, code: int) -> None:
        r = q.current
        text = read(r.log)
        if code == 0:
            meta = r.out / "run_meta.json"
            detail = ""
            if meta.exists():
                m = json.loads(read(meta) or "{}")
                detail = f"{m.get('steps_done')} steps, {m.get('gpu_hours')} GPU-h, peak {m.get('peak_memory_gb')} GB"
            q.results.append(Result(r.job, r.name, "done", detail, r.log, r.out))
        else:
            q.results.append(Result(r.job, r.name, "FAILED", f"{exit_reason(code)}: {error_line(text)}", r.log, r.out))
        q.current = None

    def lines() -> list[str]:
        out = []
        now = time.monotonic()
        for q in queues:
            r = q.current
            if r is None:
                out.append(render(q.gpu, "", "finished", 0, total, None, None,
                                  f"{sum(x.status == 'done' for x in q.results)} done, "
                                  f"{sum(x.status == 'FAILED' for x in q.results)} failed"))
                continue
            done = steps_done(read(r.out / "train_log.jsonl").splitlines())
            if done and r.first is None:
                r.first = (now, done)
            per_step = (now - r.first[0]) / (done - r.first[1]) if r.first and done > r.first[1] else None
            mtimes = [p.stat().st_mtime for p in (r.log, r.out / "train_log.jsonl") if p.exists()]
            idle = time.time() - max(mtimes) if mtimes else None
            label = f"{r.name} ({len(q.results) + 1}/{len(q.results) + 1 + len(q.jobs)})"
            out.append(render(r.job.gpu, label, "running", done, total, per_step, idle, last_line(read(r.log))[:80],
                              elapsed=now - r.started))
        failed = [x for q in queues for x in q.results if x.status == "FAILED"]
        out += [f"  FAILED {x.name} on GPU {x.job.gpu}: {x.detail[:120]}" for x in failed]
        return out

    for q in queues:
        start(q)
    shown, last_print = 0, 0.0
    try:
        while any(q.current for q in queues):
            for q in queues:
                if q.current is not None and (code := q.current.proc.poll()) is not None:
                    finish(q, code)
                    if q.jobs:
                        start(q)
            text = lines()
            if tty:
                sys.stdout.write((f"\x1b[{shown}F" if shown else "") + "".join(f"\x1b[2K{s}\n" for s in text))
                sys.stdout.flush()
                shown = len(text)
            elif time.monotonic() - last_print > 60:
                print("\n".join(text), flush=True)
                last_print = time.monotonic()
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        for q in queues:
            if q.current is not None:
                os.killpg(q.current.proc.pid, signal.SIGTERM)
        print("\nstopped every run this launcher started", file=sys.stderr)
        return 130
    results = [x for q in queues for x in q.results]
    print("\nSummary")
    for x in results:
        print(f"  {x.status:<6} GPU {x.job.gpu}  {x.name:<22} {x.detail}\n         log {x.log}, output {x.out}")
    return 1 if any(x.status == "FAILED" for x in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
