"""Start training runs on several GPUs at once, one queue per GPU, with one live progress line per GPU.

    PYTHONPATH=src python -m train.launch \\
        --queue 1=grpo@5e-5 --queue 2=dpo@5e-5,distill_self@5e-5 --queue 3=base@5e-5,sft@5e-5 \\
        --out-root data/sweep/pilot/cost -- --pilot --max-steps 20 --gradient-checkpointing on

Each --queue is GPU=arm@lr[,arm@lr...]: that GPU runs those jobs one after another, so no GPU ever holds two. Everything
after `--` goes to every `python -m train.run`; --queue-args GPU="..." adds arguments for that GPU's jobs only, so one
launch can compare settings side by side (the same arm and LR may then appear on two GPUs, and the extra arguments
are added to its directory name):

    PYTHONPATH=src python -m train.launch --queue 2=grpo@5e-5 --queue 3=grpo@5e-5 \
        --queue-args 2="--micro-batch 1" --queue-args 3="--micro-batch 4" \
        --out-root data/sweep/pilot/t -- --pilot --max-steps 10 --gradient-checkpointing on

With --out-root, each run writes to <out-root>_<arm>_lr<lr>[_<extra args>]/; without it, train.run's default directory.
Each run's output goes to logs/<run dir name>.log. The top line tracks the whole launch: runs finished and failed, and
the training steps done across every queued run.

Before starting, every requested GPU must be idle (no process on it, by nvidia-smi); the launcher refuses otherwise.
While running, each GPU's line shows its current job: steps done (from the run's train_log.jsonl), time per step, time
left, and how long since the run last wrote anything, so a hang shows. A run that exits with an error is marked FAILED
with its error line, and the GPU moves on to its next job. At the end a summary lists every run, and the launcher exits
1 if any failed.

A job may also name an M2 configuration and seed, arm@lr@config@seed; its run is then named <arm>_<config>_s<seed>
(short.files.run_name) and train.run gets --config and --seed. The one-day run (docs/OPDiverse_short.md) uses a
shared queue instead of one per GPU:

    PYTHONPATH=src python -m train.launch --plan data/short/queues.json --gpus 0 1 2 3 --retries 1 \
        --runner short.job --out-dir data/short/runs

--plan reads the ordered "jobs" of a queues.json (python -m short queue) into one list for the --gpus, and whichever
GPU is free takes the next job. --retries N puts a failed job back at the end of its queue up to N times; it resumes
where it stopped, and each attempt keeps its own log (<name>.log, <name>.retry1.log, ...). --runner sets the module
each job runs (default train.run); --out-dir writes each run to <out-dir>/<run name>/.

Run it inside tmux, so the runs survive a disconnect; Ctrl+C stops every run it started.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from etl import pinned
from generators.progress import clock

from converters import masks
from short.files import run_name as m2_run_name

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
    extra: tuple[str, ...] = ()
    config: str | None = None
    seed: int | None = None

    @property
    def name(self) -> str:
        """The run's name: run_name (or, with a configuration, short.files.run_name), plus the GPU's extra arguments
        if it has any ('grpo_lr5e-05_micro-batch-4', 'grpo_m2-cvss_s1')."""
        suffix = "-".join(re.sub(r"[^A-Za-z0-9.]+", "-", a).strip("-") for a in self.extra)
        base = run_name(self.arm, self.lr) if self.config is None else m2_run_name(self.arm, self.config, self.seed)
        return base + (f"_{suffix}" if suffix else "")

    @property
    def args(self) -> list[str]:
        """The runner's arguments that say which run this is."""
        return ["--arm", self.arm, "--lr", repr(self.lr)] + \
            ([] if self.config is None else ["--config", self.config, "--seed", str(self.seed)])


def parse_job(item: str, gpu: str = "", extra: tuple[str, ...] = ()) -> Job:
    """'dpo@5e-5' or 'grpo@5e-5@m2-cvss@1' -> a Job."""
    parts = item.strip().split("@")
    if len(parts) not in (2, 4) or parts[0] not in pinned.SWEEP_ARMS:
        raise ValueError(f"{item!r} is not arm@lr or arm@lr@config@seed with an arm from {pinned.SWEEP_ARMS}")
    try:
        lr = float(parts[1])
    except ValueError:
        raise ValueError(f"{item!r}: {parts[1]!r} is not a learning rate") from None
    if len(parts) == 2:
        return Job(gpu, parts[0], lr, extra)
    if parts[2] not in masks.CONFIGS or not parts[3].isdigit():
        raise ValueError(f"{item!r}: expected a configuration from {masks.CONFIGS} and an integer seed")
    return Job(gpu, parts[0], lr, extra, parts[2], int(parts[3]))


def parse_queue_args(specs: list[str]) -> dict[str, tuple[str, ...]]:
    """['3=--micro-batch 4'] -> {'3': ('--micro-batch', '4')}."""
    out = {}
    for spec in specs:
        gpu, sep, rest = spec.partition("=")
        if not sep or not gpu.strip().isdigit():
            raise ValueError(f"--queue-args {spec!r}: expected GPU=\"arguments\"")
        if "--out" in shlex.split(rest):
            raise ValueError("give --out-root to the launcher, not --out: each run needs its own directory")
        out[gpu.strip()] = tuple(shlex.split(rest))
    return out


def parse_queue(spec: str, extra: dict[str, tuple[str, ...]] | None = None) -> list[Job]:
    """'2=dpo@5e-5,distill_self@5e-5' -> that GPU's jobs, in order, each carrying that GPU's extra arguments."""
    gpu, sep, rest = spec.partition("=")
    if not sep or not gpu.strip().isdigit() or not rest:
        raise ValueError(f"--queue {spec!r}: expected GPU=arm@lr[,arm@lr...]")
    try:
        return [parse_job(item, gpu.strip(), (extra or {}).get(gpu.strip(), ())) for item in rest.split(",")]
    except ValueError as e:
        raise ValueError(f"--queue {spec!r}: {e}") from None


def parse_plan(plan: dict) -> list[Job]:
    """A queues.json's ordered "jobs" -> one shared list (the GPU is set when a job starts)."""
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError('the plan has no "jobs" list')
    return [parse_job(item) for item in jobs]


def check_queues(queues: list[list[Job]]) -> None:
    gpus = [q[0].gpu for q in queues]
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"a GPU appears in two queues ({gpus}); give each GPU one queue")
    check_names([j for q in queues for j in q])


def check_names(jobs: list[Job]) -> None:
    runs = [j.name for j in jobs]
    if len(set(runs)) != len(runs):
        raise ValueError("the same arm and LR is queued twice with the same arguments; two runs would share one "
                         "output directory")


def total_steps(passthrough: list[str]) -> int | None:
    """--max-steps from train.run's arguments (the last one given, as argparse reads it), else pinned.TRAIN_STEPS."""
    found = None
    for i, a in enumerate(passthrough):
        if a == "--max-steps" and i + 1 < len(passthrough):
            found = int(passthrough[i + 1])
        elif a.startswith("--max-steps="):
            found = int(a.split("=", 1)[1])
    return pinned.TRAIN_STEPS if found is None else found


def overall(finished: int, failed: int, runs: int, steps: int, all_steps: int | None, elapsed: float) -> str:
    """The launch's top line: runs finished (and failed) of all queued, and training steps done across all of them."""
    head = f"ALL   {finished}/{runs} runs finished" + (f", {failed} FAILED" if failed else "")
    progress = f"{bar(steps, all_steps)} {steps}/{all_steps} steps" if all_steps else ""
    return f"{head:<37} {progress}  elapsed {clock(elapsed)}"


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


def log_name(out: Path, attempt: int) -> str:
    """The log of one attempt at a run: <run dir name>.log, then <run dir name>.retry<k>.log."""
    return f"{out.name}.log" if attempt == 0 else f"{out.name}.retry{attempt}.log"


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
    attempt: int = 0
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
    ap.add_argument("--queue", action="append", default=[], help="GPU=arm@lr[,arm@lr...]; repeat per GPU")
    ap.add_argument("--queue-args", action="append", default=[], help='GPU="extra train.run arguments for that GPU"')
    ap.add_argument("--plan", type=Path, default=None, help="a queues.json: one shared queue for the --gpus")
    ap.add_argument("--gpus", nargs="+", default=None, help="with --plan: the GPUs that share its queue")
    ap.add_argument("--retries", type=int, default=0, help="times a failed job goes back to the end of its queue")
    ap.add_argument("--runner", default=None, help=f"the module each job runs (default {RUNNER})")
    ap.add_argument("--out-root", default=None, help="each run writes to <out-root>_<run name>/")
    ap.add_argument("--out-dir", type=Path, default=None, help="each run writes to <out-dir>/<run name>/")
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--allow-busy", action="store_true", help="start even if a requested GPU has a process on it")
    args = ap.parse_args(own)
    if "--out" in passthrough:
        raise SystemExit("give --out-root to the launcher, not --out to train.run: each run needs its own directory")
    if bool(args.queue) == bool(args.plan) or bool(args.plan) != bool(args.gpus):
        raise SystemExit("give either --queue (one per GPU) or --plan with --gpus")
    if args.out_root and args.out_dir:
        raise SystemExit("give --out-root or --out-dir, not both")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    runner = args.runner or RUNNER
    shared = args.plan is not None
    try:
        if shared:
            if args.queue_args:
                raise ValueError("--queue-args is for --queue; a plan's jobs share their arguments")
            if len(set(args.gpus)) != len(args.gpus) or not all(g.isdigit() for g in args.gpus):
                raise ValueError(f"--gpus {args.gpus}: give distinct GPU indices")
            jobs = parse_plan(json.loads(args.plan.read_text(encoding="utf-8")))
            check_names(jobs)
            queues = [Queue(g, jobs) for g in args.gpus]          # one list: a free GPU takes the next job
            queued = list(jobs)
        else:
            extra = parse_queue_args(args.queue_args)
            queues = [Queue(jobs[0].gpu, jobs) for jobs in (parse_queue(q, extra) for q in args.queue)]
            unknown = set(extra) - {q.gpu for q in queues}
            if unknown:
                raise ValueError(f"--queue-args for GPU(s) {sorted(unknown)} that have no --queue")
            check_queues([q.jobs for q in queues])
            queued = [j for q in queues for j in q.jobs]
    except (ValueError, OSError) as e:
        raise SystemExit(str(e))
    totals = {j.name: total_steps([*passthrough, *j.extra]) for j in queued}
    all_steps = None if None in totals.values() else sum(totals.values())
    attempts = {name: 0 for name in totals}
    retried: list[Result] = []
    launched = time.monotonic()
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
        job = dataclasses.replace(q.jobs.pop(0), gpu=q.gpu)
        name = job.name
        if args.out_root:
            out = Path(f"{args.out_root}_{name}")
        elif args.out_dir:
            out = args.out_dir / name
        else:
            out = Path("data/sweep") / ("pilot" if "--pilot" in passthrough else "runs") / name
        log = args.logs / log_name(out, attempts[name])
        cmd = [sys.executable, "-m", runner, *job.args, "--out", str(out), *passthrough, *job.extra]
        f = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**env_base, "CUDA_VISIBLE_DEVICES": job.gpu},
                                start_new_session=True)
        f.close()
        q.current = Running(job, name, out, log, proc, time.monotonic(), attempts[name])

    def finish(q: Queue, code: int) -> None:
        r = q.current
        text = read(r.log)
        if code == 0:
            meta = r.out / "run_meta.json"
            detail = ""
            if meta.exists():
                m = json.loads(read(meta) or "{}")
                detail = f"{m.get('steps_done')} steps, {m.get('gpu_hours')} GPU-h, peak {m.get('peak_memory_gb')} GB"
            if r.attempt:
                detail += f" (attempt {r.attempt + 1})"
            q.results.append(Result(r.job, r.name, "done", detail, r.log, r.out))
        elif attempts[r.name] < args.retries:
            attempts[r.name] += 1
            retried.append(Result(r.job, r.name, "RETRY", f"{exit_reason(code)}: {error_line(text)}", r.log, r.out))
            q.jobs.append(r.job)
        else:
            q.results.append(Result(r.job, r.name, "FAILED", f"{exit_reason(code)}: {error_line(text)}", r.log, r.out))
        q.current = None

    def lines() -> list[str]:
        out = []
        now = time.monotonic()
        steps_all = 0
        for q in queues:
            r = q.current
            steps_all += sum(totals[x.job.name] or 0 for x in q.results if x.status == "done")
            if r is None:
                out.append(render(q.gpu, "", "finished", 0, None, None, None,
                                  f"{sum(x.status == 'done' for x in q.results)} done, "
                                  f"{sum(x.status == 'FAILED' for x in q.results)} failed"))
                continue
            total = totals[r.job.name]
            done = steps_done(read(r.out / "train_log.jsonl").splitlines())
            steps_all += done
            if done and r.first is None:
                r.first = (now, done)
            per_step = (now - r.first[0]) / (done - r.first[1]) if r.first and done > r.first[1] else None
            mtimes = [p.stat().st_mtime for p in (r.log, r.out / "train_log.jsonl") if p.exists()]
            idle = time.time() - max(mtimes) if mtimes else None
            retry = f" retry {r.attempt}" if r.attempt else ""
            label = f"{r.name}{retry}" if shared else \
                f"{r.name}{retry} ({len(q.results) + 1}/{len(q.results) + 1 + len(q.jobs)})"
            out.append(render(r.job.gpu, label, "running", done, total, per_step, idle, last_line(read(r.log))[:80],
                              elapsed=now - r.started))
        failed = [x for q in queues for x in q.results if x.status == "FAILED"]
        finished = sum(len(q.results) for q in queues)
        head = overall(finished, len(failed), len(totals), steps_all, all_steps, now - launched)
        out.insert(0, head + (f"  ({len(queues[0].jobs)} waiting)" if shared else ""))
        out += [f"  RETRY  {x.name} after failing on GPU {x.job.gpu}: {x.detail[:110]}" for x in retried]
        out += [f"  FAILED {x.name} on GPU {x.job.gpu}: {x.detail[:120]}" for x in failed]
        return out

    shown, last_print = 0, 0.0
    try:
        while any(q.current or q.jobs for q in queues):
            for q in queues:
                if q.current is not None and (code := q.current.proc.poll()) is not None:
                    finish(q, code)
                if q.current is None and q.jobs:
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
        print(f"  {x.status:<6} GPU {x.job.gpu}  {x.name:<30} {x.detail}\n         log {x.log}, output {x.out}")
    for x in retried:
        print(f"  retried after a failure on GPU {x.job.gpu}: {x.name}: {x.detail}\n         log {x.log}")
    return 1 if any(x.status == "FAILED" for x in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
