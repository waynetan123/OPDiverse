"""Run the frozen-model session's request files through vLLM on a GPU machine. The only step-5 module that imports vllm.

    pip install -r requirements-gpu.txt
    PYTHONPATH=src python -m frozen_model.run_vllm \\
        --requests data/frozen_model/audit_requests.jsonl \\
        --requests data/frozen_model/rationale_requests.jsonl [--check-only]

The engine loads once and serves every request file given. Requests are generated in chunks; each
finished chunk is appended to <prefix>_generations.partial.jsonl, so an interrupted run resumes
where it stopped. When a file is complete its rows are written, in request order, to
<prefix>_generations.jsonl with <prefix>_run_meta.json, and the partial file is removed.

Refuses to run on a dirty tree, under a vLLM other than pinned.VLLM_VERSION, or if any request
departs from the pinned backbone, its sampling config, or Qwen's chat template.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned
from etl.build import read_jsonl
from etl.manifest import file_sha256, git_state
from etl.paths import ROOT
from generators.bank.files import generations_for, partial_for, run_meta_for
from generators.progress import Bar
from probe.run_vllm import check_rendering, llm_kwargs, pinned_tokenizer

from .prepare import JOBS, max_model_len, request_id, sampling_for

CHUNK = 1_000
DETERMINISM_CHECK = 20


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without vllm)
# ---------------------------------------------------------------------------


def load_requests(path: Path) -> list[dict]:
    requests = read_jsonl(path)
    ids = [r["request_id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate request_id in {path}")
    return requests


def check_request_pins(requests: list[dict]) -> None:
    """Every request names the pinned backbone and exactly the sampling sampling_for gives it."""
    for r in requests:
        rid = r["request_id"]
        if (r["model"], r["revision"]) != (pinned.TOKENIZER_REPO, pinned.TOKENIZER_REVISION):
            raise SystemExit(f"{rid}: model {r['model']}@{r['revision']} is not the pinned backbone")
        if r["job"] not in JOBS or rid != request_id(r["job"], r["item_id"], r["attempt"]):
            raise SystemExit(f"{rid}: request_id does not match its job, item and attempt")
        if r["sampling"] != sampling_for(r["job"], r["item_id"], r["attempt"]):
            raise SystemExit(f"{rid}: sampling {r['sampling']} differs from the pinned config for {r['job']} attempt {r['attempt']}")


def sampling_kwargs(sampling: dict, field_names: set[str]) -> tuple[dict, list[str]]:
    """SamplingParams arguments. vLLM 0.30.0 reports `watermarking=True` by default (step 3); it is
    not pinned and may bias token choice, so it is switched off explicitly wherever it exists."""
    kwargs, notes = dict(sampling), []
    if "watermarking" in field_names:
        kwargs["watermarking"] = False
        notes.append("watermarking=False set explicitly")
    return kwargs, notes


def output_row(request: dict, out, requests_sha: str) -> dict:
    return {
        "request_id": request["request_id"],
        "item_id": request["item_id"],
        "job": request["job"],
        "attempt": request["attempt"],
        "requests_sha256": requests_sha,
        "prompt_sha256": hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest(),
        "n_prompt_tokens": len(out.prompt_token_ids),
        "outputs": [{"text": c.text, "finish_reason": c.finish_reason, "token_ids": list(c.token_ids)} for c in out.outputs],
    }


def load_partial(path: Path, requests_sha: str) -> dict[str, dict]:
    """request_id -> row from an earlier, interrupted run of the same request file."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a line cut short by a crash; that request is generated again
            if row.get("requests_sha256") != requests_sha:
                raise SystemExit(f"{path} came from a different requests file; move it aside before running")
            done[row["request_id"]] = row
    return done


def run_file(generate, requests: list[dict], partial: Path, requests_sha: str, chunk: int = CHUNK,
             label: str = "requests") -> dict[str, dict]:
    """generate(requests) -> one vLLM RequestOutput per request, in order. Appends each finished chunk
    to `partial` and returns every row by request_id, including those from earlier runs."""
    done = load_partial(partial, requests_sha)
    todo = [r for r in requests if r["request_id"] not in done]
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "a", encoding="utf-8") as f, Bar(label, len(requests)) as bar:
        bar.update(len(done), note=f"resuming: {len(done):,} already done" if done else "")
        for start in range(0, len(todo), chunk):
            part = todo[start:start + chunk]
            for r, out in zip(part, generate(part), strict=True):
                row = output_row(r, out, requests_sha)
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                done[r["request_id"]] = row
            f.flush()
            bar.update(len(done))
    return done


def file_stats(rows: list[dict]) -> dict:
    outputs = [o for r in rows for o in r["outputs"]]
    lengths = sorted(len(o["token_ids"]) for o in outputs)
    return {
        "finish_reasons": dict(sorted(Counter(o["finish_reason"] for o in outputs).items())),
        "n_outputs": len(outputs),
        "output_tokens": {"mean": round(sum(lengths) / len(lengths), 1) if lengths else None,
                          "max": lengths[-1] if lengths else None},
    }


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def _field_names(cls) -> set[str]:
    names = set(getattr(cls, "__struct_fields__", ()))
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    try:
        names |= set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
        pass
    return names


def _watermarking_source(module) -> list[str]:
    """Source lines mentioning watermarking, with context, for the decision record."""
    try:
        lines = inspect.getsource(module).splitlines()
    except (OSError, TypeError):
        return ["(source unavailable)"]
    hits = [i for i, line in enumerate(lines) if "watermark" in line.lower()]
    keep = sorted({j for i in hits for j in range(max(0, i - 3), min(len(lines), i + 4))})
    return [f"{j + 1}: {lines[j]}" for j in keep]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m frozen_model.run_vllm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=Path, action="append", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=CHUNK, help=f"requests per checkpoint (default {CHUNK})")
    ap.add_argument("--check-only", action="store_true", help="verify pins, tokenizer and template, then stop")
    args = ap.parse_args(argv)

    git = git_state(ROOT)
    if git["commit"] is None or git["dirty"]:
        raise SystemExit(f"commit the code and pins before the GPU run (git state: {git})")

    import torch
    import transformers
    import vllm
    import vllm.sampling_params

    if vllm.__version__ != pinned.VLLM_VERSION:
        raise SystemExit(f"vLLM {vllm.__version__} is installed; pinned.VLLM_VERSION is {pinned.VLLM_VERSION}")
    todo = []
    for path in args.requests:
        if generations_for(path).exists():
            print(f"{generations_for(path)} already exists: {path.name} is complete, skipped")
            continue
        requests = load_requests(path)
        check_request_pins(requests)
        todo.append((path, requests))
    if not todo:
        return 0
    tokenizer, tok_sha, stop_ids = pinned_tokenizer()
    for _, requests in todo:
        check_rendering(requests, lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    names = _field_names(vllm.SamplingParams)
    sample_kwargs, sampling_notes = sampling_kwargs({}, names)
    capacity = max_model_len([r for _, requests in todo for r in requests])
    print(f"checks passed: {sum(len(r) for _, r in todo):,} requests in {len(todo)} file(s), tokenizer sha256 ok, "
          f"chat template ok, stop ids {stop_ids}, max_model_len {capacity:,}; {sampling_notes or 'no watermarking field'}")
    if args.check_only:
        print("\n".join(["vLLM source mentioning watermarking:", *_watermarking_source(vllm.sampling_params)]))
        return 0

    kwargs, notes = llm_kwargs(_field_names(vllm.EngineArgs), capacity)
    for note in notes:
        warnings.warn(note)
    t0 = time.perf_counter()
    llm = vllm.LLM(**kwargs, gpu_memory_utilization=args.gpu_memory_utilization)
    load_s = time.perf_counter() - t0
    checked_tokens = False

    def generate(part: list[dict]):
        nonlocal checked_tokens
        params = [vllm.SamplingParams(**sampling_kwargs(r["sampling"], names)[0], stop_token_ids=stop_ids) for r in part]
        outs = llm.generate([r["prompt"] for r in part], params)
        if not checked_tokens:
            if list(outs[0].prompt_token_ids) != tokenizer(part[0]["prompt"], add_special_tokens=False)["input_ids"]:
                raise SystemExit("vLLM tokenised the first prompt differently from the pinned tokenizer")
            checked_tokens = True
        return outs

    for path, requests in todo:
        started = datetime.now(timezone.utc)
        requests_sha = file_sha256(path)
        t0 = time.perf_counter()
        done = run_file(generate, requests, partial_for(path), requests_sha, args.chunk, label=path.stem)
        gen_s = time.perf_counter() - t0
        rows = [done[r["request_id"]] for r in requests]
        greedy = [r for r in requests if r["sampling"]["temperature"] == 0][:DETERMINISM_CHECK]
        recheck = generate(greedy) if greedy else []
        mismatched = [r["request_id"] for r, out in zip(greedy, recheck) if out.outputs[0].text != done[r["request_id"]]["outputs"][0]["text"]]
        out = generations_for(path)
        out.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        meta = {
            "last_session_started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requests_sha256": requests_sha,
            "n_requests": len(requests),
            "git": git,
            "model": pinned.TOKENIZER_REPO,
            "revision": pinned.TOKENIZER_REVISION,
            "tokenizer_sha256": tok_sha,
            "llm_kwargs": kwargs,
            "notes": notes + sampling_notes,
            "sampling_params_example": repr(vllm.SamplingParams(**sampling_kwargs(requests[0]["sampling"], names)[0], stop_token_ids=stop_ids)),
            "stop_token_ids": stop_ids,
            "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                         "cuda": torch.version.cuda, "python": platform.python_version()},
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "seconds": {"load": round(load_s, 1), "generate_this_session": round(gen_s, 1)},
            "determinism_recheck": {"n": len(greedy), "mismatched": mismatched},
            **file_stats(rows),
        }
        run_meta_for(path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial_for(path).unlink()
        print(f"{path.name}: wrote {len(rows):,} rows; finish reasons {meta['finish_reasons']}; "
              f"determinism recheck mismatches {len(mismatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
