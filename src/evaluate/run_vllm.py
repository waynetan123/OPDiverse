"""Run one evaluation request file through vLLM on a GPU machine. The evaluation runner for steps 7 and 12.

    pip install -r requirements-gpu.txt
    PYTHONPATH=src python -m evaluate.run_vllm --requests data/engine_check/vllm_a_requests.jsonl [--check-only]
    PYTHONPATH=src python -m evaluate.run_vllm --requests data/sweep/dev_requests.jsonl \\
        --model data/sweep/runs/sft_lr5e-05/step100/merged --out data/sweep/runs/sft_lr5e-05/step100

One request file per invocation, so every pass gets a fresh engine and no prefix cache carries over
from another file. Requests are generated in chunks; each finished chunk is appended to
<prefix>_generations.partial.jsonl, so an interrupted run resumes where it stopped. When the file is
complete its rows are written, in request order, to <prefix>_generations.jsonl with
<prefix>_run_meta.json, and the partial file is removed. These sit beside the request file, or in --out.

--model serves a merged checkpoint (pinned.EVAL_LORA: each LoRA adapter merged into the base weights) instead of
the backbone; the tokenizer, chat template and decoding stay pinned, and run_meta records the checkpoint's sha256.

Refuses to run on a dirty tree, under a vLLM other than pinned.VLLM_VERSION, or if any request departs
from the pinned backbone, pinned.EVAL_SAMPLING, Qwen's chat template or pinned.EVAL_MAX_MODEL_LEN. Every
prompt's token ids are checked against the pinned tokenizer, not only the first.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned
from etl.manifest import file_sha256, git_state, tree_sha256
from etl.paths import ROOT
from frozen_model.run_vllm import CHUNK, DETERMINISM_CHECK, _field_names, file_stats, load_requests, run_file, sampling_kwargs
from generators.bank.files import generations_for, partial_for, run_meta_for
from probe.run_vllm import check_rendering, llm_kwargs, pinned_tokenizer

JOB = "eval"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without vllm)
# ---------------------------------------------------------------------------


def request_id(item_id: str) -> str:
    return f"{JOB}:{item_id}"


def output_paths(requests: Path, out: Path | None = None) -> tuple[Path, Path, Path]:
    """(generations, partial, run_meta) for a request file: beside it, or in `out`."""
    paths = (generations_for(requests), partial_for(requests), run_meta_for(requests))
    return paths if out is None else tuple(out / p.name for p in paths)


def check_request_pins(requests: list[dict]) -> None:
    """Every request names the pinned backbone, exactly pinned.EVAL_SAMPLING, and fits EVAL_MAX_MODEL_LEN."""
    for r in requests:
        rid = r["request_id"]
        if (r["model"], r["revision"]) != (pinned.TOKENIZER_REPO, pinned.TOKENIZER_REVISION):
            raise SystemExit(f"{rid}: model {r['model']}@{r['revision']} is not the pinned backbone")
        if r["job"] != JOB or r["attempt"] != 1 or rid != request_id(r["item_id"]):
            raise SystemExit(f"{rid}: not an evaluation request for {r['item_id']}")
        if r["sampling"] != pinned.EVAL_SAMPLING:
            raise SystemExit(f"{rid}: sampling {r['sampling']} differs from pinned.EVAL_SAMPLING")
        if r["prompt_tokens"] + pinned.EVAL_MAX_TOKENS > pinned.EVAL_MAX_MODEL_LEN:
            raise SystemExit(f"{rid}: {r['prompt_tokens']} prompt tokens + {pinned.EVAL_MAX_TOKENS} exceed "
                             f"EVAL_MAX_MODEL_LEN {pinned.EVAL_MAX_MODEL_LEN}")


def check_prompt_ids(part: list[dict], outs, encode) -> None:
    """encode(prompt) -> token ids under the pinned tokenizer; vLLM must have tokenised every prompt the same way."""
    for r, out in zip(part, outs, strict=True):
        if list(out.prompt_token_ids) != encode(r["prompt"]):
            raise SystemExit(f"{r['request_id']}: vLLM tokenised the prompt differently from the pinned tokenizer")


def engine_settings(llm) -> dict:
    """What the engine actually ran with, read defensively (attribute names move between vLLM versions)."""
    out = {}
    config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    for section, field in (("cache_config", "enable_prefix_caching"), ("scheduler_config", "max_num_seqs"),
                           ("scheduler_config", "max_num_batched_tokens"), ("model_config", "dtype"),
                           ("model_config", "max_model_len")):
        value = getattr(getattr(config, section, None), field, None)
        out[f"{section}.{field}"] = value if isinstance(value, (bool, int, float, str, type(None))) else str(value)
    return out


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluate.run_vllm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help="a merged checkpoint directory (default: the backbone)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: beside the request file)")
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

    if vllm.__version__ != pinned.VLLM_VERSION:
        raise SystemExit(f"vLLM {vllm.__version__} is installed; pinned.VLLM_VERSION is {pinned.VLLM_VERSION}")
    out_path, partial_path, meta_path = output_paths(args.requests, args.out)
    if args.model is not None and not (args.model / "config.json").exists():
        raise SystemExit(f"{args.model} is not a model directory (no config.json)")
    if out_path.exists():
        print(f"{out_path} already exists: {args.requests.name} is complete")
        return 0
    requests = load_requests(args.requests)
    check_request_pins(requests)
    tokenizer, tok_sha, stop_ids = pinned_tokenizer()
    check_rendering(requests, lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    names = _field_names(vllm.SamplingParams)
    sampling_notes = sampling_kwargs({}, names)[1]
    print(f"checks passed: {len(requests):,} requests, tokenizer sha256 ok, chat template ok, stop ids {stop_ids}, "
          f"max_model_len {pinned.EVAL_MAX_MODEL_LEN:,}; {sampling_notes or 'no watermarking field'}")
    if args.check_only:
        return 0

    model_sha = None if args.model is None else tree_sha256(args.model)
    kwargs, notes = llm_kwargs(_field_names(vllm.EngineArgs), pinned.EVAL_MAX_MODEL_LEN,
                               None if args.model is None else str(args.model))
    for note in notes:
        warnings.warn(note)
    t0 = time.perf_counter()
    llm = vllm.LLM(**kwargs, gpu_memory_utilization=args.gpu_memory_utilization)
    load_s = time.perf_counter() - t0
    encode = lambda prompt: tokenizer(prompt, add_special_tokens=False)["input_ids"]  # noqa: E731
    checked = 0

    def generate(part: list[dict], count: bool = True):
        nonlocal checked
        params = [vllm.SamplingParams(**sampling_kwargs(r["sampling"], names)[0], stop_token_ids=stop_ids) for r in part]
        outs = llm.generate([r["prompt"] for r in part], params)
        check_prompt_ids(part, outs, encode)
        checked += len(part) if count else 0
        return outs

    started = datetime.now(timezone.utc)
    requests_sha = file_sha256(args.requests)
    t0 = time.perf_counter()
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
    done = run_file(generate, requests, partial_path, requests_sha, args.chunk, label=args.requests.stem)
    gen_s = time.perf_counter() - t0
    rows = [done[r["request_id"]] for r in requests]
    recheck_requests = requests[:DETERMINISM_CHECK]
    recheck = generate(recheck_requests, count=False)
    mismatched = [r["request_id"] for r, out in zip(recheck_requests, recheck)
                  if list(out.outputs[0].token_ids) != done[r["request_id"]]["outputs"][0]["token_ids"]]
    meta = {
        "last_session_started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests_sha256": requests_sha,
        "n_requests": len(requests),
        "git": git,
        "engine": "vllm",
        "model": pinned.TOKENIZER_REPO,
        "revision": pinned.TOKENIZER_REVISION,
        "tokenizer_sha256": tok_sha,
        **({} if args.model is None else {"checkpoint": str(args.model), "checkpoint_sha256": model_sha}),
        "llm_kwargs": kwargs,
        "engine_settings": engine_settings(llm),
        "notes": notes + sampling_notes,
        "sampling_params_example": repr(vllm.SamplingParams(**sampling_kwargs(requests[0]["sampling"], names)[0],
                                                            stop_token_ids=stop_ids)),
        "stop_token_ids": stop_ids,
        "prompt_token_ids_checked_this_session": checked,
        "resumed_from_earlier_sessions": len(requests) - checked,
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda, "python": platform.python_version()},
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "seconds": {"load": round(load_s, 1), "generate_this_session": round(gen_s, 1)},
        "determinism_recheck": {"n": len(recheck_requests), "mismatched": mismatched},
        **file_stats(rows),
    }
    # The meta first: the generations file marks the run complete, so it is written last.
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    partial_path.unlink()
    print(f"{args.requests.name}: wrote {len(rows):,} rows; finish reasons {meta['finish_reasons']}; "
          f"determinism recheck mismatches {len(mismatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
