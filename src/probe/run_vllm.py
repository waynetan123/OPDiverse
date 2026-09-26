"""Run the probe's requests through vLLM on a GPU machine. The only module that imports vllm.

    pip install -r requirements-gpu.txt
    PYTHONPATH=src python -m probe.run_vllm --requests data/probe/requests.jsonl --out data/probe

Writes <out>/generations.jsonl and <out>/run_meta.json. Refuses to run if the tokenizer, the
chat template or the sampling settings differ from the pins.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned

DETERMINISM_CHECK = 20


# ---------------------------------------------------------------------------
# Pure checks (unit-tested without vllm)
# ---------------------------------------------------------------------------


def load_requests(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    ids = [r["request_id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate request_id in the request file")
    return requests


def check_request_pins(requests: list[dict]) -> None:
    """Every request must name the pinned model, revision and sampling settings."""
    for r in requests:
        if (r["model"], r["revision"]) != (pinned.TOKENIZER_REPO, pinned.TOKENIZER_REVISION):
            raise SystemExit(f"{r['request_id']}: model {r['model']}@{r['revision']} is not the pinned backbone")
        if r["sampling"] != pinned.EVAL_SAMPLING:
            raise SystemExit(f"{r['request_id']}: sampling {r['sampling']} differs from pinned.EVAL_SAMPLING")


def check_rendering(requests: list[dict], apply_chat_template) -> None:
    """apply_chat_template(messages) -> str must reproduce each stored prompt byte for byte."""
    for r in requests:
        rendered = apply_chat_template(r["messages"])
        if rendered != r["prompt"]:
            raise SystemExit(f"{r['request_id']}: the model's chat template renders a different prompt:\n"
                             f"{rendered!r}\n!=\n{r['prompt']!r}")


def llm_kwargs(engine_arg_names: set[str]) -> tuple[dict, list[str]]:
    """Engine arguments; `generation_config="vllm"` stops vLLM applying Qwen's sampling defaults.
    Versions without that argument never applied model defaults, so explicit sampling suffices there."""
    kwargs = {
        "model": pinned.TOKENIZER_REPO,
        "revision": pinned.TOKENIZER_REVISION,
        "tokenizer_revision": pinned.TOKENIZER_REVISION,
        "dtype": pinned.EVAL_DTYPE,
        "seed": pinned.EVAL_SAMPLING["seed"],
        "max_model_len": pinned.EVAL_MAX_MODEL_LEN,
    }
    notes = []
    if "generation_config" in engine_arg_names:
        kwargs["generation_config"] = "vllm"
    else:
        notes.append("this vLLM has no generation_config argument; the explicit sampling parameters still apply")
    return kwargs, notes


def generation_rows(requests: list[dict], outputs) -> list[dict]:
    rows = []
    for r, out in zip(requests, outputs, strict=True):
        completion = out.outputs[0]
        rows.append({
            "request_id": r["request_id"],
            "cve_id": r["cve_id"],
            "question": r["question"],
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "n_prompt_tokens": len(out.prompt_token_ids),
            "n_output_tokens": len(completion.token_ids),
            "prompt_sha256": hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest(),
        })
    return rows


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m probe.run_vllm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--check-only", action="store_true", help="verify tokenizer and template, then stop")
    args = ap.parse_args(argv)

    import torch
    import transformers
    import vllm
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    requests = load_requests(args.requests)
    check_request_pins(requests)
    tok_path = hf_hub_download(pinned.TOKENIZER_REPO, "tokenizer.json", revision=pinned.TOKENIZER_REVISION)
    tok_sha = hashlib.sha256(Path(tok_path).read_bytes()).hexdigest()
    if tok_sha != pinned.TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json sha256 {tok_sha} != pinned {pinned.TOKENIZER_SHA256}")
    tokenizer = AutoTokenizer.from_pretrained(pinned.TOKENIZER_REPO, revision=pinned.TOKENIZER_REVISION)
    check_rendering(requests, lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in pinned.EVAL_STOP_TOKENS]
    if any(i is None or i == tokenizer.unk_token_id for i in stop_ids):
        raise SystemExit(f"stop tokens {pinned.EVAL_STOP_TOKENS} not all in the vocabulary: {stop_ids}")
    print(f"checks passed: {len(requests)} requests, tokenizer sha256 ok, chat template ok, stop ids {stop_ids}")
    if args.check_only:
        return 0

    kwargs, notes = llm_kwargs({f.name for f in dataclasses.fields(vllm.EngineArgs)})
    for note in notes:
        warnings.warn(note)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    llm = vllm.LLM(**kwargs, gpu_memory_utilization=args.gpu_memory_utilization)
    load_s = time.perf_counter() - t0
    params = vllm.SamplingParams(**pinned.EVAL_SAMPLING, stop_token_ids=stop_ids)

    t0 = time.perf_counter()
    outputs = llm.generate([r["prompt"] for r in requests], params)
    gen_s = time.perf_counter() - t0
    first_ids = list(outputs[0].prompt_token_ids)
    expected_ids = tokenizer(requests[0]["prompt"], add_special_tokens=False)["input_ids"]
    if first_ids != expected_ids:
        raise SystemExit("vLLM tokenised the first prompt differently from the pinned tokenizer")
    rows = generation_rows(requests, outputs)

    recheck = llm.generate([r["prompt"] for r in requests[:DETERMINISM_CHECK]], params)
    mismatched = [r["request_id"] for r, a, b in zip(requests, outputs, recheck) if a.outputs[0].text != b.outputs[0].text]

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "generations.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    meta = {
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "n_requests": len(requests),
        "model": pinned.TOKENIZER_REPO,
        "revision": pinned.TOKENIZER_REVISION,
        "tokenizer_sha256": tok_sha,
        "llm_kwargs": kwargs,
        "notes": notes,
        "sampling_params": repr(params),
        "stop_token_ids": stop_ids,
        "versions": {
            "vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(),
        },
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "seconds": {"load": round(load_s, 1), "generate": round(gen_s, 1)},
        "determinism_recheck": {"n": min(DETERMINISM_CHECK, len(requests)), "mismatched": mismatched},
        "finish_reasons": {k: sum(r["finish_reason"] == k for r in rows) for k in sorted({r["finish_reason"] for r in rows})},
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} generations in {gen_s:.1f}s; determinism recheck mismatches: {len(mismatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
