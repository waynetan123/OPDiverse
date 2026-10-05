"""The HuggingFace reference for the engine-agreement check, on a GPU machine. The only module that generates with transformers.

    pip install -r requirements-gpu.txt
    # after vLLM pass A has finished (vllm_a_generations.jsonl exists), one process per GPU:
    CUDA_VISIBLE_DEVICES=k PYTHONPATH=src python -m engine_check.run_hf --shard k --shards 4 [--check-only]

Reads data/engine_check/vllm_a_requests.jsonl (shard k takes every n-th request) and writes
hf_shard{k}of{n}_generations.jsonl in the vLLM runner's row schema, plus hf_shard{k}of{n}_run_meta.json.
Each finished request is appended to a .partial file first, so an interrupted run resumes.

The reference (pinned in docs/decisions/step7_decision_record.md): the pinned backbone in EVAL_DTYPE,
SDPA attention, batch size 1 (no padding), greedy, 512 new tokens, stopping on either EVAL_STOP_TOKENS.
Qwen's generation_config is replaced by one built from scratch, so its repetition_penalty 1.05 (which
would apply even under greedy) and its sampling defaults never take effect, and every generated token is
asserted to be the argmax of the raw logits, which no hidden logits processor could survive.

Where HF's tokens leave vLLM pass A's, the row records the first divergence: its index, both tokens, and
HF's logit margin between them. HF's logits at that step are exactly the teacher-forced logits for vLLM's
prefix, since the two sequences agree up to it. A small margin is a near-tie flip; a large one is not.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned
from etl.build import read_jsonl
from etl.manifest import file_sha256, git_state
from etl.paths import DEFAULT, ROOT, Paths
from evaluate.run_vllm import check_request_pins
from frozen_model.run_vllm import load_requests
from generators.bank.files import generations_for
from generators.progress import Bar
from probe.run_vllm import check_rendering, pinned_tokenizer

from .files import EngineFiles

STOP = -1  # one symbol for "stopped", whichever stop token ended the sequence
PAD_TOKEN_ID = 151643  # <|endoftext|>; only used to silence generate(), batch size 1 never pads


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without torch)
# ---------------------------------------------------------------------------


def shard_of(requests: list[dict], shard: int, shards: int) -> list[dict]:
    if not 0 <= shard < shards:
        raise SystemExit(f"--shard must be in [0, {shards})")
    return requests[shard::shards]


def finish_reason(token_ids: list[int], stop_ids: set[int]) -> str:
    """As vLLM reports it: 'stop' if the sequence ended on a stop token, else 'length'."""
    return "stop" if token_ids and token_ids[-1] in stop_ids else "length"


def body(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    """The generated tokens without a trailing stop token (vLLM may or may not include it)."""
    return list(token_ids[:-1]) if token_ids and token_ids[-1] in stop_ids else list(token_ids)


def normalised(output: dict, stop_ids: set[int]) -> list[int]:
    """An output row's tokens with any stop token as STOP, so both engines compare symbol for symbol."""
    return body(output["token_ids"], stop_ids) + ([STOP] if output["finish_reason"] == "stop" else [])


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """The first index where two token sequences differ (a prefix differs at the shorter one's end)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def hf_row(request: dict, token_ids: list[int], text: str, stop_ids: set[int], requests_sha: str) -> dict:
    """The vLLM runner's row schema (frozen_model.run_vllm.output_row), one output."""
    return {
        "request_id": request["request_id"],
        "item_id": request["item_id"],
        "job": request["job"],
        "attempt": request["attempt"],
        "requests_sha256": requests_sha,
        "prompt_sha256": hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest(),
        "n_prompt_tokens": request["prompt_tokens"],
        "outputs": [{"text": text, "finish_reason": finish_reason(token_ids, stop_ids), "token_ids": list(token_ids)}],
    }


def load_partial(path: Path, requests_sha: str) -> dict[str, dict]:
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


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def generation_config(transformers, stop_ids: list[int]):
    """Built from scratch: nothing from Qwen's generation_config.json can reach generate()."""
    return transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=pinned.EVAL_MAX_TOKENS, repetition_penalty=1.0,
        temperature=None, top_p=None, top_k=None, eos_token_id=list(stop_ids), pad_token_id=PAD_TOKEN_ID)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m engine_check.run_hf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--check-only", action="store_true", help="verify pins, tokenizer, template and generate() options, then stop")
    args = ap.parse_args(argv)
    files = EngineFiles.of(Paths(args.data_dir.resolve()))

    git = git_state(ROOT)
    if git["commit"] is None or git["dirty"]:
        raise SystemExit(f"commit the code and pins before the GPU run (git state: {git})")

    import torch
    import transformers

    requests_path = files.vllm_requests("a")
    out_path = files.hf_generations(args.shard, args.shards)
    if out_path.exists():
        print(f"{out_path} already exists: shard {args.shard} of {args.shards} is complete")
        return 0
    requests = load_requests(requests_path)
    check_request_pins(requests)
    vllm_path = generations_for(requests_path)
    if not vllm_path.exists():
        raise SystemExit(f"{vllm_path} is missing: run vLLM pass A first (`python -m evaluate.run_vllm --requests {requests_path}`)")
    vllm_rows = {r["request_id"]: r for r in read_jsonl(vllm_path)}
    requests_sha = file_sha256(requests_path)
    if set(vllm_rows) != {r["request_id"] for r in requests} or any(r["requests_sha256"] != requests_sha for r in vllm_rows.values()):
        raise SystemExit(f"{vllm_path.name} does not match {requests_path.name}")
    tokenizer, tok_sha, stop_list = pinned_tokenizer()
    stop_ids = set(stop_list)
    check_rendering(requests, lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    gen_config = generation_config(transformers, stop_list)
    generate_options = {"return_dict_in_generate": True, "output_logits": True}
    if "use_model_defaults" in inspect.signature(transformers.GenerationMixin.generate).parameters:
        generate_options["use_model_defaults"] = False
    mine = shard_of(requests, args.shard, args.shards)
    print(f"checks passed: {len(requests):,} requests, shard {args.shard}/{args.shards} has {len(mine):,}; tokenizer sha256 ok, "
          f"chat template ok, stop ids {stop_list}; generation config {gen_config.to_diff_dict()}; options {generate_options}")
    if args.check_only:
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    t0 = time.perf_counter()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pinned.TOKENIZER_REPO, revision=pinned.TOKENIZER_REVISION, dtype=getattr(torch, pinned.EVAL_DTYPE),
        attn_implementation=pinned.HF_ATTENTION).to("cuda").eval()
    model.generation_config = gen_config
    load_s = time.perf_counter() - t0

    def generate(request: dict):
        """(generated token ids including any stop token, per-step raw logits)."""
        ids = tokenizer(request["prompt"], add_special_tokens=False)["input_ids"]
        if len(ids) != request["prompt_tokens"] or len(ids) != vllm_rows[request["request_id"]]["n_prompt_tokens"]:
            raise SystemExit(f"{request['request_id']}: HF tokenised the prompt to {len(ids)} tokens; the bank and vLLM say "
                             f"{request['prompt_tokens']} and {vllm_rows[request['request_id']]['n_prompt_tokens']}")
        input_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                 generation_config=gen_config, **generate_options)
        gen = out.sequences[0, len(ids):]
        logits = torch.cat(out.logits, dim=0)  # (steps, vocab), raw: before any logits processor
        if logits.shape[0] != gen.shape[0] or not torch.equal(logits.argmax(dim=-1), gen):
            raise SystemExit(f"{request['request_id']}: a generated token is not the argmax of the raw logits; "
                             "something other than greedy decoding is acting on generate()")
        return gen.tolist(), logits

    def divergence(request: dict, token_ids: list[int], logits) -> dict | None:
        v_out = vllm_rows[request["request_id"]]["outputs"][0]
        hf_seq = body(token_ids, stop_ids) + ([STOP] if finish_reason(token_ids, stop_ids) == "stop" else [])
        k = first_divergence(hf_seq, normalised(v_out, stop_ids))
        if k is None:
            return None
        step = logits[k].float()
        hf_tok = token_ids[k]
        v_seq = normalised(v_out, stop_ids)
        if v_seq[k] == STOP:
            last = v_out["token_ids"][-1] if v_out["token_ids"] else None
            v_tok = last if last in stop_ids else max(stop_list, key=lambda i: float(step[i]))
        else:
            v_tok = v_seq[k]
        return {"index": k, "hf_token": hf_tok, "vllm_token": v_tok,
                "margin": float(step[hf_tok] - step[v_tok]),
                "vllm_token_rank": int((step > step[v_tok]).sum())}  # 1 = HF's second choice

    partial = files.hf_partial(args.shard, args.shards)
    done = load_partial(partial, requests_sha)
    todo = [r for r in mine if r["request_id"] not in done]
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    with open(partial, "a", encoding="utf-8") as f, Bar(f"HF shard {args.shard}", len(mine)) as bar:
        bar.update(len(done), note=f"resuming: {len(done):,} already done" if done else "")
        for r in todo:
            token_ids, logits = generate(r)
            row = hf_row(r, token_ids, tokenizer.decode(body(token_ids, stop_ids), skip_special_tokens=True), stop_ids, requests_sha)
            row["divergence_from_vllm_a"] = divergence(r, token_ids, logits)
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            done[r["request_id"]] = row
            bar.update(len(done))
    gen_s = time.perf_counter() - t0

    recheck = mine[: pinned.HF_DETERMINISM_CHECK]
    mismatched = [r["request_id"] for r in recheck if generate(r)[0] != done[r["request_id"]]["outputs"][0]["token_ids"]]
    rows = [done[r["request_id"]] for r in mine]
    meta = {
        "last_session_started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests_sha256": requests_sha,
        "vllm_a_generations_sha256": file_sha256(vllm_path),
        "shard": args.shard, "shards": args.shards, "n_requests": len(mine),
        "git": git,
        "engine": "hf",
        "model": pinned.TOKENIZER_REPO,
        "revision": pinned.TOKENIZER_REVISION,
        "tokenizer_sha256": tok_sha,
        "dtype": pinned.EVAL_DTYPE,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "sdpa_backends_enabled": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                                  "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                  "math": torch.backends.cuda.math_sdp_enabled()},
        "generation_config": json.loads(gen_config.to_json_string()),
        "generate_options": generate_options,
        "batch_size": 1,
        "argmax_checked": True,
        "generated_this_session": len(todo),
        "stop_token_ids": stop_list,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0),
        "seconds": {"load": round(load_s, 1), "generate_this_session": round(gen_s, 1)},
        "determinism_recheck": {"n": len(recheck), "mismatched": mismatched},
    }
    # The meta first: the generations file marks the shard complete, so it is written last.
    files.hf_run_meta(args.shard, args.shards).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    partial.unlink()
    print(f"shard {args.shard}/{args.shards}: wrote {len(rows):,} rows; determinism recheck mismatches {len(mismatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
