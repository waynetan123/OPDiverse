"""Train one LoRA run on one GPU (step 10 onwards).

    pip install -r requirements-gpu.txt
    PYTHONPATH=src python -m train.run --arm sft --lr 5e-5 [--check-only]
    PYTHONPATH=src python -m train.run --arm sft_dpo --lr 5e-5 --init data/sweep/runs/sft_lr5e-05/step200/merged
    PYTHONPATH=src python -m train.run --arm grpo --lr 5e-5 --pilot --max-steps 20 --gradient-checkpointing on
    PYTHONPATH=src python -m train.run --arm grpo --lr 5e-5 --vllm-server-port 8000   # server mode: `trl vllm-serve` running

Reads the frozen step-9 file for (arm, configuration, seed) through manifest.json, in the run order (train.data), and
trains pinned.TRAIN_STEPS optimizer steps of TRAIN_EXAMPLES_PER_STEP rows with the pinned LoRA and optimizer. Adapters
are saved at the CHECKPOINTS steps to <out>/trainer/checkpoint-<step>; an interrupted run resumes from the latest.
Writes <out>/train_log.jsonl as it goes and <out>/run_meta.json at the end.

Before any step it checks, and exits on any failure:
- the tree is clean, and the libraries and TRL defaults are the pinned ones (a --pilot run records them instead);
- every row's reference ids (train.data.encode_rows): prompts as the bank counted them, one end token per completion;
- the trainer's prepared ids equal the reference ids (no truncation, no second end token), and a collated row is
  unpadded, with labels exactly on the loss tokens, the end token included;
- every arm's shared configuration fields are identical.
--check-only stops there (GRPO: after one capped vLLM call).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from converters.files import read_rows
from etl import pinned
from etl.manifest import git_state
from etl.paths import DEFAULT, ROOT, Paths

from . import config as cfgmod
from . import data, grpo_logic

LIBS = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "liger_kernel", "vllm")
N_PARAMS_FLOPS = 6  # forward + backward FLOPs per parameter per trained token (the usual estimate)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a GPU)
# ---------------------------------------------------------------------------


def installed_versions(names=LIBS) -> dict[str, str | None]:
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def gpu_problem(count: int, visible: str | None) -> str | None:
    """One training job per GPU (plan). With more than one GPU visible, transformers' Trainer wraps the model in
    torch DataParallel and splits every batch across them; with none, there is nothing to train on."""
    if count == 1:
        return None
    seen = f"CUDA_VISIBLE_DEVICES={visible!r}" if visible is not None else "CUDA_VISIBLE_DEVICES is not set"
    return (f"this process sees {count} GPUs ({seen}); a run trains on exactly one. Start it with "
            "CUDA_VISIBLE_DEVICES=<one GPU>; in GRPO server mode, `trl vllm-serve` runs on a different GPU in its own process")


def readiness(pilot: bool, arm: str | None = None) -> list[str]:
    """Pins the sweep cannot run without; a pilot runs before them and records what it measured."""
    if pilot:
        return []
    names = ("TRAIN_STEPS", "GRADIENT_CHECKPOINTING", "TRAIN_LIBS", "TRL_DEFAULTS") + \
        (("GRPO_VLLM_MODE",) if arm is not None and cfgmod.TRAINER[arm] == "grpo" else ())
    return [name for name in names if getattr(pinned, name) is None]


def vllm_mode(arm: str, pilot: bool, given: str | None) -> str | None:
    """Where GRPO's vLLM runs: a pilot says (--vllm-mode, default colocate); a sweep run uses pinned.GRPO_VLLM_MODE."""
    if cfgmod.TRAINER[arm] != "grpo":
        if given is not None:
            raise SystemExit("--vllm-mode is for GRPO only")
        return None
    if not pilot:
        if given is not None:
            raise SystemExit("--vllm-mode is for --pilot runs; sweep runs use pinned.GRPO_VLLM_MODE")
        return pinned.GRPO_VLLM_MODE
    return given or "colocate"


def server_fields(config_cls, host: str, port: int, group_port: int) -> dict:
    """TRL's server-mode config fields, by whichever names this TRL has (host and port, or a base URL)."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    out = {}
    if "vllm_server_base_url" in names and "vllm_server_port" not in names:
        out["vllm_server_base_url"] = f"http://{host}:{port}"
    elif "vllm_server_port" in names:
        out.update({k: v for k, v in (("vllm_server_host", host), ("vllm_server_port", port)) if k in names})
    else:
        raise SystemExit(f"this TRL's GRPOConfig has no server address field; vLLM fields: {sorted(n for n in names if 'vllm' in n)}")
    if "vllm_group_port" in names:
        out["vllm_group_port"] = group_port
    return out


def version_drift(installed: dict, pinned_libs: dict | None) -> list[str]:
    if pinned_libs is None:
        return []
    return [f"{k}: installed {installed.get(k)}, pinned {v}" for k, v in pinned_libs.items() if installed.get(k) != v]


def expected_labels(input_ids: list[int], mask: list[int]) -> list[int]:
    return [i if m else -100 for i, m in zip(input_ids, mask, strict=True)]


def label_problems(labels: list[int], position_ids: list[int], expected: list[int]) -> list[str]:
    """Collated labels against the reference labels, as one row (position_ids restart at 0 where a sequence starts).
    The first token of each sequence may be unlabelled either way (nothing precedes it); everything else must match."""
    if len(labels) != len(expected) or len(position_ids) != len(expected):
        return [f"the collated row has {len(labels)} tokens, the reference {len(expected)}: padded or truncated"]
    bad = [k for k, (a, b, p) in enumerate(zip(labels, expected, position_ids)) if a != b and not (p == 0 and a == -100)]
    return [f"labels differ from the loss mask at {len(bad)} positions, e.g. {bad[:5]}"] if bad else []


# Arguments that only switch truncation off. Some TRL versions have dropped them (and with them the truncation), so
# they are passed only where the installed config has them; which were absent is recorded. Every other argument is
# required: a pinned setting the config lacks fails loudly.
NO_TRUNCATION = {"max_prompt_length": None, "max_completion_length": None}


def optional_fields(config_cls, wanted: dict) -> tuple[dict, list[str]]:
    """(the `wanted` arguments the config class has, the names it lacks)."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    return {k: v for k, v in wanted.items() if k in names}, sorted(k for k in wanted if k not in names)


def without_end(text: str, end: str = pinned.COMPLETION_END) -> str:
    if not text.endswith(end):
        raise ValueError("completion does not end in its end token")
    return text[: -len(end)]


def sft_tokens(order_keys: list[tuple], encoded: dict) -> dict:
    return {"forward": sum(len(encoded[k].input_ids) for k in order_keys),
            "loss": sum(sum(encoded[k].loss_mask) for k in order_keys)}


def dpo_tokens(order_keys: list[tuple], encoded: dict) -> dict:
    pair = sum(2 * len(encoded[k].prompt_ids) + len(encoded[k].chosen_ids) + len(encoded[k].rejected_ids) for k in order_keys)
    return {"forward": pair, "reference_forward": pair,
            "loss": sum(len(encoded[k].chosen_ids) + len(encoded[k].rejected_ids) for k in order_keys)}


def flops(n_params: int, tokens: dict) -> float:
    """6N per trained token, plus 2N per reference-only token (an estimate, labelled as such)."""
    return N_PARAMS_FLOPS * n_params * tokens["forward"] + 2 * n_params * tokens.get("reference_forward", 0)


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m train.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=pinned.SWEEP_ARMS)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--config", default=pinned.SWEEP_CONFIG)
    ap.add_argument("--seed", type=int, default=pinned.SWEEP_SEED)
    ap.add_argument("--init", type=Path, default=None, help="SFT->DPO: the merged SFT winner")
    ap.add_argument("--out", type=Path, default=None, help="run directory (default data/sweep/runs/<arm>_lr<lr>, "
                                                             "or data/sweep/pilot/<arm>_lr<lr> with --pilot)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT.data)
    ap.add_argument("--pilot", action="store_true", help="costing and memory run: records versions and TRL defaults")
    ap.add_argument("--max-steps", type=int, default=None, help="--pilot only")
    ap.add_argument("--gradient-checkpointing", choices=("on", "off"), default=None, help="--pilot only")
    ap.add_argument("--longest-first", action="store_true",
                    help="--pilot only: train on the longest rows first, so a short run meets the worst-case memory")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="GRPO only: completions per forward pass (default 1); the step still holds "
                         "TRAIN_EXAMPLES_PER_STEP prompts. The other arms run pinned.TRAIN_MICRO_BATCH rows.")
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45, help="GRPO colocate mode only")
    ap.add_argument("--vllm-mode", choices=pinned.GRPO_VLLM_MODES, default=None,
                    help="GRPO --pilot only (default colocate); sweep runs use pinned.GRPO_VLLM_MODE")
    ap.add_argument("--vllm-server-host", default="127.0.0.1", help="GRPO server mode: where `trl vllm-serve` listens")
    ap.add_argument("--vllm-server-port", type=int, default=8000, help="GRPO server mode: its HTTP port")
    ap.add_argument("--vllm-group-port", type=int, default=51216,
                    help="GRPO server mode: the weight-sync port (each concurrent server needs its own)")
    ap.add_argument("--check-only", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pilot and (args.max_steps is not None or args.gradient_checkpointing is not None or args.longest_first):
        raise SystemExit("--max-steps, --gradient-checkpointing and --longest-first are for --pilot runs only")
    if args.pilot and args.gradient_checkpointing is None:
        raise SystemExit("a pilot states --gradient-checkpointing on|off, so the memory measurement is explicit")
    args.micro_batch = micro_batch(args.arm, args.micro_batch)
    args.vllm_mode = vllm_mode(args.arm, args.pilot, args.vllm_mode)
    paths = Paths(args.data_dir.resolve())
    git = git_state(ROOT)
    if git["commit"] is None or git["dirty"]:
        raise SystemExit(f"commit the code and pins before the GPU run (git state: {git})")
    missing = readiness(args.pilot, args.arm)
    if missing:
        raise SystemExit(f"pins still None: {missing}; the pilot sets them (docs/decisions/step10_decision_record.md)")
    versions = installed_versions()
    drift = version_drift(versions, pinned.TRAIN_LIBS)
    if drift:
        raise SystemExit("library versions differ from pinned.TRAIN_LIBS: " + "; ".join(drift))
    steps = cfgmod.train_steps(args.max_steps if args.pilot else None)
    grad_ckpt = (args.gradient_checkpointing == "on") if args.pilot else pinned.GRADIENT_CHECKPOINTING
    name = cfgmod.run_name(args.arm, args.lr)
    out = args.out or (paths.sweep / ("pilot" if args.pilot else "runs") / name)

    import datasets
    import peft
    import torch
    import transformers
    import trl

    from frozen_model.prepare import load_bank
    from frozen_model.run_vllm import _field_names, sampling_kwargs
    from probe.run_vllm import pinned_tokenizer

    problem = gpu_problem(torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    if problem:
        raise SystemExit(problem)
    trl_defaults = {f.name: f.default for f in dataclasses.fields(trl.GRPOConfig) if f.name in pinned.GRPO_RECORDED_DEFAULTS}
    trl_defaults = json.loads(json.dumps(trl_defaults, default=str))
    if pinned.TRL_DEFAULTS is not None and trl_defaults != pinned.TRL_DEFAULTS:
        raise SystemExit(f"TRL GRPOConfig defaults {trl_defaults} differ from pinned.TRL_DEFAULTS {pinned.TRL_DEFAULTS}")

    init = None
    if args.init is not None:
        meta = args.init / "merge_meta.json"
        if not meta.exists():
            raise SystemExit(f"{args.init} is not a merged checkpoint (no merge_meta.json)")
        init = {"kind": "merged", "path": str(args.init), "weights_sha256": json.loads(meta.read_text())["weights_sha256"]}
    trainer_kind = cfgmod.TRAINER[args.arm]
    cfg = cfgmod.resolved(args.arm, args.lr, steps, init, args.config, args.seed,
                          trl_defaults if trainer_kind == "grpo" else None)
    cfg["gradient_checkpointing"] = grad_ckpt
    if trainer_kind == "grpo":
        cfg["grpo"]["vllm_mode"] = args.vllm_mode
    cfgmod.check_shared([cfg] + [dict(cfgmod.resolved(a, args.lr, steps, {"kind": "merged"} if a == "sft_dpo" else None,
                                                      args.config, args.seed), gradient_checkpointing=grad_ckpt)
                                 for a in pinned.SWEEP_ARMS if a != args.arm and not (a == "base" and args.config != "m1")])

    # --- rows, order, reference ids ------------------------------------------------------------------
    file_path, file_sha = data.training_file(paths, cfg["reads"], args.config, args.seed)
    rows = read_rows(file_path)
    order = data.run_order(rows, steps * pinned.TRAIN_EXAMPLES_PER_STEP, args.seed)
    keys = [data.order_key(r) for r in order]
    bank, _ = load_bank(paths)
    tokenizer, tok_sha, stop_ids = pinned_tokenizer()
    encode = lambda texts: tokenizer(texts, add_special_tokens=False)["input_ids"]  # noqa: E731
    encoded, problems = data.encode_rows(trainer_kind, rows, encode, tokenizer.convert_tokens_to_ids,
                                         {r["item_id"]: r["prompt_tokens"] for r in bank})
    if problems:
        raise SystemExit(f"{len(problems)} reference-id problems, e.g. {problems[:5]}")
    caps = {data.order_key(r): r["max_completion_tokens"] for r in rows} if trainer_kind == "grpo" else None
    max_length = data.longest(encoded, caps)
    if args.longest_first:
        order = data.longest_first(order, encoded, caps)
        keys = [data.order_key(r) for r in order]

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    def write_log(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    # --- model -----------------------------------------------------------------------------------------
    source = {"pretrained_model_name_or_path": str(args.init)} if args.init else \
        {"pretrained_model_name_or_path": pinned.TOKENIZER_REPO, "revision": pinned.TOKENIZER_REVISION}
    t0 = time.perf_counter()
    model = transformers.AutoModelForCausalLM.from_pretrained(**source, dtype=getattr(torch, pinned.TRAIN_DTYPE),
                                                              attn_implementation=pinned.TRAIN_ATTENTION)
    n_params = sum(p.numel() for p in model.parameters())
    lora = peft.LoraConfig(**{**pinned.LORA, "target_modules": list(pinned.LORA["target_modules"])}, task_type="CAUSAL_LM")
    common = dict(
        output_dir=str(out / "trainer"), learning_rate=args.lr, max_steps=steps, **pinned.OPTIMIZER,
        bf16=True, gradient_checkpointing=grad_ckpt, use_liger_kernel=pinned.TRAIN_LIGER, seed=args.seed,
        logging_steps=1, save_strategy="no", report_to=[], remove_unused_columns=False,
    )
    sequential = _sequential_mixin(torch)
    callbacks = [_save_at(transformers, cfg["checkpoint_steps"]), _jsonl_logger(transformers, write_log)]
    extra_meta: dict = {}

    if trainer_kind == "sft":
        # The labels carry the loss mask (-100 off the loss tokens); TRL uses them as given on a prepared dataset.
        ds = datasets.Dataset.from_list([{"input_ids": list(encoded[k].input_ids),
                                          "labels": expected_labels(list(encoded[k].input_ids), list(encoded[k].loss_mask))}
                                         for k in keys])
        targs = trl.SFTConfig(**common, per_device_train_batch_size=args.micro_batch,
                              gradient_accumulation_steps=_accum(pinned.TRAIN_EXAMPLES_PER_STEP, args.micro_batch),
                              max_length=max_length, padding_free=False, packing=False,
                              completion_only_loss=False, dataset_kwargs={"skip_prepare_dataset": True})
        Trainer = type("PinnedSFTTrainer", (sequential, trl.SFTTrainer), {})
        trainer = Trainer(model=model, args=targs, train_dataset=ds, processing_class=tokenizer, peft_config=lora,
                          callbacks=callbacks)
        problems = data.compare_prepared([encoded[k].input_ids for k in keys], trainer.train_dataset["input_ids"], "input_ids")
        for i, k in enumerate(keys[:4]):   # one row per forward pass: each collates alone, unpadded
            labels = trainer.data_collator([trainer.train_dataset[i]])["labels"].view(-1).tolist()
            problems += label_problems(labels, list(range(len(labels))),
                                       expected_labels(list(encoded[k].input_ids), list(encoded[k].loss_mask)))
        tokens = sft_tokens(keys, encoded)
    elif trainer_kind == "dpo":
        opt, extra_meta["truncation_args_absent"] = optional_fields(trl.DPOConfig, NO_TRUNCATION)
        targs = trl.DPOConfig(**common, **opt, per_device_train_batch_size=args.micro_batch,
                              gradient_accumulation_steps=_accum(pinned.TRAIN_EXAMPLES_PER_STEP, args.micro_batch),
                              beta=pinned.DPO["beta"], loss_type=pinned.DPO["loss_type"], max_length=max_length,
                              padding_free=False, precompute_ref_log_probs=False)
        Trainer = type("PinnedDPOTrainer", (sequential, trl.DPOTrainer), {})
        problems, trainer, end_form, tried = [], None, None, {}
        for end_form in ("ours", "trl"):   # does TRL append the end token itself? the reference ids decide
            strip = (lambda t: without_end(t)) if end_form == "trl" else (lambda t: t)
            ds = datasets.Dataset.from_list([{"prompt": r["prompt"], "chosen": strip(r["chosen"]), "rejected": strip(r["rejected"])}
                                             for r in order])
            trainer = Trainer(model=model, args=targs, train_dataset=ds, processing_class=tokenizer, peft_config=lora,
                              callbacks=callbacks)
            prep = trainer.train_dataset
            problems, layout = data.dpo_prepared_problems([encoded[k] for k in keys], prep.column_names,
                                                          lambda name: prep[name])
            extra_meta["dpo_prepared_columns"] = layout
            tried[end_form] = problems
            if not problems:
                break
            model = trainer.model.unload()   # the next attempt wraps the bare model again
        if problems:
            problems = [f"{form}: {p}" for form, ps in tried.items() for p in ps]
        extra_meta["dpo_end_token_added_by"] = end_form
        probe = _dpo_probe(order, encoded)
        trainer.add_callback(_dpo_monitor(transformers, torch, trainer, probe, write_log))
        tokens = dpo_tokens(keys, encoded)
    else:
        import vllm

        from etl.cwe_graph import load_cwe_graph

        from .grpo import trainer_class
        if vllm.__version__ != pinned.VLLM_VERSION:
            raise SystemExit(f"vLLM {vllm.__version__} is installed; pinned.VLLM_VERSION is {pinned.VLLM_VERSION}")
        ds_rows = data.grpo_dataset_rows(order)
        rows_by_item = {r["item_id"]: r for r in data.grpo_dataset_rows(rows)}
        per_step = pinned.TRAIN_EXAMPLES_PER_STEP * pinned.GRPO_GENERATIONS
        opt, extra_meta["truncation_args_absent"] = optional_fields(trl.GRPOConfig, {"max_prompt_length": None})
        if args.vllm_mode == "colocate":
            placement = {"vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                         "vllm_max_model_length": pinned.EVAL_MAX_MODEL_LEN}
        else:
            placement = server_fields(trl.GRPOConfig, args.vllm_server_host, args.vllm_server_port, args.vllm_group_port)
            extra_meta["vllm_server"] = placement
        targs = trl.GRPOConfig(**common, **pinned.GRPO, **opt, per_device_train_batch_size=args.micro_batch,
                               gradient_accumulation_steps=_accum(per_step, args.micro_batch),
                               use_vllm=True, vllm_mode=args.vllm_mode, **placement)
        holder: dict = {}

        def dense(prompts, completions, **kwargs):
            return holder["trainer"].reward(prompts, completions, **kwargs)

        by_text = {r["prompt"]: r["max_completion_tokens"] for r in rows}
        by_ids = {encoded[data.order_key(r)].prompt_ids: r["max_completion_tokens"] for r in rows}
        dynamic = {r["type"] for r in rows if r["dynamic_sampling"]}
        canary = [{"prompt": r["prompt"], "prompt_ids": list(encoded[data.order_key(r)].prompt_ids)}
                  for t in ("mcq", "cvss") for r in [next(x for x in order if x["type"] == t)]]
        monitor = grpo_logic.Monitor()
        Trainer = trainer_class(trl, torch, vllm, sampling_kwargs, _field_names(vllm.SamplingParams))
        trainer = Trainer(model=model, args=targs, train_dataset=datasets.Dataset.from_list(ds_rows),
                          processing_class=tokenizer, peft_config=lora, reward_funcs=[dense], callbacks=callbacks,
                          caps=grpo_logic.CapLookup(by_text, by_ids), queue=grpo_logic.TypeQueue.of(order, dynamic),
                          dynamic_types=dynamic, rows_by_item=rows_by_item, graph=load_cwe_graph(paths.cwe_xml),
                          monitor=monitor, canary=canary, stop_ids=stop_ids, write_log=write_log)
        holder["trainer"] = trainer
        extra_meta.update(vllm_mode=args.vllm_mode, vllm_engine_at=trainer.vllm_path, weight_sync_hook=trainer.sync_hook)
        print(f"vLLM {args.vllm_mode} mode, at trainer.{trainer.vllm_path}; weight-sync hook {trainer.sync_hook}")
        trainer.add_callback(_monitor_flush(transformers, monitor, write_log))
        problems = []
        if trainer.args.gradient_accumulation_steps * args.micro_batch != per_step:
            problems.append("one optimizer step does not hold TRAIN_EXAMPLES_PER_STEP prompts x GRPO_GENERATIONS")
        gen_batch = getattr(trainer.args, "generation_batch_size", per_step)
        if gen_batch != per_step:
            problems.append(f"TRL generates {gen_batch} completions per batch, not one optimizer step's {per_step}")
        if not problems:   # one call through the cap wrapper: an MCQ prompt and a line-localisation prompt
            first = [next(r["prompt"] for r in ds_rows if r["type"] == t) for t in ("mcq", "line_loc")]
            seen = trainer.trial_generate(first)
            if any(n > by_text[p] for n, p in zip(seen, first)):
                problems.append(f"vLLM returned {seen} tokens under caps {[by_text[p] for p in first]}")
            print(f"capped vLLM call: token counts {seen} under caps {[by_text[p] for p in first]}")
            problems += trainer.backbone_check()   # vLLM serves the pinned backbone (matters most in server mode)
        tokens = None  # counted after training: prompts x GRPO_GENERATIONS plus generated tokens

    if problems:
        raise SystemExit(f"{len(problems)} preparation problems, e.g. {problems[:5]}")
    load_s = time.perf_counter() - t0
    print(f"checks passed: {args.arm} lr {args.lr:.0e}, {steps} steps x {pinned.TRAIN_EXAMPLES_PER_STEP} rows "
          f"({data.passes(len(rows), steps):.2f} passes over {len(rows):,} rows), max length {max_length:,}; "
          f"prepared ids equal the reference; config sha256 {cfgmod.config_sha(cfg)[:12]}")
    if args.check_only:
        return 0

    last = sorted((out / "trainer").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(last[-1]) if last else None)
    train_s = time.perf_counter() - t0

    from .merge import adapter_sha256
    if tokens is None:
        prompts = sum(len(encoded[k].prompt_ids) for k in keys) * pinned.GRPO_GENERATIONS
        tokens = {"forward": prompts + trainer.generated_tokens, "generated": trainer.generated_tokens,
                  "note": "generated tokens counted this session (a resumed run counts only its own)"}
        extra_meta.update(sync_checks=trainer.sync_checks, spliced_keys_untouched=sorted(trainer.spliced_keys),
                          logps_row_chunked_calls=trainer.logps_chunked_calls,
                          replacements={k: v for k, v in trainer.queue.replacements.most_common()})
    events = [json.loads(line).get("monitor", {}).get("events", []) for line in open(log_path, encoding="utf-8")] \
        if log_path.exists() else []
    meta = {
        "run": name, "pilot": args.pilot, "git": git, "versions": {**versions, "python": platform.python_version(),
                                                                   "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name(0), "config": cfg, "config_sha256": cfgmod.config_sha(cfg),
        "trl_grpo_defaults": trl_defaults, "training_file": str(file_path), "training_file_content_sha256": file_sha,
        "tokenizer_sha256": tok_sha, "rows_in_file": len(rows), "passes": data.passes(len(rows), steps),
        "max_length": max_length, "micro_batch": args.micro_batch, "longest_first": args.longest_first, "steps_done": trainer.state.global_step,
        "tokens": tokens, "n_params": n_params, "flops": flops(n_params, tokens),
        "seconds": {"load_and_checks": round(load_s, 1), "train_this_session": round(train_s, 1)},
        "gpu_hours": round(train_s / 3600, 3), "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapters": {str(s): adapter_sha256(out / "trainer" / f"checkpoint-{s}") for s in cfg["checkpoint_steps"]
                     if (out / "trainer" / f"checkpoint-{s}").exists()},
        "monitor_events": [e for es in events for e in es],
        **extra_meta,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"{name}: {meta['steps_done']} steps in {train_s / 3600:.2f} h; adapters at {sorted(meta['adapters'])}")
    return 0


# ---------------------------------------------------------------------------
# Trainer pieces (built against the imported libraries)
# ---------------------------------------------------------------------------


def micro_batch(arm: str, given: int | None) -> int:
    """Rows (GRPO: completions) per forward pass. Under SDPA the cross-entropy and DPO arms are pinned to
    TRAIN_MICRO_BATCH, so no row is padded or packed beside another."""
    if cfgmod.TRAINER[arm] == "grpo":
        return 1 if given is None else given
    if given not in (None, pinned.TRAIN_MICRO_BATCH):
        raise SystemExit(f"--micro-batch is pinned to {pinned.TRAIN_MICRO_BATCH} for {arm} (pinned.TRAIN_MICRO_BATCH)")
    return pinned.TRAIN_MICRO_BATCH


def _accum(rows_per_step: int, micro: int) -> int:
    if rows_per_step % micro:
        raise SystemExit(f"--micro-batch {micro} does not divide {rows_per_step} rows per step")
    return rows_per_step // micro


def _sequential_mixin(torch):
    class Sequential:
        """The run order is ours (train.data.run_order); the trainer must not shuffle it."""

        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    return Sequential


def _save_at(transformers, steps: list[int]):
    class SaveAt(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in steps:
                control.should_save = True
            return control

    return SaveAt()


def _jsonl_logger(transformers, write_log):
    class JsonlLogger(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            write_log({"step": state.global_step, "trainer": logs})

    return JsonlLogger()


def _monitor_flush(transformers, monitor, write_log):
    class MonitorFlush(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % pinned.MONITOR_EVERY == 0:
                write_log({"monitor": monitor.summary(state.global_step)})
                monitor.reset()

    return MonitorFlush()


def _dpo_probe(order: list[dict], encoded: dict, per_type: int = 4, max_tokens: int = 4096) -> list[tuple[str, object]]:
    """A fixed handful of pairs per type (the first in run order that fit max_tokens) for the log-prob monitor."""
    out: list[tuple[str, object]] = []
    for t in pinned.BANK_TYPES:
        picked = [encoded[data.order_key(r)] for r in order if r["type"] == t and encoded[data.order_key(r)].length <= max_tokens]
        out += [(t, e) for e in list(dict.fromkeys(picked))[:per_type]]
    return out


def _dpo_monitor(transformers, torch, trainer, probe, write_log):
    """Plan: log the mean log-probability of chosen and rejected separately throughout training (DPO's pathology is
    driving both down). Per type, per token, on the fixed probe pairs, every MONITOR_EVERY steps."""

    def mean_logp(prompt, completion):
        model = trainer.accelerator.unwrap_model(trainer.model)
        ids = torch.tensor([list(prompt) + list(completion)], device=trainer.accelerator.device)
        logits = model(input_ids=ids).logits[0, len(prompt) - 1:len(prompt) - 1 + len(completion)].float()
        lp = torch.log_softmax(logits, dim=-1)[torch.arange(len(completion)), torch.tensor(list(completion), device=ids.device)]
        return float(lp.mean())

    class DpoMonitor(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % pinned.MONITOR_EVERY:
                return
            out: dict[str, dict] = {}
            with torch.no_grad():
                for t, e in probe:
                    d = out.setdefault(t, {"chosen": [], "rejected": []})
                    d["chosen"].append(mean_logp(e.prompt_ids, e.chosen_ids))
                    d["rejected"].append(mean_logp(e.prompt_ids, e.rejected_ids))
            write_log({"dpo_logps": {"step": state.global_step,
                                     **{t: {k: sum(v) / len(v) for k, v in d.items()} for t, d in out.items()}}})

    return DpoMonitor()


if __name__ == "__main__":
    raise SystemExit(main())
