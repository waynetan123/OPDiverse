"""Merge one LoRA checkpoint into its starting weights for evaluation (pinned.EVAL_LORA = "merged"; step 7: merge
in fp32, then cast to bf16). Also gives SFT->DPO its starting checkpoint (the merged SFT winner).

    PYTHONPATH=src python -m train.merge --adapter data/sweep/runs/sft_lr5e-05/trainer/checkpoint-100 \\
        --out data/sweep/runs/sft_lr5e-05/step100/merged [--init DIR]

--init is the run's own starting checkpoint (SFT->DPO: the merged SFT winner), read from the adapter's run_meta.json
when present. Writes the merged model, safetensors, and merge_meta.json with the adapter's and the output's sha256.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from etl import pinned
from etl.manifest import file_sha256, git_state, tree_sha256
from etl.paths import ROOT

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def adapter_sha256(adapter: Path) -> str:
    """sha256 over the adapter's own files (a checkpoint directory also holds optimizer state)."""
    missing = [n for n in ADAPTER_FILES if not (adapter / n).exists()]
    if missing:
        raise SystemExit(f"{adapter} is not a LoRA checkpoint: missing {missing}")
    return pinned.sha256_text("".join(f"{n}:{file_sha256(adapter / n)}\n" for n in ADAPTER_FILES))


def run_init(adapter: Path) -> str | None:
    """The starting checkpoint recorded by the run that wrote this adapter (trainer/checkpoint-N -> run dir)."""
    meta = adapter.parent.parent / "run_meta.json"
    if not meta.exists():
        return None
    init = json.loads(meta.read_text(encoding="utf-8"))["config"]["init"]
    return init.get("path")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.merge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None)
    args = ap.parse_args(argv)
    git = git_state(ROOT)
    if git["commit"] is None or git["dirty"]:
        raise SystemExit(f"commit the code and pins before merging (git state: {git})")
    if (args.out / "merge_meta.json").exists():
        print(f"{args.out} is already merged")
        return 0
    recorded = run_init(args.adapter)
    init = str(args.init) if args.init else recorded
    if recorded is not None and init != recorded:
        raise SystemExit(f"--init {init} is not the run's recorded starting checkpoint {recorded}")
    a_sha = adapter_sha256(args.adapter)

    import peft
    import torch
    import transformers

    source = {"pretrained_model_name_or_path": init} if init else \
        {"pretrained_model_name_or_path": pinned.TOKENIZER_REPO, "revision": pinned.TOKENIZER_REVISION}
    base = transformers.AutoModelForCausalLM.from_pretrained(**source, dtype=torch.float32)
    merged = peft.PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    merged = merged.to(getattr(torch, pinned.EVAL_DTYPE))
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    meta = {
        "adapter": str(args.adapter), "adapter_sha256": a_sha, "init": init or f"{pinned.TOKENIZER_REPO}@{pinned.TOKENIZER_REVISION}",
        "merge_dtype": "float32", "saved_dtype": pinned.EVAL_DTYPE, "weights_sha256": tree_sha256(args.out), "git": git,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                     "python": platform.python_version()},
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "merge_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged {args.adapter} -> {args.out} (weights sha256 {meta['weights_sha256'][:12]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
