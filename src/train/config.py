"""The resolved configuration of one training run: everything that defines it, from the pins. Pure.

Every arm shares the fields in SHARED (LoRA, optimizer, steps, examples per step, checkpoints, precision, attention,
the run order); `check_shared` asserts it, and run_meta.json records the whole dict and its sha256.
"""

from __future__ import annotations

import json

from etl import pinned

# Which TRL trainer each arm uses, and which step-9 file it reads (SFT->DPO reads the dpo file).
TRAINER = {"base": "sft", "sft": "sft", "distill_self": "sft", "dpo": "dpo", "sft_dpo": "dpo", "grpo": "grpo"}
LOSS = {"base": "all_tokens", "sft": "completion_tokens", "distill_self": "completion_tokens",
        "dpo": "dpo", "sft_dpo": "dpo", "grpo": "grpo"}
SHARED = ("config", "seed", "steps", "examples_per_step", "checkpoint_steps", "lora", "optimizer", "dtype",
          "attention", "liger", "gradient_checkpointing", "order_salt", "backbone")
assert set(TRAINER) == set(pinned.SWEEP_ARMS) == set(LOSS)


def reads(arm: str) -> str:
    """The step-9 file an arm trains on."""
    return pinned.SFT_DPO_READS if arm == "sft_dpo" else arm


def run_name(arm: str, lr: float) -> str:
    """'sft_lr5e-05': the run's directory name."""
    return f"{arm}_lr{lr:.0e}"


def checkpoint_steps(steps: int, k: int = pinned.CHECKPOINTS) -> list[int]:
    """The k evenly spaced save steps, round half up of j * steps / k for j = 1..k; the last is `steps`."""
    if steps < k:
        raise ValueError(f"{steps} steps cannot hold {k} distinct checkpoints")
    return [(2 * j * steps + k) // (2 * k) for j in range(1, k + 1)]


def train_steps(pilot_steps: int | None = None) -> int:
    """pinned.TRAIN_STEPS, or a pilot's reduced count. The sweep refuses to run before the owner pins it."""
    if pilot_steps is not None:
        return pilot_steps
    if pinned.TRAIN_STEPS is None:
        raise SystemExit("pinned.TRAIN_STEPS is None: the owner pins it from the pilot's measured GPU-hours "
                         "(docs/decisions/step10_decision_record.md); only --pilot runs before that")
    return pinned.TRAIN_STEPS


def resolved(arm: str, lr: float, steps: int, init: dict | None = None, config: str = pinned.SWEEP_CONFIG,
             seed: int = pinned.SWEEP_SEED, trl_defaults: dict | None = None) -> dict:
    """The full configuration of one run. `init` describes a non-backbone starting checkpoint (SFT->DPO only);
    `trl_defaults` are the recorded TRL GRPOConfig defaults (pinned.TRL_DEFAULTS once the pilot has set them)."""
    if arm not in pinned.SWEEP_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if lr not in pinned.LR_GRID:
        raise ValueError(f"LR {lr} is not in pinned.LR_GRID {pinned.LR_GRID}")
    if (arm == "sft_dpo") != (init is not None):
        raise ValueError("SFT->DPO starts from the merged SFT winner (--init); every other arm from the backbone")
    if arm == "base" and config != "m1":
        raise ValueError("the base arm is built for M1 only")
    cfg = {
        "arm": arm,
        "trainer": TRAINER[arm],
        "reads": reads(arm),
        "loss": LOSS[arm],
        "lr": lr,
        "config": config,
        "seed": seed,
        "steps": steps,
        "examples_per_step": pinned.TRAIN_EXAMPLES_PER_STEP,
        "checkpoint_steps": checkpoint_steps(steps),
        "lora": {**pinned.LORA, "target_modules": list(pinned.LORA["target_modules"])},
        "optimizer": dict(pinned.OPTIMIZER),
        "dtype": pinned.TRAIN_DTYPE,
        "attention": pinned.TRAIN_ATTENTION,
        "liger": pinned.TRAIN_LIGER,
        "gradient_checkpointing": pinned.GRADIENT_CHECKPOINTING,
        "order_salt": pinned.TRAIN_ORDER_SALT,
        "backbone": f"{pinned.TOKENIZER_REPO}@{pinned.TOKENIZER_REVISION}",
        "init": init or {"kind": "backbone"},
        "loss_normalisation": "token mean over the step's examples" if TRAINER[arm] == "sft" else "trainer default",
    }
    if cfg["trainer"] == "dpo":
        cfg["dpo"] = dict(pinned.DPO)
        cfg["reference"] = "the starting checkpoint, adapter disabled"
    if cfg["trainer"] == "grpo":
        cfg["grpo"] = {**pinned.GRPO, "trl_defaults": trl_defaults}
        cfg["rollout_sampling"] = dict(pinned.ROLLOUT_SAMPLING)
        cfg["dynamic_sampling_retries"] = pinned.DYNAMIC_SAMPLING_RETRIES
        cfg["monitor_every"] = pinned.MONITOR_EVERY
    return cfg


def shared(cfg: dict) -> dict:
    return {k: cfg[k] for k in SHARED}


def check_shared(cfgs: list[dict]) -> None:
    """Every arm's shared fields are identical (the plan's tuning and efficiency parity)."""
    first = shared(cfgs[0])
    for c in cfgs[1:]:
        diff = sorted(k for k in SHARED if shared(c)[k] != first[k])
        if diff:
            raise SystemExit(f"{c['arm']} differs from {cfgs[0]['arm']} on shared fields {diff}")


def config_sha(cfg: dict) -> str:
    return pinned.sha256_text(json.dumps(cfg, sort_keys=True))
