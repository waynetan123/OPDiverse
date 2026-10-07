"""Pinned definitions for the OPDiverse fact table, test window, hierarchy scoring, probe, question bank,
frozen-model session, external-model jobs, engine-agreement check, primary test, seed partitions,
converters and the LR sweep.

Everything here can move a census count, a split or a score. It is frozen before step 0: any
change needs a matching entry in docs/decisions/step{1,...,10}_decision_record.md. Pure functions,
stdlib only, no I/O.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, Protocol

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

CWE_RELEASE = "4.20"
CWE_VIEW = "1000"
NVD_SOURCE = "nvd@nist.gov"
CWE_PLACEHOLDERS = frozenset({"NVD-CWE-Other", "NVD-CWE-noinfo"})

TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"
# HF commit of Qwen/Qwen2.5-7B-Instruct (lastModified 2025-01-12) and the sha256 of its tokenizer.json.
TOKENIZER_REVISION: str | None = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_SHA256: str | None = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
# Public release date of Qwen2.5. Its pretraining cutoff is not published; reported beside the split boundary.
BACKBONE_RELEASED = "2024-09-19"

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

TOKEN_CAP = 10_000
TOKEN_CAP_REPORT = (1_024, 2_048, 4_096)
PATCH_FRACTION = Fraction(1, 5)

# ---------------------------------------------------------------------------
# Test window (step 2)
# ---------------------------------------------------------------------------

TEST_FRACTION = Fraction(3, 20)
NEAR_DUP_JACCARD = Fraction(4, 5)
SHINGLE_N = 3

# ---------------------------------------------------------------------------
# Seed partitions (step 8)
# ---------------------------------------------------------------------------

# Owner, step 8: partitions are drawn for five seeds; step 11 decides whether seeds 3-4 are trained.
PARTITION_SEEDS = (0, 1, 2, 3, 4)
# The late window holds LATE_WINDOW_MULTIPLE x the dev target, ceil(DEV_FRACTION x all facts), so that a
# random half of it gives the 70/15/15 fallback's dev (owner, step 8; the plan's 20% gave ~193 dev CVEs).
DEV_FRACTION = Fraction(3, 20)
LATE_WINDOW_MULTIPLE = 2
DEV_SALT = "dev-partition"
# The fixed checkpoint-selection subsample: CHECKPOINT_CVES dev CVEs per seed, all six items each.
CHECKPOINT_CVES = 150
CHECKPOINT_SALT = "checkpoint-subsample"

# ---------------------------------------------------------------------------
# Converters (step 9)
# ---------------------------------------------------------------------------

# One training file per arm, per matrix configuration, per seed: a selection over the frozen bank and the cached
# step-5 and step-6 files. SFT->DPO trains on the dpo file of the same configuration and seed (it differs from
# DPO-from-base only in initialisation); distill-external was dropped at step 6.
CONVERTER_ARMS = ("base", "sft", "distill_self", "dpo", "grpo")
QUESTION_ARMS = ("sft", "distill_self", "dpo", "grpo")  # base has no question items, so it is built for M1 only
SFT_DPO_READS = "dpo"
# M2 drops one type from train (and from dev at checkpoint selection) and upsamples the rest back to M1's row
# count. M1-volume drops the same share of items uniformly at random, redrawn per seed, then upsamples the same
# way. The share is 1/6 for the one-item types and 1/3 for find_error, which has two of a CVE's six items
# (owner, step 9; the plan's 20% matched no M2 loop).
M1V_FRACTIONS = (Fraction(1, 6), Fraction(1, 3))
CONVERTER_SALTS = {"mask": "m1v-mask", "upsample": "upsample", "order": "order"}
# Completions end in the token evaluation stops on; base documents end in Qwen's end-of-document token.
COMPLETION_END = "<|im_end|>"
BASE_DOC_END = "<|endoftext|>"
# The base arm's raw text (owner, step 9): the facts every other arm trains on, with no question and no answer
# format. The description and functions are the redacted strings the prompts show; the CWE name is MITRE's.
BASE_DOC_TEMPLATE = (
    "A vulnerability is described as follows:\n\n{description}\n\n"
    "Weakness: {cwe}: {name}\n"
    "CVSS v3 base vector: {vector}\n\n"
    "Vulnerable function:\n\n```c\n{vulnerable}\n```\n\n"
    "Patched function:\n\n```c\n{patched}\n```"
)

# ---------------------------------------------------------------------------
# Exact-ID hierarchy credit (step 2)
# ---------------------------------------------------------------------------

# If the best constant answer's mean symmetric score over non-test facts exceeds this, the
# direction-aware schedule is adopted. The computed number decides; see baselines.json.
EXACT_ID_THRESHOLD = Fraction(1, 5)
SCHEDULES = ("symmetric", "direction_aware")
# Decided at step 2: best constant CWE-119 scores 0.262 under the symmetric schedule (> 0.2).
EXACT_ID_SCHEDULE = "direction_aware"
SYMMETRIC_CREDIT = {0: Fraction(1), 1: Fraction(1, 2), 2: Fraction(1, 4)}
# Relation of prediction to gold -> credit. At distance 2, a prediction that is an ancestor of
# gold (by any path) takes the ancestor discount; every other 2-hop relation scores 1/4.
DIRECTION_AWARE_CREDIT = {
    "exact": Fraction(1),
    "child": Fraction(1, 2),
    "parent": Fraction(1, 4),
    "descendant_2": Fraction(1, 4),   # grandchild
    "sibling": Fraction(1, 4),        # shares a parent
    "coparent": Fraction(1, 4),       # shares a child
    "ancestor_2": Fraction(1, 8),     # grandparent, or an ancestor also reachable in 2 hops
    "far": Fraction(0),
}

# ---------------------------------------------------------------------------
# Evaluation decoding (step 3; the plan's uniform evaluation settings)
# ---------------------------------------------------------------------------

EVAL_MAX_TOKENS = 512
# Greedy. Qwen's generation_config.json defaults to temperature 0.7 / top-p 0.8 / top-k 20 /
# repetition penalty 1.05, and vLLM applies it unless told not to; every field is set explicitly.
EVAL_SAMPLING = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": EVAL_MAX_TOKENS,
    "seed": 0,
}
EVAL_DTYPE = "bfloat16"
# Qwen's generation_config lists both as EOS; with that config ignored, the runner stops on both explicitly.
EVAL_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
# The longest question-bank prompt (line localisation, 16,473 tokens) + EVAL_MAX_TOKENS; set at step 4.
# The probe ran at 1,024 (its prompts are ~80 tokens). `generators bank check` fails if this stops covering the bank.
EVAL_MAX_MODEL_LEN = 16_985
# The Qwen2.5 chat template inserts this when no system message is given.
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
# From the probe's run_meta.json (step 3); pinned at step 5. Every later vLLM run must match it.
VLLM_VERSION: str | None = "0.30.0"

# ---------------------------------------------------------------------------
# Contamination probe (step 3)
# ---------------------------------------------------------------------------

PROBE_SAMPLE_SIZE = 300
PROBE_SALT = "contamination-probe"
PROBE_PROMPTS = {
    "cwe": (
        "Which CWE (Common Weakness Enumeration) weakness does {cve_id} correspond to? "
        "Reply with a single CWE ID in the form CWE-<number> and nothing else. "
        "If you are not sure, give your best guess."
    ),
    "cvss": (
        "What is the CVSS v3.1 base vector of {cve_id}? "
        "Reply with the vector only, in the form AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_, and nothing else. "
        "If you are not sure, give your best guess."
    ),
}
PROBE_MATERIAL = Fraction(1, 20)   # recall margin >= 5 points (with p < alpha) -> sensitivity run
PROBE_LARGE = Fraction(3, 20)      # recall margin >= 15 points -> reconsider backbone / extend forward
PROBE_ALPHA = 0.05
PROBE_PERMUTATIONS = 10_000
PROBE_BOOTSTRAP = 2_000
PROBE_SEED = 0

# ---------------------------------------------------------------------------
# Question bank (step 4)
# ---------------------------------------------------------------------------

BANK_TEMPLATE_VERSION = "v1"
BANK_TYPES = ("mcq", "exact_id", "cvss", "find_error", "line_loc")
ITEMS_PER_CVE = 6  # find_error contributes two: index 0 vulnerable, index 1 patched
MCQ_LETTERS = "ABCD"

# Every template ends by asking for the answer in the target format, so the same prompt serves
# arms that answer directly (SFT) and arms that reason first (distill); the parsers read the end.
BANK_PROMPTS = {
    "mcq": (
        "Here is the NVD description of a vulnerability:\n\n{description}\n\n"
        "Which CWE (Common Weakness Enumeration) weakness does it describe?\n\n{options}\n\n"
        "End your reply with the letter of the correct option, in the form ANSWER: <letter>."
    ),
    "exact_id": (
        "Here is the NVD description of a vulnerability:\n\n{description}\n\n"
        "Which CWE (Common Weakness Enumeration) weakness does it describe? "
        "End your reply with a single CWE ID, in the form CWE-<number>."
    ),
    "cvss": (
        "Here is the NVD description of a vulnerability:\n\n{description}\n\n"
        "What is its CVSS v3 base vector? "
        "End your reply with the vector, in the form AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_."
    ),
    "find_error": (
        "Here is a C/C++ function:\n\n```c\n{function}\n```\n\n"
        "Does this function contain a security vulnerability, and if so, which CWE (Common Weakness "
        "Enumeration) weakness is it? End your reply with VULNERABLE: yes, CWE-<number> or VULNERABLE: no."
    ),
    "line_loc": (
        "Here is the NVD description of a vulnerability:\n\n{description}\n\n"
        "Here is the vulnerable C/C++ function. Each line starts with its line number, counting from 1, "
        "then a colon and a space.\n\n```\n{numbered}\n```\n\n"
        "Which lines does the fix change? A line counts if the fix modifies or deletes it. Code the fix adds "
        "counts against the nearest code line above it, or the first code line if it is added before all of "
        "them. Blank and comment-only lines never count. End your reply with the line numbers in ascending "
        "order, in the form LINES: 12, 13, 17."
    ),
}
MCQ_OPTION = "{letter}. {cwe}: {name}"
# Literal CWE IDs in a description give the answer away on MCQ and exact-ID; the item's own CVE ID
# is removed wherever it appears. The stored fact is unchanged.
CWE_LITERAL = re.compile(r"(?<![A-Za-z0-9])CWE[-_ ]?\d+(?!\d)", re.IGNORECASE)
CWE_REDACTED = "CWE-[redacted]"
CVE_REDACTED = "CVE-[redacted]"

# Line localisation trivial baselines.
LINE_KEYWORDS = ("memcpy", "strcpy", "alloc")  # case-sensitive substrings
EVERY_KTH_LINE = (2, 3)                          # lines 1, 1+k, 1+2k, ...

# MCQ distractors: the external model proposes, the rules admit, names come from the XML.
MCQ_DISTRACTORS = 3
MCQ_REQUEST_MAX = 8
MCQ_ATTEMPTS = 2                   # the first request plus one regeneration, then the prior-matched draw
MCQ_PILOT_SIZE = 100
MCQ_SHORTCUT_MAX = Fraction(1, 2)  # above this on non-test, the prior-matched draw replaces the model's picks
MCQ_SALTS = {"letter": "mcq-letter", "slot": "mcq-slot", "draw": "mcq-draw", "pilot": "mcq-pilot"}
MCQ_REQUEST_PROMPT = (
    "I am building a multiple-choice question for a benchmark that tests whether a model can classify a "
    "vulnerability from its description. The question shows the description below and four CWE options. "
    "One option is the correct answer; I need the other three.\n\n"
    "Description:\n{description}\n\n"
    "Correct answer: {cwe}: {name}\n\n"
    "List up to {n} CWE IDs that would make good wrong options, ordered from most to least plausible. Each "
    "should be tempting to someone reading the description, but wrong once the description is read "
    "carefully. Prefer weaknesses that commonly occur in C and C++ code. Do not include the correct answer, "
    "and do not include any weakness that is a more general or a more specific version of it (its ancestors "
    "or descendants under ChildOf in the CWE-1000 Research Concepts view), because those would also be "
    "defensible answers. Use weakness IDs from CWE release {release} only, not categories or views, "
    "written as CWE-<number>."
)
MCQ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"distractors": {"type": "array", "items": {"type": "string"}}},
    "required": ["distractors"],
    "additionalProperties": False,
}

# The external model (steps 4 and 6): one model, one configuration, every external job. No dated
# snapshot exists and sampling cannot be set, so its outputs are cached and audited, not regenerated.
EXTERNAL_MODEL = {
    "provider": "anthropic",
    "model": "claude-opus-5-5",
    "effort": "medium",
    "max_tokens": 16_000,
    "thinking": {"type": "adaptive"},
}

# ---------------------------------------------------------------------------
# Frozen-model session (step 5): the GRPO signal audit and distill-self rationales
# ---------------------------------------------------------------------------

# One sampling config for the audit and for GRPO training rollouts: TRL GRPOConfig's defaults.
# n, max_tokens and a per-request seed are added per request (frozen_model.requests.sampling_for).
ROLLOUT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
ROLLOUT_N = 8
# The plan's per-type rollout caps. The GRPO cap per type is the smallest of these and
# ROLLOUT_CAP_CANDIDATES at which at most ROLLOUT_CUTOFF_MAX of audit rollouts are cut off.
ROLLOUT_MAX_TOKENS = {"mcq": 16, "exact_id": 24, "cvss": 48, "find_error": 64, "line_loc": 64}
ROLLOUT_CAP_CANDIDATES = (128, 256, 512)
ROLLOUT_CUTOFF_MAX = Fraction(1, 10)

# The audit samples once at AUDIT_MAX_TOKENS; a shorter cap's rollout is a prefix of that sample.
AUDIT_PER_TYPE = 200   # find_error: both items of AUDIT_PER_TYPE // 2 CVEs
AUDIT_MAX_TOKENS = 512
AUDIT_LIVE = Fraction(1, 2)    # live-group fraction >= this: the column trains as specified
AUDIT_FLOOR = Fraction(1, 10)  # below this: floored; in between: dynamic sampling (GRPO)
AUDIT_SALT = "signal-audit"

# distill-self: attempt 1 is greedy (EVAL_SAMPLING); the one regeneration samples under ROLLOUT_SAMPLING.
RATIONALE_MAX_TOKENS = EVAL_MAX_TOKENS
RATIONALE_SALT = "distill-self"
RATIONALE_PROMPT = (
    "{user}\n\n"
    "The correct answer is:\n{target}\n\n"
    "Write out, step by step, the reasoning that leads from the information above to this answer, as if you "
    "were working it out yourself. Do not say or suggest that you were given the answer. Finish with the "
    "answer alone on the last line, written exactly as: {target}"
)
# A rationale whose reasoning matches any of these restates the hint rather than justifying the answer.
HINT_LEAK = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:we|i)\s+(?:were|was|are|am|have\s+been|'ve\s+been)\s+(?:told|given|informed)\b",
    r"\b(?:given|provided|supplied|stated)\s+(?:correct\s+)?(?:answer|solution)\b",
    r"\b(?:answer|solution)\s+(?:was|is|has\s+been)\s+(?:given|provided|supplied|stated)\b",
    r"\bthe\s+hint\b",
    r"\b(?:you|the\s+(?:prompt|question|user))\s+(?:said|says|stated|states|told|tells)\b[^.\n]{0,40}\banswer\b",
))
# Substitution, per type, on the FIRST-attempt surface-validity pass rate (owner, step 5): fires if the
# pass rate is below SUBSTITUTION_MAX or the fallback-to-gold-only rate exceeds it. If it fires on
# SUBSTITUTION_DROP_AT or more types, distill-self leaves the primary test.
SUBSTITUTION_MAX = Fraction(1, 2)
SUBSTITUTION_DROP_AT = 3

# ---------------------------------------------------------------------------
# External-model jobs (step 6): DPO rejected answers
# ---------------------------------------------------------------------------

# distill-external is not run (owner, step 6): the external model refused all 120 pilot trace requests
# under its `reasoning_extraction` category. See docs/decisions/step6_decision_record.md.
TEACHER_PILOT_CVES = 20
TEACHER_SALTS = {"pilot": "teacher-pilot", "dpo": "dpo-rule"}

# DPO: chosen is the gold target; rejected is a near miss one unit of error from gold, in the same
# canonical format. The external model picks which near miss, the rules in generators.teacher.dpo decide
# what is admissible, and a rule-constructed near miss fills in after one regeneration. MCQ sends no
# request (owner, step 6): its rejected letter is the distractor closest to gold in the hierarchy.
DPO_REQUEST_TYPES = ("exact_id", "cvss", "find_error", "line_loc")
DPO_PROMPT = (
    "I am building preference pairs to train a model on the question below. The correct answer is:\n{target}\n\n"
    "I need the single most plausible wrong answer: the one a careful but mistaken expert would most likely "
    "give. It must follow this rule: {rule}\n\n"
    "The question, exactly as the model sees it:\n<question>\n{user}\n</question>"
)
# Keyed by type, and by find_error_{index} (0 vulnerable, 1 patched).
DPO_RULES = {
    "exact_id": (
        "it must be one of these weaknesses, each one ChildOf step away from the correct answer in the "
        "CWE-1000 Research Concepts view:\n{neighbours}\nGive its ID as CWE-<number>."
    ),
    "cvss": (
        "change exactly one of the eight components of the correct vector to another valid value and keep the "
        "other seven. Give the whole vector in the form AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_."
    ),
    "find_error_0": (
        "keep the verdict that the function is vulnerable, but name a wrong weakness: one of these, each one "
        "ChildOf step away from the correct one in the CWE-1000 Research Concepts view:\n{neighbours}\n"
        "Give its ID as CWE-<number>."
    ),
    "find_error_1": (
        "the function is the patched version of one whose vulnerability was {cwe}: {name}, so the wrong answer "
        "calls it vulnerable. Give the weakness that someone who wrongly judged it vulnerable would most "
        "plausibly name, as CWE-<number>: a weakness from CWE release {release}, not a category or view."
    ),
    "line_loc": (
        "change the correct set of lines in exactly one way: either leave out one of the correct lines (only if "
        "there are two or more), or replace one of them with a different code line at least 2 lines away from "
        "it and not next to another correct line (answers within one line of a correct line are scored as "
        "correct). Blank and comment-only lines never count. Give the line numbers in ascending order."
    ),
}
DPO_FIELDS = {"exact_id": "cwe", "cvss": "vector", "find_error": "cwe", "line_loc": "lines"}
DPO_SCHEMAS = {
    field: {
        "type": "object",
        "properties": {field: {"type": "array", "items": {"type": "integer"}} if field == "lines" else {"type": "string"}},
        "required": [field],
        "additionalProperties": False,
    }
    for field in ("cwe", "vector", "lines")
}

# ---------------------------------------------------------------------------
# Engine agreement (step 7): HuggingFace versus vLLM on the raw backbone
# ---------------------------------------------------------------------------

# The sample (owner, step 7): the ENGINE_CVES non-test CVEs with the lowest stable_rank(cve_id, ENGINE_SALT)
# within the late window, all six items each. The late window is the plan's pool for dev: the latest
# ENGINE_LATE_WINDOW of the non-test pool by (published, cve_id). Greedy EVAL_SAMPLING; test is never read.
ENGINE_LATE_WINDOW = Fraction(1, 5)
ENGINE_CVES = 200
ENGINE_SALT = "engine-agreement"
# Material disagreement (owner, step 7): on any type, |HF - vLLM| >= ENGINE_GAP and the paired bootstrap
# interval over CVEs at ENGINE_CI excludes 0. Lenient parsers decide; strict is reported. 99% per type
# keeps the false-alarm rate over five types near 5% (Bonferroni).
ENGINE_GAP = Fraction(1, 100)
ENGINE_CI = Fraction(99, 100)
ENGINE_BOOTSTRAP = 10_000
ENGINE_SEED = 0
# The HF reference: batch size 1 (no padding), EVAL_DTYPE, PyTorch SDPA attention, greedy with every
# sampling field explicit. Qwen's generation_config would add repetition_penalty 1.05 even under greedy.
HF_ATTENTION = "sdpa"
HF_DETERMINISM_CHECK = 5  # requests re-generated per shard
# Evaluation (step 12) merges each LoRA adapter into the base weights and serves the merged checkpoint
# through evaluate.run_vllm, the code path this check covers.
EVAL_LORA = "merged"
# Parser review: replies shown per (source, type, category) in parser_review.md, by stable_rank.
PARSER_REVIEW_SAMPLE = 10
PARSER_REVIEW_SALT = "parser-review"

# ---------------------------------------------------------------------------
# Primary test and minimum detectable effect (frozen at step 7)
# ---------------------------------------------------------------------------

# The four primary post-trained arms; the test runs on three if distill-self leaves it (it did not at step 5).
PRIMARY_ARMS = ("sft", "distill_self", "dpo", "grpo")
# The null (owner, step 7): a parametric seed bootstrap, replacing the plan's item permutation, which ignores
# training-run noise. PRIMARY_REPLICATES simulated experiments: additive cell means (no interaction) plus
# Gaussian run noise at each column's pooled seed SD, the SDs re-estimated in every replicate.
PRIMARY_REPLICATES = 10_000
PRIMARY_SEED = 0
PRIMARY_ALPHA = 0.05
# T* >= T is read as T* >= T * (1 - PRIMARY_TIE_RTOL), so float summation order cannot break a tie.
PRIMARY_TIE_RTOL = 1e-12
# MDE: MDE_SIMULATIONS experiments drawn from the same null (MDE_SEED), with delta points planted in one
# (arm, type) cell, every arm in turn; power = share with p < PRIMARY_ALPHA against the frozen test's null
# replicates. MDE = the smallest grid delta from which power stays >= MDE_POWER.
MDE_SIMULATIONS = 1_000
MDE_SEED = 1
MDE_POWER = Fraction(4, 5)
MDE_GRID_STEP = Fraction(1, 4)  # points
MDE_GRID_MAX = 50               # points
# sha256 of src/analysis/{primary_test,mde}.py (analysis.source_sha256). `python -m analysis` refuses to run
# on any other source; a change needs a new value here and a decision-record entry.
PRIMARY_TEST_SHA256: str | None = "15e75eb78f5a3ba16f064ae2665bc9b2b0574088d78aebacab456d7df345a207"

# ---------------------------------------------------------------------------
# LR sweep (step 10)
# ---------------------------------------------------------------------------

# The sweep (plan, Tuning parity): M1, partition seed 0, every arm, LR the only swept knob. SFT->DPO starts from
# the merged SFT winner. The seed-0 winners are also M1's seed-0 keepers (owner, step 10: 123 -> 117 runs).
SWEEP_SEED = 0
SWEEP_CONFIG = "m1"
SWEEP_ARMS = ("base", "sft", "distill_self", "dpo", "grpo", "sft_dpo")
# One grid for every arm (owner, step 10). No extension if a winner lands on an edge; edges are flagged.
LR_GRID = (1e-5, 5e-5, 2e-4)
# Training randomness (LoRA init, GRPO sampling) is seeded with the partition seed.
TRAIN_ORDER_SALT = "train-order"
# Matched optimizer steps: every arm takes TRAIN_STEPS steps of TRAIN_EXAMPLES_PER_STEP rows (GRPO: prompts, each
# with GRPO_GENERATIONS rollouts). 9,576 / 24 = 399 steps is one pass over a question-arm file. TRAIN_STEPS is
# pinned by the owner from the pilot's measured GPU-hours (1 or 3 passes); the sweep refuses to run while it is None.
TRAIN_EXAMPLES_PER_STEP = 24
TRAIN_STEPS: int | None = None
CHECKPOINTS = 4  # at round-half-up(k * TRAIN_STEPS / CHECKPOINTS), k = 1..CHECKPOINTS
# LoRA, identical for every arm (plan: rank 16 on attention and MLP projections). alpha = r: scale 1; the LR sweep
# absorbs the scale.
LORA = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}
TRAIN_DTYPE = "bfloat16"
# PyTorch SDPA (owner, step 10), not flash-attn, which would not build against torch 2.13 + CUDA 13 on the GPU machine.
# SDPA cannot keep sequences apart inside one packed row, so the cross-entropy and DPO arms run TRAIN_MICRO_BATCH = 1
# row per forward pass and build each step's TRAIN_EXAMPLES_PER_STEP rows by gradient accumulation: no padding, no
# attention across rows, the same loss. GRPO's micro-batches of completions are padded and masked, which SDPA handles.
TRAIN_ATTENTION = "sdpa"
TRAIN_MICRO_BATCH = 1
# transformers TrainingArguments defaults, written out so a library change cannot move them.
OPTIMIZER = {
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
}
# Fused linear cross-entropy (logits of a 16k-token row over a 152k vocabulary would not fit), every arm.
TRAIN_LIGER = True
# Set from the pilot's memory measurement (the plan's default is off); None until then.
GRADIENT_CHECKPOINTING: bool | None = None
# DPO (plan: beta fixed at its published default, TRL's 0.1). pi_ref is the starting checkpoint, adapter disabled.
DPO = {"beta": 0.1, "loss_type": "sigmoid"}
# GRPO: what the plan and step 5 pin. Everything else is TRL GRPOConfig's default at the pinned TRL version, read by
# the pilot and recorded in TRL_DEFAULTS; the sweep refuses to run while that is None.
GRPO_GENERATIONS = 8
GRPO = {
    "num_generations": GRPO_GENERATIONS,
    **{k: ROLLOUT_SAMPLING[k] for k in ("temperature", "top_p", "repetition_penalty")},  # step 5; vLLM gets all of it
    "scale_rewards": "group",              # A_i = (r_i - mean(r)) / std(r), within the group (plan)
    "num_iterations": 1,
    "mask_truncated_completions": False,    # a cut-off rollout is scored on its text, as at step 5
    "shuffle_dataset": False,               # the run order is ours (TRAIN_ORDER_SALT)
    "max_completion_length": 512,           # the largest step-5 cap; each request carries its type's own cap
}
GRPO_RECORDED_DEFAULTS = ("beta", "loss_type", "epsilon", "epsilon_high", "importance_sampling_level",
                          "vllm_importance_sampling_correction", "top_k", "min_p")
# Dynamic sampling (step 5 bands): on a type banded "dynamic_sampling", a group whose rewards all tie is dropped and
# replaced by the next prompt of the same type from a per-type queue in run order (wrapping). One replacement per
# dropped prompt; a tied replacement is kept. How often each prompt was seen is logged.
DYNAMIC_SAMPLING_RETRIES = 1
# Running monitor (plan): every MONITOR_EVERY optimizer steps, per type, over that window's rollouts.
MONITOR_EVERY = 10
COLLAPSE_RATE = Fraction(9, 10)  # find_error: one predicted class above this over a window is a collapse
# Weight-sync canary (plan: the one failure that gives a wrong number). Every MONITOR_EVERY steps vLLM decodes the
# canary prompts greedily and the trainer scores those tokens under the current weights and the starting weights.
# Once the policy has moved (gap to the start > SYNC_MIN_DRIFT nats per token), vLLM must sit nearer the current
# weights than the start: gap_current < SYNC_RATIO x gap_start.
SYNC_CANARY_TOKENS = 32
SYNC_MIN_DRIFT = 0.05
SYNC_RATIO = 0.5
# Where a vLLM server returns no log-probs, the check reads greedy choices instead: on vLLM's canary tokens, the share
# that are the argmax under the current weights and under the starting weights. Once either share is at most
# 1 - SYNC_MIN_FLIP (the policy has moved enough to change greedy choices), vLLM must agree more with the current weights.
SYNC_MIN_FLIP = 0.1
# Before training, vLLM's greedy canary tokens must be the backbone's argmax at least this often (bf16 near-ties
# aside): the server is serving the pinned backbone.
SERVER_BACKBONE_AGREE = 0.9
# Where GRPO's vLLM runs (plan: decided by measurement on the pilot): "colocate", on the training GPU, or "server",
# TRL's `trl vllm-serve` on a second GPU. Set from the pilot; the sweep refuses to run while it is None.
GRPO_VLLM_MODES = ("colocate", "server")
GRPO_VLLM_MODE: str | None = None
# Library versions and the TRL defaults above, recorded by the pilot (as VLLM_VERSION was by the probe).
TRAIN_LIBS: dict | None = None
TRL_DEFAULTS: dict | None = None
# Selection (owner, step 10): the column-standardised mean across types, each type divided by its per-CVE spread:
# for each evaluation, the ddof-1 SD over CVEs of the per-CVE score (find_error: the paired score); pooled as the
# root mean of those variances over the scale evaluations (SCALE_ARMS x LR_GRID x CHECKPOINTS, on the checkpoint
# subsample). Frozen before any checkpoint or LR is chosen; reused for checkpoint selection at steps 11 and 15.
SCALE_ARMS = ("base", "sft", "distill_self", "dpo", "grpo")
SELECTION_SCALE: dict | None = None
# Paired cluster bootstrap over dev CVEs for each LR pair, reported (selection is the argmax; ties: lower LR,
# earlier checkpoint).
SWEEP_CI = Fraction(95, 100)
SWEEP_BOOTSTRAP = 10_000
SWEEP_BOOTSTRAP_SEED = 0
# The outcome: arm -> winning LR (set when step 10 is recorded).
SWEEP_LR: dict | None = None

# Deprecated CWE-1000 weaknesses -> replacement, read from each entry's Description in the
# 4.20 XML. A replacement is recorded only where MITRE names exactly one successor; None
# means the row is dropped. Keys must cover every deprecated weakness in the release.
DEPRECATED_REPLACEMENT: dict[str, str | None] = {
    "71": "62",      # "Please refer to CWE-62"
    "92": "75",      # "CWE-75 is a more appropriate mapping"
    "132": "170",    # duplicate of CWE-170
    "216": None,     # no successor named
    "217": None,     # split into CWE-766 and CWE-767
    "218": "493",    # duplicate of CWE-493
    "225": "199",    # "can be found at CWE-199" (a category, so the row still drops)
    "247": "350",    # duplicate of CWE-350
    "249": "785",    # "most of its content has been transferred to CWE-785"
    "292": "350",    # duplicate of CWE-350
    "365": None,     # no successor named
    "373": None,     # overlaps CWE-362 and CWE-662
    "423": "441",    # duplicate of CWE-441
    "443": "113",    # "can be found at CWE-113"
    "458": None,     # description duplicated CWE-454, name suggested CWE-665
    "516": "385",    # "can be found at CWE-385"
    "533": "532",    # "See CWE-532"
    "534": "532",    # "See CWE-532"
    "542": "532",    # "See CWE-532"
    "545": None,     # "partially overlaps CWE-470" - not a replacement
    "592": "287",    # redundant with CWE-287
    "596": "1023",   # "Its closest equivalent is CWE-1023"
    "769": "774",    # duplicate of CWE-774
    "1187": "908",   # duplicate of CWE-908
    "1324": "319",   # "integrated into CWE-319"
}

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_rank(*parts: str) -> int:
    """Deterministic 64-bit rank. Never use hash(): it is salted per process."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Function text
# ---------------------------------------------------------------------------


def split_lines(text: str) -> list[str]:
    """Split on LF only. str.splitlines() also breaks on \\f, \\v, \\x1c-\\x1e, \\x85, \\u2028."""
    return text.split("\n")


def clean_function(raw: str) -> str:
    """The stored, model-facing function: CRLF/CR -> LF, leading/trailing blank lines removed.

    Nothing else changes; line numbers everywhere refer to this text, 1-indexed.
    """
    lines = split_lines(raw.replace("\r\n", "\n").replace("\r", "\n"))
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def render_numbered(text: str) -> str:
    """The line-numbered form the model sees for line localisation."""
    return "\n".join(f"{i}: {line}" for i, line in enumerate(split_lines(text), 1))


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _raw_string_start(text: str, i: int) -> bool:
    """True if the quote at text[i] opens a C++ raw string (R"..., u8R"..., LR"...)."""
    if i == 0 or text[i - 1] != "R":
        return False
    k = i - 1
    if text[max(0, k - 2):k] == "u8":
        k -= 2
    elif k >= 1 and text[k - 1] in "uUL":
        k -= 1
    return k == 0 or not _is_ident_char(text[k - 1])


def _digit_separator(text: str, i: int) -> bool:
    """True if the apostrophe at text[i] is a C++14 digit separator (1'000'000)."""
    if i == 0 or i + 1 >= len(text) or not _is_ident_char(text[i - 1]) or not text[i + 1].isalnum():
        return False
    k = i - 1
    while k > 0 and (_is_ident_char(text[k - 1]) or text[k - 1] == "'"):
        k -= 1
    return text[k].isdigit()


def mask_comments(text: str) -> tuple[str, bool]:
    """Blank out // and /* */ comments, keeping every newline so line numbers survive.

    String and character literals are skipped so comment markers inside them are left alone.
    Returns (text, False) if a literal or block comment is unterminated; the caller then falls
    back to the unmasked text.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = i
            while j < n and text[j] != "\n":
                if text[j] == "\\" and j + 1 < n and text[j + 1] == "\n":
                    out[j] = " "
                    j += 2  # backslash-newline continues the comment; keep the newline
                    continue
                out[j] = " "
                j += 1
            i = j
        elif c == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                return text, False
            for j in range(i, end + 2):
                if text[j] != "\n":
                    out[j] = " "
            i = end + 2
        elif c == '"' and _raw_string_start(text, i):
            paren = text.find("(", i + 1)
            if paren < 0 or paren - i - 1 > 16:
                return text, False
            terminator = ")" + text[i + 1:paren] + '"'
            end = text.find(terminator, paren + 1)
            if end < 0:
                return text, False
            i = end + len(terminator)
        elif c == "'" and _digit_separator(text, i):
            i += 1
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    return text, False
                j += 1
            if j >= n:
                return text, False
            i = j + 1
        else:
            i += 1
    return "".join(out), True


def line_key(line: str) -> str:
    """Comparison key for one line: all whitespace removed."""
    return "".join(line.split())


def _code_lines(text: str) -> list[tuple[int, str]]:
    """(1-indexed line number, key) for every line whose key is non-empty."""
    return [(i, k) for i, line in enumerate(split_lines(text), 1) if (k := line_key(line))]


def _code_keys(text: str) -> list[str]:
    masked, ok = mask_comments(text)
    return [k for _, k in _code_lines(masked if ok else text)]


def norm_body_hash(text: str) -> str:
    """Whitespace- and comment-insensitive hash of a function, for near-duplicate checks."""
    return sha256_text("\n".join(_code_keys(text)))


def code_shingles(text: str) -> frozenset[tuple[str, ...]]:
    """SHINGLE_N consecutive code-line keys (comment-masked, whitespace removed). A function
    shorter than SHINGLE_N code lines is one shingle."""
    keys = _code_keys(text)
    return frozenset(tuple(keys[i:i + SHINGLE_N]) for i in range(max(1, len(keys) - SHINGLE_N + 1)))


def jaccard_at_least(a: frozenset, b: frozenset, threshold: Fraction = NEAR_DUP_JACCARD) -> bool:
    """|a & b| / |a | b| >= threshold, in exact integer arithmetic."""
    inter = len(a & b)
    return inter * threshold.denominator >= (len(a) + len(b) - inter) * threshold.numerator


# ---------------------------------------------------------------------------
# Patch line set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchResult:
    lines: tuple[int, ...]        # gold patch line set on the vulnerable function, sorted
    n_lines: int                  # all lines of the vulnerable function
    n_code_lines: int             # lines with a non-empty key (the 20% denominator)
    n_inserted: int               # patched-side lines not matched ("+" lines)
    n_deleted: int                # vulnerable-side lines not matched ("-" lines)
    mask_ok: bool                 # comment masking applied to both sides
    opcodes: tuple[tuple[str, int, int, int, int], ...]  # difflib opcodes over code lines
    vuln_code: tuple[int, ...]    # original line number of each vulnerable code line
    patched_code: tuple[int, ...]  # original line number of each patched code line


def patch_line_set(vuln: str, patched: str) -> PatchResult:
    """Gold lines for line localisation. Both inputs are clean_function() output.

    Lines are compared on their key after comment masking; blank and comment-only lines take
    no part in the diff. Replaced and deleted lines are gold; an insertion is attributed to
    the preceding code line, or to the first code line if it comes before all of them.
    """
    masked_v, ok_v = mask_comments(vuln)
    masked_p, ok_p = mask_comments(patched)
    ok = ok_v and ok_p
    a = _code_lines(masked_v if ok else vuln)
    b = _code_lines(masked_p if ok else patched)
    matcher = difflib.SequenceMatcher(None, [k for _, k in a], [k for _, k in b], autojunk=False)
    opcodes = tuple(matcher.get_opcodes())
    gold: set[int] = set()
    n_inserted = n_deleted = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        n_deleted += i2 - i1
        n_inserted += j2 - j1
        if tag in ("replace", "delete"):
            gold.update(a[i][0] for i in range(i1, i2))
        elif a:  # insert
            gold.add(a[i1 - 1][0] if i1 > 0 else a[0][0])
    return PatchResult(
        lines=tuple(sorted(gold)),
        n_lines=len(split_lines(vuln)),
        n_code_lines=len(a),
        n_inserted=n_inserted,
        n_deleted=n_deleted,
        mask_ok=ok,
        opcodes=opcodes,
        vuln_code=tuple(ln for ln, _ in a),
        patched_code=tuple(ln for ln, _ in b),
    )


def exceeds_fraction(count: int, denominator: int) -> bool:
    """count > PATCH_FRACTION * denominator, in exact integer arithmetic."""
    return count * PATCH_FRACTION.denominator > denominator * PATCH_FRACTION.numerator


# ---------------------------------------------------------------------------
# CVSS v3.x
# ---------------------------------------------------------------------------

CVSS_VERSIONS = ("3.0", "3.1")
CVSS_ORDER = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")
CVSS_ALLOWED = {"AV": "NALP", "AC": "LH", "PR": "NLH", "UI": "NR", "S": "UC", "C": "HLN", "I": "HLN", "A": "HLN"}
_CVSS_DECOMPOSED = {
    "AV": "attackVector", "AC": "attackComplexity", "PR": "privilegesRequired", "UI": "userInteraction",
    "S": "scope", "C": "confidentialityImpact", "I": "integrityImpact", "A": "availabilityImpact",
}
_CVSS_METRIC_KEYS = (("3.1", "cvssMetricV31"), ("3.0", "cvssMetricV30"))


def parse_cvss_v3(vector: str) -> tuple[str | None, dict[str, str]] | None:
    """Parse a v3.x base vector in any component order, case-insensitively.

    Returns (declared version or None if unprefixed, components in canonical order), or None
    if any base metric is missing, duplicated, unknown or out of range.
    """
    parts = vector.strip().split("/")
    version = None
    if parts[0].upper().startswith("CVSS:"):
        version = parts[0][5:]
        if version not in CVSS_VERSIONS:
            return None
        parts = parts[1:]
    components: dict[str, str] = {}
    for part in parts:
        key, sep, value = part.partition(":")
        key, value = key.strip().upper(), value.strip().upper()
        if not sep or key not in CVSS_ALLOWED or key in components or len(value) != 1 or value not in CVSS_ALLOWED[key]:
            return None
        components[key] = value
    if len(components) != len(CVSS_ORDER):
        return None
    return version, {k: components[k] for k in CVSS_ORDER}


def canonical_cvss(components: dict[str, str]) -> str:
    return "/".join(f"{k}:{components[k]}" for k in CVSS_ORDER)


def select_nvd_cvss(metrics: dict) -> tuple[dict | None, str]:
    """NVD's own v3.x vector, preferring v3.1. Filters on source, never on type.

    Returns ({version, vector, components, both_versions}, "ok") or (None, drop reason).
    """
    by_version = {}
    for version, key in _CVSS_METRIC_KEYS:
        entries = [m for m in metrics.get(key, []) if m.get("source") == NVD_SOURCE]
        if entries:
            by_version[version] = entries
    if not by_version:
        if metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30"):
            return None, "cna_only_v3"
        if metrics.get("cvssMetricV2"):
            return None, "v2_only"
        if metrics.get("cvssMetricV40"):
            return None, "v4_only"
        return None, "no_cvss"
    version = "3.1" if "3.1" in by_version else "3.0"
    entries = by_version[version]
    if len({m["cvssData"]["vectorString"] for m in entries}) > 1:
        return None, "conflicting_nvd_vectors"
    data = entries[0]["cvssData"]
    parsed = parse_cvss_v3(data["vectorString"])
    if parsed is None:
        return None, "invalid_vector"
    declared, components = parsed
    if declared != version or data.get("version", version) != version:
        return None, "version_mismatch"
    for key, field in _CVSS_DECOMPOSED.items():
        full = data.get(field)
        if full is not None and full[:1].upper() != components[key]:
            return None, "decomposed_mismatch"
    return {
        "version": version,
        "vector": canonical_cvss(components),
        "components": components,
        "both_versions": len(by_version) == 2,
    }, "ok"


# ---------------------------------------------------------------------------
# CWE
# ---------------------------------------------------------------------------


class CweLookup(Protocol):
    def is_deprecated_weakness(self, cwe: str) -> bool: ...
    def is_category(self, cwe: str) -> bool: ...
    def is_view(self, cwe: str) -> bool: ...
    def in_view(self, cwe: str) -> bool: ...
    def parents(self, cwe: str) -> frozenset[str]: ...
    def children(self, cwe: str) -> frozenset[str]: ...
    def ancestors(self, cwe: str) -> frozenset[str]: ...
    def descendants(self, cwe: str) -> frozenset[str]: ...


_CWE_ID = re.compile(r"CWE-(\d+)")


def nvd_cwe_values(weaknesses: list[dict]) -> tuple[bool, list[str]]:
    """(any NVD-sourced weakness block present, sorted distinct values across those blocks)."""
    blocks = [w for w in weaknesses if w.get("source") == NVD_SOURCE]
    values = {
        d["value"].strip()
        for w in blocks
        for d in w.get("description", [])
        if d.get("lang") == "en" and d.get("value", "").strip()
    }
    return bool(blocks), sorted(values)


def resolve_cwe(has_nvd_block: bool, values: list[str], graph: CweLookup) -> tuple[str | None, str, tuple[str, ...]]:
    """One CWE-1000 weakness per CVE, or a drop reason.

    Order: strip placeholders -> map deprecated -> dedupe -> multi-CWE -> must be a
    non-deprecated view-1000 weakness. Returns (CWE-id or None, reason, mapped-from IDs).
    """
    if not has_nvd_block:
        return None, "no_nvd_cwe", ()
    real = [v for v in values if v not in CWE_PLACEHOLDERS]
    if not real:
        return None, "placeholder_only", ()
    ids: set[str] = set()
    mapped: list[str] = []
    for value in real:
        m = _CWE_ID.fullmatch(value)
        if not m:
            return None, "malformed_cwe", ()
        cwe = m.group(1)
        if graph.is_deprecated_weakness(cwe):
            replacement = DEPRECATED_REPLACEMENT[cwe]
            if replacement is None:
                return None, "deprecated_no_replacement", (f"CWE-{cwe}",)
            mapped.append(f"CWE-{cwe}")
            cwe = replacement
        ids.add(cwe)
    mapped_from = tuple(sorted(mapped, key=lambda s: int(s[4:])))
    if len(ids) > 1:
        return None, "multi_cwe", mapped_from
    (cwe,) = ids
    if graph.is_category(cwe):
        return None, "category", mapped_from
    if graph.is_view(cwe):
        return None, "view", mapped_from
    if not graph.in_view(cwe):
        return None, "not_in_view_1000", mapped_from
    return f"CWE-{cwe}", "ok", mapped_from


def _neighbours(cwe: str, graph: CweLookup) -> frozenset[str]:
    return graph.parents(cwe) | graph.children(cwe)


def cwe_distance(a: str, b: str, graph: CweLookup) -> int | None:
    """Shortest undirected ChildOf distance in view 1000 if it is at most 2, else None. Bare IDs."""
    if a == b:
        return 0
    near = _neighbours(a, graph)
    if b in near:
        return 1
    if any(b in _neighbours(n, graph) for n in near):
        return 2
    return None


def cwe_relation(pred: str, gold: str, graph: CweLookup) -> str:
    """Key into DIRECTION_AWARE_CREDIT. Bare IDs; gold must be a live view-1000 weakness."""
    if not graph.in_view(pred):
        return "far"
    d = cwe_distance(pred, gold, graph)
    if d == 0:
        return "exact"
    if d == 1:
        return "child" if pred in graph.children(gold) else "parent"
    if d == 2:
        if pred in graph.ancestors(gold):
            return "ancestor_2"
        if pred in graph.descendants(gold):
            return "descendant_2"
        if any(pred in graph.children(p) for p in graph.parents(gold)):
            return "sibling"
        return "coparent"
    return "far"


def hierarchy_score(pred: str, gold: str, graph: CweLookup, schedule: str = EXACT_ID_SCHEDULE) -> Fraction:
    """Exact-ID hierarchy credit in [0, 1]. 'CWE-n' strings; anything else as pred scores 0."""
    m, g = _CWE_ID.fullmatch(pred.strip().upper()), _CWE_ID.fullmatch(gold)
    if g is None or not graph.in_view(g.group(1)):
        raise ValueError(f"gold {gold!r} is not a live view-{CWE_VIEW} weakness")
    if m is None or not graph.in_view(m.group(1)):
        return Fraction(0)
    p, g = m.group(1), g.group(1)
    if schedule == "symmetric":
        return SYMMETRIC_CREDIT.get(cwe_distance(p, g, graph), Fraction(0))
    if schedule == "direction_aware":
        return DIRECTION_AWARE_CREDIT[cwe_relation(p, g, graph)]
    raise ValueError(f"unknown schedule {schedule!r}")


def siblings(cwe: str, graph: CweLookup) -> list[str]:
    """Other children of any view-1000 parent (2 ChildOf edges away), excluding the CWE's own
    ancestors and descendants. Bare IDs in, 'CWE-n' out, sorted numerically."""
    found: set[str] = set()
    for parent in graph.parents(cwe):
        found |= graph.children(parent)
    found -= {cwe} | graph.ancestors(cwe) | graph.descendants(cwe)
    return [f"CWE-{c}" for c in sorted(found, key=int)]


# ---------------------------------------------------------------------------
# One function per CVE
# ---------------------------------------------------------------------------

_NOT_A_NAME = frozenset(
    "if else for while do switch case return sizeof typeof alignof _Alignof __alignof__ defined "
    "__attribute__ __attribute __declspec alignas _Alignas decltype __typeof__ asm __asm__ "
    "static_assert _Static_assert noexcept throw "
    "int char void long short unsigned signed float double bool const volatile static inline "
    "extern struct union enum register auto".split()
)
_NAME_BEFORE_PAREN = re.compile(r"((?:[A-Za-z_]\w*\s*::\s*)*~?\s*[A-Za-z_]\w*)\s*$")
_FIRST_ARG = re.compile(r"\s*([A-Za-z_]\w*)")
_MACRO = re.compile(r"[A-Z][A-Z0-9_]+")
_HEADER_LIMIT = 2_000


def extract_function_names(func: str) -> tuple[str, ...]:
    """Candidate names from the header (text before the first '{').

    Every identifier directly before a depth-0 '(' counts; qualified names also contribute
    their last component, and ALL-CAPS macros (PHP_FUNCTION(x), SYSCALL_DEFINE4(f, ...))
    their first argument. Extra candidates are harmless: they only break ties within one CVE.
    """
    masked, ok = mask_comments(func)
    text = (masked if ok else func)[:_HEADER_LIMIT]
    brace = text.find("{")
    header = text if brace < 0 else text[:brace]
    names: list[str] = []
    depth = 0
    for pos, ch in enumerate(header):
        if ch == "(":
            if depth == 0 and (m := _NAME_BEFORE_PAREN.search(header[:pos])):
                full = re.sub(r"\s+", "", m.group(1))
                last = full.rsplit("::", 1)[-1].lstrip("~")
                if last not in _NOT_A_NAME:
                    names += [full, last]
                    if _MACRO.fullmatch(last) and (arg := _FIRST_ARG.match(header, pos + 1)):
                        if arg.group(1) not in _NOT_A_NAME:
                            names.append(arg.group(1))
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return tuple(dict.fromkeys(names))


def name_in_description(names: tuple[str, ...], description: str) -> bool:
    """Whole-word, case-sensitive match of any candidate name."""
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", description) for n in names)


class Candidate(NamedTuple):
    pair_id: str
    vuln_norm_hash: str
    name_in_desc: bool


def choose_function(cve_id: str, candidates: list[Candidate]) -> tuple[str, str]:
    """Pick one surviving pair per CVE: a function named in the NVD description first, then
    the lowest stable_rank(cve_id, vuln_norm_hash), with pair_id as the final tie-break."""
    if len(candidates) == 1:
        return candidates[0].pair_id, "single"
    named = [c for c in candidates if c.name_in_desc]
    if len(named) == 1:
        return named[0].pair_id, "name_in_description"
    pool, rule = (named, "name_in_description+hash") if named else (candidates, "hash")
    best = min(pool, key=lambda c: (stable_rank(cve_id, c.vuln_norm_hash), c.pair_id))
    return best.pair_id, rule
