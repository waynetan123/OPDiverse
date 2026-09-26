# Step 3 decision record — contamination probe

Pre-registration for the contamination probe (procedure step 3). All pins below are implemented in `src/etl/pinned.py`, `src/etl/verifiers.py` and `src/probe/`. **They must be committed before the GPU run**, so the rule is verifiably fixed before any output exists. "User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## Question

Has the raw backbone (Qwen/Qwen2.5-7B-Instruct @ `a09a354…`) memorised the test CVEs? It is given only each CVE ID, with no description or code.

## Verifiers (written at this step; reused by every later step)

`src/etl/verifiers.py`, parser version **v1**. A verifier turns a reply plus gold into a `Verdict`:
- `parsed`: the canonical answer;
- `parse_ok`: the lenient parser found an answer;
- `strict_ok`: the reply is exactly the target format;
- `metric`: the reported score;
- `dense`: the training signal.

Both scores are in [0, 1] and both are 0 on a parse failure. The probe, the signal audit, DPO negative validation, the GRPO reward and evaluation all call these functions.

| | Exact-ID | CVSS |
|---|---|---|
| Target format | `CWE-787` | `AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H` |
| Lenient parser | The last `CWE[-_ :]*<digits>` in the reply, case-insensitive, normalised | The last valid value of each of the 8 base metrics; letter boundaries stop `AC:L` counting as `C:L`; absent or invalid metrics stay missing |
| Strict parser | The reply is exactly `CWE-<digits>` | The reply is the 8 metrics in canonical order, uppercase, optional `CVSS:3.x/` prefix |
| `metric` | Exact match (0/1) | Share of the 8 components correct (missing = wrong) |
| `dense` | `hierarchy_score`, direction-aware (step 2) | Same as `metric` |

It is string matching on normalised IDs; CWE names are not mapped to IDs. Parsers are re-tuned only on dev outputs (step 7); the probe is then re-scored and both versions reported. Probe outputs are never used to tune parsers.

## Pins

| Pin | Value | Source |
|---|---|---|
| Sample | The 300 test CVEs with the lowest `stable_rank(cve_id, "contamination-probe")` | default |
| Prompts | `PROBE_PROMPTS` in `pinned.py`: one CWE question and one CVSS v3.1 question, each giving only the CVE ID, stating the target format, and asking for a best guess if unsure | default |
| Rendering | Qwen chat template, default system prompt, generation prompt on. Stored per request; the runner asserts byte equality with `apply_chat_template` at the pinned revision. | default |
| Engine | vLLM on the GPU cluster, the plan's pinned evaluation engine. The version used becomes `pinned.VLLM_VERSION` for every later evaluation. | user |
| Decoding | `EVAL_SAMPLING`: greedy (temperature 0), top-p 1, repetition penalty 1.0, n = 1, 512 max new tokens, seed 0; bf16; stop on `<|im_end|>` and `<|endoftext|>`; `generation_config="vllm"`, so Qwen's defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05) are **not** applied | default |
| Recall measure | Per primary measure (CWE exact, CVSS agreement): margin = observed − control. Control = the mean score of each answer against every **other** CVE's gold, which is what the model's answering habits earn without CVE-specific memory. | user |
| Significance | One-sided permutation test (gold shuffled across items, 10,000 times, seed 0), p = (1 + #{T* ≥ T}) / 10,001; 95% bootstrap CI (2,000 resamples, seed 0) | default |
| Decision | **Large** if either primary margin ≥ 15 points with p < 0.05 → reconsider the backbone or extend the data forward. **Material** if ≥ 5 points with p < 0.05 → add the sensitivity run of the primary test excluding `recalled.jsonl`. Otherwise **not detected** → proceed unchanged. Lenient parsers decide; strict is reported. | user |
| Recalled CVEs | Exactly right with an answer that isn't the model's most common answer (for CWE or for CVSS) | default |
| Reported alongside | The plan's constant-baseline comparison (CWE-125 exact; the majority CVSS vector; CWE-125 hierarchy), and, descriptively, the best constant on the sample itself | doc |

## Why a permutation control and not only the plan's constants

On the pinned sample, the training-era constant CWE-125 scores 0.117 exact, but always answering CWE-787 scores 0.170. CWE-787 dominates the 2021–22 window. A model with no CVE-specific memory can therefore clear the constant by 5.3 points. Verified: with fake "always CWE-787" replies on the real sample, `probe score` reports +5.3 points vs the constant, but a 0.0-point margin vs the control, so not detected.

## Runbook

1. `PYTHONPATH=src python -m probe prepare` → `data/probe/requests.jsonl` (600 requests; sha256 `a8a4eed3…d946fca`).
2. Commit the code, pins and this record.
3. On a GPU machine, at that commit:
   - `pip install -r requirements-gpu.txt`
   - copy `data/probe/requests.jsonl` over (`data/` is git-ignored)
   - `PYTHONPATH=src python -m probe.run_vllm --requests data/probe/requests.jsonl --out data/probe --check-only`, then again without `--check-only`
4. Copy `generations.jsonl` and `run_meta.json` back to `data/probe/`, then run `PYTHONPATH=src python -m probe score`.
5. Record the outcome below; set `pinned.VLLM_VERSION` from `run_meta.json`.

## Outcome

_Pending the GPU run._

| | |
|---|---|
| Outcome | |
| CWE exact: observed / control / margin / p | |
| CVSS: observed / control / margin / p | |
| Parse rates (lenient / strict) | |
| vLLM version, GPU | |
