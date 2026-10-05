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

From `data/probe/report.md` and `report.json`. Generations sha256 `78788b6e…fe430d06`, from requests `a8a4eed3…d946fca`. The pins were committed at `725eeb8` (2026-09-26 22:02 UTC), before the GPU run started (2026-09-29 00:09 UTC).

| | |
|---|---|
| Outcome | **Not detected.** Memorisation was probed and not detected; proceed unchanged. |
| CWE exact: observed / control / margin / p | 0.020 / 0.032 / **−1.2 pts** (95% CI −2.6, 0.2) / 0.956 |
| CVSS: observed / control / margin / p | 0.515 / 0.498 / **+1.7 pts** (95% CI 0.5, 2.9) / 0.005 |
| CWE hierarchy (secondary): observed / control / margin / p | 0.027 / 0.041 / −1.5 pts (95% CI −2.9, −0.1) / 0.967 |
| vs the plan's constants | CWE exact −9.7 pts vs CWE-125 (0.117); CVSS −18.3 pts vs the majority vector (0.698); hierarchy −13.8 pts vs CWE-125 (0.165) |
| Parse rates (lenient / strict) | 1.00 / 1.00 on both questions; the strict margins equal the lenient ones |
| Recalled CVEs (`recalled.jsonl`) | 6 |
| vLLM version, GPU | vLLM 0.30.0 (torch 2.13.0+cu130, transformers 5.17.0, CUDA 13.0, Python 3.13.11); 4 × NVIDIA A40 |
| Run | 600 / 600 finished on a stop token; none hit the 512-token cap. Load 666.5 s, generation 6.4 s |
| Determinism recheck | 2 of 20 re-generated outputs differed: `CVE-2018-25033:cvss`, `CVE-2020-19860:cwe` |

### Reading

- **CVSS is significant but immaterial.** The +1.7-point margin clears p < 0.05 but not the 5-point bar, so the rule gives "not detected". Most of the 0.515 is the model's habitual vector matching common component values, which the control absorbs.
- **The CWE answers are a habit, not recall.** Across 300 CVEs the model gives only 5 distinct CWEs: CWE-78 × 185, CWE-787 × 57, CWE-79 × 51, CWE-789 × 5, CWE-665 × 2.
- **The 6 "recalled" CVEs are chance hits of a habitual answer.** All 6 are CWE-787 answers on CWE-787 gold (CVE-2021-21704, CVE-2021-28021, CVE-2022-27666, CVE-2022-30292, CVE-2022-36041, CVE-2022-37434). The pinned definition excludes only the single most common answer (CWE-78), so CWE-787, the second most common, counts as non-habitual. Answering CWE-787 57 times against 51 CWE-787 golds would hit ≈ 9.7 times by chance, and it hit 6. The file is recorded as specified. It has no use at this outcome, since the sensitivity run applies only to a material result. If the probe is ever re-run, the definition is too loose to identify recall.
- **The determinism mismatches don't affect the outcome.** Under greedy decoding, the recheck is expected to agree. Two disagreements are consistent with batch-dependent floating-point differences. They cannot move a margin this far from the thresholds. Step 7's engine-agreement check should expect this level of run-to-run variation.

## Open for later steps

- **`watermarking=True` in the recorded `SamplingParams`.** The runner doesn't set it, and it isn't pinned. Before `pinned.VLLM_VERSION = "0.30.0"` is fixed for every later evaluation, confirm what this does in vLLM 0.30.0 and whether it alters greedy token choice. If it does, disable it explicitly in `EVAL_SAMPLING` and record the change here. It may also be behind the determinism mismatches.
- **`pinned.VLLM_VERSION`** is still `None`. Set it once the watermarking question is settled.

*Step 5:* `pinned.VLLM_VERSION` is now `"0.30.0"`. Every later runner sets `watermarking=False` explicitly wherever the field exists, rather than relying on its default; `frozen_model.run_vllm --check-only` prints the vLLM source that defines it, recorded in `step5_decision_record.md`. The probe's own outputs stand as generated.

*Resolved at step 5:* `watermarking` is a per-request switch for a watermark configured on the engine. The engine-level defaults in vLLM 0.30.0 are `watermark = 0.0` and `watermark_config = None`, and the probe passed neither, so no watermark was configured and the probe's outputs are plain greedy decoding. Watermarking is **not** behind the two determinism mismatches; those are batch-dependent floating-point differences (step 5 saw 18 of 20 on 512-token rationales). Details in `step5_decision_record.md`, Outcome.

*Step 7:* the parser re-tuning promised above runs at step 7, on the untrained backbone's non-test replies (the step-5 audit and the engine check), never on probe replies. If it changes a parser (v2), the probe is re-scored and both versions are reported. Outcome: `step7_decision_record.md`.
