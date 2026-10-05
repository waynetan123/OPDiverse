# Step 5 decision record — frozen-model session

Pre-registration for procedure step 5: the GRPO signal audit and distill-self rationale generation, on the untrained backbone. The pins are implemented in:
- `src/etl/pinned.py` (the "Frozen-model session" section);
- `src/frozen_model/` (requests, the GPU runner, scoring).

**They must be committed before the GPU run**, so the rules are verifiably fixed before any output exists. The runner refuses a dirty tree. "User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## What this step does and does not cover

- **GRPO audit.** Do the base model's 8 rollouts on one prompt disagree? GRPO learns only from groups whose rewards are not all equal ("live" groups).
- **distill-self.** One hint-conditioned rationale per non-test item (11,598), cached as `distill_self.jsonl`, keyed by `item_id`. The substitution trigger is evaluated per type.
- **Not here.** The DPO line of the audit (near-miss constructibility) and the distill-external line need the step-6 external outputs, so they are computed at step 6.

One engine load serves the audit and rationale attempt 1. The one regeneration of invalid rationales is a second, smaller run, as the plan says.

## Pins

| Pin | Value | Source |
|---|---|---|
| Rollout sampling | `ROLLOUT_SAMPLING`: temperature 1.0, top-p 1.0, repetition penalty 1.0, presence and frequency penalties 0. These are TRL `GRPOConfig`'s defaults, so the audit and GRPO training share one config. n = 8. Per-request seed `stable_rank(item_id, "signal-audit") mod 2³¹`. | user |
| Rollout caps | **A rule, replacing the plan's fixed caps.** The plan's caps (MCQ 16 / exact-ID 24 / CVSS 48 / find-error 64 / line-loc 64) assumed terse replies. The step-4 templates say "End your reply with …", which invites reasoning first, and a reasoning-first reply cut off at 16 tokens scores 0. If all 8 rollouts are cut off, the group is dead, and a length setting would floor a column. **Rule:** per type, the GRPO cap is the smallest of {plan cap, 128, 256, 512} at which at most **10%** of audit rollouts are cut off, else 512. | user |
| How the caps are measured | The audit samples once at 512 tokens. A cap-k rollout is the first k tokens of that sample: same seed, same tokens. The stop token counts toward k, as in vLLM. Each sample is scored at every candidate cap, so one run gives the whole cap-to-dead-groups curve. | default |
| Audit sample | The 200 non-test CVEs with the lowest `stable_rank(cve_id, "signal-audit")`. It takes their mcq, exact_id, cvss and line_loc items, and both find_error items of the lowest-ranked 100. That is 200 prompts per type and 1,000 requests, with balanced classes on find_error. | default |
| Audit scores | Dense is the GRPO reward, `Verdict.dense`, which is 0 on a parse failure. Binary is `metric == 1`. **The band is read on dense** at the chosen cap: ≥ 50% live trains as specified, below 10% is floored, and in between enables dynamic sampling. Binary is reported alongside. | default |
| Audit diagnostics | Per type and per cap: parse rate, cut-off rate, live groups (dense and binary), mean reward. Also groups whose 8 texts are identical (a seeding check), find-the-error predicted-vulnerable rate per class, line-localisation predicted-set size, and re-decode mismatches against vLLM's text. | default |
| Rationale prompt | `RATIONALE_PROMPT`: the item's own user message, then "The correct answer is: {target}". It asks for step-by-step reasoning "as if you were working it out yourself", without saying or suggesting the answer was given, with the answer alone on the last line, exactly as `{target}`. **The training prompt stays the item's byte-identical `prompt`**; the hint appears only at generation. | default |
| Rationale decoding | Attempt 1 is greedy (`EVAL_SAMPLING`, 512 tokens): the model's single most likely rationale. The regeneration samples under `ROLLOUT_SAMPLING`, n = 1, 512 tokens, seed `stable_rank(item_id, "distill-self", "2")`, because greedy would only repeat the failed output. | user |
| Surface validity | All of these must hold, checked in this order, with the first failure logged as the reason:<br>1. it stopped on a stop token (`truncated`);<br>2. it has a last non-empty line (`empty`);<br>3. that line parses under the item's lenient verifier (`no_final_answer`);<br>4. the parsed value is exactly the gold's (`wrong_final_answer`). Exact, not ±1, so "LINES: 12, 14" for gold {12, 13} fails;<br>5. there is reasoning above that line (`no_reasoning`);<br>6. the reasoning matches none of `HINT_LEAK` (`hint_leak`): "we were told/given", "given/provided answer", "the answer was given", "the hint", "the prompt says … answer". | default |
| Training target | The reasoning, a blank line, then the canonical target. The last line is replaced by its canonical form, so `**ANSWER: B**` becomes `ANSWER: B`. Fallback after a failed regeneration is the gold target alone. Each row records `source` (`self_a1` / `self_a2` / `gold_only`) and the reasons. | default |
| Substitution trigger | Per type, on the **first-attempt** pass rate: it fires if the pass rate is below 50% or the fallback rate is above 50%. Under this reading the fallback clause never fires on its own, because fallback ≤ 1 − pass rate. Example: 45 of 100 valid first time and 30 of 55 recovered gives a fallback rate of 25%, but the trigger fires on the 45%. Exactly 50% does not fire. **If it fires on 3 or more types, distill-self leaves the primary test** (run it on SFT, DPO, GRPO). | user |
| distill-self band | The plan's ≥ 50% / 10–50% / < 10% bands, on the first-attempt pass rate, reported per type. | doc |
| vLLM | **`VLLM_VERSION = "0.30.0"`**, from the probe's `run_meta.json`. The runner refuses any other version. | default |
| Watermarking | vLLM 0.30.0 printed `watermarking=True` in the probe's `SamplingParams`. The field is not pinned and may bias token choice, which would distort sampled rollouts. **The runner sets `watermarking=False` explicitly** wherever the field exists. `--check-only` prints the vLLM source lines that mention it, to be recorded under Outcome. | default |
| Stop tokens | `EVAL_STOP_TOKENS` only, with no stop strings, since a stop string would cut reasoning-first replies short. | default |
| Session `max_model_len` | The longest prompt plus its own `max_tokens`: **17,059** on the real files (longest rationale prompt 16,547 tokens + 512). It sets capacity only and changes no output. `EVAL_MAX_MODEL_LEN` (16,985) is unchanged. | default |
| Checkpointing | The runner generates in chunks of 1,000 and appends each chunk to `<prefix>_generations.partial.jsonl`. An interrupted run resumes without regenerating finished requests. Rows keep every sample's text, finish reason and token ids (the audit cuts by token). | doc + default |
| Request files | `data/frozen_model/audit_requests.jsonl`, 1,000 requests, sha256 `6005153a…5e8f09f2`. `rationale_requests.jsonl`, 11,598 requests, sha256 `7b75e4b1…5e022a3e`. Both built from `bank_nontest.jsonl` `4d03cc38…0a327a4d`. A rebuild under a different `PYTHONHASHSEED` is byte-identical. | — |

## Runbook

1. Commit the code, the pins, this record and the plan-document edits.
2. `PYTHONPATH=src python -m frozen_model prepare` (already written; the rerun is byte-identical).
3. On the GPU machine, at that commit:
   - `pip install -r requirements-gpu.txt`;
   - copy `data/frozen_model/*_requests.jsonl` over (`data/` is git-ignored);
   - `PYTHONPATH=src python -m frozen_model.run_vllm --requests data/frozen_model/audit_requests.jsonl --requests data/frozen_model/rationale_requests.jsonl --check-only`, and record the watermarking lines it prints;
   - run the same command again without `--check-only`.
4. Copy `*_generations.jsonl` and `*_run_meta.json` back, then run `PYTHONPATH=src python -m frozen_model prepare-retry`.
5. On the GPU machine: `python -m frozen_model.run_vllm --requests data/frozen_model/rationale_retry_requests.jsonl`. If the retry file is empty, skip this step and delete the empty file.
6. Copy the files back, then run `python -m frozen_model audit`, `rationales` and `report`. Record the outcome below from `step5_report.md`, `audit_report.md` and `rationale_report.md`.

If a run is interrupted, rerun the same command: only unfinished requests are generated.

## Outcome

From `data/frozen_model/step5_report.md`, `audit_report.md`, `rationale_report.md` and the three run metas. The pins were committed at `066d597` (2026-10-02 00:39 UTC) and merged to `main` as `9dc2bf9` (00:49 UTC), with identical `src/` and `tests/`. All three GPU runs ran at clean `9dc2bf9`, the first starting 2026-10-02 09:13 UTC.

| | |
|---|---|
| Watermarking in vLLM 0.30.0 | **Settled: no watermark was applied to any output, in step 3 or step 5.** `SamplingParams.watermarking: bool = True` is documented in `vllm/sampling_params.py` line 257 as *"Whether to apply the engine's configured watermark to this request."* It is a per-request switch for a watermark configured on the engine, not a watermark in itself. The engine-level settings in `EngineArgs` default to `watermark = 0.0` and `watermark_config = None` (checked on the GPU machine, vLLM 0.30.0). Neither the probe nor any step-5 run passed either setting: the recorded `llm_kwargs` are only model, revision, tokenizer revision, dtype, seed, `max_model_len` and `generation_config`. So no watermark was configured, and the probe's switch-on had nothing to apply; its outputs are plain greedy decoding. Every step-5 run also set the switch to `False`. Reading `watermark = 0.0` as the strength of an unconfigured watermark is an inference from the default values, not from vLLM documentation. **Pin for later runs:** keep `watermarking=False` per request and pass neither engine setting. |
| GRPO caps and bands per type | See the audit table below. Two types are banded **dynamic sampling** (MCQ, exact-ID) and three **train as specified** (CVSS, find-the-error, line localisation). **No type is floored.** |
| distill-self pass / fallback / hint-leak rates per type | See the rationale table below. Of 11,598 targets: 9,308 from attempt 1, 953 recovered by the regeneration, **1,337 gold only (11.5%)**. Every type is in the ≥ 50% band. |
| Substituted types; distill-self in the primary test? | **None substituted; distill-self stays in the primary test** as the fourth arm (`substitution.json`: `"substituted_types": []`, `"drop_distill_self_from_primary_test": false`). The closest type is line localisation, at 56.5% against the 50% bar. |
| Run (GPUs, timing, finish reasons) | 4 × NVIDIA A40; vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0, CUDA 13.0, Python 3.13.11; `max_model_len` 17,059. **Audit:** 1,000 requests × 8 = 8,000 rollouts in 840 s (mean 122 output tokens; 7,770 stop / 230 length). **Rationales, attempt 1:** 11,598 in 3,979 s (mean 419 tokens; 9,470 stop / 2,128 length). **Regeneration:** 2,290 in 1,236 s (mean 482 tokens; 1,008 stop / 1,282 length). Engine load 46 s, then 27 s. Re-decoded audit samples match vLLM's text exactly (0 mismatches). |
| Determinism recheck | **18 of 20** greedy rationale-1 requests produced different text when re-generated in a 20-request batch. The probe gave 2 of 20, but its answers were a few tokens long. A 512-token rationale diverges for good after one batch-dependent floating-point flip. **`rationale_generations.jsonl` is the artifact and cannot be regenerated identically, even greedily.** The audit and regeneration are sampled and have no greedy recheck. |
| sha256s | Every request, generation and output file is listed in the sha256 table below. |

### GRPO signal audit

| Type | Plan cap | GRPO cap | Cut off at that cap | Parsed | Live groups (dense) | Live groups (binary) | Mean reward | Band |
|---|---|---|---|---|---|---|---|---|
| MCQ | 16 | **16** | 2.9% | 93.2% | 25.0% | 25.0% | 0.664 | dynamic sampling |
| Exact-ID | 24 | **24** | 0.6% | 100.0% | 49.5% | 28.0% | 0.383 | dynamic sampling |
| CVSS | 48 | **256** | 1.2% | 96.7% | 97.0% | 15.0% | 0.592 | as specified |
| Find-the-error | 64 | **512** | 0.6% | 99.9% | 69.5% | 66.0% | 0.409 | as specified |
| Line localisation | 64 | **512** | **13.8%** | 98.8% | 76.0% | 9.0% | 0.184 | as specified |

- **Line localisation's cap is the rule's fallback.** No candidate cap kept the cut-off at or below 10% (64: 99.9%, 128: 97.1%, 256: 62.7%, 512: 13.8%), so the rule gives 512. GRPO trains this column with about one rollout in seven cut off, and a cut-off rollout scores 0 unless its truncated text already parses.
- **The plan's caps would have been wrong for three types.** At the plan cap, CVSS cuts off 54.7%, find-the-error 99.4% and line localisation 99.9% of rollouts. The base model reasons before answering on these types.
- **Exact-ID misses "as specified" by one group:** 99 of 200 live, against 100 needed.
- **MCQ's dead groups are mostly all-correct.** The mean reward is 0.664, and 31% of groups are 8 identical texts at temperature 1.0. CVSS, find-the-error and line localisation have no identical groups, so this is the model's confidence, not a seeding fault.
- **Find-the-error does not separate the classes.** The model calls 38.1% of vulnerable and 39.8% of patched rollouts vulnerable. Groups are live, but the base model has no class signal to start from. The collapse diagnostics at step 11 matter here.
- **Line localisation:** predicted set sizes at the 512 cap are mostly 1–4 lines (1,190 of 1,581 parsed); 222 predict 10 or more.

### distill-self rationales

| Type | Items | Valid at attempt 1 | Recovered by regeneration | Gold only | Restates hint (attempt 1) | Substitution |
|---|---|---|---|---|---|---|
| MCQ | 1,933 | 1,869 (96.7%) | 39 | 25 (1.3%) | 0.1% | no |
| Exact-ID | 1,933 | 1,897 (98.1%) | 35 | 1 (0.1%) | 0.5% | no |
| CVSS | 1,933 | 1,666 (86.2%) | 158 | 109 (5.6%) | 0.9% | no |
| Find-the-error | 3,866 | 2,783 (72.0%) | 419 | 664 (17.2%) | 0.1% | no |
| Line localisation | 1,933 | 1,093 (56.5%) | 302 | 538 (27.8%) | 1.1% | no |

- **Almost every failure is a cut-off reply, not wrong reasoning.** At attempt 1, truncation accounts for 1,048 of 1,083 find-the-error failures and 822 of 840 line-localisation failures. Wrong final answers are rare on every type (3–51 per type). These are the two types whose prompts carry the whole function.
- **Two columns are partly bare gold.** 27.8% of line-localisation and 17.2% of find-the-error distill-self targets carry no reasoning. On those items the arm trains like SFT. Report this beside every distill-self cell in those columns; it is not substitution and is not flagged in the primary test.
- **The regeneration recovered little where it mattered,** because 56% of its replies were cut off as well (1,282 of 2,290).

### sha256s

| File | sha256 |
|---|---|
| `audit_requests.jsonl` | `6005153a…5e8f09f2` |
| `audit_generations.jsonl` | `2488bdc1…433896d7` |
| `audit_scores.jsonl` | `22f67c82…03a5a745` |
| `rationale_requests.jsonl` | `7b75e4b1…5e022a3e` |
| `rationale_generations.jsonl` | `a9511a46…9991babe` |
| `rationale_retry_requests.jsonl` | `a4e93520…372f437d` |
| `rationale_retry_generations.jsonl` | `d4a16298…11241118` |
| `distill_self.jsonl` | `45b7509d…5633cc87` |
| `substitution.json` | `d40db52b…18b26d13` |

## For later steps

- **Step 6.** No type was substituted, so step 6 generates **no** distill-self substitution rationales. It produces the distill-external traces and the DPO rejected completions, then computes the DPO and distill-external lines of the audit.
- **Step 11, GRPO.** Use `ROLLOUT_SAMPLING` and the per-type caps recorded above, through TRL's vLLM integration at `VLLM_VERSION`, with watermarking off. Apply dynamic sampling on MCQ and exact-ID, the two types banded "dynamic sampling". No GRPO cell is floored at step 0; the running monitor can still floor one mid-training.
- **Converters (step 9).** distill-self reads `distill_self.jsonl` by `item_id`. The prompt comes from the bank, and the `target` is the rationale target.

## Differences from the experiment plan (now reflected in its text)

- GRPO rollout caps are set by a pre-registered rule measured at the audit, not fixed at 16/24/48/64/64.
- The rollout sampling config is named: TRL's defaults, temperature 1.0 and top-p 1.0.
- The substitution trigger reads the first-attempt pass rate.
- The DPO and distill-external audit lines move to step 6, where their data is produced.
