# Step 7 decision record — engine agreement, parser freeze, primary-test freeze

Pre-registration for procedure step 7:
- the engine-agreement check (HuggingFace versus vLLM);
- the parser review and freeze;
- the freeze of the primary test and the minimum-detectable-effect (MDE) simulation.

The pins are implemented in:
- `src/etl/pinned.py` (the "Engine agreement (step 7)" and "Primary test and minimum detectable effect" sections);
- `src/evaluate/run_vllm.py` (the evaluation runner, reused at step 12);
- `src/engine_check/` (the sample, the HF reference, the comparison and the parser review);
- `src/analysis/` (the primary test and the MDE).

**They must be committed before the GPU run**, so the rules are verifiably fixed before any output exists. Both GPU runners refuse a dirty tree. "User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## What this step covers

- **Engine agreement.** Do HF and vLLM, the evaluation engine, score the same model the same? This is checked before any trained model exists, so a setup bug (chat template, stop tokens, hidden sampling defaults) is found before results exist.
- **Parsers.** They are reviewed on the untrained backbone's non-test replies and frozen before step 10, because they also compute the GRPO reward.
- **Primary test and MDE.** Written, tested on synthetic data and frozen by hash, before step 9 writes any training file.

## Owner decisions

| Question | Decision | Source |
|---|---|---|
| Which checkpoint and prompts | **The raw backbone, now.** No trained checkpoint exists, and dev is drawn at step 8, so the prompts come from the late window of the non-test pool, the plan's pool for dev. Test is never read. | user |
| LoRA at evaluation | Each LoRA adapter is **merged into the base weights** before vLLM serves it (`pinned.EVAL_LORA = "merged"`), so evaluation runs the code path checked here. | user |
| Material disagreement | On any type, \|HF − vLLM A\| ≥ 1 point **and** the interval excludes 0. | user |
| Interval level | **99% per type**, so the false-alarm rate over five types stays near 5% (Bonferroni). Five 95% checks would stop the pipeline on pure noise about 23% of the time. | user (accepted with the plan) |
| Parsers | **Review now, then freeze.** The owner approves any change, which makes the parsers v2. | user |
| MDE | **Frozen now, with a single-cell effect**: one arm δ better on one type, reported per type and overall. | user |
| Primary test's null | **A parametric seed bootstrap**, replacing the plan's item-level permutation (below). | user |
| More seeds instead | Considered and rejected: they do not fix the permutation null (below). 50 seeds would mean about 1,570 training runs instead of 114. | user |

## Engine-agreement check

| Pin | Value | Source |
|---|---|---|
| Late window | The latest ⌈20% × 1,933⌉ = **387** non-test CVEs by (`published`, `cve_id`), published 2020-01-22 → 2022-05-08. Step 8 may re-pin the late window (step 2's open item). This sample stays as drawn. | doc + default |
| Sample | The **200** window CVEs with the lowest `stable_rank(cve_id, "engine-agreement")`, all six items each: **1,200 prompts**, 200 per type, with find-the-error as 200 pairs. The prompts are the bank's byte-identical `prompt`. | default |
| Decoding | `EVAL_SAMPLING`: greedy, 512 tokens, seed 0, stop on `EVAL_STOP_TOKENS`, bf16. | doc |
| vLLM | `evaluate.run_vllm` at `VLLM_VERSION` 0.30.0. **Pass A**, bank order, **decides**. **Pass B**, the same rows in a seeded shuffle (`stable_rank(request_id, "engine-agreement", "b")`), is vLLM's own batch noise and is reference only. One file per invocation, so each pass gets a fresh engine. Settings: `generation_config="vllm"`, `watermarking=False`, `max_model_len = EVAL_MAX_MODEL_LEN` (16,985). Every prompt's token ids are checked against the pinned tokenizer. Engine settings are recorded. | default |
| HF reference | `engine_check.run_hf`: the pinned backbone, bf16, `attn_implementation="sdpa"`, **batch size 1** (no padding), an explicit attention mask.<br>• `generation_config` is built from scratch (`do_sample=False`, `repetition_penalty=1.0`, both stop ids, `max_new_tokens=512`) and `use_model_defaults=False` is passed where `generate` accepts it. Qwen's own `generation_config.json` would add `repetition_penalty=1.05` even under greedy.<br>• **Every generated token is asserted to be the argmax of the raw logits**, which no hidden logits processor could survive.<br>• Every prompt's length is asserted equal to the bank's and vLLM's count.<br>• Four shards run in parallel, one per GPU, each resumable. A 5-request determinism recheck runs per shard. | default |
| Near-tie diagnostic | Where HF's tokens leave pass A's, the row records the first divergence: its index, both tokens, HF's logit margin between them, and the rank of vLLM's token under HF's logits. Rank 1, HF's second choice, is a near-tie flip. HF's logits at that step are the teacher-forced logits for vLLM's prefix, since the sequences agree up to it. Reported, not decided on. | default |
| Score unit | One per CVE per type: the item's `metric` for MCQ, exact-ID, CVSS and line localisation, and `paired_accuracy` (both labels right) for find-the-error. | doc |
| Interval | A paired percentile bootstrap over CVEs of the mean HF − vLLM difference: **10,000** resamples, seed 0, 99%. | default |
| Decision | **Material** if on any type \|gap\| ≥ 1 point and the 99% interval excludes 0; otherwise **agree**. The lenient parsers decide, and strict parse rates are reported (as at step 3). A material outcome stops the pipeline before step 8. The first diagnostic is a pass at `max_num_seqs=1`, to separate kernel effects from batching; then template, stops, generation config and dtype are checked. Fix, re-run and record. | user |
| Also reported | Per engine and type: parse rate (lenient and strict), cut-off rate and mean output tokens. Per pair: identical-text rate and which engine was better on discordant CVEs. The first-divergence index and margin distributions. Text that is not the decoding of its own tokens. Each run's determinism recheck. | default |
| Request files | `data/engine_check/vllm_a_requests.jsonl` sha256 `47d34da8…e196de36`; `vllm_b_requests.jsonl` `9081bc79…ad54d696`. Both built from `bank_nontest.jsonl` `4d03cc38…0a327a4d`. A rebuild under a different `PYTHONHASHSEED` is byte-identical. | — |

## Parser review

`python -m engine_check parser-review` → `data/engine_check/parser_review.{json,md}`.

**Sources** are the untrained backbone's non-test replies only:
- the step-5 audit (8,000 rollouts, sampled, 512 tokens);
- vLLM pass A and HF (1,200 greedy each).

Probe replies are never used, as step 3 forbids.

| Category | Meaning |
|---|---|
| `parse_failure` | The lenient parser found no answer in a reply that was not cut off. Every one is listed in the JSON. |
| `no_field` | An answer was found, but a fallback decided: there was no `ANSWER:`, `VULNERABLE:` or `LINES:` field, the CVSS vector was incomplete, or the exact-ID answer was not on the last line. |
| `lenient_not_strict` | Parsed, but not in the exact target format. This is expected when the model reasons first. 10 per source and type are sampled by `stable_rank(…, "parser-review")`. |

**Rule for a change:** a parser is changed only where a human reader would accept the stated answer without doubt and the parser missed or misread it. The owner approves each change.

**If anything changes:**
- `PARSER_VERSION` becomes `"v2"`, with a unit test per case;
- the probe is re-scored, and both versions are reported;
- the step-5 and step-6 decisions stand as made under v1, with re-scored figures reported descriptively;
- these guards are re-run: the bank `check` (every target still scores 1 strictly), every `dpo.jsonl` rejected answer still strict-valid with dense < 1, and every `distill_self` target's last line still parsing to gold;
- the engine comparison is re-run, and its decision is recorded under the final parser, and under both if they differ.

## Primary test

### The statistic: the plan's, unchanged

- `y[a,t,s]` is run (a, t, s)'s mean score over the test CVEs: arm a trained without type t, seed s, scored on t. Find-the-error's two items collapse to one paired score per CVE.
- `sd_t` is the pooled seed-level SD: sqrt(mean over the tested arms of the ddof-1 variance across seeds of `y[a,t,·]`). For 4 arms × 3 seeds that is 8 degrees of freedom.
- `m = mean_s y / sd_t`, and `T = Σ (P m)²`, where P removes the row, column and grand means.
- T depends only on cell means, so the plan's item-level residualisation gives the same T. A test asserts this.

### The null: changed from the plan (owner)

**The flaw.** The plan's null shuffles arm labels within each test CVE. That models test-question noise only. Training-run noise, the seed-to-seed shift of an arm's score on a type, survives the three-seed average and reads as interaction.

The evidence comes from simulated M2 experiments with **no interaction**, at the real size: 343 CVEs, 4 arms, 5 types, 3 seeds, binary and partial-credit items, and shared item difficulty. Training noise is the SD of a run's mean around its cell's true mean.

| Training noise per cell | Plan's item permutation rejects (p < 0.05) | Seed bootstrap rejects |
|---|---|---|
| 0 points | 2 / 40 | 2 / 40 |
| 1 point | 4 / 40 | 1 / 40 |
| 2 points | **19 / 40** | 2 / 40 |
| 4 points | **40 / 40** | 3 / 40 |

A correct 5% test rejects about 2 of 40. More seeds do not fix the permutation null, because they shrink the noise it measures as fast as the noise it misses: with 10 seeds it rejected 23 / 30 at 2 points, and with 30 seeds 10 / 20.

**Two alternatives were checked and not adopted:**
- An F-test on the same T, using the seeds as replicates, rejected 6, 3, 9 and 10 of 40. The SDs estimated with 8 degrees of freedom inflate it.
- The permutation null with seed noise added was too strict at low noise (0 / 40) and too loose at high noise (8 / 40).

**The pinned null.** Under "no interaction", each cell's expected run mean is the additive fit (grand + arm + type effects) to m, times `sd_t`.
- Each of **10,000** replicates (`random.Random(0).gauss`, in arm, type, seed order) draws all runs as that mean plus `sd_t` × N(0, 1).
- It **re-estimates every `sd_t`** and recomputes T\*.
- p = (1 + #{T\* ≥ T·(1 − 10⁻¹²)}) / 10,001; significant at p < 0.05.

The unit tests confirm about 5% rejection at 0, 2 and 4 points of training noise, and with an arm main effect of 15 points.

**What it treats as fixed**, stated in the write-up:
- the test CVEs, so the conclusion is about this test set;
- run-to-run noise as normal, with one SD per column across arms. An arm whose noise differs greatly from the others, such as a collapsed GRPO, can bend it.

| Pin | Value | Source |
|---|---|---|
| Arms | `PRIMARY_ARMS` = sft, distill_self, dpo, grpo. Three arms if distill-self leaves the test; it did not at step 5. | doc |
| Input | M2 rows `{arm, seed, item_id, metric, dropped_type}`. Only the dropped type's items enter. Missing or duplicate scores fail. | default |
| Column with `sd_t` = 0 | Excluded from T and reported. There is no floor for a tiny `sd_t`; the per-cell contributions expose one. | default |
| Reported alongside | Run means, cell means, interaction residuals, per-cell contributions to T, and flagged cells. The **leave-flagged-columns-out** sensitivity run drops every column with a floored or substituted cell. An exclude-CVEs list is available but unused, since the probe found nothing. | doc |
| Ties | `T* ≥ T·(1 − PRIMARY_TIE_RTOL)`, with `PRIMARY_TIE_RTOL` = 10⁻¹². | default |
| Arithmetic | Stdlib, float64, fixed summation order: reproducible and needing no numpy. At M2 size the test takes about 0.5 s. | default |

## MDE

| Pin | Value | Source |
|---|---|---|
| Experiments | **1,000**, drawn from the primary test's null (the additive fit plus seed noise at the measured SDs), with `random.Random(1)`. | default |
| Planted effect | δ points added to every run of one (arm, type) cell, **on each arm in turn**. The planted interaction is (1 − 1/arms)(1 − 1/types) of δ, which is 0.6δ for four arms and five types. Each type's headroom below 100 points is reported. | user |
| Exactness | δ shifts a cell's seed mean and leaves the seed SDs unchanged, so T(δ) = a + 2bx + cx² with x = δ / (100 `sd_t`), exactly. Each experiment is read against the frozen test's 10,000 null replicates. | default |
| Power and MDE | Over the grid 0, ¼, …, 50 points, power is the share of the 4,000 trials with p < 0.05. **MDE = the smallest δ from which power stays ≥ 80%**, or "> 50 points". Reported per type, and overall from the type-averaged power curve. About 2 s at M2 size. | default |

## Freeze

`pinned.PRIMARY_TEST_SHA256` = **`15e75eb7…df345a207`**, which is `analysis.source_sha256()`: the sha256 of `src/analysis/primary_test.py` and `mde.py`. It is set in the pre-registration commit, so the test is frozen before any step-7 GPU output exists.
- `python -m analysis` refuses to run on any other source, and a unit test asserts the match.
- A change needs a new pin and an entry here.

## Runbook

1. Commit the code, the pins, this record and the plan-document edits.
2. `PYTHONPATH=src python -m engine_check prepare`. Already written; the rerun is byte-identical.
3. On the GPU machine, at that commit, with `pip install -r requirements-gpu.txt` and `data/engine_check/vllm_{a,b}_requests.jsonl` copied over:
   - `python -m evaluate.run_vllm --requests data/engine_check/vllm_a_requests.jsonl --check-only`, then without `--check-only`;
   - the same for `vllm_b_requests.jsonl`, as a separate invocation;
   - `python -m engine_check.run_hf --shard 0 --shards 4 --check-only`, then, in parallel, `CUDA_VISIBLE_DEVICES=k python -m engine_check.run_hf --shard k --shards 4` for k = 0..3.
4. Copy back `vllm_{a,b}_generations.jsonl`, `vllm_{a,b}_run_meta.json` and `hf_shard*of4_*`. Then run `python -m engine_check compare` and `python -m engine_check parser-review`.
5. The owner reviews `parser_review.md`. If a change is approved, implement v2 and run its guards (above), then re-run `compare`.
6. Record the outcome below.

If a run is interrupted, rerun the same command: only unfinished requests are generated.

## Outcome

*To be filled in from `data/engine_check/engine_report.md`, `parser_review.md` and the run metas.*

| | |
|---|---|
| Engine agreement (outcome; gaps and intervals per type) | |
| vLLM A vs B (batch noise) | |
| Divergences (count; near-tie share; margins) | |
| Runs (versions, settings, token-id checks, argmax check, determinism) | |
| Parser review (failures per type; changes; version frozen) | |
| Freeze (commit; `PRIMARY_TEST_SHA256`) | |
| sha256s | |

## For later steps

- **Step 8.** Pin the late-window size (step 2's open item: the plan's rule gives about 193 dev CVEs, not about 341) and the fixed 150-CVE checkpoint-selection subsample.
- **Step 10.** The LR sweep runs at seed 0 only, so the "pooled seed SD" its column-standardised selection asks for does not exist then. That needs a pin.
- **Steps 11–12.**
  - Merge each LoRA adapter in fp32 and cast to bf16, then evaluate through `evaluate.run_vllm` (a model-path argument is added then).
  - The MCQ log-likelihood column is engine-dependent and its scoring is not yet defined.
  - The plan's "second decoding configuration" for the robustness check is not yet pinned.
- **Step 11, seeds.** The seed bootstrap gains power from more seeds, because the SDs are better estimated. 5 seeds instead of 3 adds about 60 runs. That is a budget decision, independent of this freeze.
- **Step 19.** `python -m analysis primary-test` and `python -m analysis mde` on the M2 scores, with `--seeds` (the seeds trained) and `--flagged` for floored or substituted cells. The scores must cover exactly the test pool and those seeds.

## Differences from the experiment plan (now reflected in its text)

- The engine check runs on the raw backbone over late-window non-test prompts, not on a trained checkpoint over dev, because neither exists yet.
- The primary test keeps the plan's statistic and replaces its item-level permutation null with a parametric seed bootstrap.
- The MDE draws its experiments from that null and plants the effect on each arm in turn.
- Evaluation serves merged LoRA checkpoints.
