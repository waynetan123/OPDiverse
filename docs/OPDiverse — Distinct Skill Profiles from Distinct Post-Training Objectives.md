## Goal

Test whether different post-training objectives produce teacher models with distinct and complementary skill profiles.

Four post-trained arms plus a control, with one additional arm (SFT→DPO) reported descriptively. A second, distill-external, was planned and dropped at step 6. All arms train on the same facts and the same prompts. What differs is the learning signal: imitate a target string (SFT), imitate a reasoning trace (distillation), prefer one completion over another (DPO), or maximise a verifiable reward (GRPO). If performance diverges across question types, that divergence is attributable to the objective, subject to the compute-matching caveat below.

The secondary question — and the one that makes the result non-trivial — is whether any observed advantage is a real transferable skill or just familiarity with an output format.

**What "identical facts" does and does not mean.** All arms see the same CVE set and the same prompt strings. Every arm's file covers the full fact set: no arm's converter can remove a CVE from any other arm's file. They do not see the same token budget: DPO sees two completions per prompt, distillation sees full targets, GRPO sees k rollouts per step. Tokens-seen and FLOPs are reported per arm. The token-matched secondary comparison is described in the budget section and runs only if the optional allocation is available.

**Backbone.** A Qwen-family 7B instruct checkpoint, named and version-pinned in-repo. One backbone for every arm.

**External model.** A single named external LLM, Claude Opus 5.5 (`claude-opus-5-5`, effort `medium`), was pinned for four jobs; three ran (distill-external's traces were refused at step 6):
- chooses the MCQ distractors;
- supplies distill-external's reasoning traces (refused at step 6; the arm is dropped);
- writes every DPO rejected completion except MCQ's, which is rule-defined (the distractor closest to gold in the hierarchy);
- stands as the substitution source for distill-self if its self-generated rationales fail the quality bar.

One model, one configuration, every external job, so that any finding about externally sourced data is a finding about one identified system rather than an unspecified mixture. It has no dated snapshot and accepts no sampling parameters. Its outputs are therefore cached and audited rather than regenerated. Pins are in `docs/decisions/step4_decision_record.md`.

## Architecture

```
NVD records ──┐
              ├─→ fact table ─→ chronological split ─→ question bank ─→ converters ─→ training files ─→ models
PrimeVul   ───┘   (join on        (test window drawn     (6 items per     (one per        (base/SFT/distill-self/
                   CVE ID)         first, on dates        fact, frozen)    objective)      DPO/GRPO + 2 secondary)
                                   alone)
```

**Pipeline ordering is load-bearing.** The bank is generated _after_ the test window is drawn and _before_ any per-seed partition. That ordering is what makes the generated artifacts — distill-self rationales and DPO negatives (and distill-external traces, as planned) — cacheable static files reusable across every seed and every matrix, which the compute budget depends on. An earlier draft generated the bank from "training and dev facts," which are per-seed quantities; that would have forced regeneration of ~20,000 items three times over and made the cached-file claim false. It also ran the signal audit and rationale generation _before_ the bank existed, which is simply unrunnable. Corrected below.

**Fact table.** One row per CVE, keyed on CVE ID, carrying description, CWE, sibling CWEs, CVSS v3.x base vector, vulnerable function, patched function, patch line set, publication date. Rows missing any field are dropped; an empty sibling list is a value, not a missing field. Functions come from PrimeVul v0.1's paired files. Where a CVE has several vulnerable functions, one is kept: the function named in the NVD description, else a fixed hash choice. Pins and outcomes are in `docs/decisions/step1_decision_record.md`.

The patch line set is derived by diffing the vulnerable function against its patched counterpart. Lines are compared on a normalised key (whitespace removed, comments masked), so reindentation and comment edits don't inflate the target. Blank and comment-only lines take no part in the diff, and the stored function keeps its original text. Insertions are attributed to the preceding code line, uniformly. Rows where the patch touches more than 20% of the function's code lines are dropped — a rewrite is not a localisation item. So are rows that insert more than 20% of that count, since insertion attribution would otherwise hide a rewrite behind one gold line.

**Function length cap.** Functions longer than 10,000 tokens under the backbone's tokenizer (the longer of the vulnerable and patched versions) are dropped at the fact-table stage, before any split is drawn. Applying the cap as a census filter makes it uniform by construction and visible in the filtering table, rather than a training-time truncation applied inconsistently across arms. Report N dropped. The cap was raised from 1,024 because 1,024 would have left 1,327 CVEs, below the reduce-scope line. The cost is that long functions (median ~800 tokens, p95 ~5,200) dominate the batches they land in, since attention cost is quadratic in sequence length. Length bucketing and the compute estimates below must account for that.

Line numbering presented to the model must match the numbering of the stored function, 1-indexed. It is also the diff's numbering, because normalisation maps lines one-to-one. Verify on a sample of 50 items before generating the bank; a silent off-by-N invalidates the entire column. The prompt states the convention explicitly, so numbering is a property of the task rather than something the model must infer.

**Filtering census.** Report N surviving each stage before any split is drawn: after pair integrity, after the NVD join, after field completeness, after CWE resolution, after CVSS v3.x pinning, after the empty-patch rule, after the 20% patch rule, after the insertion guard, after the length cap, after the one-function-per-CVE selection. Split percentages assume roughly 2500–4000 surviving CVEs; below 2500 apply the split fallback, and below roughly 1500 reduce scope (fewer arms or fewer types) rather than shaving splits further.

**Census gate on time range.** The CVSS v3.x requirement is not a uniform thinning — it is a cut at the old end of the range. CVSS v3.0 arrived in 2015 and NVD only assigned v3 vectors routinely from late 2015; everything older carries v2 and nothing else, so a large share of PrimeVul's BigVul-derived CVEs are dropped. That matters because the whole design rests on a chronological split producing a genuine distribution shift between train and test. Over a compressed window, train and test resemble each other, the M1→M2 gap shrinks, and the minimum detectable effect comes back large.

So the census reports the v3.x availability drop **as a publication-year histogram, before and after**, not as a single N, and a pre-registered rule applies:

- **Surviving range ≥ 5 years** — proceed unchanged.
- **Surviving range < 5 years** — choose, before any split is drawn, between (a) accepting it and stating in the limitations that the chronological shift is mild and gaps are expected to be smaller, or (b) extending the fact table forward with newer NVD entries carrying fix-commit links (the CVEfixes-style pipeline is open), restoring range length from the recent end instead of the old one. Recent years thin out because of the NVD analysis backlog, so (b) buys less than it looks like it should.

This is the same lever the contamination probe's escalation path uses; if both fire, extend once and serve both.

**v2 is not admitted to solve this.** v2 is a real format break — different components, different allowed values — and mixing it would make the eight-component scoring rule ill-defined.

## Contamination probe

Run after the test window is drawn and before the question bank is generated, on the raw backbone, no training of any kind. It needs the test-window CVE list and nothing else; it does not need the bank, which is why it sits here.

**Why.** A chronological split controls leakage inside the dataset, not against the backbone. PrimeVul's upstream sources end well before any current Qwen checkpoint's crawl, so the test window almost certainly sits inside pretraining. That overlap is not automatically harmless: objectives differ in how effectively they surface memorised facts — SFT teaching the answer format may unlock recall that GRPO instead sharpens — so contamination can manufacture an arm × question-type interaction, which is exactly the statistic under test.

**Why it is nonetheless likely to come back clean at 7B.** Memorisation scales steeply with parameter count. A 7B will have absorbed Log4Shell and Heartbleed, which appear on thousands of pages, but is unlikely to have memorised the CVSS vector of an obscure media-parser CVE appearing on three. Most of the fact table is the second kind.

**Procedure.** Sample 300 CVEs from the drawn test window. Prompt the raw backbone with **the CVE ID alone** — no description, no code, no options — for the CWE, then separately for the CVSS vector. Score with the same verifiers used in the main experiment. Compare against the most-frequent-CWE and majority-component-CVSS baselines computed on training-era facts. With nothing to reason from, any margin above those baselines is recall.

**Control.** A training-era constant is not enough on its own: the test window's label mix differs from the training era's (CWE-787 leads in 2021–22), so a model with no CVE-specific memory can beat the constant just by guessing the currently common label. On the pinned 300-CVE sample, always answering CWE-787 scores 0.170 exact against CWE-125's 0.117. Recall is therefore measured as the **margin over a permutation control**: each answer's score against its own CVE's gold, minus its mean score against every other CVE's gold. The constant-baseline comparison is reported alongside.

**Decision rule, fixed in advance.** This applies to either primary measure (CWE exact match; CVSS per-component agreement), with a one-sided permutation p-value (10,000 shuffles, seed 0). Details and pins are in `docs/decisions/step3_decision_record.md`.

- **At or near baseline** (margin < 5 points, or p ≥ 0.05) — record it, state in the write-up that memorisation was probed and not detected, proceed unchanged.
- **Materially above baseline** (margin ≥ 5 points with p < 0.05) — report the margin, and add a pre-registered sensitivity run of the primary test restricted to test CVEs the probe shows are not recalled. **Large** (margin ≥ 15 points with p < 0.05): reconsider the backbone or extend the fact table forward via newer NVD entries with fix-commit links, as under the census gate above.

Report the result either way.

## Generation infrastructure

Six workloads in this experiment generate text, and generation — not backpropagation — is the bulk of the compute. The engine choices below are pinned as frozen definitions, in the same class as the parsers and the hierarchy scorer, because a decoding change is a change to the measurement.

### Why vLLM

Generation has two phases. **Prefill** reads the prompt, processing all tokens in parallel — fast per token, and a shape GPUs handle well. **Decode** writes the answer one token at a time, each token requiring the full model weights to be re-read from memory; on an A40 that is bounded by ~696 GB/s of memory bandwidth, not by arithmetic.

This experiment's workload is prefill-heavy and decode-light. Half the items per CVE are long, because find-the-error and line localisation carry a whole C function (median ~900 and ~1,300 prompt tokens, up to 10,000 and 16,500). The label types carry only the description (~130–190 tokens). Answers are tiny (`LINES: 12, 13, 17` is about 15 tokens). That is the shape where naive generation wastes the most, and where vLLM's three mechanisms pay off hardest.

|Mechanism|What it fixes|
|---|---|
|**Prefix sharing via `n=k`**|One request for k completions prefills the shared prompt **once** and forks k branches, instead of re-reading the function k times|
|**PagedAttention**|KV cache allocated per token actually generated, in small pages — no padding across mixed-length functions, no memory reserved for `max_new_tokens` that is never used|
|**Continuous batching**|A finished sequence frees its slot immediately; no waiting for the slowest member of a batch|

For an 800-token function (roughly the fact-table median) generating 8 rollouts of ~15 tokens, prefix sharing alone removes roughly 86% of total token work, and more for longer functions. It is a genuine reduction in computation, not better scheduling.

### Pinned engine configuration

|Decision|Pin|
|---|---|
|Engine|vLLM, one specific version, recorded in-repo|
|Scope|**All** evaluation, for every arm and every matrix. Engines are never mixed|
|Rollout sampling|One sampling config used identically for GRPO training rollouts and for the signal audit: TRL's GRPO defaults, temperature 1.0, top-p 1.0|
|Rollout / audit caps|Per type, the smallest of {MCQ 16 / exact-ID 24 / CVSS 48 / find-the-error 64 / line localisation 64, 128, 256, 512} at which ≤ 10% of audit rollouts are cut off, measured by sampling the audit once at 512 tokens. Stop on end-of-sequence only. Pins in `docs/decisions/step5_decision_record.md`|
|**Evaluation caps**|**512 tokens, uniform across all arms and types**|
|External model|Claude Opus 5.5, effort `medium`, for MCQ distractors, DPO negatives, and the distill-self substitution source (distill-external traces: refused, arm dropped at step 6)|
|Evaluation checkpoints|Each LoRA adapter merged into the base weights, then served by `evaluate.run_vllm`, the runner the step-7 engine check went through|

**Why evaluation caps are uniform and generous.** Tight caps suit GRPO rollouts, where the model is being trained toward terse answers and every arm sees the same cap on the same untrained base — provided the base model's replies fit them. The bank's templates ask for the answer at the end of the reply, which invites reasoning first, so a fixed 16-token cap could cut off every rollout and floor a column for a length setting. The rollout cap per type is therefore set by the audit's cut-off rate under a pre-registered rule (step 5). They are wrong at evaluation, because distill-self is trained to reason before answering. A 24-token evaluation cap would truncate those arms while leaving SFT intact — an efficiency setting that silently penalises specific arms and would be read as an objective effect. Decode is cheap when a model stops early, so a 512 cap costs little and binds only on the arms that must not be truncated.

**Engine-agreement check, before any results exist.** Take one checkpoint, score it on dev under both HuggingFace and vLLM, confirm agreement. Different kernels and floating-point accumulation orders can flip a near-tie under greedy decoding. If the two disagree materially, that must be known before the results exist, not after.

_Pinned at step 7 (`docs/decisions/step7_decision_record.md`); outcome: **agree** on every type, under parsers v1 and v2._ No trained checkpoint exists at step 7 and dev is drawn at step 8, so the check runs on the raw backbone, over all six items of 200 CVEs from the late window (the latest 20% of the non-test pool, the plan's pool for dev). Test is never read. vLLM runs twice: pass A in bank order, and pass B in a shuffled order, which measures vLLM's own batch noise. The HF reference runs at batch size 1 with SDPA attention and greedy decoding, from a generation config built from scratch, and every token is asserted to be the argmax of the raw logits. **Material** means that on any type the HF − vLLM A gap is ≥ 1 point and its 99% paired bootstrap interval over CVEs excludes 0 (99% per type keeps the five-type false-alarm rate near 5%). A material result stops the pipeline before step 8.

**Weight-sync assertion.** GRPO's model changes after every optimiser step, so the inference engine's copy goes stale immediately. Weight syncing is handled by TRL's `GRPOTrainer` vLLM integration, not implemented by hand. A runtime assertion confirms rollouts originate from current weights. This is the one failure in the whole integration that produces a _wrong number_ rather than a slow run: if syncing silently fails, GRPO trains against its own past self, reward curves look plausible, and nothing in the results table reveals it.

### Per-workload integration

**1. Frozen-model session.** The signal audit and distill-self rationale generation run on the same untrained backbone, after the bank is frozen. vLLM loads once and serves both in a single session, before any training code exists. (The contamination probe runs earlier, on its own, since it needs no bank; it is a few minutes of `n=1` requests and does not justify holding the session open.)

- _Audit:_ 1,000 requests at `n=8` under the pinned rollout sampling config, replacing 8,000 separate calls. Prompts drawn from non-test items only. The audit's group-variance numbers only describe the runs they gate if the sampling matches training exactly.
- _distill-self rationales:_ one request per non-test item — roughly 20,000 at a 3,400-CVE non-test pool — `n=1`, 512-token cap. These are the longest outputs in the experiment, and where continuous batching earns its keep because rationale lengths vary widely. Output is cached to disk as a static file keyed by `(CVE ID, type, item index)`; the generation loop is checkpointed so a crash at item 16,000 does not restart from zero. The single regeneration pass over surface-invalid rationales is a second, smaller batch.

Because the pool is _all_ non-test items rather than one seed's train set, this file is generated once and every seed's partition reads from it. That is the entire reason the ordering was changed.

**2. External-model jobs.** DPO rejected completions (9,665; MCQ needs no request) run against the pinned external model over the same frozen non-test item pool. distill-external traces were planned here too, but the model refused them at the step-6 pilot and the arm is dropped. The DPO job produces a static file, cached to disk, generated once and reused across every seed and matrix. Pin the model and its configuration, archive the generation prompts, and record request-level metadata so the files can be audited. They cannot be regenerated identically, because the model has no dated snapshot and no sampling control, so the cached files are the artifact. Pins in `docs/decisions/step6_decision_record.md`.

**3. GRPO rollouts.** TRL `GRPOTrainer` with `use_vllm=True`, `num_generations=8` (which becomes `n=8` with prefix sharing), per-type completion caps, and `gpu_memory_utilization` tuned against OOM. Deployment mode — `colocate` (all four cards run full GRPO jobs) versus `server` (one card serves inference, three train) — is decided by measurement on a single pilot run, not by assumption. Colocate is the likely choice on 4×A40, since a 7B in bf16 is 14GB and the workload is ~24 independent GRPO runs that parallelise trivially across cards.

**4. Evaluation and checkpoint selection.** All batch mode through vLLM with frozen weights; no syncing involved. Volume is reduced as specified in the selection rules below.

### Other efficiency pins, applied uniformly

An efficiency change applied to some arms and not others is a confound. Everything here is uniform, or its non-applicability is structural and recorded.

- **One training job per GPU.** A 7B with LoRA fits on a single A40. Sharding across four cards with FSDP or ZeRO-3 gains nothing and costs 30–50% to PCIe communication on hardware without NVLink. The workload is 120 embarrassingly parallel runs; run four at once.
- **Flash Attention 2** everywhere (Ampere is supported).
- **Gradient checkpointing off** unless memory measurement says otherwise. It trades ~30% more compute for memory you likely do not need.
- **Sequence packing** for the cross-entropy arms (base, SFT, distill-self), with attention masking so no sequence attends across a document boundary. Structurally inapplicable to DPO (paired) and GRPO (per-prompt groups); recorded in the per-arm config rather than treated as an oversight.
- **Length bucketing** for DPO and GRPO, recovering most of what packing would have.
- **LoRA rank 16** on attention and MLP projections, identical across all arms.

## Pinned definitions

Every item below is a knob that can move a score by several points after results are visible. All are frozen before step 0, implemented as a single module with unit tests, with the MITRE XML release checked into the repository. Nothing here may be revisited once the test set has been touched.

### CWE hierarchy scoring

|Decision|Pin|
|---|---|
|View|CWE-1000 (Research Concepts)|
|Version|One MITRE release, cited by number; XML archived in-repo|
|Edge types|ChildOf only. PeerOf, CanPrecede, CanAlsoBe excluded — not hierarchy|
|Distance|Shortest undirected path over ChildOf edges|
|Scores|see direction-aware table below|
|Multiple parents|Minimum distance over all parents|
|Placeholders|Drop NVD-CWE-noinfo, NVD-CWE-Other, or absent from CWE-1000. Report N|
|Multiple CWEs per CVE|Drop multi-CWE rows. Report N|
|Deprecated IDs|Map to replacement where MITRE specifies one; otherwise drop|

**Hub-answer check and direction-aware credit.** In C/C++ code a large share of gold labels sit beneath CWE-119: CWE-787, CWE-125, CWE-120 are all one hop below it. Answering "CWE-119" every time earns roughly 0.5 on a large fraction of items at zero effort. GRPO drifts to that hub, groups tie at 0.5, the gradient dies, and binary exact-ID stays near the floor. DPO negatives built on a score margin learn the same preference for generality.

1. **Constant-answer baseline, computed first.** Over the non-test facts, compute the mean hierarchy score of the single best constant CWE answer. Report the value and the CWE achieving it as a trivial baseline row in every results table.
2. **Direction-aware credit, conditional on that value.** If the baseline exceeds 0.2:

|Relation of prediction to gold|Score|
|---|---|
|Exact match|1.0|
|Child, 1 hop|0.5|
|Ancestor, 1 hop|0.25|
|Any other relation, 2 hops (sibling, grandchild, co-parent)|0.25|
|Ancestor of gold at 2 hops (grandparent, or an ancestor also reachable as a sibling)|0.125|
|Otherwise|0|

_Outcome (step 2, `docs/decisions/step2_decision_record.md`):_ symmetric baseline CWE-119 = 0.262 > 0.2, so the direction-aware schedule is adopted; re-reported under it, the best constant is CWE-125 = 0.236.

At or below 0.2, the symmetric schedule (1.0 / 0.5 / 0.25 / 0) stands. The computed number decides, not preference, and the decision is recorded before any training run. Re-report the constant-answer baseline under the adopted schedule.

_Known limitation:_ NVD often assigns a Class-level CWE where a Base-level answer is arguably more precise, so a precise answer can score 0.5 rather than 1.0. The ancestor discount slightly worsens this, trading a small unfairness for removal of a degenerate optimum. Deliberate.

### CVSS

|Decision|Pin|
|---|---|
|Version|**v3.0 and v3.1 only.** Drop rows with neither; report N **and the publication-year histogram of the drop**|
|Source|NVD's own v3.x record. CNA-only vectors dropped|
|Components|AV, AC, PR, UI, S, C, I, A. Temporal and environmental excluded|
|Scoring|Proportion of the eight matching exactly, case-normalised. Identical for both versions|
|Canonical order|`AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_`; parser accepts any order, scores per-component|
|Version logging|Record per row; report the v3.0/v3.1 proportion per split|
|v2 / v4.0|Excluded. v2 is the real format break; v4.0 would be a separate column if ever added|

v3.0 and v3.1 share the same eight base metrics with the same allowed values; 3.1 changed guidance and rounding, not format. Accepting both rather than v3.1 alone avoids deleting a block from the middle of the time range.

The v3.x requirement still truncates the old end of the range, which is what the census time-range gate above exists to measure and decide on. That gate runs before any split is drawn.

**Parity check after the census, before any split:** compare mean per-component agreement of the majority baseline on v3.0 rows against v3.1 rows. Investigate if they differ materially.

Also pinned: diff normalisation; insertion attribution; the 20% patch threshold and insertion guard; the 10,000-token function cap; the line-set matcher; MCQ gold-letter assignment; near-miss construction rules, the trace and near-miss prompts, and the external model's identity and version; fixed second-knob values; the vLLM version and both sampling configurations; all answer parsers.

## Splits

**Unit.** The split key is CVE ID, drawn before the question bank is generated. Each CVE contributes six items across five types (find-the-error contributes two — the vulnerable function and its patch); splitting at item level would put the same function, CWE and description on both sides and measure recall of a specific CVE rather than transferable skill. Verify no CVE ID appears in two splits.

**Near-duplicates.** PrimeVul merges four upstream datasets, and the same function can surface under two CVE IDs or as a backport. Hash normalised function bodies, check collisions across split boundaries, move colliding CVEs to the **earlier** side so a train-era function never lands in test. Report N moved.

**Proportions.** 75 / 10 / 15, chronological by publication date: train oldest, dev, then test newest. Dev sits between train and test in time rather than inside the training window, so it lies on the same side of the distribution shift as test.

At 4000 CVEs: 3000 / 400 / 600. At 2500: 1875 / 250 / 375. **Fallback:** below 2500, shift to 70 / 15 / 15. Never shave test.

**Test window is drawn first and once**, immediately after the census — chronologically latest 15%, near-duplicate clusters kept intact. It depends only on publication dates, so it can and must precede bank generation. _Drawn at step 2:_ boundary 2021-07-30, 343 test CVEs (15.07%), 1 moved to the earlier side as a near-duplicate (exact normalised hash, or 3-line-shingle Jaccard ≥ 0.8).

**No absolute dev floor.** The earlier 500-CVE minimum counted the wrong unit: each dev CVE contributes **six** question items, so 400 CVEs is 2,400 dev items. Correlation within a CVE puts the effective n below 2,400 but far above 400. The floor was also unsatisfiable — 10% never reaches 500 in the stated census range — so it silently overrode the split it accompanied. Three conditions replace it:

1. **Selection on the mean across question types, never per type.** Per type the count really is 250–400 (800 for find-the-error) and the precision worry bites; across types it does not. Column-standardised, using the same pooled seed SD as the primary test.
2. **Paired comparisons.** Candidates are scored on identical items, so what matters is precision of the _difference_, which depends only on items where candidates disagree. At 10% disagreement over 2,400 items, SE of the difference ≈ 0.65 points — well below the 2-point gaps being resolved.
3. **A small grid.** Three learning rates; four checkpoints.

**Per-seed resampling is selection, not regeneration.** The test window is held out once and frozen, and the question bank over the non-test pool is generated once and frozen. For each seed, re-partition the non-test pool — and the re-partition must actually vary. "The latest 10% of the remainder" is a deterministic function of publication dates: every seed gets byte-identical train and dev, and the claim that seed variance includes data composition becomes false. Instead:

1. Define the **late window** as the chronologically latest 20% of the non-test pool.
2. Per seed, draw half of it uniformly at random as dev; the rest returns to train.
3. Near-duplicate clusters are assigned as units.

Dev stays chronologically late but its membership differs across seeds. Report pairwise dev overlap as a check that randomisation is live.

Because the bank and every generated artifact are keyed by `(CVE ID, type, item index)` over the whole non-test pool, a seed's partition is a lookup, not a generation job. No cached file is invalidated by re-partitioning.

**Contamination.** Report split boundaries relative to the backbone's cutoff, alongside the probe result. Report CWE distribution and CVSS version proportion per split.

## Question bank

Five types per fact, six prompts per fact, one prompt template each. All generation-based and parsed, so all five are trainable by every objective.

Generated **once**, after the test window is drawn: one pass over the non-test pool, one pass over the test pool, both frozen before any generated artifact or any seed partition exists.

**What each prompt shows.** The context is split by source, and no prompt shows the CVE ID:
- MCQ, exact-ID and CVSS see the NVD description only;
- find-the-error sees the bare function only;
- line localisation sees the description and the numbered vulnerable function.

Literal CWE IDs in a description, and the CVE's own ID anywhere, are redacted. Every template ends "End your reply with … in the form …", so one byte-identical prompt serves arms that answer directly and arms that reason first. Templates, targets and parsers are pinned in `docs/decisions/step4_decision_record.md`.

|Type|Prompts per CVE|Prompt|Eval metric|Training score (verifier-consuming arms only)|
|---|---|---|---|---|
|MCQ|1|which CWE, four lettered options|parsed letter, exact match|binary; chance 25% gives natural spread|
|Exact-ID|1|name the CWE|exact string after normalisation|CWE-hierarchy distance|
|CVSS|1|give the v3.x base vector|per-component partial credit|same, already dense|
|Find-the-error|2|is this function vulnerable, which CWE|paired accuracy|per-function composite, normalised to [0,1]|
|Line localisation|1|which lines does the fix touch|F1, one-to-one ±1 matching|same F1, same matcher, parse-gated|

**Item accounting.** Six items per CVE, not five. This propagates: 400 dev CVEs is 2,400 dev items; a 3,400-CVE non-test pool is ~20,400 items per generated artifact; the 150-CVE checkpoint subsample is 900 items. (The primary test's item-level permutation, which applied one arm-label permutation to all six of a CVE's items, was replaced at step 7; see the primary test.) Every count in this document uses six.

**Dense scores.** Reported metric and training score are separate objects. The reported metric never changes. The dense score exists only where an arm consumes a verifier — DPO pair construction and GRPO reward — because binary verifiers produce no gradient at the floor.

This creates a real asymmetry: GRPO's training signal is dense where its eval metric is binary, on exact-ID and find-the-error. The alternative was an arm that takes no gradient steps. Stated here rather than discovered by a reviewer.

**MCQ interface.** Scored by generating a letter, not by ranking option likelihoods. Likelihood ranking is not a function of a sampled completion, so no generation-based objective can optimise it; training one interface and evaluating another would measure the mismatch and attribute it to the objective. Length-normalised log-likelihood ranking is retained as a secondary diagnostic on all arms, never trained against. Read together: high rank with low letter accuracy is knowledge without format; low rank with high letter accuracy is a decoding habit without a shift in belief.

**MCQ construction.** The external model proposes the distractors and fixed rules decide which are admitted. Options are generated once and frozen, so every arm and every seed sees identical option sets.
- **Why not the sibling field.** An earlier draft drew distractors from the fact table's sibling-CWE field. On non-test that lets "pick the option that is most often a gold label" score 0.895 against a chance rate of 0.25. SFT could learn that label prior faster than GRPO and manufacture an arm × type interaction.
- **Admission.** A distractor must be a live CWE-1000 weakness. It must not be gold and must not be an ancestor or descendant of gold, since either would be a second defensible answer. Option names come from the MITRE XML.
- **Gaps.** One regeneration, then a draw weighted by non-test gold frequency fills any missing slot.
- **Pre-registered guard.** If the most-familiar-option shortcut exceeds 0.5 on non-test, every MCQ item is rebuilt from that draw alone. The draw scores 0.348.

**Gold letter assignment.** Assigned by a deterministic hash of the CVE ID into {A, B, C, D}, fixed at bank-generation time. The earlier pin — "balanced at exactly 25% per letter within each split" — is retired because it is incompatible with freezing options before splits exist, and per-seed rebalancing would make MCQ a different item between M1 and M2, which is exactly the comparison the matrices rest on. Hashing gives 25% per letter in expectation and approximately within any large subset. **Report the realised per-letter marginals per split** as a check; the "always A" baseline catches the failure if a split comes out skewed.

## Answer formats and parsers

The target format is what the gold answer looks like and what SFT trains toward. The parser is what scoring accepts, and is deliberately looser. The gap is the format ramp: a model can earn reward while its output is rough, without being paid for formatting alone.

**Line localisation target:** `LINES: 12, 13, 17` — one line, 1-indexed, comma-separated, ascending, deduplicated. No ranges (`12-17` forces an endpoint-inclusivity convention the model gets wrong in both directions). No JSON (bracket failures for zero information gain). Empty prediction is `LINES: none`. Gold always sorted and deduplicated, so SFT has one canonical target per item.

**Line localisation parser:**

1. If `LINES:` appears, take everything after the last occurrence; otherwise the whole response.
2. Extract all integers.
3. Drop any outside `[1, function_length]`.
4. Deduplicate and sort.
5. Empty result, or literal `none` → parse success, empty set.
6. No integers and no `none` → parse failure.

Step 3 does real work: without it, a response mentioning CWE-787 contributes 787 and CVE-2021-44228 contributes two garbage numbers, silently wrecking precision. Step 5 matters because an empty prediction has a well-defined F1 — treating it as failure would conflate "declined" with "produced garbage."

**Line-set matching — one-to-one, pinned.** The ±1 tolerance must be a matching, not a membership test. Under "within 1 of any gold line," tolerance becomes free precision:

- Gold `{12}`, prediction `{11, 12, 13}` → precision 3/3, recall 1/1, **F1 = 1.0**. Tripling the guess costs nothing.
- Gold of 2, prediction of all 40 → up to 6 fall inside the windows, giving F1 ≈ 0.26, not the 0.095 the anti-spray argument relies on.

The pinned matcher builds a maximum matching where **each gold line may be claimed by at most one prediction and vice versa**. Among maximum matchings it prefers the most exact hits, then the lowest line numbers. That tie-break decides which pairs are matched, never how many. Matching exact hits first and then greedily at distance 1 is *not* equivalent: gold {11, 12} against prediction {12, 13} scores 0.5 that way and 1.0 under the maximum matching. TP is matched pairs; precision TP/|predicted|, recall TP/|gold|. Under it, `{11, 12, 13}` against `{12}` gives F1 = 0.5 and spray-all returns to 0.095. The identical matcher runs in the GRPO reward and in evaluation.

**All five parsers** are tuned against dev outputs only, never test, and frozen before test is touched. Leniency is worth several points if tuned after seeing results. The same parser runs on every arm, base especially, since base will answer in prose. Every score is also reported under a strict parser as a robustness column; disagreement on arm ordering is a finding about format sensitivity, not a bug.

_Step 7:_ the parsers are reviewed on the untrained backbone's non-test replies (the step-5 audit and the engine check's greedy replies; never the probe's) and frozen before any training, because they also compute the GRPO reward. Any change is approved by the owner and becomes v2, and the probe is re-scored with both versions reported. _Outcome:_ **parsers v2**, frozen. Three owner-approved changes:
- an MCQ answer written as one option's text reads as that option's letter;
- CVSS accepts the specification's value names (`AV:Network`);
- without a `LINES:` field, echoed code is not read as predicted lines.

The probe re-scores identically, and step 5's decisions stand. Details are in `docs/decisions/step7_decision_record.md`.

## Reward shaping

```
if parses:  reward = dense_score      # may be 0.0
else:       reward = 0
```

No additive format credit anywhere. The earlier design paid credit for well-formed line numbers, any values, reasoning that this lets a model climb the format before the content. It does the opposite: `LINES: 7` collects that credit every time at zero effort while genuine localisation returns F1 near zero, so the cheap behaviour dominates. Reward climbs, the loss curve looks healthy, F1 stays flat at zero.

The lenient parser is the ramp instead. A rough prose answer containing plausible line numbers still parses and still earns F1 if any are right, so groups disagree and gradients flow; `LINES: 7` parses perfectly and scores 0 unless 7 is gold.

**Every dense score lies in [0, 1].** A unit-tested constraint on the reward module, not a coincidence of the individual definitions.

**On spray-and-pray:** under the one-to-one matcher, 40 predictions with 2 correct gives F1 ≈ 0.095. Because tolerance makes _sparse_ spraying the cheapest remaining trick, "every k-th line" for k = 2, 3 joins the trivial baselines.

Logged beside every reward: parse-failure rate and predicted-set-size distribution per arm.

## Converters

The prompt is byte-identical across all arms. Only what sits beside it changes. Converters read the frozen bank plus the frozen generated-artifact files, and emit one training file per arm per seed by selecting the seed's train items. They generate nothing.

|Arm|Role|Consumes|Per-item form|
|---|---|---|---|
|Base|primary|raw text|no question items at all|
|SFT|primary|prompt + gold|prompt → gold string|
|Distill-self|primary|prompt + own hint-conditioned rationales|base model writes a rationale _given the gold answer_; rationale + gold answer is the target|
|DPO-from-base|primary|prompt + chosen/rejected|chosen is gold; rejected written by the external model in identical format|
|GRPO|primary|prompt + verifier|k=8 rollouts, dense reward, group-normalised advantage|
|~~Distill-external~~|dropped at step 6|—|planned: written-out reasoning from the external model; refused|
|SFT→DPO|secondary|prompt + chosen/rejected|DPO initialised from the SFT checkpoint|

Base, SFT and distill-self never touch a verifier for data selection. Every shaping decision lands in DPO and GRPO only. No converter can delete a row from another converter's file.

### Distillation — two arms, the substitution rule, and what it costs

**distill-self is the arm in the primary test, built by hint-conditioned generation.** The earlier plan specified it two incompatible ways, both unworkable:

- _Filter to correct traces._ A 7B base may produce almost none on line localisation, so the file silently loses most of that column — violating full coverage. Deciding "correct" on CVSS or line localisation also needs a threshold, putting a verifier in charge of SFT-family data.
- _Keep everything, overwrite the answer._ This manufactures self-contradictory targets: a trace reasoning toward an out-of-bounds _read_ with `Answer: CWE-787` (a write) stapled on. The arm learns that its reasoning and its answer are causally unrelated.

**Pinned procedure.** The base checkpoint is shown the prompt _and the gold answer_ and asked to produce the reasoning that justifies it. The target is that rationale followed by the gold answer in canonical format. The hint appears only during generation; the training prompt stays byte-identical to every other arm's. Generated in the frozen-model vLLM session over the **whole non-test pool**, cached to disk.

Coverage is 100% by construction, no verifier touches the data, no target contradicts itself. Rationales are checked for surface validity only — they must end in the gold answer and must not restate the hint as given ("the answer is CWE-787 because we were told it is"). Failures are regenerated once, then fall back to gold-only; report the fallback rate per type.

**Naming.** This reframes the arm: it is SFT on self-generated rationales, not R1-style trace distillation, and the write-up must call it that. The comparison against SFT stays clean and stays interesting — it isolates whether producing intermediate reasoning before committing helps — but it is no longer a comparison against trace imitation. Label it **SFT-rationale (distill-self)** in every table.

**Substitution rule, pre-registered.** If distill-self's self-generated rationales are too poor to constitute a real arm, its rationales are substituted with the external model's traces for the affected types. The rule must be fixed now, because deciding it after seeing the arm's scores would make it a free parameter worth several points.

_Trigger, evaluated per type at the signal audit, before any training:_ substitution fires on a type when **either** the surface-validity pass rate falls below 50% **or** the fallback-to-gold-only rate exceeds 50% — i.e. the majority of that column's targets would otherwise be bare gold strings with no reasoning, which is SFT with extra steps rather than a distinct arm. The pass rate is read on the **first attempt**, before the regeneration (step 5); under that reading the fallback clause is implied by the first.

_Scope:_ substitution is **per type, not per arm**. If line localisation fires and the other four do not, only that column's rationales come from the external model. Which types were substituted is recorded and reported in every table containing the distill-self row.

_What it costs, stated plainly._ A substituted column is no longer independent of distill-external. It carries two explanations for any effect — "reasoning in the target helps" and "the external model knows things the backbone doesn't" — which is precisely why distill-external is excluded from the primary test. So:

- Any substituted cell in the distill-self row is **flagged in the primary test** alongside floored cells, with its contribution to the test statistic reported separately.
- The **leave-flagged-columns-out sensitivity run** covers substituted cells as well as floored ones.
- If substitution fires on **three or more of the five types**, distill-self has effectively become distill-external and must not be reported as an independent arm. In that case, drop distill-self from the primary test, run it on three arms (SFT, DPO, GRPO), and say so. Pre-registering this prevents a quietly-contaminated fourth row from carrying the headline.

_Not run (step 6):_ the external model refused all 120 pilot trace requests under its `reasoning_extraction` category, so distill-external is dropped. No trace was generated and the arm is not trained. See `docs/decisions/step6_decision_record.md`. **What this loses:** the gap between the two distill arms, which would have bounded how much of a distill benefit is teacher capability rather than the presence of reasoning in the target. The write-up states that the bound is unavailable. distill-self's reading — "producing intermediate reasoning before answering helps" — is unaffected.

**distill-external** (as planned) uses the same external model to generate unhinted traces in the ordinary way, reported descriptively, never in the interaction test. A strong teacher makes the row carry two explanations at once, which cannot be separated, and a contaminated row distorts the interaction statistic for all four arms rather than only its own cell. The gap between the two distill arms bounds how much of the benefit is teacher capability rather than the presence of reasoning in the target — a bound that becomes uninformative for any substituted type, which is another reason to record substitution per type.

A summarised trace is a different object from a full one, and a reviewer will flag distillation on it. The pinned external model, Claude Opus 5.5, never returns its hidden reasoning, so a trace here is the reasoning the model writes out in its reply before the answer. The owner has permission to request this. The write-up calls it written-out reasoning, not raw chain-of-thought. Report the dense-score distribution of external traces per type as a covariate.


### DPO — initialisation, reference policy, negatives

**DPO-from-base** initialises from the same starting checkpoint as every other arm. π_ref is a frozen copy of that checkpoint. "Base" is ambiguous in this document, denoting both the raw backbone and the base arm (continued pretraining on raw text with LoRA); π_ref is the raw starting checkpoint, not the base arm's weights.

**β is fixed, not swept** — see tuning parity.

**SFT→DPO** is secondary because it is standard practice but not in the primary test: it has had SFT plus DPO training, so it is neither independent of the SFT arm nor step-matched, and any advantage is partly the SFT stage.

Log mean log-probability of chosen and rejected separately throughout training. DPO's known pathology is driving both down; seen only through a final score that is indistinguishable from "DPO is bad at this type," and they mean opposite things.

**Negatives — written by the external model, format-matched.** Sampling rejections from the base policy fails on both sides. Unparseable sampled rejections make DPO train formatting while the write-up calls it content — and the same confound arrives through the _chosen_ side: with chosen a terse canonical `CWE-787` and rejected a rambling prose sample, the loss can be driven down entirely by "be short and canonical," never touching 787-versus-125.

- **Chosen** is the gold string in canonical target format.
- **Rejected** is written by the pinned external model, emitted in the _identical_ canonical format, differing only in answer content. Model, version and generation prompt archived in-repo; outputs cached as a static file over the whole non-test pool and reused across every seed and matrix.
- **Difficulty: hardest available near-miss.** The model produces the most plausible wrong answer at one unit of error, validated per type:

|Type|Rejected must be|Validity check|
|---|---|---|
|MCQ|the letter of the distractor closest to gold in the hierarchy (ties: the external model's step-4 order)|rule-defined at step 6: no request, not gold|
|Exact-ID|a CWE exactly 1 hop from gold in CWE-1000|in-hierarchy, distance 1, not gold|
|CVSS|gold with exactly one component changed|parses, exactly one component differs|
|Find-the-error|on vulnerable functions, correct label with a 1-hop wrong CWE; on patched, flipped label with the most plausible CWE|fields valid, not gold|
|Line localisation|gold with one line dropped, or one line displaced by ≥2|parses, F1 < 1.0 under the pinned matcher|

- **Validation is mechanical and non-negotiable.** Every generated rejection is checked against its rule and scored by the verifier; anything failing is regenerated once, then replaced by a rule-constructed near-miss meeting the same constraint. The external model chooses _which_ near-miss; the rules decide what is admissible. Report the rule-fallback rate per type — a type where the model rarely produces a valid near-miss is one where the negatives are effectively rule-generated, and the write-up should say so rather than claim model-written negatives throughout.
- **Line localisation displacement must be ≥2**, since a ±1 shift scores as correct under the tolerance and would produce a pair whose two sides are worth the same.
- **Margin.** The earlier "minimum chosen–rejected margin" pin is retired: a margin floor and a hardest-near-miss policy pull in opposite directions, since the near-miss _minimises_ the margin. The one-unit-of-error rule fixes the margin at the smallest non-zero value the metric admits, uniformly across types.

_Outcome (step 6, `docs/decisions/step6_decision_record.md`):_ `dpo.jsonl` built for all 11,598 non-test items. Of the 9,665 requested near misses, 99.3% are the external model's (96.5–100% valid at the first attempt per type, all in the ≥ 50% band) and 64 (0.7%) are rule-built: CVSS 2.1%, line localisation 1.2%, exact-ID and find-the-error 0%. MCQ's 1,933 are rule-defined. No refusals. Cost $104.19.

**What this costs, stated rather than hidden.** Negatives no longer come from the policy, so DPO's gradient sits where _we_ judge the model is likely to err rather than where it demonstrably does. A sampled near-miss has the virtue of being the model's actual error; a constructed one is format-matched and available on every item. This plan takes the second trade because the first makes DPO coverage depend on base-policy competence — near zero for a 7B on the code-reasoning types, the columns the experiment exists to measure — and because an uncontrolled format difference between chosen and rejected would invalidate the arm outright. Any finding about DPO here is a finding about this negative construction, and the construction, including the external model's identity, is part of the method.

### Base arm definition

Base is continued pretraining on raw text, with the same LoRA config and optimizer steps as the other arms. It is not a zero-shot control. The distinction governs how the base reference line in M2 is read.

## Tuning parity and hyperparameters

**Learning rate is the only swept hyperparameter, three values, identical grid for every arm.** Everything else is fixed at a published default, named in-repo, identical across seeds and matrices.

The earlier plan swept LR for every arm _and_ β for DPO, while GRPO's analogous knob was fixed. An arm allowed fifteen dev configurations and keeping its best has more chances to land a lucky setting than one allowed five, so part of any DPO–GRPO gap would measure tuning effort rather than the objective — the same confound this plan invokes against a shared learning rate.

|Arm|Swept|Fixed at default|
|---|---|---|
|Base|LR ∈ {3}|—|
|SFT|LR ∈ {3}|epochs|
|Distill-self|LR ∈ {3}|epochs|
|DPO-from-base|LR ∈ {3}|**β**|
|GRPO|LR ∈ {3}|**KL coefficient**, k=8 rollouts|
|SFT→DPO|LR ∈ {3}|β|

State the configuration count per arm in the paper.

**Sweep once, reuse everywhere.** The sweep runs at **M1, seed 0 only**, on **full dev** — this is the comparison that most needs precision and it happens once. The winning LR per arm is reused for every seed, every M2 loop, and M1-volume.

**Checkpoint selection per run, reduced.** Four checkpoints, evaluated on a **fixed 150-CVE dev subsample (900 items)**, the same subsample across all runs within a seed so comparisons stay paired. Pin the subsample before training. As originally specified (10 checkpoints × 2,400 items × 120 runs) selection would have required 2.88 million generations — larger than it appears and entirely absent from the budget; the reduction brings it to ~432k. Four evenly spaced checkpoints is enough to catch the peak on a fine-tuning curve this short.

**The trade-off, stated.** The shared LR was chosen on a seed-0 dev set containing all five types, so every M2 run inherits a trace of exposure to its dropped type. That channel is one scalar informing five columns, far weaker than the gradient exposure M2 removes and weaker than per-run checkpoint selection, which stays clean. It is not zero, and the write-up says so.

## Procedure

1. **Build the fact table**, apply all pinned definitions, report the filtering census including the **v3.x publication-year histogram**. Apply the time-range gate and record the decision. Compute the constant-answer hierarchy baseline and fix the exact-ID credit schedule. Run the CVSS version-parity check.
2. **Draw the frozen test window:** chronologically latest 15%, by publication date. Verify no CVE ID or near-duplicate function crosses the boundary. Report N moved.
3. **Contamination probe** on the raw backbone, over 300 CVEs from the drawn test window. Record the result and the decision-rule outcome.
4. **Generate the question bank, once.** One pass over the whole non-test pool, one pass over the test pool. Six items per CVE. MCQ distractors proposed by the external model and admitted by rule, with the shortcut guard applied. Options frozen, gold letters assigned by CVE-ID hash; report realised per-letter marginals per pool. Freeze the bank.
5. **Frozen-model vLLM session:** run the GRPO signal audit on non-test items and set the per-type rollout caps (the DPO audit line needs step 6's outputs and is computed there; the distill-external line was never computed, as the arm was dropped); generate distill-self hint-conditioned rationales over the **whole non-test pool**; cache to disk keyed by `(CVE ID, type, item index)`. Evaluate the **substitution trigger** per type and record which types, if any, will draw rationales from the external model.
6. **External-model jobs:** generate all DPO rejected completions over the whole non-test pool, plus substituted distill-self rationales where triggered (none: step 5 substituted no type). A 20-CVE pilot runs first and prices the full run. distill-external traces were planned here; the pilot's trace requests were all refused, and the arm is dropped. Validate every rejection against its per-type rule; regenerate once, then rule-construct. Cache all outputs; report rule-fallback rates. Do not proceed until the audit, substitution decisions and fallback rates are recorded.
7. Run the **engine-agreement check**: HF versus vLLM on the raw backbone, over 200 late-window non-test CVEs (dev is not drawn until step 8, and no trained checkpoint exists yet). Review and freeze the parsers. Freeze the primary-test and MDE implementation (a seed-bootstrap null; see the primary test).
8. **For each seed, partition the non-test pool:** late window = latest 20%, draw half at random as dev, rest to train, clusters intact. This is selection over the frozen bank — nothing is regenerated. Report inter-seed dev overlap.
9. **Run the converters;** produce one training file per arm for this seed by selecting that seed's train items from the frozen bank and the cached artifact files.
10. **At M1, seed 0 only:** sweep LR over three values per arm on full dev, selecting on the column-standardised mean across types via paired per-item differences. Record the winner per arm.
11. **Train the arms**, three seeds, one job per GPU, recorded LRs. Select checkpoints per run on the fixed 150-CVE dev subsample.
12. **Evaluate** every model on the whole frozen test set via vLLM, 512-token cap, all five types, every run.
13. **Record M1.**
14. Return to step 8. **Mask one question type** out of the train and dev item selection; upsample the remaining four so example count and optimizer steps match step 9 exactly. Test is untouched and keeps all five types.
15. Repeat 9–12 with recorded LRs (no sweep), selecting checkpoints on the mean over the four retained dev types.
16. Record M2 for the dropped type only; retain the other four columns as diagnostics.
17. Repeat 14–16 for each type.
18. Run M1-volume.
19. Analyse, and report the minimum detectable effect.

**Re-entry is at step 8, the partition, not at bank generation.** Type removal is a selection mask over an already-frozen bank, applied once per M2 loop, so it cannot drift between the bank and the converters and it costs no generation. This is the second consequence of the reordering: under the old design, dropping a type meant regenerating the bank, which meant regenerating the cached rationales, traces and negatives, five times over.

**Why dev drops the type too.** Dev produces no gradients but produces selection. Checkpoints are chosen by dev score, so leaving type X in dev means selecting the checkpoint that does best at the type the arm is claimed never to have seen. That biases M2 upward, flattering the headline. Test keeps type X because scoring the removed type _is_ M2.

**Upsampling and its control.** Without upsampling, M2 trains on 20% less data than M1. With it, M2 differs from M1 in two ways at once: never saw type X, and saw the other four with duplication. M1-volume isolates the second.

**M1-volume, corrected.** Same procedure as M2, with the removal spread across all five types: mask 20% of items uniformly at random, then upsample survivors back to full count with matched steps.

|Config|Total examples|Distinct|Steps|
|---|---|---|---|
|M1|1000|1000|S|
|M2|1000|800 (200 duplicated)|S|
|M1-volume, old|800|800|0.8S|
|M1-volume, corrected|1000|800 (200 duplicated)|S|

The old version differs from M2 along an axis M2 never moved (total example count) and takes 20% fewer steps as well, so subtracting it removes the wrong quantity twice. The corrected version differs in exactly one respect: whether the missing 20% came from one type or was spread across five. Duplication is a dataloader sampling weight, not generated data. The masked 20% is redrawn per seed; drawn once, the comparator would carry error bars it has not earned. Discarded M2 columns are kept as diagnostics — a leave-MCQ-out run should still score normally on CVSS, and if it doesn't, the upsampling is wrong.

## Compute budget

**The earlier budget's error.** It stated 99 runs while specifying LR sweeps per arm _within each seed_, β sweeps for DPO, and re-selection inside every M2 loop. Those are incompatible: the 99 counted keeper models, not models trained and discarded during tuning. Carried through, the real figure was roughly 600 training jobs, and 2.88 million generations for checkpoint selection appeared nowhere at all.

**Run count:**

|Component|Count|
|---|---|
|LR sweep: 6 arms × 3 LRs, M1 seed 0 only|18|
|M1 keepers: 5 primary arms × 3 seeds|15|
|M2 keepers: 4 post-trained arms × 5 types × 3 seeds|60|
|Base noise floor: one leave-one-out config × 3 fresh seeds|3|
|M1-volume: × 3 seeds|15|
|Secondary at M1: SFT→DPO × 3 seeds|3|
|**Required total**|**114**|

The base arm's M2 equals its M1 by construction and is not re-run beyond the noise-floor configuration. If substitution fires on three or more types and distill-self is dropped from the primary test, M2 falls to 45 and the total to 99. (The total was 120 with distill-external, dropped at step 6: 3 sweep runs and 3 secondary runs.)

**GPU-hours, with the vLLM integration and reduced selection:**

|Component|Naive|Integrated|
|---|---|---|
|GRPO training|480–960|150–300|
|Other arms|290–580|120–250|
|Test evaluation|~40|~15|
|Checkpoint selection|~145 (uncounted)|~25|
|Probe, audit, rationales|~35|~6|
|**Total**|**~990–1,760**|**~315–595**|

Rationale generation is costed **once over the whole non-test pool**, not once per seed. Under the old ordering it would have been three times this, plus a further five regenerations for the M2 loops.

On 4×A40 at realistic utilisation: **4–8 days of wall clock**, with contingency of **2–3 weeks** for failures and re-runs. External-model inference for traces and negatives is costed separately as API spend, not GPU-hours, and is a one-time cost over the non-test pool reused across every run and every matrix.

**These are estimates, to be replaced.** Run one GRPO pilot — single config, reduced steps, dev-only evaluation — measuring colocate against server mode, and substitute measured per-arm GPU-hours for the table above before committing the cluster. This is not a power gate; it is a costing measurement.

**Optional, only if the allocation allows:** the secondary arm at M2 (1 × 5 × 3 = 15), and the token-matched secondary comparison, which must be costed explicitly before it is promised. If unfunded, remove the token-matched claim from the opening section rather than leaving it unsupported.

**Scheduling.** GRPO dominates and appears in 15 of 60 M2 keepers plus its share of M1, M1-volume and the sweep. Cost in GPU-hours weighted by arm, not job counts, or the GRPO rows will overrun.

## Per-arm signal audit

Run in the frozen-model session, after the bank is frozen and before any training. 200 prompts per question type, drawn from non-test items, on the base checkpoint, under the pinned rollout sampling config.

The underlying failure is shared: on a type where the base policy almost never produces a distinguishable answer, three of the four primary post-trained arms break — by different mechanisms, which is why one diagnostic does not cover them. A 7B backbone makes this more likely, not less.

**GRPO — group variance.** 8 rollouts per prompt (one `n=8` request), measuring the fraction of groups with nonzero reward variance under both binary and dense scoring. GRPO normalises advantages within a group, A_i = (r_i − mean(r)) / std(r). If all rollouts score identically, every advantage is exactly zero and the prompt contributes no gradient. A group only teaches when its members disagree. At a 5% per-sample success rate with G=8, two-thirds of groups are dead; at 1%, 92% are. The run completes, the loss curve looks normal, and the reported number says "GRPO underperforms" when the truth is "GRPO took almost no steps."

**DPO — near-miss constructibility.** Coverage is 100% by construction, so this measures the fraction of prompts where the external model produces a valid 1-hop negative without rule fallback, per type, with fallback reasons logged. MCQ's negative is rule-defined, so its line is reported as not applicable.

**Distillation — rationale quality and the substitution trigger.** For distill-self: surface-validity pass rate, fallback-to-gold-only rate, and the fraction of rationales merely restating the hint. These feed the substitution rule directly. For distill-external: fraction of prompts where the teacher reaches the correct answer, plus dense-score distribution per type (not computed: the arm was dropped at step 6).

**Decision rule**, per arm per type, fixed in advance:

- **≥50%** — the column trains as specified for that arm.
- **10–50%** — for GRPO, enable dynamic sampling: discard tied groups and resample until the batch is full, logging retained-prompt distribution per type and deviation from the identical-fact-set claim. For DPO, report the fallback subset as a covariate. For distill, report rationale quality per type.
- **<10%** — that arm is **floored** on that type. The cell is still produced, still reported, and still enters the primary test, flagged.

A type can clear the bar for one arm and fail it for another. That asymmetry is itself a result about the objectives.

**What flagging does and does not do to the primary test.** Floored and substituted cells remain in the matrix and in the pre-registered test; the test runs on all five columns regardless. A deliberate choice with a known cost, pinned so it is not revisited after results are seen. A floored cell contributes a large residual that is partly a training failure rather than a skill difference; a substituted cell carries an extra explanation. So a significant result driven by flagged cells is a weaker claim than one driven by intact cells. Three requirements make that visible:

1. Every floored or substituted (arm, type) cell is marked in every table and plot.
2. Per-cell contributions to the test statistic are reported.
3. A **leave-flagged-columns-out sensitivity run** is reported alongside. It is not the primary result and cannot replace it, but a primary surviving only with flagged columns included is described as such.

Dropping flagged columns from the test was rejected because it lets the audit silently change which hypothesis is tested, and because a column can be floored for one arm and fine for three others.

**Running monitor.** The audit is re-evaluated every N steps during training, per type per run, against the same bands. A step-0 check cannot catch the common failure where the model learns to parse, then ties at zero on content, and silently stops learning — early on, inconsistent formatting creates spread and the check passes; later, uniform formatting with uniform wrongness means every group ties. Removing the additive format credit makes this more likely, not less, since there is no longer an artificial source of spread. Any run dropping below the floor mid-training is flagged.

Log throughout training, per type per run: nonzero-advantage fraction, reward mean and variance, and their trajectories.

**Expected trouble spots.** CVSS is dense by construction and fine. MCQ has 25% chance and disagrees freely. Exact-ID is borderline under binary scoring and fine under hierarchy scoring — watch the hub answer. Find-the-error and line localisation are the risks, and they carry the actual code reasoning. At 7B, expect floors here: PrimeVul's paired metric drops StarCoder2 from 68 F1 to 3. Line localisation is also the type most likely to trigger distill-self substitution.

### Find-the-error reward — per-function, normalised, failure mode admitted

Paired accuracy is defined across two prompts, while a GRPO rollout exists on one. The reward is therefore **per-function**; paired accuracy is computed afterward by pairing rollouts across the two prompts and remains the headline. A likelihood margin between the two functions is a useful diagnostic but cannot be a reward, for the same reason likelihood-ranked MCQ cannot.

The two prompts are the two items this type contributes per CVE. They are never split across a partition boundary: the pair is the unit, assigned together, which follows automatically from CVE-level splitting.

**Normalisation.** The earlier formulation summed label correctness and CWE credit, so a vulnerable function could earn up to 2.0 while a patched function capped at 1.0 — a thumb on the scale toward "vulnerable" before any learning occurs, and a violation of the [0, 1] invariant. Pinned:

```
vulnerable prompt:  reward = 0.5 * label_correct + 0.5 * label_correct * cwe_score
patched prompt:     reward = 1.0 * label_correct
```

CWE credit is gated on the label being right. Both span [0, 1] and a correct answer is worth the same on either side.

**The known failure mode, retained deliberately.** Per-function reward does not close the loop on constant-answer collapse. A policy always answering "vulnerable, CWE-119" earns up to 1.0 on every vulnerable prompt and 0 on every patched one — mean reward around 0.5 — while paired accuracy is exactly 0. Once consistent, all 8 rollouts on a patched prompt agree, the group ties, the advantage is zero, and GRPO is never pushed off it. Reward looks healthy; the headline metric is at the floor.

Pair-level grouping would fix this by making the reward equal the headline metric. This plan does not do that. If GRPO collapses, **the collapse is reported as the result for that cell**: _GRPO collapses to a constant label under a per-function reward on find-the-error._ That is a real finding, and it is a finding about **GRPO combined with this reward design**, not about GRPO as an objective. The write-up states that framing wherever the cell is discussed, including in the abstract if the cell drives the headline. "Would pair-level grouping have fixed it?" is a fair question, and the honest answer — probably, and we did not run it — belongs in the limitations.

Detection is not left to inference. Logged per run per step: prediction rate by class, fraction of tied groups on patched prompts specifically, per-function accuracy split by class, and paired accuracy. Collapse is declared when prediction rate for one class exceeds 0.9 over a logging window, and the step is reported.

## Scoring

|Type|Metric|Scale|
|---|---|---|
|MCQ|parsed letter matches gold|binary → accuracy|
|Exact-ID|exact match after normalisation|binary → accuracy|
|CVSS|proportion of eight v3.x components correct|continuous → mean|
|Find-the-error|both members of a pair classified correctly|binary → paired accuracy|
|Line localisation|F1, one-to-one ±1 matching|continuous → mean|

All five are generation-based and parsed. Parse-failure rate is logged beside every score. An arm at 0.0 with 90% parse failures has a format problem, not a knowledge problem, and that distinction must survive to analysis.

MCQ carries a second column: length-normalised log-likelihood ranking over option strings, all arms, never trained against. Length normalisation is required — raw log-likelihood favours short options.

Find-the-error is scored per pair, collapsing its two items into one number per CVE. PrimeVul's test split pairs each vulnerable function with its patch, so answering "vulnerable" every time gets 50% per-function and looks competent. Paired accuracy is the headline; per-function accuracy, prediction rate by class, and parse rate are logged beside it. The paired metric requires **the correct label only**, with CWE-correct paired accuracy as a second, stricter column.

Exact-ID carries the hierarchy score as a logged secondary column for all arms.

Line localisation logs parse-failure rate and predicted-set-size distribution, scored under the pinned one-to-one matcher. The ±1 tolerance exists because the insertion-attribution convention is chosen and the model cannot know it. Report under both conventions if they differ materially, and stratify F1 by gold-set size.

Response length is logged alongside every score. DPO is known to lengthen outputs, which confounds parse rate. The uniform 512-token evaluation cap ensures no arm is truncated where another is not.

Decoding is fixed by the pinned configuration for all evaluation, through vLLM, with a second configuration as a robustness check. Greedy versus sampled changes results unevenly across arms, since RL-trained models often have collapsed entropy.

**Trivial baselines**, no training compute: most-frequent-CWE for exact-ID and MCQ; the best constant CWE under the adopted hierarchy schedule; always-"A" for MCQ; majority-class-per-component for CVSS; always-"vulnerable" for find-the-error at both per-function and paired accuracy; a keyword heuristic for line localisation (`memcpy`, `strcpy`, `alloc`); every-k-th-line for k = 2, 3.

**Seeds.** One number per (arm, type, seed); mean over three seeds with seed SD reported. Each seed carries its own train/dev partition as well as its own training randomness, so the variance estimate includes data composition — the term that usually dominates. The test set is identical across seeds, so this excludes test-set sampling variance. Report it as such. Because LR is swept once at seed 0 and reused, seeds differ in data and training randomness but not in learning rate. The base arm's noise floor applies to base; RL seed variance is typically several times larger, so GRPO needs its own floor estimate or the "exceeds seed variance" trigger will fire constantly on one row and never on another.

Note what the frozen bank does and does not remove from seed variance: item _content_ is now identical across seeds by construction, so seed variance reflects which items each seed trained on, not how those items happened to be phrased. That is the intended quantity — phrasing variation was never part of the hypothesis — and it makes the estimate cleaner, not narrower in a misleading way.

## The matrices

**M1** — all arms trained on all five types, tested on held-out items.

**M1-volume** — 20% of examples masked uniformly across types, survivors upsampled to full count and matched steps. The comparator for M2.

**M2** — for each type in turn, retrain the four primary post-trained arms with that type masked out of train and dev and the rest upsampled. Test on the removed type. Five loops, five columns.

The base arm, trained on raw text, has no question types in its training data at all. Masking one changes nothing, so its M2 equals its M1 by construction. This makes the base row a fixed reference line: the score reachable having never seen the format. Any arm whose M2 falls below base in a column has been actively harmed by its objective on that type, which is sharper than a gap.

_Caveat:_ on exact-ID and line localisation a 7B base may floor for format reasons rather than knowledge reasons. A floor at zero means nothing can fall below it and the "actively harmed" finding is unavailable in that column. Run base with few-shot prompting as a diagnostic to separate "doesn't know" from "won't format," and report both.

## Interpretation

M2 is trained on strictly less relevant data than M1, so M2 ≤ M1 is the expected direction.

||M2 high|M2 low|
|---|---|---|
|**M1 high**|Small gap → transferable skill. Advantage.|Large gap → needed the format to perform. Memorisation.|
|**M1 low**|Investigate.|Disadvantage.|

The M1-low / M2-high cell is not automatically a bug. Negative interference is real: training on type X can induce a format habit that suppresses performance the other four types would have preserved. Find-the-error is where to expect it, since constant-answer collapse is its known failure mode and is now an explicitly anticipated outcome there.

M2 identifies which teacher is genuinely good at a question type. The gap explains why — skill or formatting. Report both; the gap alone is misleading, since a model with nothing to lose loses nothing. Compare the gap against M1-volume, not M1.

Compare within a column only. Some types are inherently harder to reach sideways, and a uniform gap down a column is a fact about that type, not about any objective.

**A note on gap size under a compressed time range.** If the census gate found a short surviving range and the accept branch was taken, train and test resemble each other more than the design assumes, and every gap in this table shrinks toward zero for reasons that have nothing to do with the objectives. That interacts directly with the minimum-detectable-effect report and must be stated where the gaps are discussed, not only in the census section.

### Primary test, pre-registered

One test on one matrix: the objective × question-type interaction on M2, over the four primary post-trained arms (SFT, distill-self, DPO-from-base, GRPO) — three arms if the substitution rule removed distill-self. All five columns included, flagged cells marked, sensitivity run reported alongside.

**Why the earlier specification was invalid.** It removed row means from the observed matrix, then built its null by shuffling arm labels independently within each column. Those steps are incompatible. Suppose GRPO is one SD above every other arm in every column — a pure arm effect, no interaction. In the real matrix, row-centering subtracts that uniform advantage and the statistic is small. In a shuffled matrix, GRPO's high values scatter into a different row in each column, so no row owns them, row-centering cannot remove them, and the shuffled statistic is large. The null is inflated by exactly the main effect the statistic was meant to exclude, and real interactions become harder to detect precisely when one objective is simply better overall. Separately, "permute at the CVE-cluster level" cannot be applied to a statistic defined on 20 aggregate cells, so the plan specified two different tests.

**The fix: residualise first, permute at item level.** Main effects are removed from the _data_ before permutation, so they cannot re-enter through the shuffle, and the statistic operates on per-item scores so cluster-level permutation is meaningful.

The scale comes from outside the matrix: dividing by an SD estimated from four numbers is routinely off by a factor of two at n=4, and that error lands directly on the statistic. Use the pooled seed-level SD per column, across arms and seeds, independent of the between-arm differences being tested. And the null is empirical: standardising columns fixes total variance by construction, leaving no residual variance inside the matrix to serve as an error term.

```
S[a,t,c] = score of arm a, type t, test CVE c, averaged over seeds
           (find-the-error: the CVE's two items collapse to one paired score)
sd_t     = pooled seed-level SD for column t, across arms and seeds
z        = S / sd_t                                  # column standardisation

m[a,t]   = mean_c z[a,t,c]
grand    = mean_{a,t} m[a,t]
R[a]     = mean_t m[a,t] - grand                     # arm main effect
C[t]     = mean_a m[a,t] - grand                     # type main effect

e[a,t,c] = z[a,t,c] - R[a] - C[t] - grand            # residualised items
T        = sum_{a,t} ( mean_c e[a,t,c] )^2           # observed statistic

repeat 10,000 times:
    for each test CVE c:
        draw one permutation pi_c of the arm labels
        apply pi_c to ALL SIX of that CVE's items, across ALL FIVE types
    recompute T* from the permuted residuals

p = (1 + #{ T* >= T }) / 10001
```

One permutation per CVE applied across all six of that CVE's items preserves the within-CVE correlation cluster-level permutation exists to respect — the same test CVE generates every one of them, and shuffling them independently would treat correlated observations as independent and understate the null. The find-the-error pair is inside that cluster, so it inherits the CVE's permutation rather than receiving one of its own. Because main effects are stripped before shuffling, no arm-level advantage can leak into the null.

**Changed at step 7 (owner): the null.** The statistic T above is kept, column scaling included. The item-level permutation is replaced by a **parametric seed bootstrap**.

_Why._ Permuting arms within a test CVE models test-question noise only. Training-run noise is the seed-to-seed shift of an arm's score on a type, and averaging three seeds does not remove it, so it reads as interaction. On simulated M2 data with no interaction at all, at the real size (343 CVEs, four arms, five types, three seeds), the permutation test rejected at p < 0.05:
- 2 of 40 experiments at 0 points of training noise per cell;
- 4 of 40 at 1 point;
- 19 of 40 at 2 points;
- 40 of 40 at 4 points.

More seeds do not help, because they shrink the noise the shuffle measures as fast as the noise it misses. With 30 seeds it still rejected 10 of 20 at 2 points.

_The replacement._ Under "no interaction", each cell's expected run mean is the additive fit (grand, arm and type effects) to the standardised cell means, times sd_t. Each of 10,000 replicates (seed 0) draws all 60 runs as that mean plus Gaussian noise at the column's pooled seed SD, re-estimates every sd_t, and recomputes T\*. Then p = (1 + #{T\* ≥ T}) / 10,001. In the same simulations it rejected 2, 1, 2 and 3 of 40, and about 5% with a large arm main effect too.

_What it treats as fixed:_ the test CVEs, so the conclusion is about this test set. Run noise is taken as normal, with one SD per column across arms. Both are stated in the write-up.

Frozen at step 7, before any training file is written: `src/analysis/primary_test.py` and `mde.py`, whose sha256 is pinned as `pinned.PRIMARY_TEST_SHA256`.

### Minimum detectable effect, reported not gated

The earlier plan gated progress on a power simulation requiring 80% power before proceeding. That gate is removed: it depended on a seed-level noise estimate nothing before it produced — the signal audit samples only the frozen base checkpoint and says nothing about how much trained models wobble between seeds — so it ran on a guess, and the guess mattered, since RL seed variance is plausibly several times SFT's.

What remains is the part that answers the reviewer's question. After training, using the pooled seed SDs actually measured: simulate M2 matrices across a range of effect sizes with noise at the observed magnitude; run the frozen primary test on 1,000 matrices per effect size; report the smallest effect detected at 80% power; state that number beside the result, **whatever the result is**.

_Pinned at step 7 (owner): a single-cell effect._
- Experiments are drawn from the primary test's own null, the additive fit plus seed noise at the measured SDs, 1,000 of them.
- δ points are planted in one (arm, type) cell, on each arm in turn.
- Each experiment is read against the frozen test's null replicates.
- Over a grid of 0–50 points in quarter-point steps, the MDE is the smallest δ from which power stays ≥ 80%. It is reported per type and overall.

No GPU time, no pilot required. A significant result is stronger for carrying it; a null result becomes interpretable — "no interaction detected, and this design detects effects of X points or larger" — rather than uninterpretable.

**What is given up, in the limitations.** Test-set size is committed at 15% without a prior check that it suffices, because by the time the noise estimate exists the test window is frozen. That is the accepted risk of removing the gate, and the write-up says so rather than omitting the detectable effect when it is inconveniently large. If the census gate also forced a compressed time range, both effects push the same direction and should be reported together.

### Scope of inference

Everything else — M1, the gaps, the diagnostic columns, the secondary arm — is reported without inferential claims. With 20 cells and a second gap matrix, the multiple-comparison surface is large; the single pre-registered test prevents fishing.

The base arm is excluded: its M2 values are copied from its M1 by construction rather than produced by the M2 procedure, and including a row generated differently would partly test "does an untrained model differ from trained ones," which is not the hypothesis. Base stays as an annotated reference line. SFT→DPO is excluded because it carries a second explanation for any effect (as distill-external would have, had it run) — and distill-self joins them if substitution fired on three or more types.

Line localisation and find-the-error share the same input function, so a leave-line-loc-out run still trains on that code through find-the-error. That is cross-type transfer, which is what M2 measures, not a leak — but the two columns are not independent and shouldn't be read as separate evidence for the same objective.

Any cell where M2 exceeds M1 by more than seed variance is a signal to investigate — training-data leak, volume mismatch, negative interference, or noise — not a finding.

## On "complementary"

The design establishes divergence: whether objectives produce different skill profiles. It does not establish complementarity: whether combining them beats the best single arm. If the downstream claim is multi-teacher composition, add at least one composition run — a model trained on the union, or an inference-time ensemble — showing a combination exceeds the best single arm. Without it, "complementary" in the goal statement is stronger than what the matrices support.

The five types span roughly three dimensions. MCQ and exact-ID are the same question at different output formats; find-the-error and line localisation share the same input code. "Five distinct skills" overstates the design.

In the other direction, sharpened by the hint-conditioned pin: SFT and distill-self share an objective (token-level cross-entropy) _and_, where rationales are self-generated, a data source, differing only in whether the target contains intermediate reasoning. If distill-self beats SFT, the honest reading is "producing intermediate reasoning before answering helps here," a claim about output form rather than learning signal. The factor is best described as **training recipe** rather than objective, and the write-up should say so in the framing, not only in the limitations. Where substitution fired, even that reading is unavailable for the affected column, since the reasoning came from a different and stronger model.

## NVD

The National Vulnerability Database, run by NIST. A vulnerability receives a CVE ID (CVE-2021-44228); the CVE system is a naming registry. NVD enriches each CVE with structured analysis: a prose description, a CWE category identifying the kind of mistake (CWE-502 is deserialisation of untrusted data), a CVSS vector encoding severity as labelled components (`AV:N/AC:L/PR:N/...`), affected products, and reference links including patch commits.

The CWE relationship graph is published and machine-readable, which makes hierarchy-weighted scoring a lookup rather than a judgment call — provided view, edge types, placeholder handling and credit direction are pinned.

NVD's v3 coverage begins in late 2015. Everything earlier carries v2 only, which is why the v3.0/v3.1 pin is a constraint on the _time range_ of the fact table and not merely on its size.

## PrimeVul

A C/C++ vulnerability detection dataset: 6,968 vulnerable and 228,800 benign functions across 140 CWEs, built by merging BigVul, CrossVul, CVEfixes and DiverseVul with corrected labels, deduplication, and chronological splits. Its test split contains paired samples — a vulnerable function alongside its patch — to test whether a model can distinguish the two. It carries CVE descriptions and NVD links as metadata.

Its upstream sources reach back well before 2015, so the v3.x pin removes a real block from its old end. The census year histogram measures exactly how much.

## Why these two

Neither is sufficient alone. NVD is prose and structured labels with no code; PrimeVul is code with no descriptions. Joined on CVE ID, one row carries both, and all five question types fall out of the same fact. The paired structure does double duty: the honest metric for find-the-error and the gold line sets for localisation.

The join is sound by construction: PrimeVul was built by matching NVD descriptions to function names in fix commits, so the linkage already exists rather than being imposed.

**Verifiability.** Every type has a canonical answer and a short verifier. This is the binding constraint — GRPO cannot train without one, and no off-the-shelf cyber benchmark offers both verifiability and format diversity. It is also why _explain_ was dropped: an LLM judge or n-gram overlap against the NVD description is the one number a reviewer attacks first, and feeding judge noise into the residuals would contaminate every arm.

**Contamination control.** Building from raw NVD rather than a published benchmark means you own the split, and publication dates make it chronological rather than random. Contamination against the backbone is handled separately by the probe.

**Clean evaluation.** Keeping CTI-Bench, CyberMetric and SecBench out of training leaves them available as independent external checks.

Two caveats to verify before committing: PrimeVul excludes anything absent from NVD, which biases the sample. It does contain multi-function vulnerabilities (446 CVEs in v0.1's paired files have more than one pair), and the fact table keeps one function per CVE; and models perform poorly on it — StarCoder2 drops from 68 F1 on BigVul to 3 on PrimeVul, largely because of the paired metric. The signal audit is the check, and it decides whether the find-the-error column can rank anything for any arm before training compute is spent. Line localisation partly hedges: same code, continuous scoring, so partial overlap still separates arms where a binary label gives everyone zero.