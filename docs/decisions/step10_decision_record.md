# Step 10 decision record — LR sweep

Pre-registration for procedure step 10: the learning-rate sweep at M1, seed 0, for every arm, and the selection rule that picks one LR per arm. It is the first step that trains models. The pins are implemented in:
- `src/etl/pinned.py` (the "LR sweep (step 10)" section);
- `src/train/` (the training runs and the merge);
- `src/sweep/` (the evaluation requests, the type scale, selection and the report);
- `src/evaluate/run_vllm.py`, which now serves a merged checkpoint (`--model`, `--out`).

**They must be committed before any GPU run**, so the rules are verifiably fixed before any output exists. Every GPU entry point refuses a dirty tree. "User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## What this step does

- **Runs.** 6 arms × 3 LRs = **18 runs**, on seed 0's M1 files from step 9: base, SFT, distill-self, DPO, GRPO and SFT→DPO.
  - SFT→DPO trains on the `dpo` file, starting from the merged SFT winner, so its 3 runs follow SFT's selection.
- **Evaluation.**
  - Each run saves 4 checkpoints. Each is merged and evaluated on seed 0's 150-CVE checkpoint subsample (900 items).
  - The best checkpoint is then evaluated on full dev (337 CVEs, 2,022 items).
  - Test is never read.
- **Selection.**
  - Per arm, the LR whose chosen checkpoint has the highest full-dev statistic wins.
  - The winner is that arm's LR for every later run (steps 11, 15 and 18).
  - The winning seed-0 run is also that arm's M1 seed-0 run.

## Owner decisions

| Question | Decision | Source |
|---|---|---|
| The selection scale (step 7's open item: there is no pooled seed SD at seed 0) | **Per-CVE spread.** For each type, the ddof-1 SD over CVEs of the per-CVE score (find-the-error: the paired score) in each evaluation, pooled as the root mean of those variances over the 60 scale evaluations (base, SFT, distill-self, DPO and GRPO × 3 LRs × 4 checkpoints, on the checkpoint subsample). Frozen before any choice. Reused for checkpoint selection at steps 11 and 15. | user |
| LR grid | **{1e-5, 5e-5, 2e-4}**, identical for every arm. If a winner lands on an edge, the grid is not extended; the edge is flagged. | user |
| Run length | **Pinned after the pilot**, from measured GPU-hours, before any sweep run. One pass over a question-arm file is 399 steps; the published default of 3 epochs would be 1,197. `pinned.TRAIN_STEPS` stays `None` until then, and every non-pilot run refuses to start. | user |
| Seed-0 winners | **Reused as the M1 seed-0 keepers.** Retraining would repeat the same data, LR and run seed. The run count falls from 123 to **117**. The seed-0 keeper is "the best of three on dev"; the write-up says so. Test plays no part in the choice. | user |

### Why the scale is the per-CVE spread

The statistic averages five types that sit on different curves. Example: one arm at LRs A and B scores CVSS 72 → 74 (+2) and find-the-error 10 → 8 (−2).
- The raw mean calls it a tie.
- A typical CVE sits about 15 points from the mean on CVSS (partial credit) and about 30 on find-the-error (0 or 1).
- In those units it is +0.13 against −0.07, so B wins.

The plan's yardstick, the pooled seed SD, needs several seeds. At seed 0 the per-CVE spread is measurable, the same for every arm, and fixed before any choice.

## Pins

### Training

| Pin | Value | Source |
|---|---|---|
| Matched steps | Every arm takes `TRAIN_STEPS` optimizer steps of **24 rows** each (`TRAIN_EXAMPLES_PER_STEP`). For GRPO a row is a prompt with 8 rollouts, so 192 completions per step. One pass over a question-arm file (9,576 rows) is 399 steps. Base cycles its 1,596 documents: about 6 passes per question-arm pass, so each CVE is seen about 6 times, as the question arms see its six items. | doc + default |
| Run order | Ours, not the trainer's. Pass p over the file is sorted by `stable_rank(key, "train-order", seed, p)`, with key = (item_id, copy), or cve_id for base. Every question arm of a configuration and seed holds the same rows, so step i trains on the same items in every arm. The trainers' samplers are sequential (TRL GRPO: `shuffle_dataset=False`). This is step 9's "the trainer shuffles with its own run seed", done once for every arm. | default |
| Checkpoints | 4, at round-half-up(k · S / 4): 100, 200, 299, 399 at S = 399. | doc + default |
| LoRA | r 16, alpha 16, dropout 0, no bias, on q, k, v, o, gate, up and down projections, every arm. alpha = r makes the scale 1; the LR sweep absorbs it. | doc + default |
| Optimizer | AdamW (`adamw_torch`), linear decay to 0 over the run, no warmup, weight decay 0, betas 0.9 / 0.999, eps 1e-8, gradient-norm clip 1.0: transformers' defaults, written out. | default |
| Precision, attention | bf16 weights; Flash Attention 2; Liger kernels (fused linear cross-entropy), every arm. A 16k-token row's logits over a 152k vocabulary would not otherwise fit. | doc + default |
| Batching | **Padding-free** within a step: a micro-batch's sequences are concatenated with `position_ids`, so there is no padding and no attention across documents. This replaces the plan's cross-example *packing* for the cross-entropy arms. Packing would put a different number of examples in each arm's step and break matched steps. | default |
| Gradient checkpointing | Set from the pilot's memory measurement (the plan's default is off). It is expected to be on: the longest rows are 16–20k tokens. | doc |
| Loss | SFT and distill-self: completion tokens only. Base: every token. All three are normalised as a token mean over the step's 24 rows. distill-self's ~400-token rationales therefore weigh more per row than SFT's ~10-token answers, as cross-entropy does. | default |
| DPO | β 0.1, sigmoid loss (TRL's defaults, written out). π_ref is the starting checkpoint with the adapter disabled. **SFT→DPO** starts from the merged SFT winner (selected LR and checkpoint), and π_ref is that model. | doc + default |
| DPO log-probs | Every 10 steps, the mean per-token log-probability of chosen and of rejected, per type, on a fixed probe of 4 pairs per type: the first in run order with at most 4,096 tokens. This is the plan's check for "both driven down". | doc + default |
| GRPO | TRL `GRPOTrainer`, vLLM colocate, 8 generations. `ROLLOUT_SAMPLING` (temperature 1, top-p 1) is enforced on every request. `scale_rewards="group"`, `num_iterations` 1, `mask_truncated_completions=False` (a cut-off rollout is scored on its text, as at step 5). Every other GRPOConfig field (β, loss type, ε, importance-sampling options, top-k, min-p) is TRL's default at the pinned TRL version. The pilot records those values as `pinned.TRL_DEFAULTS`, and a later run refuses any difference. | doc + default |
| GRPO reward | `verifiers.verify_item(type, completion, gold).dense`, the parsers v2 dense score, which is 0 on a parse failure. | doc |
| Per-type caps | Every rollout request reaches vLLM with its own SamplingParams: `ROLLOUT_SAMPLING`, the row's step-5 cap (MCQ 16, exact-ID 24, CVSS 256, find-the-error 512, line localisation 512), both stop tokens and watermarking off. TRL's own params are checked against the pins first. | doc (step 5) + default |
| Dynamic sampling | MCQ and exact-ID, the step-5 "dynamic sampling" bands. After a generation batch is scored, every group of these types whose 8 rewards tie is replaced by the next prompt of the same type, from a per-type queue in run order that wraps. The replacement is generated, scored and spliced in. A tied replacement is kept: one replacement per dropped prompt. How often each prompt was used as a replacement is logged, which is the plan's "deviation from the identical-fact-set claim". | doc + default |
| Running monitor | Every 10 steps, per type: live groups (dense and binary), reward mean and variance, parse and cut-off rates, line-localisation set sizes, and find-the-error predicted-vulnerable rate overall and by class. Events: a type below the 10% live floor; a find-the-error **collapse**, when one predicted class exceeds 90% of parsed rollouts in the window. Events are logged and flag the run; they do not stop it (plan). | doc + default |
| Weight-sync check | Every 10 steps, after the weights reach vLLM: vLLM decodes two canary prompts greedily (32 tokens), and the trainer scores those tokens under the current weights and under the starting weights (adapter off). Once the policy has moved more than 0.05 nats per token from the start, vLLM's log-probs must sit at less than half the distance from the current weights that they sit from the start. Otherwise the run stops. This catches the plan's "GRPO trains against its own past self". | doc + default |
| Guards before any step | `python -m train.run --check-only` runs these, and every run repeats them:<ul><li>the tree is clean, and the libraries and TRL defaults are the pinned ones;</li><li>the file is the frozen step-9 file (content sha256 against `converters_meta.json`);</li><li>every prompt tokenises to the bank's count;</li><li>every completion ends in exactly one `<|im_end|>` and every base document in exactly one `<|endoftext|>`;</li><li>the trainer's prepared ids equal these reference ids exactly (no truncation, no second end token);</li><li>a collated batch is padding-free, with labels exactly on the loss tokens, the end token included;</li><li>every arm's shared configuration fields are identical;</li><li>GRPO: one optimizer step holds 24 prompts × 8 completions, and one call through the cap wrapper returns no more than each cap.</li></ul> | default |
| DPO end token | Whether TRL appends the end token itself is not assumed. The run prepares the pairs with the end token, and if TRL adds a second one, prepares them without it. The prepared ids must equal the reference ids either way. `run_meta.json` records which. | default |
| Record per run | `run_meta.json`: library versions, GPU, the resolved configuration and its sha256, the training file's content sha256, steps done, tokens (forward, loss, generated), a FLOPs estimate (6N per trained token, 2N per reference-only token), GPU-hours, peak memory, every adapter's sha256, monitor events, weight-sync checks and dynamic-sampling replacements. `train_log.jsonl` holds the trainer's logs, the monitor, the DPO log-probs and each check. | doc + default |

### Evaluation and selection

| Pin | Value | Source |
|---|---|---|
| Merge | `python -m train.merge`: the adapter is merged into its starting weights in fp32, cast to bf16 and saved as safetensors. `merge_meta.json` records the adapter's and the weights' sha256. The merged directory is deleted after its evaluations (about 15 GB each), except the SFT winner, which SFT→DPO starts from. | doc (step 7) |
| Evaluation | `python -m evaluate.run_vllm --model <merged> --out <step dir>`: step 7's runner, with the same pins (`EVAL_SAMPLING`, greedy, 512 tokens, both stop tokens, `generation_config="vllm"`, watermarking off, every prompt's ids checked). The tokenizer stays the pinned backbone's; run_meta records the checkpoint's sha256. Without `--model` it behaves exactly as at step 7. | doc |
| Requests | `data/sweep/dev_requests.jsonl` (2,022 items, 337 CVEs) and `checkpoint_requests.jsonl` (900, 150 CVEs, a subset of dev), built by `python -m sweep prepare` from the bank and seed 0's partition. They use step 7's evaluation request rows. The command asserts the subsample sits inside dev, dev outside train and test, and agreement with the converters' manifest. | doc + default |
| Score unit | One per CVE per type: the item's `metric`, or paired accuracy for find-the-error (step 7's `unit_scores`). | doc |
| Statistic | The mean over the five types of (mean per-CVE score ÷ the type's scale). Every dev CVE has all five types, so this is also the mean over CVEs of a per-CVE standardised score. | user + default |
| Freeze of the scale | `python -m sweep scale` computes it from all 60 scale evaluations and writes `scale.json`. Its values are then pinned as `pinned.SELECTION_SCALE` in a commit **before** `python -m sweep select` runs: `data/` is not in git, so the commit is the verifiable freeze. `select` refuses to run until they agree. SFT→DPO is not in the scale; its runs come after it. | user + default |
| Checkpoint choice | The highest statistic on the checkpoint subsample; an exact tie goes to the earlier step. | doc + default |
| LR choice | The highest full-dev statistic at each run's chosen checkpoint; an exact tie goes to the lower LR. | doc + default |
| Reported beside it | For each other LR against the winner: the paired difference and its 95% cluster bootstrap interval over CVEs (10,000 resamples, seed 0). Edge winners. The raw backbone on full dev as a reference row. Base will likely pick the LR that does least harm to the question formats it never trains on; the write-up says so. | doc + default |
| Freeze of choices | A choice in `selection.json` cannot change: `select` refuses if a re-evaluation would move one. | default |

## Pilot (decides nothing about results)

Its outputs never enter selection and go to `data/sweep/pilot/`. It records:
- **Memory.** Each arm at the middle LR (5e-5), `--longest-first` (the longest rows train first), a few steps, with gradient checkpointing off and then on. The worst cases are the 19,796-token base document, DPO pairs of about 2 × 16k, a line-localisation GRPO group (8 × about 16.5k) and a 16.4k distill-self row. This sets `GRADIENT_CHECKPOINTING`, each arm's micro-batch, and the vLLM memory fraction (at least about 0.45 in colocate mode).
- **Cost.** About 20 steps per arm in pilot order, giving measured GPU-hours per step. Before the pilot, the token counts suggest roughly 3–9 GPU-h for a 399-step run of the cross-entropy and DPO arms, and **20–30 GPU-h for GRPO**. That is 2–3× the plan's budget per GRPO run.
- **Versions and defaults.** `TRAIN_LIBS` (torch, transformers, trl, peft, accelerate, datasets, liger-kernel, flash-attn, vLLM) and `TRL_DEFAULTS`.

**Server mode:** the per-type cap wrapper works in TRL's colocate mode, the plan's expected choice on 4 × A40. Measuring server mode would first need the wrapper extended to TRL's vLLM client. If colocate is too slow, that becomes an owner decision, recorded here.

**After the pilot**, the owner pins `TRAIN_STEPS` from the measured hours. `GRADIENT_CHECKPOINTING`, `TRAIN_LIBS` and `TRL_DEFAULTS` are set from the pilot's run metas. The pilot outcome below is recorded and committed before any sweep run.

## Runbook

1. Commit the code, the pins, this record and the plan-document edits.
2. `PYTHONPATH=src python -m sweep prepare`. Already written; the rerun is byte-identical: `dev_requests.jsonl` `e27e5f9b…39a85082`, `checkpoint_requests.jsonl` `afe19ded…667b3944`.
3. On the GPU machine, at that commit:
   - `pip install -r requirements-gpu.txt`;
   - copy `data/` over: `bank/`, `combined_dataset/`, `converters/`, `mitre_cwe/`, `sweep/*_requests.jsonl`.
4. **Pilot**, for each arm A in base, sft, distill_self, dpo and grpo:
   - `python -m train.run --arm A --lr 5e-5 --pilot --max-steps 8 --longest-first --gradient-checkpointing off --check-only`, then without `--check-only`;
   - the same with `--gradient-checkpointing on` if off runs out of memory;
   - then a cost run of 20 steps without `--longest-first`.

   Record the outcome below. The owner pins `TRAIN_STEPS`; set `GRADIENT_CHECKPOINTING`, `TRAIN_LIBS` and `TRL_DEFAULTS`; commit.
5. **Sweep runs**, one per GPU, GRPO first: `python -m train.run --arm A --lr X` for A in base, sft, distill_self, dpo and grpo, and X in the grid. Each run ends with a `run_meta.json`; an interrupted run resumes from its latest checkpoint.
6. For every run and each of its 4 checkpoints:
   - `python -m train.merge --adapter <run>/trainer/checkpoint-N --out <run>/stepN/merged`;
   - `python -m evaluate.run_vllm --requests data/sweep/checkpoint_requests.jsonl --model <run>/stepN/merged --out <run>/stepN`;
   - delete the merged directory unless it is needed again (full dev, or the SFT winner).
7. `python -m sweep scale`, then pin its values as `pinned.SELECTION_SCALE` and commit.
8. `python -m sweep select`. It chooses each run's checkpoint and lists the full-dev evaluations still needed. Run them:
   - `python -m evaluate.run_vllm --requests data/sweep/dev_requests.jsonl --model <run>/stepN/merged --out <run>/stepN`;

   then run `select` again, which gives the LRs of the five arms.
9. **SFT→DPO:** keep the SFT winner's merged directory. Run `python -m train.run --arm sft_dpo --lr X --init <sft winner>/stepN/merged` for the three LRs. Evaluate as in steps 6 and 8, then run `select`.
10. The raw-backbone reference: `python -m evaluate.run_vllm --requests data/sweep/dev_requests.jsonl --out data/sweep/backbone`.
11. `python -m sweep report`. Record the outcome below, set `pinned.SWEEP_LR`, and commit.

## Pilot outcome

*Pending.*

## Outcome

*Pending.*

## For later steps

- **Step 11.**
  - M1 seeds 1–2, M2 and M1-volume use `pinned.SWEEP_LR`, the same training pins and `TRAIN_STEPS`.
  - Checkpoint selection uses the same statistic with the frozen `SELECTION_SCALE`. M2 uses the four retained dev types (`manifest.json` `dev_types`).
  - The seed-0 M1 keepers are this step's winners.
- **Step 12:** each M1 seed-0 keeper is evaluated on test from its winning checkpoint.
- **Report beside every cell:** the distill-self gold-only share and the DPO rule-built share (step 9), and for GRPO the monitor's events and dynamic-sampling replacements.

## Differences from the experiment plan (now reflected in its text)

- Selection standardises each type by its per-CVE spread, not the pooled seed SD, which does not exist at seed 0.
- The grid is {1e-5, 5e-5, 2e-4}, and run length is pinned from the pilot.
- The seed-0 winners are the M1 seed-0 keepers: 117 runs, not 123.
- The cross-entropy arms use padding-free batching instead of cross-example packing, so every arm's step holds the same 24 rows.
