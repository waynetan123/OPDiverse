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
| Training attention | **PyTorch SDPA**, not flash-attn. flash-attn would not build on the GPU machine: torch 2.13 is built for CUDA 13.0, and the machine's CUDA compiler is 12.8. SDPA is built into PyTorch and was step 7's HF reference attention. It cannot keep sequences apart inside one packed row, so the cross-entropy and DPO arms run **one row per forward pass** (`TRAIN_MICRO_BATCH` = 1) and build each 24-row step by gradient accumulation. There is no padding and no attention across rows, and the loss is the same. The cost: short rows can no longer share a forward pass. | user |
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
| Precision, attention | bf16 weights; PyTorch SDPA attention (owner, above); Liger kernels (fused linear cross-entropy), every arm. A 16k-token row's logits over a 152k vocabulary would not otherwise fit. | user + default |
| Batching | **One row per forward pass** for base, SFT, distill-self, DPO and SFT→DPO (a DPO row is its chosen and rejected pair). The step's 24 rows are accumulated, so there is no padding and no attention across documents. This replaces the plan's cross-example *packing*, which would put a different number of examples in each arm's step and break matched steps, and which SDPA cannot mask. GRPO's micro-batches of completions are padded and masked; their size is set by the pilot. | user + default |
| Gradient checkpointing | Set from the pilot's memory measurement (the plan's default is off). It is expected to be on: the longest rows are 16–20k tokens. | doc |
| Loss | SFT and distill-self: completion tokens only. Base: every token. All three are normalised as a token mean over the step's 24 rows. distill-self's ~400-token rationales therefore weigh more per row than SFT's ~10-token answers, as cross-entropy does. | default |
| DPO | β 0.1, sigmoid loss (TRL's defaults, written out). π_ref is the starting checkpoint with the adapter disabled. **SFT→DPO** starts from the merged SFT winner (selected LR and checkpoint), and π_ref is that model. | doc + default |
| DPO log-probs | Every 10 steps, the mean per-token log-probability of chosen and of rejected, per type, on a fixed probe of 4 pairs per type: the first in run order with at most 4,096 tokens. This is the plan's check for "both driven down". | doc + default |
| GRPO | TRL `GRPOTrainer`, vLLM colocate, 8 generations. `ROLLOUT_SAMPLING` (temperature 1, top-p 1) is enforced on every request. `scale_rewards="group"`, `num_iterations` 1, `mask_truncated_completions=False` (a cut-off rollout is scored on its text, as at step 5). Every other GRPOConfig field (β, loss type, ε, importance-sampling options, top-k, min-p) is TRL's default at the pinned TRL version. The pilot records those values as `pinned.TRL_DEFAULTS`, and a later run refuses any difference. | doc + default |
| GRPO reward | `verifiers.verify_item(type, completion, gold).dense`, the parsers v2 dense score, which is 0 on a parse failure. | doc |
| GRPO vLLM placement | `GRPO_VLLM_MODE`, set from the pilot (the plan: "decided by measurement on a single pilot run"). **Colocate:** vLLM inside the training process, on the same GPU, 1 GPU per run. **Server:** TRL's `trl vllm-serve` on a second GPU, serving the pinned backbone from a local snapshot of the pinned revision, with the trainer on the first; 2 GPUs per run. The weights, sampling and caps are the same either way. Every sweep GRPO run uses the pinned mode. | doc + default |
| Per-type caps | Colocate: every rollout request reaches vLLM with its own SamplingParams: `ROLLOUT_SAMPLING`, the row's step-5 cap (MCQ 16, exact-ID 24, CVSS 256, find-the-error 512, line localisation 512), both stop tokens and watermarking off. Server: TRL's client sends one length limit per call, so each batch is split into one call per cap, and the replies are put back in order; the stop tokens go in `generation_kwargs` where the client takes them. Either way TRL's sampling values are checked against the pins first, and the check-only call confirms no reply exceeds its cap. | doc (step 5) + default |
| Dynamic sampling | MCQ and exact-ID, the step-5 "dynamic sampling" bands. After a generation batch is scored, every group of these types whose 8 rewards tie is replaced by the next prompt of the same type, from a per-type queue in run order that wraps. The replacement is generated, scored and spliced in. A tied replacement is kept: one replacement per dropped prompt. How often each prompt was used as a replacement is logged, which is the plan's "deviation from the identical-fact-set claim". | doc + default |
| Running monitor | Every 10 steps, per type: live groups (dense and binary), reward mean and variance, parse and cut-off rates, line-localisation set sizes, and find-the-error predicted-vulnerable rate overall and by class. Events: a type below the 10% live floor; a find-the-error **collapse**, when one predicted class exceeds 90% of parsed rollouts in the window. Events are logged and flag the run; they do not stop it (plan). | doc + default |
| Weight-sync check | Every 10 steps, after the weights reach vLLM: vLLM decodes two canary prompts greedily (32 tokens), and the trainer scores those tokens under the current weights and under the starting weights (adapter off). Once the policy has moved more than 0.05 nats per token from the start, vLLM's log-probs must sit at less than half the distance from the current weights that they sit from the start. Otherwise the run stops. This catches the plan's "GRPO trains against its own past self". A server that returns no log-probs is read by greedy choices instead: on vLLM's canary tokens, the share that are the argmax under the current and under the starting weights; once either is at most 0.9, the current weights must win. **Before training**, in either mode, vLLM's greedy canary tokens must be the backbone's argmax at least 90% of the time (it is serving the pinned backbone), and where the server reports its prompt ids, they must equal ours. | doc + default |
| Guards before any step | `python -m train.run --check-only` runs these, and every run repeats them:<ul><li>the tree is clean, and the libraries and TRL defaults are the pinned ones;</li><li>the file is the frozen step-9 file (content sha256 against `converters_meta.json`);</li><li>every prompt tokenises to the bank's count;</li><li>every completion ends in exactly one `<|im_end|>` and every base document in exactly one `<|endoftext|>`;</li><li>the trainer's prepared ids equal these reference ids exactly (no truncation, no second end token);</li><li>a collated batch is padding-free, with labels exactly on the loss tokens, the end token included;</li><li>every arm's shared configuration fields are identical;</li><li>the process sees exactly one GPU (one training job per GPU; with more, transformers would split every batch across them with DataParallel);</li><li>GRPO: one optimizer step holds 24 prompts × 8 completions, and one call through the cap wrapper returns no more than each cap.</li></ul> | default |
| DPO end token | Whether TRL appends the end token itself is not assumed. The run prepares the pairs with the end token, and if TRL adds a second one, prepares them without it. The prepared ids must equal the reference ids either way. TRL versions also name the prepared columns differently, and some store chosen and rejected after the prompt; the check accepts either and stops on anything else, listing the columns it found. `run_meta.json` records the end-token form and the column layout. | default |
| GRPO hooks into TRL | Where TRL keeps the vLLM engine, and what its weight-sync method is called, change between versions. The trainer searches itself for the one `vllm.LLM` instance (server mode: the one `VLLMClient`), and looks for the sync method by name on the trainer and then on the engine's holder. It also requires `_generate_and_score_completions`. Each hook must be found, or the trainer stops before any step and lists what this TRL has instead; no hook can be silently skipped. `run_meta.json` records where the engine and the sync hook were found. | default |
| GRPO log-prob scoring | TRL 1.14.2's Liger path (`_chunked_logps`) ignores `batch_size` and runs the model over every row it is given at once. Before each training step that is the whole generation batch (192 completions of up to ~17k tokens), which asked for 68 GiB in the pilot. The GRPO trainer feeds it the rows in batches of the micro-batch and joins the results. Each row's numbers are unchanged, and Liger stays on as pinned. `run_meta.json` counts the chunked calls. | default |
| GRPO padding | TRL left-pads every prompt of a generation batch to the longest one. In run order, a step mixes code prompts of up to ~16k tokens with ~130-token label prompts, so a one-row micro-batch was ~16k columns, nearly all padding. In the cost pilot, step 1 took 29 minutes for 178,592 real tokens. Before each forward pass (scoring and training), the leading columns that are padding in every row of the micro-batch are cut off; the completion columns are never cut. This changes no token's attention: padding is masked, and rotary positions enter only through distances between tokens. A one-row micro-batch then starts at position 0, as vLLM placed it when sampling, so results match up to bf16 rounding. `run_meta.json` records the padding columns cut. | default |
| Truncation arguments | TRL's `max_prompt_length` and `max_completion_length` exist only to switch truncation off, and some TRL versions have dropped them (and the truncation with them). They are passed only where the installed config has them; `run_meta.json` records which were absent. SFT and DPO still compare the prepared ids with the reference, so any truncation would stop the run. Every pinned argument stays required. | default |
| SFT labels | The prepared SFT dataset carries `labels`: the token ids on the loss tokens, −100 elsewhere (base: every token). TRL uses them as given; the collated-label check confirms it. | default |
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
- **Memory.** Each arm at the middle LR (5e-5), `--longest-first` (the longest rows train first), a few steps, with gradient checkpointing off and then on. The worst cases are the 19,796-token base document, DPO pairs of about 2 × 16k, a line-localisation GRPO group (8 × about 16.5k) and a 16.4k distill-self row. This sets `GRADIENT_CHECKPOINTING`, GRPO's micro-batch, and the vLLM memory fraction (at least about 0.45 in colocate mode).
- **Cost.** About 20 steps per arm in pilot order, giving measured GPU-hours per step. Before the pilot, the token counts suggest roughly 3–9 GPU-h for a 399-step run of the cross-entropy and DPO arms, and **20–30 GPU-h for GRPO**. That is 2–3× the plan's budget per GRPO run.
- **Versions and defaults.** `TRAIN_LIBS` (torch, transformers, trl, peft, accelerate, datasets, liger-kernel, vLLM) and `TRL_DEFAULTS`.

**GRPO placement:** colocate is measured first. If GRPO does not fit beside its vLLM on one card, or is too slow, it is measured in server mode (`--vllm-mode server`, runbook below), and the result sets `GRPO_VLLM_MODE`. Server mode doubles GRPO's GPUs per run, so at most two GRPO runs go at once on 4 × A40.

**After the pilot**, the owner pins `TRAIN_STEPS` from the measured hours. `GRADIENT_CHECKPOINTING`, `GRPO_VLLM_MODE`, `TRAIN_LIBS` and `TRL_DEFAULTS` are set from the pilot's run metas. The pilot outcome below is recorded and committed before any sweep run.

## Runbook

1. Commit the code, the pins, this record and the plan-document edits.
2. `PYTHONPATH=src python -m sweep prepare`. Already written; the rerun is byte-identical: `dev_requests.jsonl` `e27e5f9b…39a85082`, `checkpoint_requests.jsonl` `afe19ded…667b3944`.
3. On the GPU machine, at that commit:
   - `pip install -r requirements-gpu.txt`, holding the installed torch, transformers and vLLM where they are (a constraints file of their `pip freeze` lines). flash-attn is not used;
   - copy `data/` over: `bank/`, `combined_dataset/`, `converters/`, `mitre_cwe/`, `sweep/*_requests.jsonl`.
4. **Pilot**, for each arm A in base, sft, distill_self, dpo and grpo:
   - `python -m train.run --arm A --lr 5e-5 --pilot --max-steps 8 --longest-first --gradient-checkpointing off --check-only`, then without `--check-only`;
   - the same with `--gradient-checkpointing on` if off runs out of memory;
   - then a cost run of 20 steps without `--longest-first`.

   GRPO in server mode, if colocate does not fit: download the pinned snapshot once, `hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --local-dir models/qwen2.5-7b-instruct-a09a354`. Then start `CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model models/qwen2.5-7b-instruct-a09a354 --dtype bfloat16 --max-model-len 16985 --port 8000`, and once it reports ready run `CUDA_VISIBLE_DEVICES=1 python -m train.run --arm grpo --lr 5e-5 --pilot --vllm-mode server --vllm-server-port 8000 --max-steps 8 --longest-first --gradient-checkpointing on`. A second concurrent server needs its own `--port` and the trainer its own `--vllm-server-port` and `--vllm-group-port`.

   Record the outcome below. The owner pins `TRAIN_STEPS`; set `GRADIENT_CHECKPOINTING`, `GRPO_VLLM_MODE`, `TRAIN_LIBS` and `TRL_DEFAULTS`; commit.
5. **Sweep runs**, one per GPU, GRPO first: `python -m train.run --arm A --lr X` for A in base, sft, distill_self, dpo and grpo, and X in the grid. Each run ends with a `run_meta.json`; an interrupted run resumes from its latest checkpoint. `python -m train.launch` starts one queue of runs per GPU, inside tmux. It refuses GPUs that already have a process on them, and keeps a live progress line per GPU. A failed run is marked with its error line and does not stop the other runs.
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
- Training attention is PyTorch SDPA, not Flash Attention 2 (owner: flash-attn would not build against CUDA 13).
- The cross-entropy and DPO arms run one row per forward pass instead of cross-example packing, so every arm's step holds the same 24 rows.
