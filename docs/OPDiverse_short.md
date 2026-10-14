# OPDiverse — one-day run

A reduced version of the experiment, cut to fit a hard limit of **one day of GPU access: 4 × A40 for 24 hours, 96 GPU-hours** in total. The full experiment needs roughly 480–1,380 GPU-hours, so this is not a scaled-down copy of it. It keeps one thing intact, the pre-registered primary test, and drops everything else.

All costs here are estimates from token counts until the pilot at the start of the day measures them. The step-count rule below is written so that the measurement, not a guess, sets the run length.

## The question it still answers

Four arms train the same backbone (Qwen2.5-7B-Instruct, LoRA) on the same facts with different learning signals:

- **SFT:** imitate the answer;
- **distill-self:** imitate a reasoning trace;
- **DPO:** prefer a good answer over a near-miss;
- **GRPO:** maximise a verified reward.

**M2** trains each arm with one of the five question types hidden, then tests it on that hidden type: MCQ, exact-ID, CVSS, find-the-error and line localisation.

The **primary test** asks whether the arms differ in *which* types they handle well, beyond what chance and overall ability explain. This is the arm × type interaction on M2. It measures chance by how much a cell moves between seeds, so it needs at least two seeds. The frozen code in `src/analysis/` accepts two seeds and three or four arms, so it runs unchanged and its pinned hash stays valid.

## Full experiment vs one-day run

| | Full experiment | One-day run |
|---|---|---|
| **Arms** | base, SFT, distill-self, DPO, GRPO, plus SFT→DPO | SFT, distill-self, DPO, GRPO (GRPO dropped if it doesn't fit, see the fallback) |
| **Learning rate** | Swept over {1e-5, 5e-5, 2e-4} per arm at seed 0 (18 runs) | Fixed at **5e-5** for every arm, no sweep |
| **M2** (one type hidden, the primary test) | 4 arms × 5 types × 3 seeds = 60 runs | 4 arms × 5 types × **2 seeds** (0, 1) = **40 runs** |
| **M1** (nothing hidden) | 5 arms × 3 seeds | Not run |
| **M1-volume** (the control for M2's data loss) | 4 arms × 2 shares × 3 seeds = 24 runs | Not run |
| **Base noise floor** | 3 runs | Not run |
| **SFT→DPO** (secondary arm) | 3 seeds | Not run |
| **Run length** | 200 or 399 steps (half or one pass) | **S steps, set by formula after the pilot**: about 60–100 expected, so each run sees about a fifth of its rows |
| **Snapshots saved** | 4 per run | 4 per run (unchanged) |
| **Snapshot choice** | Each of the 4 checked on dev, best one kept | **Last snapshot used**. The other three stay on disk, unevaluated. |
| **Test evaluation** | Every run, all five types | Every run, all five types (unchanged) |
| **Analysis** | Primary test, MDE, M1/M2 gaps against M1-volume, descriptive tables | Primary test and MDE only |
| **Total runs** | 117 | **40** |
| **GPU-hours** | ~480–760 (half pass), ~890–1,380 (one pass) | **96** |
| **Wall clock on 4 GPUs** | ~6–10 days (half pass), ~12–18 days (one pass), plus 2–3 weeks contingency | **1 day** |

## What it can and cannot claim

**Can claim:**
- whether the four objectives produce distinct skill profiles on hidden question types (the primary test, p-value);
- the minimum detectable effect (MDE): the smallest per-cell difference this design would catch 80% of the time. It is reported beside the result whatever the result is.

**Cannot claim:**
- **Skill vs format.** "Is an advantage real skill, or familiarity with the output format?" needs M1 and M1-volume, which are not run.
- **Arms tuned fairly.** Every arm uses the same untuned learning rate. If an arm does badly, its learning rate being wrong for it can't be ruled out, and the write-up must say so.
- **Fully trained models.** At about a fifth of a pass, the models are lightly trained. Scores are lower than after a full pass, and smaller differences may not show.
- **Sensitivity.** Two seeds instead of three, and short training, make the MDE larger. A null result is weaker than it would be in the full design.

## Design

### Runs

Every combination of:
- arm ∈ {`sft`, `distill_self`, `dpo`, `grpo`};
- config ∈ {`m2-mcq`, `m2-exact_id`, `m2-cvss`, `m2-find_error`, `m2-line_loc`};
- seed ∈ {0, 1}.

That is 40 runs, all at LR 5e-5 and `S` steps, on the frozen step-9 files. Every other training pin is unchanged from step 10: LoRA, optimizer, batching, GRPO settings, monitor and weight-sync check.

### Run length: the step-count formula

This is fixed before the day, so the run length depends only on measured cost, never on results.

From the pilot's cost runs, for each arm `a`:
- `t_a` = median seconds per step, over steps 3–10;
- `o_a` = fixed per-run overhead (model load, vLLM start, guards) = run time − steps × `t_a`.

Also:
- `E` = merge plus test evaluation per run, fixed at 0.25 h;
- `H` = hours left in the window once the pins are committed;
- `R` = 3 h reserve.

`S` is the largest multiple of 4, at most 399, that satisfies both conditions:

1. **Total work fits on 4 GPUs:** `10 × Σ_a (o_a + E) + 10 × S × Σ_a t_a ≤ 4 × (H − R)`
2. **The busiest GPU fits.** Ten GRPO runs on four GPUs means one GPU runs three: `3 × (o_grpo + S × t_grpo + E) ≤ H − R`

### Fallback: dropping GRPO

Drop GRPO and recompute `S` over the other three arms if any of these holds:
- `S < 48`;
- GRPO does not fit in colocate mode. Server mode needs a second GPU per run, which the day cannot afford.

The primary test then runs on three arms with `--arms sft distill_self dpo`, which the frozen CLI allows. GRPO is reported as dropped for compute.

### Snapshots and evaluation

- Each run saves its 4 snapshots as pinned, at round-half-up(k · S / 4).
- Only the last snapshot (`checkpoint-S`) is merged and evaluated, on the full test set: 2,058 items, 343 CVEs, all five types.
- The merged weights (~15 GB) are deleted after evaluation. The adapters stay (~0.16 GB per snapshot); each snapshot's optimizer state (~0.32 GB) is deleted once its run has finished training (see "Changes during the run").
- The other snapshots can be checked on dev later if more GPU time appears. That would be reported as an addition, not as part of this run.
- Dev is not used at all on the day.

## Budget

| Item | GPU-h |
|---|---|
| Setup checks and pilot (memory plus cost) | ~6 |
| 40 training runs | ~65–70 |
| 40 merges plus test evaluations (~15 min each) | ~10 |
| Reserve: one crash and rerun | ~12 |
| **Total** | **96** |

GRPO is about 70% of the training time. An 80-step GRPO run is about 4–6 hours, if the post-padding-fix step time is 3–4.5 minutes as estimated. The pilot measures this, and the formula absorbs it.

## Before the day (no GPU)

Every GPU entry point refuses a modified tree, so everything below must be committed before the day starts.

1. **Commit pending work:** the current changes to `src/train/grpo.py`, `src/train/launch.py` and `tests/test_launch.py`.
2. **Record the deviations** in a decision-record entry: the list at the end of this document, the step formula and the GRPO fallback rule.
3. **Pin the learning rate.** `pinned.SWEEP_LR` = 5e-5 for every arm, marked as fixed without a sweep.
4. **Build what doesn't exist yet,** with tests (built: `src/short/`, `tests/test_short.py`):
   - **Test requests:** `python -m short prepare` builds `data/short/test_requests.jsonl` from `data/bank/bank_test.jsonl`, in step 7's evaluation request format, as `sweep prepare` does for dev.
   - **Test scoring:** `python -m short score` turns each run's test generations into the analysis input rows `{"arm", "seed", "item_id", "metric", "dropped_type"}`, reusing the verifiers and `unit_scores`, plus a per-run summary.
   - **Per-run jobs:** `train.launch` takes config and seed per job (`grpo@5e-5@m2-cvss@1`, run directory `grpo_m2-cvss_s1`), and `python -m short.job` runs the commands under "Each run" below in sequence: train, merge, test evaluation, delete the merged weights.
   - **Queue builder:** `python -m short queue` orders the runs longest estimated first (GRPO first, so a GRPO failure surfaces while there is still time to rerun it) and forecasts each GPU's load. The launcher runs them as one shared queue: whichever GPU frees first takes the next run, so real run times decide the assignment (owner's choice, replacing an up-front per-GPU assignment; it changes no run's content). A failed run goes back to the end of the queue once (`--retries 1`).
   - **Step formula:** `python -m short steps` reads the cost pilot's `train_log.jsonl` (now timed per step, `elapsed_s`) and `run_meta.json` files and prints `S`, both conditions and whether the fallback fires. The constants are pinned (`pinned.SHORT_*`).
5. **Prepare the GPU machine:**
   - install `requirements-gpu.txt` with torch, transformers and vLLM held where they are;
   - copy `data/` (`bank/`, `combined_dataset/`, `converters/`, `mitre_cwe/`, `short/test_requests.jsonl`). `combined_dataset/` holds `split.jsonl`, which the analysis needs;
   - download the pinned backbone snapshot;
   - start a tmux session.
6. **Dry-run everything off the GPU:** the builders, the scoring on a fake generation file, the queue script with a stand-in runner.

## On the day

Times are hours from the start of the GPU window.

**0:00–0:20 — Checks.**
- Confirm the tree is clean and all four GPUs are idle (`nvidia-smi`).
- Run `--check-only` for one run per arm:

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m train.run --arm grpo --lr 5e-5 --pilot \
    --max-steps 4 --gradient-checkpointing on --check-only
```

**0:20–0:50 — Memory pilot.** The longest rows go first. Gradient checkpointing is on: the longest rows are 16–20k tokens, and there is no time to test it off.

```
PYTHONPATH=src python -m train.launch \
    --queue 0=grpo@5e-5 --queue 1=dpo@5e-5 --queue 2=distill_self@5e-5 --queue 3=sft@5e-5 \
    --out-root data/short/pilot/mem -- --pilot --max-steps 4 --longest-first --gradient-checkpointing on
```

If GRPO runs out of memory in colocate mode, the fallback fires. Skip the GRPO cost run.

**0:50–1:40 — Cost pilot,** 10 steps per arm in run order:

```
PYTHONPATH=src python -m train.launch \
    --queue 0=grpo@5e-5 --queue 1=dpo@5e-5 --queue 2=distill_self@5e-5 --queue 3=sft@5e-5 \
    --out-root data/short/pilot/cost -- --pilot --max-steps 10 --gradient-checkpointing on
```

**1:40–2:00 — Pin and commit.**
- Run the step formula on the cost pilot's logs: `PYTHONPATH=src python -m short steps --hours-left <H>`. It records `S` and whether the fallback fired in `data/short/step_count.json`.
- Set and commit the pins:
  - `TRAIN_STEPS = S`;
  - `GRADIENT_CHECKPOINTING = True`;
  - `GRPO_VLLM_MODE = "colocate"`;
  - `TRAIN_LIBS` and `TRL_DEFAULTS`, from the pilot's `run_meta.json`.
- Build the queue: `PYTHONPATH=src python -m short queue`. It prints the launch command below.

**2:00–~21:00 — The 40 runs,** one shared queue, one job per GPU at a time, GRPO first:

```
PYTHONPATH=src python -m train.launch --plan data/short/queues.json --gpus 0 1 2 3 --retries 1 \
    --runner short.job --out-dir data/short/runs -- --scratch /dev/shm/opdiverse
```

Each run (directory `data/short/runs/<arm>_<config>_s<seed>`) is one `short.job`, which runs:

```
CUDA_VISIBLE_DEVICES=<g> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python -m train.run \
    --arm <arm> --lr 5e-5 --config <config> --seed <seed> --out data/short/runs/<arm>_<config>_s<seed>
rm data/short/runs/<arm>_<config>_s<seed>/trainer/checkpoint-*/optimizer.pt     # only once all S steps are done
CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=src python -m train.merge \
    --adapter data/short/runs/<arm>_<config>_s<seed>/trainer/checkpoint-<S> \
    --out /dev/shm/opdiverse/<arm>_<config>_s<seed>/merged
CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=src python -m evaluate.run_vllm --requests data/short/test_requests.jsonl \
    --model /dev/shm/opdiverse/<arm>_<config>_s<seed>/merged --out data/short/runs/<arm>_<config>_s<seed>/final
rm -rf /dev/shm/opdiverse/<arm>_<config>_s<seed>/merged    # also if the evaluation fails; merge_meta.json is kept in final/
```

While the runs go, watch the monitor events: a type below the 10% live-group floor, or a find-the-error collapse. They flag a run but do not stop it.

A failed run resumes from its latest snapshot. The launcher requeues it once, at the end of the queue. If it fails again, `short queue` rebuilds a queue of the runs not yet evaluated, to relaunch on the free GPUs.

**~21:00–24:00 — Reserve.** For a rerun. If nothing failed, it is unused. Don't start new work that might not finish.

**The primary test needs all 40 cells** (or all 30 under the fallback). A partial matrix cannot be tested, so a run that can't finish in time is worth more as a rerun slot than as extra work.

## After the day (no GPU)

1. Score every run's test generations into `data/short/m2_scores.jsonl`: `PYTHONPATH=src python -m short score` (add `--arms sft distill_self dpo` only if `step_count.json` is absent; it defaults to that file's arms). `score_summary.md` holds the diagnostic columns and monitor events.
2. Run the frozen analysis:

```
PYTHONPATH=src python -m analysis primary-test --scores data/short/m2_scores.jsonl --seeds 0 1 --out data/short/analysis
PYTHONPATH=src python -m analysis mde --scores data/short/m2_scores.jsonl --seeds 0 1 --out data/short/analysis
```

   - Add `--arms sft distill_self dpo` if the fallback fired.
   - Add `--flagged arm:type ...` for floored cells, as the plan specifies.
3. Report:
   - the p-value with the MDE beside it;
   - per-arm tokens, FLOPs and GPU-hours from `run_meta.json`;
   - the monitor events and dynamic-sampling replacements for GRPO;
   - the four diagnostic columns of each M2 run.

## Deviations from the pre-registered plan

To be recorded in a decision-record entry before any GPU run:

1. **No LR sweep.** One learning rate, 5e-5, for every arm, replacing step 10's sweep. The plan's tuning-parity argument against a shared learning rate is stated as a limitation.
2. **Two seeds, not three.**
3. **Run length set by a cost formula.** `S` is fixed by the formula above from measured cost, not chosen from 1 or 3 passes. About a fifth of a pass is expected, so each run sees only part of its file. All arms see the same rows in the same order.
4. **No checkpoint selection.** The last snapshot is evaluated. All four are saved.
5. **M1, M1-volume, the base arm and its noise floor, and SFT→DPO are not run.** The skill-vs-format analysis is out of scope.
6. **The GRPO fallback rule** above, fixed in advance and based on cost only.
7. **The pilot is trimmed:**
   - gradient checkpointing is on without an off-run;
   - GRPO micro-batch stays at 1;
   - GRPO colocate only, with no server-mode measurement.

## Changes during the run

Recorded before relaunching and before any test output is read. None changes what a run computes.

**First launch.** Only GRPO had started (the queue puts it first), and it failed four ways:
- `data/short/test_requests.jsonl` had not been copied to gpu17, so two evaluations failed (exact-ID s1, find-the-error s1).
- Their merged weights stayed on disk and filled it, so two merges failed for lack of space (CVSS s1, line localisation s0). All four had finished training.
- CVSS s0, exact-ID s0 and find-the-error s0 ran out of GPU memory between steps 15 and 27. Each failed to allocate 596 MiB while PyTorch held 1.17 GiB reserved but free, which is fragmentation. They resume from checkpoint-14.
- The weight-sync check stopped both MCQ runs: s0 at step 10 (gap ratio 0.52 against the 0.5 limit) and s1 at step 20 (ratio 1.24, vLLM nearer the starting weights). Fifteen checks in six other GRPO runs passed with a verdict (ratios 0.04–0.43), so sync works and these are treated as failed runs. Both are rerun from scratch as the plan's one retry. mcq_s1 is not resumed from checkpoint-14, since that snapshot sits between a passing and a failing check. The failed attempts are kept in `data/short/failed/`. If either rerun fails the check again, the owner decides before scoring.

**Changes (owner):**
1. **Merged weights in memory.** `short.job --scratch /dev/shm/opdiverse` merges into gpu17's `/dev/shm` (126 GB, in RAM) instead of `final/merged`. The run may use at most 52 GB of disk. The weights are identical, and scoring checks them by sha256, not by path. The merged weights are now deleted whether the evaluation succeeds or fails, so a failure cannot fill the disk again.
2. **Optimizer states deleted once a run finishes training.** When `run_meta.json` records all `S` steps, `short.job` deletes `optimizer.pt` (~0.32 GB) from each of the run's snapshots. This leaves 33 runs × 4 snapshots at about 21 GB instead of about 66 GB. Every adapter stays, so every snapshot can still be merged and evaluated, but no finished run can be trained further. A run that has not finished keeps its optimizer state to resume from.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (`pinned.SHORT_CUDA_ALLOC_CONF`) for every training stage from the relaunch on. It changes how PyTorch lays out GPU memory, not what is computed. `run_meta.json` now records it as `cuda_alloc_conf`. Runs finished in the first launch ran without it.
4. **Relaunch with `--retries 0`.** The MCQ reruns are already the one retry, and an out-of-memory run would fail on the same row again. A run that fails is left to the owner.

**Second launch.** 37 of 40 runs finished, with all 30 SFT, distill-self and DPO runs and 7 of 10 GRPO runs evaluated. The weight-sync check stopped GRPO mcq_s0 at step 20 (ratio 1.10), mcq_s1 at step 20 (0.88) and line_loc_s1 at step 30 (0.77). No test output had been read when the decision below was made.

The evidence, from the training logs only:
- **The rollouts came from the current weights.** TRL compares vLLM's log-probs with the trainer's on all 192 rollouts of every step (`sampling/sampling_logp_difference/mean`). At step 1, when the two hold identical weights, it reads 0.015–0.025, the two engines' numerical floor. It stays at 0.012–0.043 to the end in every GRPO run, the three failed runs included (0.016–0.031 up to their failure). A stale vLLM would make it grow with the policy's drift: cvss_s0 drifted 0.29 on the canary and stayed at about 0.03.
- **The failed canaries sit between the starting and the current weights.** In each failure, gap_current + gap_start ≈ drift (mcq_s1: 0.051 + 0.057 = 0.108), so vLLM's canary log-probs lie between the two on the canary tokens. The canary prompts are the same at every check and are training prompts. vLLM's prefix cache keeps the attention it computed over a prompt it has seen, under the weights it had then. If the check runs before that cache is emptied, the canary is scored partly under earlier weights, while the rollouts, generated after TRL resets it, are not.

**Change (owner):**
5. **The canary runs on an emptied prefix cache.** `sync_check` resets vLLM's prefix cache before the canary that decides. The rule is unchanged: drift above 0.05, then gap_current < 0.5 × gap_start. The same canary also runs first on the cache as the sync left it, and is recorded as `before_cache_reset` beside each check. The check decides on the post-reset canary only. The first record of each run (`vllm_engine`) states whether prefix caching is on. The three runs resume from their latest snapshots (checkpoint-14, -14, -28). The explanation is a hypothesis: if a resumed run fails the check again, it is not resumed a second time, and the primary test runs on the three other arms (`--arms sft distill_self dpo`), with GRPO reported as excluded for failing the weight-sync check.
