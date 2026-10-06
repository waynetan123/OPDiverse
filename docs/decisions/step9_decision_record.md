# Step 9 decision record — converters

Procedure step 9 writes the training files: one per arm, per matrix configuration, per seed. Each file selects that seed's train items from the frozen bank, the step-8 partitions and the cached step-5 and step-6 files. Nothing is generated and no GPU is used.
- Pins: `src/etl/pinned.py`, section "Converters (step 9)".
- Code: `src/converters/`.
- Commands: `PYTHONPATH=src python -m converters build`, then `check`, then `report`. They write to `data/converters/`.

"User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## What this step does

Every arm sees the same prompt for an item: the bank's byte-identical `prompt`. Only what sits beside it differs.

| Arm | Row (beyond `item_id, cve_id, type, index, copy`) | From |
|---|---|---|
| sft | `prompt`, `completion` = gold target + `<|im_end|>` | bank |
| distill_self | `prompt`, `completion` = step-5 target + `<|im_end|>`, `source` (`self_a1` / `self_a2` / `gold_only`) | `distill_self.jsonl` |
| dpo | `prompt`, `chosen` and `rejected` (each + `<|im_end|>`), `rejected_source`, `rejected_dense` | `dpo.jsonl` |
| grpo | `prompt`, `gold` (the bank's dict, as `verifiers.verify_item` takes it), `max_completion_tokens`, `band`, `dynamic_sampling` | bank, `step5_report.json` |
| base (M1 only) | `cve_id`, `text` = one plain document + `<|endoftext|>` | `facts.jsonl`, MITRE XML |

SFT→DPO trains on the `dpo` file of the same configuration and seed, since it differs from DPO-from-base only in initialisation. `manifest.json` maps it there rather than duplicating the file. There is no distill-external converter (step 6).

## Pins

| Pin | Value | Source |
|---|---|---|
| Base arm text | One document per train CVE, holding the facts every other arm trains on and no question or answer format (`BASE_DOC_TEMPLATE`): the description as the prompts show it (literal CWE IDs and the CVE's own ID redacted); `Weakness: CWE-n: <MITRE name>`; the CVSS v3 base vector; the vulnerable function; the patched function (own-CVE redaction as in the prompts). It ends in `<|endoftext|>`, with no chat template and no CVE ID. Alternatives considered: context only, without labels; and labels plus a unified diff instead of the patched function. | user |
| Configurations | **All built now, for seeds 0–4**, frozen before any training: `m1`; `m2-<type>` for each of the five types; `m1v-1of6` and `m1v-1of3`. That is 8 × 5 = 40 file sets and 165 files. Steps 14 and 18 read them instead of re-running step 9. | user |
| M1-volume shares | **1/6 and 1/3, replacing the plan's 20%.** A CVE has six items, so dropping one type removes 1/6 of them, or 1/3 for find-the-error. `m1v-1of6` is the comparator for the M2 loops on MCQ, exact-ID, CVSS and line localisation; `m1v-1of3` for find-the-error. Each pair then removes and duplicates exactly the same number of items. | user |
| M2 mask | Every train item of the dropped type. Dev drops it too, at checkpoint selection (`manifest.json` `dev_types`). Test keeps all five types. | doc |
| M1-volume mask | The N·f train items with the lowest `stable_rank(item_id, "m1v-mask", f, seed)`: uniform over items, as the plan says, and redrawn per seed. N = 9,576 is a multiple of 6, so N·f is exact. | doc + default |
| Upsampling | R items survive. Each gets ⌊N/R⌋ copies, and the N mod R items with the lowest `stable_rank(item_id, "upsample", config, seed)` get one more. On the real data that is 1 or 2 copies. Duplicates are explicit rows (`copy` 0, 1), so the trainer needs no custom sampler, and every configuration has exactly M1's row count. | doc + default |
| One selection per configuration | The mask, copies and order depend only on (item_id, configuration, seed), so every arm of a configuration and seed has the identical (item_id, copy) rows. | default |
| Row order | A fixed shuffle, `stable_rank(item_id, copy, "order", config, seed)`, so that sequence packing never groups a CVE's six items. Base documents use the same rule with copy 0. The trainer still shuffles with its own run seed. | default |
| End tokens | Completions end in `<|im_end|>` (`COMPLETION_END`), the token evaluation stops on; it tokenises to that single id. Base documents end in `<|endoftext|>` (`BASE_DOC_END`). The trainer must not append a second one. | default |
| GRPO fields | `max_completion_tokens` is the step-5 cap per type (MCQ 16, exact-ID 24, CVSS 256, find-the-error 512, line localisation 512). `dynamic_sampling` is true where step 5 banded the type "dynamic sampling" (MCQ, exact-ID). | doc (step 5) |
| Base in other configurations | Base is built for M1 only: it has no items to mask, and its M2 equals its M1 by construction. | doc |
| Storage | `seed{s}/{config}/{arm}.jsonl.gz`, gzip with no file name and a zeroed timestamp. Uncompressed, the 165 files would be several GB of repeated prompts. | default |
| Upstream guards | `build` refuses unless all of these hold:<ul><li>the bank's `facts` and `split` hashes match (`load_bank`);</li><li>`distill_self.jsonl` and `dpo.jsonl` are the files whose sha256 `step5_report.json` and `step6_report.json` recorded, from the same bank;</li><li>no type was substituted;</li><li>`partition.json` was drawn from the current facts and split;</li><li>`partition.jsonl` has exactly the pinned seeds;</li><li>distill-self, DPO and the partition cover exactly the non-test bank.</li></ul> | default |
| Freeze | A rebuild that would change any file's content is refused (`converters_meta.json` records each file's content sha256, gzip sha256 and row count). | default |
| `check` | Re-derives the invariants from the inputs and the plan, not from the builder's selection code. It returns violations and exits 1 on any:<ul><li>M1 is every train item once;</li><li>no dev or test item appears anywhere;</li><li>M2 has no dropped-type item and every other train item;</li><li>M1-volume masks exactly N·f;</li><li>copies stay within one of each other;</li><li>every file has M1's row count;</li><li>rows are identical across arms;</li><li>prompts are the bank's;</li><li>each completion is right: SFT is the target; distill-self is the step-5 target, ending in gold; DPO chosen is the target and rejected is step 6's strict near miss with dense < 1;</li><li>GRPO gold and caps match;</li><li>base has one document per train CVE, its labels, no CVE ID and no question;</li><li>M1-volume masks differ across seeds.</li></ul> | default |

## Run count

M1-volume becomes 4 post-trained arms × 2 shares × 3 seeds = **24** runs instead of 15, so the required total is **123** instead of 114. The plan's 15 counted five arms. Base has nothing to mask, so it has no M1-volume run. If the owner keeps a base row there, the total is 129.

## Runbook

1. Commit the code, the pins, this record and the plan-document edits.
2. `PYTHONPATH=src python -m converters build`, which takes about 6 minutes; a rebuild of identical files takes about 1 minute.
3. `python -m converters check` (0 violations), then `python -m converters report`.
4. Record the outcome below.

If the build is interrupted, rerun it: files already written are compared, not rewritten.

## Outcome

From `data/converters/converters_report.md` and `converters_meta.json`. The pins and code were committed at `a496cab` (branch `step9_converters`), and the build ran at that clean commit. A trial build from the same code in a scratch directory before the commit gave a byte-identical `converters_meta.json`, and a rebuild compares every file as identical. **`check`: 0 violations.**

| | |
|---|---|
| Files | **165**: 5 seeds × (M1's 5 arms + 7 configurations × 4 question arms), 1.3 GB gzipped. |
| Rows per seed | **9,576 in every question-arm file** (1,596 train CVEs × 6). **1,596 base documents.** No dev or test item in any file. |
| M2, one-item types | 7,980 distinct items, 1,596 duplicated rows. |
| M2, find-the-error | 6,384 distinct items, 3,192 duplicated rows. |
| M1-volume | `m1v-1of6`: 7,980 distinct items, 1,596 duplicated. `m1v-1of3`: 6,384 distinct, 3,192 duplicated. The masks are spread over every type, with find-the-error about twice as often as the others. Mask overlap between seeds has Jaccard 0.074–0.085 (1/6) and 0.168–0.191 (1/3), as independent draws over largely shared train sets should. |
| Tokens per epoch, M1 | Prompts 9.14–9.34M per question arm. Completions: SFT 0.11M; distill-self 3.42–3.43M; DPO 0.22M (chosen + rejected). Base 4.76–4.87M document tokens. GRPO's completions are generated in training. |
| Longest sequence | Base **19,796**; SFT, distill-self and DPO **16,481**; GRPO 16,985 (the longest prompt plus its 512 cap). |
| distill-self gold-only share (M1, by seed) | MCQ 1.3–1.4%, exact-ID 0.1%, CVSS 4.8–5.3%, **find-the-error 17.2–17.6%, line localisation 27.3–28.1%**. |
| DPO rule-built share (M1) | MCQ 100% (rule-defined), exact-ID 0%, CVSS 1.8–2.1%, find-the-error 0%, line localisation 1.0–1.2%. |
| End tokens | Under the pinned tokenizer every checked completion ends in the single `<|im_end|>` id, and each base document in the single `<|endoftext|>` id. |

**sha256s**

| File | sha256 |
|---|---|
| `manifest.json` | `13832f62…2b9b11e1` |
| `converters_meta.json` | `d961779b…07dba4f2` (holds every training file's content and gzip sha256) |
| `converters_report.json` | `e274968d…13d40b25` |
| `seed0/m1/sft.jsonl.gz` (content) | `159963e1…271b206e` |
| `seed0/m1/base.jsonl.gz` (content) | `6fc152c5…d4cb2766` |

## For later steps

- **Step 10, trainer setup.**
  - Read only `data/converters/manifest.json`. It gives each (arm, configuration, seed) file, the dev types for checkpoint selection, and the 150-CVE checkpoint subsample.
  - Assert that no second end token is appended.
  - Set the maximum sequence length to at least each arm's longest sequence in `converters_report.md`, so that nothing is truncated. Base documents are the longest, because they hold two functions.
- **Step 10/11, GRPO.**
  - TRL's `GRPOTrainer` has one `max_completion_length`, but the per-type caps run from 16 to 512 tokens; how they are applied needs a pin.
  - So does the implementation of dynamic sampling for MCQ and exact-ID.
- **Step 10/11, matched steps.** Base has 1,596 documents per seed against 9,576 items for the question arms. The optimizer step count is set in the trainer, the same for every arm; it does not follow from file length.
- **Report beside every cell:**
  - the distill-self gold-only share, which is high on line localisation and find-the-error;
  - the DPO rule-built share, with MCQ rule-defined.
- **Step 14 and step 18** use the `m2-*` and `m1v-*` files built here. Nothing is re-run at step 9.

## Differences from the experiment plan (now reflected in its text)

- The base arm's raw text is defined: one plain document per train CVE, holding its facts.
- M1-volume runs at two shares, 1/6 and 1/3, instead of 20%. The run count is 123, not 114.
- Step 9 builds the M2 and M1-volume files as well as M1's, so steps 14 and 18 read frozen files.
