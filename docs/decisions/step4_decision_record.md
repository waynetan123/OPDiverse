# Step 4 decision record — question bank

Pre-registration for generating the question bank (procedure step 4). The pins are implemented in:
- `src/etl/pinned.py` (templates, MCQ rules, the external model);
- `src/etl/verifiers.py` (the three new verifiers);
- `src/generators/` (the bank and the external runner).

**They must be committed before the first request to the external model**, so the rules are verifiably fixed before any output exists. `generators.external` refuses to run on a dirty tree. "User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## What the bank is

The bank has six items per CVE across five types, built once over each pool and then frozen:
- 1,933 non-test CVEs give 11,598 items;
- 343 test CVEs give 2,058 items.

Every later artifact is keyed by `item_id = "{cve_id}:{type}:{index}"`: distill-self rationales, external traces, DPO negatives, seed partitions and evaluation. Every arm sees the same `prompt` for an item. Only what sits beside it differs.

| Type | Prompt shows | Target (what SFT trains on) |
|---|---|---|
| `mcq` | NVD description and four options such as `A. CWE-787: Out-of-bounds Write` | `ANSWER: B` |
| `exact_id` | NVD description | `CWE-787` |
| `cvss` | NVD description | `AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H` |
| `find_error` (0 = vulnerable, 1 = patched) | The stored function, not numbered | `VULNERABLE: yes, CWE-787` or `VULNERABLE: no` |
| `line_loc` | NVD description and the vulnerable function, numbered with `render_numbered` | `LINES: 12, 13, 17` (ascending, deduplicated) |

The prompt is Qwen's chat rendering of the user message (`probe.prompts.render_qwen_chat`, the rendering the probe checked byte-for-byte against the pinned revision).

## Pins

| Pin | Value | Source |
|---|---|---|
| Prompt context | MCQ, exact-ID and CVSS see the description only. Find-the-error sees the bare function only. Line localisation sees the description and the numbered function. No prompt shows the CVE ID. | user |
| Templates | `pinned.BANK_PROMPTS`, `BANK_TEMPLATE_VERSION = "v1"`. Each ends "End your reply with … in the form …", not "reply with only …", so one byte-identical prompt serves arms that answer directly (SFT) and arms that reason first (distill). The parsers read the end of the reply. The line-localisation template states the 1-indexed numbering and the insertion convention: added code counts against the nearest code line above, and blank and comment-only lines never count. | default |
| Literal CWE IDs | `CWE-<n>` in a description becomes `CWE-[redacted]` in every prompt (11 descriptions). The stored fact is unchanged. | user |
| Own CVE ID | Removed wherever it appears (`CVE-[redacted]`): 1 description (CVE-2018-5745) and 5 patched functions that cite their own CVE in a comment. Redaction stays within a line, so line numbers are unchanged. | default |
| CWE names in words | Kept. 427 descriptions contain the gold CWE's name ("out-of-bounds read"). That is the signal the label types are meant to use. | default |
| MCQ option text | `{letter}. {cwe}: {name}`, with the name from the MITRE 4.20 XML, never from model text. | default |
| MCQ distractors | Proposed by the external model and admitted by rule (next section). This replaces the plan's "distractors from the sibling field". Measured on non-test, sibling distractors let "pick the option most often gold" score 0.895, against a chance rate of 0.25. | user |
| Gold letter | `MCQ_LETTERS[stable_rank(cve_id, "mcq-letter") % 4]`. Distractors fill the other letters in `stable_rank(cve_id, "mcq-slot", cwe)` order. | doc + default |
| Shortcut guard | **0.5** (`MCQ_SHORTCUT_MAX`). Details below. | user |
| Line matcher | A maximum one-to-one matching within ±1. Ties between maximum matchings go to the most exact hits, then the lowest lines. F1 = 2·TP / (\|pred\| + \|gold\|). The tie-break changes which pairs are matched, never how many, so F1 does not depend on it. This settles the plan text, which said both "maximum matching" and "exact hits first, then ±1". Those disagree: gold {11, 12} against prediction {12, 13} scores 1.0 under the first and 0.5 under the second. | user |
| MCQ parser | Lenient: the capital letter after the last `ANSWER:`; else a reply that is a single letter; else the last `(X)`, `option X` or `answer is X`. Capitals only, so "answer: a buffer overflow" is not read as A. Strict: exactly `ANSWER: X`. Metric and dense score are both 0/1. | default |
| Find-the-error parser | Label: the yes/no after the last `VULNERABLE:`. Without that field, the last vulnerability statement in the reply decides, with negations ("not vulnerable", "no vulnerability") checked first. CWE: the last CWE ID in the reply. Strict: exactly `VULNERABLE: yes, CWE-<n>` or `VULNERABLE: no`. | default |
| Find-the-error scores | `metric` is per-function label accuracy. `dense` is 0.5·label + 0.5·label·hierarchy credit on the vulnerable item, and the label on the patched item (plan). The headline is `paired_accuracy`; `paired_cwe_accuracy` is the stricter column. | doc |
| Line-localisation parser | The plan's six steps over `[1, n_lines]`. Integers found but all out of range parse as the empty set; no integers and no `none` is a failure. Gold is never empty (step 1 dropped empty patches), so an empty prediction scores F1 = 0. | doc |
| Parser version | Stays **v1**: the three verifiers were added before v1 scored any bank output. | default |
| Trivial baselines | Always-A; the most frequent CWE; the best constant under the adopted schedule; the majority CVSS vector; always-vulnerable, per-function and paired; every k-th line for k = 2 and 3 (lines 1, 1+k, 1+2k, …); keywords (every line containing `memcpy`, `strcpy` or `alloc`, case-sensitive, else `LINES: none`). All are scored by the same verifiers the arms use. | doc + default |
| `EVAL_MAX_MODEL_LEN` | **16,985**: the longest bank prompt (line localisation, 16,473 tokens) plus the 512-token evaluation cap. The probe ran at 1,024. `bank check` fails if this stops covering the bank. The longest prompt does not depend on MCQ options. | default |
| Freeze | `bank_meta.json` records the sha256s of `facts.jsonl`, `split.jsonl`, the templates, the tokenizer and the MCQ generations. `build` refuses to write any output byte that differs from an existing bank, and `check` fails if any input changed. A rebuild under a different `PYTHONHASHSEED` is byte-identical. | default |

## MCQ construction

1. **Request.** There is one request per CVE (2,276), built by `bank prepare-mcq` with `MCQ_REQUEST_PROMPT`.
   - The input is the redacted description plus the gold CWE's ID and XML name.
   - The model returns up to 8 wrong-but-plausible CWE IDs as structured output (`{"distractors": [...]}`), ranked, and preferring weaknesses common in C and C++ code.
   - It is told not to propose gold or anything more general or more specific than gold.
2. **Admission.** A proposal is kept only if all of these hold:
   - it is written `CWE-<n>`;
   - it is a live view-1000 weakness;
   - it is not gold;
   - it is not an ancestor or descendant of gold, because either would be a second defensible answer;
   - it is not a duplicate.

   The first 3 admitted IDs are kept, in the model's order. Every rejection is logged with its reason: `malformed`, `not_live_weakness`, `gold`, `ancestor`, `descendant` or `duplicate`.
3. **One regeneration.** If attempt 1 was refused, failed, or gave fewer than 3 admitted IDs, the same request is sent once more (`prepare-mcq --retry`). Attempt 1's admitted IDs are kept first, then attempt 2's.
4. **Prior-matched draw.** Any slots still empty are filled by drawing without replacement.
   - The candidates are the CWEs that are gold in non-test, under the same exclusions.
   - Each is weighted by how often it is gold in non-test. Test labels never enter.
   - The arithmetic is exact integer: slot s picks by `stable_rank(cve_id, "mcq-draw", s)` modulo the remaining weight.
   - The report gives how many distractors came from each source, per pool. The write-up states that share rather than claiming model-written options throughout.
5. **Shortcut guard.**
   - The shortcut picks, on each non-test MCQ item, the option most often gold in non-test, and scores 1/|ties| when gold is among the top options.
   - **If its mean over non-test is above 0.5, every MCQ item in both pools is rebuilt from the prior-matched draw alone.** The model's picks are kept on record in `mcq_decisions.jsonl` as `discarded_model_decision`.
   - This happens before any training, so nothing downstream sees the discarded version.
   - The draw alone scores **0.348** on the real non-test pool. My closest offline stand-in for model picks (the nearest non-test labels) scored 0.68, so the guard may fire.
6. **Pilot.** The 100 non-test CVEs with the lowest `stable_rank(cve_id, "mcq-pilot")` go first. The pilot checks the prompt, refusals and the admission rate, and gives an early reading of the shortcut. **It decides nothing.** The guard runs on the full non-test bank.
7. **Caveats, stated.**
   - The external model sees test descriptions and gold labels, but only to build items.
   - It chose the options that look plausible to itself, so distill-external's MCQ column, and any MCQ comparison against it, is not neutral.

## External model

**Claude Opus 5.5** (`pinned.EXTERNAL_MODEL`) is used for every external job: MCQ distractors (step 4), and distill-external traces, DPO rejected completions and distill-self substitution (step 6).

| Pin | Value | Source |
|---|---|---|
| Model | `claude-opus-5-5`. GPT o1 was considered and rejected: it returns no reasoning, OpenAI forbids extracting it, and the owner's permission for written-out reasoning covers Claude. | user |
| Effort | `output_config.effort = "medium"`, set explicitly even though it is this model's default | user |
| Thinking | Adaptive, which cannot be disabled on this model. The display is left at `omitted`, so thinking text is neither returned nor stored. | default |
| Sampling | None. The model rejects `temperature`, `top_p`, `top_k` and a seed, and has no dated snapshot. **Outputs are cached and audited, not regenerated.** Every row records `model`, `message_id`, `request_id`, `stop_reason`, `usage` and the custom ID. `run_meta.json` records the transport, concurrency, commit, SDK version and dates. | default |
| Transport | **Messages API**, called concurrently by `python -m generators.external` (8 requests at a time by default, `--workers`; the SDK retries 429, 5xx and connection errors up to 8 times with backoff). Each result is checkpointed to `<prefix>_generations.partial.jsonl` as it arrives, and an interrupted run resumes without re-sending finished requests. A request that still fails in transport is re-sent by the next run, so it never uses up the item's one regeneration; a refusal is the model's answer and is kept. `custom_id = item_id` with `:` → `_`, plus `_a{attempt}`. Results are matched by `custom_id`, never by position. **Changed after the pilot** from Message Batches to direct calls (owner): the full batch took hours with no results. The request parameters and model are unchanged, so the pilot's results (made through Batches) stand. Cost: about twice the batch price, roughly $16 for the full run. | user |
| Refusals | No server-side fallbacks, which would silently switch models. A `refusal` is logged with its `stop_details.category`, regenerated once, then filled by the rule. | default |
| Request file | `data/bank/mcq_requests.jsonl`, 2,276 requests, sha256 `5627c5c3…b41ffd3`. Pilot: `data/bank/pilot/mcq_requests.jsonl`, 100 requests, sha256 `e834f104…96ab6225`. | — |

## Line-numbering check

The plan asks for a 50-item manual check before the bank is built. It exists to catch one failure: model-facing line numbers shifted by N from the gold lines, which would silently invalidate the whole line-localisation column. **Owner decision: the gate is closed on the 42 items where the gold lines equal an independent `git diff`, and the remaining 8 are not read by hand.**

- **The 8 disagreeing sample items are explained, and none is a shift.** Each difference is one of two kinds:
  - comment lines, which git counts and we mask by design (items 9, 18, 25, 28, 43, 50);
  - a repeated line (`}`, `return;`, a repeated call) matched differently (items 9, 11, 48, 50).

  The classifications are recorded in `verification/line_sheet.csv`.
- **Whole-table evidence** comes from the git cross-check in `bank_report.md`, over all 2,276 CVEs:
  - exact agreement: 1,872 (82.2%);
  - differ only on comment lines: 118;
  - differ only in where inserted code is anchored: 253;
  - other repeated-line alignments: 33;
  - replaced and deleted gold lines all found by git too: **2,243 of 2,276 (98.6%)**. A numbering error would shift these lines, so this is the test that matters.
- **The pinned ±1 matcher.** Scored against our gold with it, git's code lines reach mean F1 0.963.
- **Automated guard in `bank check`**, applied to every CVE:
  - gold lines are in range and are code lines;
  - `patch_line_set` still reproduces them from the stored functions;
  - line i of every numbered prompt is exactly `f"{i}: {line}"`.
- **Residual ambiguity, stated.** On 52 CVEs git anchors an insertion more than one line from ours, which scores F1 = 0 under ±1:
  - 32 because git anchors it on a comment line, where our convention uses the code line above;
  - the rest because an insertion can slide two lines past a closing `}`.

  This is ambiguity in where a diff places inserted code, not a numbering error. See "For later steps".

## Dry run on the real data (pre-registration)

`generators bank build --dry-mcq` fills every MCQ from the prior-matched draw, so it needs no external request. It is what the bank becomes if the guard fires. The run used `facts.jsonl` sha256 `a9aaf284…9e5a08b` and `split.jsonl` sha256 `179defb4…070534d8`. `bank check` found **0 violations**.

| | |
|---|---|
| Items | Non-test 11,598 (1,933 CVEs); test 2,058 (343) |
| Gold-letter marginals | Non-test A 479 / B 500 / C 449 / D 505; test A 85 / B 88 / C 91 / D 79 |
| Prompt tokens, median (p100) | MCQ 189 (693) · exact-ID 125 (632) · CVSS 139 (646) · find-the-error 898 (10,000) · line localisation 1,312 (16,473) |
| Baselines, non-test | MCQ always-A 0.248 · MCQ most familiar option 0.348 · exact-ID CWE-125 0.155 exact / 0.236 hierarchy · CVSS majority 0.719 · always-vulnerable 0.500 per function / 0.000 paired · every 2nd line 0.111 · every 3rd line **0.140** · keywords 0.042 |

Every 3rd line beats the keyword heuristic because, under ±1 tolerance, it lands within one line of every line of the function. This is the plan's "sparse spraying" trick, and it is why every-k-th-line is a baseline row.

## Runbook

1. Commit the code, the pins, this record and the plan-document edits.
2. `PYTHONPATH=src python -m generators bank prepare-mcq --pilot` (already written; the rerun is byte-identical).
3. On a machine with `pip install -r requirements-external.txt` and API credentials, at that commit:
   `PYTHONPATH=src python -m generators.external --requests data/bank/pilot/mcq_requests.jsonl`, then `python -m generators bank pilot-report`. Record the result below. The pilot's `usage_totals` give the cost estimate for the full run.
4. `python -m generators.external --requests data/bank/mcq_requests.jsonl`, then `python -m generators bank prepare-mcq --retry`, then `python -m generators.external --requests data/bank/mcq_retry_requests.jsonl` (skip the last command if the retry file is empty). If a run is interrupted, or ends reporting failed requests, rerun the same command: only requests without a result are sent.
5. `python -m generators bank build`, then `check`, then `report`. Record the outcome below. The bank is now frozen.

## Outcome

From `data/bank/pilot/pilot_report.md`, `data/bank/bank_report.md` and `bank_meta.json`. The pins were committed before the first external request: the pilot ran at clean commit `8539539` (finished 2026-09-30 23:10 UTC), and the full run and regeneration at clean commit `3cacfc9` (2026-10-01 20:53–22:26 UTC). `bank check` found **0 violations**.

| | |
|---|---|
| Pilot | 100 / 100 `ok`, all with three admissible on the first attempt. Rejections: 7 ancestor, 7 descendant. No refusals. Shortcut 0.490. Decided nothing. |
| MCQ: shortcut with model picks / guard | **0.440** (2,551 / 5,799) against the cut-off 0.5: **not fired, model picks kept.** The draw alone scores 0.348. |
| MCQ: distractor sources (model / regeneration / draw), refusals | Non-test 5,790 / 9 / 0; test 1,023 / 3 / 3 (one test item has drawn distractors). Attempt 1: 2,276 `ok`; 5 items regenerated, all `ok`. Rejections: 179 descendant, 100 ancestor, 48 malformed, 4 duplicate, 3 not a live weakness, 1 gold. **No refusals.** |
| Usage | Full run 1.38M input / 0.52M output tokens; regeneration 2.8k / 0.4k. Every reply `end_turn`, model `claude-opus-5-5`, no transport errors. |
| Bank | Non-test 11,598 items (1,933 CVEs), test 2,058 (343). Gold letters: non-test A 479 / B 500 / C 449 / D 505; test A 85 / B 88 / C 91 / D 79. Baselines unchanged from the dry run except MCQ most-familiar-option 0.440. |
| Bank sha256s | `bank_nontest.jsonl` `4d03cc38…0a327a4d`; `bank_test.jsonl` `6f6fbb5d…c7c3d4ad`; `mcq_decisions.jsonl` `4626e671…bf831dcb`; `bank_meta.json` `63fbea47…503d1131`. Inputs: `mcq_generations.jsonl` `09fda8e4…1be876a2`, `mcq_retry_generations.jsonl` `a7be7f35…15776655`, templates `c895958a…80681ca9`. |

## For later steps

- **Step 6, distill-external traces.** Opus 5.5 never returns its hidden reasoning. The owner has permission to ask Claude to write its reasoning out in the reply, so a trace will be that written reasoning followed by the answer. The write-up must call it written-out reasoning, not raw chain of thought, which replaces the plan's preference for "an open-weight reasoning model exposing its full chain-of-thought". A written-reasoning request can still be declined under `reasoning_extraction`, so step 6 pins the fallback for refused traces.
- **Step 12, line localisation.** Also report F1 against git's gold lines, as the plan's "report under both conventions if they differ materially", because the insertion anchor is ambiguous on 52 CVEs.
- **Step 7.** `VLLM_VERSION` and the watermarking question are unchanged. `EVAL_MAX_MODEL_LEN` is now 16,985.

## Differences from the experiment plan (now reflected in its text)

- MCQ distractors come from the external model under admission rules and a shortcut guard, not from the sibling field.
- The line matcher is the maximum matching. The "exact hits first" sentence now describes the tie-break only.
- The prompt context is split by source. Only find-the-error and line localisation carry a function, so three of the six items per CVE are short.
- The external model is Claude Opus 5.5 and has four jobs, not three. Its files are audited, not regenerated identically.
- distill-external traces are written-out reasoning, not a full chain of thought.
