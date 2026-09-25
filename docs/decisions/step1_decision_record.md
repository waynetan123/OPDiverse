# Step 1 decision record — fact table

Pins for building the fact table (procedure step 1). All of them are implemented in `src/etl/pinned.py`, and none may change after the test window is drawn. "User" marks a decision the experiment owner made. "Default" marks one proposed during planning and accepted with the plan.

## Inputs

| Input | Pin |
|---|---|
| PrimeVul | v0.1 release: `primevul_{train,valid,test}_paired.jsonl` only (4,704 pairs, 4,057 CVEs). The unpaired files and `file_info.json` are not used, because only the paired files say which patched function belongs to which vulnerable one. |
| NVD | Yearly JSON 2.0 feeds `nvdcve-2.0-{2002..2026}.json.gz`. Each file is checked against its `.meta` sha256 and recorded with its feed `timestamp` in the manifest. |
| CWE | MITRE CWE 4.20 (2026-04-30), `data/mitre_cwe/cwec_v4.20.xml`, committed to the repo. |
| Backbone tokenizer | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`. `tokenizer.json` sha256 `c0382117…87539`. Counted via `tokenizers` without special tokens. |
| Diff engine | Python `difflib.SequenceMatcher(autojunk=False)` under Python 3.14; the version is recorded in the manifest. |

## Pins

| Pin | Value | Source |
|---|---|---|
| CWE source | NVD's own weakness blocks, taking every block with `source == "nvd@nist.gov"` and never filtering on `type` (NVD's own entry is sometimes `Secondary`). | user |
| CWE resolution | Strip placeholders → map deprecated → dedupe → drop if more than one remains (`multi_cwe`) → drop if not a non-deprecated view-1000 weakness (`category`, `view`, `not_in_view_1000`). Categories count toward the multi-CWE check; only NVD placeholders are stripped first. | user + default |
| Deprecated IDs | `DEPRECATED_REPLACEMENT` covers all 25 deprecated weaknesses in 4.20. A replacement is recorded only where MITRE's description names exactly one successor; six entries map to `None` and drop (216, 217, 365, 373, 458, 545). Two entries need an owner's eye: 596 → 1023 (MITRE says "closest equivalent"), and 225 → 199, which is a category, so those rows still drop. | default |
| CVSS | NVD-sourced v3.1 if present, else NVD-sourced v3.0. An NVD v3.0 beats a CNA v3.1. CNA-only v3, v2-only, v4-only and no-CVSS rows drop with separate reasons. Conflicting NVD vectors, invalid vectors, prefix/version mismatches, and vectors disagreeing with NVD's decomposed fields also drop. | user + default |
| Description, date | NVD English description; NVD `published` timestamp, stored verbatim. | default |
| Stored function | CRLF/CR → LF and leading/trailing blank lines removed, nothing else. Split on `\n` only. Line numbers are 1-indexed on this text. | user |
| Diff key | Per line: comments masked with the line count preserved, then all whitespace removed. Lines with an empty key (blank or comment-only) take no part in the diff. If masking fails on either side (an unterminated literal or comment), both sides are compared unmasked and the row records `comment_mask_ok = false`. | user + default |
| Patch line set | Replaced and deleted vulnerable-side lines are gold. An insertion is attributed to the preceding code line, or to the first code line if it comes before all of them. Gold lines are therefore always code lines. | doc + default |
| 20% rule | Drop if `5 × |S| > n_code_lines`, where `n_code_lines` counts lines with a non-empty key. Exactly 20% is kept. | doc + default |
| Rewrite guard | Also drop if `5 × n_inserted > n_code_lines`, where `n_inserted` counts every patched-side line not matched (git's `+` lines, including the new side of a replacement). Of the 4,668 intact pairs, **376** pass the 20% rule but fail the guard. Stricter or looser readings would give 205 (pure insertions only) or 279 (net growth, `+` minus `−`). In census order the guard drops 242 pairs / 190 CVEs. | user; reading of "inserted" = default |
| Token cap | 10,000 tokens, applied to `max(tokens(vulnerable), tokens(patched))` on the stored text (not the numbered form). The census also reports survivors at 1,024 / 2,048 / 4,096. | user |
| Siblings | Other children of any view-1000 parent (two ChildOf edges away), minus the CWE's own ancestors and descendants. For example, CWE-672 is excluded from CWE-415's siblings because it is also 415's grandparent via CWE-825. Sorted numerically. An empty list is a valid value, not a missing field; CWE-119 has none. | user + default |
| Multi-function CVEs | One function per CVE. The per-function filters run first, and a CVE survives if any candidate survives. Then pick: the single candidate whose name (whole-word, case-sensitive) appears in the NVD description; else the lowest `stable_rank(cve_id, vuln_norm_hash)` among the named candidates, or among all of them if none is named; `pair_id` breaks exact ties. | user |
| Pair integrity | Drop pairs whose patched row has a different commit or CVE (19 in v0.1). Drop every pair whose patched function is shared with another pair (17 pairs over 7 functions). Also drop empty functions, identical cleaned text and malformed CVE IDs. | default |
| Time-range gate | Full min–max span of surviving `published` dates, with the gate at 5 years. The 5th–95th percentile span is reported alongside for information only. | user |
| Splits, constant-answer baseline | Not part of step 1; they follow it. | user |
| Keys | `facts.jsonl` is keyed on `cve_id` alone (unique, asserted). `candidates.jsonl` is keyed on (`cve_id`, `pair_id`), with `pair_id = sha256(commit_id, sha256(raw vuln), sha256(raw patched))`; PrimeVul's own `idx` is not unique. | user + default |
| Determinism | No `hash()`; sha256 everywhere; sorted output; the gzip cache has a zeroed mtime. The only run timestamp is in `manifest.json`. | default |
| Line-numbered form | `render_numbered`: `"{n}: {line}"`, 1-indexed. Step 4 must reuse it. | default |

## Differences from the experiment plan (now reflected in its text)

- The token cap is 10,000, not 1,024 (plan lines 31, 73, 81, 179). The compute budget and vLLM `max_model_len` assumptions need revisiting.
- Normalisation is for comparison only; numbering is on the stored text (plan lines 29, 33).
- The census has more stages than plan line 35 lists: pair integrity, the rewrite guard, and the one-function-per-CVE selection.
- Siblings are defined as 2 ChildOf edges away (plan line 152, "Sibling or child, 1 hop"). The credit schedule for siblings still needs pinning before the scoring module is written.
- PrimeVul does contain multi-function vulnerabilities: 446 CVEs have more than one pair (plan line 666).

## Outcomes

From the build of 2026-09-25 (`data/combined_dataset/census.md`, `parity.json`, `manifest.json`).

| Decision | Outcome |
|---|---|
| Fact table size / size band | **2,276 CVEs.** Below 2,500 and above 1,500, so the plan's split fallback applies: **70/15/15** (≈1,593 / 341 / 342), with no scope reduction. |
| Where the rows went | 4,057 CVEs loaded. Pair integrity −12 (19 cross-commit pairs, 17 pairs sharing a patched function). NVD join −3 (Rejected). CWE −838 (category 354, no NVD CWE 191, multi-CWE 158, placeholder-only 135). CVSS v3.x −455 (v2-only 451, CNA-only 4). Empty patch −1. 20% rule −228. Rewrite guard −190. Token cap −54. |
| Time-range gate | **Proceed.** Full span 14.27 years (2008-07-09 → 2022-10-17). The robust 5th–95th percentile span is also ≥ 5 years: 5.92 years (2016-06-05 → 2022-05-08). Only 37 surviving rows predate 2016. |
| CVSS parity (report only) | Mean agreement with the pooled majority vector: v3.0 0.724, v3.1 0.709; difference −0.015 (95% CI −0.026, −0.004). Restricted to 2015–2019, where both versions occur, the difference is −0.003 (CI −0.017, +0.011). The all-years gap is change over time, not a format difference. No NVD record carried both versions. |
| Token cap context | Survivors entering the cap stage under alternative caps: 1,024 → 1,327 CVEs; 2,048 → 1,831; 4,096 → 2,099; 10,000 → 2,276. |
| One function per CVE | 227 CVEs had more than one surviving candidate. The name rule decided 129 of them (28 with a single named candidate, 101 with the hash breaking a tie between named candidates); the hash alone decided 98. |
| Determinism | Rebuilt under a different `PYTHONHASHSEED`: all outputs byte-identical. `check` found 0 invariant violations. |
| 50-item line sheet | 42 of 50 agree exactly with an independent `git diff` restricted to code lines; the other 8 show no uniform offset. Six involve comment edits that git counts and we mask; the rest are alternative alignments of repeated lines (e.g. an added `}`). **Pending owner review** of `verification/line_sheet.md`, with a verdict per item recorded in the CSV. |
