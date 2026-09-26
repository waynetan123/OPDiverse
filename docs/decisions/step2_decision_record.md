# Step 2 decision record — frozen test window and exact-ID credit schedule

Pins for drawing the test window (procedure step 2), plus the constant-answer baseline and exact-ID credit schedule, which were moved here from step 1. All are implemented in `src/etl/pinned.py` and `src/etl/split.py`; `PYTHONPATH=src python -m etl.build test-window` produces the outputs. "User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## Split proportions

The fact table has 2,276 CVEs, below 2,500, so the plan's fallback **70 / 15 / 15** applies and test stays at 15%.

| Pool | Share | CVEs | Drawn |
|---|---|---|---|
| Test | 15% (15.07%) | 343 | step 2, once, frozen |
| Dev | 15% | ≈ 341 | per seed at step 8, from the 1,933-CVE non-test pool |
| Train | 70% | ≈ 1,592 | per seed at step 8 |

PrimeVul's own train/valid/test split is not used. `primevul_split` is kept in `facts.jsonl` for provenance only. Our test window takes 174 of PrimeVul's test rows, 135 of its train rows and 34 of its valid rows.

## Pins

| Pin | Value | Source |
|---|---|---|
| Test size | k = ⌈15% × N⌉, computed exactly (`Fraction(3, 20)`) | doc |
| Boundary | Day D = UTC date of the k-th latest `published`. Test is every CVE published on or after D, so same-day CVEs all go to test. | default |
| Near-duplicate link | Two CVEs are linked if any function of one (vulnerable or patched) has the same `norm_body_hash` as any function of the other, **or** their 3-line shingle sets have Jaccard ≥ 4/5. Shingles are built from comment-masked, whitespace-free code-line keys. The comparison is in integer arithmetic. | user |
| Clusters | Connected components over the links, computed over the whole fact table. `cluster_id` is the earliest member by (`published`, `cve_id`). Step 8 must keep clusters intact within the non-test pool. | doc |
| Moves | A cluster with members on both sides moves entirely to non-test (the earlier side). Test is not refilled. If moves would push test below k, `test-window` stops (never shave test). | doc + default |
| Frozen | `test-window` records the sha256 of the `facts.jsonl` it drew from. It refuses to run if that file has changed, or if a redraw would change `split.jsonl`. | default |
| Symmetric credit | Shortest undirected ChildOf distance in view 1000: 0 / 1 / 2 → 1 / ½ / ¼, else 0. | doc |
| Direction-aware credit | Exact 1 · child ½ · parent ¼ · sibling ¼ · grandchild ¼ · co-parent (shares a child) ¼ · **any ancestor of gold at distance 2** ⅛ · else 0. | user (sibling ¼) + default |
| Ancestor discount | At distance 2, a prediction that is an ancestor of gold by any path scores ⅛, even if it is also a sibling. Example: CWE-672 for gold CWE-415 is a sibling via CWE-666 and a grandparent via CWE-825, so it scores ⅛. This matches the step 1 sibling definition, which excludes ancestors. | default |
| Invalid prediction | Anything that isn't a live view-1000 weakness (category, deprecated, malformed) scores 0. Gold must be live, or the scorer raises an error. | default |
| Baseline | Candidates are all 944 live view-1000 weaknesses, scored over the non-test facts with exact fractions; ties go to the lower CWE number. Direction-aware is adopted if the symmetric best is > ⅕. | doc |
| Schedule pin | `pinned.EXACT_ID_SCHEDULE = "direction_aware"`. `test-window` fails if the computed decision ever disagrees with it. | doc |

## Outcomes

From `data/combined_dataset/test_window.md` and `baselines.json`, drawn from `facts.jsonl` sha256 `a9aaf284…9e5a08b`.

| Decision | Outcome |
|---|---|
| Boundary | **2021-07-30.** k = 342; 344 CVEs fall on or after the boundary day. |
| Near-duplicates | 76 linked CVE pairs (8 exact, 68 fuzzy). Clusters: 42 of size 2, 6 of size 3, 1 of size 5, 1 of size 6; all inspected and all real function families (ImageMagick readers, Chrome `IDNSpoofChecker`, WavPack). One cluster crossed the boundary. |
| Moved | **1: CVE-2022-28463** (ImageMagick `ReadCINImage`). Its vulnerable function shares 84% of its shingles with CVE-2019-11470's patched version. |
| Final pools | **Test 343 (15.07%)**, non-test 1,933. No cluster spans both pools. Rebuilt under a different `PYTHONHASHSEED`: byte-identical. |
| CVSS versions | Non-test 934 v3.0 / 999 v3.1; **test 0 / 343**, because NVD moved to v3.1 in mid-2019, before the window. The step 1 parity check found no version effect on overlap years. |
| Unseen labels | 6 test CVEs carry a CWE never seen in the non-test pool (CWE-116, 273, 552, 1284). No objective can learn those labels from training data. |
| Backbone | Qwen2.5 was released 2024-09-19, 703 days after the latest test CVE (2022-10-17). The whole window is likely inside pretraining; step 3's probe measures recall. |
| Constant-answer baseline, symmetric | **CWE-119 = 0.2622 > 0.2 → direction-aware credit adopted.** |
| Constant-answer baseline, direction-aware | **CWE-125 = 0.2355.** CWE-119 drops to 0.1848: the hub no longer wins, and the best constant is now the most frequent label. |
| Other non-test baselines (for step 3) | Most frequent CWE: CWE-125, exact accuracy 0.1547. Majority CVSS vector `AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`, mean per-component agreement 0.7190. |

## Open for step 8

- **The late-window rule gives too little dev.** The plan's "latest 20% of the non-test pool, half to dev" yields ≈ 193 dev CVEs (8.5% of the table), not ≈ 341 (15%). This needs a pin before step 8, e.g. a late window twice the dev size.
