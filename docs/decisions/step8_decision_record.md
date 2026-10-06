# Step 8 decision record — seed partitions

Procedure step 8 splits the 1,933-CVE non-test pool into train and dev, once per seed. It only selects from the frozen bank: nothing is regenerated and no GPU is used.
- Pins: `src/etl/pinned.py`, section "Seed partitions (step 8)".
- Code: `split.partition` in `src/etl/split.py`.
- Command: `PYTHONPATH=src python -m etl.build partition`, which writes `partition.jsonl`, `partition.json` and `partition.md` in `data/combined_dataset/`.

"User" marks a decision the experiment owner made; "default" marks one proposed in planning and accepted with the plan.

## Pins

| Pin | Value | Source |
|---|---|---|
| Seeds | **0–4.** All five partitions are frozen now. Step 11 decides whether seeds 3–4 are trained. | user |
| Late-window size | **2 × ⌈15% × 2,276⌉ = 684** non-test CVEs. This closes step 2's open item. The plan's "latest 20%" (387) would have given about 193 dev CVEs. 684 gives the 70/15/15 fallback's 15% dev. | user |
| Boundary | Day D is the UTC date of the 684th-latest non-test CVE by (`published`, `cve_id`). Every non-test CVE published on or after D is in the window. | default (as step 2) |
| Clusters | A near-duplicate cluster with members on both sides of D goes **wholly to train**, the earlier side. The window is not refilled. This keeps any dev function's near-duplicate out of train. | default (as step 2) |
| Dev, per seed | The window's clusters are ordered by `stable_rank(cluster_id, "dev-partition", seed)`. Whole clusters are added until dev holds ≥ ⌈\|window\| / 2⌉ CVEs. The rest of non-test is train. | doc + default |
| Checkpoint subsample, per seed | The **150** dev CVEs with the lowest `stable_rank(cve_id, "checkpoint-subsample", seed)`, all six items each (900). Every run within a seed uses the same subsample. | doc + default |
| Freeze | `partition` refuses to overwrite a `partition.jsonl` that differs from the redraw. | default |

## Outcome

From `data/combined_dataset/partition.md`, drawn from `facts.jsonl` `a9aaf284…9e5a08b` and `split.jsonl` `179defb4…070534d8`.

| | |
|---|---|
| Window | Boundary **2019-06-13**. 684 CVEs fall on or after it. **11 are moved to train** (clusters straddling the boundary), which leaves **673**. |
| Per seed | **Dev 337 CVEs (2,022 items), train 1,596 (9,576), checkpoint subsample 150 (900).** Dev spans 2019-06 to 2021-07 and is 93–95% CVSS v3.1. |
| Dev overlap | Pairs of seeds share 162–174 of their 337 dev CVEs (Jaccard 0.32–0.35). A random half is expected to share about half, so the randomisation is live. |
| Unseen labels | 8–18 dev CVEs per seed carry a CWE absent from that seed's train. |
| MCQ letters | They are balanced in every split; see `partition.md`. They match the bank's gold letters on all 1,933 items. |
| Determinism | A rebuild under a different `PYTHONHASHSEED` is byte-identical. `partition.jsonl` sha256 `16e0febf…7252105f`. |

Train's latest CVE is CVE-2022-28463 (2022-05-08). Step 2 moved it out of test as a near-duplicate, and step 8 moved it out of the window for the same reason.

## Differences from the experiment plan (now reflected in its text)

- The late window is 684 CVEs (2 × the dev target), not 20% of the non-test pool.
- Partitions are drawn for five seeds, not three.
