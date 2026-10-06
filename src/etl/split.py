"""Step 2: the frozen test window, near-duplicate clusters, and the non-test baselines."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date
from fractions import Fraction

from . import parity, pinned
from .cwe_graph import CweGraph

POOLS = ("nontest", "test")


# ---------------------------------------------------------------------------
# Near-duplicates
# ---------------------------------------------------------------------------


def near_dup_links(facts: list[dict]) -> list[dict]:
    """Every pair of CVEs whose functions (vulnerable or patched, either way round) share a
    normalised-body hash or have shingle Jaccard >= NEAR_DUP_JACCARD. One entry per CVE pair,
    with the strongest evidence."""
    funcs = []  # (n_shingles, cve_id, side, shingles, norm_hash)
    for row in facts:
        for side in ("vuln", "patched"):
            sh = pinned.code_shingles(row[f"{side}_func"])
            funcs.append((len(sh), row["cve_id"], side, sh, row[f"{side}_norm_hash"]))
    funcs.sort(key=lambda f: (f[0], f[1], f[2]))
    t = pinned.NEAR_DUP_JACCARD
    best: dict[tuple[str, str], dict] = {}
    for i, (n_i, cve_i, side_i, sh_i, h_i) in enumerate(funcs):
        for j in range(i + 1, len(funcs)):
            n_j, cve_j, side_j, sh_j, h_j = funcs[j]
            if n_j * t.numerator > n_i * t.denominator:
                break  # |B| > |A| / t: Jaccard cannot reach t, nor for any larger B
            if cve_i == cve_j:
                continue
            exact = h_i == h_j
            if not exact and not pinned.jaccard_at_least(sh_i, sh_j):
                continue
            inter = len(sh_i & sh_j)
            jac = Fraction(inter, n_i + n_j - inter)
            (a, sa), (b, sb) = sorted(((cve_i, side_i), (cve_j, side_j)))
            link = {"a": a, "b": b, "kind": "exact" if exact else "fuzzy", "sides": f"{sa}-{sb}", "jaccard": jac}
            prior = best.get((a, b))
            if prior is None or (link["kind"] == "exact", jac, link["sides"]) > (prior["kind"] == "exact", prior["jaccard"], prior["sides"]):
                best[(a, b)] = link
    out = []
    for key in sorted(best):
        link = best[key]
        out.append({**link, "jaccard": round(float(link["jaccard"]), 4)})
    return out


def clusters(facts: list[dict], links: list[dict]) -> dict[str, str]:
    """cve_id -> cluster_id, where cluster_id is the cluster's earliest member by (published, cve_id)."""
    parent = {row["cve_id"]: row["cve_id"] for row in facts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for link in links:
        ra, rb = find(link["a"]), find(link["b"])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    order = {row["cve_id"]: (row["published"], row["cve_id"]) for row in facts}
    members: dict[str, list[str]] = {}
    for cve in parent:
        members.setdefault(find(cve), []).append(cve)
    return {cve: min(group, key=order.__getitem__) for group in members.values() for cve in group}


# ---------------------------------------------------------------------------
# Test window
# ---------------------------------------------------------------------------


def draw_test_window(facts: list[dict]) -> dict:
    """The chronologically latest TEST_FRACTION by publication day, with near-duplicate clusters
    kept intact by moving any straddling cluster to the earlier (non-test) side."""
    keys = [(r["published"], r["cve_id"]) for r in facts]
    assert keys == sorted(keys), "facts must be sorted by (published, cve_id)"
    assert len({r["cve_id"] for r in facts}) == len(facts), "cve_id must be unique"
    n = len(facts)
    k = math.ceil(n * pinned.TEST_FRACTION)  # exact: Fraction.__ceil__
    boundary = facts[n - k]["published"][:10]
    initial = {r["cve_id"] for r in facts if r["published"][:10] >= boundary}

    links = near_dup_links(facts)
    cluster_of = clusters(facts, links)
    members: dict[str, list[str]] = {}
    for row in facts:
        members.setdefault(cluster_of[row["cve_id"]], []).append(row["cve_id"])
    moved = sorted(
        c for group in members.values()
        if 0 < sum(c in initial for c in group) < len(group)
        for c in group if c in initial
    )
    test = initial - set(moved)
    if len(test) < k:
        raise SystemExit(f"test window fell to {len(test)} < {k} after moving near-duplicates; "
                         "the plan says never shave test, so decide how to refill before drawing")

    assignments = []
    for row in facts:
        cve = row["cve_id"]
        cid = cluster_of[cve]
        assignments.append({
            "cve_id": cve,
            "pool": "test" if cve in test else "nontest",
            "cluster_id": cid,
            "cluster_size": len(members[cid]),
            "moved": cve in moved,
        })
    _check(facts, assignments, boundary)
    pool_of = {a["cve_id"]: a["pool"] for a in assignments}
    linked = {}
    for link in links:
        linked.setdefault(link["a"], []).append(link)
        linked.setdefault(link["b"], []).append(link)
    return {
        "n": n,
        "k": k,
        "boundary_day": boundary,
        "initial": {"test": len(initial), "nontest": n - len(initial)},
        "final": {"test": len(test), "nontest": n - len(test)},
        "moved": [{"cve_id": c, "cluster_id": cluster_of[c], "links": linked[c]} for c in moved],
        "links": links,
        "multi_cve_clusters": [
            {"cluster_id": cid, "members": group, "pool": pool_of[cid]}
            for cid, group in members.items()  # insertion order follows facts, i.e. (published, cve_id)
            if len(group) > 1
        ],
        "cluster_sizes": sorted(Counter(len(g) for g in members.values()).items()),
        "assignments": assignments,
    }


def _check(facts: list[dict], assignments: list[dict], boundary: str) -> None:
    pool_of = {a["cve_id"]: a["pool"] for a in assignments}
    assert set(pool_of) == {r["cve_id"] for r in facts}, "every CVE gets exactly one pool"
    by_cluster: dict[str, set[str]] = {}
    for a in assignments:
        by_cluster.setdefault(a["cluster_id"], set()).add(a["pool"])
    assert all(len(p) == 1 for p in by_cluster.values()), "a near-duplicate cluster spans both pools"
    for r, a in zip(facts, assignments):
        day = r["published"][:10]
        if a["pool"] == "test":
            assert day >= boundary, f"{r['cve_id']} in test before the boundary"
        elif not a["moved"]:
            assert day < boundary, f"{r['cve_id']} in non-test after the boundary without being moved"


# ---------------------------------------------------------------------------
# Seed partitions (step 8)
# ---------------------------------------------------------------------------


def partition(facts: list[dict], assignments: list[dict]) -> dict:
    """Per seed, a random half of the non-test late window (whole clusters) is dev; the rest of non-test
    is train. A cluster straddling the window's boundary day goes wholly to the earlier side, as in step 2."""
    pool_of = {a["cve_id"]: a["pool"] for a in assignments}
    cluster_of = {a["cve_id"]: a["cluster_id"] for a in assignments}
    nontest = sorted((r["published"], r["cve_id"]) for r in facts if pool_of[r["cve_id"]] == "nontest")
    w = pinned.LATE_WINDOW_MULTIPLE * math.ceil(len(facts) * pinned.DEV_FRACTION)
    boundary = nontest[len(nontest) - w][0][:10]
    initial = {cve for pub, cve in nontest if pub[:10] >= boundary}
    members: dict[str, list[str]] = {}
    for _, cve in nontest:
        members.setdefault(cluster_of[cve], []).append(cve)
    moved = sorted(c for group in members.values() if 0 < sum(c in initial for c in group) < len(group)
                   for c in group if c in initial)
    window = initial - set(moved)
    target = math.ceil(len(window) / 2)

    dev, checkpoint = {}, {}
    for seed in pinned.PARTITION_SEEDS:
        chosen: set[str] = set()
        for cid in sorted({cluster_of[c] for c in window}, key=lambda c: (pinned.stable_rank(c, pinned.DEV_SALT, str(seed)), c)):
            if len(chosen) >= target:
                break
            chosen.update(members[cid])
        if len(chosen) < pinned.CHECKPOINT_CVES:
            raise SystemExit(f"seed {seed}: dev has {len(chosen)} CVEs, fewer than the {pinned.CHECKPOINT_CVES}-CVE checkpoint subsample")
        dev[seed] = chosen
        ranked = sorted(chosen, key=lambda c: (pinned.stable_rank(c, pinned.CHECKPOINT_SALT, str(seed)), c))
        checkpoint[seed] = set(ranked[: pinned.CHECKPOINT_CVES])

    for seed in pinned.PARTITION_SEEDS:
        assert all(len({c in dev[seed] for c in group}) == 1 for group in members.values()), "a cluster spans train and dev"
        assert dev[seed] <= window and checkpoint[seed] <= dev[seed]
    assert len({frozenset(d) for d in dev.values()}) == len(dev), "two seeds drew the same dev set"

    rows = [{
        "cve_id": cve,
        "cluster_id": cluster_of[cve],
        "late_window": cve in window,
        "moved_out_of_window": cve in moved,
        "role": {str(s): "dev" if cve in dev[s] else "train" for s in pinned.PARTITION_SEEDS},
        "checkpoint_seeds": [s for s in pinned.PARTITION_SEEDS if cve in checkpoint[s]],
    } for _, cve in nontest]
    return {
        "n_facts": len(facts),
        "n_nontest": len(nontest),
        "window_target": w,
        "boundary_day": boundary,
        "on_or_after_boundary": len(initial),
        "moved": moved,
        "window": len(window),
        "dev_target": target,
        "rows": rows,
    }


def partition_stats(facts: list[dict], rows: list[dict]) -> dict:
    """Per seed and role (train, dev, checkpoint): size, dates, CVSS versions, CWEs and MCQ gold letters;
    and the pairwise dev overlap between seeds."""
    by_cve = {r["cve_id"]: r for r in facts}
    seeds = [str(s) for s in pinned.PARTITION_SEEDS]
    letter = lambda cve: pinned.MCQ_LETTERS[pinned.stable_rank(cve, pinned.MCQ_SALTS["letter"]) % 4]  # noqa: E731
    out: dict = {"seeds": {}, "dev_overlap": {}}
    for s in seeds:
        groups = {
            "train": [r["cve_id"] for r in rows if r["role"][s] == "train"],
            "dev": [r["cve_id"] for r in rows if r["role"][s] == "dev"],
            "checkpoint": [r["cve_id"] for r in rows if int(s) in r["checkpoint_seeds"]],
        }
        train_cwes = {by_cve[c]["cwe"] for c in groups["train"]}
        out["seeds"][s] = {}
        for role, cves in groups.items():
            facts_ = [by_cve[c] for c in cves]
            cwes = Counter(f["cwe"] for f in facts_)
            out["seeds"][s][role] = {
                "cves": len(cves),
                "items": len(cves) * pinned.ITEMS_PER_CVE,
                "published_min": min(f["published"] for f in facts_),
                "published_max": max(f["published"] for f in facts_),
                "cvss_versions": dict(sorted(Counter(f["cvss_version"] for f in facts_).items())),
                "distinct_cwes": len(cwes),
                "top_cwes": [[c, n] for c, n in sorted(cwes.items(), key=lambda kv: (-kv[1], int(kv[0][4:])))[:3]],
                "mcq_letters": {x: sum(letter(c) == x for c in cves) for x in pinned.MCQ_LETTERS},
                "cves_with_cwe_unseen_in_train": sum(f["cwe"] not in train_cwes for f in facts_),
            }
    devs = {s: {r["cve_id"] for r in rows if r["role"][s] == "dev"} for s in seeds}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            out["dev_overlap"][f"{a}-{b}"] = {"shared": len(devs[a] & devs[b]),
                                             "jaccard": round(len(devs[a] & devs[b]) / len(devs[a] | devs[b]), 4)}
    return out


def render_partition_markdown(part: dict, stats: dict) -> str:
    out = [
        "# Step 8: seed partitions", "",
        f"Drawn from `facts.jsonl` sha256 `{part['facts_sha256']}` and `split.jsonl` sha256 `{part['split_sha256']}`.", "",
        f"- **Late window:** {pinned.LATE_WINDOW_MULTIPLE} × ⌈{pinned.DEV_FRACTION} × {part['n_facts']:,}⌉ = {part['window_target']:,} "
        f"of the {part['n_nontest']:,} non-test CVEs. Boundary day **{part['boundary_day']}**; "
        f"{part['on_or_after_boundary']:,} CVEs fall on or after it.",
        f"- **Moved out of the window** (near-duplicate clusters straddling the boundary, sent wholly to train): "
        f"{len(part['moved'])}" + (f" ({', '.join(part['moved'])})" if part["moved"] else "") + ".",
        f"- **Window: {part['window']:,} CVEs.** Per seed, whole clusters are drawn until dev holds ≥ {part['dev_target']:,}; "
        f"the rest of non-test is train. The checkpoint subsample is {pinned.CHECKPOINT_CVES} dev CVEs per seed.",
        "",
        "## Per seed", "",
    ]
    rows = []
    for s, roles in stats["seeds"].items():
        for role, x in roles.items():
            rows.append((
                s, role, f"{x['cves']:,}", f"{x['items']:,}",
                f"{x['published_min'][:10]} → {x['published_max'][:10]}",
                f"{x['cvss_versions'].get('3.0', 0):,} / {x['cvss_versions'].get('3.1', 0):,}",
                x["distinct_cwes"], ", ".join(f"{c} ({n})" for c, n in x["top_cwes"]),
                " / ".join(str(x["mcq_letters"][k]) for k in pinned.MCQ_LETTERS),
                "" if role == "train" else x["cves_with_cwe_unseen_in_train"],
            ))
    out += [_table(("Seed", "Role", "CVEs", "Items", "Published", "CVSS v3.0 / v3.1", "CWEs", "Top CWEs",
                    "MCQ A / B / C / D", "CVEs with a CWE unseen in train"), rows), ""]
    out += ["## Dev overlap between seeds", "",
            "A random half of the window is expected to share about half its CVEs with another seed's.", "",
            _table(("Seeds", "Shared dev CVEs", "Jaccard"),
                   ((k, v["shared"], f"{v['jaccard']:.3f}") for k, v in stats["dev_overlap"].items())), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Per-pool description
# ---------------------------------------------------------------------------


def pool_stats(facts: list[dict], assignments: list[dict]) -> dict:
    pool_of = {a["cve_id"]: a["pool"] for a in assignments}
    out = {}
    nontest_cwes = {r["cwe"] for r in facts if pool_of[r["cve_id"]] == "nontest"}
    for pool in POOLS:
        rows = [r for r in facts if pool_of[r["cve_id"]] == pool]
        cwes = Counter(r["cwe"] for r in rows)
        out[pool] = {
            "n": len(rows),
            "published_min": min(r["published"] for r in rows),
            "published_max": max(r["published"] for r in rows),
            "years": dict(sorted(Counter(r["published"][:4] for r in rows).items())),
            "cvss_versions": dict(sorted(Counter(r["cvss_version"] for r in rows).items())),
            "distinct_cwes": len(cwes),
            "top_cwes": [[c, n] for c, n in sorted(cwes.items(), key=lambda kv: (-kv[1], int(kv[0][4:])))[:15]],
            "primevul_split": dict(sorted(Counter(r["primevul_split"] for r in rows).items())),
        }
    test_rows = [r for r in facts if pool_of[r["cve_id"]] == "test"]
    unseen = [r for r in test_rows if r["cwe"] not in nontest_cwes]
    out["test"]["cwe_unseen_in_nontest"] = {
        "cves": len(unseen),
        "cwes": sorted({r["cwe"] for r in unseen}, key=lambda c: int(c[4:])),
    }
    return out


# ---------------------------------------------------------------------------
# Baselines on the non-test pool
# ---------------------------------------------------------------------------


def _best_constants(gold: Counter, n: int, graph: CweGraph, schedule: str, top: int = 10) -> list[dict]:
    candidates = sorted((c for c in graph.weaknesses if graph.in_view(c)), key=int)
    scored = []
    for c in candidates:
        total = sum(cnt * pinned.hierarchy_score(f"CWE-{c}", g, graph, schedule) for g, cnt in gold.items())
        scored.append((Fraction(total, n), c))
    scored.sort(key=lambda sc: (-sc[0], int(sc[1])))
    return [{"cwe": f"CWE-{c}", "mean": float(v), "mean_exact": str(v)} for v, c in scored[:top]]


def baselines(nontest: list[dict], graph: CweGraph) -> dict:
    n = len(nontest)
    gold = Counter(r["cwe"] for r in nontest)
    symmetric = _best_constants(gold, n, graph, "symmetric")
    best_symmetric = Fraction(symmetric[0]["mean_exact"])
    decision = "direction_aware" if best_symmetric > pinned.EXACT_ID_THRESHOLD else "symmetric"
    adopted = _best_constants(gold, n, graph, decision)
    most_frequent, count = min(gold.items(), key=lambda kv: (-kv[1], int(kv[0][4:])))
    vectors = [pinned.parse_cvss_v3(r["cvss_vector"])[1] for r in nontest]
    majority = parity.majority(vectors)
    agreement = sum(sum(v[key] == majority[key] for key in pinned.CVSS_ORDER) for v in vectors) / (n * len(pinned.CVSS_ORDER))
    return {
        "pool": "nontest",
        "n": n,
        "exact_id_hierarchy": {
            "threshold": str(pinned.EXACT_ID_THRESHOLD),
            "symmetric_top": symmetric,
            "decision": decision,
            "adopted_schedule_top": adopted,
            "direction_aware_credit": {k: str(v) for k, v in pinned.DIRECTION_AWARE_CREDIT.items()},
            "symmetric_credit": {str(k): str(v) for k, v in pinned.SYMMETRIC_CREDIT.items()},
        },
        "most_frequent_cwe": {"cwe": most_frequent, "count": count, "exact_accuracy": count / n},
        "cvss_majority": {
            "vector": pinned.canonical_cvss(majority),
            "mean_component_agreement": agreement,
            "per_component": {key: sum(v[key] == majority[key] for v in vectors) / n for key in pinned.CVSS_ORDER},
        },
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(tw: dict, base: dict) -> str:
    fin, ini = tw["final"], tw["initial"]
    stats = tw["pools"]
    released = date.fromisoformat(pinned.BACKBONE_RELEASED)
    latest = date.fromisoformat(stats["test"]["published_max"][:10])
    out = [
        "# Step 2: frozen test window", "",
        f"Drawn from `facts.jsonl` sha256 `{tw['facts_sha256']}`.", "",
        "## Outcome", "",
        f"- **Boundary day: {tw['boundary_day']}.** Test is every CVE published on or after it "
        f"(k = ⌈{pinned.TEST_FRACTION} × {tw['n']:,}⌉ = {tw['k']:,}; {ini['test']:,} fall on or after the boundary day).",
        f"- **Near-duplicate moves: {len(tw['moved'])}** to the non-test side.",
        f"- **Test: {fin['test']:,} CVEs ({fin['test'] / tw['n']:.2%}); non-test: {fin['nontest']:,}.**",
        f"- **Backbone:** {pinned.TOKENIZER_REPO} was released {pinned.BACKBONE_RELEASED}; the latest test CVE "
        f"({latest}) predates that by {(released - latest).days:,} days, so the whole window is likely inside pretraining. "
        "Step 3's probe measures whether it is recalled.",
        "",
    ]
    if tw["moved"]:
        out += ["## Moved", ""]
        out += [_table(("CVE", "Cluster", "Linked to (kind, sides, Jaccard)"),
                       ((m["cve_id"], m["cluster_id"],
                         "; ".join(f"{l['b'] if l['a'] == m['cve_id'] else l['a']} ({l['kind']}, {l['sides']}, {l['jaccard']})"
                                   for l in m["links"])) for m in tw["moved"])), ""]
    out += ["## Near-duplicate clusters", "",
            f"{len(tw['links']):,} linked CVE pairs "
            f"({sum(l['kind'] == 'exact' for l in tw['links']):,} exact hash, {sum(l['kind'] == 'fuzzy' for l in tw['links']):,} fuzzy "
            f"≥ {pinned.NEAR_DUP_JACCARD}). Cluster sizes: "
            + ", ".join(f"{s} × {c:,}" for s, c in tw["cluster_sizes"]) + ". "
            "Every cluster sits wholly in one pool; step 8 keeps them intact within the non-test pool.", ""]

    out += ["## Pools", ""]
    rows = [
        ("CVEs", f"{stats['nontest']['n']:,}", f"{stats['test']['n']:,}"),
        ("Published", f"{stats['nontest']['published_min'][:10]} → {stats['nontest']['published_max'][:10]}",
         f"{stats['test']['published_min'][:10]} → {stats['test']['published_max'][:10]}"),
        ("CVSS v3.0 / v3.1", *(f"{s['cvss_versions'].get('3.0', 0):,} / {s['cvss_versions'].get('3.1', 0):,}"
                               for s in (stats['nontest'], stats['test']))),
        ("Distinct CWEs", stats["nontest"]["distinct_cwes"], stats["test"]["distinct_cwes"]),
        ("PrimeVul's own split label", *(", ".join(f"{k} {v:,}" for k, v in s["primevul_split"].items())
                                         for s in (stats['nontest'], stats['test']))),
    ]
    out += [_table(("", "Non-test", "Test"), rows), ""]
    unseen = stats["test"]["cwe_unseen_in_nontest"]
    out += [f"{unseen['cves']:,} test CVEs carry a CWE that never occurs in the non-test pool: "
            + (", ".join(unseen["cwes"]) or "none") + ".", ""]
    years = sorted(set(stats["nontest"]["years"]) | set(stats["test"]["years"]))
    out += ["**By publication year**", "",
            _table(("Year", "Non-test", "Test"), ((y, stats["nontest"]["years"].get(y, 0), stats["test"]["years"].get(y, 0)) for y in years)), ""]
    top = max(len(stats["nontest"]["top_cwes"]), len(stats["test"]["top_cwes"]))
    pad = lambda lst, i: f"{lst[i][0]} ({lst[i][1]})" if i < len(lst) else ""  # noqa: E731
    out += ["**Top CWEs**", "",
            _table(("#", "Non-test", "Test"), ((i + 1, pad(stats["nontest"]["top_cwes"], i), pad(stats["test"]["top_cwes"], i)) for i in range(top))), ""]

    h = base["exact_id_hierarchy"]
    sym, ado = h["symmetric_top"][0], h["adopted_schedule_top"][0]
    out += ["## Exact-ID credit schedule", "",
            f"Best constant answer over the {base['n']:,} non-test facts under the symmetric schedule: "
            f"**{sym['cwe']} = {sym['mean']:.4f}** (threshold {float(Fraction(h['threshold'])):.1f}). "
            f"**Adopted: {h['decision'].replace('_', '-')}.** Re-reported under it: **{ado['cwe']} = {ado['mean']:.4f}**.", "",
            _table(("Rank", "Symmetric", "Adopted"),
                   ((i + 1, f"{s['cwe']} {s['mean']:.4f}", f"{a['cwe']} {a['mean']:.4f}")
                    for i, (s, a) in enumerate(zip(h["symmetric_top"], h["adopted_schedule_top"])))), ""]
    mf, cv = base["most_frequent_cwe"], base["cvss_majority"]
    out += ["## Other non-test baselines (for step 3's probe)", "",
            f"- Most frequent CWE: {mf['cwe']} ({mf['count']:,} of {base['n']:,}, exact accuracy {mf['exact_accuracy']:.4f}).",
            f"- Majority CVSS vector: `{cv['vector']}`, mean per-component agreement {cv['mean_component_agreement']:.4f}.", ""]
    return "\n".join(out)
