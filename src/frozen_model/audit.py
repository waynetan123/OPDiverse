"""GRPO signal audit: how often do the base model's 8 rollouts on one prompt disagree?

GRPO normalises rewards within a group of rollouts, so a group whose rollouts all score the same
contributes no gradient. Per type this reports the fraction of "live" groups (nonzero reward variance)
under the dense GRPO reward and under binary scoring (metric == 1).

Caps (owner, step 5). The audit samples once at AUDIT_MAX_TOKENS. A cap-k rollout is the first k
tokens of that sample: same seed, same tokens, then cut. Per type the GRPO cap is the smallest of the
plan's cap and ROLLOUT_CAP_CANDIDATES at which at most ROLLOUT_CUTOFF_MAX of rollouts are cut off,
and the band (as specified / dynamic sampling / floored) is read at that cap from the dense reward.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction

from etl import pinned, verifiers
from etl.build import write_json
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import Paths
from etl.tokens import TokenCounter
from generators.bank.files import generations_for
from generators.progress import track

from .files import SessionFiles
from .prepare import load_bank, load_results

BANDS = ("as_specified", "dynamic_sampling", "floored")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def caps_for(item_type: str) -> tuple[int, ...]:
    plan = pinned.ROLLOUT_MAX_TOKENS[item_type]
    return (plan, *(c for c in pinned.ROLLOUT_CAP_CANDIDATES if c > plan))


def cut(token_ids: list[int], finish_reason: str, k: int, stop_ids: set[int]) -> tuple[list[int], bool]:
    """(the tokens a cap of k would have returned, stop token excluded; whether that run is cut off).

    The stop token counts toward the cap, as in vLLM. Works whether or not the sample's token_ids
    end with the stop token."""
    if k > pinned.AUDIT_MAX_TOKENS:
        raise ValueError(f"cap {k} exceeds the sampled length {pinned.AUDIT_MAX_TOKENS}")
    ids = list(token_ids)
    stopped = finish_reason == "stop"
    if stopped and ids and ids[-1] in stop_ids:
        body, used = ids[:-1], len(ids)
    else:
        body, used = ids, len(ids) + stopped
    if stopped and used <= k:
        return body, False
    return body[:k], True


def live(values: list) -> bool:
    """A group teaches only if its rollouts do not all score the same."""
    return len(set(values)) > 1


def choose_cap(cut_rates: dict[int, Fraction]) -> int:
    """The smallest cap (dict in ascending order) cut off at most ROLLOUT_CUTOFF_MAX of the time, else the largest."""
    for cap, rate in cut_rates.items():
        if rate <= pinned.ROLLOUT_CUTOFF_MAX:
            return cap
    return max(cut_rates)


def band(live_fraction: Fraction) -> str:
    if live_fraction >= pinned.AUDIT_LIVE:
        return "as_specified"
    return "dynamic_sampling" if live_fraction >= pinned.AUDIT_FLOOR else "floored"


def _frac(num: int, den: int) -> Fraction:
    return Fraction(num, den) if den else Fraction(0)


def type_summary(groups: list[dict], item_type: str) -> dict:
    """groups: per prompt, {cap: [(text, Verdict, cut), ...]} plus 'item_id' and 'index'. Fractions as strings."""
    by_cap = {}
    for cap in caps_for(item_type):
        rollouts = [x for g in groups for x in g[cap]]
        n = len(rollouts)
        by_cap[cap] = {
            "cut_rate": _frac(sum(c for _, _, c in rollouts), n),
            "parse_rate": _frac(sum(v.parse_ok for _, v, _ in rollouts), n),
            "live_dense": _frac(sum(live([v.dense for _, v, _ in g[cap]]) for g in groups), len(groups)),
            "live_binary": _frac(sum(live([v.metric == 1 for _, v, _ in g[cap]]) for g in groups), len(groups)),
            "mean_dense": sum((v.dense for _, v, _ in rollouts), Fraction(0)) / n if n else Fraction(0),
            "mean_binary": _frac(sum(v.metric == 1 for _, v, _ in rollouts), n),
        }
    chosen = choose_cap({cap: s["cut_rate"] for cap, s in by_cap.items()})
    full = pinned.AUDIT_MAX_TOKENS if pinned.AUDIT_MAX_TOKENS in by_cap else max(by_cap)
    out = {
        "n_prompts": len(groups),
        "plan_cap": pinned.ROLLOUT_MAX_TOKENS[item_type],
        "chosen_cap": chosen,
        "band": band(by_cap[chosen]["live_dense"]),
        "by_cap": {str(cap): {k: str(v) for k, v in s.items()} for cap, s in by_cap.items()},
        # A seeding failure would make all 8 samples identical; read at the full sampled length.
        "identical_groups": str(_frac(sum(len({t for t, _, _ in g[full]}) == 1 for g in groups), len(groups))),
    }
    at = [(g["index"], v) for g in groups for _, v, _ in g[chosen] if v.parse_ok]
    if item_type == "find_error":
        out["predicted_vulnerable_rate"] = {
            cls: str(_frac(sum(v.parsed.startswith("VULNERABLE: yes") for i, v in at if i == idx),
                           sum(i == idx for i, _ in at)))
            for idx, cls in ((0, "vulnerable_items"), (1, "patched_items"))}
    if item_type == "line_loc":
        sizes = [0 if v.parsed == "LINES: none" else len(v.parsed[len("LINES: "):].split(", ")) for _, v in at]
        out["predicted_set_size"] = dict(sorted(Counter(min(s, 10) for s in sizes).items()))  # 10 = "10 or more"
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(paths: Paths, tokens: TokenCounter) -> dict:
    files = SessionFiles.of(paths)
    bank, bank_sha = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    graph = load_cwe_graph(paths.cwe_xml)
    requests, results = load_results(files.audit_requests)
    if any(r["bank_sha256"] != bank_sha for r in requests):
        raise SystemExit("audit requests were built from a different bank_nontest.jsonl")
    stop_ids = {i for i in (tokens.token_id(t) for t in pinned.EVAL_STOP_TOKENS) if i is not None}

    groups: dict[str, list[dict]] = {t: [] for t in pinned.BANK_TYPES}
    scores, mismatches = [], 0
    for r in track(requests, "Audit scoring"):
        item, outputs = by_id[r["item_id"]], results[r["request_id"]]["outputs"]
        group = {"item_id": item["item_id"], "index": item["index"]}
        for cap in caps_for(item["type"]):
            scored = []
            for o in outputs:
                ids, was_cut = cut(o["token_ids"], o["finish_reason"], cap, stop_ids)
                text = o["text"] if not was_cut else tokens.decode(ids)
                scored.append((text, verifiers.verify_item(item["type"], text, item["gold"], graph), was_cut))
            group[cap] = scored
        for o in outputs:
            if o["finish_reason"] == "stop" and tokens.decode(cut(o["token_ids"], "stop", pinned.AUDIT_MAX_TOKENS, stop_ids)[0]) != o["text"]:
                mismatches += 1
        groups[item["type"]].append(group)

    types = {t: type_summary(g, t) for t, g in groups.items() if g}
    for t, g in groups.items():
        cap = types[t]["chosen_cap"] if g else None
        for grp in g:
            scores.append({"item_id": grp["item_id"], "type": t, "index": grp["index"], "cap": cap,
                           "dense": [str(v.dense) for _, v, _ in grp[cap]],
                           "binary": [int(v.metric == 1) for _, v, _ in grp[cap]],
                           "parsed": [v.parsed for _, v, _ in grp[cap]],
                           "cut": [c for _, _, c in grp[cap]]})
    report = {
        "types": types,
        "grpo_caps": {t: s["chosen_cap"] for t, s in types.items()},
        "bands": {t: s["band"] for t, s in types.items()},
        "decode_mismatches": mismatches,
        "rule": {"cutoff_max": str(pinned.ROLLOUT_CUTOFF_MAX), "live": str(pinned.AUDIT_LIVE),
                 "floor": str(pinned.AUDIT_FLOOR), "band_scored_on": "dense", "n": pinned.ROLLOUT_N},
        "sources": {"audit_requests_sha256": file_sha256(files.audit_requests),
                    "audit_generations_sha256": file_sha256(generations_for(files.audit_requests)),
                    "bank_nontest_sha256": bank_sha, "tokenizer_sha256": tokens.sha256},
    }
    files.audit_scores.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in scores), encoding="utf-8")
    write_json(files.audit_report_json, report)
    files.audit_report_md.write_text(render_markdown(report), encoding="utf-8")
    return report


LABELS = {"as_specified": "trains as specified", "dynamic_sampling": "dynamic sampling", "floored": "**floored**"}


def render_markdown(r: dict) -> str:
    pct = lambda s: f"{100 * float(Fraction(s)):.1f}%"  # noqa: E731
    rows = []
    for t, s in r["types"].items():
        at = s["by_cap"][str(s["chosen_cap"])]
        rows.append(f"| {t} | {s['plan_cap']} | **{s['chosen_cap']}** | {pct(at['cut_rate'])} | {pct(at['parse_rate'])} "
                    f"| {pct(at['live_dense'])} | {pct(at['live_binary'])} | {float(Fraction(at['mean_dense'])):.3f} "
                    f"| {LABELS[s['band']]} |")
    detail = []
    for t, s in r["types"].items():
        detail.append(f"| {t} | " + " · ".join(
            f"{cap}: cut {pct(x['cut_rate'])}, live {pct(x['live_dense'])}" for cap, x in s["by_cap"].items()) + " |")
    extra = []
    for t, s in r["types"].items():
        extra.append(f"- {t}: groups with 8 identical texts {pct(s['identical_groups'])}"
                     + (f"; predicted vulnerable on vulnerable / patched items "
                        f"{pct(s['predicted_vulnerable_rate']['vulnerable_items'])} / {pct(s['predicted_vulnerable_rate']['patched_items'])}"
                        if "predicted_vulnerable_rate" in s else "")
                     + (f"; predicted set sizes {s['predicted_set_size']}" if "predicted_set_size" in s else ""))
    return "\n".join([
        "# Step 5: GRPO signal audit", "",
        f"{pinned.ROLLOUT_N} rollouts per prompt under the pinned rollout sampling, sampled once at {pinned.AUDIT_MAX_TOKENS} tokens. "
        f"A group is live if its rollouts' rewards are not all equal. Cap rule: the smallest cap cutting off at most "
        f"{pct(r['rule']['cutoff_max'])} of rollouts. Band at that cap, on the dense reward: ≥ {pct(r['rule']['live'])} live "
        f"trains as specified, below {pct(r['rule']['floor'])} is floored, in between enables dynamic sampling.", "",
        "| Type | Plan cap | GRPO cap | Cut off | Parsed | Live (dense) | Live (binary) | Mean reward | Band |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows, "",
        "## Every candidate cap", "",
        "| Type | cap: cut-off rate, live groups (dense) |", "|---|---|",
        *detail, "",
        "## Diagnostics", "",
        *extra,
        f"- Stopped samples whose re-decoded text differs from vLLM's: {r['decode_mismatches']}", "",
    ])
