"""Parser review (step 7): where do the pinned parsers miss, or over-read, the base model's non-test replies?

Sources, all from the untrained backbone on non-test items:
- the step-5 GRPO signal audit (8 sampled rollouts per prompt, 512 tokens);
- the engine-agreement check's greedy replies, vLLM pass A and HF.
Probe replies are never used (step 3 forbids tuning on them), and test is never read.

Categories, per source and type:
- parse_failure: the lenient parser found no answer in a reply that was not cut off;
- no_field: it found one, but the reply never wrote the answer field (ANSWER:, VULNERABLE:, LINES:), the
  CVSS vector is incomplete, or the exact-ID answer is not on the last line, so the fallback decided;
- lenient_not_strict: parsed, but not in the exact target format (expected for replies that reason first).

The owner reads parser_review.md and approves any change; a change makes the parsers v2.
"""

from __future__ import annotations

from collections import Counter

from etl import pinned, verifiers
from etl.build import write_json
from etl.cwe_graph import load_cwe_graph
from etl.paths import Paths
from etl.build import read_jsonl
from frozen_model.files import SessionFiles
from frozen_model.prepare import load_bank, load_results
from generators.teacher.files import TeacherFiles

from .compare import load_engines
from .files import EngineFiles

CATEGORIES = ("parse_failure", "no_field", "lenient_not_strict")
EXCERPT = 600  # characters from the end of the reply, where the answer is asked for


def category(item_type: str, reply: str, finish_reason: str, v: verifiers.Verdict) -> str | None:
    if not v.parse_ok:
        return "parse_failure" if finish_reason == "stop" else None
    if item_type == "mcq" and not verifiers._MCQ_ANSWER.search(reply):
        return "no_field"
    if item_type == "find_error" and not verifiers._FE_FIELD.search(reply):
        return "no_field"
    if item_type == "line_loc" and not verifiers._LINES_FIELD.search(reply):
        return "no_field"
    if item_type == "cvss" and "?" in v.parsed:
        return "no_field"
    if item_type == "exact_id" and verifiers.parse_cwe(reply.rstrip().split("\n")[-1]) != v.parsed:
        return "no_field"
    return None if v.strict_ok else "lenient_not_strict"


def replies(paths: Paths) -> list[tuple[str, str, int, str, str]]:
    """(source, item_id, output index, text, finish_reason) for every non-test base-model reply under review."""
    out = []
    audit_requests, audit_rows = load_results(SessionFiles.of(paths).audit_requests)
    for r in audit_requests:
        for k, o in enumerate(audit_rows[r["request_id"]]["outputs"]):
            out.append(("audit", r["item_id"], k, o["text"], o["finish_reason"]))
    requests, engines, _ = load_engines(EngineFiles.of(paths))
    for engine in ("vllm_a", "hf"):
        for r in requests:
            o = engines[engine][r["request_id"]]["outputs"][0]
            out.append((engine, r["item_id"], 0, o["text"], o["finish_reason"]))
    return out


def run(paths: Paths) -> dict:
    files = EngineFiles.of(paths)
    bank, _ = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    graph = load_cwe_graph(paths.cwe_xml)
    counts: Counter = Counter()
    totals: Counter = Counter()
    listed = []
    for source, iid, k, text, finish in replies(paths):
        item = by_id[iid]
        v = verifiers.verify_item(item["type"], text, item["gold"], graph)
        totals[(source, item["type"])] += 1
        cut = finish != "stop"
        counts[(source, item["type"], "cut_off")] += cut
        cat = category(item["type"], text, finish, v)
        if cat is None:
            continue
        counts[(source, item["type"], cat)] += 1
        listed.append({"source": source, "item_id": iid, "output": k, "type": item["type"], "category": cat,
                       "finish_reason": finish, "parsed": v.parsed, "metric": str(v.metric), "target": item["target"],
                       "rank": pinned.stable_rank(source, iid, str(k), pinned.PARSER_REVIEW_SALT), "excerpt": text[-EXCERPT:]})
    listed.sort(key=lambda x: (x["source"], pinned.BANK_TYPES.index(x["type"]), CATEGORIES.index(x["category"]), x["rank"]))
    table = {f"{s}:{t}": {"replies": n, **{c: counts[(s, t, c)] for c in ("cut_off", *CATEGORIES)}}
             for (s, t), n in sorted(totals.items(), key=lambda kv: (kv[0][0], pinned.BANK_TYPES.index(kv[0][1])))}
    taken: Counter = Counter()
    sample = []
    for x in listed:
        if x["category"] == "lenient_not_strict" and taken[(x["source"], x["type"])] < pinned.PARSER_REVIEW_SAMPLE:
            taken[(x["source"], x["type"])] += 1
            sample.append(x)
    report = {"parser_version": verifiers.PARSER_VERSION, "counts": table,
              "listed": [x for x in listed if x["category"] != "lenient_not_strict"],
              "lenient_not_strict_sample": sample}
    write_json(files.parser_review_json, report)
    files.parser_review_md.write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(r: dict) -> str:
    rows = [f"| {k.split(':')[0]} | {k.split(':')[1]} | {c['replies']:,} | {c['cut_off']:,} | {c['parse_failure']:,} | "
            f"{c['no_field']:,} | {c['lenient_not_strict']:,} |" for k, c in r["counts"].items()]
    shown: Counter = Counter()
    blocks = []
    for x in r["listed"] + r["lenient_not_strict_sample"]:
        key = (x["source"], x["type"], x["category"])
        shown[key] += 1
        if shown[key] > pinned.PARSER_REVIEW_SAMPLE:
            continue
        blocks += [f"### {x['category']} · {x['source']} · `{x['item_id']}` (output {x['output']})", "",
                   f"Target `{x['target']}`; parsed `{x['parsed']}`; metric {x['metric']}; finish {x['finish_reason']}.", "",
                   "````", x["excerpt"], "````", ""]
    return "\n".join([
        "# Step 7: parser review", "",
        f"Parsers {r['parser_version']}, on the untrained backbone's non-test replies: the step-5 audit (8 sampled rollouts per "
        "prompt) and this step's greedy replies (vLLM pass A, HF). Probe replies are not used.", "",
        "- **parse_failure**: no answer found in a reply that was not cut off.",
        "- **no_field**: an answer was found, but not from the answer field (ANSWER:, VULNERABLE:, LINES:), the CVSS vector is "
        "incomplete, or the exact-ID answer is not on the last line, so a fallback decided.",
        "- **lenient_not_strict**: parsed, not in the exact target format (expected when the model reasons first).", "",
        "A change is made only where a human reader would accept the stated answer without doubt and the parser missed or "
        "misread it. The owner approves each one.", "",
        "| Source | Type | Replies | Cut off | parse_failure | no_field | lenient_not_strict |", "|---|---|---|---|---|---|---|",
        *rows, "",
        f"## Examples (up to {pinned.PARSER_REVIEW_SAMPLE} per source, type and category; all failures are in parser_review.json)", "",
        *blocks,
    ])


# ---------------------------------------------------------------------------
# Guards: a parser change must not change what a frozen artifact means
# ---------------------------------------------------------------------------


def guards(paths: Paths) -> list[str]:
    """Under the current parsers, every frozen target still scores 1 under its own verifier (bank targets
    strictly), every DPO rejected answer is still a strict-format near miss with the dense score recorded at
    step 6, and every distill-self target still ends in the gold answer. Returns the violations."""
    bank, _ = load_bank(paths)
    by_id = {r["item_id"]: r for r in bank}
    graph = load_cwe_graph(paths.cwe_xml)
    bad = []
    for item in bank:
        v = verifiers.verify_item(item["type"], item["target"], item["gold"], graph)
        if not (v.strict_ok and v.metric == 1 and v.dense == 1):
            bad.append(f"{item['item_id']}: bank target no longer scores 1 strictly")
    for row in read_jsonl(TeacherFiles.of(paths).dpo):
        item = by_id[row["item_id"]]
        v = verifiers.verify_item(item["type"], row["rejected"], item["gold"], graph)
        if not (v.strict_ok and v.dense < 1 and str(v.dense) == row["rejected_dense"]):
            bad.append(f"{row['item_id']}: DPO rejected {row['rejected']!r} scores {v.dense} (step 6: {row['rejected_dense']})")
        if row["chosen"] != item["target"]:
            bad.append(f"{row['item_id']}: DPO chosen is not the bank target")
    for row in read_jsonl(SessionFiles.of(paths).distill_self):
        item = by_id[row["item_id"]]
        v = verifiers.verify_item(item["type"], row["target"], item["gold"], graph)
        last = row["target"].rstrip().split("\n")[-1]
        if not (v.metric == 1 and last == item["target"]):
            bad.append(f"{row['item_id']}: distill-self target no longer ends in the gold answer (metric {v.metric})")
    return bad
