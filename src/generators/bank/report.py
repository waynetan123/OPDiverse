"""bank_report.{json,md}: MCQ construction, prompt lengths, trivial baselines, and the git cross-check.

Baselines are computed on the non-test pool only. Test is never scored here.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from fractions import Fraction

from etl import pinned, verifiers
from etl.build import read_jsonl, write_json
from etl.census import quantiles
from etl.cwe_graph import CweGraph
from etl.paths import Paths
from etl.split import POOLS
from etl.verify_sheet import git_gold

from . import mcq
from .build import load_inputs
from .files import BankFiles


def _mean(values: list[Fraction]) -> float:
    return float(sum(values, Fraction(0)) / len(values)) if values else float("nan")


def distance(a: str, b: str, graph: CweGraph, cap: int = 6) -> int | None:
    """Shortest undirected ChildOf distance between two 'CWE-n' IDs, or None beyond cap."""
    a, b = a[4:], b[4:]
    seen, frontier = {a: 0}, deque([a])
    while frontier:
        x = frontier.popleft()
        if x == b:
            return seen[x]
        if seen[x] == cap:
            continue
        for y in graph.parents(x) | graph.children(x):
            if y not in seen:
                seen[y] = seen[x] + 1
                frontier.append(y)
    return None


def mcq_section(bank: dict[str, list[dict]], decisions: list[dict], gens: list[dict], meta: dict, graph: CweGraph) -> dict:
    letters = {pool: dict(sorted(Counter(r["gold"]["letter"] for r in rows if r["type"] == "mcq").items()))
               for pool, rows in bank.items()}
    dist = Counter()
    for d in decisions:
        for c in d["distractors"]:
            n = distance(c, d["gold"], graph)
            dist["far" if n is None else str(n)] += 1
    statuses, reasons = Counter(), Counter()
    for d in decisions:
        for a in d.get("discarded_model_decision", d)["attempts"]:
            statuses[f"attempt {a['attempt']}: {a['status']}"] += 1
            reasons.update(r for _, r in a["rejected"])
    return {
        "guard": meta["mcq_guard"],
        "gold_letter_marginals": letters,
        "distractor_sources": {pool: dict(sorted(Counter(s for d in decisions if d["pool"] == pool for s in d["sources"]).items()))
                               for pool in POOLS},
        "items_with_any_draw": {pool: sum("draw" in d["sources"] for d in decisions if d["pool"] == pool) for pool in POOLS},
        "attempt_statuses": dict(sorted(statuses.items())),
        "rejection_reasons": dict(sorted(reasons.items())),
        "refusal_categories": dict(sorted(Counter(str(g["stop_category"]) for g in gens if g.get("stop_reason") == "refusal").items())),
        "distractor_distance": dict(sorted(dist.items())),
    }


def tokens_section(bank: dict[str, list[dict]]) -> dict:
    out = {}
    for t in pinned.BANK_TYPES:
        values = [r["prompt_tokens"] for rows in bank.values() for r in rows if r["type"] == t]
        out[t] = quantiles(values)
    longest = max(r["prompt_tokens"] for rows in bank.values() for r in rows)
    return {"quantiles": out, "max_prompt_tokens": longest,
            "eval_max_model_len_needed": longest + pinned.EVAL_MAX_TOKENS,
            "eval_max_model_len_pinned": pinned.EVAL_MAX_MODEL_LEN}


def _every_kth(n_lines: int, k: int) -> str:
    return "LINES: " + ", ".join(map(str, range(1, n_lines + 1, k)))


def _keyword_lines(func: str) -> str:
    hits = [i for i, line in enumerate(pinned.split_lines(func), 1) if any(k in line for k in pinned.LINE_KEYWORDS)]
    return "LINES: " + (", ".join(map(str, hits)) if hits else "none")


def baselines_section(rows: list[dict], facts: dict[str, dict], counts: Counter, shortcut: Fraction, base: dict, graph: CweGraph) -> dict:
    """Trivial baselines on non-test items, scored by the same verifiers the arms are."""
    def score(t: str, reply_for) -> tuple[list[verifiers.Verdict], list[dict]]:
        chosen = [r for r in rows if r["type"] == t]
        return [verifiers.verify_item(t, reply_for(r), r["gold"], graph) for r in chosen], chosen

    most_frequent = min(counts.items(), key=lambda kv: (-kv[1], int(kv[0][4:])))[0]
    best_constant = base["exact_id_hierarchy"]["adopted_schedule_top"][0]["cwe"]
    majority = base["cvss_majority"]["vector"]
    out = {}
    v, _ = score("mcq", lambda r: "ANSWER: A")
    out["mcq_always_A"] = _mean([x.metric for x in v])
    out["mcq_most_familiar_option"] = float(shortcut)
    for name, cwe in (("exact_id_most_frequent", most_frequent), ("exact_id_best_constant", best_constant)):
        v, _ = score("exact_id", lambda r, c=cwe: c)
        out[name] = {"answer": cwe, "exact": _mean([x.metric for x in v]), "hierarchy": _mean([x.dense for x in v])}
    v, _ = score("cvss", lambda r: majority)
    out["cvss_majority_vector"] = {"answer": majority, "agreement": _mean([x.metric for x in v])}
    v, chosen = score("find_error", lambda r: "VULNERABLE: yes")
    by_cve: dict[str, dict[int, verifiers.Verdict]] = {}
    for verdict, r in zip(v, chosen):
        by_cve.setdefault(r["cve_id"], {})[r["index"]] = verdict
    out["find_error_always_vulnerable"] = {
        "per_function": _mean([x.metric for x in v]),
        "paired": _mean([verifiers.paired_accuracy(p[0], p[1]) for p in by_cve.values()]),
    }
    for k in pinned.EVERY_KTH_LINE:
        v, _ = score("line_loc", lambda r, k=k: _every_kth(r["gold"]["n_lines"], k))
        out[f"line_loc_every_{k}th_line"] = _mean([x.metric for x in v])
    v, _ = score("line_loc", lambda r: _keyword_lines(facts[r["cve_id"]]["vuln_func"]))
    out["line_loc_keywords"] = {"keywords": list(pinned.LINE_KEYWORDS), "f1": _mean([x.metric for x in v])}
    return out


def git_section(facts: list[dict]) -> dict:
    """Our gold lines against an independent Myers diff (etl.verify_sheet.git_gold) over the whole
    table. A numbering error would shift replaced/deleted lines; comment edits (which git counts and
    we mask) and where an insertion is anchored are expected differences, and are counted apart."""
    kinds: Counter = Counter()
    rd_agree = 0
    f1s: list[Fraction] = []
    for f in facts:
        g = git_gold(f["vuln_func"], f["patched_func"])
        if g is None:
            kinds["unavailable"] += 1
            continue
        ours = set(f["patch_lines"])
        r = pinned.patch_line_set(f["vuln_func"], f["patched_func"])
        text = pinned.mask_comments(f["vuln_func"])[0] if f["comment_mask_ok"] else f["vuln_func"]
        code = {i for i, line in enumerate(pinned.split_lines(text), 1) if pinned.line_key(line)}
        replaced = {r.vuln_code[i] for tag, i1, i2, *_ in r.opcodes if tag in ("replace", "delete") for i in range(i1, i2)}
        g_code = g & code
        rd_agree += replaced <= g_code
        f1s.append(verifiers.line_f1(sorted(g_code), sorted(ours)) if g_code else Fraction(0))
        if g == ours:
            kinds["agree"] += 1
        elif g_code == ours:
            kinds["comment_lines_only"] += 1
        elif ours - g_code <= ours - replaced:
            kinds["insertion_anchor_only"] += 1
        else:
            kinds["other_alignment"] += 1
    checked = sum(v for k, v in kinds.items() if k != "unavailable")
    return {
        "kinds": dict(sorted(kinds.items())),
        "agreement_rate": kinds["agree"] / checked if checked else None,
        "replaced_deleted_lines_agree": rd_agree,
        "checked": checked,
        "git_f1_under_pinned_matcher": _mean(f1s),
        "git_f1_zero": sum(x == 0 for x in f1s),
    }


def report(paths: Paths, dry: bool = False, git: bool = True) -> dict:
    files = BankFiles.of(paths, dry=dry)
    facts, split, graph = load_inputs(paths)
    by_cve = {f["cve_id"]: f for f in facts}
    meta = json.loads(files.meta.read_text(encoding="utf-8"))
    bank = {pool: read_jsonl(files.bank(pool)) for pool in POOLS}
    decisions = read_jsonl(files.mcq_decisions)
    gens = [] if dry else read_jsonl(files.mcq_generations) + (read_jsonl(files.mcq_retry_generations)
                                                              if files.mcq_retry_generations.exists() else [])
    counts = mcq.gold_counts(facts, {c: s["pool"] for c, s in split.items()})
    base = json.loads(paths.baselines_json.read_text(encoding="utf-8"))
    out = {
        "dry": dry,
        "counts": meta["counts"],
        "mcq": mcq_section(bank, decisions, gens, meta, graph),
        "tokens": tokens_section(bank),
        "baselines_nontest": baselines_section(bank["nontest"], by_cve, counts, mcq.shortcut(decisions, counts), base, graph),
        "git_cross_check": git_section(facts) if git else None,
    }
    write_json(files.report_json, out)
    files.report_md.write_text(render_markdown(out), encoding="utf-8")
    return out


def pilot_report(paths: Paths) -> dict:
    """The pilot decides nothing: admission, refusals, and an early reading of the shortcut."""
    files = BankFiles.of(paths, pilot=True)
    facts, split, graph = load_inputs(paths)
    pool_of = {c: s["pool"] for c, s in split.items()}
    counts = mcq.gold_counts(facts, pool_of)
    gens = {r["custom_id"]: r for r in read_jsonl(files.mcq_generations)}
    requests = read_jsonl(files.mcq_requests)
    by_cve = {f["cve_id"]: f for f in facts}
    decisions, statuses, reasons, full = [], Counter(), Counter(), 0
    for r in requests:
        f = by_cve[r["cve_id"]]
        d = mcq.decide(f, pool_of[f["cve_id"]], [gens.get(r["custom_id"])], counts, graph)
        decisions.append(d)
        a = d["attempts"][0]
        statuses[a["status"]] += 1
        reasons.update(x for _, x in a["rejected"])
        full += len(a["admitted"]) >= pinned.MCQ_DISTRACTORS
    out = {
        "n": len(requests),
        "statuses": dict(sorted(statuses.items())),
        "three_admissible_on_first_attempt": full,
        "rejection_reasons": dict(sorted(reasons.items())),
        "refusal_categories": dict(sorted(Counter(str(g["stop_category"]) for g in gens.values()
                                                  if g["stop_reason"] == "refusal").items())),
        "shortcut_model_first_attempt_draw_filled": float(mcq.shortcut(decisions, counts)),
        "threshold": float(pinned.MCQ_SHORTCUT_MAX),
    }
    write_json(files.pilot_report_json, out)
    files.pilot_report_md.write_text("\n".join([
        "# Step 4: MCQ pilot", "",
        f"{out['n']} non-test CVEs. Statuses: {out['statuses']}. Three admissible on the first attempt: "
        f"{full} of {out['n']}. Rejections: {out['rejection_reasons'] or 'none'}. Refusals: {out['refusal_categories'] or 'none'}.", "",
        f"Shortcut on these items (first attempt, gaps filled by the draw): **{out['shortcut_model_first_attempt_draw_filled']:.3f}** "
        f"against the pre-registered cut-off {out['threshold']}. The pilot decides nothing; the guard runs on the full non-test bank.", "",
    ]), encoding="utf-8")
    return out


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(r: dict) -> str:
    m, t, b = r["mcq"], r["tokens"], r["baselines_nontest"]
    g = m["guard"]
    guard = ("dry run: every MCQ from the prior-matched draw" if g["mode"] == "dry" else
             f"shortcut with the model's picks {float(Fraction(g['shortcut_model'])):.3f} vs cut-off "
             f"{float(Fraction(g['threshold'])):.1f}: **{'fired, prior-matched draw used' if g['fired'] else 'not fired, model picks kept'}**")
    out = [
        "# Step 4: question bank" + (" (dry run, not frozen)" if r["dry"] else ""), "",
        "## Size", "",
        _table(("Pool", "CVEs", "Items"), ((p, f"{c['cves']:,}", f"{c['items']:,}") for p, c in r["counts"].items())), "",
        "## MCQ", "",
        f"- Guard: {guard}. The draw alone scores {float(Fraction(g['shortcut_draw'])):.3f}.",
        f"- Gold-letter marginals: " + "; ".join(f"{p} {v}" for p, v in m["gold_letter_marginals"].items()),
        f"- Distractor sources: " + "; ".join(f"{p} {v}" for p, v in m["distractor_sources"].items()),
        f"- Items with at least one drawn distractor: " + ", ".join(f"{p} {v:,}" for p, v in m["items_with_any_draw"].items()),
        f"- Attempt statuses: {m['attempt_statuses'] or '—'}; rejection reasons: {m['rejection_reasons'] or '—'}; "
        f"refusal categories: {m['refusal_categories'] or '—'}",
        f"- Distractor distance from gold (ChildOf hops): {m['distractor_distance']}", "",
        "## Prompt tokens", "",
        _table(("Type", *(k for k, _ in next(iter(t["quantiles"].values())))),
               ((k, *(f"{v:,}" for _, v in q)) for k, q in t["quantiles"].items())), "",
        f"Longest prompt {t['max_prompt_tokens']:,} tokens, so evaluation needs max_model_len ≥ "
        f"{t['eval_max_model_len_needed']:,} (pinned now: {t['eval_max_model_len_pinned']:,}).", "",
        "## Trivial baselines (non-test)", "",
        _table(("Baseline", "Score"), [
            ("MCQ always A", f"{b['mcq_always_A']:.3f}"),
            ("MCQ most familiar option", f"{b['mcq_most_familiar_option']:.3f}"),
            (f"Exact-ID most frequent ({b['exact_id_most_frequent']['answer']})",
             f"{b['exact_id_most_frequent']['exact']:.3f} exact, {b['exact_id_most_frequent']['hierarchy']:.3f} hierarchy"),
            (f"Exact-ID best constant ({b['exact_id_best_constant']['answer']})",
             f"{b['exact_id_best_constant']['exact']:.3f} exact, {b['exact_id_best_constant']['hierarchy']:.3f} hierarchy"),
            ("CVSS majority vector", f"{b['cvss_majority_vector']['agreement']:.3f}"),
            ("Find-the-error always vulnerable",
             f"{b['find_error_always_vulnerable']['per_function']:.3f} per function, {b['find_error_always_vulnerable']['paired']:.3f} paired"),
            *((f"Line loc every k-th line, k = {k}", f"{b[f'line_loc_every_{k}th_line']:.3f}") for k in pinned.EVERY_KTH_LINE),
            (f"Line loc keywords ({', '.join(b['line_loc_keywords']['keywords'])})", f"{b['line_loc_keywords']['f1']:.3f}"),
        ]), "",
    ]
    gc = r["git_cross_check"]
    if gc:
        k = gc["kinds"]
        out += ["## Gold lines vs an independent git diff (whole table)", "",
                f"- Exact agreement: {k.get('agree', 0):,} of {gc['checked']:,} ({gc['agreement_rate']:.1%}).",
                f"- Differ only on comment lines (git counts them, we mask them): {k.get('comment_lines_only', 0):,}.",
                f"- Differ only in where inserted code is anchored: {k.get('insertion_anchor_only', 0):,}.",
                f"- Other alignment differences (repeated lines matched differently): {k.get('other_alignment', 0):,}.",
                f"- Replaced/deleted gold lines all flagged by git too: {gc['replaced_deleted_lines_agree']:,} of {gc['checked']:,}. "
                "A numbering error would break this.",
                f"- git's code lines scored against our gold under the pinned ±1 matcher: mean F1 {gc['git_f1_under_pinned_matcher']:.3f}; "
                f"{gc['git_f1_zero']:,} CVEs at 0 (an insertion anchored more than one line away).", ""]
    return "\n".join(out)
