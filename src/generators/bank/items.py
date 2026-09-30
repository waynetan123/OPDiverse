"""One fact -> its six question-bank items. Pure; the templates are pinned.BANK_PROMPTS."""

from __future__ import annotations

import re

from etl import pinned
from probe.prompts import render_qwen_chat


def redact_own_cve(text: str, cve_id: str) -> str:
    return re.sub(rf"(?<![A-Za-z0-9]){re.escape(cve_id)}(?!\d)", pinned.CVE_REDACTED, text, flags=re.IGNORECASE)


def prompt_description(fact: dict) -> str:
    """The NVD description as prompts show it: literal CWE IDs and the CVE's own ID redacted."""
    return redact_own_cve(pinned.CWE_LITERAL.sub(pinned.CWE_REDACTED, fact["description"]), fact["cve_id"])


def prompt_function(fact: dict, side: str) -> str:
    """The stored function with the CVE's own ID redacted (5 patched functions cite it in a
    comment). Redaction stays inside a line, so line numbers are unchanged."""
    return redact_own_cve(fact[f"{side}_func"], fact["cve_id"])


def item_id(cve_id: str, item_type: str, index: int = 0) -> str:
    return f"{cve_id}:{item_type}:{index}"


def render_options(options: list[dict]) -> str:
    return "\n".join(pinned.MCQ_OPTION.format(**o) for o in options)


def line_target(lines: list[int]) -> str:
    return "LINES: " + (", ".join(map(str, lines)) if lines else "none")


def build_items(fact: dict, pool: str, cluster_id: str, options: list[dict], gold_letter: str) -> list[dict]:
    """The six items in BANK_TYPES order (find_error: 0 vulnerable, 1 patched), without prompt_tokens."""
    cve, cwe = fact["cve_id"], fact["cwe"]
    desc = prompt_description(fact)
    vuln = prompt_function(fact, "vuln")
    specs = [
        ("mcq", 0, {"description": desc, "options": render_options(options)}, f"ANSWER: {gold_letter}",
         {"letter": gold_letter, "cwe": cwe, "options": options}),
        ("exact_id", 0, {"description": desc}, cwe, {"cwe": cwe}),
        ("cvss", 0, {"description": desc}, fact["cvss_vector"], {"vector": fact["cvss_vector"]}),
        ("find_error", 0, {"function": vuln}, f"VULNERABLE: yes, {cwe}", {"vulnerable": True, "cwe": cwe}),
        ("find_error", 1, {"function": prompt_function(fact, "patched")}, "VULNERABLE: no", {"vulnerable": False, "cwe": cwe}),
        ("line_loc", 0, {"description": desc, "numbered": pinned.render_numbered(vuln)}, line_target(fact["patch_lines"]),
         {"lines": fact["patch_lines"], "n_lines": fact["n_lines"]}),
    ]
    rows = []
    for item_type, index, fields, target, gold in specs:
        user = pinned.BANK_PROMPTS[item_type].format(**fields)
        rows.append({
            "item_id": item_id(cve, item_type, index),
            "cve_id": cve,
            "type": item_type,
            "index": index,
            "pool": pool,
            "cluster_id": cluster_id,
            "user": user,
            "prompt": render_qwen_chat(user),
            "target": target,
            "gold": gold,
            "template_version": pinned.BANK_TEMPLATE_VERSION,
        })
    assert len(rows) == pinned.ITEMS_PER_CVE
    return rows
