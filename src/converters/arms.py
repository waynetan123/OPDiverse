"""Per-arm training rows. Pure: every arm reads the bank's byte-identical prompt, and only what sits beside it
differs. Completions end in pinned.COMPLETION_END; the trainer must not append a second one."""

from __future__ import annotations

from etl import pinned
from etl.cwe_graph import CweGraph
from generators.bank.items import prompt_description, prompt_function

END = pinned.COMPLETION_END


def _common(item: dict, copy: int) -> dict:
    return {"item_id": item["item_id"], "cve_id": item["cve_id"], "type": item["type"], "index": item["index"],
            "copy": copy}


def sft(item: dict, copy: int) -> dict:
    return {**_common(item, copy), "prompt": item["prompt"], "completion": item["target"] + END}


def distill_self(item: dict, copy: int, rationale: dict) -> dict:
    """The step-5 target: the rationale, a blank line and the canonical answer, or the answer alone (gold_only)."""
    return {**_common(item, copy), "prompt": item["prompt"], "completion": rationale["target"] + END,
            "source": rationale["source"]}


def dpo(item: dict, copy: int, pair: dict) -> dict:
    """Chosen is the gold target; rejected is the step-6 near miss. Also read by SFT->DPO."""
    return {**_common(item, copy), "prompt": item["prompt"], "chosen": pair["chosen"] + END,
            "rejected": pair["rejected"] + END, "rejected_source": pair["source"],
            "rejected_dense": pair["rejected_dense"]}


def grpo(item: dict, copy: int, grpo_types: dict) -> dict:
    """The prompt and what the reward needs: the bank's gold (verifiers.verify_item takes it as is), the
    step-5 rollout cap and the audit band (dynamic_sampling on the types it names)."""
    t = grpo_types[item["type"]]
    return {**_common(item, copy), "prompt": item["prompt"], "gold": item["gold"],
            "max_completion_tokens": t["cap"], "band": t["band"],
            "dynamic_sampling": t["band"] == "dynamic_sampling"}


def base_document(fact: dict, graph: CweGraph) -> str:
    return pinned.BASE_DOC_TEMPLATE.format(
        description=prompt_description(fact),
        cwe=fact["cwe"],
        name=graph.weaknesses[fact["cwe"][4:]].name,
        vector=fact["cvss_vector"],
        vulnerable=prompt_function(fact, "vuln"),
        patched=prompt_function(fact, "patched"),
    )


def base(fact: dict, graph: CweGraph) -> dict:
    return {"cve_id": fact["cve_id"], "text": base_document(fact, graph) + pinned.BASE_DOC_END}


QUESTION_ROW = {"sft": sft, "distill_self": distill_self, "dpo": dpo, "grpo": grpo}
