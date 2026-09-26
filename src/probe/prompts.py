"""Build the probe's request file: the pinned sample, prompts rendered with Qwen's chat template."""

from __future__ import annotations

import json

from etl import pinned
from etl.manifest import file_sha256
from etl.paths import Paths

QUESTIONS = ("cwe", "cvss")


def render_qwen_chat(user: str) -> str:
    """Qwen2.5's chat template for one user message, no system message, generation prompt on.
    run_vllm.py checks this byte for byte against tokenizer.apply_chat_template at the pinned revision."""
    return (
        f"<|im_start|>system\n{pinned.QWEN_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def sample(test_ids: list[str]) -> list[str]:
    """The PROBE_SAMPLE_SIZE test CVEs with the lowest stable rank, returned sorted by CVE ID."""
    ranked = sorted(test_ids, key=lambda c: (pinned.stable_rank(c, pinned.PROBE_SALT), c))
    if len(ranked) < pinned.PROBE_SAMPLE_SIZE:
        raise ValueError(f"only {len(ranked)} test CVEs; the probe needs {pinned.PROBE_SAMPLE_SIZE}")
    return sorted(ranked[: pinned.PROBE_SAMPLE_SIZE])


def build_requests(paths: Paths) -> list[dict]:
    with open(paths.split, encoding="utf-8") as f:
        test_ids = [row["cve_id"] for row in map(json.loads, f) if row["pool"] == "test"]
    sources = {"split_sha256": file_sha256(paths.split), "facts_sha256": file_sha256(paths.facts)}
    requests = []
    for cve_id in sample(test_ids):
        for question in QUESTIONS:
            user = pinned.PROBE_PROMPTS[question].format(cve_id=cve_id)
            requests.append({
                "request_id": f"{cve_id}:{question}",
                "cve_id": cve_id,
                "question": question,
                "messages": [{"role": "user", "content": user}],
                "prompt": render_qwen_chat(user),
                "model": pinned.TOKENIZER_REPO,
                "revision": pinned.TOKENIZER_REVISION,
                "sampling": pinned.EVAL_SAMPLING,
                **sources,
            })
    return requests


def write_requests(paths: Paths) -> list[dict]:
    requests = build_requests(paths)
    paths.probe.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in requests)
    paths.probe_requests.write_text(body, encoding="utf-8")
    return requests
