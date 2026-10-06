"""Training rows for one run: the frozen step-9 file, the run order, and the reference tokenisation. Pure.

The run order is ours, not the trainer's: pass p over the file is sorted by stable_rank(key, TRAIN_ORDER_SALT, seed,
p), where the key is (item_id, copy), or cve_id for base documents. Every question arm of a configuration and seed
holds the same (item_id, copy) rows, so optimizer step i trains on the same items in every arm. A run of S steps reads
S x TRAIN_EXAMPLES_PER_STEP rows, cycling the file (base: about six passes per question-arm pass).

The reference tokenisation is what the model must see: the bank's prompt ids exactly as evaluation tokenises them,
then the completion tokenised on its own (the generation-time boundary), ending in exactly one end token, which
carries a label. The trainer's prepared ids are compared with it before any step (`compare_prepared`), which catches
truncation, a second end token and a different prompt-completion boundary.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from converters.files import ConverterFiles, read_content, sha256_bytes
from etl import pinned
from etl.paths import Paths

Encode = Callable[[list[str]], list[list[int]]]


# ---------------------------------------------------------------------------
# The frozen file
# ---------------------------------------------------------------------------


def training_file(paths: Paths, arm: str, config: str, seed: int) -> tuple[Path, str]:
    """(path, content sha256) of the step-9 file an arm reads, from manifest.json, checked against
    converters_meta.json so a run can never read anything but the frozen file."""
    files = ConverterFiles.of(paths)
    manifest = json.loads(files.manifest.read_text(encoding="utf-8"))
    meta = json.loads(files.meta.read_text(encoding="utf-8"))
    rel = manifest["seeds"][str(seed)]["files"][config][arm]
    path = files.dir / rel
    sha = sha256_bytes(read_content(path))
    if meta["outputs"][rel]["content_sha256"] != sha:
        raise SystemExit(f"{rel} is not the frozen step-9 file (content sha256 differs from converters_meta.json)")
    return path, sha


def order_key(row: dict) -> tuple[str, ...]:
    return (row["cve_id"],) if "text" in row else (row["item_id"], str(row["copy"]))


def run_order(rows: list[dict], n: int, seed: int) -> list[dict]:
    """The n rows a run reads, in order: whole passes over `rows`, each in its own seeded shuffle."""
    if not rows:
        raise ValueError("no rows")
    out: list[dict] = []
    for p in range(-(-n // len(rows))):
        out += sorted(rows, key=lambda r: (pinned.stable_rank(*order_key(r), pinned.TRAIN_ORDER_SALT, str(seed), str(p)),
                                           order_key(r)))
    return out[:n]


def passes(n_rows: int, steps: int) -> float:
    return steps * pinned.TRAIN_EXAMPLES_PER_STEP / n_rows


# ---------------------------------------------------------------------------
# Reference tokenisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoded:
    """One row's reference ids. SFT-style rows: input_ids with a loss mask; DPO rows: prompt, chosen and rejected;
    GRPO rows: the prompt only (the completion is generated)."""
    key: tuple[str, ...]
    input_ids: tuple[int, ...] = ()
    loss_mask: tuple[int, ...] = ()
    prompt_ids: tuple[int, ...] = ()
    chosen_ids: tuple[int, ...] = ()
    rejected_ids: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        if self.chosen_ids:
            return len(self.prompt_ids) + max(len(self.chosen_ids), len(self.rejected_ids))
        return len(self.input_ids) or len(self.prompt_ids)


def end_problems(ids: list[int], end_id: int, other_end_id: int | None, where: str) -> list[str]:
    """A completion or document must end in exactly one `end_id` and hold no other end token."""
    bad = []
    if not ids or ids[-1] != end_id:
        bad.append(f"{where}: does not end in its end token")
    if end_id in ids[:-1]:
        bad.append(f"{where}: holds a second end token")
    if other_end_id is not None and other_end_id in ids:
        bad.append(f"{where}: holds the other end token")
    return bad


def encode_rows(trainer: str, rows: list[dict], encode: Encode, token_id: Callable[[str], int | None],
                prompt_tokens: dict[str, int]) -> tuple[dict[tuple[str, ...], Encoded], list[str]]:
    """Reference ids for the distinct rows of a file (keyed by order_key), and every guard violation:
    a prompt whose ids differ in count from the bank's (evaluation's) count, and end-token problems."""
    im_end, eot = token_id(pinned.COMPLETION_END), token_id(pinned.BASE_DOC_END)
    if im_end is None or eot is None:
        raise SystemExit("the tokenizer lacks the pinned end tokens")
    distinct = {order_key(r): r for r in rows}
    keys = list(distinct)
    bad: list[str] = []
    out: dict[tuple[str, ...], Encoded] = {}
    if trainer == "sft" and all("text" in r for r in distinct.values()):          # base documents
        for k, ids in zip(keys, encode([distinct[k]["text"] for k in keys])):
            bad += end_problems(ids, eot, im_end, f"{k[0]} document")
            out[k] = Encoded(k, tuple(ids), (1,) * len(ids))
        return out, bad
    prompts = encode([distinct[k]["prompt"] for k in keys])
    for k, p in zip(keys, prompts):
        item = distinct[k]["item_id"]
        if len(p) != prompt_tokens[item]:
            bad.append(f"{item}: the prompt tokenises to {len(p)} ids, the bank counted {prompt_tokens[item]}")
    if trainer == "sft":
        comps = encode([distinct[k]["completion"] for k in keys])
        for k, p, c in zip(keys, prompts, comps):
            bad += end_problems(c, im_end, eot, f"{k[0]} completion")
            out[k] = Encoded(k, tuple(p + c), (0,) * len(p) + (1,) * len(c))
    elif trainer == "dpo":
        chosen = encode([distinct[k]["chosen"] for k in keys])
        rejected = encode([distinct[k]["rejected"] for k in keys])
        for k, p, c, r in zip(keys, prompts, chosen, rejected):
            bad += end_problems(c, im_end, eot, f"{k[0]} chosen") + end_problems(r, im_end, eot, f"{k[0]} rejected")
            if c == r:
                bad.append(f"{k[0]}: chosen and rejected are identical")
            out[k] = Encoded(k, prompt_ids=tuple(p), chosen_ids=tuple(c), rejected_ids=tuple(r))
    elif trainer == "grpo":
        for k, p in zip(keys, prompts):
            if len(p) + distinct[k]["max_completion_tokens"] > pinned.EVAL_MAX_MODEL_LEN:
                bad.append(f"{k[0]}: prompt plus its cap exceeds EVAL_MAX_MODEL_LEN")
            out[k] = Encoded(k, prompt_ids=tuple(p))
    else:
        raise ValueError(f"unknown trainer {trainer!r}")
    return out, bad


def longest(encoded: dict[tuple[str, ...], Encoded], cap: dict[tuple[str, ...], int] | None = None) -> int:
    """The trainer's maximum length: the longest reference row (GRPO: prompt plus its cap), so nothing is cut."""
    return max(e.length + (cap or {}).get(k, 0) for k, e in encoded.items())


def longest_first(order: list[dict], encoded: dict[tuple[str, ...], Encoded],
                  cap: dict[tuple[str, ...], int] | None = None) -> list[dict]:
    """Pilot only: the same rows, longest first (GRPO: prompt plus cap), so a short run meets the worst-case memory."""
    size = lambda r: encoded[order_key(r)].length + (cap or {}).get(order_key(r), 0)  # noqa: E731
    return sorted(order, key=lambda r: (-size(r), order_key(r)))


def compare_prepared(reference: list[tuple[int, ...]], prepared: list[list[int]], label: str, limit: int = 3) -> list[str]:
    """The trainer's prepared ids, row by row, against the reference ids. Any difference stops the run."""
    if len(reference) != len(prepared):
        return [f"{label}: the trainer prepared {len(prepared)} rows, the reference has {len(reference)}"]
    bad = [i for i, (a, b) in enumerate(zip(reference, prepared)) if list(a) != list(b)]
    out = []
    for i in bad[:limit]:
        a, b = list(reference[i]), list(prepared[i])
        j = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        out.append(f"{label}: row {i} differs from the reference at token {j} (reference {len(a)} ids, trainer {len(b)}; "
                   f"reference tail {a[-3:]}, trainer tail {b[-3:]})")
    if len(bad) > limit:
        out.append(f"{label}: {len(bad) - limit} more rows differ")
    return out


def grpo_dataset_rows(rows: list[dict]) -> list[dict]:
    """What the GRPO trainer's dataset carries per row: the prompt text and what the reward and caps need. The gold
    is JSON text, because the types' gold dicts have different fields."""
    return [{"prompt": r["prompt"], "item_id": r["item_id"], "type": r["type"], "index": r["index"],
             "gold_json": json.dumps(r["gold"], sort_keys=True), "max_completion_tokens": r["max_completion_tokens"],
             "dynamic_sampling": r["dynamic_sampling"]} for r in rows]
