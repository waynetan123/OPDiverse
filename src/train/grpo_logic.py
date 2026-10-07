"""The GRPO pieces that do not need a GPU: the reward, per-type caps, dynamic sampling, the running monitor and the
weight-sync decision. train.grpo wires them into TRL's GRPOTrainer. Pure.

Reward (plan, Reward shaping): verifiers.verify_item(type, completion, gold).dense, which is 0 on a parse failure.
No format credit. A rollout cut off at its type's cap is scored on its text (mask_truncated_completions=False), as
the step-5 audit scored it.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from etl import pinned, verifiers


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def verdicts(types: list[str], completions: list[str], golds_json: list[str], graph) -> list[verifiers.Verdict]:
    return [verifiers.verify_item(t, c, json.loads(g), graph) for t, c, g in zip(types, completions, golds_json, strict=True)]


def rewards(vs: list[verifiers.Verdict]) -> list[float]:
    return [float(v.dense) for v in vs]


# ---------------------------------------------------------------------------
# Finding TRL's internals (they move between TRL versions)
# ---------------------------------------------------------------------------

# Methods that push the trainer's weights into vLLM, by TRL version, looked for on the trainer, then on the object
# holding the vLLM engine.
SYNC_METHODS = ("_move_model_to_vllm", "sync_weights", "_sync_weights", "update_vllm_weights", "load_weights")


def find_instances(root, cls, depth: int = 2) -> list[tuple[str, object]]:
    """(attribute path, object) for every instance of `cls` reachable from `root` through at most `depth` plain
    attributes (instance __dict__ only, so no property runs), each object once, shortest path first."""
    found, seen = [], {id(root)}
    level = [("", root)]
    for _ in range(depth):
        nxt = []
        for path, obj in level:
            for name, value in list(getattr(obj, "__dict__", {}).items()):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                p = f"{path}.{name}" if path else name
                if isinstance(value, cls):
                    found.append((p, value))
                else:
                    nxt.append((p, value))
        level = nxt
    return found


def resolve(root, path: str):
    obj = root
    for name in filter(None, path.split(".")):
        obj = getattr(obj, name)
    return obj


def parent_path(path: str) -> str:
    return path.rpartition(".")[0]


def find_method(owners: list[tuple[str, object]], names: tuple[str, ...]) -> tuple[str, object, str] | None:
    """(owner path, owner, method name) for the first of `names` callable on the first owner that has one."""
    for path, owner in owners:
        for name in names:
            if callable(getattr(owner, name, None)):
                return path, owner, name
    return None


def names_like(obj, words: tuple[str, ...]) -> list[str]:
    """Attribute names on obj (class and instance) containing any of `words`, for an error that says where to look."""
    return sorted({n for n in dir(obj) if any(w in n.lower() for w in words)})


# ---------------------------------------------------------------------------
# Per-type caps
# ---------------------------------------------------------------------------


class CapLookup:
    """Prompt (text, or token ids) -> its row's max_completion_tokens. Every training prompt is known; anything else
    (a canary) is an error unless asked for explicitly."""

    def __init__(self, by_text: dict[str, int], by_ids: dict[tuple[int, ...], int]):
        self.by_text, self.by_ids = by_text, by_ids

    def __call__(self, prompt) -> int:
        if isinstance(prompt, str):
            key, table = prompt, self.by_text
        elif isinstance(prompt, dict):
            if "prompt_token_ids" in prompt:
                key, table = tuple(prompt["prompt_token_ids"]), self.by_ids
            else:
                key, table = prompt["prompt"], self.by_text
        else:
            key, table = tuple(prompt), self.by_ids
        if key not in table:
            raise SystemExit("a rollout prompt is not a training prompt of this run; its cap is unknown")
        return table[key]


def sampling_fields(cap: int, n: int, logprobs) -> dict:
    """The complete vLLM SamplingParams for one rollout request: ROLLOUT_SAMPLING, its type's cap, the stop tokens
    are added by the caller. Nothing is left to vLLM's or Qwen's defaults that the pins cover."""
    return {**pinned.ROLLOUT_SAMPLING, "n": n, "max_tokens": cap, "logprobs": logprobs}


# ---------------------------------------------------------------------------
# Dynamic sampling
# ---------------------------------------------------------------------------


@dataclass
class TypeQueue:
    """Replacement prompts per type, in run order, wrapping; counts every prompt's appearances (the logged deviation
    from the identical-fact-set claim)."""
    rows: dict[str, list[dict]]
    pos: dict[str, int] = field(default_factory=dict)
    replacements: Counter = field(default_factory=Counter)

    @classmethod
    def of(cls, ordered_rows: list[dict], types: set[str]) -> TypeQueue:
        by_type: dict[str, list[dict]] = defaultdict(list)
        seen = set()
        for r in ordered_rows:
            if r["type"] in types and r["item_id"] not in seen:
                seen.add(r["item_id"])
                by_type[r["type"]].append(r)
        return cls(dict(by_type))

    def next(self, item_type: str) -> dict:
        rows = self.rows[item_type]
        i = self.pos.get(item_type, 0)
        self.pos[item_type] = (i + 1) % len(rows)
        self.replacements[rows[i]["item_id"]] += 1
        return rows[i]


def groups(item_ids: list[str], size: int) -> list[list[int]]:
    """Consecutive runs of `size` completions of one prompt (TRL's RepeatSampler order); exits if the batch is not
    laid out that way."""
    if len(item_ids) % size:
        raise SystemExit(f"{len(item_ids)} completions are not whole groups of {size}")
    out = []
    for g in range(len(item_ids) // size):
        idx = list(range(g * size, (g + 1) * size))
        if len({item_ids[i] for i in idx}) != 1:
            raise SystemExit("a group of completions mixes prompts; the batch layout is not TRL's")
        out.append(idx)
    return out


def tied(group_rewards: list[float]) -> bool:
    return len(set(group_rewards)) == 1


def to_replace(group_types: list[str], group_rewards: list[list[float]], dynamic_types: set[str]) -> list[int]:
    """Groups to drop and replace: tied groups of a dynamic-sampling type."""
    return [g for g, (t, rs) in enumerate(zip(group_types, group_rewards, strict=True)) if t in dynamic_types and tied(rs)]


# ---------------------------------------------------------------------------
# Running monitor
# ---------------------------------------------------------------------------


def _predicted_vulnerable(v: verifiers.Verdict) -> bool | None:
    return None if not v.parse_ok else v.parsed.startswith("VULNERABLE: yes")


def _set_size(v: verifiers.Verdict) -> int | None:
    if not v.parse_ok:
        return None
    body = v.parsed[len("LINES: "):]
    return 0 if body == "none" else len(body.split(", "))


@dataclass
class Monitor:
    """The plan's running audit, per type over a window of steps: live groups (dense and binary), reward mean and
    variance, parse and cut-off rates, line-localisation set sizes, find-the-error predictions by class. Events:
    a type below the AUDIT_FLOOR live fraction, and a find-the-error collapse (one predicted class above
    COLLAPSE_RATE of parsed rollouts)."""
    rollouts: dict = field(default_factory=lambda: defaultdict(list))   # type -> [(index, verdict, truncated, n_tokens)]
    group_live: dict = field(default_factory=lambda: defaultdict(list))  # type -> [(dense live, binary live)]

    def add_group(self, item_type: str, index: int, vs: list[verifiers.Verdict], truncated: list[bool],
                  n_tokens: list[int]) -> None:
        self.rollouts[item_type] += [(index, v, c, n) for v, c, n in zip(vs, truncated, n_tokens, strict=True)]
        self.group_live[item_type].append((len({v.dense for v in vs}) > 1, len({v.metric == 1 for v in vs}) > 1))

    def summary(self, step: int) -> dict:
        types, events = {}, []
        for t in pinned.BANK_TYPES:
            rs, gs = self.rollouts.get(t, []), self.group_live.get(t, [])
            if not rs:
                continue
            dense = [float(v.dense) for _, v, _, _ in rs]
            mean = math.fsum(dense) / len(dense)
            s = {
                "groups": len(gs),
                "live_dense": sum(d for d, _ in gs) / len(gs),
                "live_binary": sum(b for _, b in gs) / len(gs),
                "reward_mean": mean,
                "reward_var": math.fsum((x - mean) ** 2 for x in dense) / len(dense),
                "parse_rate": sum(v.parse_ok for _, v, _, _ in rs) / len(rs),
                "cut_off_rate": sum(c for _, _, c, _ in rs) / len(rs),
                "mean_tokens": sum(n for *_, n in rs) / len(rs),
            }
            if s["live_dense"] < pinned.AUDIT_FLOOR:
                events.append({"step": step, "type": t, "event": "below_floor", "live_dense": s["live_dense"]})
            if t == "line_loc":
                sizes = Counter(_set_size(v) for _, v, _, _ in rs if v.parse_ok)
                s["set_size"] = {str(k): n for k, n in sorted(sizes.items())}
            if t == "find_error":
                preds = [(i, _predicted_vulnerable(v)) for i, v, _, _ in rs]
                parsed = [p for _, p in preds if p is not None]
                for cls, name in ((0, "vulnerable_items"), (1, "patched_items")):
                    mine = [p for i, p in preds if i == cls and p is not None]
                    s[f"predicted_vulnerable_{name}"] = sum(mine) / len(mine) if mine else None
                rate = sum(parsed) / len(parsed) if parsed else None
                s["predicted_vulnerable"] = rate
                if rate is not None and max(rate, 1 - rate) > pinned.COLLAPSE_RATE:
                    events.append({"step": step, "type": t, "event": "collapse", "predicted_vulnerable": rate})
            types[t] = s
        return {"step": step, "types": types, "events": events}

    def reset(self) -> None:
        self.rollouts.clear()
        self.group_live.clear()


# ---------------------------------------------------------------------------
# Weight sync
# ---------------------------------------------------------------------------


def mean_abs_gap(a: list[float], b: list[float]) -> float:
    return math.fsum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def sync_verdict(gap_current: float, gap_start: float, drift: float) -> tuple[bool | None, str]:
    """(ok, reason). gap_current: vLLM's token logprobs against the trainer's current weights; gap_start: against
    the starting weights; drift: current against start, on vLLM's tokens. Undecidable (None) until the policy has
    moved more than SYNC_MIN_DRIFT; then vLLM must sit nearer the current weights than the start."""
    if drift <= pinned.SYNC_MIN_DRIFT:
        return None, f"policy drift {drift:.4f} <= {pinned.SYNC_MIN_DRIFT}: too early to tell"
    if gap_current < pinned.SYNC_RATIO * gap_start:
        return True, f"vLLM matches the current weights ({gap_current:.4f}) far better than the start ({gap_start:.4f})"
    return False, (f"vLLM's logprobs sit {gap_current:.4f} from the current weights and {gap_start:.4f} from the start: "
                   "rollouts are not coming from the current weights")
