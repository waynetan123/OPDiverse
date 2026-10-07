"""TRL's GRPOTrainer with the step-5 and plan pins applied. Imported on the GPU machine only (by train.run).

- Per-type caps: in colocate mode every rollout request reaches vLLM with its own SamplingParams, built whole from
  ROLLOUT_SAMPLING, the row's max_completion_tokens, both stop tokens and watermarking off, so nothing is left to TRL's,
  vLLM's or Qwen's defaults. TRL's own params are checked against the pins first.
- Dynamic sampling (types banded "dynamic_sampling" at step 5): after a generation batch is scored, every tied group
  of those types is replaced by a group for the next prompt of the same type (grpo_logic.TypeQueue), generated and
  scored the same way, and spliced in; a tied replacement is kept.
- The running monitor sees every kept group; train.run writes a summary every MONITOR_EVERY steps.
- The weight-sync canary runs every MONITOR_EVERY steps after the weights are moved to vLLM (grpo_logic.sync_verdict);
  a failure stops the run.

Everything here leans on TRL internals that move between versions: where the vLLM engine is kept, the weight-sync
method, `_generate_and_score_completions` and its output dict's keys. The engine is found by searching the trainer for
the vllm.LLM instance, and the sync method by name (grpo_logic.SYNC_METHODS); each hook must exist, or the trainer
stops at construction and lists what this TRL has instead, so no hook can be silently skipped.
`python -m train.run --arm grpo --check-only` exercises each one before any step.
"""

from __future__ import annotations

from etl import pinned

from . import grpo_logic

PROMPT_SIDE = ("prompt_ids", "prompt_mask")  # left-padded in TRL; everything else is right-padded


def pad_to(torch, t, width: int, left: bool, fill):
    if t.shape[1] == width:
        return t
    pad = torch.full((t.shape[0], width - t.shape[1]), fill, dtype=t.dtype, device=t.device)
    return torch.cat([pad, t] if left else [t, pad], dim=1)


def splice(torch, out: dict, new: dict, replaced: list[int], size: int, pad_id: int) -> list[str]:
    """Put `new`'s groups (in order) into `out` at the replaced groups' rows. Returns keys left as they were."""
    n = out["completion_ids"].shape[0]
    rows = [g * size + j for g in replaced for j in range(size)]
    src = list(range(len(rows)))
    untouched = []
    for key, v in list(out.items()):
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            w = new[key]
            if v.dim() == 2 and v.shape[1] != w.shape[1]:
                width, left = max(v.shape[1], w.shape[1]), key in PROMPT_SIDE
                fill = pad_id if key.endswith("_ids") else 0
                v, w = pad_to(torch, v, width, left, fill), pad_to(torch, w, width, left, fill)
            v = v.clone()
            v[rows] = w[src].to(v.dtype)
            out[key] = v
        elif key != "num_items_in_batch":
            untouched.append(key)
    if "num_items_in_batch" in out:
        total = out["completion_mask"].sum()
        out["num_items_in_batch"] = total if torch.is_tensor(out["num_items_in_batch"]) else int(total)
    return untouched


def trainer_class(trl, torch, vllm, sampling_kwargs, field_names):
    """The subclass, built against the imported libraries."""

    class PinnedGRPOTrainer(trl.GRPOTrainer):
        def __init__(self, *args, caps: grpo_logic.CapLookup, queue: grpo_logic.TypeQueue, dynamic_types: set[str],
                     rows_by_item: dict[str, dict], graph, monitor: grpo_logic.Monitor, canary: list[dict],
                     stop_ids: list[int], write_log, **kwargs):
            super().__init__(*args, **kwargs)
            self.caps, self.queue, self.dynamic_types = caps, queue, dynamic_types
            self.rows_by_item, self.graph, self.monitor, self.canary = rows_by_item, graph, monitor, canary
            self.stop_ids, self.write_log = stop_ids, write_log
            self.pending: list = []
            self.spliced_keys: set[str] = set()
            self.sync_checks: list[dict] = []
            self.generated_tokens = 0
            if not callable(getattr(trl.GRPOTrainer, "_generate_and_score_completions", None)):
                raise SystemExit("this TRL's GRPOTrainer has no _generate_and_score_completions (dynamic sampling and the "
                                 f"monitor hook it); generation-related methods: {grpo_logic.names_like(self, ('generat',))}")
            found = grpo_logic.find_instances(self, vllm.LLM)
            if len(found) != 1:
                raise SystemExit(f"expected one vLLM engine on the trainer (colocate mode), found {[p for p, _ in found]}; "
                                 f"vLLM-related attributes: {grpo_logic.names_like(self, ('vllm', 'llm'))}")
            self.vllm_path, self.vllm_llm = found[0]
            self.raw_generate = self.vllm_llm.generate
            self.vllm_llm.generate = self.capped_generate
            holder_path = grpo_logic.parent_path(self.vllm_path)
            owners = [("trainer", self)] + ([(holder_path, grpo_logic.resolve(self, holder_path))] if holder_path else [])
            hook = grpo_logic.find_method(owners, grpo_logic.SYNC_METHODS)
            if hook is None:
                raise SystemExit("found no weight-sync method " + str(grpo_logic.SYNC_METHODS) + "; candidates: "
                                 + "; ".join(f"{p}: {grpo_logic.names_like(o, ('sync', 'weight', 'move'))}" for p, o in owners))
            owner_path, owner, name = hook
            self.sync_hook = f"{owner_path}.{name}"
            original = getattr(owner, name)

            def synced(*a, **kw):
                result = original(*a, **kw)
                self.after_sync()
                return result

            setattr(owner, name, synced)

        # --- per-type caps ---------------------------------------------------------------------------------

        def capped_generate(self, *args, **kwargs):
            args = list(args)
            prompts = kwargs.pop("prompts") if "prompts" in kwargs else args.pop(0)
            sp = kwargs.pop("sampling_params") if "sampling_params" in kwargs else (args.pop(0) if args else None)
            sp = sp[0] if isinstance(sp, list) else sp
            if sp is None:
                raise SystemExit("TRL called vLLM without SamplingParams")
            for k in ("temperature", "top_p", "repetition_penalty"):
                if getattr(sp, k) != pinned.ROLLOUT_SAMPLING[k]:
                    raise SystemExit(f"TRL's rollout {k} is {getattr(sp, k)}, pinned {pinned.ROLLOUT_SAMPLING[k]}")
            if getattr(sp, "guided_decoding", None) is not None:
                raise SystemExit("TRL asked for guided decoding")
            prompts = prompts if isinstance(prompts, list) else [prompts]
            params = [vllm.SamplingParams(**sampling_kwargs(grpo_logic.sampling_fields(self.caps(p), sp.n, sp.logprobs),
                                                            field_names)[0], stop_token_ids=self.stop_ids)
                      for p in prompts]
            outs = self.raw_generate(prompts, params, *args, **kwargs)
            self.generated_tokens += sum(len(c.token_ids) for o in outs for c in o.outputs)
            return outs

        # --- reward, dynamic sampling, monitor ---------------------------------------------------------------

        def reward(self, prompts, completions, **kwargs):
            vs = grpo_logic.verdicts(kwargs["type"], completions, kwargs["gold_json"], self.graph)
            self.pending.append(vs)
            return grpo_logic.rewards(vs)

        def _scored(self, inputs):
            self.pending = []
            out = super()._generate_and_score_completions(inputs)
            if len(self.pending) != 1 or len(self.pending[0]) != len(inputs):
                raise SystemExit("the reward function was not called once over the whole batch")
            return out, self.pending[0]

        def _observe(self, inputs, out, vs, group_rows):
            lengths = out["completion_mask"].sum(dim=1).tolist()
            last = [int(out["completion_ids"][i, max(n - 1, 0)]) for i, n in enumerate(lengths)]
            for g in group_rows:
                row = inputs[g[0]]
                cap = row["max_completion_tokens"]
                self.monitor.add_group(row["type"], row["index"], [vs[i] for i in g],
                                       [lengths[i] >= cap and last[i] not in self.stop_ids for i in g],
                                       [int(lengths[i]) for i in g])

        def _generate_and_score_completions(self, inputs):
            size = self.num_generations
            out, vs = self._scored(inputs)
            gs = grpo_logic.groups([x["item_id"] for x in inputs], size)
            types = [inputs[g[0]]["type"] for g in gs]
            dense = grpo_logic.rewards(vs)
            replaced = grpo_logic.to_replace(types, [[dense[i] for i in g] for g in gs], self.dynamic_types)
            kept = [g for k, g in enumerate(gs) if k not in set(replaced)]
            self._observe(inputs, out, vs, kept)
            if replaced:
                new_inputs = []
                for k in replaced:
                    row = self.queue.next(types[k])
                    new_inputs += [self.rows_by_item[row["item_id"]]] * size
                new, new_vs = self._scored(new_inputs)
                self._observe(new_inputs, new, new_vs, grpo_logic.groups([x["item_id"] for x in new_inputs], size))
                self.spliced_keys |= set(splice(torch, out, new, replaced, size, self.processing_class.pad_token_id))
            self.write_log({"dynamic_sampling": {"step": self.state.global_step, "replaced_groups": len(replaced),
                                           "by_type": {t: sum(types[k] == t for k in replaced) for t in self.dynamic_types}}})
            return out

        # --- weight sync ------------------------------------------------------------------------------------

        def after_sync(self):
            """Runs after every weight push to vLLM; the canary check every MONITOR_EVERY steps."""
            step = self.state.global_step
            if step and step % pinned.MONITOR_EVERY == 0 and all(c["step"] != step for c in self.sync_checks):
                self.sync_check(step)

        def token_logprobs(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
            model = self.accelerator.unwrap_model(self.model)
            ids = torch.tensor([prompt_ids + tokens], device=self.accelerator.device)
            logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:len(prompt_ids) - 1 + len(tokens)].float()
            lp = torch.log_softmax(logits, dim=-1)
            return lp[torch.arange(len(tokens)), torch.tensor(tokens, device=lp.device)].tolist()

        def sync_check(self, step: int) -> dict:
            params = vllm.SamplingParams(**sampling_kwargs({**pinned.EVAL_SAMPLING, "max_tokens": pinned.SYNC_CANARY_TOKENS,
                                                            "logprobs": 0}, field_names)[0], stop_token_ids=self.stop_ids)
            outs = self.raw_generate([c["prompt"] for c in self.canary], [params] * len(self.canary))
            v, cur, start = [], [], []
            with torch.no_grad():
                for c, o in zip(self.canary, outs, strict=True):
                    toks = [t for t in o.outputs[0].token_ids if t not in self.stop_ids]
                    if not toks:
                        continue
                    v += [o.outputs[0].logprobs[i][t].logprob for i, t in enumerate(toks)]
                    cur += self.token_logprobs(c["prompt_ids"], toks)
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        start += self.token_logprobs(c["prompt_ids"], toks)
            gap_cur, gap_start = grpo_logic.mean_abs_gap(v, cur), grpo_logic.mean_abs_gap(v, start)
            drift = grpo_logic.mean_abs_gap(cur, start)
            ok, reason = grpo_logic.sync_verdict(gap_cur, gap_start, drift)
            record = {"step": step, "gap_current": gap_cur, "gap_start": gap_start, "drift": drift, "ok": ok,
                      "reason": reason, "tokens": len(v)}
            self.sync_checks.append(record)
            self.write_log({"sync_check": record})
            if ok is False:
                raise SystemExit(f"weight-sync check failed at step {step}: {reason}")
            return record

    return PinnedGRPOTrainer
