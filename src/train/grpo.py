"""TRL's GRPOTrainer with the step-5 and plan pins applied. Imported on the GPU machine only (by train.run).

- vLLM runs in one of two modes (pinned.GRPO_VLLM_MODE): "colocate", inside the training process on the same GPU, or
  "server", a `trl vllm-serve` process on a second GPU that the trainer reaches through TRL's VLLMClient.
- Per-type caps. Colocate: every rollout request reaches vLLM with its own SamplingParams, built whole from
  ROLLOUT_SAMPLING, the row's max_completion_tokens, both stop tokens and watermarking off. Server: TRL's client sends one
  max_tokens per call, so each batch is split into one call per cap and the replies are put back in order; the stop
  tokens go in generation_kwargs where the client takes them. Either way TRL's sampling values are checked against the
  pins first.
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

import inspect

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


def join_rows(torch, parts: list[tuple]) -> tuple:
    """Join per-chunk results of _get_per_token_logps_and_entropies, (log-probs, entropies, aux loss), along the row
    dimension. A position that is None in every chunk stays None; a per-batch scalar (the MoE aux loss) cannot be
    split by rows and stops the run."""
    out = []
    for values in zip(*parts, strict=True):
        if all(v is None for v in values):
            out.append(None)
        elif all(torch.is_tensor(v) and v.dim() >= 1 for v in values):
            out.append(torch.cat(values, dim=0))
        else:
            raise SystemExit("a log-prob scoring result is not per-row; it cannot be computed in row chunks")
    return tuple(out)


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
            self.logps_chunked_calls = 0
            if not callable(getattr(trl.GRPOTrainer, "_get_per_token_logps_and_entropies", None)):
                raise SystemExit("this TRL's GRPOTrainer has no _get_per_token_logps_and_entropies, which the row-chunked "
                                 "log-prob scoring overrides; methods with 'logp' in their name: "
                                 f"{grpo_logic.names_like(self, ('logp',))}")
            if not callable(getattr(trl.GRPOTrainer, "_generate_and_score_completions", None)):
                raise SystemExit("this TRL's GRPOTrainer has no _generate_and_score_completions (dynamic sampling and the "
                                 f"monitor hook it); generation-related methods: {grpo_logic.names_like(self, ('generat',))}")
            self.vllm_mode = getattr(self.args, "vllm_mode", "colocate")
            target = vllm.LLM if self.vllm_mode == "colocate" else grpo_logic.is_vllm_client
            found = grpo_logic.find_instances(self, target)
            if len(found) != 1:
                raise SystemExit(f"expected one {'vLLM engine' if self.vllm_mode == 'colocate' else 'VLLMClient'} on the "
                                 f"trainer ({self.vllm_mode} mode), found {[p for p, _ in found]}; vLLM-related attributes: "
                                 f"{grpo_logic.names_like(self, ('vllm', 'llm'))}")
            self.vllm_path, engine = found[0]
            if self.vllm_mode == "colocate":
                self.vllm_llm = engine
                self.raw_generate = engine.generate
                engine.generate = self.capped_generate
            else:
                self.vllm_client = engine
                self.raw_client_generate = engine.generate
                self.client_sig = inspect.signature(engine.generate)
                if "prompts" not in self.client_sig.parameters or "max_tokens" not in self.client_sig.parameters:
                    raise SystemExit(f"this TRL's VLLMClient.generate takes {list(self.client_sig.parameters)}; "
                                     "the per-type caps need its prompts and max_tokens")
                engine.generate = self.capped_client_generate
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

        def client_call(self, **values):
            """Call the server client with the arguments its signature has (and only those)."""
            params = self.client_sig.parameters
            out = {k: v for k, v in values.items() if k in params}
            var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            if var_kw:
                out.update({k: v for k, v in values.items() if k not in params})
            if "generation_kwargs" in params:
                out["generation_kwargs"] = {**(values.get("generation_kwargs") or {}), "stop_token_ids": list(self.stop_ids)}
            return self.raw_client_generate(**out)

        def capped_client_generate(self, *args, **kwargs):
            bound = self.client_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            values = {}
            for name, value in bound.arguments.items():
                if self.client_sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    values.update(value)
                else:
                    values[name] = value
            for k in ("temperature", "top_p", "repetition_penalty"):
                if k in values and values[k] != pinned.ROLLOUT_SAMPLING[k]:
                    raise SystemExit(f"TRL's rollout {k} is {values[k]}, pinned {pinned.ROLLOUT_SAMPLING[k]}")
            if values.get("guided_decoding_regex") is not None:
                raise SystemExit("TRL asked for guided decoding")
            prompts, n = list(values["prompts"]), values.get("n", 1)
            parts = []
            for cap, idx in grpo_logic.group_by_cap([self.caps(p) for p in prompts]):
                parts.append((idx, self.client_call(**{**values, "prompts": [prompts[i] for i in idx], "max_tokens": cap})))
            merged = grpo_logic.merge_parts(parts, len(prompts), n)
            self.generated_tokens += sum(len(c) for c in grpo_logic.completion_ids(merged))
            return merged

        def trial_generate(self, prompts: list[str]) -> list[int]:
            """One sampled call through the cap wrapper, asking for 512 tokens; the token counts returned."""
            if self.vllm_mode == "colocate":
                outs = self.vllm_llm.generate(prompts, vllm.SamplingParams(**{**pinned.ROLLOUT_SAMPLING, "n": 1, "max_tokens": 512}))
                counts = [len(o.outputs[0].token_ids) for o in outs]
            else:
                kw = {k: v for k, v in {**pinned.ROLLOUT_SAMPLING, "n": 1, "max_tokens": 512}.items()
                      if k in self.client_sig.parameters}
                counts = [len(c) for c in grpo_logic.completion_ids(self.vllm_client.generate(prompts=prompts, **kw))]
            self.generated_tokens = 0
            return counts

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

        # --- log-prob scoring, in row chunks ---------------------------------------------------------------

        def _get_per_token_logps_and_entropies(self, model, *args, batch_size=None, **kwargs):
            """TRL 1.14's Liger path ("_chunked_logps") drops batch_size and runs the backbone over every row it is
            given at once: before training, the whole generation batch (24 prompts x 8 completions, up to ~17k tokens
            each), which asked for 68 GiB in the pilot. Here the rows go in batches of batch_size (default the
            micro-batch) and the results are joined. Each row's computation is unchanged."""
            n = args[0].shape[0]
            size = batch_size or self.args.per_device_train_batch_size
            if n <= size:
                return super()._get_per_token_logps_and_entropies(model, *args, batch_size=batch_size, **kwargs)
            parts = []
            for lo, hi in grpo_logic.row_chunks(n, size):
                def cut(x):
                    return x[lo:hi] if torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] == n else x
                parts.append(super()._get_per_token_logps_and_entropies(
                    model, *[cut(a) for a in args], batch_size=batch_size, **{k: cut(v) for k, v in kwargs.items()}))
            self.logps_chunked_calls += 1
            return join_rows(torch, parts)

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

        def token_scores(self, prompt_ids: list[int], tokens: list[int]) -> tuple[list[float], list[int]]:
            """(log-prob of each token, the argmax at each position), teacher-forced under the trainer's weights."""
            model = self.accelerator.unwrap_model(self.model)
            ids = torch.tensor([prompt_ids + tokens], device=self.accelerator.device)
            logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:len(prompt_ids) - 1 + len(tokens)].float()
            lp = torch.log_softmax(logits, dim=-1)
            return lp[torch.arange(len(tokens)), torch.tensor(tokens, device=lp.device)].tolist(), lp.argmax(dim=-1).tolist()

        def canary_outputs(self) -> tuple[list[tuple[list[int], list[float] | None]], list | None]:
            """vLLM's greedy canary continuations: per prompt (tokens, their log-probs or None), stop tokens removed;
            and the prompt ids the server tokenised, where it says."""
            prompts = [c["prompt"] for c in self.canary]
            out, prompt_ids = [], None
            if self.vllm_mode == "colocate":
                params = vllm.SamplingParams(**sampling_kwargs({**pinned.EVAL_SAMPLING, "max_tokens": pinned.SYNC_CANARY_TOKENS,
                                                                "logprobs": 0}, field_names)[0], stop_token_ids=self.stop_ids)
                for o in self.raw_generate(prompts, [params] * len(prompts)):
                    c = o.outputs[0]
                    pairs = [(t, c.logprobs[i][t].logprob) for i, t in enumerate(c.token_ids) if t not in self.stop_ids]
                    out.append(([t for t, _ in pairs], [x for _, x in pairs]))
                return out, prompt_ids
            res = self.client_call(prompts=prompts, n=1, temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                                   max_tokens=pinned.SYNC_CANARY_TOKENS)
            comps = grpo_logic.completion_ids(res)
            lps = res.get("logprobs") if isinstance(res, dict) else None
            prompt_ids = res.get("prompt_ids") if isinstance(res, dict) else None
            for k, toks in enumerate(comps):
                lp = lps[k] if lps is not None and len(lps[k]) == len(toks) and all(isinstance(x, float) for x in lps[k]) else None
                keep = [i for i, t in enumerate(toks) if t not in self.stop_ids]
                out.append(([toks[i] for i in keep], None if lp is None else [lp[i] for i in keep]))
            return out, prompt_ids

        def canary_scores(self) -> dict:
            outs, prompt_ids = self.canary_outputs()
            v, toks_all, cur, start, am_cur, am_start = [], [], [], [], [], []
            with torch.no_grad():
                for c, (toks, lps) in zip(self.canary, outs, strict=True):
                    if not toks:
                        continue
                    lp_c, a_c = self.token_scores(c["prompt_ids"], toks)
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        lp_s, a_s = self.token_scores(c["prompt_ids"], toks)
                    toks_all += toks
                    cur, start, am_cur, am_start = cur + lp_c, start + lp_s, am_cur + a_c, am_start + a_s
                    v = None if v is None or lps is None else v + lps
            if not toks_all:
                raise SystemExit("vLLM returned no canary tokens")
            return {"vllm_logprobs": v, "current": cur, "start": start, "tokens": toks_all,
                    "agree_current": grpo_logic.agreement(toks_all, am_cur),
                    "agree_start": grpo_logic.agreement(toks_all, am_start), "server_prompt_ids": prompt_ids}

        def backbone_check(self) -> list[str]:
            """Before training: vLLM serves the pinned backbone (its greedy canary tokens are the backbone's argmax),
            and, where the server reports them, it tokenised the canary prompts as we do."""
            sc = self.canary_scores()
            problems = []
            if sc["agree_start"] < pinned.SERVER_BACKBONE_AGREE:
                problems.append(f"vLLM's greedy tokens are the backbone's argmax only {sc['agree_start']:.3f} of the time "
                                f"(< {pinned.SERVER_BACKBONE_AGREE}): it is not serving the pinned backbone")
            if sc["server_prompt_ids"] is not None and [list(x) for x in sc["server_prompt_ids"]] != [c["prompt_ids"] for c in self.canary]:
                problems.append("the vLLM server tokenised the canary prompts differently from the pinned tokenizer")
            self.write_log({"backbone_check": {"agree": sc["agree_start"], "tokens": len(sc["tokens"]),
                                               "server_prompt_ids_reported": sc["server_prompt_ids"] is not None,
                                               "problems": problems}})
            return problems

        def sync_check(self, step: int) -> dict:
            sc = self.canary_scores()
            drift = grpo_logic.mean_abs_gap(sc["current"], sc["start"])
            if sc["vllm_logprobs"] is not None:
                gap_cur = grpo_logic.mean_abs_gap(sc["vllm_logprobs"], sc["current"])
                gap_start = grpo_logic.mean_abs_gap(sc["vllm_logprobs"], sc["start"])
                ok, reason = grpo_logic.sync_verdict(gap_cur, gap_start, drift)
                method = "log-probs"
            else:
                gap_cur = gap_start = None
                ok, reason = grpo_logic.sync_verdict_argmax(sc["agree_current"], sc["agree_start"])
                method = "greedy choices"
            record = {"step": step, "method": method, "gap_current": gap_cur, "gap_start": gap_start, "drift": drift,
                      "agree_current": sc["agree_current"], "agree_start": sc["agree_start"], "ok": ok,
                      "reason": reason, "tokens": len(sc["tokens"])}
            self.sync_checks.append(record)
            self.write_log({"sync_check": record})
            if ok is False:
                raise SystemExit(f"weight-sync check failed at step {step}: {reason}")
            return record

    return PinnedGRPOTrainer
