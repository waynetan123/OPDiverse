"""Step 10, training side, on the step-9 fixture: the resolved configuration and its shared fields, checkpoint steps,
the run order, the frozen-file guard, the reference ids and their guards, the GRPO pieces (caps, dynamic-sampling
queue, groups, monitor, weight-sync decision), and the runner's pure checks. Nothing here needs a GPU."""

import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from converters import __main__ as conv_cli
from converters.files import ConverterFiles, gzip_bytes, read_rows
from etl import pinned, verifiers
from etl.build import jsonl_bytes
from evaluate import run_vllm
from probe.run_vllm import llm_kwargs
from test_converters import conv, run as conv_run  # noqa: F401  (conv is a fixture)
from test_frozen_model import data  # noqa: F401  (a fixture)
from train import config, data as tdata, grpo_logic, merge, run

END, EOT = pinned.COMPLETION_END, pinned.BASE_DOC_END


class Enc:
    """Character-level ids with the two end tokens as single ids 1 and 2."""
    ids = {END: 1, EOT: 2}

    @classmethod
    def encode(cls, texts):
        out = []
        for t in texts:
            for tok, i in cls.ids.items():
                t = t.replace(tok, chr(i))
            out.append([ord(c) for c in t])
        return out

    @classmethod
    def token_id(cls, tok):
        return cls.ids.get(tok)


@pytest.fixture()
def built(conv):  # noqa: F811
    assert conv_run(conv, "build") == 0
    return conv


def file_rows(paths, arm, cfg="m1", seed=0):
    return read_rows(tdata.training_file(paths, arm, cfg, seed)[0])


def prompt_tokens(rows):
    return {r["item_id"]: len(Enc.encode([r["prompt"]])[0]) for r in rows if "prompt" in r}


# --- Configuration ------------------------------------------------------------------------------


def test_checkpoint_steps_round_half_up():
    assert config.checkpoint_steps(399) == [100, 200, 299, 399]
    assert config.checkpoint_steps(1197) == [299, 599, 898, 1197]
    assert config.checkpoint_steps(4) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        config.checkpoint_steps(3)


def test_train_steps_wait_for_the_owner(monkeypatch):
    monkeypatch.setattr(pinned, "TRAIN_STEPS", None)
    with pytest.raises(SystemExit, match="TRAIN_STEPS"):
        config.train_steps()
    assert config.train_steps(20) == 20
    monkeypatch.setattr(pinned, "TRAIN_STEPS", 399)
    assert config.train_steps() == 399


def test_resolved_configs_share_every_parity_field():
    cfgs = [config.resolved(a, 5e-5, 399, {"kind": "merged"} if a == "sft_dpo" else None) for a in pinned.SWEEP_ARMS]
    config.check_shared(cfgs)
    assert {c["trainer"] for c in cfgs} == {"sft", "dpo", "grpo"}
    by = {c["arm"]: c for c in cfgs}
    assert by["sft_dpo"]["reads"] == "dpo" and by["base"]["loss"] == "all_tokens" and by["sft"]["loss"] == "completion_tokens"
    assert by["grpo"]["grpo"]["num_generations"] == 8 and by["grpo"]["rollout_sampling"] == pinned.ROLLOUT_SAMPLING
    assert by["dpo"]["dpo"] == {"beta": 0.1, "loss_type": "sigmoid"}
    assert by["sft"]["checkpoint_steps"] == [100, 200, 299, 399]
    assert config.config_sha(by["sft"]) == config.config_sha(config.resolved("sft", 5e-5, 399))
    assert config.config_sha(by["sft"]) != config.config_sha(config.resolved("sft", 2e-4, 399))
    with pytest.raises(SystemExit, match="shared fields"):
        config.check_shared([by["sft"], dict(by["dpo"], steps=400)])
    with pytest.raises(ValueError, match="LR_GRID"):
        config.resolved("sft", 3e-5, 399)
    with pytest.raises(ValueError, match="SFT->DPO"):
        config.resolved("sft_dpo", 5e-5, 399)
    with pytest.raises(ValueError, match="SFT->DPO"):
        config.resolved("sft", 5e-5, 399, {"kind": "merged"})
    with pytest.raises(ValueError, match="M1 only"):
        config.resolved("base", 5e-5, 399, config="m2-mcq")
    assert config.run_name("sft", 5e-5) == "sft_lr5e-05" and config.run_name("grpo", 2e-4) == "grpo_lr2e-04"


# --- Rows and order -------------------------------------------------------------------------------


def test_run_order_is_shared_by_every_question_arm(built):
    orders = {a: [tdata.order_key(r) for r in tdata.run_order(file_rows(built, a), 100, 0)]
              for a in pinned.QUESTION_ARMS}
    assert len({tuple(o) for o in orders.values()}) == 1
    rows = file_rows(built, "sft")
    order = tdata.run_order(rows, 3 * len(rows) + 5, 0)
    keys = [tdata.order_key(r) for r in order]
    n = len(rows)
    for p in range(3):                                         # every pass is a permutation of the file
        assert sorted(keys[p * n:(p + 1) * n]) == sorted(tdata.order_key(r) for r in rows)
    assert keys[:n] != keys[n:2 * n]                           # each pass reshuffled
    assert keys == [tdata.order_key(r) for r in tdata.run_order(list(reversed(rows)), 3 * n + 5, 0)]
    assert keys[:n] != [tdata.order_key(r) for r in tdata.run_order(rows, n, 1)]
    base = file_rows(built, "base")
    cycled = tdata.run_order(base, 5 * len(base) + 1, 0)
    counts = {c: sum(r["cve_id"] == c for r in cycled) for c in {r["cve_id"] for r in base}}
    assert set(counts.values()) <= {5, 6} and tdata.passes(len(base), 10) == 10 * 24 / len(base)


def test_training_file_must_be_the_frozen_one(built):
    path, sha = tdata.training_file(built, "dpo", "m2-cvss", 3)
    assert path.name == "dpo.jsonl.gz" and len(sha) == 64
    rows = read_rows(path)
    path.write_bytes(gzip_bytes(jsonl_bytes(rows[1:])))
    with pytest.raises(SystemExit, match="frozen"):
        tdata.training_file(built, "dpo", "m2-cvss", 3)


# --- Reference ids and their guards ----------------------------------------------------------------


def test_reference_ids_for_each_trainer(built):
    rows = file_rows(built, "sft")
    enc, bad = tdata.encode_rows("sft", rows, Enc.encode, Enc.token_id, prompt_tokens(rows))
    assert bad == [] and len(enc) == len(rows)
    e = enc[tdata.order_key(rows[0])]
    assert e.input_ids[-1] == 1 and e.loss_mask[-1] == 1 and e.loss_mask[0] == 0
    assert sum(e.loss_mask) == len(Enc.encode([rows[0]["completion"]])[0])
    drows = file_rows(built, "dpo")
    denc, bad = tdata.encode_rows("dpo", drows, Enc.encode, Enc.token_id, prompt_tokens(drows))
    d = denc[tdata.order_key(drows[0])]
    assert bad == [] and d.chosen_ids[-1] == d.rejected_ids[-1] == 1 and d.length == len(d.prompt_ids) + max(
        len(d.chosen_ids), len(d.rejected_ids))
    grows = file_rows(built, "grpo")
    genc, bad = tdata.encode_rows("grpo", grows, Enc.encode, Enc.token_id, prompt_tokens(grows))
    caps = {tdata.order_key(r): r["max_completion_tokens"] for r in grows}
    assert bad == [] and tdata.longest(genc, caps) == max(len(genc[k].prompt_ids) + c for k, c in caps.items())
    brows = file_rows(built, "base")
    benc, bad = tdata.encode_rows("sft", brows, Enc.encode, Enc.token_id, {})
    assert bad == [] and all(e.input_ids[-1] == 2 and all(e.loss_mask) for e in benc.values())


def test_reference_guards(built):
    rows = file_rows(built, "sft")
    counts = prompt_tokens(rows)
    doubled = [dict(rows[0], completion=rows[0]["completion"] + END)] + rows[1:]
    assert any("second end token" in b for b in tdata.encode_rows("sft", doubled, Enc.encode, Enc.token_id, counts)[1])
    bare = [dict(rows[0], completion=rows[0]["completion"][: -len(END)])] + rows[1:]
    assert any("does not end" in b for b in tdata.encode_rows("sft", bare, Enc.encode, Enc.token_id, counts)[1])
    off = dict(counts, **{rows[0]["item_id"]: counts[rows[0]["item_id"]] + 1})
    assert any("bank counted" in b for b in tdata.encode_rows("sft", rows, Enc.encode, Enc.token_id, off)[1])
    brows = file_rows(built, "base")
    wrong = [dict(brows[0], text=brows[0]["text"][: -len(EOT)] + END)] + brows[1:]
    bad = tdata.encode_rows("sft", wrong, Enc.encode, Enc.token_id, {})[1]
    assert any("does not end" in b for b in bad) and any("other end token" in b for b in bad)
    drows = file_rows(built, "dpo")
    same = [dict(drows[0], rejected=drows[0]["chosen"])] + drows[1:]
    assert any("identical" in b for b in tdata.encode_rows("dpo", same, Enc.encode, Enc.token_id, prompt_tokens(drows))[1])
    with pytest.raises(SystemExit, match="end tokens"):
        tdata.encode_rows("sft", rows, Enc.encode, lambda t: None, counts)


def test_longest_first(built):
    rows = file_rows(built, "base")
    enc, _ = tdata.encode_rows("sft", rows, Enc.encode, Enc.token_id, {})
    first = tdata.longest_first(tdata.run_order(rows, len(rows), 0), enc)
    sizes = [enc[tdata.order_key(r)].length for r in first]
    assert sizes == sorted(sizes, reverse=True) and len(first) == len(rows)


def test_compare_prepared():
    ref = [(5, 6, 1), (7, 1)]
    assert tdata.compare_prepared(ref, [[5, 6, 1], [7, 1]], "x") == []
    assert "token 2" in tdata.compare_prepared(ref, [[5, 6], [7, 1]], "x")[0]          # truncated
    assert "row 1" in tdata.compare_prepared(ref, [[5, 6, 1], [7, 1, 1]], "x")[0]      # a second end token
    assert "prepared 1 rows" in tdata.compare_prepared(ref, [[5, 6, 1]], "x")[0]


def test_dpo_prepared_columns_any_layout():
    ref = [tdata.Encoded(("a", "0"), prompt_ids=(7, 8), chosen_ids=(5, 1), rejected_ids=(6, 1))]
    old = {"prompt_input_ids": [[7, 8]], "chosen_input_ids": [[5, 1]], "rejected_input_ids": [[6, 1]], "prompt": ["x"]}
    assert tdata.dpo_prepared_problems(ref, list(old), old.get) == ([], "prompt_input_ids, chosen_input_ids, "
                                                                        "rejected_input_ids (separate)")
    new = {"prompt_ids": [[7, 8]], "chosen_ids": [[7, 8, 5, 1]], "rejected_ids": [[7, 8, 6, 1]]}
    assert tdata.dpo_prepared_problems(ref, list(new), new.get)[1] == "prompt_ids, chosen_ids, rejected_ids (after the prompt)"
    doubled = dict(new, chosen_ids=[[7, 8, 5, 1, 1]])
    bad, layout = tdata.dpo_prepared_problems(ref, list(doubled), doubled.get)
    assert layout is None and any("chosen" in b for b in bad)
    bad, layout = tdata.dpo_prepared_problems(ref, ["prompt", "chosen"], {}.get)
    assert layout is None and "['chosen', 'prompt']" in bad[0]


def test_grpo_dataset_rows(built):
    rows = file_rows(built, "grpo")
    ds = tdata.grpo_dataset_rows(rows)
    assert json.loads(ds[0]["gold_json"]) == rows[0]["gold"] and ds[0]["prompt"] == rows[0]["prompt"]
    assert {r["type"] for r in ds if r["dynamic_sampling"]} == {"mcq", "exact_id"}


# --- GRPO pieces ---------------------------------------------------------------------------------------


def test_caps_and_sampling():
    caps = grpo_logic.CapLookup({"p1": 16, "p2": 512}, {(1, 2): 16})
    assert caps("p1") == 16 and caps({"prompt": "p2"}) == 512 and caps({"prompt_token_ids": [1, 2]}) == 16 and caps([1, 2]) == 16
    with pytest.raises(SystemExit, match="cap"):
        caps("canary")
    f = grpo_logic.sampling_fields(24, 8, 0)
    assert f == {**pinned.ROLLOUT_SAMPLING, "n": 8, "max_tokens": 24, "logprobs": 0}


def test_reward_is_the_dense_verdict(graph):
    gold = {"cwe": "CWE-787"}
    vs = grpo_logic.verdicts(["exact_id"] * 3, ["CWE-787", "so CWE-119", "no idea"], [json.dumps(gold)] * 3, graph)
    assert grpo_logic.rewards(vs) == [1.0, float(pinned.hierarchy_score("CWE-119", "CWE-787", graph)), 0.0]


@pytest.fixture(scope="module")
def graph():
    from etl.cwe_graph import load_cwe_graph
    from etl.paths import DEFAULT
    return load_cwe_graph(DEFAULT.cwe_xml)


def test_type_queue_wraps_and_counts():
    order = [{"item_id": f"c{i}:{t}:0", "type": t} for i in range(3) for t in ("mcq", "cvss")]
    order += order[:2]                                                  # a second pass repeats items
    q = grpo_logic.TypeQueue.of(order, {"mcq"})
    assert set(q.rows) == {"mcq"} and [r["item_id"] for r in q.rows["mcq"]] == ["c0:mcq:0", "c1:mcq:0", "c2:mcq:0"]
    got = [q.next("mcq")["item_id"] for _ in range(4)]
    assert got == ["c0:mcq:0", "c1:mcq:0", "c2:mcq:0", "c0:mcq:0"] and q.replacements["c0:mcq:0"] == 2


def test_groups_and_ties():
    ids = ["a"] * 3 + ["b"] * 3
    assert grpo_logic.groups(ids, 3) == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(SystemExit):
        grpo_logic.groups(ids[:5], 3)
    with pytest.raises(SystemExit, match="mixes"):
        grpo_logic.groups(["a", "b", "a", "b", "a", "b"], 3)
    rewards = [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert grpo_logic.to_replace(["mcq", "mcq", "exact_id", "cvss"], rewards, {"mcq", "exact_id"}) == [0, 2]


def verdict(t, reply, gold, graph):
    return verifiers.verify_item(t, reply, gold, graph)


def test_monitor_summary_and_events(graph):
    m = grpo_logic.Monitor()
    g = {"vulnerable": True, "cwe": "CWE-787"}
    m.add_group("find_error", 0, [verdict("find_error", "VULNERABLE: yes, CWE-787", g, graph)] * 4, [False] * 4, [9] * 4)
    m.add_group("find_error", 1, [verdict("find_error", "VULNERABLE: yes, CWE-787", dict(g, vulnerable=False), graph)] * 4,
                [False] * 4, [9] * 4)
    lg = {"lines": [3], "n_lines": 6}
    m.add_group("line_loc", 0, [verdict("line_loc", r, lg, graph) for r in ("LINES: 3", "LINES: 1, 2", "LINES: none", "x")],
                [False, False, False, True], [5, 6, 7, 512])
    s = m.summary(10)
    fe, ll = s["types"]["find_error"], s["types"]["line_loc"]
    assert fe["predicted_vulnerable"] == 1.0 and fe["predicted_vulnerable_patched_items"] == 1.0 and fe["live_dense"] == 0
    assert {e["event"] for e in s["events"] if e["type"] == "find_error"} == {"collapse", "below_floor"}
    assert ll["live_dense"] == 1.0 and ll["parse_rate"] == 0.75 and ll["cut_off_rate"] == 0.25
    assert ll["set_size"] == {"0": 1, "1": 1, "2": 1} and ll["mean_tokens"] == (5 + 6 + 7 + 512) / 4
    m.reset()
    assert m.summary(20) == {"step": 20, "types": {}, "events": []}


def test_sync_verdict():
    assert grpo_logic.sync_verdict(0.01, 0.02, 0.01)[0] is None                       # too early to tell
    assert grpo_logic.sync_verdict(0.02, 0.30, 0.30)[0] is True
    ok, reason = grpo_logic.sync_verdict(0.30, 0.02, 0.30)                             # vLLM still serves the start
    assert ok is False and "not coming from the current weights" in reason
    assert grpo_logic.mean_abs_gap([0.0, -1.0], [0.5, -0.5]) == 0.5


def test_splice_puts_replacements_in_place():
    torch = pytest.importorskip("torch")
    from train.grpo import splice
    out = {"prompt_ids": torch.tensor([[0, 5], [0, 6], [7, 8], [7, 9]]), "completion_ids": torch.tensor([[1], [2], [3], [4]]),
           "completion_mask": torch.ones(4, 1, dtype=torch.long), "advantages": torch.zeros(4), "num_items_in_batch": 4}
    new = {"prompt_ids": torch.tensor([[1, 2, 3], [1, 2, 4]]), "completion_ids": torch.tensor([[5, 6], [7, 8]]),
           "completion_mask": torch.ones(2, 2, dtype=torch.long), "advantages": torch.tensor([1.0, -1.0]), "num_items_in_batch": 4}
    assert splice(torch, out, new, [0], 2, pad_id=0) == []
    assert out["prompt_ids"].tolist() == [[1, 2, 3], [1, 2, 4], [0, 7, 8], [0, 7, 9]]       # left-padded prompts
    assert out["completion_ids"].tolist() == [[5, 6], [7, 8], [3, 0], [4, 0]]              # right-padded completions
    assert out["advantages"].tolist() == [1.0, -1.0, 0.0, 0.0] and out["num_items_in_batch"] == 6


# --- Runner checks -----------------------------------------------------------------------------------------


def test_label_checks():
    ids, mask = [10, 11, 12, 1], [0, 0, 1, 1]
    exp = run.expected_labels(ids, mask)
    assert exp == [-100, -100, 12, 1]
    two = exp + run.expected_labels([20, 21, 2], [1, 1, 1])
    pos = [0, 1, 2, 3, 0, 1, 2]
    assert run.label_problems(two, pos, two) == []
    assert run.label_problems(two[:4] + [-100] + two[5:], pos, two) == []          # a sequence's first token
    assert "positions" in run.label_problems(two[:3] + [-100] + two[4:], pos, two)[0]  # the end token unlabelled
    assert "padded or truncated" in run.label_problems(two + [-100], pos + [4], two)[0]
    assert run.without_end("ANSWER: B" + END) == "ANSWER: B"
    with pytest.raises(ValueError):
        run.without_end("ANSWER: B")


def test_optional_truncation_fields():
    import dataclasses

    @dataclasses.dataclass
    class Old:
        max_prompt_length: int | None = 512
        beta: float = 0.1

    @dataclasses.dataclass
    class New:
        beta: float = 0.1

    assert run.optional_fields(Old, run.NO_TRUNCATION) == ({"max_prompt_length": None}, ["max_completion_length"])
    assert run.optional_fields(New, {"max_prompt_length": None}) == ({}, ["max_prompt_length"])


def test_readiness_and_versions(monkeypatch):
    for name in ("TRAIN_STEPS", "GRADIENT_CHECKPOINTING", "TRAIN_LIBS", "TRL_DEFAULTS"):
        monkeypatch.setattr(pinned, name, None)
    assert run.readiness(pilot=True) == [] and len(run.readiness(pilot=False)) == 4
    assert run.version_drift({"trl": "1.0"}, None) == []
    assert run.version_drift({"trl": "1.0", "peft": "2"}, {"trl": "1.1", "peft": "2"}) == ["trl: installed 1.0, pinned 1.1"]
    assert set(run.installed_versions(("pytest", "no_such_lib"))) == {"pytest", "no_such_lib"}
    assert run.installed_versions(("no_such_lib",))["no_such_lib"] is None


def test_token_accounting():
    e = tdata.Encoded(("a", "0"), (1, 2, 3, 4), (0, 0, 1, 1))
    assert run.sft_tokens([e.key, e.key], {e.key: e}) == {"forward": 8, "loss": 4}
    d = tdata.Encoded(("b", "0"), prompt_ids=(1, 2), chosen_ids=(3,), rejected_ids=(4, 5))
    t = run.dpo_tokens([d.key], {d.key: d})
    assert t == {"forward": 7, "reference_forward": 7, "loss": 3}
    assert run.flops(10, t) == 6 * 10 * 7 + 2 * 10 * 7
    assert run._accum(192, 8) == 24
    with pytest.raises(SystemExit):
        run._accum(24, 5)


def test_micro_batch_is_pinned_outside_grpo():
    assert run.micro_batch("sft", None) == pinned.TRAIN_MICRO_BATCH == run.micro_batch("dpo", 1)
    assert run.micro_batch("grpo", None) == 1 and run.micro_batch("grpo", 8) == 8
    with pytest.raises(SystemExit, match="pinned"):
        run.micro_batch("sft_dpo", 4)
    assert pinned.TRAIN_ATTENTION == "sdpa" and "flash_attn" not in run.LIBS


def test_run_cli_guards(monkeypatch):
    with pytest.raises(SystemExit, match="pilot runs only"):
        run.main(["--arm", "sft", "--lr", "5e-5", "--max-steps", "4"])
    with pytest.raises(SystemExit, match="pilot runs only"):
        run.main(["--arm", "sft", "--lr", "5e-5", "--longest-first"])
    with pytest.raises(SystemExit, match="gradient-checkpointing"):
        run.main(["--arm", "sft", "--lr", "5e-5", "--pilot", "--max-steps", "4"])


# --- Merge and evaluation -------------------------------------------------------------------------------------


def test_adapter_identity(tmp_path):
    ckpt = tmp_path / "run" / "trainer" / "checkpoint-4"
    ckpt.mkdir(parents=True)
    with pytest.raises(SystemExit, match="not a LoRA checkpoint"):
        merge.adapter_sha256(ckpt)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"w")
    sha = merge.adapter_sha256(ckpt)
    (ckpt / "optimizer.pt").write_bytes(b"state")                       # optimizer state is not the adapter
    assert merge.adapter_sha256(ckpt) == sha
    assert merge.run_init(ckpt) is None
    (tmp_path / "run" / "run_meta.json").write_text(json.dumps({"config": {"init": {"kind": "merged", "path": "/m"}}}))
    assert merge.run_init(ckpt) == "/m"


def test_eval_runner_serves_a_merged_checkpoint(tmp_path):
    kw, _ = llm_kwargs({"generation_config"})
    assert list(kw)[:3] == ["model", "revision", "tokenizer_revision"] and kw["model"] == pinned.TOKENIZER_REPO
    kw, _ = llm_kwargs({"generation_config"}, model="/ckpt")
    assert kw["model"] == "/ckpt" and kw["tokenizer"] == pinned.TOKENIZER_REPO and "revision" not in kw
    assert kw["tokenizer_revision"] == pinned.TOKENIZER_REVISION and kw["generation_config"] == "vllm"
    req = tmp_path / "dev_requests.jsonl"
    assert [p.name for p in run_vllm.output_paths(req)] == ["dev_generations.jsonl", "dev_generations.partial.jsonl",
                                                            "dev_run_meta.json"]
    assert all(p.parent == tmp_path / "o" for p in run_vllm.output_paths(req, tmp_path / "o"))


def test_shared_fields_cover_the_plan_parity_pins():
    assert {"steps", "examples_per_step", "lora", "optimizer", "checkpoint_steps"} <= set(config.SHARED)
    assert Fraction(9576, pinned.TRAIN_EXAMPLES_PER_STEP) == 399


def test_find_trl_internals_wherever_they_live():
    class LLM:
        def generate(self): ...

    class Generation:          # newer TRL: the engine sits on a helper object, which owns the sync
        def __init__(self):
            self.llm = LLM()

        def sync_weights(self): ...

    class OldTrainer:           # older TRL: trainer.llm, trainer._move_model_to_vllm
        def __init__(self):
            self.llm = LLM()
            self.args = SimpleNamespace(x=1)

        def _move_model_to_vllm(self): ...

    class NewTrainer:
        def __init__(self):
            self.vllm_generation = Generation()

    old, new = OldTrainer(), NewTrainer()
    assert [p for p, _ in grpo_logic.find_instances(old, LLM)] == ["llm"]
    (path, llm), = grpo_logic.find_instances(new, LLM)
    assert path == "vllm_generation.llm" and llm is new.vllm_generation.llm
    holder = grpo_logic.parent_path(path)
    assert holder == "vllm_generation" and grpo_logic.resolve(new, holder) is new.vllm_generation
    owners = [("trainer", new), (holder, grpo_logic.resolve(new, holder))]
    assert grpo_logic.find_method(owners, grpo_logic.SYNC_METHODS)[::2] == ("vllm_generation", "sync_weights")
    assert grpo_logic.find_method([("trainer", old)], grpo_logic.SYNC_METHODS)[::2] == ("trainer", "_move_model_to_vllm")
    assert grpo_logic.find_method([("trainer", NewTrainer())], grpo_logic.SYNC_METHODS) is None
    assert "sync_weights" in grpo_logic.names_like(Generation(), ("sync",))
    shared = LLM()                                      # one engine reachable two ways is one engine
    both = SimpleNamespace(a=shared, b=SimpleNamespace(c=shared))
    assert len(grpo_logic.find_instances(both, LLM)) == 1
