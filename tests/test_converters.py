"""Step 9 on a small fixture: the mask and upsampling arithmetic, every arm's rows, the base documents, the
freeze, the upstream guards, each `check` invariant (by corrupting a file), the report, and determinism."""

import gzip
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from converters import __main__ as cli
from converters import build, check, masks
from converters.files import ConverterFiles, gzip_bytes, read_rows
from etl import pinned, verifiers
from etl.build import jsonl_bytes, read_jsonl, write_json
from etl.cwe_graph import load_cwe_graph
from etl.manifest import file_sha256
from etl.paths import DEFAULT
from frozen_model import rationale
from frozen_model.files import SessionFiles
from generators.bank.build import load_inputs as bank_inputs
from generators.bank.check import code_lines
from generators.bank.files import BankFiles
from generators.teacher import dpo
from generators.teacher.files import TeacherFiles
from test_frozen_model import data  # noqa: F401  (a fixture: the dry bank, 8 non-test CVEs)

ROOT = Path(__file__).resolve().parents[1]
GRAPH = load_cwe_graph(DEFAULT.cwe_xml)
END = pinned.COMPLETION_END
# Five seeds over a 4-CVE late window, each a different pair (as step 8's draw guarantees).
DEV = [(4, 5), (6, 7), (4, 7), (5, 6), (4, 6)]


def run(paths, *args):
    return cli.main([*args, "--data-dir", str(paths.data), "--tokenizer", str(paths.data / "tokenizer.json"),
                     "--unpinned-tokenizer"])


def gen(value):
    return {"result_type": "succeeded", "stop_reason": "end_turn", "stop_category": None, "text": json.dumps(value)}


def write_upstream(paths, gold_only=("line_loc",)):
    """distill_self.jsonl, dpo.jsonl, their reports and substitution.json, built with the real step-5 and step-6
    decision functions; and a hand-drawn partition.jsonl for seeds 0-4."""
    bank = read_jsonl(BankFiles.of(paths).bank("nontest"))
    bank_sha = file_sha256(BankFiles.of(paths).bank("nontest"))
    session, teacher = SessionFiles.of(paths), TeacherFiles.of(paths)

    distill = []
    for item in bank:
        good = {"text": f"Reasoning about {item['type']}.\n{item['target']}", "finish_reason": "stop"}
        cut = {"text": "Thinking", "finish_reason": "length"}
        distill.append(rationale.decide(item, cut if item["type"] in gold_only else good,
                                        cut if item["type"] in gold_only else None, GRAPH))
    session.dir.mkdir(parents=True, exist_ok=True)
    session.distill_self.write_bytes(jsonl_bytes(distill))
    write_json(session.substitution, {"substituted_types": []})
    write_json(session.report_json, {
        "substituted_types": [],
        "grpo": {t: {"cap": cap, "band": "dynamic_sampling" if t in ("mcq", "exact_id") else "as_specified"}
                 for t, cap in zip(pinned.BANK_TYPES, (16, 24, 256, 512, 512))},
        "sources": {"bank_nontest_sha256": bank_sha, "distill_self_sha256": file_sha256(session.distill_self)}})

    facts, _, graph = bank_inputs(paths)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    decisions = {d["cve_id"]: d for d in read_jsonl(BankFiles.of(paths).mcq_decisions)}
    pairs = []
    for item in bank:
        if item["type"] == "mcq":
            pairs.append(dpo.decide(item, None, None, code[item["cve_id"]], graph, decisions[item["cve_id"]]))
        else:
            value = dpo.rule_value(item, code[item["cve_id"]], graph)
            pairs.append(dpo.decide(item, gen({pinned.DPO_FIELDS[item["type"]]: value}), None, code[item["cve_id"]], graph))
    teacher.dir.mkdir(parents=True, exist_ok=True)
    teacher.dpo.write_bytes(jsonl_bytes(pairs))
    write_json(teacher.report_json, {"sources": {"bank_nontest_sha256": bank_sha, "dpo_sha256": file_sha256(teacher.dpo)}})

    split = read_jsonl(paths.split)
    nontest = [f for f in sorted(facts, key=lambda f: (f["published"], f["cve_id"]))
               if next(s for s in split if s["cve_id"] == f["cve_id"])["pool"] == "nontest"]
    rows = []
    for k, f in enumerate(nontest):
        role = {str(s): "dev" if k in DEV[s] else "train" for s in pinned.PARTITION_SEEDS}
        rows.append({"cve_id": f["cve_id"], "cluster_id": f["cve_id"], "late_window": k >= 4,
                     "moved_out_of_window": False, "role": role,
                     "checkpoint_seeds": [s for s in pinned.PARTITION_SEEDS if k == DEV[s][0]]})
    paths.partition.write_bytes(jsonl_bytes(rows))
    write_json(paths.partition_json, {"facts_sha256": file_sha256(paths.facts), "split_sha256": file_sha256(paths.split),
                                      "seeds": list(pinned.PARTITION_SEEDS)})
    return bank


@pytest.fixture()
def conv(data):  # noqa: F811
    write_upstream(data)
    return data


def built(conv):
    assert run(conv, "build") == 0
    return ConverterFiles.of(conv)


def rewrite(path, edit):
    """Apply edit(rows) -> rows to one training file, keeping its gzip framing."""
    rows = edit(read_rows(path))
    path.write_bytes(gzip_bytes(jsonl_bytes(rows)))


# --- Masks and upsampling ---------------------------------------------------------


def items(n_cves):
    out = []
    for c in range(n_cves):
        for t, idx in (("mcq", 0), ("exact_id", 0), ("cvss", 0), ("find_error", 0), ("find_error", 1), ("line_loc", 0)):
            out.append((f"CVE-2019-{c}:{t}:{idx}", t))
    return out


def test_configs_and_comparators():
    assert masks.CONFIGS == ("m1", "m2-mcq", "m2-exact_id", "m2-cvss", "m2-find_error", "m2-line_loc", "m1v-1of6", "m1v-1of3")
    assert {c: masks.comparator(c) for c in masks.M2} == {
        "m2-mcq": "m1v-1of6", "m2-exact_id": "m1v-1of6", "m2-cvss": "m1v-1of6",
        "m2-find_error": "m1v-1of3", "m2-line_loc": "m1v-1of6"}
    assert masks.dev_types("m2-cvss") == ["mcq", "exact_id", "find_error", "line_loc"]
    assert masks.dev_types("m1v-1of3") == list(pinned.BANK_TYPES) == masks.dev_types("m1")
    assert masks.arms("m1")[0] == "base" and "base" not in masks.arms("m2-mcq")


@pytest.mark.parametrize("config, removed", [
    ("m1", 0), ("m2-mcq", 1596), ("m2-find_error", 3192), ("m1v-1of6", 1596), ("m1v-1of3", 3192)])
def test_selection_at_the_real_size(config, removed):
    its = items(1596)  # one seed's train set: 9,576 items
    rows = masks.selection(its, config, seed=0)
    count = Counter(i for i, _ in rows)
    assert len(rows) == 9576 and len(count) == 9576 - removed
    assert sum(k == 2 for k in count.values()) == removed and set(count.values()) <= {1, 2}
    assert Counter(c for _, c in rows) == Counter({0: 9576 - removed, 1: removed}) - Counter()
    if config.startswith("m2-"):
        assert config[3:] not in masks.composition(rows)
    assert masks.selection(its, config, seed=0) == rows               # deterministic
    if config != "m1":
        assert masks.selection(its, config, seed=1) != rows           # redrawn per seed
    assert rows != sorted(rows)                                        # shuffled, not bank order


def test_upsample_remainder():
    kept = [f"x{i}" for i in range(7)]
    n = masks.copies(kept, 30, "m2-mcq", 0)   # 30 = 4 x 7 + 2
    assert sorted(n.values()) == [4] * 5 + [5] * 2 and sum(n.values()) == 30
    with pytest.raises(ValueError, match="whole number"):
        masks.masked(items(1)[:5], "m1v-1of6", 0)


def test_m1v_mask_is_uniform_over_items():
    its = items(1596)
    gone = masks.masked(its, "m1v-1of6", 0)
    by_type = Counter(i.split(":")[1] for i in gone)
    assert len(gone) == 1596 and set(by_type) == set(pinned.BANK_TYPES)
    assert by_type["find_error"] > by_type["mcq"]   # two items per CVE, so about twice as many masked


# --- Build -------------------------------------------------------------------------


def test_build_writes_every_file(conv):
    files = built(conv)
    bank = {r["item_id"]: r for r in read_jsonl(BankFiles.of(conv).bank("nontest"))}
    manifest = json.loads(files.manifest.read_text())
    meta = json.loads(files.meta.read_text())
    assert len(meta["outputs"]) == 5 * (5 + 7 * 4) == len(build.specs())
    assert manifest["seeds"]["0"]["train_cves"] == 6 and manifest["seeds"]["0"]["train_items"] == 36
    assert manifest["seeds"]["0"]["files"]["m2-cvss"]["sft_dpo"] == "seed0/m2-cvss/dpo.jsonl.gz"
    assert manifest["configs"]["m2-find_error"]["m1v_comparator"] == "m1v-1of3"
    nontest = [r["cve_id"] for r in read_jsonl(conv.partition)]
    assert manifest["seeds"]["1"]["checkpoint_cves"] == [nontest[DEV[1][0]]] and manifest["seeds"]["1"]["dev_cves"] == 2

    sft = read_rows(files.training(0, "m1", "sft"))
    assert len(sft) == 36 and all(r["prompt"] == bank[r["item_id"]]["prompt"] for r in sft)
    assert all(r["completion"] == bank[r["item_id"]]["target"] + END for r in sft)
    distill = read_rows(files.training(0, "m1", "distill_self"))
    ll = [r for r in distill if r["type"] == "line_loc"]
    assert {r["source"] for r in ll} == {"gold_only"} and all(r["completion"] == bank[r["item_id"]]["target"] + END for r in ll)
    cv = next(r for r in distill if r["type"] == "cvss")
    assert cv["completion"] == f"Reasoning about cvss.\n\n{bank[cv['item_id']]['target']}{END}"
    for r in read_rows(files.training(0, "m1", "dpo")):
        item = bank[r["item_id"]]
        v = verifiers.verify_item(item["type"], r["rejected"][: -len(END)], item["gold"], GRAPH)
        assert r["chosen"] == item["target"] + END and v.strict_ok and v.dense < 1
    grpo = read_rows(files.training(0, "m1", "grpo"))
    assert {(r["type"], r["max_completion_tokens"], r["dynamic_sampling"]) for r in grpo} == {
        ("mcq", 16, True), ("exact_id", 24, True), ("cvss", 256, False), ("find_error", 512, False), ("line_loc", 512, False)}
    assert all(r["gold"] == bank[r["item_id"]]["gold"] for r in grpo)

    keys = {a: [(r["item_id"], r["copy"]) for r in read_rows(files.training(2, "m2-find_error", a))]
            for a in pinned.QUESTION_ARMS}
    assert len({tuple(k) for k in keys.values()}) == 1        # one selection for every arm
    assert len(keys["sft"]) == 36 and sum(c for _, c in keys["sft"]) == 12


def test_base_documents(conv):
    files = built(conv)
    facts = {f["cve_id"]: f for f in read_jsonl(conv.facts)}
    docs = read_rows(files.training(0, "m1", "base"))
    assert len(docs) == 6 and not files.training(0, "m2-mcq", "base").exists()
    for d in docs:
        f, text = facts[d["cve_id"]], d["text"]
        assert text.endswith(pinned.BASE_DOC_END) and text.count(pinned.BASE_DOC_END) == 1
        assert f"Weakness: {f['cwe']}: {GRAPH.weaknesses[f['cwe'][4:]].name}\n" in text
        assert f"CVSS v3 base vector: {f['cvss_vector']}\n" in text and "End your reply" not in text
        assert d["cve_id"] not in text and "<|im_start|>" not in text
    redacted = [d["text"] for d in docs if pinned.CVE_REDACTED in d["text"]]
    assert redacted  # fixture facts 0 and 1 cite their own CVE ID; both are train for seed 0


def test_rebuild_is_identical_and_frozen(conv, monkeypatch):
    files = built(conv)
    before = files.meta.read_bytes()
    assert run(conv, "build") == 0 and files.meta.read_bytes() == before
    files.training(1, "m1", "sft").unlink()                     # a missing file is rewritten identically
    assert run(conv, "build") == 0 and files.meta.read_bytes() == before
    monkeypatch.setitem(pinned.CONVERTER_SALTS, "order", "changed")
    with pytest.raises(SystemExit, match="frozen"):
        run(conv, "build")


def test_upstream_guards(conv):
    teacher = TeacherFiles.of(conv)
    rows = read_jsonl(teacher.dpo)
    rows[0]["rejected"] = "ANSWER: D"
    teacher.dpo.write_bytes(jsonl_bytes(rows))
    with pytest.raises(SystemExit, match="not the file step 6 reported"):
        run(conv, "build")


def test_substitution_refused(conv):
    session = SessionFiles.of(conv)
    write_json(session.substitution, {"substituted_types": ["line_loc"]})
    with pytest.raises(SystemExit, match="substituted"):
        run(conv, "build")


def test_partition_guard(conv):
    rows = read_jsonl(conv.partition)
    for r in rows:
        del r["role"]["4"]
    conv.partition.write_bytes(jsonl_bytes(rows))
    with pytest.raises(SystemExit, match="seeds"):
        run(conv, "build")


def test_determinism_across_hash_seeds(conv):
    files = built(conv)
    first = files.meta.read_bytes()
    shutil.rmtree(files.dir)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "123"}
    subprocess.run([sys.executable, "-m", "converters", "build", "--data-dir", str(conv.data)],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    assert files.meta.read_bytes() == first
    assert gzip.decompress(files.training(3, "m1v-1of3", "dpo").read_bytes())  # readable gzip


# --- Check -------------------------------------------------------------------------


def test_check_passes(conv):
    built(conv)
    assert check.check(conv) == []
    assert run(conv, "check") == 0


def _violations(conv, seed, config, arm, edit):
    files = ConverterFiles.of(conv)
    rewrite(files.training(seed, config, arm), edit)
    return check.check(conv)


def test_check_catches_a_wrong_completion(conv):
    built(conv)

    def edit(rows):
        rows[0]["completion"] = "ANSWER: Z" + END
        return rows
    bad = _violations(conv, 0, "m1", "sft", edit)
    assert any("completion is not the bank target" in b for b in bad)
    assert any("content differs from converters_meta.json" in b for b in bad)


def test_check_catches_a_missing_row_and_arm_mismatch(conv):
    built(conv)
    bad = _violations(conv, 1, "m1", "grpo", lambda rows: rows[1:])
    assert any("grpo rows differ" in b for b in bad)


def test_check_catches_a_dev_item(conv):
    built(conv)
    bank = read_jsonl(BankFiles.of(conv).bank("nontest"))
    dev_cve = next(r["cve_id"] for r in read_jsonl(conv.partition) if r["role"]["0"] == "dev")
    dev_item = next(i for i in bank if i["cve_id"] == dev_cve and i["type"] == "mcq")
    for arm in pinned.QUESTION_ARMS:
        def edit(rows, arm=arm):
            src = {"sft": lambda: {"completion": dev_item["target"] + END},
                   "distill_self": lambda: {"completion": dev_item["target"] + END, "source": "gold_only"},
                   "dpo": lambda: {}, "grpo": lambda: {}}[arm]()
            rows[0] = {**rows[0], "item_id": dev_item["item_id"], "cve_id": dev_cve, "prompt": dev_item["prompt"], **src}
            return rows
        rewrite(ConverterFiles.of(conv).training(0, "m2-cvss", arm), edit)
    bad = check.check(conv)
    assert any("outside the seed's train set" in b for b in bad)


def test_check_catches_a_leaked_type_and_wrong_mask(conv):
    built(conv)
    files = ConverterFiles.of(conv)
    m1 = read_rows(files.training(0, "m1", "sft"))
    for arm in pinned.QUESTION_ARMS:   # make m2-mcq a copy of m1: the dropped type is back, nothing duplicated
        rewrite(files.training(0, "m2-mcq", arm), lambda rows, arm=arm: read_rows(files.training(0, "m1", arm)))
    bad = check.check(conv)
    assert any("seed0/m2-mcq: kept items are not the train items without mcq" in b for b in bad)
    assert len(m1) == 36


def test_check_catches_a_base_document_with_its_cve_id(conv):
    built(conv)

    def edit(rows):
        rows[0]["text"] = rows[0]["cve_id"] + " " + rows[0]["text"]
        return rows
    bad = _violations(conv, 0, "m1", "base", edit)
    assert any("contains its own CVE ID" in b for b in bad)


def test_check_catches_a_double_end_token(conv):
    built(conv)

    def edit(rows):
        rows[0]["chosen"] += END
        return rows
    bad = _violations(conv, 0, "m1", "dpo", edit)
    assert any("chosen is not the bank target" in b for b in bad)


def test_check_without_build(conv):
    assert check.check(conv) == ["converters_meta.json is missing: run `python -m converters build`"]


# --- Report ------------------------------------------------------------------------


def test_report(conv):
    files = built(conv)
    assert run(conv, "report") == 0
    rep = json.loads(files.report_json.read_text())
    m1 = rep["files"]["0"]["m1"]
    assert m1["sft"]["rows"] == 36 and m1["sft"]["duplicated_items"] == 0
    assert rep["files"]["0"]["m2-find_error"]["dpo"]["duplicated_items"] == 12
    assert m1["distill_self"]["gold_only_share"]["line_loc"] == "1" and m1["distill_self"]["gold_only_share"]["cvss"] == "0"
    assert m1["dpo"]["rule_share"]["mcq"] == "1"
    assert set(rep["max_sequence_tokens"]) == set(pinned.CONVERTER_ARMS)
    assert rep["m1v_masks"]["m1v-1of3"]["masked_by_type"]["0"] and len(rep["m1v_masks"]["m1v-1of6"]["overlap_jaccard"]) == 10
    md = files.report_md.read_text()
    assert md.startswith("# Step 9: converters") and "## M1-volume masks" in md


# --- The real files ------------------------------------------------------------------


def test_real_files():
    files = ConverterFiles.of(DEFAULT)
    if not files.meta.exists():
        pytest.skip("data/converters not built")
    meta = json.loads(files.meta.read_text())
    assert len(meta["outputs"]) == len(build.specs()) == 165
    assert all(o["rows"] == 9_576 for rel, o in meta["outputs"].items() if not rel.endswith("/base.jsonl.gz"))
    assert all(o["rows"] == 1_596 for rel, o in meta["outputs"].items() if rel.endswith("/base.jsonl.gz"))
    from etl.tokens import pinned_counter
    tok = pinned_counter(DEFAULT.tokenizer_json)
    row = read_rows(files.training(0, "m1", "sft"))[0]
    ids = tok._tokenizer.encode(row["completion"], add_special_tokens=False).ids
    assert ids[-1] == tok.token_id(pinned.COMPLETION_END) and ids.count(ids[-1]) == 1


def test_fractions_pin_matches_types():
    assert {Fraction(n, pinned.ITEMS_PER_CVE) for n in masks.PER_CVE.values()} == set(pinned.M1V_FRACTIONS)
