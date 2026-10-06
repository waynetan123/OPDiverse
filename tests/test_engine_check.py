"""Step 7 on a small fixture: the engine-check sample and requests, the evaluation runner's checks, the HF
runner's pure helpers, the comparison and its decision rule, and the parser review, with fake generations."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine_check import __main__ as cli
from engine_check import compare, parser_review, prepare, run_hf
from engine_check.files import EngineFiles
from etl import pinned, verifiers
from etl.paths import DEFAULT
from evaluate import run_vllm
from frozen_model import __main__ as frozen_cli
from frozen_model.files import SessionFiles
from generators import __main__ as bank_cli
from generators.bank.files import BankFiles, generations_for
from test_frozen_model import STOP, FakeTokens
from test_frozen_model import write_generations as write_audit_generations
from test_generators_bank import make_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def data(tmp_path, monkeypatch):
    """A frozen (dry-built) bank of 8 non-test CVEs; the late window is the latest 2, both sampled."""
    paths = make_data(tmp_path)
    tok = ["--data-dir", str(paths.data), "--tokenizer", str(paths.data / "tokenizer.json"), "--unpinned-tokenizer"]
    assert bank_cli.main(["bank", "build", "--dry-mcq", *tok]) == 0
    for f in paths.bank_dry.iterdir():
        shutil.copy(f, paths.bank / f.name)
    monkeypatch.setattr(pinned, "ENGINE_CVES", 2)
    return paths


def read(path):
    return [json.loads(line) for line in path.open()]


def prepared(data):
    assert cli.main(["prepare", "--data-dir", str(data.data)]) == 0
    return read(EngineFiles.of(data).vllm_requests("a"))


def gen_row(r, text, finish="stop", requests_sha=None):
    ids = FakeTokens.encode(text, finish == "stop")
    return {"request_id": r["request_id"], "item_id": r["item_id"], "job": r["job"], "attempt": r["attempt"],
            "requests_sha256": requests_sha, "prompt_sha256": pinned.sha256_text(r["prompt"]),
            "n_prompt_tokens": r["prompt_tokens"], "outputs": [{"text": text, "finish_reason": finish, "token_ids": ids}]}


def write_engines(data, vllm_reply, hf_reply, shards=2, b_reply=None):
    """*_reply(request) -> text. Writes vLLM passes A and B and the HF shards as the runners would."""
    files = EngineFiles.of(data)
    requests = read(files.vllm_requests("a"))
    a_sha = hashlib.sha256(files.vllm_requests("a").read_bytes()).hexdigest()
    b_sha = hashlib.sha256(files.vllm_requests("b").read_bytes()).hexdigest()
    a_rows = [gen_row(r, vllm_reply(r), requests_sha=a_sha) for r in requests]
    generations_for(files.vllm_requests("a")).write_text("".join(json.dumps(x) + "\n" for x in a_rows))
    b_rows = [gen_row(r, (b_reply or vllm_reply)(r), requests_sha=b_sha) for r in read(files.vllm_requests("b"))]
    generations_for(files.vllm_requests("b")).write_text("".join(json.dumps(x) + "\n" for x in b_rows))
    stop = {STOP}
    by_id = {x["request_id"]: x for x in a_rows}
    for k in range(shards):
        rows = []
        for r in run_hf.shard_of(requests, k, shards):
            row = gen_row(r, hf_reply(r), requests_sha=a_sha)
            hf_seq = run_hf.normalised(row["outputs"][0], stop)
            i = run_hf.first_divergence(hf_seq, run_hf.normalised(by_id[r["request_id"]]["outputs"][0], stop))
            row["divergence_from_vllm_a"] = None if i is None else {"index": i, "hf_token": 1, "vllm_token": 2,
                                                                    "margin": 0.01, "vllm_token_rank": 1}
            rows.append(row)
        files.hf_generations(k, shards).write_text("".join(json.dumps(x) + "\n" for x in rows))


def bank_of(data):
    return {r["item_id"]: r for r in read(BankFiles.of(data).bank("nontest"))}


# --- Sample and requests ------------------------------------------------------------------------------


def test_sample_is_the_late_window_and_never_test(data):
    rows = prepared(data)
    facts = {f["cve_id"]: f for f in read(data.facts)}
    split = {s["cve_id"]: s["pool"] for s in read(data.split)}
    nontest = sorted((f["published"], c) for c, f in facts.items() if split[c] == "nontest")
    assert {r["cve_id"] for r in rows} == {c for _, c in nontest[-2:]}         # ceil(8 / 5) = 2 latest
    assert len(rows) == 2 * pinned.ITEMS_PER_CVE and all(split[r["cve_id"]] == "nontest" for r in rows)
    bank = bank_of(data)
    assert all(r["prompt"] == bank[r["item_id"]]["prompt"] and r["sampling"] == pinned.EVAL_SAMPLING for r in rows)
    b = read(EngineFiles.of(data).vllm_requests("b"))
    assert sorted(map(json.dumps, b)) == sorted(map(json.dumps, rows)) and b != rows
    run_vllm.check_request_pins(rows)
    first = EngineFiles.of(data).vllm_requests("a").read_bytes()
    assert prepared(data) and EngineFiles.of(data).vllm_requests("a").read_bytes() == first


def test_sample_rank_and_window():
    window = [f"CVE-2020-{i:04d}" for i in range(250)]
    chosen = prepare.sample(window)
    ranked = sorted(window, key=lambda c: pinned.stable_rank(c, pinned.ENGINE_SALT))[:pinned.ENGINE_CVES]
    assert chosen == sorted(ranked) and prepare.sample(list(reversed(window))) == chosen
    with pytest.raises(ValueError, match="late window"):
        prepare.sample(window[:pinned.ENGINE_CVES - 1])
    facts = [{"cve_id": f"C{i}", "published": f"2020-01-{10 + i}"} for i in range(11)]
    pool = {f"C{i}": "test" if i == 10 else "nontest" for i in range(11)}
    assert prepare.late_window(facts, pool) == ["C8", "C9"]                       # ceil(10 / 5), test excluded


def test_requests_are_byte_identical_across_hash_seeds(data):
    prepared(data)
    before = EngineFiles.of(data).vllm_requests("b").read_bytes()
    env = {**os.environ, "PYTHONHASHSEED": "123", "PYTHONPATH": str(ROOT / "src")}
    code = ("from etl import pinned; pinned.ENGINE_CVES = 2; from engine_check import __main__ as m; "
            f"m.main(['prepare', '--data-dir', {str(data.data)!r}])")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
    assert EngineFiles.of(data).vllm_requests("b").read_bytes() == before


def test_eval_runner_rejects_departures(data):
    r = prepared(data)[0]
    with pytest.raises(SystemExit, match="EVAL_SAMPLING"):
        run_vllm.check_request_pins([dict(r, sampling={**r["sampling"], "temperature": 0.7})])
    with pytest.raises(SystemExit, match="backbone"):
        run_vllm.check_request_pins([dict(r, revision="main")])
    with pytest.raises(SystemExit, match="evaluation request"):
        run_vllm.check_request_pins([dict(r, request_id="audit:x")])
    with pytest.raises(SystemExit, match="EVAL_MAX_MODEL_LEN"):
        run_vllm.check_request_pins([dict(r, prompt_tokens=pinned.EVAL_MAX_MODEL_LEN)])
    out = SimpleNamespace(prompt_token_ids=[1, 2])
    run_vllm.check_prompt_ids([r], [out], lambda p: [1, 2])
    with pytest.raises(SystemExit, match="tokenised"):
        run_vllm.check_prompt_ids([r], [out], lambda p: [1, 3])
    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True), scheduler_config=SimpleNamespace(max_num_seqs=256))))
    s = run_vllm.engine_settings(llm)
    assert s["cache_config.enable_prefix_caching"] is True and s["model_config.dtype"] is None
    assert run_vllm.engine_settings(object())["scheduler_config.max_num_seqs"] is None


def test_real_request_files_match_pins():
    files = EngineFiles.of(DEFAULT)
    if not files.vllm_requests("a").exists():
        pytest.skip("data/engine_check not prepared")
    rows = run_vllm.load_requests(files.vllm_requests("a"))
    run_vllm.check_request_pins(rows)
    assert len(rows) == pinned.ENGINE_CVES * pinned.ITEMS_PER_CVE


# --- HF runner helpers -------------------------------------------------------------------------------------


def test_hf_helpers():
    stop = {STOP, 9}
    assert run_hf.finish_reason([5, 6, STOP], stop) == "stop" and run_hf.finish_reason([5, 6], stop) == "length"
    assert run_hf.body([5, 9], stop) == [5] and run_hf.body([5, 6], stop) == [5, 6]
    # vLLM may or may not keep the stop token; both normalise to the same symbols
    assert run_hf.normalised({"token_ids": [5, STOP], "finish_reason": "stop"}, stop) == \
        run_hf.normalised({"token_ids": [5], "finish_reason": "stop"}, stop) == [5, run_hf.STOP]
    assert run_hf.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert run_hf.first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert run_hf.first_divergence([1, 2, run_hf.STOP], [1, 2, 7, run_hf.STOP]) == 2
    assert run_hf.first_divergence([1, 2], [1, 2, 3]) == 2
    reqs = [{"request_id": i} for i in range(7)]
    assert [r["request_id"] for r in run_hf.shard_of(reqs, 1, 3)] == [1, 4]
    with pytest.raises(SystemExit):
        run_hf.shard_of(reqs, 3, 3)


def test_hf_rows_use_the_vllm_schema():
    from frozen_model.run_vllm import output_row
    r = {"request_id": "eval:x", "item_id": "x", "job": "eval", "attempt": 1, "prompt": "p", "prompt_tokens": 3}
    out = SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[SimpleNamespace(text="ab", finish_reason="stop", token_ids=[7, 8, STOP])])
    assert run_hf.hf_row(r, [7, 8, STOP], "ab", {STOP}, "sha") == output_row(r, out, "sha")


# --- Comparison and the decision -----------------------------------------------------------------------------


def test_decision_boundaries():
    gap = pinned.ENGINE_GAP
    assert compare.material(gap, (0.001, 0.02)) and compare.material(-gap, (-0.02, -0.001))
    assert not compare.material(gap - Fraction(1, 10**6), (0.001, 0.02))       # below the 1-point bar
    assert not compare.material(gap, (0.0, 0.02))                              # interval touches zero
    assert compare.bootstrap_interval([0.0, 1.0, 0.0, 1.0]) == compare.bootstrap_interval([0.0, 1.0, 0.0, 1.0])
    assert compare.bootstrap_interval([0.5] * 5) == (0.5, 0.5)


def test_compare_agree(data):
    prepared(data)
    bank = bank_of(data)
    write_engines(data, lambda r: "reasoning...\n" + bank[r["item_id"]]["target"], lambda r: "reasoning...\n" + bank[r["item_id"]]["target"])
    rep = compare.run(data, FakeTokens())
    assert rep["outcome"] == "agree" and rep["material_types"] == []
    c = rep["comparisons"]["vllm_a_vs_hf"]
    assert all(v["gap"] == 0 and v["mean_x"] == 1 for v in c.values()) and c["find_error"]["n"] == 2
    assert rep["identical_text"]["vllm_a_vs_hf"]["mcq"] == 1 and rep["divergence_hf_from_vllm_a"]["diverged"] == 0
    assert rep["decode_mismatches"] == {"vllm_a": 0, "vllm_b": 0, "hf": 0}
    assert EngineFiles.of(data).report_md.read_text().startswith("# Step 7: engine-agreement check")


def test_compare_material_and_reference_noise(data):
    prepared(data)
    bank = bank_of(data)
    target = lambda r: bank[r["item_id"]]["target"]  # noqa: E731
    wrong_mcq = lambda r: "ANSWER: " + ("A" if target(r) != "ANSWER: A" else "B") if r["type"] == "mcq" else target(r)  # noqa: E731
    write_engines(data, target, wrong_mcq, b_reply=lambda r: "no idea" if r["type"] == "cvss" else target(r))
    rep = compare.run(data, FakeTokens())
    assert rep["outcome"] == "material" and rep["material_types"] == ["mcq"]
    assert rep["comparisons"]["vllm_a_vs_hf"]["mcq"]["gap"] == -1
    assert rep["comparisons"]["vllm_a_vs_vllm_b"]["cvss"]["material"]          # reported, never decides
    assert rep["engines"]["vllm_b"]["cvss"]["parse_rate"] == 0
    assert rep["divergence_hf_from_vllm_a"]["diverged"] == 2


def test_compare_refuses_mismatched_runs(data):
    prepared(data)
    bank = bank_of(data)
    write_engines(data, lambda r: bank[r["item_id"]]["target"], lambda r: bank[r["item_id"]]["target"])
    files = EngineFiles.of(data)
    files.hf_generations(1, 2).unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        compare.run(data, FakeTokens())
    write_engines(data, lambda r: bank[r["item_id"]]["target"], lambda r: bank[r["item_id"]]["target"])
    rows = read(generations_for(files.vllm_requests("a")))
    rows[0]["prompt_sha256"] = "0" * 64
    generations_for(files.vllm_requests("a")).write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(SystemExit, match="different prompt"):
        compare.run(data, FakeTokens())


# --- Parser review -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("item_type, reply, finish, expected", [
    ("mcq", "I think it is B.", "stop", "parse_failure"),
    ("mcq", "I think it is B", "length", None),                  # cut off: not a parser question
    ("mcq", "The answer is (B).", "stop", "no_field"),
    ("mcq", "ANSWER: B", "stop", None),
    ("mcq", "So:\nANSWER: B", "stop", "lenient_not_strict"),
    ("find_error", "The function is not vulnerable.", "stop", "no_field"),
    ("line_loc", "Lines 3 and 4 change.", "stop", "no_field"),
    ("cvss", "AV:N/AC:L", "stop", "no_field"),
    ("exact_id", "CWE-787 fits.\nNot CWE-125.", "stop", "lenient_not_strict"),  # the last line's CWE decides
    ("exact_id", "It is CWE-787.\nThat is all.", "stop", "no_field"),
])
def test_parser_review_categories(item_type, reply, finish, expected):
    gold = {"mcq": {"letter": "B"}, "find_error": {"vulnerable": True, "cwe": "CWE-787"},
            "line_loc": {"lines": [3], "n_lines": 9}, "cvss": {"vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
            "exact_id": {"cwe": "CWE-787"}}[item_type]
    from etl.cwe_graph import load_cwe_graph
    v = verifiers.verify_item(item_type, reply, gold, load_cwe_graph(DEFAULT.cwe_xml))
    assert parser_review.category(item_type, reply, finish, v) == expected


def test_parser_review_end_to_end(data, monkeypatch):
    monkeypatch.setattr(pinned, "AUDIT_PER_TYPE", 4)
    tok = ["--data-dir", str(data.data), "--tokenizer", str(data.data / "tokenizer.json"), "--unpinned-tokenizer"]
    assert frozen_cli.main(["prepare", *tok]) == 0
    write_audit_generations(SessionFiles.of(data).audit_requests,
                            lambda r: [("thinking", "stop")] * 4 + [("cut", "length")] * 2 + [("x\nANSWER: A", "stop")] * 2)
    prepared(data)
    bank = bank_of(data)
    write_engines(data, lambda r: bank[r["item_id"]]["target"], lambda r: "Probably " + bank[r["item_id"]]["target"])
    rep = parser_review.run(data)
    counts = rep["counts"]
    assert counts["audit:mcq"] == {"replies": 32, "cut_off": 8, "parse_failure": 16, "no_field": 0, "lenient_not_strict": 8}
    assert counts["vllm_a:mcq"]["lenient_not_strict"] == 0 and counts["hf:mcq"]["lenient_not_strict"] == 2
    assert all(x["category"] != "lenient_not_strict" for x in rep["listed"])
    assert len(rep["lenient_not_strict_sample"]) <= 3 * len(pinned.BANK_TYPES) * pinned.PARSER_REVIEW_SAMPLE
    assert EngineFiles.of(data).parser_review_md.read_text().startswith("# Step 7: parser review")


def test_parser_guards_on_the_real_artifacts():
    """Every frozen target, DPO pair and distill-self target still means the same under the current parsers."""
    from etl.paths import DEFAULT
    from frozen_model.files import SessionFiles
    from generators.teacher.files import TeacherFiles
    if not (TeacherFiles.of(DEFAULT).dpo.exists() and SessionFiles.of(DEFAULT).distill_self.exists()):
        pytest.skip("data/teacher or data/frozen_model not built")
    assert parser_review.guards(DEFAULT) == []
