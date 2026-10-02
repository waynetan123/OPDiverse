"""Step 5 on a small fixture: request files, the cap rule and bands, rationale validity, the
substitution trigger, the runner's resumable loop, and the whole CLI path with fake generations."""

import hashlib
import json
import shutil
from fractions import Fraction
from types import SimpleNamespace

import pytest

from etl import pinned
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT
from frozen_model import __main__ as cli
from frozen_model import audit, distill, prepare, rationale, run_vllm
from frozen_model.files import SessionFiles
from generators import __main__ as bank_cli
from generators.bank.files import BankFiles, generations_for
from probe.prompts import render_qwen_chat
from test_generators_bank import make_data

STOP = 0  # the fake tokenizer's stop token


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    """A frozen (dry-built) bank of 8 non-test CVEs; the audit takes 4 of them."""
    paths = make_data(tmp_path)
    tok = ["--data-dir", str(paths.data), "--tokenizer", str(paths.data / "tokenizer.json"), "--unpinned-tokenizer"]
    assert bank_cli.main(["bank", "build", "--dry-mcq", *tok]) == 0
    for f in paths.bank_dry.iterdir():
        shutil.copy(f, paths.bank / f.name)
    monkeypatch.setattr(pinned, "AUDIT_PER_TYPE", 4)
    return paths


def run(paths, *args):
    return cli.main([*args, "--data-dir", str(paths.data), "--tokenizer", str(paths.data / "tokenizer.json"),
                     "--unpinned-tokenizer"])


def read(path):
    return [json.loads(line) for line in path.open()]


class FakeTokens:
    """Token ids are character codes; id 0 is the stop token."""
    sha256 = "f" * 64

    @staticmethod
    def encode(text, stopped=True):
        return [ord(c) for c in text] + ([STOP] if stopped else [])

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids if i != STOP)

    @staticmethod
    def token_id(token):
        return STOP if token == "<|im_end|>" else None


def write_generations(requests_path, reply_for):
    """reply_for(request) -> [(text, finish_reason), ...], one per sample."""
    sha = hashlib.sha256(requests_path.read_bytes()).hexdigest()
    rows = []
    for r in read(requests_path):
        outs = []
        for text, finish in reply_for(r):
            ids = FakeTokens.encode(text, finish == "stop")
            if finish == "length":
                ids, text = ids[:r["sampling"]["max_tokens"]], text[:r["sampling"]["max_tokens"]]
            outs.append({"text": text, "finish_reason": finish, "token_ids": ids})
        rows.append({"request_id": r["request_id"], "item_id": r["item_id"], "job": r["job"], "attempt": r["attempt"],
                     "requests_sha256": sha, "prompt_sha256": pinned.sha256_text(r["prompt"]), "n_prompt_tokens": 10,
                     "outputs": outs})
    generations_for(requests_path).write_text("".join(json.dumps(x) + "\n" for x in rows))


# --- Requests ------------------------------------------------------------------


def test_sampling_for_matches_the_pins():
    a = prepare.sampling_for("audit", "CVE-1:mcq:0")
    assert a == {**pinned.ROLLOUT_SAMPLING, "n": 8, "max_tokens": 512, "seed": a["seed"]}
    assert a["temperature"] == 1.0 and a["top_p"] == 1.0 and 0 <= a["seed"] < 2**31
    assert prepare.sampling_for("audit", "CVE-1:mcq:0") == a != prepare.sampling_for("audit", "CVE-2:mcq:0")
    assert prepare.sampling_for("rationale", "CVE-1:mcq:0") == pinned.EVAL_SAMPLING
    r2 = prepare.sampling_for("rationale", "CVE-1:mcq:0", 2)
    assert r2["temperature"] == 1.0 and r2["n"] == 1 and r2["max_tokens"] == 512
    with pytest.raises(ValueError):
        prepare.sampling_for("audit", "CVE-1:mcq:0", 2)


def bank_rows(n_cves):
    rows = []
    for i in range(n_cves):
        for t, idx in (("mcq", 0), ("exact_id", 0), ("cvss", 0), ("find_error", 0), ("find_error", 1), ("line_loc", 0)):
            rows.append({"cve_id": f"CVE-2020-{i}", "type": t, "index": idx, "pool": "nontest",
                         "item_id": f"CVE-2020-{i}:{t}:{idx}"})
    return rows


def test_audit_sample():
    rows = bank_rows(300)
    s = prepare.audit_sample(rows)
    by_type = {t: [r for r in s if r["type"] == t] for t in pinned.BANK_TYPES}
    assert all(len(v) == 200 for v in by_type.values())
    fe = by_type["find_error"]
    assert len({r["cve_id"] for r in fe}) == 100 and sum(r["index"] for r in fe) == 100
    assert {r["cve_id"] for r in fe} <= {r["cve_id"] for r in by_type["mcq"]}
    assert prepare.audit_sample(list(reversed(rows))) == list(reversed(s))  # same set, bank order kept
    with pytest.raises(ValueError):
        prepare.audit_sample(bank_rows(199))
    with pytest.raises(ValueError):
        prepare.audit_sample([dict(rows[0], pool="test")])


def test_prepare_on_fixture(data):
    assert run(data, "prepare") == 0
    files = SessionFiles.of(data)
    audit_rows, rat = read(files.audit_requests), read(files.rationale_requests)
    assert len(audit_rows) == 4 * 5 and len(rat) == 8 * pinned.ITEMS_PER_CVE
    bank = {r["item_id"]: r for r in read(BankFiles.of(data).bank("nontest"))}
    for r in audit_rows:
        assert r["prompt"] == bank[r["item_id"]]["prompt"]
    for r in rat:
        item = bank[r["item_id"]]
        assert r["messages"][0]["content"].startswith(item["user"]) and r["prompt"] == render_qwen_chat(r["messages"][0]["content"])
        assert r["messages"][0]["content"].endswith(f"written exactly as: {item['target']}")
    run_vllm.check_request_pins(audit_rows + rat)
    first = files.rationale_requests.read_bytes()
    assert run(data, "prepare") == 0 and files.rationale_requests.read_bytes() == first
    # once generations exist, the request file may not change
    write_generations(files.rationale_requests, lambda r: [("x", "stop")])
    lines = first.decode().splitlines(keepends=True)
    files.rationale_requests.write_text("".join(lines[:-1]))
    with pytest.raises(SystemExit, match="already has generations"):
        run(data, "prepare")


def test_runner_rejects_tampered_requests(data):
    run(data, "prepare")
    r = read(SessionFiles.of(data).audit_requests)[0]
    with pytest.raises(SystemExit, match="sampling"):
        run_vllm.check_request_pins([dict(r, sampling={**r["sampling"], "temperature": 0.7})])
    with pytest.raises(SystemExit, match="backbone"):
        run_vllm.check_request_pins([dict(r, revision="main")])
    with pytest.raises(SystemExit, match="request_id"):
        run_vllm.check_request_pins([dict(r, attempt=2)])


def test_sampling_kwargs_switch_off_watermarking():
    kw, notes = run_vllm.sampling_kwargs({"temperature": 1.0}, {"temperature", "watermarking"})
    assert kw == {"temperature": 1.0, "watermarking": False} and notes
    assert run_vllm.sampling_kwargs({"temperature": 1.0}, {"temperature"}) == ({"temperature": 1.0}, [])


# --- Audit -------------------------------------------------------------------


def test_cut():
    stop = {STOP}
    assert audit.cut([5, 6, STOP], "stop", 16, stop) == ([5, 6], False)
    assert audit.cut([5, 6, STOP], "stop", 3, stop) == ([5, 6], False)      # the stop token fits exactly
    assert audit.cut([5, 6, STOP], "stop", 2, stop) == ([5, 6], True)       # no room for the stop token
    assert audit.cut([5, 6, 7, STOP], "stop", 2, stop) == ([5, 6], True)
    assert audit.cut([5, 6], "stop", 3, stop) == ([5, 6], False)            # stop token not in token_ids
    assert audit.cut([5, 6], "stop", 2, stop) == ([5, 6], True)
    assert audit.cut([5] * 512, "length", 512, stop) == ([5] * 512, True)
    with pytest.raises(ValueError):
        audit.cut([5], "stop", 513, stop)


def test_caps_cap_rule_and_bands():
    assert audit.caps_for("mcq") == (16, 128, 256, 512) and audit.caps_for("line_loc") == (64, 128, 256, 512)
    tenth = pinned.ROLLOUT_CUTOFF_MAX
    assert audit.choose_cap({16: Fraction(1, 2), 128: tenth, 256: Fraction(0)}) == 128     # exactly 10% qualifies
    assert audit.choose_cap({16: Fraction(0), 128: Fraction(0)}) == 16
    assert audit.choose_cap({16: Fraction(1), 128: Fraction(1, 2), 512: tenth + Fraction(1, 1000)}) == 512
    assert audit.band(Fraction(1, 2)) == "as_specified"
    assert audit.band(Fraction(1, 2) - Fraction(1, 10**6)) == "dynamic_sampling"
    assert audit.band(Fraction(1, 10)) == "dynamic_sampling"
    assert audit.band(Fraction(1, 10) - Fraction(1, 10**6)) == "floored"
    assert audit.live([0, 0, Fraction(1, 2)]) and not audit.live([Fraction(1, 4)] * 8)


def test_audit_end_to_end(data):
    run(data, "prepare")
    files = SessionFiles.of(data)
    bank = {r["item_id"]: r for r in read(BankFiles.of(data).bank("nontest"))}

    def rollouts(r):
        target = bank[r["item_id"]]["target"]
        t = r["type"]
        if t == "mcq":            # half right, half wrong: every group live
            return [(f"ANSWER: {'ABCD'[i % 4]}", "stop") for i in range(8)]
        if t == "exact_id":       # always the same answer: every group dead
            return [("CWE-125", "stop")] * 8
        if t == "cvss":           # ~140 tokens of reasoning first: cut at 48 and 128, whole at 256
            return [("x" * 100 + f" {i} " + target, "stop") for i in range(8)]
        if t == "find_error":     # always "vulnerable": live only through the CWE credit on vulnerable items
            return [("VULNERABLE: yes, CWE-125" if i % 2 else "VULNERABLE: yes, CWE-787", "stop") for i in range(8)]
        return [("y" * 600, "length")] * 8   # line_loc: never finishes

    write_generations(files.audit_requests, rollouts)
    rep = audit.run(data, FakeTokens())
    assert rep["grpo_caps"] == {"mcq": 16, "exact_id": 24, "cvss": 256, "find_error": 64, "line_loc": 512}
    t = rep["types"]
    assert t["mcq"]["band"] == "as_specified" and t["exact_id"]["band"] == "floored"
    assert t["cvss"]["by_cap"]["48"]["cut_rate"] == "1" and t["cvss"]["by_cap"]["256"]["cut_rate"] == "0"
    assert t["line_loc"]["by_cap"]["512"]["parse_rate"] == "0" and t["line_loc"]["band"] == "floored"
    assert t["find_error"]["predicted_vulnerable_rate"] == {"vulnerable_items": "1", "patched_items": "1"}
    assert t["exact_id"]["identical_groups"] == "1" and t["mcq"]["identical_groups"] == "0"
    assert rep["decode_mismatches"] == 0
    scores = read(files.audit_scores)
    assert len(scores) == 20 and all(len(s["dense"]) == 8 for s in scores)
    assert files.audit_report_md.read_text().startswith("# Step 5: GRPO signal audit")


# --- Rationales ------------------------------------------------------------------


MCQ = {"item_id": "CVE-1:mcq:0", "cve_id": "CVE-1", "type": "mcq", "index": 0, "target": "ANSWER: B",
       "gold": {"letter": "B", "cwe": "CWE-787", "options": []}}
LINES = {"item_id": "CVE-1:line_loc:0", "cve_id": "CVE-1", "type": "line_loc", "index": 0, "target": "LINES: 12, 13",
         "gold": {"lines": [12, 13], "n_lines": 30}}
FE = {"item_id": "CVE-1:find_error:0", "cve_id": "CVE-1", "type": "find_error", "index": 0,
      "target": "VULNERABLE: yes, CWE-787", "gold": {"vulnerable": True, "cwe": "CWE-787"}}


@pytest.mark.parametrize("item, reply, finish, expected", [
    (MCQ, "The check is missing, so B fits.\n\n**ANSWER: B**\n", "stop", "The check is missing, so B fits.\n\nANSWER: B"),
    (MCQ, "Option B fits.\nANSWER: C", "stop", "wrong_final_answer"),
    (MCQ, "Option B fits.\nANSWER: B", "length", "truncated"),
    (MCQ, "ANSWER: B", "stop", "no_reasoning"),
    (MCQ, "", "stop", "empty"),
    (MCQ, "Option B fits.\nThat is all.", "stop", "no_final_answer"),
    (MCQ, "We were told the answer, so it must be B.\nANSWER: B", "stop", "hint_leak"),
    (MCQ, "Since the provided answer is B, B.\nANSWER: B", "stop", "hint_leak"),
    (MCQ, "Given the description, B fits.\nANSWER: B", "stop", "Given the description, B fits.\n\nANSWER: B"),
    (LINES, "The bounds check is at 12 and 13.\nLINES: 12, 13", "stop", "The bounds check is at 12 and 13.\n\nLINES: 12, 13"),
    (LINES, "Lines 12 and 14.\nLINES: 12, 14", "stop", "wrong_final_answer"),   # F1 would be 1.0 under ±1; still wrong
    (FE, "memcpy overflows t.\nVULNERABLE: yes, CWE-787", "stop", "memcpy overflows t.\n\nVULNERABLE: yes, CWE-787"),
    (FE, "memcpy overflows t.\nVULNERABLE: yes, CWE-119", "stop", "wrong_final_answer"),
])
def test_validate(graph, item, reply, finish, expected):
    target, reason = rationale.validate(item, reply, finish, graph)
    assert (target if target is not None else reason) == expected


def rows_for(t, n, a1_ok, a2_ok):
    out = []
    for i in range(n):
        source = "self_a1" if i < a1_ok else "self_a2" if i < a1_ok + a2_ok else "gold_only"
        out.append({"type": t, "source": source, "a1_reason": None if source == "self_a1" else "wrong_final_answer",
                    "a2_reason": "truncated" if source == "gold_only" else None, "a1_hint_leak": i == 0})
    return out


def test_substitution_reads_the_first_attempt():
    # the plan's example: 45 valid first time, 30 of 55 recovered -> fallback 25%, but it fires on the pass rate
    s = rationale.summarise(rows_for("mcq", 100, 45, 30) + rows_for("cvss", 100, 50, 0) + rows_for("exact_id", 100, 90, 5))
    assert s["types"]["mcq"]["substitution_fires"] and s["types"]["mcq"]["fallback_rate"] == "1/4"
    assert not s["types"]["cvss"]["substitution_fires"]          # exactly 50% does not fire
    assert s["types"]["exact_id"]["band"] == "as_specified" and s["types"]["mcq"]["band"] == "report_quality"
    assert s["substituted_types"] == ["mcq"] and not s["drop_distill_self_from_primary_test"]
    s = rationale.summarise(rows_for("mcq", 10, 4, 0) + rows_for("cvss", 10, 0, 9) + rows_for("line_loc", 10, 3, 3))
    assert s["substituted_types"] == ["mcq", "cvss", "line_loc"] and s["drop_distill_self_from_primary_test"]
    assert s["types"]["cvss"]["band"] == "floored"


def test_decide_needs_the_regeneration(graph):
    bad = {"text": "ANSWER: C", "finish_reason": "stop"}
    with pytest.raises(SystemExit, match="no regeneration"):
        rationale.decide(MCQ, bad, None, graph)
    row = rationale.decide(MCQ, bad, {"text": "B fits.\nANSWER: B", "finish_reason": "stop"}, graph)
    assert row["source"] == "self_a2" and row["target"] == "B fits.\n\nANSWER: B"
    row = rationale.decide(MCQ, bad, bad, graph)
    assert row["source"] == "gold_only" and row["target"] == "ANSWER: B" and row["a2_reason"] == "wrong_final_answer"


def test_rationales_end_to_end(data):
    run(data, "prepare")
    files = SessionFiles.of(data)
    bank = {r["item_id"]: r for r in read(BankFiles.of(data).bank("nontest"))}
    bad_cves = sorted({r["cve_id"] for r in bank.values()})[:5]

    def first(r):  # line_loc and five CVEs' mcq fail the first time
        item = bank[r["item_id"]]
        if item["type"] == "line_loc" or (item["type"] == "mcq" and item["cve_id"] in bad_cves):
            return [("Thinking...", "length")]
        return [(f"Reasoning about {item['type']}.\n{item['target']}", "stop")]

    write_generations(files.rationale_requests, first)
    with pytest.raises(SystemExit, match="no regeneration"):
        run(data, "rationales")
    assert run(data, "prepare-retry") == 0
    retry = read(files.rationale_retry_requests)
    assert {r["type"] for r in retry} == {"line_loc", "mcq"} and len(retry) == 8 + 5
    assert all(r["attempt"] == 2 and r["sampling"]["temperature"] == 1.0 for r in retry)
    write_generations(files.rationale_retry_requests,  # line_loc recovers; the mcq stay invalid
                      lambda r: [(f"Second look.\n{bank[r['item_id']]['target']}", "stop")] if r["type"] == "line_loc"
                      else [("We were told: B\nANSWER: B", "stop")])
    assert run(data, "rationales") == 0
    rows = read(files.distill_self)
    assert len(rows) == len(bank) and {r["item_id"] for r in rows} == set(bank)
    sub = json.loads(files.substitution.read_text())
    assert sub["types"]["line_loc"]["pass_rate_a1"] == "0" and sub["types"]["line_loc"]["sources"]["self_a2"] == 8
    assert sub["types"]["mcq"]["pass_rate_a1"] == "3/8" and sub["types"]["mcq"]["fallback_rate"] == "5/8"
    assert sub["substituted_types"] == ["mcq", "line_loc"] and not sub["drop_distill_self_from_primary_test"]
    ok = next(r for r in rows if r["type"] == "cvss")
    assert ok["source"] == "self_a1" and ok["target"] == f"Reasoning about cvss.\n\n{bank[ok['item_id']]['target']}"

    write_generations(files.audit_requests, lambda r: [(bank[r["item_id"]]["target"], "stop")] * 8)
    audit.run(data, FakeTokens())
    assert run(data, "report") == 0
    assert "Substituted types: mcq, line_loc." in files.report_md.read_text()


# --- Runner loop (vLLM itself is not importable here) ----------------------------------


def fake_out(r):
    return SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(text=f"t {r['request_id']}",
                                                                           finish_reason="stop", token_ids=[7, STOP])])


def test_run_file_resumes_and_refuses_another_file(tmp_path):
    requests = [{"request_id": f"r{i}", "item_id": f"i{i}", "job": "audit", "attempt": 1, "prompt": f"p{i}"} for i in range(7)]
    partial = tmp_path / "x_generations.partial.jsonl"
    calls = []

    def generate(part):
        calls.append([r["request_id"] for r in part])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return [fake_out(r) for r in part]

    with pytest.raises(KeyboardInterrupt):
        run_vllm.run_file(generate, requests, partial, "sha", chunk=3)
    with open(partial, "a") as f:
        f.write('{"request_id": "r3", "torn')            # a line cut short by a crash
    calls.clear()
    done = run_vllm.run_file(lambda part: (calls.append([r["request_id"] for r in part]), [fake_out(r) for r in part])[1],
                             requests, partial, "sha", chunk=3)
    assert calls == [["r3", "r4", "r5"], ["r6"]] and sorted(done) == [f"r{i}" for i in range(7)]
    assert done["r0"]["outputs"] == [{"text": "t r0", "finish_reason": "stop", "token_ids": [7, STOP]}]
    with pytest.raises(SystemExit, match="different requests file"):
        run_vllm.run_file(generate, requests, partial, "other-sha")
    assert run_vllm.file_stats(list(done.values()))["finish_reasons"] == {"stop": 7}


def test_real_request_files_match_pins():
    files = SessionFiles.of(DEFAULT)
    if not files.audit_requests.exists():
        pytest.skip("data/frozen_model not prepared")
    for path in (files.audit_requests, files.rationale_requests):
        reqs = run_vllm.load_requests(path)
        run_vllm.check_request_pins(reqs)
        run_vllm.check_rendering(reqs, lambda m: render_qwen_chat(m[0]["content"]))
    assert len(run_vllm.load_requests(files.audit_requests)) == 5 * pinned.AUDIT_PER_TYPE
