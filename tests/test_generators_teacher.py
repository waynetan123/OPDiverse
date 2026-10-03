"""Step 6 on a small fixture: request files, trace validity, the DPO near-miss rules and their rule-built
fallback, the MCQ rule, the one regeneration, the pilot report, and the whole CLI path with fake generations."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from etl import pinned, verifiers
from etl.build import read_jsonl
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT
from generators import __main__ as cli
from generators import external
from generators.bank.build import load_inputs
from generators.bank.check import code_lines
from generators.bank.files import BankFiles, generations_for, run_meta_for
from generators.bank.report import distance
from generators.teacher import dpo, prepare, traces
from generators.teacher.files import TeacherFiles
from test_frozen_model import FE, LINES, MCQ, data  # noqa: F401  (data is a fixture)

ROOT = Path(__file__).resolve().parents[1]
VEC = "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


def run(paths, *args):
    return cli.main(["teacher", *args, "--data-dir", str(paths.data), "--tokenizer", str(paths.data / "tokenizer.json"),
                     "--unpinned-tokenizer"])


def words(text):
    return len(text.split())


def gen(text="", stop="end_turn", category=None):
    return {"result_type": "succeeded", "stop_reason": stop, "stop_category": category, "text": text}


def write_generations(requests_path, reply_for, meta=False):
    """reply_for(request) -> (text, stop_reason[, category]). Writes the runner's final file (and a run_meta)."""
    sha = hashlib.sha256(requests_path.read_bytes()).hexdigest()
    rows = []
    for r in read_jsonl(requests_path):
        text, stop, *cat = reply_for(r)
        rows.append({"custom_id": r["custom_id"], "requests_sha256": sha, "result_type": "succeeded",
                     "model": "claude-opus-5-5", "message_id": "msg_x", "request_id": "req_x", "stop_reason": stop,
                     "stop_category": cat[0] if cat else None, "text": text, "usage": {"input_tokens": 100, "output_tokens": 10},
                     "error": None})
    generations_for(requests_path).write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    if meta:
        run_meta_for(requests_path).write_text(json.dumps({
            "n_requests": len(rows), "usage_totals": {"input_tokens": 100 * len(rows), "output_tokens": 10 * len(rows)},
            "stop_reasons": {}, "models": ["claude-opus-5-5"], "git": {}, "finished_utc": "x"}))


def bank_of(paths):
    return {r["item_id"]: r for r in read_jsonl(BankFiles.of(paths).bank("nontest"))}


# --- Requests ------------------------------------------------------------------


def test_prepare_on_fixture(data):
    assert run(data, "prepare") == 0
    files = TeacherFiles.of(data)
    bank = bank_of(data)
    trace, dpo_rows = read_jsonl(files.requests("trace")), read_jsonl(files.requests("dpo"))
    assert len(trace) == len(bank) == 8 * pinned.ITEMS_PER_CVE
    assert len(dpo_rows) == len(bank) - 8 and "mcq" not in {r["type"] for r in dpo_rows}
    external.check_request_pins(trace + dpo_rows)
    assert all(r["custom_id"].startswith("trace_") for r in trace) and all(r["custom_id"].startswith("dpo_") for r in dpo_rows)
    for r in trace:
        user = r["params"]["messages"][0]["content"]
        item = bank[r["item_id"]]
        assert user.startswith(item["user"]) and f"at most {pinned.TRACE_WORDS} words" in user
        assert item["target"] not in user.removeprefix(item["user"]) and "format" not in r["params"]["output_config"]
    for r in dpo_rows:
        item, params = bank[r["item_id"]], r["params"]
        user = params["messages"][0]["content"]
        assert f"The correct answer is:\n{item['target']}" in user and f"<question>\n{item['user']}\n</question>" in user
        assert params["output_config"]["format"]["schema"] == pinned.DPO_SCHEMAS[pinned.DPO_FIELDS[item["type"]]]
    ex = next(r for r in dpo_rows if r["type"] == "exact_id" and bank[r["item_id"]]["gold"]["cwe"] == "CWE-787")
    assert "- CWE-119: " in ex["params"]["messages"][0]["content"]  # 787's parent is listed by name
    first = files.requests("trace").read_bytes()
    assert run(data, "prepare") == 0 and files.requests("trace").read_bytes() == first


def test_pilot_sample(data, monkeypatch):
    monkeypatch.setattr(pinned, "TEACHER_PILOT_CVES", 3)
    assert run(data, "prepare", "--pilot") == 0
    files = TeacherFiles.of(data, pilot=True)
    trace = read_jsonl(files.requests("trace"))
    assert len(trace) == 18 and len({r["cve_id"] for r in trace}) == 3 and len(read_jsonl(files.requests("dpo"))) == 15
    assert not TeacherFiles.of(data).requests("trace").exists()


def test_prepare_is_byte_identical_across_hash_seeds(data):
    digests = []
    for seed in ("0", "4242"):
        subprocess.run([sys.executable, "-m", "generators", "teacher", "prepare", "--data-dir", str(data.data)],
                       check=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": seed})
        files = TeacherFiles.of(data)
        digests.append([hashlib.sha256(files.requests(j).read_bytes()).hexdigest() for j in pinned.TEACHER_JOBS])
    assert digests[0] == digests[1]


def test_teacher_rejects_bank_commands(data):
    with pytest.raises(SystemExit):
        run(data, "check")


# --- Traces --------------------------------------------------------------------

CVSS = {"item_id": "CVE-1:cvss:0", "cve_id": "CVE-1", "type": "cvss", "index": 0, "target": VEC, "gold": {"vector": VEC}}


@pytest.mark.parametrize("item, row, expected", [
    (MCQ, gen("The bound is never checked, so B.\n\n**ANSWER: B**\n"), ("The bound is never checked, so B.\n\nANSWER: B", True)),
    (MCQ, gen("C fits better.\nANSWER: C"), ("C fits better.\n\nANSWER: C", False)),       # a wrong answer is kept
    (MCQ, gen("", "refusal", "reasoning_extraction"), "refusal"),
    (MCQ, gen("B fits", "max_tokens"), "max_tokens"),
    (MCQ, gen(""), "empty"),
    (MCQ, gen("Thinking.\nNo idea."), "no_final_answer"),
    (MCQ, gen("ANSWER: B"), "no_reasoning"),
    (MCQ, gen("word " * 600 + "\nANSWER: B"), "too_long"),
    (CVSS, gen("Remote, no auth.\nAV:N/AC:L"), "incomplete_answer"),
    (CVSS, gen("Remote, no auth.\nCVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:L"),
     ("Remote, no auth.\n\nAV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:L", False)),
    (FE, gen("memcpy overflows t.\nVULNERABLE: yes"), "incomplete_answer"),
    (FE, gen("memcpy overflows t.\nVULNERABLE: yes, cwe 0787"), ("memcpy overflows t.\n\nVULNERABLE: yes, CWE-787", True)),
    (LINES, gen("The check is at 12.\nLINES: 12, 99"), ("The check is at 12.\n\nLINES: 12", False)),
])
def test_trace_validate(graph, item, row, expected):
    valid, reason = traces.validate(item, row, words, graph)
    if isinstance(expected, str):
        assert valid is None and reason == expected
    else:
        assert reason is None and (valid["target"], valid["correct"]) == expected
        assert valid["target_tokens"] == words(valid["target"])


def test_trace_length_boundary(graph):
    fits = gen("w " * (pinned.TRACE_MAX_TOKENS - 3) + "\nANSWER: B")   # 509 words + 2 answer tokens = 511
    assert traces.validate(MCQ, fits, words, graph)[1] is None
    over = gen("w " * (pinned.TRACE_MAX_TOKENS - 2) + "\nANSWER: B")
    assert traces.validate(MCQ, over, words, graph)[1] == "too_long"


def test_trace_decide(graph):
    bad = gen("", "refusal", "reasoning_extraction")
    with pytest.raises(SystemExit, match="no regeneration"):
        traces.decide(MCQ, bad, None, words, graph)
    row = traces.decide(MCQ, bad, gen("B fits.\nANSWER: B"), words, graph)
    assert (row["source"], row["a1_reason"], row["a1_stop_category"]) == ("ext_a2", "refusal", "reasoning_extraction")
    row = traces.decide(MCQ, bad, bad, words, graph)
    assert row["source"] == "gold_only" and row["target"] == "ANSWER: B" and row["correct"] is None
    s = traces.summarise([row, traces.decide(MCQ, gen("C.\nANSWER: C"), None, words, graph)])
    assert s["types"]["mcq"]["refusal_categories"] == {"reasoning_extraction": 2}
    assert s["types"]["mcq"]["correct_rate"] == "0" and s["types"]["mcq"]["band"] == "floored"


# --- DPO rules -------------------------------------------------------------------

EXACT = {"item_id": "CVE-1:exact_id:0", "cve_id": "CVE-1", "type": "exact_id", "index": 0, "target": "CWE-787",
         "gold": {"cwe": "CWE-787"}}
PATCHED = {"item_id": "CVE-1:find_error:1", "cve_id": "CVE-1", "type": "find_error", "index": 1,
           "target": "VULNERABLE: no", "gold": {"vulnerable": False, "cwe": "CWE-787"}}
LINE1 = {"item_id": "CVE-1:line_loc:0", "cve_id": "CVE-1", "type": "line_loc", "index": 0, "target": "LINES: 12",
         "gold": {"lines": [12], "n_lines": 30}}
CODE = set(range(1, 31)) - {20}


@pytest.mark.parametrize("item, value, expected", [
    (EXACT, "CWE-119", "CWE-119"),                 # parent
    (EXACT, " cwe-0119 ", "CWE-119"),
    (EXACT, "CWE-787", "gold"),
    (EXACT, "CWE-125", "not_one_hop"),             # sibling via 119
    (EXACT, "CWE-399", "not_live_weakness"),       # a category
    (EXACT, "buffer overflow", "malformed"),
    (FE, "CWE-119", "VULNERABLE: yes, CWE-119"),
    (FE, "CWE-125", "not_one_hop"),
    (PATCHED, "CWE-787", "VULNERABLE: yes, CWE-787"),
    (PATCHED, "CWE-416", "VULNERABLE: yes, CWE-416"),
    (PATCHED, "CWE-399", "not_live_weakness"),
    (CVSS, "AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", "AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"),
    (CVSS, "CVSS:3.1/A:L/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H", "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:L"),
    (CVSS, VEC, "gold"),
    (CVSS, "AV:L/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H", "not_one_component"),
    (CVSS, "AV:N/AC:L", "malformed"),
    (LINES, [12], "LINES: 12"),                    # one line left out
    (LINES, [12, 16], "LINES: 12, 16"),            # 13 moved to 16
    (LINES, [12, 13], "gold"),
    (LINES, [12, 14], "too_close"),
    (LINES, [12, 20], "not_code_line"),
    (LINES, [11, 12], "not_a_near_miss"),          # 13 -> 11 still matches everything within ±1
    (LINES, [], "not_one_change"),
    (LINES, [12, 13, 16], "not_one_change"),
    (LINES, [12, 40], "out_of_range"),
    (LINE1, [], "single_line_drop"),
    (LINE1, [14], "LINES: 14"),
    (LINE1, [13], "too_close"),
])
def test_dpo_check(graph, item, value, expected):
    text, reason = dpo.check(item, value, CODE, graph)
    assert (text if text is not None else reason) == expected
    if text is not None:
        v = verifiers.verify_item(item["type"], text, item["gold"], graph)
        assert v.strict_ok and v.dense < 1


@pytest.mark.parametrize("text, stop, expected", [
    ('{"cwe": "CWE-119"}', "end_turn", ("ok", "CWE-119")),
    ('{"cwe": 119}', "end_turn", ("bad_json", None)),
    ("not json", "end_turn", ("bad_json", None)),
    ("", "refusal", ("refusal", None)),
])
def test_parse_proposal(text, stop, expected):
    assert dpo.parse_proposal(EXACT, gen(text, stop)) == expected
    assert dpo.parse_proposal(LINES, gen('{"lines": [true, 12]}')) == ("bad_json", None)
    assert dpo.parse_proposal(LINES, gen('{"lines": [12]}')) == ("ok", [12])


def test_rule_values_are_admissible(graph):
    for item in (EXACT, FE, PATCHED, CVSS, LINES, LINE1):
        text, reason = dpo.check(item, dpo.rule_value(item, CODE, graph), CODE, graph)
        assert reason is None, (item["item_id"], reason)
    assert dpo.rule_value(PATCHED, CODE, graph) == "CWE-787"
    assert dpo.rule_value(LINE1, CODE, graph) in ([10], [14])            # nearest code line 2 away
    assert dpo.rule_value(LINE1, CODE, graph) == dpo.rule_value(LINE1, CODE, graph)
    assert dpo.rule_value(dict(LINE1, gold={"lines": [19], "n_lines": 30}), CODE, graph) in ([17], [21])
    assert dpo.rule_value(dict(LINE1, gold={"lines": [22], "n_lines": 30}), CODE, graph) == [24]  # 20 is blank


def test_rule_values_on_the_real_bank(graph):
    bank_path = BankFiles.of(DEFAULT).bank("nontest")
    if not bank_path.exists():
        pytest.skip("data/bank not built")
    facts, _, _ = load_inputs(DEFAULT)
    code = {f["cve_id"]: code_lines(f) for f in facts}
    for item in read_jsonl(bank_path):
        if item["type"] in pinned.DPO_REQUEST_TYPES:
            c = code[item["cve_id"]]
            assert dpo.check(item, dpo.rule_value(item, c, graph), c, graph)[1] is None, item["item_id"]


def mcq_item(options, gold_letter):
    return {"item_id": "CVE-1:mcq:0", "cve_id": "CVE-1", "type": "mcq", "index": 0, "target": f"ANSWER: {gold_letter}",
            "gold": {"letter": gold_letter, "cwe": "CWE-787",
                     "options": [{"letter": l, "cwe": c, "name": ""} for l, c in zip("ABCD", options)]}}


def test_mcq_rule(graph):
    assert distance("CWE-805", "CWE-787", graph) == distance("CWE-125", "CWE-787", graph) == 2
    assert distance("CWE-416", "CWE-787", graph) not in (0, 1, 2)
    item = mcq_item(["CWE-416", "CWE-787", "CWE-125", "CWE-805"], "B")
    decision = {"gold": "CWE-787", "gold_letter": "B", "distractors": ["CWE-416", "CWE-805", "CWE-125"]}
    assert dpo.mcq_rejected(item, decision, graph) == "ANSWER: D"        # tie at 2 -> the earlier in step-4 order
    decision["distractors"] = ["CWE-416", "CWE-125", "CWE-805"]
    assert dpo.mcq_rejected(item, decision, graph) == "ANSWER: C"
    row = dpo.decide(item, None, None, set(), graph, decision)
    assert (row["source"], row["chosen"], row["rejected_dense"]) == ("rule", "ANSWER: B", "0")
    with pytest.raises(SystemExit, match="does not match"):
        dpo.mcq_rejected(item, {**decision, "gold_letter": "A"}, graph)


def test_dpo_decide(graph):
    with pytest.raises(SystemExit, match="no regeneration"):
        dpo.decide(EXACT, gen('{"cwe": "CWE-125"}'), None, CODE, graph)
    row = dpo.decide(EXACT, gen('{"cwe": "CWE-125"}'), gen('{"cwe": "CWE-119"}'), CODE, graph)
    assert (row["source"], row["rejected"], row["a1_reason"], row["proposals"]) == (
        "model_a2", "CWE-119", "not_one_hop", ["CWE-125", "CWE-119"])
    row = dpo.decide(EXACT, gen("", "refusal", "cyber"), gen("x", "end_turn"), CODE, graph)
    assert row["source"] == "rule" and row["a2_reason"] == "bad_json" and row["rejected"] in dpo.neighbours("CWE-787", graph)
    s = dpo.summarise([row])["types"]["exact_id"]
    assert s["constructible_a1"] == "0" and s["rule_fallback_rate"] == "1" and s["refusal_categories"] == {"cyber": 1}


# --- The whole path --------------------------------------------------------------


def test_end_to_end(data):
    assert run(data, "prepare") == 0
    files = TeacherFiles.of(data)
    bank = bank_of(data)

    def trace_a1(r):
        item = bank[r["item_id"]]
        if item["type"] == "line_loc":
            return "word " * 600 + "\n" + item["target"], "end_turn"            # too long
        if item["type"] == "cvss":
            return "", "refusal", "reasoning_extraction"
        return f"Reasoning about {item['type']}.\n{item['target']}", "end_turn"

    def dpo_a1(r):
        item = bank[r["item_id"]]
        if item["type"] == "cvss":
            return json.dumps({"vector": "AV:L/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H"}), "end_turn"   # two changes
        if item["type"] == "line_loc":
            return json.dumps({"lines": []}), "end_turn"                                     # single-line drop
        if item["type"] == "find_error" and item["index"] == 1:
            return json.dumps({"cwe": item["gold"]["cwe"]}), "end_turn"
        return json.dumps({"cwe": dpo.neighbours(item["gold"]["cwe"], GRAPH)[0]}), "end_turn"

    write_generations(files.requests("trace"), trace_a1, meta=True)
    write_generations(files.requests("dpo"), dpo_a1, meta=True)
    with pytest.raises(SystemExit, match="no regeneration"):
        run(data, "build")
    assert run(data, "prepare-retry") == 0
    trace_retry, dpo_retry = read_jsonl(files.requests("trace", True)), read_jsonl(files.requests("dpo", True))
    assert {r["type"] for r in trace_retry} == {"line_loc", "cvss"} and len(trace_retry) == 16
    assert {r["type"] for r in dpo_retry} == {"line_loc", "cvss"} and all(r["attempt"] == 2 for r in dpo_retry)
    first = {r["item_id"]: r for r in read_jsonl(files.requests("trace"))}
    assert all(r["params"] == first[r["item_id"]]["params"] for r in trace_retry)   # the same request, sent again

    # line_loc traces recover; cvss stays refused. DPO cvss recovers; line_loc falls to the rule.
    write_generations(files.requests("trace", True), lambda r: (
        (f"Second look.\n{bank[r['item_id']]['target']}", "end_turn") if r["type"] == "line_loc"
        else ("", "refusal", "reasoning_extraction")), meta=True)
    write_generations(files.requests("dpo", True), lambda r: (
        (json.dumps({"vector": "AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}), "end_turn") if r["type"] == "cvss"
        else (json.dumps({"lines": []}), "end_turn")), meta=True)
    assert run(data, "build") == 0

    rows = read_jsonl(files.distill_external)
    assert [r["item_id"] for r in rows] == list(bank)
    by_type = lambda rs, t: [r for r in rs if r["type"] == t]  # noqa: E731
    assert {r["source"] for r in by_type(rows, "line_loc")} == {"ext_a2"}
    assert {r["source"] for r in by_type(rows, "cvss")} == {"gold_only"}
    assert all(r["source"] == "ext_a1" and r["correct"] for r in by_type(rows, "mcq"))

    pairs = read_jsonl(files.dpo)
    assert [r["item_id"] for r in pairs] == list(bank)
    graph = GRAPH
    for r in pairs:
        item = bank[r["item_id"]]
        assert r["chosen"] == item["target"] and r["rejected"] != r["chosen"]
        v = verifiers.verify_item(item["type"], r["rejected"], item["gold"], graph)
        assert v.strict_ok and v.dense < 1
    assert {r["source"] for r in by_type(pairs, "mcq")} == {"rule"}
    assert {r["source"] for r in by_type(pairs, "cvss")} == {"model_a2"}
    assert {r["source"] for r in by_type(pairs, "line_loc")} == {"rule"}
    assert {r["source"] for r in by_type(pairs, "exact_id")} == {"model_a1"}

    rep = json.loads(files.report_json.read_text())
    assert rep["dpo"]["types"]["mcq"]["rule_defined"] and rep["dpo"]["types"]["line_loc"]["rule_fallback_rate"] == "1"
    assert rep["distill_external"]["types"]["cvss"]["refusal_categories"] == {"reasoning_extraction": 16}
    assert set(rep["usage"]) == {"trace", "trace_retry", "dpo", "dpo_retry"} and rep["total_cost_usd"] > 0
    assert files.report_md.read_text().startswith("# Step 6: external-model jobs")
    assert set(rep["sources"]) >= {"trace_requests_sha256", "dpo_retry_generations_sha256", "dpo_sha256"}


def test_pilot_report(data, monkeypatch):
    monkeypatch.setattr(pinned, "TEACHER_PILOT_CVES", 2)
    run(data, "prepare", "--pilot")
    files = TeacherFiles.of(data, pilot=True)
    bank = bank_of(data)
    write_generations(files.requests("trace"), lambda r: (f"Because.\n{bank[r['item_id']]['target']}", "end_turn"), meta=True)
    write_generations(files.requests("dpo"), lambda r: ("", "refusal", "cyber"), meta=True)
    assert run(data, "pilot-report") == 0
    out = json.loads(files.pilot_report_json.read_text())
    assert out["pilot_cves"] == 2 and out["scale_to_full_run"] == 4
    assert out["trace"]["mcq"] == {**out["trace"]["mcq"], "n": 2, "valid_a1": 2, "correct": 2}
    assert out["dpo"]["cvss"]["refusal_categories"] == {"cyber": 2}
    trace_cost = (12 * 100 * 4 + 12 * 10 * 20) / 1e6
    assert out["cost_usd"]["trace"]["full_run_estimate"] == round(trace_cost * 4, 2)


def test_real_request_files_match_pins():
    files = TeacherFiles.of(DEFAULT)
    if not files.requests("trace").exists():
        pytest.skip("data/teacher not prepared")
    bank = read_jsonl(BankFiles.of(DEFAULT).bank("nontest"))
    for job in pinned.TEACHER_JOBS:
        requests = external.load_requests(files.requests(job))
        external.check_request_pins(requests)
        assert len(requests) == sum(prepare.wanted(job, r) for r in bank)


GRAPH = load_cwe_graph(DEFAULT.cwe_xml)
