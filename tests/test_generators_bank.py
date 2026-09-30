"""Step 4 on a small fixture: the dry path, the model path with its one regeneration, the shortcut
guard, freezing, invariant checks, and the external runner's pure helpers."""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from etl import pinned
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT, Paths
from generators import __main__ as cli
from generators import external
from generators.bank import build, check, items, mcq, prepare, report
from generators.bank.files import BankFiles

ROOT = Path(__file__).resolve().parents[1]
NONTEST_CWES = ["CWE-787", "CWE-125", "CWE-476", "CWE-416", "CWE-190", "CWE-20", "CWE-401", "CWE-835"]
TEST_CWES = ["CWE-787", "CWE-125", "CWE-400", "CWE-369"]
VEC = "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
RARE = ["CWE-1335", "CWE-1339", "CWE-466", "CWE-823", "CWE-128", "CWE-193", "CWE-1284", "CWE-252", "CWE-253",
        "CWE-457", "CWE-170", "CWE-134"]


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


def function(i: int) -> tuple[str, str]:
    vuln = f"int f{i}(char *b, int n)\n{{\n    char t[8];\n    memcpy(t, b, n);\n    return t[{i % 3}];\n}}"
    if i % 2:
        patched = vuln.replace("    memcpy(t, b, n);", "    if (n > 8)\n        return -1;\n    memcpy(t, b, n);")
    else:
        patched = vuln.replace("memcpy(t, b, n);", "memcpy(t, b, n < 8 ? n : 8);")
    return vuln, patched


def make_data(root: Path) -> Paths:
    (root / "combined_dataset").mkdir(parents=True)
    (root / "mitre_cwe").mkdir()
    (root / "mitre_cwe" / "cwec_v4.20.xml").symlink_to(DEFAULT.cwe_xml)
    facts, split = [], []
    for i, cwe in enumerate(NONTEST_CWES + TEST_CWES):
        cve = f"CVE-20{16 + i // 4}-{1000 + i}"
        vuln, patched = function(i)
        desc = f"A flaw in f{i} lets remote attackers corrupt memory."
        if i == 0:
            desc += f" This is {cwe} ({cve})."           # literal CWE ID and own CVE ID: both redacted
        if i == 1:
            patched = patched.replace("{", f"{{ /* {cve} */", 1)  # own CVE ID in a patched comment
        r = pinned.patch_line_set(vuln, patched)
        facts.append({"cve_id": cve, "cwe": cwe, "description": desc, "cvss_vector": VEC, "vuln_func": vuln,
                      "patched_func": patched, "patch_lines": list(r.lines), "n_lines": r.n_lines,
                      "comment_mask_ok": r.mask_ok, "published": f"20{16 + i // 4}-01-0{1 + i % 4}T00:00:00.000"})
        split.append({"cve_id": cve, "pool": "nontest" if i < len(NONTEST_CWES) else "test", "cluster_id": cve})
    out = root / "combined_dataset"
    (out / "facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
    (out / "split.jsonl").write_text("".join(json.dumps(s) + "\n" for s in split))
    (out / "baselines.json").write_text(json.dumps({
        "exact_id_hierarchy": {"adopted_schedule_top": [{"cwe": "CWE-125"}]}, "cvss_majority": {"vector": VEC}}))
    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(root / "tokenizer.json"))
    return Paths(root)


@pytest.fixture()
def data(tmp_path):
    return make_data(tmp_path)


def run(data: Paths, *args: str) -> int:
    return cli.main(["bank", *args, "--data-dir", str(data.data), "--tokenizer", str(data.data / "tokenizer.json"),
                     "--unpinned-tokenizer"])


def facts_of(data):
    return [json.loads(line) for line in data.facts.open()]


def write_generations(path: Path, requests: list[dict], reply_for) -> None:
    """reply_for(request) -> ('ok', [ids]) | ('refusal', None) | ('errored', None)."""
    rows = []
    for r in requests:
        kind, ids = reply_for(r)
        row = {"custom_id": r["custom_id"], "result_type": "succeeded", "model": "claude-opus-5-5",
               "stop_reason": "end_turn", "stop_category": None, "text": None, "usage": {"output_tokens": 50},
               "error": None, "message_id": "msg_x"}
        if kind == "ok":
            row["text"] = json.dumps({"distractors": ids})
        elif kind == "refusal":
            row.update(stop_reason="refusal", stop_category="cyber", text="")
        else:
            row.update(result_type="errored", model=None, stop_reason=None, usage=None, error={"type": "api_error"})
        rows.append(row)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# --- Dry path ------------------------------------------------------------------


def test_dry_build_check_report(data):
    assert run(data, "build", "--dry-mcq") == 0
    assert run(data, "check", "--dry-mcq") == 0
    assert run(data, "report", "--dry-mcq") == 0
    files = BankFiles.of(data, dry=True)
    bank = read(files.bank("nontest")) + read(files.bank("test"))
    assert len(bank) == 12 * pinned.ITEMS_PER_CVE
    assert Counter(r["type"] for r in bank) == {"mcq": 12, "exact_id": 12, "cvss": 12, "find_error": 24, "line_loc": 12}
    first = [r for r in bank if r["cve_id"] == "CVE-2016-1000"]
    assert all("CWE-787" not in r["user"].split("\n\nWhich")[0] and "CVE-2016-1000" not in r["user"] for r in first)
    assert any("CWE-[redacted] (CVE-[redacted])" in r["user"] for r in first)
    patched = next(r for r in bank if r["item_id"] == "CVE-2016-1001:find_error:1")
    assert "/* CVE-[redacted] */" in patched["user"]
    meta = json.loads(files.meta.read_text())
    assert meta["mcq_guard"]["mode"] == "dry" and meta["counts"]["test"] == {"cves": 4, "items": 24}
    rep = json.loads(files.report_json.read_text())
    assert rep["baselines_nontest"]["find_error_always_vulnerable"] == {"per_function": 0.5, "paired": 0.0}
    assert rep["git_cross_check"]["checked"] == 12
    assert files.report_md.read_text().startswith("# Step 4: question bank (dry run, not frozen)")


def test_item_targets_and_prompts(data):
    run(data, "build", "--dry-mcq")
    rows = {r["item_id"]: r for r in read(BankFiles.of(data, dry=True).bank("nontest"))}
    loc = rows["CVE-2016-1001:line_loc:0"]
    assert loc["target"] == "LINES: 3" and loc["gold"] == {"lines": [3], "n_lines": 6}  # insertion -> preceding code line
    assert "1: int f1(char *b, int n)\n2: {\n3:     char t[8];" in loc["user"]
    assert rows["CVE-2016-1000:find_error:0"]["target"] == "VULNERABLE: yes, CWE-787"
    assert rows["CVE-2016-1000:find_error:1"]["target"] == "VULNERABLE: no"
    q = rows["CVE-2016-1002:mcq:0"]
    letters = [line[0] for line in q["user"].split("\n") if line[1:3] == ". "]
    assert letters == list("ABCD") and q["target"] == f"ANSWER: {q['gold']['letter']}"
    assert all(r["prompt"].startswith("<|im_start|>system\n") and r["prompt_tokens"] > 0 for r in rows.values())


def test_rebuild_is_byte_identical_across_hash_seeds(data):
    digests = []
    for seed in ("0", "12345"):
        subprocess.run([sys.executable, "-m", "generators", "bank", "build", "--dry-mcq", "--data-dir", str(data.data),
                        "--tokenizer", str(data.data / "tokenizer.json"), "--unpinned-tokenizer"],
                       check=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": seed})
        files = BankFiles.of(data, dry=True)
        digests.append([hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (files.bank("nontest"), files.bank("test"), files.mcq_decisions, files.meta)])
    assert digests[0] == digests[1]


# --- Requests ------------------------------------------------------------------


def test_prepare_requests(data, monkeypatch):
    monkeypatch.setattr(pinned, "MCQ_PILOT_SIZE", 5)
    files, pilot = prepare.prepare_mcq(data, pilot=True)
    assert len(pilot) == 5 and all(r["pool"] == "nontest" for r in pilot) and files.dir == data.bank_pilot
    _, full = prepare.prepare_mcq(data)
    assert len(full) == 12 and {r["pool"] for r in full} == {"nontest", "test"}
    for r in full:
        assert external.CUSTOM_ID.fullmatch(r["custom_id"]) and r["custom_id"].endswith("_a1")
        p = r["params"]
        assert p["model"] == "claude-opus-5-5" and p["output_config"]["effort"] == "medium"
        assert p["output_config"]["format"]["schema"] == pinned.MCQ_RESPONSE_SCHEMA
        assert "temperature" not in p and "fallbacks" not in p
        assert "Correct answer: CWE-" in p["messages"][0]["content"]
    first = next(r for r in full if r["cve_id"] == "CVE-2016-1000")
    assert "CVE-2016-1000" not in first["params"]["messages"][0]["content"].split("Correct answer")[0]


def test_request_pins_are_enforced():
    good = {"custom_id": "CVE-1_mcq_0_a1", "params": external.message_params("hi")}
    external.check_request_pins([good])
    for change in ({"temperature": 0.0}, {"fallbacks": "default"}, {"model": "claude-opus-5"},
                   {"output_config": {"effort": "high"}}, {"max_tokens": 10}):
        with pytest.raises(SystemExit):
            external.check_request_pins([{**good, "params": {**good["params"], **change}}])
    with pytest.raises(SystemExit):
        external.check_request_pins([{**good, "custom_id": "CVE-1:mcq:0"}])
    assert external.custom_id("CVE-2021-1234:mcq:0", 2) == "CVE-2021-1234_mcq_0_a2"


# --- Model path ----------------------------------------------------------------


def good_ids(gold: str, graph) -> list[str]:
    """Other non-test labels that are admissible for this gold: ties on familiarity, so no shortcut."""
    return mcq.admit([c for c in NONTEST_CWES if c != gold], gold, graph)[0][:5]


def test_model_path_with_regeneration_and_freeze(data, graph):
    files, reqs = prepare.prepare_mcq(data)
    gold = {f["cve_id"]: f["cwe"] for f in facts_of(data)}

    def first(r):
        if r["cve_id"] == "CVE-2016-1002":
            return "refusal", None
        if r["cve_id"] == "CVE-2016-1000":  # CWE-119 is an ancestor, 787 is gold: one admissible
            return "ok", ["CWE-119", "CWE-787", "CWE-125", "CWE-0125", "banana"]
        return "ok", good_ids(gold[r["cve_id"]], graph)

    write_generations(files.mcq_generations, reqs, first)
    with pytest.raises(SystemExit, match="regeneration"):
        run(data, "build")
    _, retry = prepare.prepare_mcq(data, retry=True)
    assert sorted(r["cve_id"] for r in retry) == ["CVE-2016-1000", "CVE-2016-1002"]
    assert all(r["attempt"] == 2 and r["custom_id"].endswith("_a2") for r in retry)
    write_generations(files.mcq_retry_generations, retry,
                      lambda r: ("ok", ["CWE-416"]) if r["cve_id"] == "CVE-2016-1000" else ("ok", good_ids(gold[r["cve_id"]], graph)))
    assert run(data, "build") == 0
    assert run(data, "check") == 0
    decisions = {d["cve_id"]: d for d in read(files.mcq_decisions)}
    d0 = decisions["CVE-2016-1000"]
    assert d0["sources"] == ["model", "model_retry", "draw"] and d0["distractors"][:2] == ["CWE-125", "CWE-416"]
    assert [x[1] for x in d0["attempts"][0]["rejected"]] == ["ancestor", "gold", "duplicate", "malformed"]
    assert decisions["CVE-2016-1002"]["attempts"][0]["status"] == "refusal"
    assert decisions["CVE-2016-1002"]["sources"] == ["model_retry"] * 3
    meta = json.loads(files.meta.read_text())
    assert meta["mcq_guard"]["fired"] is False and Fraction(meta["mcq_guard"]["shortcut_model"]) <= Fraction(1, 2)
    assert run(data, "report", "--skip-git") == 0
    rep = json.loads(files.report_json.read_text())
    assert rep["mcq"]["refusal_categories"] == {"cyber": 1}

    # Frozen: an identical rebuild is fine; changed generations are refused and fail the check.
    assert run(data, "build") == 0
    write_generations(files.mcq_generations, reqs, lambda r: ("ok", good_ids(gold[r["cve_id"]], graph)[::-1]))
    assert run(data, "check") == 1
    with pytest.raises(SystemExit, match="frozen"):
        run(data, "build")
    with pytest.raises(SystemExit, match="already has generations"):
        prepare.prepare_mcq(data, retry=True)  # the retry set would change under the new generations


def test_shortcut_guard_fires(data, graph):
    files, reqs = prepare.prepare_mcq(data)
    gold = {f["cve_id"]: f["cwe"] for f in facts_of(data)}
    # Rare-but-live distractors: gold is always the only familiar option, so the shortcut is 1.
    write_generations(files.mcq_generations, reqs, lambda r: ("ok", mcq.admit(RARE, gold[r["cve_id"]], graph)[0][:3]))
    assert run(data, "build") == 0
    meta = json.loads(files.meta.read_text())
    assert meta["mcq_guard"]["fired"] is True and meta["mcq_guard"]["shortcut_model"] == "1"
    assert run(data, "check") == 0
    run(data, "build", "--dry-mcq")
    real = {d["cve_id"]: d for d in read(files.mcq_decisions)}
    dry = {d["cve_id"]: d for d in read(BankFiles.of(data, dry=True).mcq_decisions)}
    assert all(real[c]["distractors"] == dry[c]["distractors"] for c in real)  # the guard's bank is the dry bank
    assert all(real[c]["discarded_model_decision"]["sources"] == ["model"] * 3 for c in real)


def test_pilot_report(data, graph, monkeypatch):
    monkeypatch.setattr(pinned, "MCQ_PILOT_SIZE", 6)
    files, reqs = prepare.prepare_mcq(data, pilot=True)
    gold = {f["cve_id"]: f["cwe"] for f in facts_of(data)}
    write_generations(files.mcq_generations, reqs,
                      lambda r: ("refusal", None) if r is reqs[0] else ("ok", good_ids(gold[r["cve_id"]], graph)))
    assert run(data, "pilot-report") == 0
    out = json.loads(files.pilot_report_json.read_text())
    assert out["n"] == 6 and out["statuses"] == {"ok": 5, "refusal": 1} and out["refusal_categories"] == {"cyber": 1}


# --- Rules ---------------------------------------------------------------------


def test_admission(graph):
    admitted, rejected = mcq.admit(["CWE-119", "CWE-787", "CWE-122", "CWE-399", "CWE-125", "CWE-0125", "CWE-416", "x"],
                                   "CWE-787", graph)
    assert admitted == ["CWE-125", "CWE-416"]
    assert rejected == [["CWE-119", "ancestor"], ["CWE-787", "gold"], ["CWE-122", "descendant"],
                        ["CWE-399", "not_live_weakness"], ["CWE-0125", "duplicate"], ["x", "malformed"]]


def test_parse_generation():
    ok = {"result_type": "succeeded", "stop_reason": "end_turn", "text": '{"distractors": ["CWE-1"]}'}
    assert mcq.parse_generation(ok) == ("ok", ["CWE-1"])
    assert mcq.parse_generation(None) == ("missing", [])
    assert mcq.parse_generation({**ok, "stop_reason": "refusal"}) == ("refusal", [])
    assert mcq.parse_generation({**ok, "stop_reason": "max_tokens"}) == ("max_tokens", [])
    assert mcq.parse_generation({**ok, "text": "not json"}) == ("bad_json", [])
    assert mcq.parse_generation({**ok, "text": '{"distractors": [1]}'}) == ("bad_json", [])
    assert mcq.parse_generation({**ok, "result_type": "errored"}) == ("errored", [])


def test_prior_draw(graph):
    counts = Counter({"CWE-125": 90, "CWE-416": 10, "CWE-476": 30, "CWE-119": 50, "CWE-20": 5})
    draws = [mcq.prior_draw(f"CVE-2020-{i}", "CWE-787", counts, graph, 3) for i in range(400)]
    assert all(len(set(d)) == 3 and "CWE-119" not in d for d in draws)  # 119 is an ancestor of 787
    assert draws[0] == mcq.prior_draw("CVE-2020-0", "CWE-787", counts, graph, 3)
    first = Counter(d[0] for d in draws)
    assert first["CWE-125"] > first["CWE-476"] > first["CWE-416"] > 0      # frequency-weighted
    with pytest.raises(SystemExit):
        mcq.prior_draw("CVE-1", "CWE-787", Counter({"CWE-125": 1}), graph, 2)


def test_layout_and_shortcut(graph):
    options, letter = mcq.layout("CVE-2020-1", "CWE-787", ["CWE-125", "CWE-416", "CWE-476"], graph)
    assert [o["letter"] for o in options] == list("ABCD")
    assert next(o for o in options if o["letter"] == letter)["cwe"] == "CWE-787"
    assert options[0]["name"] == graph.weaknesses[options[0]["cwe"][4:]].name
    letters = Counter(mcq.layout(f"CVE-2020-{i}", "CWE-787", ["CWE-125", "CWE-416", "CWE-476"], graph)[1] for i in range(4000))
    assert all(900 < letters[x] < 1100 for x in "ABCD")
    counts = Counter({"CWE-787": 5, "CWE-125": 5, "CWE-416": 1})
    d = lambda g, ds: {"pool": "nontest", "gold": g, "distractors": ds}  # noqa: E731
    assert mcq.shortcut([d("CWE-787", ["CWE-416", "CWE-1", "CWE-2"])], counts) == 1
    assert mcq.shortcut([d("CWE-787", ["CWE-125", "CWE-1", "CWE-2"])], counts) == Fraction(1, 2)
    assert mcq.shortcut([d("CWE-416", ["CWE-125", "CWE-1", "CWE-2"])], counts) == 0
    assert mcq.shortcut([{**d("CWE-787", ["CWE-416"] * 3), "pool": "test"}], counts) == 0  # test never counts


# --- Check catches problems -------------------------------------------------------


def test_check_catches_tampering(data, graph):
    run(data, "build", "--dry-mcq")
    files = BankFiles.of(data, dry=True)
    path = files.bank("nontest")
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]

    def with_change(i, fn):
        changed = [dict(r) for r in rows]
        fn(changed[i])
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in changed))
        problems = check.check(data, dry=True)
        path.write_text(original)
        return problems

    exact = next(i for i, r in enumerate(rows) if r["type"] == "exact_id")
    assert any("does not score 1" in p for p in with_change(exact, lambda r: r.update(target="CWE-1")))
    assert any("own CVE ID" in p for p in with_change(exact, lambda r: r.update(user=r["user"] + " " + r["cve_id"])))
    assert any("gold CWE ID" in p for p in with_change(exact, lambda r: r.update(user=r["user"] + " " + r["gold"]["cwe"])))
    q = next(i for i, r in enumerate(rows) if r["type"] == "mcq" and r["gold"]["cwe"] == "CWE-787")

    def ancestor_option(r):
        opts = [dict(o) for o in r["gold"]["options"]]
        slot = next(o for o in opts if o["letter"] != r["gold"]["letter"])
        slot.update(cwe="CWE-119", name=graph.weaknesses["119"].name)
        r["gold"] = {**r["gold"], "options": opts}
    assert any("inadmissible" in p for p in with_change(q, ancestor_option))
    assert any("does not match its fields" in p for p in with_change(exact, lambda r: r.update(item_id=r["item_id"] + "x")))
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[1:]))  # drop one item
    assert any("do not have 6 items" in p for p in check.check(data, dry=True))
    path.write_text(original)
    assert check.check(data, dry=True) == []


def test_line_numbering_guard(data):
    f = facts_of(data)[1]
    assert check.line_numbering(f) == []
    assert any("outside" in p for p in check.line_numbering({**f, "patch_lines": [99]}))
    assert any("no longer reproduces" in p for p in check.line_numbering({**f, "patch_lines": [4]}))
    assert any("not code lines" in p for p in check.line_numbering({**f, "vuln_func": f["vuln_func"].replace("    char t[8];", ""),
                                                                       "patch_lines": [3]}))


# --- External runner helpers (anthropic itself is not imported) ------------------


def test_result_rows():
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    ok = SimpleNamespace(custom_id="c1", result=SimpleNamespace(type="succeeded", message=SimpleNamespace(
        id="msg_1", model="claude-opus-5-5", stop_reason="end_turn", stop_details=None, usage=usage,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text='{"distractors": []}')])))
    refused = SimpleNamespace(custom_id="c2", result=SimpleNamespace(type="succeeded", message=SimpleNamespace(
        id="msg_2", model="claude-opus-5-5", stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"),
        usage=usage, content=[])))
    errored = SimpleNamespace(custom_id="c3", result=SimpleNamespace(type="errored", error=SimpleNamespace(type="api_error")))
    rows = [external.result_row(x) for x in (ok, refused, errored)]
    assert rows[0]["text"] == '{"distractors": []}' and rows[0]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert (rows[1]["stop_reason"], rows[1]["stop_category"], rows[1]["text"]) == ("refusal", "cyber", "")
    assert rows[2]["error"] == {"type": "api_error"} and rows[2]["text"] is None
    s = external.summarise(rows)
    assert s["result_types"] == {"errored": 1, "succeeded": 2} and s["refusal_categories"] == {"cyber": 1}
    assert s["usage_totals"] == {"input_tokens": 20, "output_tokens": 10} and s["models"] == ["claude-opus-5-5"]


def test_requires_clean_tree(monkeypatch):
    monkeypatch.setattr(external, "git_state", lambda root: {"commit": "abc", "dirty": True})
    with pytest.raises(SystemExit, match="commit the code"):
        external.require_clean_tree()
    monkeypatch.setattr(external, "git_state", lambda root: {"commit": "abc", "dirty": False})
    assert external.require_clean_tree() == {"commit": "abc", "dirty": False}


def test_parse_env():
    text = '# comment\n\nANTHROPIC_API_KEY="sk-ant-x"\nexport OTHER=\'a=b\'\nPLAIN = v \n'
    assert external.parse_env(text) == {"ANTHROPIC_API_KEY": "sk-ant-x", "OTHER": "a=b", "PLAIN": "v"}
    with pytest.raises(SystemExit, match="line 1"):
        external.parse_env("not a pair")


def test_api_key_comes_from_env_file_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    with pytest.raises(SystemExit, match="not found"):
        external.api_key(env)
    env.write_text("ANTHROPIC_API_KEY=\n")
    with pytest.raises(SystemExit, match="missing or empty"):
        external.api_key(env)
    env.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
    assert external.api_key(env) == "sk-ant-from-file"  # the shell's key is ignored


def test_env_file_is_git_ignored():
    out = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=DEFAULT.data.parent)
    assert out.returncode == 0, ".env must be git-ignored so the key is never committed"
