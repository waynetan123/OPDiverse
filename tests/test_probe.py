import json
import random
from fractions import Fraction
from types import SimpleNamespace

import pytest

from etl import pinned
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT, Paths
from probe import prompts, run_vllm, score

HABIT_VEC = "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
VECTORS = [HABIT_VEC, "AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", "AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H",
           "AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", "AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:H"]
RARE_CWES = ["CWE-20", "CWE-22", "CWE-78", "CWE-89", "CWE-120", "CWE-122", "CWE-369", "CWE-400", "CWE-401",
             "CWE-415", "CWE-617", "CWE-772", "CWE-835", "CWE-843", "CWE-908", "CWE-1284", "CWE-674", "CWE-770"]


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


def gold_cwes(n: int) -> list[str]:
    base = ["CWE-787"] * 55 + ["CWE-476"] * 40 + ["CWE-125"] * 35 + ["CWE-416"] * 35 + ["CWE-190"] * 15
    rest = [RARE_CWES[i % len(RARE_CWES)] for i in range(n - len(base))]
    out = base + rest
    random.Random(1).shuffle(out)
    return out


@pytest.fixture()
def data(tmp_path):
    """A data dir with 320 test and 50 non-test CVEs, the real CWE XML, and step-2 baselines."""
    (tmp_path / "combined_dataset").mkdir()
    (tmp_path / "mitre_cwe").mkdir()
    (tmp_path / "mitre_cwe" / "cwec_v4.20.xml").symlink_to(DEFAULT.cwe_xml)
    cwes = gold_cwes(320)
    facts, split = [], []
    for i in range(370):
        cve = f"CVE-2021-{10000 + i}"
        test = i >= 50
        facts.append({"cve_id": cve, "cwe": cwes[i - 50] if test else "CWE-125",
                      "cvss_vector": VECTORS[i % len(VECTORS)], "published": f"2021-01-01T00:00:{i % 60:02d}.000"})
        split.append({"cve_id": cve, "pool": "test" if test else "nontest"})
    out = tmp_path / "combined_dataset"
    (out / "facts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in facts))
    (out / "split.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split))
    (out / "baselines.json").write_text(json.dumps({
        "most_frequent_cwe": {"cwe": "CWE-125"},
        "exact_id_hierarchy": {"adopted_schedule_top": [{"cwe": "CWE-125"}]},
        "cvss_majority": {"vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"},
    }))
    return Paths(tmp_path)


@pytest.fixture()
def fast_stats(monkeypatch):
    monkeypatch.setattr(pinned, "PROBE_PERMUTATIONS", 999)
    monkeypatch.setattr(pinned, "PROBE_BOOTSTRAP", 200)


def write_generations(paths: Paths, reply_for) -> None:
    """reply_for(request, fact) -> text; writes generations.jsonl matching requests.jsonl."""
    facts = {r["cve_id"]: r for r in map(json.loads, paths.facts.open())}
    requests = [json.loads(line) for line in paths.probe_requests.open()]
    rows = []
    for r in requests:
        rows.append({"request_id": r["request_id"], "cve_id": r["cve_id"], "question": r["question"],
                     "text": reply_for(r, facts[r["cve_id"]]), "finish_reason": "stop",
                     "n_prompt_tokens": 80, "n_output_tokens": 8,
                     "prompt_sha256": pinned.sha256_text(r["prompt"])})
    paths.probe_generations.write_text("".join(json.dumps(x) + "\n" for x in rows))


# --- Requests ----------------------------------------------------------------


def test_render_qwen_chat_golden():
    assert prompts.render_qwen_chat("Hi?") == (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHi?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_sample_is_deterministic_and_sized():
    ids = [f"CVE-2022-{i}" for i in range(400)]
    s = prompts.sample(ids)
    assert len(s) == 300 and s == sorted(s) and set(s) <= set(ids)
    assert prompts.sample(list(reversed(ids))) == s
    with pytest.raises(ValueError):
        prompts.sample(ids[:299])


def test_write_requests(data):
    reqs = prompts.write_requests(data)
    assert len(reqs) == 600 and {r["question"] for r in reqs} == {"cwe", "cvss"}
    assert all(r["cve_id"] in r["prompt"] and r["sampling"] == pinned.EVAL_SAMPLING for r in reqs)
    split = {r["cve_id"]: r["pool"] for r in map(json.loads, data.split.open())}
    assert all(split[r["cve_id"]] == "test" for r in reqs)
    first = data.probe_requests.read_bytes()
    prompts.write_requests(data)
    assert data.probe_requests.read_bytes() == first


def test_real_request_file_matches_pins():
    if not DEFAULT.probe_requests.exists():
        pytest.skip("data/probe/requests.jsonl not prepared")
    reqs = run_vllm.load_requests(DEFAULT.probe_requests)
    run_vllm.check_request_pins(reqs)
    run_vllm.check_rendering(reqs, lambda m: prompts.render_qwen_chat(m[0]["content"]))
    assert len(reqs) == 600


# --- Statistics --------------------------------------------------------------


def brute_control(m: score.Measure) -> Fraction:
    n = len(m.golds)
    total = sum(m.score8(a, m.golds[j]) for i, a in enumerate(m.answers) if a is not None
                for j in range(n) if j != i)
    return Fraction(total, score.UNIT * n * (n - 1))


def test_control_matches_brute_force(graph):
    rng = random.Random(3)
    pool = ["CWE-787", "CWE-125", "CWE-119", "CWE-476", "CWE-20", None]
    golds = [rng.choice(pool[:5]) for _ in range(60)]
    answers = [rng.choice(pool) for _ in range(60)]
    vec_golds = [rng.choice(VECTORS) for _ in range(60)]
    vec_answers = [rng.choice(VECTORS + [None, "AV:N/AC:?/PR:?/UI:?/S:?/C:?/I:?/A:?"]) for _ in range(60)]
    for m in (score.Measure(answers, golds, lambda a, g: 8 * (a == g)),
              score.Measure(answers, golds, lambda a, g: int(pinned.hierarchy_score(a, g, graph) * 8)),
              score.Measure(vec_answers, vec_golds, score._cvss8)):
        observed, control = score.observed_and_control(m)
        assert control == brute_control(m)
        assert observed == Fraction(sum(m.score8(a, g) for a, g in zip(m.answers, m.golds) if a is not None), 8 * 60)


def test_constant_answers_have_zero_margin():
    golds = gold_cwes(300)
    m = score.Measure(["CWE-787"] * 300, golds, lambda a, g: 8 * (a == g))
    observed, control = score.observed_and_control(m)
    assert observed == control == Fraction(55, 300)


def test_permutation_p_reproducible_and_sensitive():
    golds = gold_cwes(300)
    null = score.Measure(["CWE-787"] * 300, golds, lambda a, g: 8 * (a == g))
    strong = score.Measure(golds[:], golds, lambda a, g: 8 * (a == g))
    assert score.permutation_p(null, 500) == score.permutation_p(null, 500) == 1.0  # every shuffle ties
    assert score.permutation_p(strong, 500) == 1 / 501


@pytest.mark.parametrize("margin, p, outcome", [
    (Fraction(1, 20), 0.049, "material"),
    (Fraction(1, 20), 0.05, "not detected"),        # p must be strictly below alpha
    (Fraction(1, 20) - Fraction(1, 10**6), 0.001, "not detected"),
    (Fraction(3, 20), 0.001, "large"),
    (Fraction(3, 20) - Fraction(1, 10**6), 0.001, "material"),
])
def test_classify_boundaries(margin, p, outcome):
    assert score.classify(margin, p) == outcome


# --- End to end with fake generations -----------------------------------------


def run_scenario(data, reply_for):
    prompts.write_requests(data)
    write_generations(data, reply_for)
    return score.run(data)


def test_oracle_is_large(data, fast_stats):
    rep = run_scenario(data, lambda r, f: f["cwe"] if r["question"] == "cwe" else f["cvss_vector"])
    assert rep["outcome"] == "large"
    assert rep["lenient"]["cwe_exact"]["observed"] == 1.0 and rep["lenient"]["cvss"]["observed"] == 1.0
    assert rep["recalled"] > 0 and data.probe_report_md.read_text().startswith("# Step 3: contamination probe")


def test_habit_is_not_detected(data, fast_stats):
    rep = run_scenario(data, lambda r, f: "CWE-787" if r["question"] == "cwe" else HABIT_VEC)
    assert rep["outcome"] == "not detected"
    assert rep["lenient"]["cwe_exact"]["margin"] == 0 and rep["lenient"]["cvss"]["margin"] == 0
    # The plan's constant baseline (CWE-125) would have seen an advantage here.
    assert rep["lenient"]["cwe_exact"]["observed"] > rep["constant_baselines"]["cwe_exact"]["score"]


def test_shuffled_correct_answers_are_not_detected(data, fast_stats):
    facts = [json.loads(line) for line in data.facts.open()]
    shifted = {facts[i]["cve_id"]: facts[(i + 1) % len(facts)] for i in range(len(facts))}  # right answer, wrong CVE
    rep = run_scenario(data, lambda r, f: shifted[f["cve_id"]]["cwe"] if r["question"] == "cwe"
                       else shifted[f["cve_id"]]["cvss_vector"])
    assert rep["outcome"] == "not detected"
    assert rep["lenient"]["cwe_exact"]["p"] > 0.05


def test_partial_recall_is_material(data, fast_stats):
    facts = [json.loads(line) for line in data.facts.open()]
    known = {f["cve_id"] for f in facts if f["cwe"] in RARE_CWES[:3]}  # recalls ~20 rarer CVEs
    rep = run_scenario(data, lambda r, f: (f["cwe"] if f["cve_id"] in known else "CWE-787")
                       if r["question"] == "cwe" else HABIT_VEC)
    cwe = rep["lenient"]["cwe_exact"]
    assert 0.05 <= cwe["margin"] < 0.15 and cwe["p"] < 0.05
    assert rep["outcome"] == "material"
    recalled = [json.loads(line)["cve_id"] for line in data.probe_recalled.open()]
    assert recalled and set(recalled) <= known


def test_score_rejects_mismatched_generations(data, fast_stats):
    prompts.write_requests(data)
    write_generations(data, lambda r, f: "CWE-787")
    lines = data.probe_generations.read_text().splitlines()
    data.probe_generations.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit, match="1 missing"):
        score.run(data)
    write_generations(data, lambda r, f: "CWE-787")
    rows = [json.loads(line) for line in data.probe_generations.open()]
    rows[0]["prompt_sha256"] = "0" * 64
    data.probe_generations.write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(SystemExit, match="different prompt"):
        score.run(data)


# --- Runner helpers (vLLM itself is not importable here) -----------------------


def test_runner_request_checks(data):
    reqs = prompts.write_requests(data)
    run_vllm.check_request_pins(reqs)
    run_vllm.check_rendering(reqs, lambda m: prompts.render_qwen_chat(m[0]["content"]))
    tampered = [dict(reqs[0], sampling={**pinned.EVAL_SAMPLING, "temperature": 0.7})]
    with pytest.raises(SystemExit, match="sampling"):
        run_vllm.check_request_pins(tampered)
    with pytest.raises(SystemExit, match="chat template"):
        run_vllm.check_rendering(reqs[:1], lambda m: "<|im_start|>user\n" + m[0]["content"])


def test_llm_kwargs_disable_model_generation_config():
    kwargs, notes = run_vllm.llm_kwargs({"model", "generation_config"})
    assert kwargs["generation_config"] == "vllm" and kwargs["revision"] == pinned.TOKENIZER_REVISION and not notes
    kwargs, notes = run_vllm.llm_kwargs({"model"})
    assert "generation_config" not in kwargs and notes


def test_generation_rows():
    req = {"request_id": "CVE-1:cwe", "cve_id": "CVE-1", "question": "cwe", "prompt": "p"}
    out = SimpleNamespace(prompt_token_ids=[1, 2, 3],
                          outputs=[SimpleNamespace(text="CWE-787", finish_reason="stop", token_ids=[9, 9])])
    (row,) = run_vllm.generation_rows([req], [out])
    assert row["text"] == "CWE-787" and row["n_prompt_tokens"] == 3 and row["n_output_tokens"] == 2
    assert row["prompt_sha256"] == pinned.sha256_text("p")
