"""The one-day run on the bank fixture, with fake runs: the test request file, the step-count rule (by hand), the shared
queue and its forecast, the train-merge-evaluate job with stand-in stages, and scoring into the frozen analysis's
input. Nothing here needs a GPU."""

import hashlib
import json
import os
import textwrap

import pytest

from analysis import primary_test
from analysis.__main__ import m2_rows
from converters import masks
from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from evaluate import run_vllm
from short import __main__ as cli
from short import job, queue, score, steps
from short.files import ShortFiles, run_name
from test_frozen_model import FakeTokens, data  # noqa: F401  (data is a fixture)

S = 8  # TRAIN_STEPS in these tests


@pytest.fixture()
def short(data, monkeypatch):  # noqa: F811
    """The bank fixture (4 test CVEs, 24 items) with test_requests.jsonl built, SWEEP_LR pinned and TRAIN_STEPS 8."""
    monkeypatch.setattr(pinned, "SWEEP_LR", {a: 5e-5 for a in pinned.SWEEP_ARMS})
    monkeypatch.setattr(pinned, "TRAIN_STEPS", S)
    assert cli.main(["prepare", "--data-dir", str(data.data)]) == 0
    return data


def bank_items(paths):
    return {r["item_id"]: r for r in read_jsonl(paths.bank / "bank_test.jsonl")}


# --- Test requests ------------------------------------------------------------------------------------------


def test_test_requests(short):
    rows = read_jsonl(ShortFiles.of(short).test_requests)
    bank = bank_items(short)
    test = {r["cve_id"] for r in read_jsonl(short.split) if r["pool"] == "test"}
    assert len(rows) == len(test) * pinned.ITEMS_PER_CVE == len(bank) and {r["cve_id"] for r in rows} == test
    assert all(r["prompt"] == bank[r["item_id"]]["prompt"] and r["job"] == "eval" for r in rows)
    run_vllm.check_request_pins(rows)


def test_test_requests_refuse_a_changed_bank(short):
    path = short.bank / "bank_test.jsonl"
    rows = read_jsonl(path)
    path.write_bytes(jsonl_bytes([r for r in rows if r["cve_id"] != rows[0]["cve_id"]]))
    with pytest.raises(SystemExit, match="covers 3 CVEs"):
        cli.main(["prepare", "--data-dir", str(short.data)])
    nontest = read_jsonl(short.bank / "bank_nontest.jsonl")[0]
    path.write_bytes(jsonl_bytes(rows + [nontest]))
    with pytest.raises(SystemExit, match="covers 5 CVEs"):
        cli.main(["prepare", "--data-dir", str(short.data)])
    files = ShortFiles.of(short)                      # requests that have generations never change
    files.test_requests.with_name("test_generations.jsonl").write_text("")
    path.write_bytes(jsonl_bytes([dict(r, prompt_tokens=r["prompt_tokens"] + 1) if i == 0 else r
                                  for i, r in enumerate(rows)]))
    with pytest.raises(SystemExit, match="already has generations"):
        cli.main(["prepare", "--data-dir", str(short.data)])


# --- Step count ---------------------------------------------------------------------------------------------


def records(t: float, slow_step: int = 3) -> list[dict]:
    """A 10-step cost pilot taking t seconds a step, except one step three times as long (the median ignores it)."""
    out, e = [], 30.0
    for k in range(1, 11):
        e += 3 * t if k == slow_step else t
        out.append({"step": k, "trainer": {"loss": 0.5}, "elapsed_s": e})
    out.append({"step": 10, "trainer": {"train_runtime": e}})     # the end-of-run summary has no loss
    return out


def test_seconds_per_step_and_cost():
    assert steps.seconds_per_step(records(20.0)) == 20.0
    meta = {"seconds": {"load_and_checks": 200.0, "train_this_session": 10 * 20.0 + 400}, "steps_done": 10}
    assert steps.arm_cost(records(20.0), meta) == {"t": 20.0, "o": 600.0, "steps_done": 10}
    with pytest.raises(ValueError, match="steps \\[5\\]"):
        steps.seconds_per_step([r for r in records(20.0) if r["step"] != 5])
    resumed = records(20.0)
    resumed[6]["elapsed_s"] = 1.0
    with pytest.raises(ValueError, match="resumed"):
        steps.seconds_per_step(resumed)


COSTS = {"grpo": {"t": 180.0, "o": 600.0}, "sft": {"t": 20.0, "o": 300.0}, "distill_self": {"t": 20.0, "o": 300.0},
         "dpo": {"t": 20.0, "o": 300.0}}


def test_step_rule_by_hand():
    # H = 24, R = 3: 21 h = 75,600 s per GPU; E = 900 s; ten runs per arm; three on the busiest GPU.
    # total:   10 x (1,500 + 3 x 1,200) + 10 x S x (180 + 3 x 20) = 51,000 + 2,400 S <= 302,400  ->  S <= 104.75
    # busiest: 3 x (600 + 180 S + 900) <= 75,600                                                ->  S <= 131.7
    out = steps.choose(COSTS, 24)
    assert (out["steps"], out["fallback"], out["arms"]) == (104, False, list(pinned.SHORT_ARMS))
    assert out["conditions"]["total"]["ok"] and out["conditions"]["busiest"]["arm"] == "grpo"
    assert not steps.conditions(COSTS, 108, 24)["total"]["ok"]
    assert steps.choose(COSTS, 200)["steps"] == 396          # the largest multiple of 4 up to 399


def test_step_rule_fallback():
    # GRPO at 1,000 s a step: total 51,000 + 10,600 S <= 302,400 -> S = 20 < 48, so GRPO is dropped; the other three
    # arms: 36,000 + 600 S <= 302,400 and 3 x (1,200 + 20 S) <= 75,600 -> capped at 396.
    out = steps.choose({**COSTS, "grpo": {"t": 1000.0, "o": 600.0}}, 24)
    assert (out["steps_over_four_arms"], out["steps"], out["fallback"]) == (20, 396, True)
    assert out["arms"] == list(pinned.SHORT_FALLBACK_ARMS) and "below 48" in out["fallback_reason"]
    no_grpo = steps.choose({a: c for a, c in COSTS.items() if a != "grpo"}, 24)
    assert no_grpo["fallback"] and "no GRPO cost run" in no_grpo["fallback_reason"] and no_grpo["steps"] == 396
    with pytest.raises(ValueError, match="only GRPO may be missing"):
        steps.choose({a: c for a, c in COSTS.items() if a != "sft"}, 24)
    assert not steps.choose({a: {"t": 1e5, "o": 0.0} for a in COSTS}, 24)["fits"]


def write_pilot(paths, arm, t, o=600.0):
    d = steps.pilot_dir(ShortFiles.of(paths).pilot / "cost", arm)
    d.mkdir(parents=True)
    (d / "train_log.jsonl").write_bytes(jsonl_bytes(records(t)))
    (d / "run_meta.json").write_text(json.dumps({"config": {"arm": arm}, "pilot": True, "steps_done": 10,
                                                 "seconds": {"load_and_checks": o, "train_this_session": 10 * t}}))


def test_steps_command_and_freeze(short, capsys):
    for arm, c in COSTS.items():
        write_pilot(short, arm, c["t"], c["o"])
    args = ["steps", "--data-dir", str(short.data), "--hours-left", "24"]
    assert cli.main(args) == 0
    assert "S = 104" in capsys.readouterr().out
    out = json.loads(ShortFiles.of(short).step_count.read_text())
    assert out["steps"] == 104 and out["costs"]["grpo"] == {"seconds_per_step": 180.0, "overhead_seconds": 600.0,
                                                            "pilot_steps": 10}
    assert cli.main(args) == 0                                     # the same computation again is fine
    with pytest.raises(SystemExit, match="frozen"):
        cli.main(["steps", "--data-dir", str(short.data), "--hours-left", "23"])


# --- Queue --------------------------------------------------------------------------------------------------


def write_step_count(paths, run_hours, arms=pinned.SHORT_ARMS, fallback=False, hours_left=24.0):
    files = ShortFiles.of(paths)
    files.step_count.write_text(json.dumps({"steps": S, "arms": list(arms), "fallback": fallback,
                                            "run_hours": run_hours, "hours_left": hours_left}))


HOURS = {"grpo": 5.0, "dpo": 1.0, "distill_self": 0.8, "sft": 0.5}


def test_queue_order_and_forecast(short):
    write_step_count(short, HOURS)
    out = queue.build(short, ["0", "1", "2", "3"])
    assert len(out["jobs"]) == 40 and all(j.startswith("grpo@") for j in out["jobs"][:10])
    assert out["jobs"][:3] == ["grpo@5e-05@m2-mcq@0", "grpo@5e-05@m2-mcq@1", "grpo@5e-05@m2-exact_id@0"]
    assert out["jobs"][10].startswith("dpo@") and out["jobs"][-1].startswith("sft@")
    grpo = [sum(j.startswith("grpo") for j in out["forecast"][g]["jobs"]) for g in "0123"]
    assert grpo == [3, 3, 2, 2]
    # GRPO leaves 15, 15, 10, 10 h; the dpo runs fill GPUs 2 and 3 to 15; distill-self adds 2.4, 2.4, 1.6, 1.6;
    # sft brings every GPU to 18.4 or 18.1.
    assert [round(out["forecast"][g]["hours"], 6) for g in "0123"] == [18.4, 18.4, 18.1, 18.1]
    assert out["forecast_finish_hours"] == pytest.approx(18.4) and not out["over_budget"]
    assert queue.build(short, ["0", "1"])["over_budget"]              # 36.5 h on two GPUs is past 24 - 3


def test_queue_leaves_out_evaluated_runs_and_needs_the_pin(short, monkeypatch):
    write_step_count(short, {a: HOURS[a] for a in pinned.SHORT_FALLBACK_ARMS}, pinned.SHORT_FALLBACK_ARMS, True)
    files = ShortFiles.of(short)
    files.generations("sft", "m2-cvss", 1).parent.mkdir(parents=True)
    files.generations("sft", "m2-cvss", 1).write_text("")
    out = queue.build(short, ["0", "1", "2", "3"])
    assert len(out["jobs"]) == 29 and out["already_evaluated"] == ["sft@5e-05@m2-cvss@1"]
    assert not any(j.startswith("grpo") for j in out["jobs"])
    monkeypatch.setattr(pinned, "TRAIN_STEPS", 12)
    with pytest.raises(SystemExit, match="TRAIN_STEPS is 12"):
        queue.build(short, ["0"])


# --- One job ------------------------------------------------------------------------------------------------

STAGES = {
    "fake_train": '''
        import argparse, json, sys
        from pathlib import Path
        ap = argparse.ArgumentParser(); ap.add_argument("--out"); a, rest = ap.parse_known_args()
        out = Path(a.out); open(out.parent / "calls", "a").write("train " + " ".join(sys.argv[1:]) + "\\n")
        (out / "trainer" / "checkpoint-8").mkdir(parents=True, exist_ok=True)
        (out / "run_meta.json").write_text(json.dumps({"steps_done": 8, "pilot": False}))
    ''',
    "fake_merge": '''
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser(); ap.add_argument("--adapter"); ap.add_argument("--out"); a = ap.parse_args()
        out = Path(a.out); open(out.parents[2] / "calls", "a").write("merge " + a.adapter + "\\n")
        assert Path(a.adapter).exists()
        out.mkdir(parents=True, exist_ok=True)
        (out / "merge_meta.json").write_text(json.dumps({"adapter": a.adapter}))
    ''',
    "fake_eval": '''
        import argparse, os, sys
        from pathlib import Path
        ap = argparse.ArgumentParser(); ap.add_argument("--requests"); ap.add_argument("--model"); ap.add_argument("--out")
        a = ap.parse_args()
        out = Path(a.out); open(out.parents[1] / "calls", "a").write("eval " + a.model + "\\n")
        assert (Path(a.model) / "merge_meta.json").exists()
        if os.environ.get("FAIL_EVAL"):
            print("RuntimeError: vLLM died", file=sys.stderr); sys.exit(3)
        (out / "test_generations.jsonl").write_text("{}\\n")
    ''',
}


@pytest.fixture()
def stages(tmp_path, monkeypatch):
    for name, body in STAGES.items():
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    monkeypatch.setattr(job, "TRAIN", "fake_train")
    monkeypatch.setattr(job, "MERGE", "fake_merge")
    monkeypatch.setattr(job, "EVALUATE", "fake_eval")
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setattr(pinned, "TRAIN_STEPS", S)
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


def run_job(runs, *extra):
    out = runs / "grpo_m2-cvss_s1"
    code = job.main(["--arm", "grpo", "--lr", "5e-05", "--config", "m2-cvss", "--seed", "1", "--out", str(out),
                     "--requests", str(runs / "test_requests.jsonl"), *extra])
    calls = (runs / "calls").read_text().splitlines() if (runs / "calls").exists() else []
    (runs / "calls").unlink(missing_ok=True)
    return code, out, calls


def test_job_runs_the_stages_in_order_and_cleans_up(stages):
    code, out, calls = run_job(stages, "--gradient-checkpointing", "on")
    assert code == 0 and [c.split()[0] for c in calls] == ["train", "merge", "eval"]
    assert "--config m2-cvss --seed 1" in calls[0] and calls[0].endswith("--gradient-checkpointing on")
    assert calls[1].endswith("trainer/checkpoint-8")
    assert not (out / "final" / "merged").exists() and (out / "final" / "merge_meta.json").exists()
    assert (out / "final" / "test_generations.jsonl").exists()
    assert run_job(stages) == (0, out, [])                          # finished: nothing runs again


def test_job_failure_and_resume(stages, monkeypatch):
    monkeypatch.setenv("FAIL_EVAL", "1")
    code, out, calls = run_job(stages)
    assert code == 3 and [c.split()[0] for c in calls] == ["train", "merge", "eval"]
    assert (out / "final" / "merged").exists()                      # kept for the rerun
    monkeypatch.delenv("FAIL_EVAL")
    code, out, calls = run_job(stages)
    assert code == 0 and [c.split()[0] for c in calls] == ["merge", "eval"]   # training is not redone
    assert not (out / "final" / "merged").exists()
    with pytest.raises(SystemExit, match="pilots"):
        run_job(stages, "--pilot")


# --- Scoring ------------------------------------------------------------------------------------------------


def changed_vector(vector: str) -> str:
    """The vector with its confidentiality metric changed: 7 of 8 components right."""
    parts = vector.split("/")
    i = next(k for k, p in enumerate(parts) if p.startswith("C:"))
    parts[i] = "C:" + ("L" if parts[i] != "C:L" else "H")
    return "/".join(parts)


def fake_run(paths, arm, config, seed, good_cves, partial_cvss=False, requests_sha=None):
    """run_meta, merge_meta, test_run_meta and generations for one run: gold answers on good_cves, else a non-answer."""
    files = ShortFiles.of(paths)
    final = files.final(arm, config, seed)
    final.mkdir(parents=True)
    (files.run_dir(arm, config, seed) / "run_meta.json").write_text(json.dumps({
        "config": {"arm": arm, "config": config, "seed": seed, "lr": 5e-5, "steps": S}, "pilot": False,
        "steps_done": S, "adapters": {str(S): "a" * 64}, "monitor_events": [{"type": "line_loc", "event": "floor"}]
        if (arm, config, seed) == ("grpo", "m2-line_loc", 0) else [], "gpu_hours": 0.5, "flops": 1.0, "tokens": {},
        "config_sha256": "c" * 64}))
    files.merge_meta(arm, config, seed).write_text(json.dumps({
        "adapter": str(files.adapter(arm, config, seed, S)), "adapter_sha256": "a" * 64, "weights_sha256": "w" * 64}))
    sha = hashlib.sha256(files.test_requests.read_bytes()).hexdigest()
    files.eval_meta(arm, config, seed).write_text(json.dumps({"checkpoint_sha256": "w" * 64, "requests_sha256": sha}))
    bank, rows = bank_items(paths), []
    for r in read_jsonl(files.test_requests):
        item = bank[r["item_id"]]
        text = item["target"] if r["cve_id"] in good_cves else "I am not sure."
        if partial_cvss and item["type"] == "cvss":
            text = changed_vector(item["gold"]["vector"])
        rows.append({"request_id": r["request_id"], "item_id": r["item_id"], "job": "eval", "attempt": 1,
                     "requests_sha256": requests_sha or sha, "prompt_sha256": pinned.sha256_text(r["prompt"]),
                     "n_prompt_tokens": 3,
                     "outputs": [{"text": text, "finish_reason": "stop", "token_ids": FakeTokens.encode(text)}]})
    files.generations(arm, config, seed).write_text("".join(json.dumps(x) + "\n" for x in rows))


def cves_of(paths):
    return sorted({r["cve_id"] for r in read_jsonl(ShortFiles.of(paths).test_requests)})


def test_score_rows_and_summary(short):
    cves = cves_of(short)
    fake_run(short, "sft", "m2-cvss", 0, cves[:2], partial_cvss=True)
    out = score.score(short)
    files = ShortFiles.of(short)
    assert not out["scores_written"] and not files.scores.exists() and len(out["missing"]) == 39
    r = out["runs"]["sft_m2-cvss_s0"]
    assert r["dropped_type"] == "cvss" and r["types"]["cvss"]["score"] == pytest.approx(7 / 8)
    assert r["types"]["mcq"]["score"] == 0.5 and r["types"]["find_error"]["parse_rate"] == 0.5
    md = files.summary_md.read_text()
    assert "| sft_m2-cvss_s0 |" in md and "**0.875**" in md and "## Missing" in md


def test_score_refuses_runs_that_do_not_match(short):
    cves = cves_of(short)
    fake_run(short, "sft", "m2-cvss", 0, cves)
    files = ShortFiles.of(short)
    meta_path = files.run_dir("sft", "m2-cvss", 0) / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "config": {**meta["config"], "seed": 1}}))
    with pytest.raises(SystemExit, match="describes sft m2-cvss seed 1"):
        score.score(short)
    meta_path.write_text(json.dumps({**meta, "adapters": {str(S): "b" * 64}}))
    with pytest.raises(SystemExit, match="not the run's last checkpoint"):
        score.score(short)
    meta_path.write_text(json.dumps(meta))
    fake_run(short, "dpo", "m2-mcq", 1, cves, requests_sha="0" * 64)
    with pytest.raises(SystemExit, match="was not generated from test_requests.jsonl"):
        score.score(short)


def test_complete_scores_feed_the_frozen_analysis(short, capsys):
    cves = cves_of(short)
    for i, (a, c, s) in enumerate((a, c, s) for a in pinned.SHORT_ARMS for c in masks.M2 for s in pinned.SHORT_SEEDS):
        fake_run(short, a, c, s, cves[:1 + i % 4], partial_cvss=(i % 3 == 0))
    assert cli.main(["score", "--data-dir", str(short.data)]) == 0
    assert "40 runs scored, 0 missing" in capsys.readouterr().out
    files = ShortFiles.of(short)
    rows = read_jsonl(files.scores)
    assert len(rows) == 40 * len(cves) * pinned.ITEMS_PER_CVE
    assert set(rows[0]) == {"arm", "seed", "item_id", "metric", "dropped_type"}
    assert {r["metric"] for r in rows} == {"0", "1", "7/8"}
    assert "grpo_m2-line_loc_s0: " in files.summary_md.read_text()          # its monitor event is listed
    m2 = m2_rows(files.scores)
    assert len(m2) == 40 * len(cves) + 8 * len(cves)                         # find-the-error has two items a CVE
    scores = primary_test.collect(m2, cves=tuple(cves), seeds=(0, 1))
    assert scores.arms == pinned.PRIMARY_ARMS and len(scores.values) == 4 * 5 * 2
    with pytest.raises(ValueError, match="seeds"):
        primary_test.collect(m2, cves=tuple(cves), seeds=(0, 1, 2))


def test_run_names_match_the_launcher():
    from train.launch import parse_job
    j = parse_job("grpo@5e-05@m2-cvss@1")
    assert j.name == run_name("grpo", "m2-cvss", 1) == "grpo_m2-cvss_s1"
