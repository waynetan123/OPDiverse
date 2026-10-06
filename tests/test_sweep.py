"""Step 10, selection side, on the step-9 fixture with fake generations: the request files, the frozen type scale (by
hand), checkpoint and LR choice with their tie rules, the freeze of both, what is still missing, and the report."""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from etl import pinned
from etl.build import jsonl_bytes, read_jsonl
from evaluate import run_vllm
from sweep import __main__ as cli
from sweep import select
from sweep.files import SweepFiles
from test_converters import conv, run as conv_run  # noqa: F401  (conv is a fixture)
from test_frozen_model import FakeTokens, data  # noqa: F401  (data is a fixture)
from train.config import run_name

ROOT = Path(__file__).resolve().parents[1]
STEPS = [2, 4, 6, 8]


@pytest.fixture()
def swept(conv, monkeypatch):  # noqa: F811
    """Converters built with seed 0's checkpoint subsample widened to both of its dev CVEs (the scale needs two);
    TRAIN_STEPS 8, so checkpoints at 2, 4, 6, 8."""
    rows = read_jsonl(conv.partition)
    for r in rows:
        if r["role"]["0"] == "dev" and 0 not in r["checkpoint_seeds"]:
            r["checkpoint_seeds"] = sorted(r["checkpoint_seeds"] + [0])
    conv.partition.write_bytes(jsonl_bytes(rows))
    assert conv_run(conv, "build") == 0
    monkeypatch.setattr(pinned, "TRAIN_STEPS", 8)
    assert cli.main(["prepare", "--data-dir", str(conv.data)]) == 0
    return conv


def bank(paths):
    return {r["item_id"]: r for r in read_jsonl(paths.bank / "bank_nontest.jsonl")}


def write_eval(paths, gen_path, which, good_cves):
    """Generations for an evaluation request file: the gold target (scores 1) on `good_cves`, else a non-answer (0)."""
    files = SweepFiles.of(paths)
    req_path = files.requests(which)
    sha = hashlib.sha256(req_path.read_bytes()).hexdigest()
    items = bank(paths)
    rows = []
    for r in read_jsonl(req_path):
        text = items[r["item_id"]]["target"] if r["cve_id"] in good_cves else "I am not sure."
        rows.append({"request_id": r["request_id"], "item_id": r["item_id"], "job": r["job"], "attempt": 1,
                     "requests_sha256": sha, "prompt_sha256": pinned.sha256_text(r["prompt"]), "n_prompt_tokens": 3,
                     "outputs": [{"text": text, "finish_reason": "stop", "token_ids": FakeTokens.encode(text)}]})
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text("".join(json.dumps(x) + "\n" for x in rows))


def dev_cves(paths):
    return sorted({r["cve_id"] for r in read_jsonl(SweepFiles.of(paths).requests("dev"))})


# --- Requests ---------------------------------------------------------------------------------------------


def test_request_files(swept):
    files = SweepFiles.of(swept)
    dev, ckpt = read_jsonl(files.requests("dev")), read_jsonl(files.requests("checkpoint"))
    part = {r["cve_id"]: r for r in read_jsonl(swept.partition)}
    test = {r["cve_id"] for r in read_jsonl(swept.split) if r["pool"] == "test"}
    assert len(dev) == 2 * pinned.ITEMS_PER_CVE and {r["item_id"] for r in ckpt} <= {r["item_id"] for r in dev}
    assert all(part[r["cve_id"]]["role"]["0"] == "dev" and r["cve_id"] not in test for r in dev)
    run_vllm.check_request_pins(dev)
    items = bank(swept)
    assert all(r["prompt"] == items[r["item_id"]]["prompt"] for r in dev)


def test_request_files_are_byte_identical_across_hash_seeds(swept):
    before = SweepFiles.of(swept).requests("dev").read_bytes()
    env = {**os.environ, "PYTHONHASHSEED": "123", "PYTHONPATH": str(ROOT / "src")}
    code = f"from sweep import __main__ as m; m.main(['prepare', '--data-dir', {str(swept.data)!r}])"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
    assert SweepFiles.of(swept).requests("dev").read_bytes() == before


def test_checkpoint_subsample_must_match_the_manifest(conv):  # noqa: F811
    assert conv_run(conv, "build") == 0
    rows = read_jsonl(conv.partition)
    next(r for r in rows if r["role"]["0"] == "dev" and 0 not in r["checkpoint_seeds"])["checkpoint_seeds"].append(0)
    conv.partition.write_bytes(jsonl_bytes(rows))
    with pytest.raises(SystemExit, match="manifest"):
        cli.main(["prepare", "--data-dir", str(conv.data)])


# --- Scale and selection ---------------------------------------------------------------------------------------


def quality(arm, lr, step):
    """How many of the 2 dev CVEs a checkpoint gets right. Every run peaks at step 4 except sft at 5e-5 (flat: the tie
    goes to step 2). On full dev, 5e-5 wins for every arm but dpo, where all three tie (the lower LR, an edge, wins)."""
    if arm == "sft" and lr == 5e-5:
        return 1
    return 2 if step == 4 else 1


def dev_quality(arm, lr):
    return 1 if arm == "dpo" else (2 if lr == 5e-5 else 1)


def write_checkpoint_evals(paths, arms):
    files, cves = SweepFiles.of(paths), dev_cves(paths)
    for arm in arms:
        for lr in pinned.LR_GRID:
            for s in STEPS:
                write_eval(paths, files.generations(arm, lr, s, "checkpoint"), "checkpoint", cves[:quality(arm, lr, s)])


def test_scale_by_hand(swept):
    with pytest.raises(SystemExit, match="missing"):
        select.freeze_scale(swept)
    write_checkpoint_evals(swept, pinned.SCALE_ARMS)
    out = select.freeze_scale(swept)
    # an evaluation with one of two CVEs right has variance 1/2 on every type; with both right, 0
    halves = sum(quality(a, lr, s) == 1 for a in pinned.SCALE_ARMS for lr in pinned.LR_GRID for s in STEPS)
    expected = math.sqrt(0.5 * halves / (len(pinned.SCALE_ARMS) * 3 * 4))
    assert all(abs(v - expected) < 1e-12 for v in out["scale"].values()) and len(out["evaluations"]) == 60
    assert select.freeze_scale(swept) == out                                      # idempotent
    files = SweepFiles.of(swept)
    write_eval(swept, files.generations("base", 1e-5, 2, "checkpoint"), "checkpoint", dev_cves(swept))
    with pytest.raises(SystemExit, match="frozen"):
        select.freeze_scale(swept)


def test_selection(swept, monkeypatch):
    files, cves = SweepFiles.of(swept), dev_cves(swept)
    write_checkpoint_evals(swept, pinned.SCALE_ARMS)
    with pytest.raises(SystemExit, match="scale.json is missing"):
        select.select(swept)
    scale = select.freeze_scale(swept)["scale"]
    with pytest.raises(SystemExit, match="commit them before any selection"):
        select.select(swept)
    monkeypatch.setattr(pinned, "SELECTION_SCALE", scale)
    out = select.select(swept)
    assert out["runs"]["sft_lr5e-05"]["chosen_step"] == 2 and out["runs"]["dpo_lr1e-05"]["chosen_step"] == 4
    assert out["arms"] == {} and sum("full-dev" in t for t in out["todo"]) == 15
    assert sum(t.startswith("sft_dpo") for t in out["todo"]) == 12
    for arm in pinned.SCALE_ARMS:
        for lr in pinned.LR_GRID:
            step = out["runs"][run_name(arm, lr)]["chosen_step"]
            write_eval(swept, files.generations(arm, lr, step, "dev"), "dev", cves[:dev_quality(arm, lr)])
    out = select.select(swept)
    assert {a: (v["lr"], v["step"], v["edge"]) for a, v in out["arms"].items()} == {
        "base": (5e-5, 4, False), "sft": (5e-5, 2, False), "distill_self": (5e-5, 4, False), "grpo": (5e-5, 4, False),
        "dpo": (1e-5, 4, True)}
    p = out["arms"]["sft"]["paired_vs_winner"]["1e-05"]
    assert p["difference"] < 0 and p["interval"][0] <= p["difference"] <= p["interval"][1]
    assert out["arms"]["dpo"]["paired_vs_winner"]["5e-05"]["difference"] == 0
    # choices are frozen: a re-evaluation that would move a winner is refused
    write_eval(swept, files.generations("grpo", 2e-4, 4, "dev"), "dev", cves)
    write_eval(swept, files.generations("grpo", 5e-5, 4, "dev"), "dev", [])
    with pytest.raises(SystemExit, match="frozen"):
        select.select(swept)
    monkeypatch.setattr(pinned, "SELECTION_SCALE", {t: 1.0 for t in pinned.BANK_TYPES})
    with pytest.raises(SystemExit, match="SELECTION_SCALE"):
        select.select(swept)


def test_argmax_ties_and_statistic():
    assert select.argmax([(1e-5, 1.0), (5e-5, 1.0), (2e-4, 0.5)]) == 1e-5
    assert select.argmax([(2, 0.1), (4, 0.3), (6, 0.3)]) == 4
    units = {"mcq": {"a": 1, "b": 0}, "cvss": {"a": 0.5, "b": 0.5}}
    assert select.statistic(units, {"mcq": 0.5, "cvss": 0.25}, ("mcq", "cvss")) == pytest.approx((1 + 2) / 2)
    with pytest.raises(SystemExit, match="different CVEs"):
        select.per_cve({"mcq": {"a": 1}, "cvss": {"b": 1}}, {"mcq": 1, "cvss": 1}, ("mcq", "cvss"))
    assert select.variance([1.0, 0.0]) == 0.5


def test_report(swept, monkeypatch):
    files, cves = SweepFiles.of(swept), dev_cves(swept)
    write_checkpoint_evals(swept, pinned.SCALE_ARMS)
    monkeypatch.setattr(pinned, "SELECTION_SCALE", select.freeze_scale(swept)["scale"])
    out = select.select(swept)
    step = out["runs"]["sft_lr5e-05"]["chosen_step"]
    for lr in pinned.LR_GRID:
        write_eval(swept, files.generations("sft", lr, out["runs"][run_name("sft", lr)]["chosen_step"], "dev"), "dev", cves[:1])
    select.select(swept)
    write_eval(swept, files.backbone_generations, "dev", [])
    run_dir = files.run_dir("sft", 5e-5)
    (run_dir / "run_meta.json").write_text(json.dumps({"steps_done": 8, "tokens": {"forward": 1234}, "flops": 1e15,
                                                       "gpu_hours": 0.5, "monitor_events": []}))
    assert conv_run(swept, "report") == 0                                     # the flags come from step 9's report
    assert cli.main(["report", "--data-dir", str(swept.data)]) == 0
    rep = json.loads(files.report_json.read_text())
    assert rep["runs"]["sft_lr5e-05"]["training"]["tokens"]["forward"] == 1234
    assert rep["backbone_dev"]["types"]["mcq"]["score"] == 0.0 and rep["selection"]["arms"]["sft"]["lr"] == 1e-5
    assert set(rep["flags"]) == {"distill_self_gold_only", "dpo_rule_built"}
    md = files.report_md.read_text()
    assert "## Winners" in md and "raw backbone (reference)" in md and f"step{step}/checkpoint ✓" in md
    assert "| sft_lr5e-05 | 8 | 1,234 |" in md and "## Still to do" in md
