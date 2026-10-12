"""The multi-GPU launcher: queue parsing, progress and error reading, the busy-GPU check, and a whole run with a
stand-in for train.run (no GPU)."""

import json
import os
import sys
import textwrap

import pytest

from train import launch


def test_parse_queue():
    jobs = launch.parse_queue("2=dpo@5e-5,distill_self@1e-5")
    assert [(j.gpu, j.arm, j.lr) for j in jobs] == [("2", "dpo", 5e-5), ("2", "distill_self", 1e-5)]
    for bad in ("dpo@5e-5", "x=dpo@5e-5", "1=dpo", "1=notanarm@5e-5", "1="):
        with pytest.raises(ValueError):
            launch.parse_queue(bad)


def test_check_queues():
    q1, q2 = launch.parse_queue("1=grpo@5e-5"), launch.parse_queue("2=sft@5e-5")
    launch.check_queues([q1, q2])
    with pytest.raises(ValueError, match="two queues"):
        launch.check_queues([q1, launch.parse_queue("1=sft@5e-5")])
    with pytest.raises(ValueError, match="queued twice"):
        launch.check_queues([q1, launch.parse_queue("2=grpo@5e-5")])
    extra = launch.parse_queue_args(["2=--micro-batch 4"])          # same arm and LR, different settings: allowed
    q3 = launch.parse_queue("2=grpo@5e-5", extra)
    launch.check_queues([q1, q3])
    assert (q1[0].name, q3[0].name) == ("grpo_lr5e-05", "grpo_lr5e-05_micro-batch-4")
    assert q3[0].extra == ("--micro-batch", "4")


def test_parse_queue_args():
    assert launch.parse_queue_args(['3=--micro-batch 4 --max-steps 5']) == {"3": ("--micro-batch", "4", "--max-steps", "5")}
    for bad in ("--micro-batch 4", "x=--micro-batch 4", "3=--out somewhere"):
        with pytest.raises(ValueError):
            launch.parse_queue_args([bad])


def test_total_steps(monkeypatch):
    assert launch.total_steps(["--pilot", "--max-steps", "20"]) == 20
    assert launch.total_steps(["--max-steps=8"]) == 8
    assert launch.total_steps(["--max-steps", "20", "--max-steps", "5"]) == 5      # a GPU's own value wins
    monkeypatch.setattr(launch.pinned, "TRAIN_STEPS", 399)
    assert launch.total_steps(["--gradient-checkpointing", "on"]) == 399


def test_steps_done_reads_training_records_only():
    lines = [json.dumps({"step": 3, "trainer": {"loss": 0.1}}),
             json.dumps({"dynamic_sampling": {"step": 9}}),
             json.dumps({"step": 20, "trainer": {"train_runtime": 1.0}}),   # the end-of-run summary
             json.dumps({"step": 5, "trainer": {"loss": 0.2}}),
             '{"step": 6, "trai']                                          # being written
    assert launch.steps_done(lines) == 5
    assert launch.steps_done([]) == 0


def test_error_and_last_lines():
    oom = ("Loading weights: 50%\rLoading weights: 100%\n[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried\n"
           "[rank0]:[W1007 ProcessGroupNCCL.cpp:1624] Warning: destroy_process_group() was not called\n")
    assert launch.error_line(oom).startswith("torch.OutOfMemoryError: CUDA out of memory")
    assert launch.error_line("this process sees 4 GPUs; a run trains on exactly one\n") \
        == "this process sees 4 GPUs; a run trains on exactly one"
    assert launch.error_line("") == "(no output)"
    assert launch.last_line("Capturing CUDA graphs: 10%\rCapturing CUDA graphs: 90%\r") == "Capturing CUDA graphs: 90%"
    assert launch.exit_reason(-11) == "killed by SIGSEGV" and launch.exit_reason(1) == "exit code 1"


def test_busy_gpus():
    gpus = "0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n"
    apps = "GPU-aaa, 2990876, 596 MiB\nGPU-bbb, 3002961, 40792 MiB\n"
    assert launch.busy_gpus(gpus, apps) == {"0": ["pid 2990876 (596 MiB)"], "1": ["pid 3002961 (40792 MiB)"]}
    assert launch.busy_gpus(gpus, "") == {}


def test_overall_line():
    line = launch.overall(2, 1, 5, 30, 100, 3725)
    assert line.startswith("ALL   2/5 runs finished, 1 FAILED") and "30/100 steps" in line and "1:02:05" in line
    assert "FAILED" not in launch.overall(0, 0, 2, 0, None, 1) and "steps" not in launch.overall(0, 0, 2, 0, None, 1)


def test_render():
    line = launch.render("1", "grpo_lr5e-05 (1/1)", "running", 5, 20, 90.0, 3.0, "")
    assert "GPU 1" in line and "5/20" in line and "90s/step" in line and "22:30 left" in line
    assert "starting (1:05): Loading weights" in launch.render("2", "dpo", "running", 0, 20, None, 1.0, "Loading weights",
                                                               elapsed=65)
    assert "starting (0:00): launching" in launch.render("2", "dpo", "running", 0, 20, None, None, "")
    assert "no output for" in launch.render("2", "dpo", "running", 5, 20, 60.0, launch.STALLED_AFTER + 1, "")
    assert launch.bar(10, 20, 10) == "[#####-----]" and launch.bar(0, None, 3) == "[???]"


FAKE = textwrap.dedent('''
    """A stand-in for train.run: writes a few training steps, then succeeds or fails by arm."""
    import argparse, json, os, sys
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm"); ap.add_argument("--lr"); ap.add_argument("--out")
    ap.add_argument("--max-steps", type=int); ap.add_argument("--pilot", action="store_true")
    a, _ = ap.parse_known_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("gpu", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    with open(out / "train_log.jsonl", "w") as f:
        for s in range(1, a.max_steps + 1):
            f.write(json.dumps({"step": s, "trainer": {"loss": 0.5}}) + "\\n")
    if a.arm == "dpo":
        print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.14 GiB", file=sys.stderr)
        sys.exit(1)
    (out / "run_meta.json").write_text(json.dumps({"steps_done": a.max_steps, "gpu_hours": 0.01, "peak_memory_gb": 20.0}))
''')


def test_whole_launch_with_a_stand_in(tmp_path, monkeypatch, capsys):
    (tmp_path / "fake_run.py").write_text(FAKE)
    monkeypatch.setattr(launch, "RUNNER", "fake_run")
    monkeypatch.setattr(launch, "REFRESH", 0.05)
    monkeypatch.setattr(launch, "nvidia_smi_busy", lambda gpus: {})
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.chdir(tmp_path)
    code = launch.main(["--queue", "1=grpo@5e-5", "--queue", "2=dpo@5e-5,sft@5e-5", "--out-root", "pilot/cost",
                        "--", "--pilot", "--max-steps", "3"])
    out = capsys.readouterr().out
    assert code == 1                                                       # one run failed
    assert "FAILED GPU 2  dpo_lr5e-05" in out and "exit code 1: torch.OutOfMemoryError" in out
    assert "done   GPU 2  sft_lr5e-05" in out                              # the queue went on after the failure
    assert "done   GPU 1  grpo_lr5e-05" in out and "3 steps, 0.01 GPU-h, peak 20.0 GB" in out
    assert (tmp_path / "logs" / "cost_sft_lr5e-05.log").read_text().startswith("gpu 2")
    assert (tmp_path / "logs" / "cost_grpo_lr5e-05.log").read_text().startswith("gpu 1")
    assert (tmp_path / "pilot" / "cost_grpo_lr5e-05" / "run_meta.json").exists()


def test_side_by_side_settings(tmp_path, monkeypatch, capsys):
    (tmp_path / "fake_run.py").write_text(FAKE)
    monkeypatch.setattr(launch, "RUNNER", "fake_run")
    monkeypatch.setattr(launch, "REFRESH", 0.05)
    monkeypatch.setattr(launch, "nvidia_smi_busy", lambda gpus: {})
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.chdir(tmp_path)
    code = launch.main(["--queue", "2=grpo@5e-5", "--queue", "3=grpo@5e-5", "--queue-args", "2=--micro-batch 1",
                        "--queue-args", "3=--micro-batch 4 --max-steps 2", "--out-root", "pilot/t",
                        "--", "--pilot", "--max-steps", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "done   GPU 2  grpo_lr5e-05_micro-batch-1" in out and "3 steps" in out
    assert "done   GPU 3  grpo_lr5e-05_micro-batch-4-max-steps-2" in out and "2 steps" in out
    assert (tmp_path / "pilot" / "t_grpo_lr5e-05_micro-batch-1" / "run_meta.json").exists()
    with pytest.raises(SystemExit, match="no --queue"):
        launch.main(["--queue", "2=grpo@5e-5", "--queue-args", "1=--micro-batch 4", "--", "--max-steps", "3"])


def test_refuses_busy_gpus_and_shared_out(monkeypatch):
    monkeypatch.setattr(launch, "nvidia_smi_busy", lambda gpus: {"1": ["pid 3002961 (40792 MiB)"]})
    with pytest.raises(SystemExit, match="GPU 1: pid 3002961"):
        launch.main(["--queue", "1=grpo@5e-5", "--", "--pilot", "--max-steps", "3"])
    with pytest.raises(SystemExit, match="--out-root"):
        launch.main(["--queue", "1=grpo@5e-5", "--", "--out", "x"])


def test_parse_job_with_config_and_seed():
    j = launch.parse_job("grpo@5e-5@m2-cvss@1", "2")
    assert (j.gpu, j.arm, j.lr, j.config, j.seed) == ("2", "grpo", 5e-5, "m2-cvss", 1)
    assert j.name == "grpo_m2-cvss_s1"
    assert j.args == ["--arm", "grpo", "--lr", "5e-05", "--config", "m2-cvss", "--seed", "1"]
    assert launch.parse_job("sft@1e-5").args == ["--arm", "sft", "--lr", "1e-05"]
    for bad in ("grpo@5e-5@m2-cvss", "grpo@5e-5@m9@1", "grpo@5e-5@m2-cvss@x", "grpo@fast@m2-cvss@1", "x@5e-5"):
        with pytest.raises(ValueError):
            launch.parse_job(bad)
    jobs = launch.parse_plan({"jobs": ["grpo@5e-05@m2-cvss@1", "sft@5e-05@m2-mcq@0"]})
    assert [j.name for j in jobs] == ["grpo_m2-cvss_s1", "sft_m2-mcq_s0"] and all(j.gpu == "" for j in jobs)
    with pytest.raises(ValueError, match="queued twice"):
        launch.check_names(launch.parse_plan({"jobs": ["sft@5e-05@m2-mcq@0", "sft@5e-05@m2-mcq@0"]}))
    with pytest.raises(ValueError, match="jobs"):
        launch.parse_plan({"queues": {}})


SHARED = textwrap.dedent('''
    """A stand-in for short.job: grpo is slow, distill_self fails on its first attempt, dpo always fails."""
    import argparse, json, os, sys, time
    from pathlib import Path
    ap = argparse.ArgumentParser()
    for a in ("--arm", "--lr", "--config", "--seed", "--out"):
        ap.add_argument(a)
    a, _ = ap.parse_known_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("gpu", os.environ["CUDA_VISIBLE_DEVICES"], a.config, a.seed, flush=True)
    if a.arm == "grpo":
        time.sleep(2.0)
    if a.arm == "dpo" or (a.arm == "distill_self" and not (out / "failed_once").exists()):
        (out / "failed_once").write_text("")
        print("RuntimeError: stage eval failed", file=sys.stderr)
        sys.exit(3)
    (out / "run_meta.json").write_text(json.dumps({"steps_done": 8, "gpu_hours": 0.01, "peak_memory_gb": 20.0}))
''')


def test_shared_queue_with_retries(tmp_path, monkeypatch, capsys):
    (tmp_path / "fake_job.py").write_text(SHARED)
    monkeypatch.setattr(launch, "REFRESH", 0.05)
    monkeypatch.setattr(launch, "nvidia_smi_busy", lambda gpus: {})
    monkeypatch.setattr(launch.pinned, "TRAIN_STEPS", 8)
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.chdir(tmp_path)
    plan = {"jobs": ["grpo@5e-05@m2-cvss@0", "sft@5e-05@m2-mcq@0", "sft@5e-05@m2-mcq@1", "distill_self@5e-05@m2-cvss@1",
                     "dpo@5e-05@m2-line_loc@0"]}
    (tmp_path / "queues.json").write_text(json.dumps(plan))
    code = launch.main(["--plan", "queues.json", "--gpus", "1", "2", "--retries", "1", "--runner", "fake_job",
                        "--out-dir", "runs"])
    out = capsys.readouterr().out
    assert code == 1                                                       # dpo failed twice
    assert "done   GPU 1  grpo_m2-cvss_s0" in out                          # the first job went to the first GPU
    assert "done   GPU 2  sft_m2-mcq_s0" in out and "done   GPU 2  sft_m2-mcq_s1" in out   # the free GPU took the next
    assert "distill_self_m2-cvss_s1" in out and "(attempt 2)" in out       # failed once, then done
    assert "FAILED GPU" in out and "dpo_m2-line_loc_s0" in out and "exit code 3: RuntimeError: stage eval failed" in out
    assert out.count("retried after a failure") == 2
    assert (tmp_path / "logs" / "dpo_m2-line_loc_s0.log").exists() and (tmp_path / "logs" / "dpo_m2-line_loc_s0.retry1.log").exists()
    assert (tmp_path / "runs" / "grpo_m2-cvss_s0" / "run_meta.json").exists()
    assert (tmp_path / "logs" / "sft_m2-mcq_s1.log").read_text().startswith("gpu 2 m2-mcq 1")


def test_plan_and_queue_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "nvidia_smi_busy", lambda gpus: {})
    (tmp_path / "q.json").write_text(json.dumps({"jobs": ["sft@5e-05@m2-mcq@0"]}))
    q = str(tmp_path / "q.json")
    for args, match in ((["--queue", "1=sft@5e-5", "--plan", q, "--gpus", "1"], "either"),
                        (["--plan", q], "either"), ([], "either"),
                        (["--plan", q, "--gpus", "1", "1"], "distinct"),
                        (["--plan", q, "--gpus", "1", "--queue-args", "1=--micro-batch 4"], "--queue-args"),
                        (["--queue", "1=sft@5e-5", "--out-root", "a", "--out-dir", "b"], "not both")):
        with pytest.raises(SystemExit, match=match):
            launch.main(args)
