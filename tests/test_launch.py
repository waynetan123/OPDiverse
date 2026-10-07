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


def test_total_steps(monkeypatch):
    assert launch.total_steps(["--pilot", "--max-steps", "20"]) == 20
    assert launch.total_steps(["--max-steps=8"]) == 8
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
    a = ap.parse_args()
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


def test_refuses_busy_gpus_and_shared_out(monkeypatch):
    monkeypatch.setattr(launch, "nvidia_smi_busy", lambda gpus: {"1": ["pid 3002961 (40792 MiB)"]})
    with pytest.raises(SystemExit, match="GPU 1: pid 3002961"):
        launch.main(["--queue", "1=grpo@5e-5", "--", "--pilot", "--max-steps", "3"])
    with pytest.raises(SystemExit, match="--out-root"):
        launch.main(["--queue", "1=grpo@5e-5", "--", "--out", "x"])
