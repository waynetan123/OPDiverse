import io
from types import SimpleNamespace

from generators import external
from generators.bank import build
from generators.progress import Bar, clock, track
from test_generators_bank import data, make_data, run  # noqa: F401  (fixture and helpers)


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_clock():
    assert (clock(0), clock(75), clock(3725)) == ("0:00", "1:15", "1:02:05")


def test_bar_renders_progress_and_note():
    out = Terminal()
    with Bar("MCQ batch", 200, stream=out) as bar:
        bar.update(50, note="in_progress: 50 ok")
        assert "[#######-----------------------]  25% 50/200" in bar.render()
        assert bar.render().endswith("in_progress: 50 ok") and "left" in bar.render()
    text = out.getvalue()
    assert text.endswith("\n") and text.count("\r") >= 2  # drawn on entry, redrawn, closed with a newline
    assert "25% 50/200" in text.splitlines()[-1]


def test_track_yields_everything_and_finishes_at_total():
    out = Terminal()
    assert list(track(range(7), "Items", stream=out)) == list(range(7))
    assert out.getvalue().rstrip("\n").endswith("7/7  0:00")


def test_no_terminal_no_output():
    out = io.StringIO()  # not a TTY: tests, logs and redirected runs stay clean
    assert list(track([1, 2, 3], "Items", stream=out)) == [1, 2, 3]
    with Bar("x", 5, stream=out) as bar:
        bar.update(3)
    assert out.getvalue() == ""


def test_overshoot_and_empty_total_are_safe():
    bar = Bar("x", 0, stream=Terminal())
    bar.update(5)
    assert " 100% 0/0" in bar.render()
    bar = Bar("x", 3, stream=Terminal())
    bar.update(10)
    assert bar.done == 3


def test_wait_for_batch_polls_until_ended(capsys):
    states = [("in_progress", 0, 0), ("in_progress", 40, 2), ("ended", 97, 3)]

    class Batches:
        calls = 0

        def retrieve(self, batch_id):
            status, ok, failed = states[min(self.calls, len(states) - 1)]
            self.calls += 1
            counts = SimpleNamespace(processing=100 - ok - failed, succeeded=ok, errored=failed, canceled=0, expired=0)
            return SimpleNamespace(id=batch_id, processing_status=status, request_counts=counts)

    batches = Batches()
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    slept = []
    batch = external.wait_for_batch(client, "msgbatch_x", 100, label="mcq batch", poll_seconds=3, sleep=slept.append)
    assert batch.processing_status == "ended" and batches.calls == 3
    assert len(slept) == 6  # 3 one-second ticks between each of the two waits
    assert external.finished(batch.request_counts) == 100
    # Without a terminal (as under pytest), one status line per poll instead of a bar.
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["  in_progress: 0 ok (0/100 done)", "  in_progress: 40 ok, 2 failed (42/100 done)",
                     "  ended: 97 ok, 3 failed (100/100 done)"]


def test_build_output_is_identical_with_bars_drawn(data, monkeypatch, tmp_path):  # noqa: F811
    """Drawing bars must never change an output byte."""
    assert run(data, "build", "--dry-mcq") == 0
    files = build.BankFiles.of(data, dry=True)
    quiet = {p: p.read_bytes() for p in (files.bank("nontest"), files.bank("test"), files.mcq_decisions, files.meta)}
    monkeypatch.setattr("sys.stderr", Terminal())
    assert run(data, "build", "--dry-mcq") == 0
    import sys
    assert "Question items [" in sys.stderr.getvalue()
    assert {p: p.read_bytes() for p in quiet} == quiet
