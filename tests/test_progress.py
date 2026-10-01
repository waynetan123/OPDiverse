import io

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
    with Bar("mcq", 200, stream=out) as bar:
        bar.update(50, note="2 failed, will be re-sent")
        assert "[#######-----------------------]  25% 50/200" in bar.render()
        assert bar.render().endswith("2 failed, will be re-sent") and "left" in bar.render()
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
