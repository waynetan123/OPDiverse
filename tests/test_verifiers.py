from fractions import Fraction

import pytest

from etl import verifiers
from etl.cwe_graph import load_cwe_graph
from etl.paths import DEFAULT

GOLD_VEC = "AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H"


@pytest.fixture(scope="module")
def graph():
    return load_cwe_graph(DEFAULT.cwe_xml)


# --- Exact-ID ----------------------------------------------------------------


@pytest.mark.parametrize("reply, parsed, strict", [
    ("CWE-787", "CWE-787", True),
    ("  CWE-787\n", "CWE-787", True),
    ("cwe 0787", "CWE-787", False),
    ("CWE_787.", "CWE-787", False),
    ("The weakness is CWE: 787 (Out-of-bounds Write).", "CWE-787", False),
    ("It is CWE-119... on reflection, CWE-787.", "CWE-787", False),  # the last ID wins
    ("CWE-0", "CWE-0", False),
])
def test_parse_cwe(reply, parsed, strict):
    assert verifiers.parse_cwe(reply) == parsed
    assert verifiers.strict_cwe(reply) is strict


@pytest.mark.parametrize("reply", [
    "I don't have information about that CVE.",
    "This is an out-of-bounds write.",          # a name without an ID
    "CVE-2022-1785",                            # a CVE ID is not a CWE ID
    "XCWE-787",                                  # must not be glued to a preceding word
    "",
])
def test_parse_cwe_failures(reply):
    assert verifiers.parse_cwe(reply) is None


def test_verify_exact_id(graph):
    v = verifiers.verify_exact_id("CWE-787", "CWE-787", graph)
    assert (v.parsed, v.parse_ok, v.strict_ok, v.metric, v.dense) == ("CWE-787", True, True, 1, 1)
    v = verifiers.verify_exact_id("A heap overflow (CWE-122).", "CWE-787", graph)
    assert (v.parsed, v.strict_ok, v.metric, v.dense) == ("CWE-122", False, 0, Fraction(1, 2))
    v = verifiers.verify_exact_id("CWE-119", "CWE-787", graph)
    assert (v.metric, v.dense) == (0, Fraction(1, 4))
    v = verifiers.verify_exact_id("CWE-399", "CWE-787", graph)  # a category parses but earns nothing
    assert (v.parse_ok, v.metric, v.dense) == (True, 0, 0)
    v = verifiers.verify_exact_id("no idea", "CWE-787", graph)
    assert v == verifiers.Verdict(None, False, False, Fraction(0), Fraction(0))


# --- CVSS --------------------------------------------------------------------


@pytest.mark.parametrize("reply, parsed, strict", [
    (GOLD_VEC, GOLD_VEC, True),
    ("CVSS:3.1/" + GOLD_VEC, GOLD_VEC, True),
    ("CVSS:3.0/" + GOLD_VEC, GOLD_VEC, True),
    (GOLD_VEC.lower(), GOLD_VEC, False),
    ("A:H/I:H/C:H/S:U/UI:N/PR:L/AC:L/AV:L", GOLD_VEC, False),        # any order parses; strict wants canonical
    ("Vector: AV:L / AC:L / PR:L / UI:N / S:U / C:H / I:H / A:H", GOLD_VEC, False),
    ("AV:N/AC:L", "AV:N/AC:L/PR:?/UI:?/S:?/C:?/I:?/A:?", False),      # partial
    ("AV:X/AC:L", "AV:?/AC:L/PR:?/UI:?/S:?/C:?/I:?/A:?", False),      # invalid value ignored
    ("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H then AV:L", "AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", False),  # last wins
    ("CVSS:4.0/" + GOLD_VEC, GOLD_VEC, False),
])
def test_parse_cvss(reply, parsed, strict):
    got = verifiers.parse_cvss(reply)
    assert "/".join(f"{k}:{got.get(k, '?')}" for k in ("AV", "AC", "PR", "UI", "S", "C", "I", "A")) == parsed
    assert verifiers.strict_cvss(reply) is strict


def test_ac_is_not_read_as_c():
    assert verifiers.parse_cvss("AC:L") == {"AC": "L"}
    assert verifiers.parse_cvss("PR:N UI:R") == {"PR": "N", "UI": "R"}


@pytest.mark.parametrize("reply", ["I don't know the vector.", "7.8 HIGH", "", "AV:X"])
def test_parse_cvss_failures(reply):
    assert verifiers.parse_cvss(reply) is None


def test_verify_cvss():
    v = verifiers.verify_cvss("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H", GOLD_VEC)
    assert (v.parse_ok, v.strict_ok, v.metric, v.dense) == (True, True, Fraction(5, 8), Fraction(5, 8))
    v = verifiers.verify_cvss("AV:L/AC:L", GOLD_VEC)  # missing metrics count as wrong
    assert v.metric == Fraction(2, 8) and v.parsed == "AV:L/AC:L/PR:?/UI:?/S:?/C:?/I:?/A:?"
    assert verifiers.verify_cvss("no idea", GOLD_VEC) == verifiers.Verdict(None, False, False, Fraction(0), Fraction(0))
    with pytest.raises(ValueError):
        verifiers.verify_cvss(GOLD_VEC, "AV:N/AC:L")


def test_scores_in_unit_interval(graph):
    replies = ["CWE-787", "CWE-119", "cwe 20", "x", "CWE-99999", "CWE-1187"]
    for r in replies:
        for gold in ("CWE-787", "CWE-125", "CWE-20"):
            v = verifiers.verify_exact_id(r, gold, graph)
            assert 0 <= v.metric <= 1 and 0 <= v.dense <= 1
    for r in (GOLD_VEC, "AV:N", "nothing", GOLD_VEC.lower()):
        v = verifiers.verify_cvss(r, GOLD_VEC)
        assert 0 <= v.metric <= 1 and v.metric == v.dense


def test_registry():
    assert set(verifiers.VERIFIERS) == {"mcq", "exact_id", "cvss", "find_error", "line_loc"}


# --- MCQ ---------------------------------------------------------------------


@pytest.mark.parametrize("reply, parsed, strict", [
    ("ANSWER: B", "B", True),
    ("  ANSWER: B\n", "B", True),
    ("answer: (C)", "C", False),
    ("**Answer:** D", "D", False),
    ("The flaw is a read, so ANSWER: A ... wait, ANSWER: C", "C", False),  # the last ANSWER: wins
    ("B", "B", False),
    ("(B).", "B", False),
    ("I think option D fits best.", "D", False),
    ("The answer is B because the read is out of bounds.", "B", False),
    ("Between (A) and (C), I pick (C).", "C", False),
])
def test_parse_mcq(reply, parsed, strict):
    assert verifiers.parse_mcq(reply) == parsed
    assert verifiers.strict_mcq(reply) is strict


@pytest.mark.parametrize("reply", [
    "answer: a buffer overflow",   # lower-case 'a' is an article, not an option
    "It is CWE-787.",
    "ANSWER: E",
    "",
])
def test_parse_mcq_failures(reply):
    assert verifiers.parse_mcq(reply) is None


def test_verify_mcq():
    assert verifiers.verify_mcq("ANSWER: B", "B") == verifiers.Verdict("B", True, True, Fraction(1), Fraction(1))
    assert verifiers.verify_mcq("ANSWER: A", "B").metric == 0
    assert verifiers.verify_mcq("no idea", "B") == verifiers.Verdict(None, False, False, Fraction(0), Fraction(0))


# --- Find-the-error ------------------------------------------------------------


@pytest.mark.parametrize("reply, label, cwe, strict", [
    ("VULNERABLE: yes, CWE-787", True, "CWE-787", True),
    ("VULNERABLE: no", False, None, True),
    ("vulnerable: YES (CWE-125)", True, "CWE-125", False),
    ("It copies without a bound. VULNERABLE: yes", True, None, False),
    ("VULNERABLE: yes, CWE-787 ... on reflection VULNERABLE: no", False, None, False),
    ("This function is not vulnerable.", False, None, False),
    ("There is no vulnerability here.", False, None, False),
    ("The function is vulnerable to an out-of-bounds read (CWE-125).", True, "CWE-125", False),
    ("It looked vulnerable at first, but it is not vulnerable.", False, None, False),
    ("VULNERABLE: no, CWE-787", False, None, False),
])
def test_parse_find_error(reply, label, cwe, strict):
    assert verifiers.parse_find_error(reply) == (label, cwe)
    assert verifiers.strict_find_error(reply) is strict


def test_parse_find_error_failure():
    assert verifiers.parse_find_error("The loop runs to len.") == (None, None)


def test_verify_find_error(graph):
    v = verifiers.verify_find_error("VULNERABLE: yes, CWE-787", True, "CWE-787", graph)
    assert (v.parsed, v.strict_ok, v.metric, v.dense) == ("VULNERABLE: yes, CWE-787", True, 1, 1)
    v = verifiers.verify_find_error("VULNERABLE: yes, CWE-119", True, "CWE-787", graph)
    assert (v.metric, v.dense) == (1, Fraction(1, 2) + Fraction(1, 8))  # label 1/2 + half of the 1/4 parent credit
    v = verifiers.verify_find_error("VULNERABLE: yes", True, "CWE-787", graph)
    assert (v.metric, v.dense) == (1, Fraction(1, 2))
    v = verifiers.verify_find_error("VULNERABLE: no", True, "CWE-787", graph)
    assert (v.metric, v.dense) == (0, 0)
    v = verifiers.verify_find_error("VULNERABLE: no", False, "CWE-787", graph)
    assert (v.metric, v.dense) == (1, 1)
    v = verifiers.verify_find_error("VULNERABLE: yes, CWE-787", False, "CWE-787", graph)
    assert (v.metric, v.dense) == (0, 0)  # a correct CWE earns nothing on the patched function


def test_paired_accuracy(graph):
    vy = verifiers.verify_find_error("VULNERABLE: yes, CWE-787", True, "CWE-787", graph)
    vy_wrong_cwe = verifiers.verify_find_error("VULNERABLE: yes, CWE-125", True, "CWE-787", graph)
    pn = verifiers.verify_find_error("VULNERABLE: no", False, "CWE-787", graph)
    py = verifiers.verify_find_error("VULNERABLE: yes, CWE-787", False, "CWE-787", graph)
    assert verifiers.paired_accuracy(vy, pn) == 1 and verifiers.paired_cwe_accuracy(vy, pn, "CWE-787") == 1
    assert verifiers.paired_accuracy(vy_wrong_cwe, pn) == 1 and verifiers.paired_cwe_accuracy(vy_wrong_cwe, pn, "CWE-787") == 0
    assert verifiers.paired_accuracy(vy, py) == 0  # always "vulnerable" scores 0 paired


# --- Line localisation ---------------------------------------------------------


@pytest.mark.parametrize("reply, parsed", [
    ("LINES: 12, 13, 17", [12, 13, 17]),
    ("LINES: 17, 12, 12", [12, 17]),
    ("LINES: none", []),
    ("The bug is CWE-787 on lines 3 and 5.", [3, 5]),            # 787 is out of range and dropped
    ("Lines 2 and 4 look off. LINES: 7", [7]),                    # only after the last LINES:
    ("LINES: 12-14", [12, 14]),                                   # no ranges: both endpoints, nothing between
    ("LINES: 0, 41", []),                                         # integers found but none in range: empty, not a failure
    ("I don't think any line is involved: none.", []),
])
def test_parse_lines(reply, parsed):
    assert verifiers.parse_lines(reply, 40) == parsed


@pytest.mark.parametrize("reply", ["I cannot tell.", "LINES:", ""])
def test_parse_lines_failures(reply):
    assert verifiers.parse_lines(reply, 40) is None


@pytest.mark.parametrize("reply, strict", [
    ("LINES: 1, 2, 40", True), ("LINES: none", True), ("LINES: 2, 1", False), ("LINES: 1, 1", False),
    ("LINES: 41", False), ("lines: 1", False), ("LINES: 1,2", False), ("LINES: 01", False),
])
def test_strict_lines(reply, strict):
    assert verifiers.strict_lines(reply, 40) is strict


@pytest.mark.parametrize("pred, gold, f1", [
    ([11, 12, 13], [12], Fraction(1, 2)),        # tripling the guess costs precision (plan's example)
    (list(range(1, 41)), [7, 30], Fraction(4, 42)),  # spray-all: ~0.095
    ([12, 13], [11, 12], Fraction(1)),           # a maximum matching, not exact-hits-first
    ([12], [12], Fraction(1)),
    ([13], [12], Fraction(1)),                   # +-1 tolerance
    ([14], [12], Fraction(0)),
    ([11, 13], [12], Fraction(2, 3)),            # one gold line is claimed once
    ([], [12], Fraction(0)),
    ([5, 6, 7], [4, 6, 8], Fraction(1)),
])
def test_line_f1(pred, gold, f1):
    assert verifiers.line_f1(pred, gold) == f1


def test_matcher_is_maximum():
    import random
    rng = random.Random(0)

    def brute(pred, gold):  # exhaustive maximum matching within +-1
        if not pred:
            return 0
        p, rest = pred[0], pred[1:]
        best = brute(rest, gold)
        for i, g in enumerate(gold):
            if abs(p - g) <= 1:
                best = max(best, 1 + brute(rest, gold[:i] + gold[i + 1:]))
        return best

    for _ in range(500):
        pred = sorted(rng.sample(range(1, 12), rng.randint(0, 6)))
        gold = sorted(rng.sample(range(1, 12), rng.randint(1, 6)))
        assert verifiers.matched_lines(pred, gold) == brute(pred, gold)


def test_verify_line_loc():
    v = verifiers.verify_line_loc("LINES: 12", [12], 40)
    assert (v.parsed, v.strict_ok, v.metric, v.dense) == ("LINES: 12", True, 1, 1)
    v = verifiers.verify_line_loc("LINES: none", [12], 40)
    assert (v.parsed, v.parse_ok, v.metric) == ("LINES: none", True, 0)
    assert verifiers.verify_line_loc("no idea", [12], 40).parse_ok is False
    with pytest.raises(ValueError):
        verifiers.verify_line_loc("LINES: 1", [], 40)


def test_new_scores_in_unit_interval(graph):
    for r in ("ANSWER: A", "B", "x"):
        v = verifiers.verify_mcq(r, "A")
        assert 0 <= v.metric <= 1 and 0 <= v.dense <= 1
    for r in ("VULNERABLE: yes, CWE-787", "VULNERABLE: yes, CWE-20", "VULNERABLE: yes", "VULNERABLE: no", "?"):
        for vulnerable in (True, False):
            v = verifiers.verify_find_error(r, vulnerable, "CWE-787", graph)
            assert 0 <= v.metric <= 1 and 0 <= v.dense <= 1
    for r in ("LINES: 1, 2, 3", "LINES: none", "LINES: " + ", ".join(map(str, range(1, 41))), "nothing"):
        v = verifiers.verify_line_loc(r, [2, 9], 40)
        assert 0 <= v.metric <= 1 and v.metric == v.dense


def test_verify_item_dispatch(graph):
    assert verifiers.verify_item("mcq", "ANSWER: C", {"letter": "C"}, graph).metric == 1
    assert verifiers.verify_item("exact_id", "CWE-787", {"cwe": "CWE-787"}, graph).metric == 1
    assert verifiers.verify_item("cvss", GOLD_VEC, {"vector": GOLD_VEC}, graph).metric == 1
    assert verifiers.verify_item("find_error", "VULNERABLE: no", {"vulnerable": False, "cwe": "CWE-787"}, graph).metric == 1
    assert verifiers.verify_item("line_loc", "LINES: 3", {"lines": [3], "n_lines": 9}, graph).metric == 1
    with pytest.raises(ValueError):
        verifiers.verify_item("explain", "x", {}, graph)


# --- Parser v2 (step 7) ----------------------------------------------------------


OPTIONS = [{"letter": "A", "cwe": "CWE-20", "name": "Improper Input Validation"},
           {"letter": "B", "cwe": "CWE-787", "name": "Out-of-bounds Write"},
           {"letter": "C", "cwe": "CWE-287", "name": "Improper Authentication"},
           {"letter": "D", "cwe": "CWE-665", "name": "Improper Initialization"}]


def test_parser_version():
    assert verifiers.PARSER_VERSION == "v2"


@pytest.mark.parametrize("reply, parsed", [
    ("CWE-287: Improper Authentication", "C"),                        # the option's text, without its letter
    ("D. CWE-665: Improper Initialization", "D"),                     # the whole option line
    ("The input is never checked.\n\nA. CWE-20: Improper Input Validation\n", "A"),
    ("ANSWER: B\nCWE-287: Improper Authentication", "B"),             # an ANSWER: field still wins
    ("CWE-287: Improper Authentication. It fits.", None),             # the last line must be exactly an option
    ("CWE-416: Use After Free", None),                                # not one of this item's options
    ("CWE-287", None),                                                # an ID alone is not an option line
])
def test_parse_mcq_option_text(reply, parsed):
    assert verifiers.parse_mcq(reply, OPTIONS) == parsed
    assert verifiers.parse_mcq("CWE-287: Improper Authentication") is None  # without options: v1 behaviour


def test_verify_item_mcq_option_text():
    gold = {"letter": "C", "cwe": "CWE-287", "options": OPTIONS}
    assert verifiers.verify_item("mcq", "CWE-287: Improper Authentication", gold, None) == \
        verifiers.Verdict("C", True, False, Fraction(1), Fraction(1))


@pytest.mark.parametrize("reply, parsed", [
    ("AV:Network/AC:Low/PR:None/UI:None/S:Unchanged/C:High/I:High/A:High", "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"),
    ("AV:NETWORK_/AC:LOW_/PR:NONE_/UI:REQUIRED/S:CHANGED/C:LOW/I:NONE/A:LOW", "AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:N/A:L"),
    ("AV:Adjacent Network/AC:H/PR:L/UI:N/S:U/C:N/I:N/A:H", "AV:A/AC:H/PR:L/UI:N/S:U/C:N/I:N/A:H"),
    ("AV: physical/AC:L/PR:H/UI:N/S:U/C:H/I:N/A:N", "AV:P/AC:L/PR:H/UI:N/S:U/C:H/I:N/A:N"),
    # words outside the specification never count
    ("AV:Networking/AC:Low/PR:Necessary/UI:Not Required/S:Single/C:N/I:N/A:H", "AV:?/AC:L/PR:?/UI:?/S:?/C:N/I:N/A:H"),
])
def test_parse_cvss_spec_words(reply, parsed):
    assert verifiers.verify_cvss(reply, "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H").parsed == parsed


def test_cvss_letters_unchanged_by_v2():
    assert verifiers.parse_cvss("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") == dict(
        zip(("AV", "AC", "PR", "UI", "S", "C", "I", "A"), "NLNNUHHH"))
    assert verifiers.parse_cvss("AV:NA_/AC:H") == {"AC": "H"}   # 'NA' is neither a letter nor a value name


@pytest.mark.parametrize("reply, parsed", [
    # an echoed numbered listing, cut off before any LINES: field, is not an answer
    ("Here is the fixed code:\n```\n1: int f(void)\n2: {\n3:     return g(4, 5);", None),
    ("Here is the fixed code:\n1: int f(void)\n2: {\n3:     return 0;\n4: }\nLine 3 changes.", [3]),
    ("The change:\n```c\nchar buf[16];\nmemcpy(buf, src, 8);\n```\nLines 12 and 14 are modified.", [12, 14]),
    ("```\nLINES: 7\n```", [7]),                                   # a LINES: field is read wherever it is
    ("Line 12 checks the length; line 17 frees it.", [12, 17]),
])
def test_parse_lines_ignores_echoed_code(reply, parsed):
    assert verifiers.parse_lines(reply, 40) == parsed
