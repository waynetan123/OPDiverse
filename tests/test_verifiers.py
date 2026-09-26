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
    assert set(verifiers.VERIFIERS) == {"exact_id", "cvss"}
