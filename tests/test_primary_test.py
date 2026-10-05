"""Step 7: the frozen primary test (the plan's statistic, a parametric seed-bootstrap null) and the MDE simulation,
on synthetic M2 scores."""

import itertools
import json
import math
import random
from fractions import Fraction

import pytest

import analysis
from analysis import __main__ as cli
from analysis import mde
from analysis import primary_test as pt
from etl import pinned

CONT = ("mcq", "exact_id", "cvss", "line_loc")  # types whose items take any value in [0, 1]


def make_rows(score, arms=pinned.PRIMARY_ARMS, types=pinned.BANK_TYPES, n_cves=30, seeds=3):
    """score(arm, type, seed, cve_index, item_index) -> metric in [0, 1]."""
    rows = []
    for a, s, c, t in itertools.product(arms, range(seeds), range(n_cves), types):
        for i in ((0, 1) if t == "find_error" else (0,)):
            rows.append({"arm": a, "seed": s, "item_id": f"CVE-2022-{c:04d}:{t}:{i}", "metric": score(a, t, s, c, i)})
    return rows


def experiment(seed=0, n_cves=30, seeds=3, training_sd=0.02, arm_effect=0.0, interaction=0.0):
    """Item scores near 0.5: per-run training noise (arm x type x seed), shared item difficulty, item noise;
    optionally an arm main effect (grpo, every type) and an interaction (sft on cvss)."""
    rng = random.Random(seed)
    run = {k: rng.gauss(0, training_sd) for k in itertools.product(pinned.PRIMARY_ARMS, CONT, range(seeds))}
    difficulty = {k: rng.uniform(-0.15, 0.15) for k in itertools.product(CONT, range(n_cves))}
    noise = {k: rng.uniform(-0.15, 0.15) for k in itertools.product(pinned.PRIMARY_ARMS, CONT, range(seeds), range(n_cves))}

    def score(a, t, s, c, i):
        return 0.5 + run[(a, t, s)] + difficulty[(t, c)] + noise[(a, t, s, c)] + arm_effect * (a == "grpo") \
            + interaction * (a == "sft" and t == "cvss")
    return score


def scores_of(score, arms=pinned.PRIMARY_ARMS, n_cves=30, seeds=3):
    return pt.collect(make_rows(score, arms=arms, types=CONT, n_cves=n_cves, seeds=seeds), arms=arms, types=CONT)


# --- Collecting scores ------------------------------------------------------------------------------


def test_collect_pairs_find_error_and_rejects_bad_input():
    rows = make_rows(lambda a, t, s, c, i: 1 if (t != "find_error" or c % 2 == 0 or i == 0) else 0, n_cves=4, seeds=2)
    sc = pt.collect(rows + [dict(rows[0], arm="base")])  # other arms are ignored
    assert sc.values[("sft", "find_error", "0")] == (1.0, 0.0, 1.0, 0.0)  # paired: both labels right
    assert sc.values[("sft", "mcq", "0")] == (1.0,) * 4 and sc.seeds == ("0", "1")
    with pytest.raises(ValueError, match="missing"):
        pt.collect([r for r in rows if r["item_id"] != "CVE-2022-0001:find_error:1"])
    with pytest.raises(ValueError, match="duplicate"):
        pt.collect(rows + rows[:1])
    with pytest.raises(ValueError, match="outside"):
        pt.collect([dict(rows[0], metric="3/2")] + rows[1:])
    with pytest.raises(ValueError, match="two seeds"):
        pt.collect([r for r in rows if r["seed"] == 0])
    assert pt.collect(rows, exclude_cves=frozenset({"CVE-2022-0000"})).cves == ("CVE-2022-0001", "CVE-2022-0002", "CVE-2022-0003")
    with pytest.raises(ValueError, match="no such item index"):
        pt.collect(rows + [dict(rows[0], item_id="CVE-2022-0000:mcq:1")])
    cves = tuple(f"CVE-2022-{c:04d}" for c in range(4))
    assert pt.collect(rows, cves=cves, seeds=(0, 1)).cves == cves
    with pytest.raises(ValueError, match="cover 4 CVEs"):                       # a test CVE with no scores at all
        pt.collect(rows, cves=cves + ("CVE-2022-9999",), seeds=(0, 1))
    with pytest.raises(ValueError, match="seeds"):
        pt.collect(rows, cves=cves, seeds=(0, 1, 2))
    assert pt.collect(rows, exclude_cves=frozenset({"CVE-2022-0000"}), cves=cves, seeds=(0, 1)).cves == cves[1:]


def test_constant_column_is_decided_exactly():
    """Equal run means that are not exact binary floats (here k/7) must still give sd_t = 0, not an ulp."""
    for k in range(1, 7):
        sc = scores_of(lambda a, t, s, c, i, k=k: Fraction(k, 7) if t == "line_loc" else experiment(6)(a, t, s, c, i),
                       n_cves=7)
        assert pt.constant_column(sc, "line_loc") and not pt.constant_column(sc, "mcq")
        assert pt.fit(sc).excluded_zero_sd == ("line_loc",)


def test_pooled_sd_by_hand():
    # run means across seeds: sft (0.2, 0.4) -> variance 0.02; the other arms (0.5, 0.5) -> 0
    sc = scores_of(lambda a, t, s, c, i: (0.2 if s == 0 else 0.4) if a == "sft" else 0.5, n_cves=3, seeds=2)
    assert pt.pooled_sd(pt.run_means(sc), sc.arms, "mcq", sc.seeds) == pytest.approx(math.sqrt(0.02 / 4))


# --- The statistic ---------------------------------------------------------------------------------------


def naive_T(sc):
    """The plan's T written out directly: seed-averaged scores over the pooled seed SD, then the interaction."""
    arms, types, seeds, n = sc.arms, sc.types, sc.seeds, len(sc.cves)
    mean = {(a, t, s): sum(sc.values[(a, t, s)]) / n for a in arms for t in types for s in seeds}
    sd = {}
    for t in types:
        v = []
        for a in arms:
            mu = sum(mean[(a, t, s)] for s in seeds) / len(seeds)
            v.append(sum((mean[(a, t, s)] - mu) ** 2 for s in seeds) / (len(seeds) - 1))
        sd[t] = math.sqrt(sum(v) / len(v))
    m = {(a, t): sum(mean[(a, t, s)] for s in seeds) / len(seeds) / sd[t] for a in arms for t in types}
    g = sum(m.values()) / len(m)
    r = {a: sum(m[(a, t)] for t in types) / len(types) - g for a in arms}
    k = {t: sum(m[(a, t)] for a in arms) / len(arms) - g for t in types}
    return sum((m[(a, t)] - r[a] - k[t] - g) ** 2 for a in arms for t in types)


def test_T_matches_the_plan_written_out():
    sc = scores_of(experiment(1, interaction=0.05))
    out = pt.primary_test(sc, replicates=50)
    assert out["T"] == pytest.approx(naive_T(sc), rel=1e-12)
    assert sum(v for row in out["contributions"].values() for v in row.values()) == pytest.approx(1)


@pytest.mark.parametrize("shift", ["arm", "type", "arm_x_cve"])
def test_main_effects_leave_T_unchanged(shift):
    """The plan's counter-example: an arm one pooled SD above the rest in every column is no interaction."""
    f = experiment(2)
    base = scores_of(f)
    sd = {t: pt.pooled_sd(pt.run_means(base), base.arms, t, base.seeds) for t in CONT}
    bump = {"arm": lambda a, t, c: sd[t] * (a == "grpo"),
            "type": lambda a, t, c: 0.1 * (t == "cvss"),
            "arm_x_cve": lambda a, t, c: sd[t] * random.Random(f"{a}{c}").uniform(-1, 1)}[shift]
    shifted = scores_of(lambda a, t, s, c, i: f(a, t, s, c, i) + bump(a, t, c))
    x, y = pt.primary_test(base, replicates=500), pt.primary_test(shifted, replicates=500)
    assert y["T"] == pytest.approx(x["T"], rel=1e-9)
    assert y["pooled_seed_sd"] == pytest.approx(x["pooled_seed_sd"], rel=1e-9)
    assert abs(y["p"] - x["p"]) <= 0.05  # the null's additive fit moves with the main effect; T does not


# --- The null: replicates, calibration, power ---------------------------------------------------------


def test_p_counts_the_null_replicates():
    sc = scores_of(experiment(3))
    out = pt.primary_test(sc, replicates=300, seed=4)
    null = pt.null_statistics(pt.fit(sc), 300, 4)
    assert out["exceeding"] == sum(t >= out["T"] * (1 - pinned.PRIMARY_TIE_RTOL) for t in null)
    assert out["p"] == (1 + out["exceeding"]) / 301
    assert pt.primary_test(sc, replicates=300, seed=4) == out                 # reproducible
    assert pt.null_statistics(pt.fit(sc), 5, 5) != pt.null_statistics(pt.fit(sc), 5, 4)


def test_null_fit_is_additive():
    sc = scores_of(experiment(4, interaction=0.2))
    f = pt.fit(sc)
    m = [f.expected[(a, t)] / f.sd[t] for a in f.arms for t in f.types]
    assert max(abs(v) for v in pt.interaction(m, len(f.arms), len(f.types))) < 1e-9


@pytest.mark.parametrize("training_sd, arm_effect", [(0.0, 0.0), (0.02, 0.0), (0.04, 0.0), (0.02, 0.15)])
def test_null_is_calibrated(training_sd, arm_effect):
    """No interaction, at several levels of training-run noise and with a large arm main effect: ~5% rejected.
    (The plan's item permutation rejected about half of these at 2 points of training noise.)"""
    rejected = sum(pt.primary_test(scores_of(experiment(1000 + k, training_sd=training_sd, arm_effect=arm_effect)),
                                   replicates=199)["p"] < 0.05 for k in range(150))
    assert 1 <= rejected <= 17  # about 7.5 of 150 expected


def test_planted_interaction_is_detected_and_attributed():
    out = pt.primary_test(scores_of(experiment(5, training_sd=0.01, interaction=0.15)), replicates=999)
    assert out["p"] <= 0.005 and out["significant"]
    top = max(((a, t) for a in out["arms"] for t in out["types_tested"]), key=lambda k: out["contributions"][k[0]][k[1]])
    assert top == ("sft", "cvss")


def test_three_arms_zero_sd_columns_and_dropped_columns():
    f = experiment(6)
    three = scores_of(f, arms=("sft", "dpo", "grpo"))
    assert pt.primary_test(three, replicates=100)["arms"] == ["sft", "dpo", "grpo"]
    sc = scores_of(lambda a, t, s, c, i: 0.25 if t == "line_loc" else f(a, t, s, c, i))
    out = pt.primary_test(sc, replicates=100, flagged_cells=(("grpo", "cvss"),))
    assert out["types_excluded_zero_sd"] == ["line_loc"] and out["types_tested"] == ["mcq", "exact_id", "cvss"]
    assert out["pooled_seed_sd"]["line_loc"] == 0 and out["flagged_cells"] == [["grpo", "cvss"]]
    sens = pt.primary_test(sc, drop_types=("cvss",), replicates=100)
    assert sens["types_tested"] == ["mcq", "exact_id"] and sens["types_dropped"] == ["cvss"]
    with pytest.raises(ValueError, match="two columns"):
        pt.primary_test(sc, drop_types=("cvss", "exact_id"), replicates=10)


def test_no_interaction_at_all_gives_p_1():
    sc = scores_of(lambda a, t, s, c, i: 0.4 + 0.1 * s)  # every arm identical: T = 0, every T* ties or exceeds
    out = pt.primary_test(sc, replicates=50)
    assert out["T"] == pytest.approx(0, abs=1e-20) and out["p"] == 1.0


# --- MDE ------------------------------------------------------------------------------------------------


def test_quadratic_equals_direct_recomputation():
    f = pt.fit(scores_of(experiment(7)))
    y = pt.null_draw(f, random.Random(3))
    a, pm, c, sd = mde.quadratic(y, f)
    n_t = len(f.types)
    for (i, arm), (j, t), delta in itertools.product(enumerate(f.arms), enumerate(f.types), (0.0, 1.5, 7.0)):
        planted = {k: v + (delta / 100 if (k[0], k[1]) == (arm, t) else 0) for k, v in y.items()}
        x = delta / (100 * sd[t])
        assert a + 2 * pm[i * n_t + j] * x + c * x * x == pytest.approx(pt.statistic(planted, f.arms, f.types, f.seeds), rel=1e-9)


def test_p_value_matches_the_test():
    sc = scores_of(experiment(8, interaction=0.05))
    out = pt.primary_test(sc, replicates=400)
    null = sorted(pt.null_statistics(pt.fit(sc), 400, pinned.PRIMARY_SEED))
    assert mde.p_value(out["T"], null) == out["p"]


def test_power_is_near_alpha_at_zero_and_rises():
    out = mde.simulate(scores_of(experiment(9)), simulations=300, replicates=999)
    overall = [Fraction(p) for p in out["overall"]["power"]]
    assert overall[0] <= Fraction(1, 10) and overall[-1] == 1
    assert out["planted_interaction_factor"] == "9/16" and out["grid_points"][:3] == ["0", "1/4", "1/2"]
    assert out["overall"]["mde_points"] is not None and out["placements_per_simulation"] == 4


def test_mde_rule_needs_power_to_stay_above_target():
    pts = [Fraction(i) for i in range(5)]
    f = lambda *xs: [Fraction(x) for x in xs]  # noqa: E731
    assert mde.mde(pts, f(0, "4/5", "1/2", "9/10", 1)) == 3      # dips below after first reaching it
    assert mde.mde(pts, f("4/5", "4/5", 1, 1, 1)) == 0
    assert mde.mde(pts, f(0, 0, 1, 1, "1/2")) is None
    assert mde.grid()[-1] == pinned.MDE_GRID_MAX and len(mde.grid()) == 201


# --- Freeze and CLI --------------------------------------------------------------------------------------


def test_source_is_frozen():
    assert pinned.PRIMARY_TEST_SHA256 == analysis.source_sha256(), \
        "src/analysis/{primary_test,mde}.py changed: the primary test is frozen (step 7 decision record)"


def test_cli(tmp_path, monkeypatch):
    f = experiment(10)
    rows = make_rows(lambda a, t, s, c, i: f(a, t, s, c, i) if t in CONT else random.Random(f"{a}{s}{c}{i}").randint(0, 1))
    m2 = [dict(r, dropped_type=r["item_id"].split(":")[1]) for r in rows]
    m2 += [dict(r, dropped_type="mcq") for r in rows if not r["item_id"].endswith(":mcq:0")]  # diagnostics: ignored
    path = tmp_path / "scores.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in m2))
    split = tmp_path / "split.jsonl"
    split.write_text("".join(json.dumps({"cve_id": f"CVE-2022-{c:04d}", "pool": "test"}) + "\n" for c in range(30))
                     + json.dumps({"cve_id": "CVE-2019-0001", "pool": "nontest"}) + "\n")
    args = ["--scores", str(path), "--out", str(tmp_path), "--seeds", "0", "1", "2", "--split", str(split)]
    monkeypatch.setattr(pinned, "PRIMARY_TEST_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="frozen test has been changed"):
        cli.main(["primary-test", *args])
    monkeypatch.setattr(pinned, "PRIMARY_TEST_SHA256", analysis.source_sha256())
    monkeypatch.setattr(pinned, "PRIMARY_REPLICATES", 200)
    assert cli.main(["primary-test", *args, "--flagged", "grpo:cvss"]) == 0
    out = json.loads((tmp_path / "primary_test.json").read_text())
    assert out["primary"]["types_tested"] == list(pinned.BANK_TYPES) and out["primary"]["replicates"] == 200
    assert out["sensitivity_leave_flagged_columns_out"]["types_tested"] == ["mcq", "exact_id", "find_error", "line_loc"]
    monkeypatch.setattr(pinned, "MDE_SIMULATIONS", 5)
    assert cli.main(["mde", *args]) == 0
    assert json.loads((tmp_path / "mde.json").read_text())["simulations"] == 5
    with pytest.raises(SystemExit, match="--arms"):
        cli.main(["primary-test", *args, "--arms", "sft", "base"])
    with pytest.raises(ValueError, match="seeds"):
        cli.main(["primary-test", *args[:5], "0", "1", "--split", str(split)])
