"""The pre-registered primary test, frozen at step 7: an objective x question-type interaction on M2.

Pinned in docs/decisions/step7_decision_record.md. The statistic is the plan's ("Primary test,
pre-registered"); its null is a parametric seed bootstrap (owner, step 7), replacing the plan's
item-level permutation, which ignores training-run noise. With seed-to-seed noise of 2 points per cell,
that permutation called pure noise significant in about half of simulated experiments; this null held
its 5% rate at every noise level checked.

    y[a,t,s]  run (a, t, s)'s mean score over the test CVEs: arm a trained without type t, seed s, scored on
              t. A CVE's score is the item's metric; find-the-error's two items collapse to one paired score.
    sd_t      the pooled seed-level SD of column t: sqrt(mean over arms of the variance, ddof 1, across seeds
              of y[a,t,.]). A column with sd_t = 0 (decided exactly, from the scores' Fraction sums) is
              excluded and reported.
    m[a,t]    mean over seeds of y[a,t,s], divided by sd_t
    T         sum over cells of (P m)^2, P the double centring (subtract row and column means, add back the grand mean)

    Null: no interaction. The additive fit to m (grand + arm + type effects), times sd_t, gives each cell's
    expected run mean. Each of PRIMARY_REPLICATES replicates draws every run as that mean plus sd_t times a
    standard normal (random.Random(PRIMARY_SEED).gauss, in arm, type, seed order), re-estimates every sd_t,
    and recomputes T*. p = (1 + #{T* >= T * (1 - PRIMARY_TIE_RTOL)}) / (1 + PRIMARY_REPLICATES).

What it treats as fixed, stated in the write-up: the test CVEs (the conclusion is about this test set), and
the run-to-run noise as normal with one SD per column across arms.

Input rows are {"arm", "seed", "item_id", "metric"}, each an M2 score. Missing, duplicated or unexpected
scores fail loudly, as do CVEs or seeds other than the expected ones when those are given.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction

from etl import pinned


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    arms: tuple[str, ...]
    types: tuple[str, ...]
    seeds: tuple[str, ...]
    cves: tuple[str, ...]
    values: dict[tuple[str, str, str], tuple[float, ...]]  # (arm, type, seed) -> score per CVE, in `cves` order
    sums: dict[tuple[str, str, str], Fraction]              # (arm, type, seed) -> exact sum of those scores


def collect(rows: list[dict], arms: tuple[str, ...] = pinned.PRIMARY_ARMS, types: tuple[str, ...] = pinned.BANK_TYPES,
            exclude_cves: frozenset[str] = frozenset(), cves: tuple[str, ...] | None = None,
            seeds: tuple[str, ...] | None = None) -> Scores:
    """Item scores -> one score per (arm, type, seed, CVE). Rows for other arms or types are ignored. Given
    `cves` (the test CVEs) and `seeds`, the rows must cover exactly those, less `exclude_cves`."""
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError(f"the test needs at least two distinct arms, got {arms}")
    items: dict[tuple[str, str, str, str, int], Fraction] = {}
    for r in rows:
        cve, t, index = r["item_id"].split(":")
        if r["arm"] not in arms or t not in types or cve in exclude_cves:
            continue
        if int(index) not in ((0, 1) if t == "find_error" else (0,)):
            raise ValueError(f"{r['item_id']}: no such item index for {t}")
        key = (r["arm"], t, str(r["seed"]), cve, int(index))
        if key in items:
            raise ValueError(f"duplicate score for {key}")
        value = Fraction(r["metric"])
        if not 0 <= value <= 1:
            raise ValueError(f"score {value} for {key} is outside [0, 1]")
        items[key] = value
    found_cves, found_seeds = tuple(sorted({k[3] for k in items})), tuple(sorted({k[2] for k in items}))
    if cves is not None and found_cves != tuple(sorted(set(cves) - exclude_cves)):
        raise ValueError(f"the scores cover {len(found_cves)} CVEs, not the {len(set(cves) - exclude_cves)} expected")
    if seeds is not None and found_seeds != tuple(sorted(str(s) for s in seeds)):
        raise ValueError(f"the scores cover seeds {found_seeds}, not {tuple(sorted(str(s) for s in seeds))}")
    cves, seeds = found_cves, found_seeds
    if len(seeds) < 2:
        raise ValueError(f"the pooled seed SD needs at least two seeds, got {seeds}")
    values, sums, missing = {}, {}, []
    for a, t, s in itertools.product(arms, types, seeds):
        column = []
        for c in cves:
            got = [items.get((a, t, s, c, i)) for i in ((0, 1) if t == "find_error" else (0,))]
            if None in got:
                missing.append((a, t, s, c))
                continue
            column.append(Fraction(all(v == 1 for v in got)) if t == "find_error" else got[0])
        values[(a, t, s)] = tuple(float(v) for v in column)
        sums[(a, t, s)] = sum(column, Fraction(0))
    if missing:
        raise ValueError(f"{len(missing)} (arm, type, seed, CVE) scores are missing, e.g. {missing[:3]}")
    return Scores(tuple(arms), tuple(types), seeds, cves, values, sums)


def constant_column(scores: Scores, t: str) -> bool:
    """sd_t = 0 exactly: every arm's run means on t are equal across seeds (decided in exact arithmetic)."""
    return all(len({scores.sums[(a, t, s)] for s in scores.seeds}) == 1 for a in scores.arms)


def run_means(scores: Scores) -> dict[tuple[str, str, str], float]:
    n = len(scores.cves)
    return {k: math.fsum(v) / n for k, v in scores.values.items()}


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def pooled_sd(y: dict[tuple[str, str, str], float], arms: tuple[str, ...], t: str, seeds: tuple[str, ...]) -> float:
    """sqrt(mean over arms of the ddof-1 variance across seeds of the arm's run means on column t)."""
    variances = []
    for a in arms:
        means = [y[(a, t, s)] for s in seeds]
        mu = math.fsum(means) / len(means)
        variances.append(math.fsum((m - mu) ** 2 for m in means) / (len(means) - 1))
    return math.sqrt(math.fsum(variances) / len(variances))


def interaction(m: list[float], n_arms: int, n_types: int) -> list[float]:
    """P applied to an arm-major flattened cell matrix: row, column and grand means removed."""
    rows = [sum(m[a * n_types:(a + 1) * n_types]) / n_types for a in range(n_arms)]
    cols = [sum(m[a * n_types + t] for a in range(n_arms)) / n_arms for t in range(n_types)]
    grand = sum(m) / (n_arms * n_types)
    return [m[a * n_types + t] - rows[a] - cols[t] + grand for a in range(n_arms) for t in range(n_types)]


def standardised(y: dict, arms: tuple[str, ...], types: tuple[str, ...], seeds: tuple[str, ...]) -> tuple[list[float], dict[str, float]]:
    """(m flattened arm-major over `types`, sd per type): seed-averaged run means over the pooled seed SD."""
    sd = {t: pooled_sd(y, arms, t, seeds) for t in types}
    m = [math.fsum(y[(a, t, s)] for s in seeds) / len(seeds) / sd[t] for a in arms for t in types]
    return m, sd


def statistic(y: dict, arms: tuple[str, ...], types: tuple[str, ...], seeds: tuple[str, ...]) -> float:
    m, _ = standardised(y, arms, types, seeds)
    return sum(v * v for v in interaction(m, len(arms), len(types)))


def exceeds(t_star: float, t_obs: float) -> bool:
    return t_star >= t_obs * (1 - pinned.PRIMARY_TIE_RTOL)


# ---------------------------------------------------------------------------
# The null
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    arms: tuple[str, ...]
    types: tuple[str, ...]                 # the tested columns
    seeds: tuple[str, ...]
    y: dict[tuple[str, str, str], float]   # observed run means
    sd: dict[str, float]                   # every column's pooled seed SD
    excluded_zero_sd: tuple[str, ...]
    dropped: tuple[str, ...]
    expected: dict[tuple[str, str], float]  # the additive (null) fit, in score units


def fit(scores: Scores, drop_types: tuple[str, ...] = ()) -> Fit:
    y = run_means(scores)
    kept = [t for t in scores.types if t not in drop_types]
    sd = {t: 0.0 if constant_column(scores, t) else pooled_sd(y, scores.arms, t, scores.seeds) for t in scores.types}
    tested = tuple(t for t in kept if sd[t] > 0)
    if len(tested) < 2:
        raise ValueError(f"the interaction needs at least two columns; tested {tested}")
    m, _ = standardised(y, scores.arms, tested, scores.seeds)
    n_a, n_t = len(scores.arms), len(tested)
    residual = interaction(m, n_a, n_t)
    expected = {(a, t): (m[i * n_t + j] - residual[i * n_t + j]) * sd[t]
                for i, a in enumerate(scores.arms) for j, t in enumerate(tested)}
    return Fit(arms=scores.arms, types=tested, seeds=scores.seeds, y=y, sd=sd,
               excluded_zero_sd=tuple(t for t in kept if sd[t] == 0), dropped=tuple(drop_types), expected=expected)


def null_draw(f: Fit, rng: random.Random) -> dict[tuple[str, str, str], float]:
    """One simulated experiment under the null: the additive fit plus Gaussian run noise at each column's SD."""
    return {(a, t, s): f.expected[(a, t)] + f.sd[t] * rng.gauss(0.0, 1.0) for a in f.arms for t in f.types for s in f.seeds}


def null_statistics(f: Fit, replicates: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [statistic(null_draw(f, rng), f.arms, f.types, f.seeds) for _ in range(replicates)]


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def primary_test(scores: Scores, drop_types: tuple[str, ...] = (), replicates: int | None = None,
                 seed: int | None = None, flagged_cells: tuple[tuple[str, str], ...] = ()) -> dict:
    """The frozen test. `drop_types` gives the leave-flagged-columns-out sensitivity run; `flagged_cells`
    (floored or substituted (arm, type) cells) are only marked in the output."""
    count = pinned.PRIMARY_REPLICATES if replicates is None else replicates
    seed = pinned.PRIMARY_SEED if seed is None else seed
    f = fit(scores, drop_types)
    n_a, n_t = len(f.arms), len(f.types)
    m, _ = standardised(f.y, f.arms, f.types, f.seeds)
    cells = interaction(m, n_a, n_t)
    t_obs = sum(v * v for v in cells)
    hits = sum(exceeds(t_star, t_obs) for t_star in null_statistics(f, count, seed))
    p = (1 + hits) / (1 + count)
    grid = lambda vals: {a: {t: vals[i * n_t + j] for j, t in enumerate(f.types)} for i, a in enumerate(f.arms)}  # noqa: E731
    seed_means = [math.fsum(f.y[(a, t, s)] for s in f.seeds) / len(f.seeds) for a in f.arms for t in f.types]
    return {
        "T": t_obs,
        "p": p,
        "alpha": pinned.PRIMARY_ALPHA,
        "significant": p < pinned.PRIMARY_ALPHA,
        "null": "parametric seed bootstrap",
        "replicates": count,
        "exceeding": hits,
        "seed": seed,
        "arms": list(f.arms),
        "types_tested": list(f.types),
        "types_dropped": list(f.dropped),
        "types_excluded_zero_sd": list(f.excluded_zero_sd),
        "n_cves": len(scores.cves),
        "seeds": list(f.seeds),
        "pooled_seed_sd": f.sd,
        "run_means": {f"{a}|{t}|{s}": v for (a, t, s), v in sorted(f.y.items())},
        "cell_means": grid(seed_means),
        "interaction_residuals": grid(cells),
        "contributions": grid([v * v / t_obs if t_obs > 0 else 0.0 for v in cells]),
        "flagged_cells": [list(c) for c in flagged_cells],
    }
