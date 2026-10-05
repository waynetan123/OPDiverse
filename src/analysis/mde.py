"""Minimum detectable effect, frozen at step 7: how large an effect could the frozen primary test have caught?

Pinned in docs/decisions/step7_decision_record.md (owner: a single-cell effect):

- Simulated experiments come from the primary test's own null: the additive fit to the observed M2 run
  means plus Gaussian run noise at each column's pooled seed SD (random.Random(MDE_SEED)).
- Planted effect: delta points added to every run of one (arm, type) cell. That shifts the cell's seed
  mean and leaves every seed SD unchanged, so with the experiment's own SDs the statistic is exactly
  T(delta) = a + 2 b x + c x^2, with x = delta / (100 sd_t). Each grid point is evaluated from that.
- Each simulated experiment's p is read against the frozen test's null replicates (PRIMARY_REPLICATES from
  PRIMARY_SEED, computed once from the observed fit), with the test's tie rule and p formula.
- Power at delta = the share of MDE_SIMULATIONS x arms (the effect placed on each arm in turn) with
  p < PRIMARY_ALPHA. Over the grid 0, MDE_GRID_STEP, ..., MDE_GRID_MAX points, the MDE is the smallest
  delta from which power stays at least MDE_POWER; none if it never does. Reported per type, and overall
  from the type-averaged power curve. The interaction a planted delta puts in its cell is
  (1 - 1/arms)(1 - 1/types) of it.
"""

from __future__ import annotations

import bisect
import random
from fractions import Fraction

from etl import pinned

from .primary_test import Fit, Scores, fit, interaction, null_draw, null_statistics, standardised


def grid() -> list[Fraction]:
    steps = int(pinned.MDE_GRID_MAX / pinned.MDE_GRID_STEP)
    return [pinned.MDE_GRID_STEP * i for i in range(steps + 1)]


def mde(points: list[Fraction], power: list[Fraction], target: Fraction = pinned.MDE_POWER) -> Fraction | None:
    """The smallest grid point from which power stays >= target, or None."""
    below = [i for i, p in enumerate(power) if p < target]
    last = below[-1] if below else -1
    return None if last == len(points) - 1 else points[last + 1]


def quadratic(y: dict, f: Fit) -> tuple[float, list[float], float, dict[str, float]]:
    """(a, P m, c, sd) for one experiment: planting x (standardised) in cell (i, j) gives T = a + 2 (P m)[i, j] x + c x^2."""
    n_a, n_t = len(f.arms), len(f.types)
    m, sd = standardised(y, f.arms, f.types, f.seeds)
    pm = interaction(m, n_a, n_t)
    return sum(v * v for v in pm), pm, (1 - 1 / n_a) * (1 - 1 / n_t), sd


def p_value(t: float, null_sorted: list[float]) -> float:
    """The frozen test's p for statistic t: (1 + #{T* >= t (1 - PRIMARY_TIE_RTOL)}) / (1 + replicates)."""
    hits = len(null_sorted) - bisect.bisect_left(null_sorted, t * (1 - pinned.PRIMARY_TIE_RTOL))
    return (1 + hits) / (1 + len(null_sorted))


def simulate(scores: Scores, simulations: int | None = None, replicates: int | None = None,
             seed: int | None = None, sim_seed: int | None = None) -> dict:
    sims = pinned.MDE_SIMULATIONS if simulations is None else simulations
    count = pinned.PRIMARY_REPLICATES if replicates is None else replicates
    f = fit(scores)
    n_a, n_t = len(f.arms), len(f.types)
    null_sorted = sorted(null_statistics(f, count, pinned.PRIMARY_SEED if seed is None else seed))
    points = grid()
    detected = [[0] * len(points) for _ in f.types]
    rng = random.Random(pinned.MDE_SEED if sim_seed is None else sim_seed)
    for _ in range(sims):
        a, pm, c, sd = quadratic(null_draw(f, rng), f)
        for j, t in enumerate(f.types):
            for i in range(n_a):
                b = pm[i * n_t + j]
                for k, d in enumerate(points):
                    x = float(d) / (100 * sd[t])
                    detected[j][k] += p_value(a + 2 * b * x + c * x * x, null_sorted) < pinned.PRIMARY_ALPHA
    trials = sims * n_a
    power = {t: [Fraction(v, trials) for v in detected[j]] for j, t in enumerate(f.types)}
    overall = [sum((power[t][k] for t in f.types), Fraction(0)) / n_t for k in range(len(points))]
    headroom = {t: 100 * (1 - max(sum(f.y[(a, t, s)] for s in f.seeds) / len(f.seeds) for a in f.arms)) for t in f.types}
    fmt = lambda v: None if v is None else str(v)  # noqa: E731
    return {
        "simulations": sims,
        "placements_per_simulation": n_a,
        "replicates": count,
        "seed": pinned.PRIMARY_SEED if seed is None else seed,
        "sim_seed": pinned.MDE_SEED if sim_seed is None else sim_seed,
        "alpha": pinned.PRIMARY_ALPHA,
        "target_power": str(pinned.MDE_POWER),
        "arms": list(f.arms),
        "types": list(f.types),
        "types_excluded_zero_sd": list(f.excluded_zero_sd),
        "grid_points": [str(d) for d in points],
        "planted_interaction_factor": str(Fraction(n_a - 1, n_a) * Fraction(n_t - 1, n_t)),
        "per_type": {t: {"pooled_seed_sd": f.sd[t], "headroom_points": headroom[t],
                         "power": [str(v) for v in power[t]], "mde_points": fmt(mde(points, power[t]))} for t in f.types},
        "overall": {"power": [str(v) for v in overall], "mde_points": fmt(mde(points, overall))},
    }
