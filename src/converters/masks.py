"""The matrix configurations and their item selections: which train items each configuration keeps, how many
copies of each, and in what order. Pure; deterministic in (item_id, configuration, seed), so every arm of one
configuration and seed gets the identical rows."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

from etl import pinned


def m1v_name(f: Fraction) -> str:
    return f"m1v-{f.numerator}of{f.denominator}"


M1 = "m1"
M2 = tuple(f"m2-{t}" for t in pinned.BANK_TYPES)
M1V = tuple(m1v_name(f) for f in pinned.M1V_FRACTIONS)
CONFIGS = (M1, *M2, *M1V)
# Items per CVE of each type: find_error has two (0 vulnerable, 1 patched), the others one.
PER_CVE = {"mcq": 1, "exact_id": 1, "cvss": 1, "find_error": 2, "line_loc": 1}
assert sum(PER_CVE.values()) == pinned.ITEMS_PER_CVE and tuple(PER_CVE) == pinned.BANK_TYPES


def dropped_type(config: str) -> str | None:
    return config[3:] if config.startswith("m2-") else None


def m1v_fraction(config: str) -> Fraction | None:
    if not config.startswith("m1v-"):
        return None
    num, den = config[4:].split("of")
    return Fraction(int(num), int(den))


def comparator(config: str) -> str | None:
    """The M1-volume configuration that removes the same share of items as an M2 loop."""
    t = dropped_type(config)
    return None if t is None else m1v_name(Fraction(PER_CVE[t], pinned.ITEMS_PER_CVE))


def dev_types(config: str) -> list[str]:
    """Types checkpoint selection reads on dev: M2 drops its type from dev too (the plan's 'Why dev drops the
    type too'); M1 and M1-volume keep all five."""
    return [t for t in pinned.BANK_TYPES if t != dropped_type(config)]


def arms(config: str) -> tuple[str, ...]:
    return pinned.CONVERTER_ARMS if config == M1 else pinned.QUESTION_ARMS


def _rank(item_id: str, salt: str, *parts: str) -> tuple[int, str]:
    return pinned.stable_rank(item_id, pinned.CONVERTER_SALTS[salt], *parts), item_id


def masked(items: list[tuple[str, str]], config: str, seed: int) -> set[str]:
    """The item_ids a configuration removes from the seed's train items, given as (item_id, type)."""
    t = dropped_type(config)
    if t is not None:
        return {i for i, it in items if it == t}
    f = m1v_fraction(config)
    if f is None:
        return set()
    k = len(items) * f
    if k.denominator != 1:
        raise ValueError(f"{config}: {len(items)} items x {f} is not a whole number")
    return set(sorted((i for i, _ in items), key=lambda i: _rank(i, "mask", str(f), str(seed)))[:int(k)])


def copies(kept: list[str], n: int, config: str, seed: int) -> dict[str, int]:
    """Upsample the kept items back to n rows: each gets n // len(kept) copies, and the n % len(kept) with the
    lowest rank one more."""
    base, extra = divmod(n, len(kept))
    plus = set(sorted(kept, key=lambda i: _rank(i, "upsample", config, str(seed)))[:extra])
    return {i: base + (i in plus) for i in kept}


def selection(items: list[tuple[str, str]], config: str, seed: int) -> list[tuple[str, int]]:
    """(item_id, copy) rows for one configuration and seed, in the file's fixed shuffled order. `items` are the
    seed's train items as (item_id, type), in bank order. Every configuration has exactly len(items) rows."""
    gone = masked(items, config, seed)
    kept = [i for i, _ in items if i not in gone]
    n = copies(kept, len(items), config, seed)
    rows = [(i, c) for i in kept for c in range(n[i])]
    rows.sort(key=lambda ic: (pinned.stable_rank(ic[0], str(ic[1]), pinned.CONVERTER_SALTS["order"], config, str(seed)), ic))
    assert len(rows) == len(items)
    return rows


def composition(rows: list[tuple[str, int]]) -> Counter:
    return Counter(i.split(":")[1] for i, _ in rows)
