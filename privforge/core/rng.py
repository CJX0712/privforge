"""Deterministic RNG factory (SPEC.md section 5).

Nobody in this repository is allowed to call `np.random.*` directly: the
numpy 2.x `default_rng` stream and the legacy `RandomState` stream are not
interchangeable, so a bare global draw silently destroys reproducibility.

Every stream is *derived* from a master seed plus a stable tag, so adding a
new consumer never shifts an existing one.
Author: 晨星
"""

from __future__ import annotations

import hashlib

import numpy as np

from privforge.core.constants import SEED_MASK


def derive_seed(master: int, *tags: object) -> int:
    """Derive a stable non negative 63 bit seed from (master, tags).

    `hash()` is salted for strings in CPython, so we use blake2b instead.
    """
    payload = "|".join([str(master)] + [str(t) for t in tags]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & SEED_MASK


def make_rng(master: int, *tags: object) -> np.random.Generator:
    """Build a `Generator` deterministically derived from (master, tags)."""
    return np.random.default_rng(derive_seed(master, *tags))


def fork(rng: np.random.Generator, tag: object) -> np.random.Generator:
    """Fork an existing generator into a named child stream.

    The child is derived from a draw of the parent, so the parent state is
    advanced exactly once and never reused.
    """
    return np.random.default_rng(derive_seed(int(rng.integers(0, SEED_MASK)), tag))


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn `n` independent child generators (numpy >= 1.25 API)."""
    return list(rng.spawn(int(n)))


def shuffle_indices(rng: np.random.Generator, n: int) -> np.ndarray:
    """A permutation of `range(n)` drawn from `rng` (never np.random)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return rng.permutation(int(n))
