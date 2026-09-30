"""SOP delivery gate tests (random-ai-system-delivery, SOP section 5/8).

These are the *engineering* invariants the SOP requires IN ADDITION to the
math invariants in ``test_invariants.py``: import hygiene, reproducibility of
the seed protocol, the three silent-failure hard constraints (no ``mu*C_t``
coupling, scale invariance, no epsilon split in ensembles), and privacy-leak
hygiene.

Author: 晨星
"""

from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from privforge.core.errors import CompositionError
from privforge.core.rng import derive_seed, make_rng
from privforge.core.types import Bounds, Budget, DPFitResult, EvalResult
from privforge.data.loader import from_arrays, train_test_split
from privforge.data.synthetic import generate_dataset
from privforge.domain.accountant import RDPAccountant
from privforge.eval.protocol import evaluate_dod
from privforge.training.adaclip_budget import AdaClipBudget, AdaClipConfig
from privforge.training.dp_sgd import train_dpsgd
from privforge.training.ensemble import assert_no_epsilon_split
from privforge.training.logistic import rescale_R


# --------------------------------------------------------------------------- #
# Import hygiene
# --------------------------------------------------------------------------- #
def test_import_direction():
    """Every public sub-package imports cleanly from a fresh interpreter."""
    for mod in [
        "privforge",
        "privforge.core",
        "privforge.data",
        "privforge.domain",
        "privforge.training",
        "privforge.eval",
        "privforge.pipeline",
        "privforge.backends",
        "privforge.hpo",
    ]:
        importlib.import_module(mod)


def test_no_legacy_numpy_aliases():
    """No numpy 2.x-removed legacy aliases (np.float/np.int/np.bool/...) anywhere."""
    root = Path(__file__).resolve().parent.parent / "privforge"
    bad = [
        r"np\.float\b",
        r"np\.int\b",
        r"np\.bool\b",
        r"np\.object\b",
        r"np\.str\b",
        r"np\.long\b",
        r"np\.unicode\b",
        r"np\.complex\b",
        r"np\.bool8\b",
        r"np\.unicode_\b",
        r"np\.string_\b",
    ]
    pat = re.compile("|".join(bad))
    offenders = []
    for p in root.rglob("*.py"):
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            offenders.append((p.name, m.group(0)))
    assert not offenders, f"legacy numpy aliases found: {offenders}"


def test_diffprivlib_importable():
    """The reference library we cross-check against is importable."""
    import diffprivlib
    from diffprivlib.mechanisms import GaussianAnalytic

    assert diffprivlib is not None
    assert GaussianAnalytic is not None


def test_default_config_no_unsafe():
    """The flagship default must NOT use the unsafe mu*C_t ablation path."""
    assert AdaClipConfig().unsafe_mu_times_C is False


# --------------------------------------------------------------------------- #
# Silent-failure hard constraint #1: ledger never couples to the clip norm
# --------------------------------------------------------------------------- #
def test_no_mu_times_C():
    """With unit-sensitivity normalization the RDP ledger depends on mu only.

    Two otherwise-identical runs that differ only in the fixed clip norm ``C_t``
    must produce byte-identical ledgers.  If the released noise scale were
    ``mu_t * C_t`` the two ledgers would diverge -- so this is a direct
    behavioural check that the hard constraint is enforced by the shipped path.
    """
    ds = generate_dataset("medium")
    R = float(ds.bounds.x_norm_max)
    Xs = rescale_R(ds.X, R)
    y = np.asarray(ds.y, dtype=np.float64)
    rng1 = make_rng(11, "cmp", "medium")
    rng2 = make_rng(11, "cmp", "medium")
    r1 = train_dpsgd(
        Xs, y, q=0.1, T=15, eps_q_per_step=0.0, mu=0.5, fixed_C=0.3, clip_mode="fixed", rng=rng1
    )
    r2 = train_dpsgd(
        Xs, y, q=0.1, T=15, eps_q_per_step=0.0, mu=0.5, fixed_C=3.0, clip_mode="fixed", rng=rng2
    )
    assert np.allclose(r1.accountant.eps_alpha, r2.accountant.eps_alpha, atol=1e-12)


# --------------------------------------------------------------------------- #
# Silent-failure hard constraint #2: scale invariance
# --------------------------------------------------------------------------- #
def test_scaling_invariance():
    """Multiplying features and the public bound by the same factor is a no-op.

    The pipeline re-scales by ``R`` so the working features are ``X/R``; doubling
    both ``X`` and ``R`` leaves ``X/R`` byte-identical. The stored coefficient
    ``w = theta/R`` therefore scales with ``1/R``, but the *model's predictions*
    (and hence its accuracy) are exactly invariant -- which is the real invariant.
    """
    ds = generate_dataset("medium")
    R = float(ds.bounds.x_norm_max)
    train, test = train_test_split(ds, seed=1)
    s = 2.0
    # scaled dataset: scale X and the bound by s; working features X/R identical
    X2 = ds.X * s
    b2 = Bounds(x_norm_max=R * s, y_abs_max=1.0)
    ds2 = from_arrays(X2, ds.y, name="medium_scaled", bounds=b2)
    train2, test2 = train_test_split(ds2, seed=1)
    b = Budget.from_fractions(epsilon=1.0, delta=1e-5)
    cfg = AdaClipConfig(T=20, q=0.1, mu_schedule="uniform")
    m1 = AdaClipBudget(cfg).fit(train, b, make_rng(1, "sc", "medium"))
    m2 = AdaClipBudget(cfg).fit(train2, b, make_rng(1, "sc", "medium"))
    # predictions on the corresponding folds must be byte-identical (invariance)
    assert m1.accuracy(test.X, test.y) == m2.accuracy(test2.X, test2.y)


# --------------------------------------------------------------------------- #
# Silent-failure hard constraint #3: ensembles never split one epsilon
# --------------------------------------------------------------------------- #
def test_ensemble_never_splits_epsilon():
    """Two independently-spent DP releases must be rejected for ensembling."""
    w = np.zeros(5)
    b1 = Budget.from_fractions(1.0, 1e-5)
    b2 = Budget.from_fractions(1.0, 1e-5)
    m_same1 = DPFitResult(w=w, budget_spent=b1)
    m_same2 = DPFitResult(w=w, budget_spent=b1)  # identical ledger object
    m_diff = DPFitResult(w=w, budget_spent=b2)  # distinct ledger object
    # distinct ledgers -> CompositionError (E401)
    with pytest.raises(CompositionError):
        assert_no_epsilon_split([m_same1, m_diff])
    # identical ledger -> allowed (post-processing of one release)
    assert_no_epsilon_split([m_same1, m_same2])


# --------------------------------------------------------------------------- #
# Seed protocol: paired, tag-sensitive, blake2b-derived
# --------------------------------------------------------------------------- #
def test_seed_protocol_is_paired():
    s1 = derive_seed(123, "a", "medium")
    s2 = derive_seed(123, "a", "medium")
    s3 = derive_seed(123, "b", "medium")
    assert s1 == s2
    assert s1 != s3
    # derived from blake2b, never hash() -- so a stable non-negative int
    assert isinstance(s1, int) and s1 >= 0 and s1 < (1 << 63)
    # make_rng is reproducible
    r1 = make_rng(7, "x")
    r2 = make_rng(7, "x")
    assert r1.integers(0, 1 << 30) == r2.integers(0, 1 << 30)


# --------------------------------------------------------------------------- #
# Poisson subsampling amplifies privacy (smaller epsilon than full-batch)
# --------------------------------------------------------------------------- #
def test_dpsgd_epsilon_amplified_lt_naive():
    q, mu, T, delta = 0.1, 0.5, 10, 1e-5
    amp = RDPAccountant()
    for _ in range(T):
        amp.step(q, mu)  # Poisson subsampled
    naive = RDPAccountant()
    for _ in range(T):
        naive.step(1.0, mu)  # full-batch, no amplification
    assert amp.epsilon(delta) < naive.epsilon(delta)


# --------------------------------------------------------------------------- #
# Privacy-leak hygiene: a normal fit must not emit a PrivacyLeakWarning
# --------------------------------------------------------------------------- #
def test_no_privacy_leak_warning(tiny_ds):
    train, _ = tiny_ds
    from diffprivlib.utils import PrivacyLeakWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", PrivacyLeakWarning)
        m = AdaClipBudget(AdaClipConfig(T=10, q=0.1)).fit(
            train, Budget.from_fractions(1.0, 1e-5), make_rng(1, "w", "medium")
        )
    assert m is not None


# --------------------------------------------------------------------------- #
# DoD gate evaluation on a multi-dataset epsilon sweep (regression)
# --------------------------------------------------------------------------- #
def _mk_row(ds: str, method: str, eps: float, seed: int, acc: float, gap: float) -> EvalResult:
    return EvalResult(
        dataset=ds,
        method=method,
        epsilon=eps,
        seed=seed,
        accuracy=acc,
        auc=acc,
        macro_f1=acc,
        eps_spent=eps,
        utility_gap=gap,
        ugc=0.5,
        uac=0.0,
    )


def test_dod_evaluation_handles_multi_dataset_sweep():
    """Two regressions locked in one test.

    1. ``evaluate_dod`` crashed with KeyError on multi-dataset input: the dead
       ``eps_sets`` dict was rebuilt per result and kept only the last dataset
       per method.
    2. DoD-2/3/4 were epsilon-blind (best baseline taken across ALL epsilons),
       a category error: an eps=0.05 flagship scoring below an eps=10 baseline
       is a different privacy price, not a regression. The best-base lookup is
       now keyed by (dataset, seed, epsilon).
    """
    ceiling = 0.77  # dataset-level non-private accuracy, same for every method
    rows = []
    for ds in ("medium", "heavy_tail"):
        for seed in (1, 2):
            for eps, f_acc, b_acc in (
                (0.05, 0.52, 0.51),
                (0.5, 0.75, 0.66),
                (10.0, 0.76, 0.74),
            ):
                rows.append(_mk_row(ds, "AQUA-DP", eps, seed, f_acc, ceiling - f_acc))
                rows.append(_mk_row(ds, "B4", eps, seed, b_acc, ceiling - b_acc))
    rep = evaluate_dod(rows)
    d = rep.as_dict()
    assert d["DoD-1_pass"] is True
    # epsilon-local: flagship@0.05 (0.52) is NOT penalised by baseline@10 (0.74)
    assert d["DoD-2_pass"] is True
    assert d["DoD-3_pass"] is True
    assert d["DoD-4_pass"] is True
    # flagship reaches the target at eps=0.5, the baseline only at eps=10
    assert d["DoD-5_pass"] is True
    assert d["DoD-6_pass"] is True
    assert d["DoD-7_pass"] is True
    assert d["all_passed"] is True


def test_dod_reference_is_strongest_single_baseline_not_envelope():
    """SPEC section 4: the DoD reference is the strongest SINGLE baseline.

    The reference method is picked per dataset by MEAN accuracy, and every
    seed is paired against that same method. A per-seed max-over-methods
    envelope is an oracle no single method can beat (B4 wins seed1, B2 wins
    seed2 -> the envelope is 0.70 on both seeds while every real method is
    below it on one seed) -- that bug made DoD-2/4 structurally unpassable.
    """
    from privforge.eval.protocol import _strongest_base

    rows = [
        _mk_row("medium", "B4", 1.0, 1, 0.70, 0.02),
        _mk_row("medium", "B4", 1.0, 2, 0.62, 0.10),
        _mk_row("medium", "B2", 1.0, 1, 0.60, 0.12),
        _mk_row("medium", "B2", 1.0, 2, 0.70, 0.02),
    ]
    ref = _strongest_base(rows)
    # B4 mean 0.66 > B2 mean 0.65 -> B4 is the reference on BOTH seeds,
    # including seed2 where B2 happened to score higher.
    assert ref == {
        ("medium", 1, 1.0): 0.70,
        ("medium", 2, 1.0): 0.62,
    }
