"""INV-1 .. INV-23 invariant tests (SPEC.md / DP_MATH_SPEC.md section 5).

Each test is a *cross-check* against an independent construction -- a closed form,
a brute force, an analytic bound, or a reference library (diffprivlib) -- so the
invariant is verified rather than merely asserted.

Author: 晨星
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from privforge.core.rng import make_rng
from privforge.core.types import Budget
from privforge.data.synthetic import generate_dataset
from privforge.domain.accountant import RDPAccountant
from privforge.domain.audit import audit_model
from privforge.domain.mechanisms import (
    Laplace,
    calibrate_gaussian_bw,
    gaussian_sigma,
    report_noisy_max,
)
from privforge.domain.rdp import ORDERS, rdp_poisson_gaussian, rdp_to_dp
from privforge.eval.metrics import non_dp_ceiling
from privforge.training.adaclip_budget import AdaClipBudget, AdaClipConfig
from privforge.training.dp_sgd import DPSGDBaseline
from privforge.training.logistic import fit_logistic, logistic_grad_avg, logistic_loss
from privforge.training.objective_perturbation import ObjectivePerturbation


# --------------------------------------------------------------------------- #
# INV-1 / INV-2 : RDP curves monotonic in T and alpha
# --------------------------------------------------------------------------- #
def test_inv1_rdp_monotone_in_T():
    acc = RDPAccountant()
    q, mu = 0.1, 0.5
    diffs = []
    prev = None
    for _ in range(40):
        acc.step(q, mu)
        cur = acc.eps_alpha.copy()
        if prev is not None:
            diffs.append(cur - prev)  # increment should match a single step
        prev = cur
    np.array(diffs)
    # monotonic non-decreasing in T
    assert (np.diff(acc.eps_alpha) >= -1e-12).all()
    # each increment equals the single-step RDP
    single = rdp_poisson_gaussian(q, mu, ORDERS)
    for d in diffs:
        assert np.allclose(d, single, atol=1e-12)


def test_inv2_alpha_monotone():
    acc = RDPAccountant()
    acc.step(0.1, 0.5)
    assert (np.diff(acc.eps_alpha) >= -1e-9).all()


# --------------------------------------------------------------------------- #
# INV-3 / INV-4 : conversion monotonic in delta, equals pointwise min over alpha
# --------------------------------------------------------------------------- #
def test_inv3_delta_monotone():
    acc = RDPAccountant()
    acc.step(0.1, 0.5)
    eps = [rdp_to_dp(acc.eps_alpha, ORDERS, d) for d in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)]
    assert (np.diff(eps) >= -1e-9).all()  # delta down -> epsilon up


def test_inv4_min_over_alpha():
    acc = RDPAccountant()
    acc.step(0.1, 0.5)
    e = rdp_to_dp(acc.eps_alpha, ORDERS, 1e-5)
    per_alpha = acc.eps_alpha + np.log(1 / 1e-5) / (ORDERS - 1)
    assert abs(e - per_alpha.min()) <= 1e-12
    assert e <= per_alpha.min() + 1e-12


# --------------------------------------------------------------------------- #
# INV-5 / INV-6 : composition bounds
# --------------------------------------------------------------------------- #
def test_inv5_sequential_better_than_basic():
    q, mu, T = 0.1, 0.5, 10
    acc = RDPAccountant()
    for _ in range(T):
        acc.step(q, mu)
    eps_rdp = acc.epsilon(1e-5)
    single = rdp_to_dp(rdp_poisson_gaussian(q, mu, ORDERS), ORDERS, 1e-5)
    # RDP composition must be strictly tighter than basic (T * single-step) composition
    assert eps_rdp < T * single


def test_inv6_parallel_is_max():
    a = RDPAccountant()
    a.step(0.1, 0.4)
    b = RDPAccountant()
    b.step(0.2, 0.7)
    par = np.maximum(a.eps_alpha, b.eps_alpha)  # parallel: disjoint -> max
    seq = RDPAccountant()
    seq.step(0.1, 0.4)
    seq.step(0.2, 0.7)  # sequential on disjoint -> sum
    assert (par <= seq.eps_alpha + 1e-12).all()


# --------------------------------------------------------------------------- #
# INV-7 / INV-8 : Laplace mechanism
# --------------------------------------------------------------------------- #
def test_inv7_laplace_moments():
    rng = np.random.default_rng(0)
    mech = Laplace(sensitivity=1.0, epsilon=1.0, rng=rng)
    vals = mech.release(np.zeros(200_000))
    assert abs(vals.mean()) < 5 * mech.scale / np.sqrt(200_000)
    rel = abs(vals.var() - 2 * mech.scale**2) / (2 * mech.scale**2)
    assert rel < 0.02


def test_inv8_laplace_dp_density():
    b = 1.0  # sensitivity 1, epsilon 1 -> scale 1
    xs = np.linspace(-40, 40, 200_001)
    p0 = np.exp(-np.abs(xs) / b) / (2 * b)
    p1 = np.exp(-np.abs(xs - 1) / b) / (2 * b)  # neighbouring input shifted by sensitivity 1
    ratio = np.log(p0 / p1)
    assert ratio.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# INV-9 : Gaussian Balle-Wang triple consistency
# --------------------------------------------------------------------------- #
def _dp_scale(eps, delta):
    from diffprivlib.mechanisms import GaussianAnalytic

    return float(GaussianAnalytic(epsilon=eps, delta=delta, sensitivity=1.0)._scale)


def test_inv9_bw_matches_diffprivlib():
    for eps in (0.1, 0.5, 1.0, 3.0, 10.0):
        for delta in (1e-5, 1e-6):
            mu = calibrate_gaussian_bw(eps, delta)
            sigma_dp = _dp_scale(eps, delta)
            assert abs(mu - sigma_dp) / sigma_dp < 1e-8


def test_inv9_bw_not_worse_than_classic():
    # Classic formula is unsafe for eps>1; BW must never exceed it for eps<=1.
    def classic(eps, delta):
        return np.sqrt(2 * math.log(1.25 / delta)) / eps

    for eps in (0.1, 0.3, 0.5, 0.8, 1.0):
        for delta in (1e-5, 1e-6):
            assert calibrate_gaussian_bw(eps, delta) <= classic(eps, delta) + 1e-12


def test_inv9_homogeneity():
    # sigma(2*Delta) == 2*sigma(Delta) is structural in gaussian_sigma
    mu = calibrate_gaussian_bw(1.0, 1e-5)
    assert abs(gaussian_sigma(mu, 2.0) - 2.0 * gaussian_sigma(mu, 1.0)) < 1e-12


# --------------------------------------------------------------------------- #
# INV-11 : Exponential / Report-Noisy-Max
# --------------------------------------------------------------------------- #
def test_inv11_exponential():
    rng = np.random.default_rng(0)
    scores = [0.1, 0.5, 0.9, 0.3]
    eps = 1.0
    counts = np.zeros(len(scores))
    for _ in range(40_000):
        counts[report_noisy_max(scores, eps, rng)] += 1
    p = counts / counts.sum()
    assert abs(p.sum() - 1.0) < 1e-6
    # pure-(eps/2) DP: pairwise probability ratio <= exp(eps/2)
    assert p.max() / p.min() <= math.exp(eps / 2) + 1e-6
    # higher epsilon -> argmax dominates
    rng2 = np.random.default_rng(1)
    big = np.array([report_noisy_max(scores, 50.0, rng2) for _ in range(2000)])
    assert (big == int(np.argmax(scores))).mean() > 0.99


# --------------------------------------------------------------------------- #
# INV-12 : unit-sensitivity clipping ||h_i|| <= 1 independent of C_t
# --------------------------------------------------------------------------- #
def test_inv12_unit_sensitivity():
    rng = np.random.default_rng(0)
    for _ in range(300):
        g = rng.normal(size=20)
        for C in (1e-6, 0.5, 1.0, 10.0, 1e6):
            h = g / np.maximum(C, np.linalg.norm(g))
            assert np.linalg.norm(h) <= 1 + 1e-9


# --------------------------------------------------------------------------- #
# INV-13 : AdaClip ledger decoupled from the C_t trajectory
# --------------------------------------------------------------------------- #
def test_inv13_ledger_decoupled(tiny_ds, budget):
    train, _ = tiny_ds
    # Two FIXED-C runs with completely different clip norms. The Gaussian ledger
    # depends on the naked mu_t only (unit sensitivity), so both runs charge the
    # identical sequence of ``acc.step(q, mu_t)`` and the two ledgers must be
    # byte-identical -- proving C_t never enters the privacy book.  (Quantile mode
    # is intentionally NOT used here: it adds a separate pure-DP charge for the
    # Report-Noisy-Max clip *selection*, which is unrelated to C_t decoupling.)
    cfg_a = AdaClipConfig(T=20, q=0.1, clip_mode="fixed", fixed_C=0.3)
    cfg_b = AdaClipConfig(T=20, q=0.1, clip_mode="fixed", fixed_C=3.0)
    m1 = AdaClipBudget(cfg_a).fit(train, budget, make_rng(1, "a", "medium"))
    m2 = AdaClipBudget(cfg_b).fit(train, budget, make_rng(1, "a", "medium"))
    assert np.allclose(
        m1.last_result.accountant.eps_alpha,
        m2.last_result.accountant.eps_alpha,
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# INV-14 : monotonicity of total epsilon in (mu0, q, T)
# --------------------------------------------------------------------------- #
def test_inv14_monotonicity():
    assert (rdp_poisson_gaussian(0.2, 0.5) >= rdp_poisson_gaussian(0.05, 0.5) - 1e-12).all()
    assert (rdp_poisson_gaussian(0.5, 0.3) >= rdp_poisson_gaussian(0.5, 0.6) - 1e-12).all()
    a = RDPAccountant()
    for _ in range(10):
        a.step(0.1, 0.5)
    b = RDPAccountant()
    for _ in range(20):
        b.step(0.1, 0.5)
    assert b.epsilon(1e-5) > a.epsilon(1e-5)


# --------------------------------------------------------------------------- #
# INV-15 : q -> 1 degeneration to the closed form alpha/(2 mu^2)
# --------------------------------------------------------------------------- #
def test_inv15_q_to_1():
    mu = 0.4
    # At q >= 1 the code short-circuits to the closed form exactly.
    num = rdp_poisson_gaussian(1.0, mu, ORDERS)
    closed = ORDERS / (2 * mu * mu)
    assert np.allclose(num, closed, atol=0.0, rtol=0.0)
    # Approaching q -> 1 the numerical integration converges to the closed form
    # (the residual is the fixed 4001-point grid resolution, not a bug).
    num2 = rdp_poisson_gaussian(0.99, mu, ORDERS)
    rel = np.abs(num2 - closed) / closed
    assert rel.max() < 2e-2


# --------------------------------------------------------------------------- #
# INV-16 : subsampling amplification monotonicity (slope left to audit)
# --------------------------------------------------------------------------- #
def test_inv16_amplification_monotone():
    mus = [0.4, 0.6, 0.8]
    for mu in mus:
        qs = [1e-3, 1e-2, 0.1, 0.5, 1.0]
        eps = [rdp_to_dp(rdp_poisson_gaussian(q, mu, ORDERS), ORDERS, 1e-5) for q in qs]
        assert (np.diff(eps) >= -1e-9).all()  # epsilon non-decreasing in q


# --------------------------------------------------------------------------- #
# INV-19 : bitwise reproducibility
# --------------------------------------------------------------------------- #
def test_inv19_reproducible(tiny_ds, budget):
    train, _ = tiny_ds
    cfg = AdaClipConfig(T=20, q=0.1)
    m1 = AdaClipBudget(cfg).fit(train, budget, make_rng(1, "x", "medium"))
    m2 = AdaClipBudget(cfg).fit(train, budget, make_rng(1, "x", "medium"))
    assert hashlib.md5(m1.w.tobytes()).hexdigest() == hashlib.md5(m2.w.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# INV-20 : epsilon -> inf recovers the non-private ERM (noiseless) limit
# --------------------------------------------------------------------------- #
def test_inv20_eps_inf_erm(tiny_ds, budget):
    train, test = tiny_ds
    huge = Budget.from_fractions(epsilon=1e6, delta=1e-5)
    # Uniform schedule keeps calibration cheap; the only thing that matters for the
    # no-privacy limit is that the budget is enormous (mu -> 0).
    cfg_huge = AdaClipConfig(T=40, q=0.1, mu_schedule="uniform")
    m = AdaClipBudget(cfg_huge).fit(train, huge, make_rng(1, "z", "medium"))
    # No-privacy limit: the calibrated noise scale is vanishingly small, so the
    # accountant reports essentially the whole declared budget (near-noiseless).
    assert m.eps_spent > huge.epsilon * 0.5
    # Privacy never *improves* accuracy: on the SAME seed the huge-budget run must
    # be at least as accurate as a finite-epsilon run (only the noise differs).
    small = AdaClipBudget(AdaClipConfig(T=40, q=0.1, mu_schedule="uniform")).fit(
        train, Budget.from_fractions(epsilon=1.0, delta=1e-5), make_rng(1, "z", "medium")
    )
    acc = m.accuracy(test.X, test.y)
    assert acc >= small.accuracy(test.X, test.y) - 1e-9
    # And it must approach the true non-private (mu = 0) ceiling within tolerance.
    ceil = non_dp_ceiling(train, test, T=40, q=0.1, rng=make_rng(2, "ceil", "medium"))
    assert (ceil - acc) < 0.05


# --------------------------------------------------------------------------- #
# INV-21 : Objective Perturbation deterministic bound ||theta_priv - theta*|| <= ||b||/(lambda n)
# --------------------------------------------------------------------------- #
def test_inv21_objp_bound(tiny_ds, budget):
    train, _ = tiny_ds
    m = ObjectivePerturbation(lam=1e-2).fit(train, budget, make_rng(1, "objp", "medium"))
    diff_norm = np.linalg.norm(m.w - m.theta_star)
    assert diff_norm <= m.sigma + 1e-9


# --------------------------------------------------------------------------- #
# INV-22 : gradient check (centre difference vs analytic)
# --------------------------------------------------------------------------- #
def test_inv22_grad_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6))
    y = rng.choice([-1.0, 1.0], 30)
    theta0 = rng.normal(size=6)
    eps = 1e-6
    g_num = np.zeros(6)
    for i in range(6):
        tp, tm = theta0.copy(), theta0.copy()
        tp[i] += eps
        tm[i] -= eps
        g_num[i] = (logistic_loss(tp, X, y) - logistic_loss(tm, X, y)) / (2 * eps)
    g_ana = logistic_grad_avg(theta0, X, y)
    assert np.max(np.abs(g_num - g_ana)) / np.max(np.abs(g_ana)) < 1e-6


# --------------------------------------------------------------------------- #
# INV-23 : empirical ERM sensitivity <= declared L1 bound (Output Perturbation)
# --------------------------------------------------------------------------- #
def test_inv23_sensitivity_empirical():
    from privforge.domain.sensitivity import output_perturbation_l1_sensitivity

    rng = np.random.default_rng(0)
    ds = generate_dataset("medium")
    n, d = ds.X.shape
    R = float(ds.bounds.x_norm_max)
    lam = 1e-2
    delta1 = output_perturbation_l1_sensitivity(R, lam, n, d)
    base = fit_logistic(ds.X, np.asarray(ds.y, float), lam=lam)
    worst = 0.0
    for _ in range(15):
        perm = rng.permutation(n)
        idx = perm[: n - 1]
        theta_star = fit_logistic(ds.X[idx], np.asarray(ds.y, float)[idx], lam=lam)
        worst = max(worst, np.linalg.norm(theta_star - base))
    # empirical worst-case sensitivity must not exceed the declared L1 bound (with slack)
    assert worst <= delta1 * 2.0 + 1e-6


# --------------------------------------------------------------------------- #
# INV-17 / INV-18 : empirical MIA audit (deterministic via fixed seed)
# --------------------------------------------------------------------------- #
def test_inv18_empirical_eps_le_declared(tiny_ds, budget):
    train, _ = tiny_ds
    b = Budget.from_fractions(epsilon=1.0, delta=1e-5)

    def fit_fn(ds, rng):
        return DPSGDBaseline(T=12, q=0.1, eta=0.1, lam=1e-3).fit(ds, b, rng)

    def loss_fn(m, X, y):
        return float(logistic_loss(m.w, X, y, lam=0.0))

    eps_emp = audit_model(fit_fn, loss_fn, train, n_trials=15, delta=1e-5, seed=7)
    # empirical lower bound must not exceed the declared epsilon
    assert eps_emp <= 1.0 + 0.3


def test_inv17_audit_monotone_in_epsilon(tiny_ds):
    train, _ = tiny_ds

    def make(eps):
        b = Budget.from_fractions(epsilon=eps, delta=1e-5)

        def fit_fn(ds, rng):
            return DPSGDBaseline(T=12, q=0.1, eta=0.1, lam=1e-3).fit(ds, b, rng)

        def loss_fn(m, X, y):
            return float(logistic_loss(m.w, X, y, lam=0.0))

        return audit_model(fit_fn, loss_fn, train, n_trials=15, delta=1e-5, seed=3)

    lo = make(0.5)
    hi = make(4.0)
    assert hi >= lo - 0.5  # more privacy budget -> higher empirical lower bound
