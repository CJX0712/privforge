"""tests/test_accountant.py — RDP 矩会计数学不变量。"""

from __future__ import annotations

import math

import pytest

from core.errors import PrivacyError
from dp.accountant import RDPAccountant, gaussian_rdp


def _manual_epsilon(T, sigma, delta):
    alphas = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 32.0, 64.0]
    best = float("inf")
    for a in alphas:
        eps = T * gaussian_rdp(a, sigma) + math.log(1.0 / delta) / (a - 1.0)
        best = min(best, eps)
    return best


def test_gaussian_rdp_formula():
    assert abs(gaussian_rdp(2.0, 1.0) - 1.0) < 1e-12
    assert abs(gaussian_rdp(4.0, 2.0) - 0.5) < 1e-12


def test_composition_additivity():
    acc = RDPAccountant()
    T, sigma = 50, 1.1
    for _ in range(T):
        acc.step(sigma)
    # rdp(α) = T · α/(2σ²)
    for a in (2.0, 5.0, 10.0):
        assert abs(acc.rdp(a) - T * gaussian_rdp(a, sigma)) < 1e-9


def test_get_epsilon_matches_manual():
    acc = RDPAccountant()
    T, sigma, delta = 80, 1.1, 1e-5
    for _ in range(T):
        acc.step(sigma)
    assert abs(acc.get_epsilon(delta) - _manual_epsilon(T, sigma, delta)) < 1e-9


def test_epsilon_monotonic_in_sigma():
    delta, T = 1e-5, 50
    lo = RDPAccountant()
    for _ in range(T):
        lo.step(1.0)
    hi = RDPAccountant()
    for _ in range(T):
        hi.step(2.0)
    assert hi.get_epsilon(delta) < lo.get_epsilon(delta)


def test_epsilon_monotonic_in_steps():
    delta, sigma = 1e-5, 1.1
    few = RDPAccountant()
    for _ in range(20):
        few.step(sigma)
    many = RDPAccountant()
    for _ in range(100):
        many.step(sigma)
    assert many.get_epsilon(delta) > few.get_epsilon(delta)


def test_epsilon_decreases_with_delta():
    # δ 越大（隐私要求越松）→ ε 越小（单调性）
    sigma, T = 1.1, 60
    acc = RDPAccountant()
    for _ in range(T):
        acc.step(sigma)
    assert acc.get_epsilon(1e-4) < acc.get_epsilon(1e-6)


def test_privacy_independent_of_clip_norm():
    # 隐私只由 (σ,T) 决定，与裁剪界无关 —— 核心性质
    a1 = RDPAccountant()
    for _ in range(40):
        a1.step(1.1)
    assert a1.get_epsilon(1e-5) > 0


def test_invalid_sigma_raises():
    acc = RDPAccountant()
    with pytest.raises(PrivacyError):
        acc.step(0.0)
    with pytest.raises(PrivacyError):
        gaussian_rdp(2.0, 0.0)
