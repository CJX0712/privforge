"""Shared fixtures for the PrivForge test suite.

Author: 晨星
"""

from __future__ import annotations

import pytest

from privforge.core.types import Budget
from privforge.data.loader import train_test_split
from privforge.data.synthetic import generate_dataset


@pytest.fixture
def tiny_ds():
    """A (train, test) split of the medium synthetic dataset, seed 1."""
    ds = generate_dataset("medium")
    return train_test_split(ds, seed=1)


@pytest.fixture
def budget():
    """A (epsilon=1.0, delta=1e-5) budget."""
    return Budget.from_fractions(epsilon=1.0, delta=1e-5)
