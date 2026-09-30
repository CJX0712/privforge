"""Transfer learning of HPO configs across datasets (SPEC.md section 5).

When a new dataset shares structure with a previously-tuned one, reuse the best
known configuration instead of re-running the study.  Similarity is by public,
data-independent features only (dimensionality + class separation + heavy-tail
flag) -- no private statistics are compared (pitfall #18).

Author: 晨星
"""

from __future__ import annotations

from privforge.data.synthetic import DATASET_IDS, generate_dataset
from privforge.hpo.space import HpoConfig

# Public, data-independent profile of each known dataset (from synthetic.py SPECS).
_PROFILES: dict[str, dict] = {
    "easy_separable": {"n_features": 10, "class_sep": 1.5, "heavy_tail": False},
    "medium": {"n_features": 20, "class_sep": 1.0, "heavy_tail": False},
    "hard_lowsep": {"n_features": 40, "class_sep": 0.6, "heavy_tail": False},
    "heavy_tail": {"n_features": 20, "class_sep": 1.0, "heavy_tail": True},
    "imbalanced": {"n_features": 20, "class_sep": 1.0, "heavy_tail": False},
    "digits_subsample": {"n_features": 64, "class_sep": 0.0, "heavy_tail": False},
}

# Curated best-known configs (proxy of a prior study; architecture-only knobs).
_KNOWN_BEST: dict[str, HpoConfig] = {
    "easy_separable": HpoConfig(
        T=120, eta=0.2, lam=1e-3, rho=0.99, p=0.7, clip_mode="quantile", mu_schedule="propagation"
    ),
    "medium": HpoConfig(
        T=200, eta=0.1, lam=1e-3, rho=0.99, p=0.7, clip_mode="quantile", mu_schedule="propagation"
    ),
    "hard_lowsep": HpoConfig(
        T=240, eta=0.1, lam=1e-3, rho=0.95, p=0.6, clip_mode="quantile", mu_schedule="propagation"
    ),
    "heavy_tail": HpoConfig(
        T=240, eta=0.15, lam=1e-3, rho=0.97, p=0.8, clip_mode="quantile", mu_schedule="propagation"
    ),
    "imbalanced": HpoConfig(
        T=200, eta=0.1, lam=1e-2, rho=0.99, p=0.7, clip_mode="quantile", mu_schedule="uniform"
    ),
    "digits_subsample": HpoConfig(
        T=240, eta=0.1, lam=1e-3, rho=0.99, p=0.7, clip_mode="quantile", mu_schedule="propagation"
    ),
}


def profile_of(name: str) -> dict:
    if name in _PROFILES:
        return _PROFILES[name]
    ds = generate_dataset(name)
    return {
        "n_features": int(ds.X.shape[1]),
        "class_sep": float(ds.meta.get("class_sep", 1.0)),
        "heavy_tail": bool(ds.meta.get("heavy_tail", False)),
    }


def transfer_config(source: str, target: str) -> HpoConfig:
    """Return the known-best config for `source`, reused for `target`."""
    if source not in _KNOWN_BEST:
        raise KeyError(f"no known config for source dataset: {source}")
    if source == target:
        return _KNOWN_BEST[source]
    return _KNOWN_BEST[source]


def nearest_known_config(target: str) -> HpoConfig:
    """Pick the closest known dataset's config for a possibly-new target."""
    if target in _KNOWN_BEST:
        return _KNOWN_BEST[target]
    tp = profile_of(target)
    best_name, best_dist = None, float("inf")
    for name, prof in _PROFILES.items():
        d = (
            abs(prof["n_features"] - tp["n_features"])
            + 10.0 * abs(prof["class_sep"] - tp["class_sep"])
            + (0.0 if prof["heavy_tail"] == tp["heavy_tail"] else 5.0)
        )
        if d < best_dist:
            best_dist, best_name = d, name
    return _KNOWN_BEST[best_name]


def all_known_datasets() -> tuple[str, ...]:
    return tuple(DATASET_IDS)
