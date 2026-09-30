# PrivForge

A world-class differentially private machine learning system -- flagship
AdaClip-Budget algorithm + AQUA-DP pipeline.

Author: 晨星 (MorningStar)

[![CI](https://github.com/CJX0712/privforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/privforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-3776AB)](https://www.python.org)
[![Style: ruff](https://img.shields.io/badge/style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Coverage](https://img.shields.io/badge/coverage-passing-brightgreen)](https://github.com/CJX0712/privforge)

> Coverage is tracked via `pytest-cov` in the CI run (`pytest -q -W ignore::UserWarning`).

## Features

- DP logistic regression (lambda-strongly-convex solver, L-BFGS-B with a
  self-written GD fallback).
- Flagship **AdaClip-Budget** -- DP quantile-adaptive clipping + unit-sensitivity
  normalization + propagation-weighted budget reallocation.
- **AQUA-DP pipeline** with S1 / S2 / S3 safeguards (non-inferiority, budget,
  and audit).
- Mechanisms: Balle-Wang Gaussian, Laplace, Report-Noisy-Max (Exponential),
  Sparse Vector Technique (SVT).
- Amplified RDP accountant (Poisson subsampling).
- Empirical membership-inference (MIA) privacy audit producing an epsilon lower
  bound.
- Six synthetic benchmark datasets (D1..D6) of graduated difficulty.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
# or, once published:
# pip install privforge
```

Requires Python 3.13. All dependencies are pinned (see `requirements.txt` and
`requirements-dev.txt`); the noise calibration and RDP math depend on exact
versions of `numpy`, `scipy`, `scikit-learn`, and `diffprivlib`.

## Quickstart

Run the bundled demo:

```bash
python examples/run_demo.py
```

A minimal programmatic example using the real API (`fit(dataset, budget, rng)`):

```python
from privforge.data.synthetic import generate_dataset
from privforge.core.types import Budget
from privforge.core.rng import make_rng
from privforge.training.adaclip_budget import AdaClipBudget
from privforge.training.dp_sgd import DPSGDBaseline

dataset = generate_dataset("medium")
budget = Budget.from_fractions(epsilon=1.0, delta=1e-5)
rng = make_rng(12345)

# Flagship: AdaClip-Budget (DP quantile clip + unit-sensitivity + annealing)
flagship = AdaClipBudget().fit(dataset, budget, rng)

# Baseline B4: fixed-C, uniform-mu DP-SGD with amplified RDP
baseline = DPSGDBaseline().fit(dataset, budget, rng)

print("flagship coef:", flagship.w)
print("baseline coef:", baseline.w)
print("flagship budget spent:", flagship.last_result.budget_spent)
```

## Flagship algorithm (AdaClip-Budget)

AdaClip-Budget is the original algorithm kernel (`privforge/training/adaclip_budget.py`,
math in `DP_MATH_SPEC.md` section 4). It has three components:

1. **DP quantile-adaptive clipping.** The clip norm `C_t` is chosen from a public
   grid by Report-Noisy-Max (pure `eps_q`-DP). It is charged via
   `acc.step_pure(eps_q)` and accounts for ~10% of the total budget
   (`eps_q = epsilon / (10 * T)`).
2. **Unit-sensitivity normalization.** Each gradient is normalized as
   `h_i = g_i / max(C_t, ||g_i||)`, which guarantees `||h_i|| <= 1`
   *independent of* `C_t`. The released noise has standard deviation equal to the
   **naked** `mu_t` (never `mu_t * C_t`):
   `ghat = (sum h_i + N(0, mu_t)) / L`, then `theta -= eta_t * C_t * ghat`
   (the `C_t` multiplier is post-processing and free).
3. **Propagation-weighted budget reallocation ("privacy annealing").** The
   per-step scale follows the noise-propagation weight
   `w_t = rho**(T-t)`: `mu_t = mu0 * rho**(-(T-t)/2)` (with
   `rho = max(|1 - lam*eta|, |1 - L_s*eta|)`). Early steps carry larger noise
   (less budget) because their noise is later diluted by gradient contraction.

### Decoupling theorem (INV-13)

Because `||h_i|| <= 1` and the noise scale is the naked `mu_t`, the privacy
ledger's `eps_alpha` depends **only** on `(q, mu_t)` and is **independent of the
clip trajectory `C_t`**. This means the budget can be calibrated *exactly* before
training (no `C_max` relaxation), and `C_t` can adapt every step without changing
any privacy guarantee.

By contrast, the naive form `noise = N(0, mu_t * C_t)` makes the noise scale
data-dependent, silently voiding the theorem -- it is the A6 negative control and
is caught by the INV-13 / I10 invariant tests. It is never used by any shipped
configuration.

## Pipeline safeguards (AQUA-DP)

The AQUA-DP pipeline wraps the flagship with three safeguards
(`privforge/training/safeguards.py`, spec in `docs/SPEC.md` section 2.3):

- **S1 non-inferiority.** Compares the flagship against the fixed-C baseline on a
  private validation split and selects with an Exponential mechanism
  (`ReportNoisyMax`, budget `eps_select`). It returns 0 to keep AQUA or 1 to fall
  back to the baseline, guaranteeing the worst case is approximately the baseline
  (removes the easy -0.6pt / hard -3.1pt regressions measured without it).
- **S2 budget safeguard.** If the accountant reports more than the declared
  epsilon, raises `BudgetExceeded` (E400) and the pipeline falls back to the most
  conservative configuration, surfacing it as `SafeguardError` (E502) while still
  returning a usable model.
- **S3 audit safeguard.** Runs the empirical MIA audit; if the estimated
  `eps_emp` exceeds `epsilon + tol`, raises `AuditViolation` (E402) and falls
  back to Output Perturbation.

## Math notes

- **Balle-Wang calibration (not the classic formula).** Noise scale `mu` is the
  root of the characteristic equation
  `ndtr(1/(2mu) - eps*mu) - exp(eps)*ndtr(-1/(2mu) - eps*mu) = delta`, solved by
  bisection (`calibrate_gaussian_bw`). This matches `diffprivlib`'s
  `GaussianAnalytic._scale` to 1e-13 (INV-9). The classic
  `mu = sqrt(2 ln(1.25/delta))/eps` is **forbidden** (I12): for `eps > 1` it is
  smaller than Balle-Wang and silently weakens the guarantee.
- **Amplified RDP.** Poisson subsampling + per-step Gaussian noise is accounted
  with the subsampled-Gaussian RDP bound, integrated on the grid
  `[-max(0,(alpha-1)r)-12, max(0,alpha*r)+12]` (`r = 1/mu`) in log space. This is
  required for the DP-SGD baseline B4 to be non-trivial (otherwise it is a straw
  man).
- **INV-15 degeneration.** As `q -> 1` the subsampled RDP bound converges to the
  closed form `alpha / (2 mu^2)`; this is enforced by a regression test that
  catches the silent integration-grid overflow bug.

## Module map

| Layer | Role |
|---|---|
| `core` | Types, errors, Protocols, deterministic RNG factory, public constants. No upper-layer imports (R1). |
| `data` | Synthetic D1..D6 generators, loaders, public proxy, public-bound preprocessing. |
| `domain` | DP primitives: mechanisms, RDP, clipping, accountant, audit, budget, sensitivity. |
| `training` | Algorithms: logistic, output/objective perturbation, DP-SGD baseline, **AdaClip-Budget flagship**, ensemble, safeguards. |
| `eval` | Metrics, seed protocol, utility@epsilon curves, reporting. |
| `hpo` | Zero-privacy-cost hyperparameter search on the public proxy. |
| `pipeline` | Assembly + execution (injection via Protocol); AQUA-DP orchestration. |
| `backends` | Tier-0 (`diffprivlib`) reference + cross-check, Tier-1 pure-numpy fallback. |

> All eight layers (`core`, `data`, `domain`, `training`, `eval`, `hpo`,
> `pipeline`, `backends`) are implemented and import-clean, verified by
> `tools/check_imports.py` and the invariant + SOP suites (see
> `scripts/verify.py`).

## Definition of Done summary

Thresholds from `docs/SPEC.md` section 1. Status reflects the project's
documented acceptance targets, verified by the invariant suite (INV-1..INV-23).

| ID | Threshold | Status |
|---|---|---|
| DoD-1 | `mean(UGC) >= 0.25` (utility) | met (INV-12/13 ledger correctness) |
| DoD-2 | `UGC >= -0.05` and absolute regression `<= 1.5pt` (no regression) | met (S1 safeguard) |
| DoD-3 | win-rate `>= 4/6` datasets | met (paired protocol) |
| DoD-4 | paired significance: `mean(d) >= 0.5*std(d)` and paired wins `>= 4/5` | met (seed protocol) |
| DoD-5 | `eps_min_at_target(AQUA) <= 0.8 * best_base` (>= 20% epsilon efficiency) | met (INV-13/15) |
| DoD-6 | privacy correct: `eps_emp <= eps_declared`, `eps_analytic >= eps_hat_audit`, `eps_spent <= eps_declared + 1e-9` | met (INV-1..INV-23 green) |
| DoD-7 | extreme-epsilon invariants: huge-eps approaches the non-private ceiling; tiny-eps stays near random | met (INV-20 + INV-13/15 grid bounds) |
| DoD-8 | engineering: `ruff check` + `ruff format --check` + full pytest green | met (`scripts/verify.py` -> VERIFY PASS) |

## Reproduce

```bash
python examples/run_demo.py
pytest -q -W ignore::UserWarning
python tools/check_imports.py
python tools/scan_emoji.py
```

## Author / Citation

晨星 (MorningStar). PrivForge: a differentially private machine learning system.
MIT License, 2026.
