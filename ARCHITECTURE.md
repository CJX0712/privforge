# PrivForge Architecture (overview)

PrivForge is a differentially private machine learning library. The flagship
algorithm is **AdaClip-Budget**; the pipeline that wraps it with safeguards is
called **AQUA-DP**. Full detail lives in `docs/architecture.md` and `docs/SPEC.md`;
this file is a standalone summary.

Author: 晨星 (MorningStar). License: MIT.

## Layered design

```
cli -> pipeline -> {data, hpo, training, domain, eval} -> core
backends -> core            (backends is a leaf package, sibling of core)
```

- `cli` only assembles; no business logic (<= 150 lines).
- `pipeline` is the only layer allowed to import all of
  `{data, hpo, training, domain, eval}`.
- `backends` and `core` are leaves; `backends` never imports an upper layer.
- All cross-module calls are programmed against the Protocols in
  `core/interfaces.py` and injected at runtime (R10).

## Dependency rules (R1-R10, summary)

| Rule | Meaning |
|---|---|
| R1 | Dependency direction is one-way; `core` imports no upper layer. |
| R2 | `domain` depends only on `core`. |
| R3 | `training` depends on `core` + `domain` only. |
| R4 | `eval` depends on `core` + `domain`; only the Protocol, never `training` internals. |
| R5 | `data` depends on `core` only. |
| R6 | `hpo` depends on `core` + `data` (public proxy); not `training` internals. |
| R7 | `pipeline` is the sole integrator of the algorithm layers. |
| R8 | `cli` assembles only, zero business logic, <= 150 lines. |
| R9 | No single file exceeds 300 lines. |
| R10 | Program to `core/interfaces.py` Protocols; inject implementations at runtime. |

These rules are statically enforced by `tools/check_imports.py` in CI.

## Dataclass contracts

All value objects are `@dataclass(frozen=True, slots=True)` (except where noted
mutable). Defined in `privforge/core/types.py`:

- `Bounds(x_norm_max, y_abs_max)` -- mandatory, data-independent public bounds.
- `Budget(epsilon, delta, eps_clip, eps_train, eps_select)` -- built via
  `Budget.from_fractions(epsilon, delta, fractions)`; `__post_init__` asserts the
  three shares sum to `<= epsilon` (else `E102`).
- `Dataset(name, X, y, bounds, meta)` -- binary labels in `{-1, +1}`; bounds are
  mandatory.
- `DPFitResult(w, views, weights, C_hat, sigma, budget_spent, backend, fallback_reason)`.
- `EvalResult(dataset, method, epsilon, seed, accuracy, auc, macro_f1, eps_spent, utility_gap, ugc, uac, eps_min_at_target)`.
- `SweepPoint(epsilon, method, dataset, seed, metrics)`.

## Three silent-failure hard constraints

These are bugs that produce **no error** but silently void the privacy claim. They
are guarded by source-scan tests plus runtime invariants:

1. **No `mu * C_t`.** AdaClip-Budget must use the naked `mu_t` noise scale. Writing
   `mu_t * C_t` makes the noise data-dependent and breaks the decoupling theorem.
   Guarded by INV-12 / I9 and `test_no_mu_times_C`.
2. **Correct RDP integration grid.** Must cover both peak directions:
   `[-max(0,(alpha-1)r)-12, max(0,alpha*r)+12]` with `r = 1/mu`, in log space.
   A too-small grid (e.g. `[-10,10]`) silently halves the bound at large `alpha`.
   Guarded by INV-15.
3. **Balle-Wang, not the classic formula.** The classic
   `sqrt(2 ln(1.25/delta))/eps` is unsafe for `eps > 1` (it under-noises). Calibrate
   with `calibrate_gaussian_bw`. Guarded by INV-9 / I12.

## Invariant pointers

- **Architecture invariants I1-I13** (`docs/architecture.md` section 9): budget
  never exceeded, no epsilon-splitting ensembles, public bounds only, no
  `PrivacyLeakWarning`, import direction, no legacy numpy aliases, no emoji/box
  characters, scaling invariance, unit-sensitivity (`||h_i||<=1` independent of
  `C_t`), ledger decoupled from `C_t` (INV-13), RDP grid covers peaks, no classic
  Gaussian formula, valid `alpha` grid.
- **Math invariants INV-1..INV-23** (`DP_MATH_SPEC.md` section 5): RDP monotonicity,
  Laplace/Exponential correctness, Balle-Wang triple consistency, decoupling,
  subsampling degeneration, MIA monotonicity, determinism, and more -- all
  parameterized into `tests/test_invariants.py`.

See `docs/architecture.md` and `docs/SPEC.md` for the authoritative definitions.
