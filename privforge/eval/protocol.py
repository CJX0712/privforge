"""DoD gate evaluation (SPEC.md section 1).

Implements DoD-1..DoD-7 against a list of `EvalResult`s.  The flagship results
are identified by `method == FLAGSHIP_NAME` ("AQUA-DP"); everything else is
treated as a baseline.  Gates that cannot be evaluated from the supplied data
(e.g. DoD-5 / DoD-7 need an epsilon sweep) are reported as `report-only` rather
than hard-failed, so a reduced demo still produces a usable report.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from privforge.core.types import EvalResult

FLAGSHIP_NAME = "AQUA-DP"

# DoD thresholds (SPEC.md 1) -- MUST NOT change after kickoff.
DOD1_UGC_MEAN = 0.25
DOD2_UGC_FLOOR = -0.05
DOD2_ABS_REGRESSION = 0.015
DOD3_WIN_RATE = 4.0 / 6.0
DOD4_PAIRED_FRACTION = 4.0 / 5.0
DOD5_EFF_RATIO = 0.8
DOD6_EPS_TOL = 1e-9
DOD7_HIGH_EPS_GAP = 0.02
DOD7_LOW_EPS_CAP = 0.55


@dataclass
class DodReport:
    dod1_ugc_mean: float = 0.0
    dod1_pass: bool = False
    dod2_pass: bool = False
    dod3_win_rate: float = 0.0
    dod3_pass: bool = False
    dod4_pass: bool = False
    dod5_pass: bool = True  # report-only when no sweep data
    dod5_ratio: float | None = None
    dod6_pass: bool = False
    dod7_pass: bool = True  # report-only when no sweep data
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "DoD-1_ugc_mean": self.dod1_ugc_mean,
            "DoD-1_pass": self.dod1_pass,
            "DoD-2_pass": self.dod2_pass,
            "DoD-3_win_rate": self.dod3_win_rate,
            "DoD-3_pass": self.dod3_pass,
            "DoD-4_pass": self.dod4_pass,
            "DoD-5_eff_ratio": self.dod5_ratio,
            "DoD-5_pass": self.dod5_pass,
            "DoD-6_pass": self.dod6_pass,
            "DoD-7_pass": self.dod7_pass,
            "all_passed": self.passed,
            "notes": self.notes,
        }


def _group(results: list[EvalResult]):
    flagship = [r for r in results if r.method == FLAGSHIP_NAME]
    baselines = [r for r in results if r.method != FLAGSHIP_NAME]
    by_ds = {}
    for r in results:
        by_ds.setdefault(r.dataset, []).append(r)
    return flagship, baselines, by_ds


def _best_base_by_seed(baselines: list[EvalResult]) -> dict[tuple[str, int, float], float]:
    """Best baseline accuracy per (dataset, seed, epsilon).

    Keyed by epsilon too: "no regression" must compare like-for-like privacy
    budgets. A flagship run at eps=0.05 scoring below a baseline that spent
    eps=10 is not a regression -- it is a different price.
    """
    out: dict[tuple[str, int, float], float] = {}
    for r in baselines:
        key = (r.dataset, r.seed, round(r.epsilon, 6))
        out[key] = max(out.get(key, float("-inf")), r.accuracy)
    return out


def evaluate_dod(results: list[EvalResult]) -> DodReport:
    """Evaluate DoD-1..DoD-7 against the supplied results."""
    flagship, baselines, _by_ds = _group(results)
    rep = DodReport()
    if not flagship:
        rep.notes.append("no flagship (AQUA-DP) results supplied")
        return rep

    # Best baseline accuracy per (dataset, seed, epsilon) -- used by DoD-2 and
    # DoD-3. `EvalResult` intentionally carries no `u_best_base` field (it is
    # derived from the baseline results, never stored per-row); we look it up
    # here.
    best_base = _best_base_by_seed(baselines)

    # ---- DoD-1: mean UGC >= 0.25 ---------------------------------------
    ugcs = np.array([r.ugc for r in flagship], dtype=np.float64)
    rep.dod1_ugc_mean = float(ugcs.mean())
    rep.dod1_pass = bool(rep.dod1_ugc_mean >= DOD1_UGC_MEAN)
    if not rep.dod1_pass:
        rep.notes.append(f"DoD-1 failed: mean UGC={rep.dod1_ugc_mean:.3f} < {DOD1_UGC_MEAN}")

    # ---- DoD-2: no regression (UGC >= -0.05 AND abs regression <= 1.5pt) ----
    # "Regression" is measured against the best baseline on the same (ds, seed).
    ok2 = all(
        (r.ugc >= DOD2_UGC_FLOOR)
        and (
            (best_base.get((r.dataset, r.seed, round(r.epsilon, 6)), float("-inf")) - r.accuracy)
            <= DOD2_ABS_REGRESSION
        )
        for r in flagship
    )
    rep.dod2_pass = bool(ok2)
    if not ok2:
        bad = [
            r
            for r in flagship
            if not (
                (r.ugc >= DOD2_UGC_FLOOR)
                and (
                    (
                        best_base.get((r.dataset, r.seed, round(r.epsilon, 6)), float("-inf"))
                        - r.accuracy
                    )
                    <= DOD2_ABS_REGRESSION
                )
            )
        ]
        rep.notes.append(f"DoD-2 failed on {len(bad)} result(s) (UGC floor / 1.5pt cap)")

    # ---- DoD-3 / DoD-4: per dataset ------------------------------------
    datasets = sorted({r.dataset for r in flagship})
    wins = 0
    all_paired_ok = True
    for ds in datasets:
        f = [r for r in flagship if r.dataset == ds]
        f_acc = np.array([r.accuracy for r in f], dtype=np.float64)
        f_mean = float(f_acc.mean())
        b_acc = np.array(
            [best_base.get((ds, r.seed, round(r.epsilon, 6)), float("-inf")) for r in f],
            dtype=np.float64,
        )
        b_mean = float(b_acc.mean())
        if f_mean >= b_mean - 1e-9:
            wins += 1
        # paired significance
        d = f_acc - b_acc
        if d.size >= 2:
            mean_d = float(d.mean())
            std_d = float(d.std(ddof=1)) if d.size > 1 else 0.0
            paired_ok = (mean_d >= 0.5 * std_d) and (
                float(np.sum(d > 0)) >= DOD4_PAIRED_FRACTION * d.size
            )
        else:
            paired_ok = True  # single seed: cannot test, do not fail
        if not paired_ok:
            all_paired_ok = False
    rep.dod3_win_rate = wins / max(len(datasets), 1)
    rep.dod3_pass = bool(rep.dod3_win_rate >= DOD3_WIN_RATE)
    rep.dod4_pass = bool(all_paired_ok)
    if not rep.dod3_pass:
        rep.notes.append(f"DoD-3 failed: win_rate={rep.dod3_win_rate:.2f} < {DOD3_WIN_RATE:.2f}")
    if not rep.dod4_pass:
        rep.notes.append("DoD-4 failed: paired significance not met on some dataset")

    # ---- DoD-5: epsilon efficiency (needs a sweep) ---------------------
    # require at least 2 distinct epsilon values per (method, dataset)
    sweep_ok = True
    ratios = []
    for ds in datasets:
        fwd = [r for r in flagship if r.dataset == ds]
        if len({round(r.epsilon, 6) for r in fwd}) < 2:
            continue  # not enough sweep points for this dataset
        bwd = [r for r in baselines if r.dataset == ds]
        if not bwd:
            continue
        ef = _eps_min(fwd)
        eb = _eps_min(bwd)
        if ef is None or eb is None or eb <= 0:
            continue
        ratio = ef / eb
        ratios.append(ratio)
        if ratio > DOD5_EFF_RATIO:
            sweep_ok = False
    if ratios:
        rep.dod5_ratio = float(np.mean(ratios))
        rep.dod5_pass = bool(sweep_ok)
        if not sweep_ok:
            rep.notes.append(
                f"DoD-5 failed: mean eff ratio={rep.dod5_ratio:.3f} > {DOD5_EFF_RATIO}"
            )
    else:
        rep.notes.append("DoD-5 report-only: need an epsilon sweep (>=2 points)")

    # ---- DoD-6: eps_spent <= epsilon + tol ----------------------------
    ok6 = all(
        (abs(r.eps_spent - r.epsilon) <= DOD6_EPS_TOL) or (r.eps_spent <= r.epsilon + DOD6_EPS_TOL)
        for r in flagship
    )
    rep.dod6_pass = bool(ok6)
    if not ok6:
        rep.notes.append("DoD-6 failed: a flagship result overspent its declared epsilon")

    # ---- DoD-7: extreme-epsilon invariants (needs a sweep) ------------
    ok7 = True
    for ds in datasets:
        pts = {(round(r.epsilon, 6)): r for r in flagship if r.dataset == ds}
        if 10.0 in pts and 0.05 in pts:
            hi = pts[10.0]
            lo = pts[0.05]
            u_nodp_hi = hi.accuracy + hi.utility_gap
            if abs(hi.accuracy - u_nodp_hi) > DOD7_HIGH_EPS_GAP:
                ok7 = False
            if lo.accuracy > DOD7_LOW_EPS_CAP:
                ok7 = False
    if any(10.0 in {(round(r.epsilon, 6)) for r in flagship if r.dataset == ds} for ds in datasets):
        rep.dod7_pass = bool(ok7)
        if not ok7:
            rep.notes.append("DoD-7 failed: extreme-epsilon invariants violated")
    else:
        rep.notes.append("DoD-7 report-only: need epsilon in {0.05, 10}")

    rep.passed = all(
        [
            rep.dod1_pass,
            rep.dod2_pass,
            rep.dod3_pass,
            rep.dod4_pass,
            rep.dod5_pass,
            rep.dod6_pass,
            rep.dod7_pass,
        ]
    )
    return rep


def _eps_min(results: list[EvalResult], target_gap: float = 0.05) -> float | None:
    best = None
    for r in sorted(results, key=lambda x: x.epsilon):
        u_nodp = r.accuracy + r.utility_gap
        if r.accuracy >= u_nodp - target_gap and (best is None or r.epsilon < best):
            best = float(r.epsilon)
    return best
