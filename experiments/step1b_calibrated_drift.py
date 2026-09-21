"""Step 1b - is the Step 1 drift distinguishable from sampling noise?

Step 1 reports Jensen-Shannon divergences with no null distribution, so a number
like 0.198 bits cannot be read on its own: two samples drawn from the SAME week
also have non-zero JS divergence. This step calibrates the measurement.

Design. The same 50 institutions are observed every week, so week 0 and week w
are paired by institution. Under the null "week w is distributed like week 0",
each institution's week-0 block and week-w block are exchangeable. Two paired
tests follow from that:

  (a) distributional test - statistic: pooled JS divergence (the Step 1 number);
      null: randomly swap each institution's week-0 and week-w blocks.
  (b) level test - statistic: mean over institutions of the per-institution
      change in weekly mean log-volume; null: random sign flips per institution.

Both are exact permutation tests that respect the within-institution correlation
of daily values (an iid bootstrap ignores it and is anti-conservative). With 39
weeks tested, p-values are corrected with Benjamini-Hochberg at FDR 5%.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from step1_benign_drift import (load_series, select_institutions, js_divergence,
                                N_BINS, DAYS_PER_WEEK, START)

SEED = 42
N_PERM_JS = 1000
N_PERM_LEVEL = 20000
FDR = 0.05


def benjamini_hochberg(p, q=FDR):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] <= q * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(ranked)[0]) + 1 if ranked.any() else 0
    keep = np.zeros(m, dtype=bool)
    keep[order[:k]] = True
    return keep


def main():
    rng = np.random.default_rng(SEED)
    flows, _ = load_series("n_flows")
    keep = select_institutions(flows)
    logv = np.log10(flows[keep] + 1.0)                     # (50 institutions, 280 days)
    n_inst = logv.shape[0]
    n_weeks = logv.shape[1] // DAYS_PER_WEEK
    edges = np.linspace(logv.min(), logv.max(), N_BINS + 1)

    def hist(x):
        h = np.histogram(x, bins=edges)[0].astype(float)
        return h / h.sum()

    def block(w):
        return logv[:, w * DAYS_PER_WEEK:(w + 1) * DAYS_PER_WEEK]

    x0 = block(0)
    rows = []
    for w in range(1, n_weeks):
        xw = block(w)

        # (a) pooled JS divergence, institution-swap permutation
        js_obs = js_divergence(hist(x0.ravel()), hist(xw.ravel()))
        swaps = rng.random((N_PERM_JS, n_inst)) < 0.5
        js_null = np.array([
            js_divergence(hist(np.where(s[:, None], x0, xw).ravel()),
                          hist(np.where(s[:, None], xw, x0).ravel()))
            for s in swaps])
        p_js = (np.sum(js_null >= js_obs) + 1) / (N_PERM_JS + 1)

        # (b) per-institution level change, sign-flip permutation
        d = xw.mean(axis=1) - x0.mean(axis=1)              # change in weekly mean log10 volume
        flips = rng.choice([-1.0, 1.0], size=(N_PERM_LEVEL, n_inst))
        lvl_null = np.abs((flips * d).mean(axis=1))
        p_lvl = (np.sum(lvl_null >= abs(d.mean())) + 1) / (N_PERM_LEVEL + 1)

        rows.append({
            "capture_week": w,
            "week_start": (START + pd.Timedelta(days=7 * w)).date(),
            "js_n_flows": js_obs,
            "js_null_p99": np.percentile(js_null, 99),
            "p_js": p_js,
            "mean_log10_change": d.mean(),
            "pct_institutions_up": 100 * np.mean(d > 0),
            "p_level": p_lvl,
        })

    out = pd.DataFrame(rows)
    out["js_significant_fdr"] = benjamini_hochberg(out.p_js)
    out["level_significant_fdr"] = benjamini_hochberg(out.p_level)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/step1b_calibrated_drift.csv", index=False)

    n = len(out)
    peak = out.loc[out.js_n_flows.idxmax()]
    print(f"{n_inst} institutions, {n} weeks tested against week 0, BH-FDR {int(FDR*100)}%\n")
    print("(a) pooled JS divergence (the Step 1 statistic)")
    print(f"    weeks with uncorrected p < 0.05 : {int((out.p_js < 0.05).sum())}/{n}")
    print(f"    weeks significant after FDR     : {int(out.js_significant_fdr.sum())}/{n}")
    print(f"    peak week {int(peak.capture_week)}: JS {peak.js_n_flows:.3f} bits, "
          f"null p99 {peak.js_null_p99:.3f}, p = {peak.p_js:.3f}")
    print("(b) per-institution level change (paired sign-flip)")
    print(f"    weeks significant after FDR     : {int(out.level_significant_fdr.sum())}/{n}")
    w33 = out.set_index("capture_week").loc[33]
    print(f"    week 33: mean change {w33.mean_log10_change:+.3f} log10 units "
          f"({100*(10**w33.mean_log10_change-1):+.0f}% volume), "
          f"{w33.pct_institutions_up:.0f}% of institutions up, p = {w33.p_level:.5f}")
    print("\nconclusion: the drift is real, but pooled JS on daily data is too weak a statistic")
    print("to show it week by week; the paired level test is the calibrated measurement.")
    print("wrote results/step1b_calibrated_drift.csv")


if __name__ == "__main__":
    main()
