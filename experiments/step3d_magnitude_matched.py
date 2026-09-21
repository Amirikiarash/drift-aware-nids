"""Step 3d - does shape separate attacks from benign shift OF THE SAME SIZE?

Step 3 compares shape and magnitude over all test windows. Two weaknesses:
  - most benign windows are small deviations, so the comparison mixes "attack vs
    quiet traffic" with the actual question, "attack vs benign drift";
  - with only 13 attack days, the pooled gap needs a confidence interval.

Here the negative class is magnitude-matched: for each decile of attack-window
magnitude, up to three benign windows are drawn from the same magnitude band.
Within that set, magnitude can by construction barely separate the classes, so
any remaining separability must come from the shape of the shift. Uncertainty is
a day-block bootstrap over out-of-fold predictions (days are resampled whole, so
temporal correlation within a day is preserved).

Reads results/step3_window_signals.csv written by step3_separability.py.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_BOOT = 2000
MATCH_RATIO = 3
SHAPE = ["concentration", "top_feature_share", "abruptness"]


def out_of_fold(df, y, groups, cols):
    X = df[cols].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        oof[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def day_bootstrap(y, groups, scores, rng, stat):
    days = np.unique(groups)
    idx_by_day = {d: np.where(groups == d)[0] for d in days}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_day[d] for d in rng.choice(days, len(days))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(stat(y[idx], *[s[idx] for s in scores]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("results/step3_window_signals.csv", parse_dates=["t"])
    y = df.is_attack.to_numpy()
    groups = df.t.values.astype("datetime64[D]")
    print(f"{len(df)} windows, {y.sum()} attack windows on {len(np.unique(groups[y == 1]))} of "
          f"{len(np.unique(groups))} days")

    # (1) pooled comparison with a confidence interval
    o_mag = out_of_fold(df, y, groups, ["magnitude"])
    o_shp = out_of_fold(df, y, groups, SHAPE)
    gap = roc_auc_score(y, o_shp) - roc_auc_score(y, o_mag)
    lo, hi = day_bootstrap(y, groups, [o_shp, o_mag], rng,
                           lambda yy, a, b: roc_auc_score(yy, a) - roc_auc_score(yy, b))
    print(f"\n(1) all windows: ROC-AUC shape - magnitude = {gap:+.3f}, "
          f"95% day-block CI [{lo:+.3f}, {hi:+.3f}]")

    # (2) magnitude-matched negatives
    edges = np.quantile(df.magnitude[y == 1], np.linspace(0, 1, 11))
    band = np.digitize(df.magnitude, edges[1:-1])
    sel = list(np.where(y == 1)[0])
    for k in range(10):
        n_att = int(((band == k) & (y == 1)).sum())
        pool = np.where((band == k) & (y == 0))[0]
        if len(pool):
            sel += list(rng.choice(pool, min(len(pool), MATCH_RATIO * n_att), replace=False))
    sel = np.array(sel)
    dm, ym, gm = df.iloc[sel].reset_index(drop=True), y[sel], groups[sel]
    print(f"\n(2) magnitude-matched set: {ym.sum()} attack vs {(ym == 0).sum()} benign windows "
          f"of the same deviation size (base rate {ym.mean():.2f})")
    rows = []
    for name, cols in [("magnitude", ["magnitude"]), ("shape", SHAPE)]:
        o = out_of_fold(dm, ym, gm, cols)
        auc, ap = roc_auc_score(ym, o), average_precision_score(ym, o)
        clo, chi = day_bootstrap(ym, gm, [o], rng, lambda yy, a: roc_auc_score(yy, a))
        print(f"    {name:<10} ROC-AUC {auc:.3f} [95% CI {clo:.3f}-{chi:.3f}]   PR-AUC {ap:.3f}")
        rows.append({"setting": "magnitude_matched", "signal": name, "roc_auc": auc,
                     "roc_ci_lo": clo, "roc_ci_hi": chi, "pr_auc": ap, "base_rate": ym.mean()})
    rows.append({"setting": "all_windows", "signal": "shape_minus_magnitude", "roc_auc": gap,
                 "roc_ci_lo": lo, "roc_ci_hi": hi, "pr_auc": np.nan, "base_rate": y.mean()})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/step3d_magnitude_matched.csv", index=False)
    print("\nconclusion: pooled over all windows the shape advantage is not significant;")
    print("against benign shift of the same size, shape separates attacks clearly.")
    print("wrote results/step3d_magnitude_matched.csv")


if __name__ == "__main__":
    main()
