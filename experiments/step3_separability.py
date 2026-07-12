"""Step 3 - are benign drift and real attacks separable, beyond shift magnitude?

Uses UGR'16 (real tier-3 ISP traffic with scheduled, labelled attacks). We take
the calibration period as the benign baseline, then walk the test period in short
windows. A naive drift detector only sees how FAR a window has moved from normal
(shift magnitude). We test whether the SHAPE of the move - how concentrated it is
across features and how abruptly it arrives - tells a real attack from benign
drift, when magnitude alone cannot.

Data: github.com/josecamachop/UGR16_FeatureData (per-minute features + per-minute
attack counts in 8 categories). `labelblacklist` is background traffic present in
every single minute, so it is NOT treated as a discrete attack.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

UGR16_DIR = os.environ.get("UGR16_DIR", "data/ugr16")
VARIANT = "UGR16v1"
WINDOW = 10          # minutes per window
STEP = 5             # window hop
SEED = 42

DISCRETE_ATTACKS = ["labeldos", "labelscan11", "labelscan44", "labelnerisbotnet",
                    "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyspam"]


def load():
    xtr = pd.read_csv(os.path.join(UGR16_DIR, f"{VARIANT}.Xtrain.csv"))
    xte = pd.read_csv(os.path.join(UGR16_DIR, f"{VARIANT}.Xtest.csv"))
    yte = pd.read_csv(os.path.join(UGR16_DIR, f"{VARIANT}.Ytest.csv"))
    feats = [c for c in xtr.columns if c != "Row"]
    return xtr, xte, yte, feats


def standardise(train, test, feats):
    """log1p then z-score using calibration (benign) statistics only."""
    tr = np.log1p(train[feats].to_numpy(dtype=float))
    te = np.log1p(test[feats].to_numpy(dtype=float))
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd == 0] = 1.0
    return (te - mu) / sd                      # per-minute deviation from benign baseline


def window_signals(dev, times):
    norm = np.linalg.norm(dev, axis=1)         # how far each minute is from normal
    jump = np.abs(np.diff(norm, prepend=norm[0]))
    jump_scale = np.median(jump) + 1e-9
    n_feat = dev.shape[1]

    rows = []
    for s in range(0, len(dev) - WINDOW + 1, STEP):
        block = dev[s:s + WINDOW]
        a = block ** 2
        pr = (a.sum(1) ** 2) / (a.sum(1) ** 2 / n_feat + (a ** 2).sum(1) + 1e-12)  # participation ratio
        concentration = 1.0 - np.clip((pr.mean() - 1) / (n_feat - 1), 0, 1)        # 0 spread .. 1 concentrated
        absd = np.abs(block)
        top_share = (absd.max(1) / (absd.sum(1) + 1e-12)).mean()                   # share of top feature
        abruptness = jump[s:s + WINDOW].max() / jump_scale
        rows.append({
            "t": times[s + WINDOW // 2],
            "magnitude": norm[s:s + WINDOW].mean(),      # baseline signal (naive drift size)
            "concentration": concentration,              # shape signal
            "top_feature_share": top_share,              # shape signal
            "abruptness": abruptness,                    # temporal signal
        })
    return pd.DataFrame(rows)


def window_labels(yte, feats_times, times):
    attack_minute = (yte[DISCRETE_ATTACKS] > 0).any(axis=1).to_numpy()
    labels = []
    for s in range(0, len(times) - WINDOW + 1, STEP):
        labels.append(int(attack_minute[s:s + WINDOW].any()))
    return np.array(labels)


def evaluate(df, y):
    """Day-grouped 5-fold CV: windows from one day never straddle train/test, and
    attack days are balanced across folds. Avoids the TESSERACT-style leakage a
    random split would introduce, while coping with attacks clustering in time."""
    groups = df["t"].to_numpy().astype("datetime64[D]")
    shape_cols = ["concentration", "top_feature_share", "abruptness"]

    def cv_score(cols):
        X = df[cols].to_numpy()
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        aps, aucs = [], []
        for tr, te in skf.split(X, y, groups):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aps.append(average_precision_score(y[te], p))
            aucs.append(roc_auc_score(y[te], p))
        return (np.mean(aps), np.std(aps), np.mean(aucs)) if aps else None

    out = {
        "magnitude_only": cv_score(["magnitude"]),
        "shape_only": cv_score(shape_cols),
        "all_signals": cv_score(["magnitude"] + shape_cols),
    }
    return out, y.mean()


def main():
    xtr, xte, yte, feats = load()
    print(f"UGR'16 {VARIANT}: calibration {len(xtr)} min (benign baseline), test {len(xte)} min")
    dev = standardise(xtr, xte, feats)
    times = pd.to_datetime(xte["Row"].astype(str), format="%Y%m%d%H%M").to_numpy()

    sig = window_signals(dev, times)
    y = window_labels(yte, feats, times)
    print(f"windows: {len(sig)}  attack windows: {y.sum()} ({100*y.mean():.1f}%)  benign: {(y==0).sum()}")

    res, base_rate = evaluate(sig, y)
    print(f"\nseparability (day-grouped 5-fold CV, base rate {100*base_rate:.1f}% attack)")
    print(f"{'signal set':<16}  PR-AUC (mean+-sd)   ROC-AUC")
    for name, r in res.items():
        if r is None:
            print(f"{name:<16}  (no valid folds)")
        else:
            ap, sd, auc = r
            print(f"{name:<16}  {ap:.3f} +- {sd:.3f}       {auc:.3f}")

    os.makedirs("results", exist_ok=True)
    summary = pd.DataFrame([
        {"signal_set": k, "pr_auc": (v[0] if v else np.nan), "pr_auc_sd": (v[1] if v else np.nan),
         "roc_auc": (v[2] if v else np.nan), "base_rate": base_rate}
        for k, v in res.items()
    ])
    summary.to_csv("results/step3_separability.csv", index=False)
    sig.assign(is_attack=y).to_csv("results/step3_window_signals.csv", index=False)
    print("\nwrote results/step3_separability.csv and results/step3_window_signals.csv")


if __name__ == "__main__":
    main()
