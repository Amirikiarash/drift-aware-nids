"""Step 3c - separability per attack family (worst-family, not pooled).

The pooled Step 3 number can hide that some attacks are easy and others are
nearly invisible. This reports the shape+gradient discriminator's AUROC on UGR'16
one attack family at a time, with a clean benign negative class (windows with no
discrete attack of any kind). The worst family is the honest headline.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from step3_separability import load, standardise, window_signals, WINDOW, STEP
from step3b_deep_separability import (train_autoencoder, gradient_signals,
                                      benign_reference, add_gradient_signals, cv_auc)

FAMILIES = {
    "DoS": "labeldos",
    "port scan (11)": "labelscan11",
    "port scan (44)": "labelscan44",
    "botnet (neris)": "labelnerisbotnet",
}
MIN_ATTACK_WINDOWS = 15


def family_windows(yte, colname, times):
    """Per-window: +1 if this family is present, 0 if the window is attack-free,
    -1 (dropped) if only OTHER families are present."""
    any_attack = (yte[list(FAMILIES.values())] > 0).any(axis=1).to_numpy()
    fam = (yte[colname] > 0).to_numpy()
    labels = []
    for s in range(0, len(times) - WINDOW + 1, STEP):
        blk_fam = fam[s:s + WINDOW].any()
        blk_any = any_attack[s:s + WINDOW].any()
        labels.append(1 if blk_fam else (0 if not blk_any else -1))
    return np.array(labels)


def main():
    xtr, xte, yte, feats = load()
    dev = np.clip(standardise(xtr, xte, feats), -10, 10)
    times = pd.to_datetime(xte["Row"].astype(str), format="%Y%m%d%H%M").to_numpy()

    tr = np.log1p(xtr[feats].to_numpy(float)); mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1
    Xtr = np.clip((tr - mu) / sd, -10, 10)
    rng = np.random.default_rng(42)
    sub = rng.choice(len(Xtr), min(20000, len(Xtr)), replace=False)
    params = train_autoencoder(Xtr[sub])
    ref_R, ref_stats = benign_reference(params, Xtr[sub])

    sig = window_signals(dev, times)
    gnz, align = gradient_signals(params, dev, ref_R, ref_stats)
    sig = add_gradient_signals(sig, gnz, align, times)
    groups = sig["t"].to_numpy().astype("datetime64[D]")
    cols = ["magnitude", "concentration", "top_feature_share", "abruptness", "grad_norm_z", "grad_align"]

    print("UGR'16 separability per attack family (all signals, day-grouped CV)")
    print(f"{'family':<18} {'attack win':>10} {'PR-AUC':>8} {'ROC-AUC':>8}")
    rows = []
    for name, col in FAMILIES.items():
        y = family_windows(yte, col, times)
        keep = y >= 0
        yk, sigk, grpk = y[keep], sig[keep].reset_index(drop=True), groups[keep]
        if yk.sum() < MIN_ATTACK_WINDOWS:
            print(f"{name:<18} {int(yk.sum()):>10}   (too few windows)")
            continue
        ap, auc = cv_auc(sigk, yk, grpk, cols)
        print(f"{name:<18} {int(yk.sum()):>10} {ap:>8.3f} {auc:>8.3f}")
        rows.append({"family": name, "attack_windows": int(yk.sum()), "pr_auc": ap, "roc_auc": auc})

    df = pd.DataFrame(rows)
    if len(df):
        worst = df.loc[df.roc_auc.idxmin()]
        print(f"\nworst family: {worst.family} (ROC-AUC {worst.roc_auc:.3f}) -- the honest headline")
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/step3c_per_family.csv", index=False)
    print("wrote results/step3c_per_family.csv")


if __name__ == "__main__":
    main()
