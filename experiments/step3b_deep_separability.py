"""Step 3b - deeper separability: influence-geometry signals + an adaptive adversary.

Two things Step 3 left open:

1. Gradient / influence geometry. Step 3's signals are model-free (shape of the
   shift). Here a small differentiable autoencoder is trained on benign traffic;
   for each minute we measure how its update gradient behaves - its norm (how hard
   the point pushes the model) and its alignment with the average benign gradient.
   A poisoning-shaped move misaligns. This is the signal an IsolationForest cannot
   give (it has no gradient); we add it and ask whether it improves separability.

2. Adaptive adversary. A real attacker who knows the detector will try to make an
   attack look like benign drift. We model an attacker that spends a fraction lambda
   of the window on benign "cover traffic" to spread and smooth the shift. That
   lowers the shape signal but also dilutes the attack. We sweep lambda and report
   the Pareto trade-off: detector ROC-AUC vs. how much attack the adversary keeps.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from step3_separability import (load, standardise, window_signals, window_labels,
                                 WINDOW, STEP, SEED)

rng = np.random.default_rng(SEED)


# ----------------------------- tiny numpy autoencoder -----------------------------
def train_autoencoder(X, hidden=16, epochs=15, batch=256, lr=1e-3):
    n, d = X.shape
    s = 1.0 / np.sqrt(d)
    W1 = rng.normal(0, s, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, d)); b2 = np.zeros(d)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for _ in range(epochs):
        for i in range(0, n, batch):
            xb = X[i:i + batch]
            h = np.tanh(xb @ W1 + b1)
            xhat = h @ W2 + b2
            r = xhat - xb
            gW2 = h.T @ (2 * r) / len(xb)
            gb2 = (2 * r).mean(0)
            dh = (2 * r) @ W2.T * (1 - h ** 2)
            gW1 = xb.T @ dh / len(xb)
            gb1 = dh.mean(0)
            grads = [gW1, gb1, gW2, gb2]
            t += 1
            for j in range(4):
                m[j] = 0.9 * m[j] + 0.1 * grads[j]
                v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
                mh = m[j] / (1 - 0.9 ** t); vh = v[j] / (1 - 0.999 ** t)
                params[j] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return params


def gradient_signals(params, X, ref_R, ref_norm_stats):
    """Per-sample last-layer gradient norm (z-scored) and alignment with benign mean."""
    W1, b1, W2, b2 = params
    h = np.tanh(X @ W1 + b1)
    r = (h @ W2 + b2) - X                       # reconstruction residual
    rn = np.linalg.norm(r, axis=1); hn = np.linalg.norm(h, axis=1)
    g_norm = rn * hn                            # ||outer(r,h)||_F, up to a constant
    Rh = h @ ref_R.T                            # (n, d)
    align = (r * Rh).sum(1) / (rn * hn * np.linalg.norm(ref_R) + 1e-12)
    med, mad = ref_norm_stats
    g_norm_z = (g_norm - med) / (mad + 1e-9)
    return g_norm_z, align


def benign_reference(params, Xb):
    W1, b1, W2, b2 = params
    h = np.tanh(Xb @ W1 + b1)
    r = (h @ W2 + b2) - Xb
    R = r.T @ h / len(Xb)                        # mean benign outer(r,h)
    g_norm = np.linalg.norm(r, axis=1) * np.linalg.norm(h, axis=1)
    return R, (np.median(g_norm), np.median(np.abs(g_norm - np.median(g_norm))))


def add_gradient_signals(sig, gnz, align, times):
    """Aggregate per-minute gradient signals into the per-window signal frame."""
    gn_w, al_w = [], []
    for s in range(0, len(gnz) - WINDOW + 1, STEP):
        gn_w.append(gnz[s:s + WINDOW].mean())
        al_w.append(align[s:s + WINDOW].mean())
    out = sig.copy()
    out["grad_norm_z"] = gn_w
    out["grad_align"] = al_w
    return out


def cv_auc(df, y, groups, cols):
    X = df[cols].to_numpy()
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    aps, aucs = [], []
    for tr, te in skf.split(X, y, groups):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], p)); aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aps)), float(np.mean(aucs))


def main():
    xtr, xte, yte, feats = load()
    dev = standardise(xtr, xte, feats)
    dev = np.clip(dev, -10, 10)
    times = pd.to_datetime(xte["Row"].astype(str), format="%Y%m%d%H%M").to_numpy()

    # benign training signal for the autoencoder: standardise calibration on itself
    tr = np.log1p(xtr[feats].to_numpy(float)); mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1
    Xtr_std = np.clip((tr - mu) / sd, -10, 10)
    sub = rng.choice(len(Xtr_std), min(20000, len(Xtr_std)), replace=False)
    params = train_autoencoder(Xtr_std[sub])
    ref_R, ref_stats = benign_reference(params, Xtr_std[sub])

    sig = window_signals(dev, times)
    y = window_labels(yte, feats, times)
    groups = sig["t"].to_numpy().astype("datetime64[D]")
    gnz, align = gradient_signals(params, dev, ref_R, ref_stats)
    sig = add_gradient_signals(sig, gnz, align, times)

    shape = ["concentration", "top_feature_share", "abruptness"]
    grad = ["grad_norm_z", "grad_align"]
    print(f"windows {len(sig)}  attacks {y.sum()} ({100*y.mean():.1f}%)")
    print("\n(1) does influence geometry add to the model-free shape signals?")
    rows = []
    for name, cols in [("magnitude", ["magnitude"]), ("shape", shape),
                       ("gradient", grad), ("shape+gradient", shape + grad),
                       ("all", ["magnitude"] + shape + grad)]:
        ap, auc = cv_auc(sig, y, groups, cols)
        print(f"    {name:<16} PR-AUC {ap:.3f}   ROC-AUC {auc:.3f}")
        rows.append({"signal_set": name, "pr_auc": ap, "roc_auc": auc})
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/step3b_signals.csv", index=False)

    # (2) adaptive adversary: dilute attack windows with benign cover traffic
    print("\n(2) adaptive adversary (cover-traffic) - Pareto: kept attack vs detector ROC-AUC")
    attack_min = (yte[["labeldos", "labelscan11", "labelscan44", "labelnerisbotnet",
                       "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyspam"]] > 0).any(axis=1).to_numpy()
    benign_idx = np.where(~attack_min)[0]
    benign_profile = dev[benign_idx].mean(0)   # coherent "average benign" cover, no injected jitter
    use = shape  # the model-free signals the adversary tries to spoof
    pareto = []
    atk_rows = np.where(attack_min)[0]
    for lam in [0.0, 0.2, 0.4, 0.6, 0.8]:
        dev_adv = dev.copy()
        dev_adv[atk_rows] = (1 - lam) * dev[atk_rows] + lam * benign_profile
        sig_adv = window_signals(dev_adv, times)
        # train on clean signals, evaluate on adversarial ones, day-grouped
        X, Xa = sig[use].to_numpy(), sig_adv[use].to_numpy()
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        for trn, tst in skf.split(X, y, groups):
            if y[trn].sum() == 0 or y[tst].sum() == 0:
                continue
            m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(X[trn], y[trn])
            aucs.append(roc_auc_score(y[tst], m.predict_proba(Xa[tst])[:, 1]))
        auc = float(np.mean(aucs))
        pareto.append({"lambda_cover": lam, "attack_kept": 1 - lam, "roc_auc": auc})
        print(f"    cover={lam:.1f}  attack kept {100*(1-lam):.0f}%  ROC-AUC {auc:.3f}")
    pd.DataFrame(pareto).to_csv("results/step3b_adversary.csv", index=False)
    print("\nwrote results/step3b_signals.csv and results/step3b_adversary.csv")


if __name__ == "__main__":
    main()
