"""Render every figure in the README from the result CSVs written by the experiments."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})
os.makedirs("figures", exist_ok=True)


def step1():
    d = pd.read_csv("results/step1_benign_drift.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d.capture_week, d.js_n_flows, "-o", ms=3, color="#c0392b", label="n_flows")
    ax.plot(d.capture_week, d.js_n_packets, "-o", ms=3, color="#2980b9", label="n_packets")
    ax.fill_between(d.capture_week, 0, d.js_n_flows, color="#c0392b", alpha=0.10)
    peak = int(d.js_n_flows.idxmax())
    ax.annotate(f"end of spring semester\n{d.week_start[peak]}  ({d.js_n_flows[peak]:.3f})",
                (peak, d.js_n_flows[peak]), xytext=(peak-13, d.js_n_flows[peak]-0.02),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlabel("capture week (0 = 2023-10-09)")
    ax.set_ylabel("Jensen-Shannon divergence\nvs. week 0 (bits)")
    ax.set_title("Step 1 - benign week-over-week drift in CESNET-TimeSeries24", loc="left", weight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/step1_benign_drift.png"); plt.close(fig)


def step2():
    d = pd.read_csv("results/step2_detector_decay.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = np.where(d.is_drift_week, "#c0392b", "#95a5a6")
    ax.bar(d.capture_week, d.false_alarm_rate * 100, color=colors, width=0.8)
    ax.axhline(1.0, color="#27ae60", lw=1.2, ls="--", label="calibrated 1% rate")
    ax.axvline(10.5, color="0.3", lw=1.0, ls=":")
    ax.text(4.5, ax.get_ylim()[1]*0.9, "training\n(weeks 0-10)", fontsize=8.5, ha="center", color="0.3")
    ax.set_xlabel("capture week"); ax.set_ylabel("false-alarm rate (%)")
    ax.set_title("Step 2 - a frozen IsolationForest's false-alarm rate (red = top-third JS weeks)",
                 loc="left", weight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/step2_detector_decay.png"); plt.close(fig)


def step2b():
    d = pd.read_csv("results/step2b_deeper_analysis.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(d.capture_week, d.pct_institutions_off_baseline * 100,
              color=np.where(d.is_drift_week, "#8e44ad", "#b2babb"), width=0.8)
    ax[0].set_title("(a) institutions off their OWN baseline (>2 IQR)", loc="left", weight="bold", fontsize=10)
    ax[0].set_xlabel("capture week"); ax[0].set_ylabel("% of institutions")
    ax[1].plot(d.capture_week, d.far_frozen_forecaster * 100, "-o", ms=3, color="#c0392b", label="frozen seasonal profile")
    ax[1].plot(d.capture_week, d.far_rolling_forecaster * 100, "-o", ms=3, color="#16a085", label="rolling (blind adaptation)")
    ax[1].set_title("(b) forecast-residual detectors under drift", loc="left", weight="bold", fontsize=10)
    ax[1].set_xlabel("capture week"); ax[1].set_ylabel("false-alarm rate (%)"); ax[1].legend()
    fig.tight_layout(); fig.savefig("figures/step2b_deeper_analysis.png"); plt.close(fig)


def step3():
    if not os.path.exists("results/step3_separability.csv"):
        return
    s = pd.read_csv("results/step3_separability.csv")
    w = pd.read_csv("results/step3_window_signals.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    labels = {"magnitude_only": "shift magnitude\n(naive)", "shape_only": "shift shape\n(ours)",
              "all_signals": "both"}
    x = np.arange(len(s))
    ax[0].bar(x - 0.2, s.roc_auc, 0.4, color="#2c3e50", label="ROC-AUC")
    ax[0].bar(x + 0.2, s.pr_auc, 0.4, color="#c0392b", label="PR-AUC")
    ax[0].axhline(s.base_rate.iloc[0], color="#7f8c8d", ls="--", lw=1, label="random PR-AUC (base rate)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([labels.get(n, n) for n in s.signal_set], fontsize=9)
    ax[0].set_ylabel("score"); ax[0].set_ylim(0, 1)
    ax[0].set_title("(a) separating real attacks from benign drift (UGR'16)", loc="left", weight="bold", fontsize=10)
    ax[0].legend(fontsize=8)

    benign = w[w.is_attack == 0]["concentration"]
    attack = w[w.is_attack == 1]["concentration"]
    ax[1].hist(benign, bins=40, density=True, alpha=0.6, color="#16a085", label="benign drift")
    ax[1].hist(attack, bins=40, density=True, alpha=0.6, color="#c0392b", label="attack")
    ax[1].set_xlabel("deviation concentration across features")
    ax[1].set_ylabel("density")
    ax[1].set_title("(b) attacks concentrate the shift on fewer features", loc="left", weight="bold", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/step3_separability.png"); plt.close(fig)


def step3b():
    if not (os.path.exists("results/step3b_signals.csv") and
            os.path.exists("results/step3b_adversary.csv")):
        return
    s = pd.read_csv("results/step3b_signals.csv")
    a = pd.read_csv("results/step3b_adversary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    colors = ["#95a5a6", "#16a085", "#2980b9", "#8e44ad", "#c0392b"]
    ax[0].bar(range(len(s)), s.roc_auc, color=colors[:len(s)])
    ax[0].set_xticks(range(len(s))); ax[0].set_xticklabels(s.signal_set, rotation=20, fontsize=8.5)
    ax[0].set_ylim(0.5, 0.82); ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("(a) influence-geometry adds to shape signals", loc="left", weight="bold", fontsize=10)
    for i, v in enumerate(s.roc_auc):
        ax[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)

    ax[1].plot(a.attack_kept * 100, a.roc_auc, "-o", color="#c0392b", ms=5)
    ax[1].axhline(0.5, color="0.5", ls="--", lw=1, label="chance")
    ax[1].set_xlabel("attack intensity the adversary keeps (%)")
    ax[1].set_ylabel("detector ROC-AUC")
    ax[1].set_ylim(0.5, 0.8); ax[1].invert_xaxis()
    ax[1].set_title("(b) adaptive adversary: evasion costs attack intensity", loc="left", weight="bold", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/step3b_deep.png"); plt.close(fig)


def step3c():
    if not os.path.exists("results/step3c_per_family.csv"):
        return
    d = pd.read_csv("results/step3c_per_family.csv").sort_values("roc_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#16a085" if v >= 0.75 else "#e67e22" if v >= 0.6 else "#c0392b" for v in d.roc_auc]
    yerr = None
    if {"roc_auc_min", "roc_auc_max"} <= set(d.columns):
        yerr = [d.roc_auc - d.roc_auc_min, d.roc_auc_max - d.roc_auc]
    ax.bar(d.family, d.roc_auc, color=colors, width=0.6, yerr=yerr, capsize=4,
           error_kw=dict(ecolor="0.25", lw=1))
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="chance")
    for x, v in zip(range(len(d)), d.roc_auc):
        ax.text(x + 0.2, v + 0.02, f"{v:.2f}", ha="left", fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("ROC-AUC")
    ax.set_title("Step 3c - separability per attack family on UGR'16 (median and range over 30 CV splits)",
                 loc="left", weight="bold", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/step3c_per_family.png"); plt.close(fig)


def step1b():
    if not os.path.exists("results/step1b_calibrated_drift.csv"):
        return
    d = pd.read_csv("results/step1b_calibrated_drift.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(d.capture_week, d.js_n_flows, "-o", ms=3, color="#c0392b", label="observed JS vs week 0")
    ax[0].plot(d.capture_week, d.js_null_p99, "--", color="0.35", lw=1.2, label="paired-permutation null, 99th pct")
    ax[0].fill_between(d.capture_week, 0, d.js_null_p99, color="0.6", alpha=0.18)
    ax[0].set_xlabel("capture week"); ax[0].set_ylabel("JS divergence (bits)")
    ax[0].set_title("(a) pooled JS on daily data stays inside its own noise", loc="left", weight="bold", fontsize=10)
    ax[0].legend(fontsize=8)
    pct = 100 * (10 ** d.mean_log10_change - 1)
    sig = d.level_significant_fdr.astype(bool)
    ax[1].bar(d.capture_week[~sig], pct[~sig], color="#b2babb", width=0.8, label="not significant")
    ax[1].bar(d.capture_week[sig], pct[sig], color="#2c3e50", width=0.8, label="significant, BH-FDR 5%")
    ax[1].axhline(0, color="0.3", lw=0.8)
    ax[1].set_xlabel("capture week"); ax[1].set_ylabel("mean volume change vs week 0 (%)")
    ax[1].set_title("(b) paired per-institution test: drift in 33 of 39 weeks", loc="left", weight="bold", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/step1b_calibrated_drift.png"); plt.close(fig)


def step3d():
    if not os.path.exists("results/step3d_magnitude_matched.csv"):
        return
    d = pd.read_csv("results/step3d_magnitude_matched.csv")
    m = d[d.setting == "magnitude_matched"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["#95a5a6", "#16a085"]
    err = [m.roc_auc - m.roc_ci_lo, m.roc_ci_hi - m.roc_auc]
    ax.bar(["shift magnitude", "shift shape"], m.roc_auc, color=colors, width=0.55,
           yerr=err, capsize=5, error_kw=dict(ecolor="0.25", lw=1.1))
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="chance")
    for i, v in enumerate(m.roc_auc):
        ax.text(i + 0.3, v, f"{v:.2f}", va="center", fontsize=9)
    ax.set_ylim(0.4, 0.85); ax.set_ylabel("ROC-AUC (95% day-block CI)")
    ax.set_title("Step 3d - attacks vs benign shift of the same size", loc="left", weight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig("figures/step3d_magnitude_matched.png"); plt.close(fig)


if __name__ == "__main__":
    step1(); step1b(); step2(); step2b(); step3(); step3b(); step3c(); step3d()
    print("wrote figures for steps 1, 1b, 2, 2b, 3, 3b, 3c and 3d")
