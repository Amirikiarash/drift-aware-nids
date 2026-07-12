"""Render the figures for Steps 1, 2 and 2b from the committed result CSVs."""
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
    ax.set_title("Step 2 - a frozen IsolationForest's false alarms track benign drift (red = drift weeks)",
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


if __name__ == "__main__":
    step1(); step2(); step2b()
    print("wrote figures/step1_benign_drift.png, step2_detector_decay.png, step2b_deeper_analysis.png")
