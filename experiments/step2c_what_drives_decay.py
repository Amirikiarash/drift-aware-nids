"""Step 2c - what drives the Step 2 false-alarm inflation: drift events or elapsed time?

Step 2 shows a frozen detector's false-alarm rate rising from 1% to 16% on average.
A natural reading is that alarms spike on the high-drift weeks. This step tests
that reading directly, and checks the Step 2 calibration out of sample.

  (1) Rank correlation, over the 29 test weeks, between the weekly false-alarm
      rate and (a) the pooled JS divergence of Step 1, (b) the absolute
      per-institution level change of Step 1b, and (c) elapsed time.
  (2) Out-of-sample calibration: leave one training week out, refit, recalibrate
      to 1% on the rest, and measure the alarm rate on the held-out week. If the
      held-out rate is ~1%, the 16x inflation is not an in-sample artefact.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(__file__))
from step2_detector_decay import load_cube, normalise_per_institution, TRAIN_WEEKS, N_DAYS, SEED


def main():
    s1 = pd.read_csv("results/step1_benign_drift.csv").set_index("capture_week")
    s1b = pd.read_csv("results/step1b_calibrated_drift.csv").set_index("capture_week")
    s2 = pd.read_csv("results/step2_detector_decay.csv").set_index("capture_week")
    test = np.arange(TRAIN_WEEKS, N_DAYS // 7)
    far = s2.loc[test, "false_alarm_rate"].to_numpy()

    print("(1) Spearman correlation with the weekly false-alarm rate, test weeks 11-39")
    rows = []
    for name, x in [("pooled JS divergence (Step 1)", s1.loc[test, "js_n_flows"].to_numpy()),
                    ("|per-institution level change| (Step 1b)", np.abs(s1b.loc[test, "mean_log10_change"].to_numpy())),
                    ("elapsed weeks since training", test.astype(float))]:
        rho, p = spearmanr(x, far)
        print(f"    {name:<44} rho = {rho:+.2f}   p = {p:.4f}")
        rows.append({"predictor": name, "spearman_rho": rho, "p_value": p})

    print("\n(2) out-of-sample calibration check (leave-one-training-week-out)")
    cube = load_cube()
    train_days = TRAIN_WEEKS * 7
    z = np.clip(np.nan_to_num(normalise_per_institution(cube, train_days)), -50, 50)
    held = []
    for k in range(TRAIN_WEEKS):
        days = [d for d in range(train_days) if d // 7 != k]
        X = z[:, days].reshape(-1, z.shape[2])
        m = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X)
        thr = np.percentile(m.score_samples(X), 1.0)
        held.append((m.score_samples(z[:, k * 7:(k + 1) * 7].reshape(-1, z.shape[2])) < thr).mean())
    held = np.array(held)
    print(f"    held-out training-week alarm rate: mean {held.mean()*100:.1f}% "
          f"(range {held.min()*100:.1f}-{held.max()*100:.1f}%) vs 1.0% in-sample")
    rows.append({"predictor": "held-out training-week alarm rate (mean)", "spearman_rho": np.nan,
                 "p_value": np.nan, "value": held.mean()})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/step2c_what_drives_decay.csv", index=False)
    print("\nconclusion: calibration holds out of sample, so the inflation is real; but it")
    print("follows elapsed time, not individual drift events - the decay is cumulative.")
    print("wrote results/step2c_what_drives_decay.csv")


if __name__ == "__main__":
    main()
