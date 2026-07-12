"""Step 2 - what does benign drift do to a static detector?

Freeze an IsolationForest on the first 11 capture weeks, calibrate its threshold
to a 1% alarm rate on that training period, then score the remaining 29 weeks.
The traffic has no labelled attacks, so every alarm the frozen model raises is a
false alarm - and we watch that false-alarm rate track the benign drift.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = os.environ.get("CESNET_DATA_DIR", "data/cesnet_institutions/agg_1_day")
START = pd.Timestamp("2023-10-09")
TRAIN_WEEKS = 11                     # weeks 0-10
N_DAYS = 280
SEED = 42

FEATURES = ["n_flows", "n_packets", "n_bytes",
            "average_n_dest_asn", "average_n_dest_ports", "average_n_dest_ip",
            "tcp_udp_ratio_packets", "dir_ratio_packets", "avg_duration", "avg_ttl"]


def _inst_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def load_cube():
    """Return (n_institutions, n_days, n_features) for full-coverage institutions."""
    cube = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=_inst_id):
        df = pd.read_csv(path).sort_values("id_time")
        if df["id_time"].max() < N_DAYS - 1 or len(df) < N_DAYS:
            continue
        cube.append(df[FEATURES].to_numpy()[:N_DAYS])
    return np.asarray(cube, dtype=float)


def normalise_per_institution(cube, train_days):
    """z-score each institution's features using its own training-period statistics only."""
    mu = cube[:, :train_days].mean(axis=1, keepdims=True)
    sd = cube[:, :train_days].std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (cube - mu) / sd


def main():
    cube = load_cube()
    n_inst = cube.shape[0]
    train_days = TRAIN_WEEKS * 7
    print(f"loaded {n_inst} institutions x {cube.shape[1]} days x {cube.shape[2]} features")

    z = normalise_per_institution(cube, train_days)
    z = np.clip(np.nan_to_num(z), -50, 50)

    train_X = z[:, :train_days].reshape(-1, z.shape[2])
    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(train_X)

    scores = model.score_samples(z.reshape(-1, z.shape[2])).reshape(n_inst, N_DAYS)
    threshold = np.percentile(model.score_samples(train_X), 1.0)   # 1% training alarm rate
    alarms = scores < threshold

    n_weeks = N_DAYS // 7
    far = np.array([alarms[:, w * 7:(w + 1) * 7].mean() for w in range(n_weeks)])

    step1 = pd.read_csv("results/step1_benign_drift.csv")
    js = step1.set_index("capture_week")["js_n_flows"]
    test_weeks = np.arange(TRAIN_WEEKS, n_weeks)
    drift_weeks = [w for w in test_weeks if js.get(w, 0) >= js.loc[test_weeks].quantile(0.66)]

    train_far = far[:TRAIN_WEEKS].mean()
    test_far = far[TRAIN_WEEKS:].mean()
    drift_far = far[drift_weeks].mean()
    peak_week = int(TRAIN_WEEKS + np.argmax(far[TRAIN_WEEKS:]))

    out = pd.DataFrame({
        "capture_week": np.arange(n_weeks),
        "week_start": (START + pd.to_timedelta(np.arange(n_weeks) * 7, unit="D")).date,
        "false_alarm_rate": far,
        "is_drift_week": [w in drift_weeks for w in range(n_weeks)],
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/step2_detector_decay.csv", index=False)

    print(f"\ncalibrated training false-alarm rate : {train_far*100:.1f}%")
    print(f"average false-alarm rate, weeks 11-39: {test_far*100:.1f}%  ({test_far/train_far:.1f}x calibrated)")
    print(f"average on drift weeks {drift_weeks}: {drift_far*100:.1f}%")
    print(f"peak false-alarm rate: {far[peak_week]*100:.1f}% at week {peak_week} ({out.week_start[peak_week]})")
    print("\nweek  start        FAR    drift?")
    for _, r in out.iterrows():
        flag = "  <-- drift" if r.is_drift_week else ""
        print(f"{int(r.capture_week):>4}  {r.week_start}  {r.false_alarm_rate*100:>5.1f}%{flag}")
    print("wrote results/step2_detector_decay.csv")


if __name__ == "__main__":
    main()
