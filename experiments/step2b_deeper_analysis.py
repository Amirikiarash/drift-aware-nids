"""Step 2b - two objections to Step 2, answered on the same real data.

(1) Is the decay just an artefact of pooling many institutions together?
    Measure each institution against its OWN training baseline and count how many
    are off-baseline on the drift weeks vs during training.

(2) Do forecast-residual detectors (the class used in operational anomaly
    prediction) behave differently? Compare a FROZEN seasonal-profile forecaster
    with a ROLLING seasonal-naive one that blindly adapts.
"""
import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("CESNET_DATA_DIR", "data/cesnet_institutions/agg_1_day")
START = pd.Timestamp("2023-10-09")
TRAIN_WEEKS = 11
N_DAYS = 280

FEATURES = ["n_flows", "n_packets", "n_bytes",
            "average_n_dest_asn", "average_n_dest_ports", "average_n_dest_ip",
            "tcp_udp_ratio_packets", "dir_ratio_packets", "avg_duration", "avg_ttl"]


def _inst_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def load_cube(cols):
    cube = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=_inst_id):
        df = pd.read_csv(path).sort_values("id_time")
        if df["id_time"].max() < N_DAYS - 1 or len(df) < N_DAYS:
            continue
        cube.append(df[cols].to_numpy()[:N_DAYS])
    return np.asarray(cube, dtype=float)


def drift_weeks():
    js = pd.read_csv("results/step1_benign_drift.csv").set_index("capture_week")["js_n_flows"]
    test = np.arange(TRAIN_WEEKS, N_DAYS // 7)
    return [w for w in test if js.get(w, 0) >= js.loc[test].quantile(0.66)]


def per_institution_offbaseline(train_days):
    """Fraction of institutions >2 IQR from their own training normal, per week."""
    cube = load_cube(FEATURES)
    mu = cube[:, :train_days].mean(axis=1, keepdims=True)
    sd = cube[:, :train_days].std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    dev = np.abs((cube - mu) / sd).mean(axis=2)                 # (n_inst, n_days) daily deviation
    q1 = np.percentile(dev[:, :train_days], 25, axis=1, keepdims=True)
    q3 = np.percentile(dev[:, :train_days], 75, axis=1, keepdims=True)
    thresh = q3 + 2.0 * (q3 - q1)                               # per-institution 2-IQR fence
    off = dev > thresh
    n_weeks = N_DAYS // 7
    week_off = np.array([off[:, w*7:(w+1)*7].mean(axis=1) for w in range(n_weeks)]).T   # (n_inst, n_weeks)
    return (week_off > 0.5).mean(axis=0)                        # fraction of institutions off-baseline that week


def forecast_residual_far(signal, train_days, mode):
    """False-alarm rate per week for a seasonal forecaster on a per-institution signal.
    mode='frozen' uses the training day-of-week profile forever; mode='rolling' predicts
    from the same weekday of the previous week (blind adaptation)."""
    cube = load_cube([signal])[:, :, 0]
    logv = np.log10(cube + 1.0)
    n_inst, n_days = logv.shape
    dow = np.arange(n_days) % 7

    if mode == "frozen":
        profile = np.array([[logv[i, :train_days][dow[:train_days] == d].mean() for d in range(7)]
                            for i in range(n_inst)])
        pred = np.array([profile[i, dow] for i in range(n_inst)])
    else:  # rolling seasonal-naive: value from the same weekday one week earlier
        pred = np.full_like(logv, np.nan)
        pred[:, 7:] = logv[:, :-7]

    resid = np.abs(logv - pred)
    scale = np.nanstd(resid[:, :train_days], axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    z = resid / scale
    thr = np.nanpercentile(z[:, :train_days], 99)               # 1% training alarm rate
    alarm = z > thr
    n_weeks = n_days // 7
    return np.array([np.nanmean(alarm[:, w*7:(w+1)*7]) for w in range(n_weeks)])


def main():
    train_days = TRAIN_WEEKS * 7
    dw = drift_weeks()

    off = per_institution_offbaseline(train_days)
    print("(1) per-institution drift (own baseline, not pooled)")
    print(f"    off-baseline during training : {off[:TRAIN_WEEKS].mean()*100:.0f}% of institutions")
    print(f"    off-baseline on drift weeks  : {off[dw].mean()*100:.0f}%")
    print(f"    off-baseline at peak week    : {off.max()*100:.0f}% (week {int(off.argmax())})")

    frozen = forecast_residual_far("n_flows", train_days, "frozen")
    rolling = forecast_residual_far("n_flows", train_days, "rolling")
    print("\n(2) forecast-residual detectors")
    print(f"    FROZEN seasonal profile - FAR on drift weeks : {frozen[dw].mean()*100:.0f}%  (peak {frozen[TRAIN_WEEKS:].max()*100:.0f}%)")
    print(f"    ROLLING seasonal-naive  - FAR on drift weeks : {rolling[dw].mean()*100:.0f}%  (peak {rolling[TRAIN_WEEKS:].max()*100:.0f}%)")

    n_weeks = N_DAYS // 7
    out = pd.DataFrame({
        "capture_week": np.arange(n_weeks),
        "week_start": (START + pd.to_timedelta(np.arange(n_weeks)*7, unit="D")).date,
        "pct_institutions_off_baseline": off,
        "far_frozen_forecaster": frozen,
        "far_rolling_forecaster": rolling,
        "is_drift_week": [w in dw for w in range(n_weeks)],
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/step2b_deeper_analysis.csv", index=False)
    print("\nwrote results/step2b_deeper_analysis.csv")


if __name__ == "__main__":
    main()
