"""Step 1 - how much does real traffic drift on its own?

Week-over-week Jensen-Shannon divergence of the daily n_flows / n_packets
distributions in CESNET-TimeSeries24, every capture week against week 0.
No attacks are involved, so any shift measured here is benign.
"""
import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("CESNET_DATA_DIR", "data/cesnet_institutions/agg_1_day")
START = pd.Timestamp("2023-10-09")          # capture day 0
FEATURES = ["n_flows", "n_packets"]
DAYS_PER_WEEK = 7
N_BINS = 60
N_INSTITUTIONS = 50                          # volume-stationary subset, see select_institutions()


def load_series(feature):
    """Return a (n_institutions, n_days) matrix for one feature, full-coverage institutions only."""
    rows, ids = [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=_inst_id):
        df = pd.read_csv(path).sort_values("id_time")
        if df["id_time"].max() < 279 or len(df) < 280:
            continue                          # drop short / onboarding-truncated institutions
        rows.append(df[feature].to_numpy()[:280])
        ids.append(_inst_id(path))
    return np.asarray(rows, dtype=float), np.asarray(ids)


def _inst_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def select_institutions(flows):
    """Pick the N institutions whose traffic volume is most stationary, so the drift we
    later measure is behavioural rather than a sensor coming online (see data/README)."""
    daily_total = flows.sum(axis=1, keepdims=True)
    daily_total[daily_total == 0] = 1.0
    share = flows / daily_total                       # each institution's daily share of its own total
    first4 = share[:, :28].mean(axis=1)
    last4 = share[:, -28:].mean(axis=1)
    volatility = np.abs(np.log((last4 + 1e-9) / (first4 + 1e-9)))
    return np.argsort(volatility)[:N_INSTITUTIONS]


def js_divergence(p, q):
    eps = 1e-12
    p, q = p + eps, q + eps
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def week_distributions(matrix, n_bins):
    """One log-scaled histogram per week, all sharing the global support."""
    logv = np.log10(matrix + 1.0)
    edges = np.linspace(logv.min(), logv.max(), n_bins + 1)
    n_weeks = matrix.shape[1] // DAYS_PER_WEEK
    dists = []
    for w in range(n_weeks):
        block = logv[:, w * DAYS_PER_WEEK:(w + 1) * DAYS_PER_WEEK].ravel()
        h = np.histogram(block, bins=edges)[0].astype(float)
        dists.append(h / h.sum() if h.sum() else h)
    return np.asarray(dists)


def main():
    flows, ids = load_series("n_flows")
    packets, _ = load_series("n_packets")
    print(f"loaded {flows.shape[0]} full-coverage institutions x {flows.shape[1]} days from {DATA_DIR}")

    keep = select_institutions(flows)
    print(f"using {len(keep)} volume-stationary institutions for the drift measurement")

    result = {}
    for name, matrix in [("n_flows", flows[keep]), ("n_packets", packets[keep])]:
        dists = week_distributions(matrix, N_BINS)
        ref = dists[0]
        scores = np.array([js_divergence(ref, d) for d in dists])
        result[name] = scores

    weeks = np.arange(len(result["n_flows"]))
    week_dates = START + pd.to_timedelta(weeks * 7, unit="D")

    out = pd.DataFrame({
        "capture_week": weeks,
        "week_start": week_dates.date,
        "js_n_flows": result["n_flows"],
        "js_n_packets": result["n_packets"],
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/step1_benign_drift.csv", index=False)

    print("\nweek  start        JS(n_flows)  JS(n_packets)")
    for _, r in out.iterrows():
        print(f"{int(r.capture_week):>4}  {r.week_start}   {r.js_n_flows:>9.3f}    {r.js_n_packets:>9.3f}")

    top = out.sort_values("js_n_flows", ascending=False).head(6)
    print("\nmost-shifted weeks by JS(n_flows):")
    for _, r in top.iterrows():
        print(f"  week {int(r.capture_week):>2} ({r.week_start}): {r.js_n_flows:.3f}")
    print(f"\npeak JS(n_flows) = {out.js_n_flows.max():.3f} bits at week {int(out.js_n_flows.idxmax())}")
    print("wrote results/step1_benign_drift.csv")


if __name__ == "__main__":
    main()
