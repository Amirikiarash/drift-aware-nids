# Telling Drift from Attack

Measuring benign non-stationarity in real network traffic, as groundwork for
intrusion detection that can tell legitimate change from adversarial manipulation.

Deployed ML-based intrusion detectors face two kinds of distribution shift that
look alike but demand opposite responses: benign concept drift (the network
legitimately evolving) and adversarially induced shift (evasion or poisoning).
Before building anything that separates the two, the benign side has to be
measured on real traffic. That is what this repository does, step by step.

## Step 1 — How much does real traffic drift on its own?

`experiments/step1_benign_drift.py` quantifies week-over-week distribution shift
in [CESNET-TimeSeries24](https://doi.org/10.1038/s41597-025-04603-x): 40 weeks of
real ISP traffic from 275 institutions (Oct 2023 - Jul 2024). For 50 randomly
selected institutions it computes the Jensen-Shannon divergence of the daily
`n_flows` and `n_packets` distributions of every capture week against week 0
(log-scaled, 60 shared histogram bins).

### Result

Benign drift is real, sustained, and it aligns with the calendar. The
most-shifted weeks reach ~0.19 bits of JS divergence with no attack involved,
and they coincide with the Czech academic calendar:

| capture week | JS(n_flows) | likely event |
|---|---|---|
| 11 | 0.160 | Christmas break |
| 32-33 | 0.13-0.19 | end of spring semester |
| 38-39 | 0.16-0.155 | start of summer vacation |

A detection model whose notion of "normal" is frozen at week 0 is already
operating on a measurably different network by these weeks. Output plot and CSV
are in `results/`.

## Run it

```bash
pip install -r requirements.txt
python experiments/step1_benign_drift.py
```

The dataset part used here (~10 MB, institutions / 1-day aggregation) downloads
automatically on first run via `cesnet-tszoo`. Set `CESNET_DATA_ROOT` to change
the download directory.

## Roadmap

- Step 2: label drift weeks with dated calendar events; show a baseline
  detector's accuracy decaying across exactly these weeks.
- Step 3: inject problem-space adversarial shift with logged ground truth and
  test whether benign and adversarial shift are statistically separable online.

## Data

Koumar, Hynek et al., *CESNET-TimeSeries24: Time Series Dataset for Network
Traffic Anomaly Detection and Forecasting*, Scientific Data (2025).
Loaded via the [cesnet-tszoo](https://github.com/CESNET/cesnet-tszoo) library.

## Author

Kiarash Amiri — amiri.kiarash@outlook.com — [linkedin.com/in/kiarash-amiri](https://www.linkedin.com/in/kiarash-amiri)

Seven years running network and security operations for a 3,000-server,
50-site retail estate; M.Sc. in Secure Computing. This project grew out of
watching detection rules go stale in production.
