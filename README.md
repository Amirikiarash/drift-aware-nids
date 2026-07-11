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
real ISP traffic (Oct 2023 - Jul 2024). For 50 institutions, it computes the
Jensen-Shannon divergence of the daily `n_flows` and `n_packets` distributions
of every capture week against week 0 (log-scaled, 60 shared histogram bins).

**Result.** Benign drift is real, sustained, and it aligns with the calendar.
The most shifted weeks reach ~0.19 bits of JS divergence with no attack
involved, and they coincide with the Czech academic calendar:

| capture week | JS(n_flows) | likely event |
|---|---|---|
| 11 | 0.160 | Christmas break |
| 32-33 | 0.13-0.19 | end of spring semester |
| 38-39 | 0.16-0.155 | start of summer vacation |

![Benign drift in CESNET-TimeSeries24](results/cesnet_drift_week_over_week.png)
<img width="1650" height="750" alt="cesnet_drift_week_over_week" src="https://github.com/user-attachments/assets/e501e83d-96f9-4a14-9abc-b000015ac031" />

## Step 2 — What does that drift do to a static detector?

`experiments/step2_detector_decay.py` freezes an IsolationForest on capture
weeks 0-10 (per-institution features normalised with training-weeks statistics
only), calibrates its threshold to a 1% alarm rate, and scores the remaining 29
weeks. The traffic contains no labelled attacks, so every alarm is a false alarm.

**Result.** The frozen detector's false-alarm rate inflates from the calibrated
1% to a 9.6x higher average after the training period, averages **21.5%** on
exactly the drift weeks identified in Step 1, and peaks near **40%** in week 33.
Two independent measurements — a model-free distributional distance and a
model's alarm behaviour — point to the same weeks. A detector whose notion of
"normal" is frozen quietly turns benign drift into operator noise; this is the
alert-fatigue failure mode that gets real detectors switched off.

![Static detector decay](results/step2_detector_decay.png)
<img width="1650" height="750" alt="step2_detector_decay" src="https://github.com/user-attachments/assets/a947624f-806e-48ee-b1f5-56cbc384b350" />

## Step 2b — Deeper: per-institution drift, and predictive detectors

`experiments/step2b_deeper_analysis.py` answers two objections.

**Is the drift an artefact of pooling institutions?** No. Measured against each
institution's own training baseline, **29%** of institutions sit more than two
interquartile ranges from their own normal on the drift weeks (**48%** at the
peak) versus 2-8% during training. The drift is broad, not a pooling artefact.

**Do forecast-residual detectors (the class used in operational anomaly
prediction) behave differently?** A frozen seasonal-profile forecaster reaches
a **33%** false-alarm rate on the drift weeks, peaking above **50%**. A rolling
seasonal-naive forecaster that blindly adapts decays far less (**13%**). That
contrast is the whole research problem in one figure: adaptation is necessary,
and blind adaptation is exactly the surface an attacker can poison.

![Predictive detectors under drift](results/step2b_predictive_decay.png)
<img width="1650" height="750" alt="step2b_predictive_decay" src="https://github.com/user-attachments/assets/979b97cb-d85b-402f-a256-9a0c70058299" />

## Step 1b — Does the same happen in metropolitan mobile traffic?

`experiments/step1b_telecom_italia.py` repeats the Step-1 measurement on the
Telecom Italia dataset (Milan city cells, two months of the mobile operator
traffic). Benign drift in call-volume distributions grows steadily, reaching
~0.13 bits of JS divergence within eight weeks — the effect is not specific to
fixed ISP networks.

## Run it

```bash
pip install -r requirements.txt
python experiments/step1_benign_drift.py
python experiments/step2_detector_decay.py
python experiments/step2b_deeper_analysis.py
python experiments/step1b_telecom_italia.py
```

Datasets download automatically on first run via `cesnet-tszoo` (CESNET
institutions/daily: ~10 MB; Telecom Italia: ~250 MB). Set `CESNET_DATA_ROOT`
to change the download directory. Plots and CSV summaries are saved to `results/`.

## Roadmap

- Step 3: inject problem-space adversarial shift with logged ground truth and
  test whether benign and adversarial shifts are statistically separable online
  (geometry, temporal signature, cross-view cost).
- Step 4: a drift-source-aware detector that adapts to a  benign shift under a
  forgetting bounds and resisting adversarial shift under an influence cap.

## Data

- Koumar, Hynek et al., *CESNET-TimeSeries24*, Scientific Data (2025).
- Barlacchi et al., *A multi-source dataset of urban life in the city of Milan
  and the Province of Trentino*, Scientific Data (2015) — Telecom Italia.
- Loaded via the [cesnet-tszoo](https://github.com/CESNET/cesnet-tszoo) library.

## Author

Kiarash Amiri — amiri.kiarash@outlook.com — [linkedin.com/in/kiarash-amiri](https://www.linkedin.com/in/kiarash-amiri)

This project grew out of watching detection rules go stale in production.
