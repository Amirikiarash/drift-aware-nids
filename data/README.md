# Data

## CESNET-TimeSeries24 (institution level, daily aggregation)

`cesnet_institutions/agg_1_day/*.csv` — one file per institution (283 institutions),
daily time series over 40 weeks (9 Oct 2023 – 14 Jul 2024) of real traffic on
CESNET3, the Czech national research and education network. `identifiers.csv`
lists the institution ids.

Each row is one day (`id_time` = 0…279) with 19 columns: volume counters
(`n_flows`, `n_packets`, `n_bytes`), destination-diversity statistics
(`*_n_dest_asn`, `*_n_dest_ports`, `*_n_dest_ip`), protocol/direction ratios
(`tcp_udp_ratio_*`, `dir_ratio_*`), and `avg_duration`, `avg_ttl`.

- Dataset: Koumar, Hynek, Čejka & Šiška, *CESNET-TimeSeries24*, Scientific Data 12 (2025).
  https://doi.org/10.1038/s41597-025-04603-x — arXiv:2409.18874
- Full data (41.5 GB, all aggregations/entity types) on Zenodo: https://zenodo.org/records/13382427
- Library: https://github.com/CESNET/cesnet-tszoo
- License: CC-BY 4.0
- Downloaded: only `institutions.tar.gz` (479 MB); only the 1-day aggregation is kept
  here (9.2 MB), which is all the experiments in this repo use.

The full dataset can be re-fetched with `cesnet-tszoo`; this daily institution
subset is committed directly so the experiments are runnable without any download.
