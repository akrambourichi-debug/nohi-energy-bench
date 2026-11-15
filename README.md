# NOHI energy bench

Energy-measurement scenario for the NOHI intrusion-detection pipeline: a gradient-boosted classifier with and without
an ontology rule layer, on CIC-IDS2017 network flows. The scenario is written for the
[Green Metrics Tool](https://docs.green-coding.io/) and the bench script also runs on any Linux machine.

## What is measured

| Phase | What it runs | Flows |
|---|---|---|
| Idle reference | container idle for 30 s | - |
| Training | XGBoost + fitting and calibration of the rule layer | stratified sample (about 300,000 flows, 60% for training) |
| Batch inference, classifier only | vectorised classifier | 3,000,000 |
| Batch inference, full NOHI | classifier + rules + fusion | 3,000,000 |
| Single-flow inference, classifier only | one flow at a time, one thread | 30,000 |
| Single-flow inference, full NOHI | same + rules + fusion + decision trace | 30,000 |

The two variants of each pair process the same flows in the same order, so the difference between them is the cost of the
rule layer. Energy per flow is the energy of a phase divided by its number of flows; the net value subtracts the idle
power multiplied by the duration of the phase.

## Files

| File | Role |
|---|---|
| `bench_energy.py` | the bench: one sub-command per measured phase |
| `nohi_v2.py` | the NOHI pipeline (models, rule layer, fusion, decision trace) |
| `Dockerfile` | bench image, library versions pinned |
| `usage_scenario.yml` | the phases measured by the Green Metrics Tool |
| `.github/workflows/smoke-test.yml` | builds the image and runs a reduced scenario |

## Run with the Green Metrics Tool

Submit the URL of this repository to a Green Metrics Tool instance; `usage_scenario.yml` is at the repository root.

## Run elsewhere

```bash
export NOHI_BENCH_DIR=$HOME/nohi_bench
python3 bench_energy.py prepare --n 300000
python3 bench_energy.py train
for phase in batch-ml batch-nohi single-ml single-nohi; do
  perf stat -a -e power/energy-pkg/,power/energy-ram/ python3 bench_energy.py $phase
done
perf stat -a -e power/energy-pkg/,power/energy-ram/ sleep 30   # idle reference
```

## Data

At build time the image downloads the public flow export of CIC-IDS2017 distributed on Hugging Face
(`rdpahalavan/CIC-IDS2017`, file `Network-Flows/CICIDS_Flow.parquet`, about 370 MB) and keeps a stratified random sample
without the identifier-like columns (flow identifier, addresses, source port, timestamp). To avoid the download, put a
`sample.parquet` produced by `python3 bench_energy.py prepare` next to the `Dockerfile`.

CIC-IDS2017 was published by I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani (ICISSP 2018).
