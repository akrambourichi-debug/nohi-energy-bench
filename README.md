# NOHI energy bench

Energy-measurement scenario for the NOHI intrusion-detection pipeline: a gradient-boosted classifier with and without
an ontology rule layer, on CIC-IDS2017 network flows. The scenario is written for the
[Green Metrics Tool](https://docs.green-coding.io/) and the bench script also runs on any Linux machine.

## What is measured

| Phase | What it runs | Flows |
|---|---|---|
| Idle reference | container idle for 30 s | - |
| Training | XGBoost + fitting and calibration of the rule layer | stratified sample (about 300,000 flows, 60% for training) |
| Start-up only (5 times) | a new process imports the libraries and loads the trained state | 0 |
| Batch, classifier only (pass 1 and pass 2) | vectorised classifier | 3,000,000 per pass |
| Batch, full NOHI (pass 1 and pass 2) | classifier + rules + fusion | 3,000,000 per pass |
| Single flow, classifier only (pass 1 and pass 2) | one flow at a time, one thread | 30,000 per pass |
| Single flow, full NOHI (pass 1 and pass 2) | same + rules + fusion + decision trace | 30,000 per pass |

The two variants of each pair process the same flows in the same order, so the difference between them is the cost of the
rule layer. Each pair is run in the order classifier, NOHI, NOHI, classifier, so that a slow drift of the machine during
the run (temperature, for instance) weighs equally on both variants.

Every inference phase is a new process, so its energy includes one start-up. For one phase:

- energy per flow = (energy of the phase - energy of one start-up) / number of flows, where the energy of one start-up
  is the energy of the start-up phase divided by 5;
- net energy per flow: the same after subtracting, from the phase and from the start-up, the idle power multiplied by
  their duration;
- cost of the rule layer = mean of the two NOHI passes - mean of the two classifier passes.

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
for phase in load batch-ml batch-nohi batch-nohi batch-ml single-ml single-nohi single-nohi single-ml; do
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
