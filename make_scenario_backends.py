#!/usr/bin/env python3
"""Writes usage_scenario_backends.yml: the phases of usage_scenario.yml, repeated for four classifiers.
The file is generated so that the phase names cannot drift from those read by the analysis script.
usage: python3 make_scenario_backends.py [output file]"""
import sys

BACKENDS = ["XGBoost", "LightGBM", "DecisionTree", "MLP"]
# flows per pass: the two cheap classifiers get more flows, so that a pass lasts tens of seconds like the others;
# the whole scenario must stay well under the 30 minutes that the free tier of the measurement service allows
FLOWS = {"XGBoost": dict(batch=2_000_000, single=20_000), "LightGBM": dict(batch=2_000_000, single=20_000),
         "DecisionTree": dict(batch=30_000_000, single=90_000), "MLP": dict(batch=16_000_000, single=90_000)}
HEAD = """---
name: NOHI energy bench, four classifiers (CIC-IDS2017 sample, with and without the ontology rule layer)
author: Akram Bourichi
description: Energy of training and of inference, vectorised and flow by flow, for each classifier alone and for the full NOHI path (classifier, rules, fusion, decision trace). Same phases as usage_scenario.yml, repeated for XGBoost, LightGBM, a decision tree and an MLP; the two cheap classifiers process more flows per pass so that a pass lasts tens of seconds.

services:
  nohibench:
    build:
      context: .
    image: nohi_energy_bench

flow:
  - name: Idle reference (30 s)
    container: nohibench
    commands:
      - type: console
        command: sleep 30
        note: container idle
"""
MODES = (("batch", "Batch - classifier only", "Batch - full NOHI", "vectorised classifier", "vectorised classifier, rules and fusion"),
         ("single", "Single flow - classifier only", "Single flow - full NOHI with trace", "one flow at a time and one thread", "one flow at a time and one thread with explanation"))


def phase(name, commands):
    s = f"  - name: {name}\n    container: nohibench\n    commands:\n"
    for command, note in commands: s += f"      - type: console\n        command: {command}\n        note: {note}\n"
    return s


def scenario():
    out = HEAD
    for b in BACKENDS:
        opt = f" --backend {b}"
        out += phase(f"{b} - Train classifier and fit rules", [(f"python3 bench_energy.py train{opt}", "training on the sample")])
        out += phase(f"{b} - Start-up only x5 (libraries and model load)", [(f"python3 bench_energy.py load{opt}", f"start-up {i} of 5") for i in range(1, 6)])
        for mode, lab_ml, lab_nohi, note_ml, note_nohi in MODES:
            n = FLOWS[b][mode]
            for variant, k in (("ml", 1), ("nohi", 1), ("nohi", 2), ("ml", 2)):   # classifier, NOHI, NOHI, classifier: a slow drift weighs equally on both
                out += phase(f"{b} - {lab_ml if variant == 'ml' else lab_nohi} - pass {k} ({n} flows)",
                             [(f"python3 bench_energy.py {mode}-{variant} --n {n}{opt}", note_ml if variant == "ml" else note_nohi)])
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "usage_scenario_backends.yml"; open(path, "w", encoding="utf-8").write(scenario()); print("written:", path)
