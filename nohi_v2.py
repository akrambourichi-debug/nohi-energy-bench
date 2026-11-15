#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOHI v2 - reproducible pipeline for the revision of ARRAY-D-26-02766.

Everything reported in the revised manuscript (tables, figures, numbers quoted in the
response letter) is produced by this ONE script, so that text and code cannot diverge.

Usage (Colab / HPC):
    python nohi_v2.py --data /path/CICIDS_Flow.parquet --out results --seeds 5 --shap
    python nohi_v2.py --data /path/to/MachineLearningCVE/ --out results      # folder of CSVs
    python nohi_v2.py --synthetic --out smoke                                  # code check ONLY

Protocol (stated once, enforced in code):
    train (60%)  -> model fitting, scaler, zero-variance filter, ontology thresholds (benign quantiles)
    val   (20%)  -> early stopping, rule-confidence calibration, tau selection, backend selection
    test  (20%)  -> final evaluation ONLY (never used for any choice)
"""
import argparse, copy, glob, hashlib, json, operator, os, pickle, platform, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                     "savefig.dpi": 300, "figure.dpi": 120})

# ----------------------------------------------------------------------------------------
# 0. Ontology definition (categories, metadata, label mapping)
# ----------------------------------------------------------------------------------------
CATS = ["DDoS", "PortScan", "BruteForce", "Malware"]          # ontology attack categories
NORMAL_CODE, UNCOV_CODE = 4, 5                                  # codes for Normal / attack not in ontology
CAT_META = {
    "DDoS":       dict(severity="Critical", impact="Availability",
                       mitigation="rate-limit or null-route sources; engage upstream scrubbing"),
    "PortScan":   dict(severity="Medium", impact="Reconnaissance",
                       mitigation="block scanning source; review exposed services"),
    "BruteForce": dict(severity="High", impact="Confidentiality",
                       mitigation="lock/throttle account; enforce MFA and fail2ban-style bans"),
    "Malware":    dict(severity="Critical", impact="Integrity/Confidentiality",
                       mitigation="isolate host; open incident response"),
}
# CIC-IDS2017 label -> ontology category.  None = attack NOT represented in the ontology.
FINE2CAT = {
    "normal": "Normal", "benign": "Normal",
    "ddos": "DDoS", "dos hulk": "DDoS", "dos goldeneye": "DDoS",
    "dos slowloris": "DDoS", "dos slowhttptest": "DDoS",
    "portscan": "PortScan",
    "ftp patator": "BruteForce", "ssh patator": "BruteForce", "web attack brute force": "BruteForce",
    "bot": "Malware", "infiltration": "Malware", "heartbleed": "Malware",
    "web attack xss": None, "web attack sql injection": None,
}
DUP_ALIAS = {}      # dropped duplicate column -> identical kept column (filled by clean(); lets rules keep their feature names)
STRAT = ["Consensus", "ML (ontology silent)", "OntologyOverride", "MLOverride", "ConservativeDefault"]

# Feature-name aliases (CICFlowMeter v2/v3 spellings after normalisation)
ALIAS = {
    "flow_packets_s": ["flow_packets_s", "flow_pkts_s"],
    "fwd_packets_s": ["fwd_packets_s", "fwd_pkts_s"],
    "flow_bytes_s": ["flow_bytes_s", "flow_byts_s"],
    "flow_duration": ["flow_duration"],
    "total_fwd_packets": ["total_fwd_packets", "tot_fwd_pkts"],
    "total_backward_packets": ["total_backward_packets", "tot_bwd_pkts"],
    "total_length_bwd": ["total_length_of_bwd_packets", "totlen_bwd_pkts"],
    "destination_port": ["destination_port", "dst_port"],
    "syn_flag_count": ["syn_flag_count", "syn_flag_cnt"],
    "rst_flag_count": ["rst_flag_count", "rst_flag_cnt"],
    "ack_flag_count": ["ack_flag_count", "ack_flag_cnt"],
    "psh_flag_count": ["psh_flag_count", "psh_flag_cnt"],
}
# 12 rules.  Thresholds given as ("q", p) are the p-quantile of the BENIGN TRAINING flows (no attack
# labels used).  Rule confidence = Laplace-smoothed precision of the rule on the VALIDATION set.
# NOTE: this rule base is a RECONSTRUCTION from the manuscript's description (4 categories, 12 rules);
# replace/edit it to match the rules actually used in the ICASET paper. Rules whose features are
# missing in the loaded dataset are disabled and reported in tables/T_rules.csv.
RULES = [
    dict(id="R01", cat="DDoS", desc="one-way packet flood",
         conds=[("flow_packets_s", ">", ("q", 0.99)), ("total_backward_packets", "<=", 1)]),
    dict(id="R02", cat="DDoS", desc="very high forward packet rate and SYN present",
         conds=[("fwd_packets_s", ">", ("q", 0.99)), ("syn_flag_count", ">=", 1)]),
    dict(id="R03", cat="DDoS", desc="low-and-slow: very long flow, few packets",
         conds=[("flow_duration", ">", ("q", 0.95)), ("total_fwd_packets", "<=", ("q", 0.50))]),
    dict(id="R04", cat="PortScan", desc="tiny flow, no meaningful reply, very short",
         conds=[("total_fwd_packets", "<=", 2), ("total_backward_packets", "<=", 1),
                ("flow_duration", "<", ("q", 0.10))]),
    dict(id="R05", cat="PortScan", desc="reset answer to a minimal probe",
         conds=[("rst_flag_count", ">=", 1), ("total_fwd_packets", "<=", 2)]),
    dict(id="R06", cat="PortScan", desc="bare SYN probe (no ACK)",
         conds=[("syn_flag_count", ">=", 1), ("ack_flag_count", "==", 0), ("total_fwd_packets", "<=", 2)]),
    dict(id="R07", cat="BruteForce", desc="FTP control port with sustained exchange",
         conds=[("destination_port", "==", 21), ("total_fwd_packets", ">=", 5)]),
    dict(id="R08", cat="BruteForce", desc="SSH port with sustained exchange",
         conds=[("destination_port", "==", 22), ("total_fwd_packets", ">=", 5)]),
    dict(id="R09", cat="BruteForce", desc="web login port, many forward packets, PSH",
         conds=[("destination_port", "in", [80, 443, 8080]), ("total_fwd_packets", ">", ("q", 0.95)),
                ("psh_flag_count", ">=", 1)]),
    dict(id="R10", cat="Malware", desc="beacon-like: long flow, very low byte rate",
         conds=[("flow_duration", ">", ("q", 0.95)), ("flow_bytes_s", "<", ("q", 0.10))]),
    dict(id="R11", cat="Malware", desc="TLS-heartbeat port 444 / oversized backward payload",
         conds=[("destination_port", "==", 444)]),
    dict(id="R12", cat="Malware", desc="asymmetric exfiltration-like transfer",
         conds=[("total_length_bwd", ">", ("q", 0.99)), ("total_fwd_packets", "<=", ("q", 0.50))]),
]
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq,
       "in": lambda x, t: x in t}


def fmt_thr(t):
    return f"{t:.4g}" if isinstance(t, (int, float, np.floating)) else str(t)


def vec_cond(x, op, t):
    return np.isin(x, t) if op == "in" else OPS[op](x, t)


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def norm_label(s):
    s = re.sub(r"[^A-Za-z0-9]+", " ", str(s)).strip()
    return "Normal" if s.lower() == "benign" else s


def rules_to_json(rules):
    return [dict(id=r["id"], cat=r["cat"], desc=r["desc"],
                 conds=[[f, op, ({"q": v[1]} if isinstance(v, tuple) else v)] for f, op, v in r["conds"]]) for r in rules]


def load_rules(path):
    """JSON list of {id, cat, desc, conds:[[feature, op, value|{"q":p}], ...]}; cat in CATS."""
    out = []
    for r in json.load(open(path, encoding="utf-8")):
        assert r["cat"] in CATS, f"rule {r['id']}: category must be one of {CATS}"
        out.append(dict(id=r["id"], cat=r["cat"], desc=r.get("desc", ""),
                        conds=[(f, op, ("q", v["q"]) if isinstance(v, dict) else v) for f, op, v in r["conds"]]))
    return out


def load_labelmap(path):
    """JSON {label (any case): category|null|"Normal"} that overrides/extends FINE2CAT."""
    for k, v in json.load(open(path, encoding="utf-8")).items():
        FINE2CAT[re.sub(r"[^a-z0-9]+", " ", k.lower()).strip()] = v


def read_table(path):
    p = Path(path)
    if p.is_dir():
        return pd.concat([pd.read_csv(f, encoding="latin1", low_memory=False) for f in sorted(glob.glob(str(p / "*.csv")))], ignore_index=True)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, encoding="latin1", low_memory=False)


_IRREG = {
    "dst_port": "destination_port", "tot_fwd_pkts": "total_fwd_packets", "tot_bwd_pkts": "total_backward_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets", "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "pkt_len_min": "min_packet_length", "pkt_len_max": "max_packet_length", "pkt_len_mean": "packet_length_mean",
    "pkt_len_std": "packet_length_std", "pkt_len_var": "packet_length_variance", "pkt_size_avg": "average_packet_size",
    "fwd_seg_size_avg": "avg_fwd_segment_size", "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_seg_size_min": "min_seg_size_forward", "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward", "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_byts_b_avg": "fwd_avg_bytes_bulk", "fwd_pkts_b_avg": "fwd_avg_packets_bulk", "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg": "bwd_avg_bytes_bulk", "bwd_pkts_b_avg": "bwd_avg_packets_bulk", "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    "fwd_iat_tot": "fwd_iat_total", "bwd_iat_tot": "bwd_iat_total",
}
_TOK = {"pkts": "packets", "pkt": "packet", "byts": "bytes", "len": "length", "cnt": "count", "seg": "segment",
        "avg": "average", "dst": "destination", "src": "source", "tot": "total"}


def canon(name):
    """Map CSE-CIC-IDS2018-style (abbreviated) CICFlowMeter names onto the CIC-IDS2017 vocabulary."""
    n = norm_col(name)
    if n in _IRREG:
        return _IRREG[n]
    return "_".join(_TOK.get(t, t) for t in n.split("_"))


def ext_label_code(lab):
    """External-dataset label -> ontology family code (keyword based; documented in the paper)."""
    k = re.sub(r"[^a-z0-9]+", " ", str(lab).lower()).strip()
    if k in ("benign", "normal"):
        return NORMAL_CODE
    if "xss" in k or "sql" in k:
        return UNCOV_CODE
    if "scan" in k:
        return CATS.index("PortScan")
    if "brute" in k or "patator" in k:
        return CATS.index("BruteForce")
    if "ddos" in k or "dos" in k:
        return CATS.index("DDoS")
    if "bot" in k or "infil" in k or "heartbleed" in k:
        return CATS.index("Malware")
    return UNCOV_CODE


def label_to_code(lab):
    key = lab.lower()
    if key not in FINE2CAT:
        print(f"  [warn] label '{lab}' not in FINE2CAT -> treated as attack not covered by ontology")
        return UNCOV_CODE
    c = FINE2CAT[key]
    return NORMAL_CODE if c == "Normal" else (UNCOV_CODE if c is None else CATS.index(c))


class Ontology:
    """Rule-based reasoner. Silent (abstains) when no rule fires."""

    def __init__(self, rules=RULES):
        self.rules = [dict(r) for r in rules]

    def fit(self, X, colnames, benign_mask):
        norm = {n: j for j, n in enumerate(colnames)}
        for d_, k_ in DUP_ALIAS.items():
            if k_ in norm and d_ not in norm:
                norm[d_] = norm[k_]
        self.colnames = list(colnames)
        self.active = []
        for r in self.rules:
            res, ok = [], True
            for f, op, v in r["conds"]:
                j = next((norm[a] for a in ALIAS.get(f, [f]) if a in norm), None)
                if j is None:
                    ok = False
                    break
                thr = float(np.quantile(X[benign_mask, j], v[1])) if isinstance(v, tuple) else v
                res.append((f, j, op, thr))
            r["resolved"], r["active"] = res, ok
            if ok:
                self.active.append(r)
        self.rule_cat = np.array([CATS.index(r["cat"]) for r in self.active], dtype=int)
        self.conf = np.full(len(self.active), 0.5)
        return self

    def fired(self, X):
        F = np.zeros((len(X), len(self.active)), dtype=bool)
        for k, r in enumerate(self.active):
            m = np.ones(len(X), dtype=bool)
            for _, j, op, t in r["resolved"]:
                m &= vec_cond(X[:, j], op, t)
            F[:, k] = m
        return F

    def calibrate(self, X_val, true_code_val):
        F = self.fired(X_val)
        rows = []
        for k, r in enumerate(self.active):
            fired = int(F[:, k].sum())
            tp = int((F[:, k] & (true_code_val == self.rule_cat[k])).sum())
            pos = int((true_code_val == self.rule_cat[k]).sum())
            self.conf[k] = (tp + 1) / (fired + 2)
            rows.append(dict(rule=r["id"], category=r["cat"], description=r["desc"],
                             conditions=" AND ".join(f"{f} {op} {fmt_thr(t)}" for f, _, op, t in r["resolved"]),
                             fired_val=fired, tp_val=tp, precision_val=tp / fired if fired else np.nan,
                             recall_of_category_val=tp / pos if pos else np.nan,
                             confidence=self.conf[k]))
        for r in self.rules:
            if not r["active"]:
                rows.append(dict(rule=r["id"], category=r["cat"], description=r["desc"],
                                 conditions="DISABLED (feature missing)", confidence=np.nan))
        return pd.DataFrame(rows)

    def infer(self, X):
        """-> code (-1 = silent), conf, best rule index, fired matrix"""
        F = self.fired(X)
        n = len(X)
        if F.shape[1] == 0:
            return np.full(n, -1), np.full(n, 0.5), np.full(n, -1), F
        score = F * self.conf[None, :]
        best = np.argmax(score, axis=1)
        any_ = F.any(axis=1)
        code = np.where(any_, self.rule_cat[best], -1)
        conf = np.where(any_, score[np.arange(n), best], 0.5)
        return code, conf, np.where(any_, best, -1), F

    def prepare_scalar(self):
        self._sc = [(k, r["id"], [(f, j, OPS[op], op, t) for f, j, op, t in r["resolved"]])
                    for k, r in enumerate(self.active)]

    def infer_one(self, x):
        best_c, best_k = 0.0, -1
        for k, rid, conds in self._sc:
            ok = True
            for _, j, fn, _, t in conds:
                if not fn(x[j], t):
                    ok = False
                    break
            if ok and self.conf[k] > best_c:
                best_c, best_k = self.conf[k], k
        if best_k < 0:
            return -1, 0.5, -1
        return int(self.rule_cat[best_k]), float(best_c), best_k


# ----------------------------------------------------------------------------------------
# 1. Fusion (Algorithm 1, explicit 15-class interpretation)
# ----------------------------------------------------------------------------------------
def fuse(proba, ont_code, ont_conf, tau, delta, cls_code, variant="literal"):
    """Vectorised Algorithm 1.
    * Labels are compared at ONTOLOGY-CATEGORY level (fine class -> category via cls_code).
    * Ontology silent (no rule fired) -> abstains; ML label is kept.
    * Consensus -> ML fine label.
    * Ontology override (C_ont > tau) -> most probable ML class INSIDE the ontology category.
    * ML override (C_ml > tau and ML says attack) -> ML fine label.
        variant='ml_trust' relaxes this to any ML label (including Normal).
    * Conservative default -> most probable ML class among ALL attack classes.
    * delta only scales the reported fused confidence (it never changes a label).
    """
    n, K = proba.shape
    ml = np.argmax(proba, 1)
    mlc = proba[np.arange(n), ml]
    mcode = cls_code[ml]
    silent = ont_code < 0
    cons = (~silent) & (ont_code == mcode)
    confl = (~silent) & ~cons
    ovr = confl & (ont_conf > tau)
    ml_ok = (mcode != NORMAL_CODE) if variant == "literal" else np.ones(n, bool)
    mlo = confl & ~ovr & (mlc > tau) & ml_ok
    dflt = confl & ~ovr & ~mlo
    final = ml.copy()
    for c in range(len(CATS)):
        rows = np.where(ovr & (ont_code == c))[0]
        idx = np.where(cls_code == c)[0]
        if len(rows) and len(idx):
            final[rows] = idx[np.argmax(proba[np.ix_(rows, idx)], 1)]
    att = np.where(cls_code != NORMAL_CODE)[0]
    rows = np.where(dflt)[0]
    if len(rows) and len(att):
        final[rows] = att[np.argmax(proba[np.ix_(rows, att)], 1)]
    strat = np.zeros(n, int)
    strat[silent], strat[cons], strat[ovr], strat[mlo], strat[dflt] = 1, 0, 2, 3, 4
    cf = mlc.copy()
    half = 0.5 * (mlc + ont_conf)
    cf[cons] = half[cons]
    cf[confl] = (half * delta)[confl]
    return final, strat, cf


def fuse_one(p, ont_code, ont_conf, tau, delta, cls_code, variant="literal"):
    ml = int(np.argmax(p))
    mlc = float(p[ml])
    mcode = int(cls_code[ml])
    if ont_code < 0:
        return ml, 1, mlc
    if ont_code == mcode:
        return ml, 0, 0.5 * (mlc + ont_conf)
    cf = 0.5 * (mlc + ont_conf) * delta
    if ont_conf > tau:
        idx = np.where(cls_code == ont_code)[0]
        return (int(idx[np.argmax(p[idx])]) if len(idx) else ml), 2, cf
    if mlc > tau and (variant != "literal" or mcode != NORMAL_CODE):
        return ml, 3, cf
    att = np.where(cls_code != NORMAL_CODE)[0]
    return int(att[np.argmax(p[att])]), 4, cf


def explain_one(strategy, final, cf, p, ont_code, ont_conf, rule_k, x, classes, ont, tau, delta):
    top = np.argsort(-p)[:3]
    s = [f"[{STRAT[strategy]}] final={classes[final]} (fused conf {cf:.2f}).",
         "ML evidence: " + ", ".join(f"{classes[i]}={p[i]:.2f}" for i in top) + "."]
    if ont_code >= 0:
        r = ont.active[rule_k]
        ev = "; ".join(f"{f}={x[j]:.4g} {op} {fmt_thr(t)}" for f, j, op, t in r["resolved"])
        m = CAT_META[CATS[ont_code]]
        s.append(f"Ontology evidence: rule {r['id']} ({r['desc']}) -> {CATS[ont_code]} "
                 f"(C_ont={ont_conf:.2f}); {ev}. Severity={m['severity']}, impact={m['impact']}; "
                 f"suggested action: {m['mitigation']}.")
    else:
        s.append("Ontology evidence: no rule fired (ontology abstains).")
    why = {0: "ML and ontology agree on the category.",
           1: "ontology silent; ML decision kept.",
           2: f"ontology confidence {ont_conf:.2f} > tau={tau:.2f}: ontology category imposed, ML picks the class within it.",
           3: f"ontology not confident enough (<= tau={tau:.2f}) but ML is confident attack: ML kept.",
           4: f"neither side confident (tau={tau:.2f}): conservative default = most probable attack class."}[strategy]
    s.append("Rationale: " + why)
    return " ".join(s)


# ----------------------------------------------------------------------------------------
# 2. Data: loading, cleaning (with a full report), splitting
# ----------------------------------------------------------------------------------------
DROP_RE = re.compile(r"label|attack|flow_id|timestamp|(^|_)(src|dst|source|destination)_?ip($|_)|"
                     r"(^|_)(src|source)_?port($|_)|unnamed|^index$|^id$")


def make_synthetic(n=40000, seed=0):
    """Stand-in data with CIC-like column names. ONLY for checking that the code runs."""
    rng = np.random.default_rng(seed)
    spec = {"Normal": .80, "DoS Hulk": .06, "PortScan": .05, "DDoS": .04, "DoS GoldenEye": .01,
            "FTP-Patator": .008, "SSH-Patator": .006, "DoS slowloris": .006, "DoS Slowhttptest": .005,
            "Bot": .004, "Web Attack - Brute Force": .003, "Web Attack - XSS": .0015,
            "Infiltration": .0008, "Web Attack - Sql Injection": .0006, "Heartbleed": .0004}
    labs = rng.choice(list(spec), size=n, p=np.array(list(spec.values())) / sum(spec.values()))
    rows = []
    for lab in labs:
        d = dict(dur=rng.lognormal(10, 2), fp=rng.poisson(8) + 1, bp=rng.poisson(8), port=int(rng.choice([80, 443, 53, 8080, 3389])),
                 fl=rng.lognormal(6, 1.5), bl=rng.lognormal(7, 1.5), syn=int(rng.random() < .1), rst=int(rng.random() < .05),
                 ack=int(rng.random() < .6), psh=int(rng.random() < .4))
        if lab in ("DDoS", "DoS Hulk", "DoS GoldenEye"):
            d.update(dur=rng.lognormal(8, 1), fp=rng.poisson(20) + 5, bp=int(rng.random() < .3), syn=1)
        elif lab in ("DoS slowloris", "DoS Slowhttptest"):
            d.update(dur=rng.lognormal(16, .5), fp=rng.poisson(3) + 1)
        elif lab == "PortScan":
            d.update(dur=rng.lognormal(2, 1), fp=1 + int(rng.random() < .3), bp=int(rng.random() < .3), port=int(rng.integers(1, 65535)),
                     syn=1, ack=0, rst=int(rng.random() < .5))
        elif lab == "FTP-Patator":
            d.update(port=21, fp=rng.poisson(10) + 5)
        elif lab == "SSH-Patator":
            d.update(port=22, fp=rng.poisson(12) + 5)
        elif lab == "Bot":
            d.update(dur=rng.lognormal(16, .5), fl=rng.lognormal(1, 1))
        elif lab == "Heartbleed":
            d.update(port=444, bl=rng.lognormal(17, .5))
        elif lab.startswith("Web Attack"):
            d.update(port=80, psh=1, fp=rng.poisson(18) + 4, dur=rng.lognormal(11, 1))
        rows.append(d)
    r = pd.DataFrame(rows)
    df = pd.DataFrame({
        " Destination Port": r.port, " Flow Duration": r.dur, " Total Fwd Packets": r.fp,
        " Total Backward Packets": r.bp, "Total Length of Bwd Packets": r.bl,
        "Flow Bytes/s": r.fl, " Flow Packets/s": (r.fp + r.bp) / (r.dur / 1e6 + 1e-6),
        " Fwd Packets/s": r.fp / (r.dur / 1e6 + 1e-6), " SYN Flag Count": r.syn, " RST Flag Count": r.rst,
        " ACK Flag Count": r.ack, " PSH Flag Count": r.psh})
    for i in range(12):
        df[f"noise_{i}"] = rng.normal(size=n) + (labs == "Normal") * 0.0
    df["const_col"] = 1.0
    df[" Label"] = np.where(labs == "Normal", "BENIGN", labs)
    # time bursts (for the grouped-split check) + dirty rows (to exercise the cleaning code)
    base = pd.Timestamp("2017-07-03")
    df["Timestamp"] = [base + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 5))) for _ in range(n)]
    df = pd.concat([df, df.sample(200, random_state=1)], ignore_index=True)
    df.loc[df.sample(50, random_state=2).index, "Flow Bytes/s"] = np.inf
    df.loc[df.sample(30, random_state=3).index, " Flow Packets/s"] = np.nan
    return df


def load_data(args):
    if args.synthetic:
        df = make_synthetic(seed=args.seed0)
    else:
        df = read_table(args.data)
    df.columns = [norm_col(c) for c in df.columns]
    lab_col = next((c for c in ("label", "attack_label", "attack", "class") if c in df.columns), None)
    if lab_col is None:
        sys.exit("No label column found")
    return df, lab_col


def clean(df, lab_col, out):
    """Cleaning on float32 feature vectors (the representation the models see):
       1) rows with non-finite values removed; 2) feature vectors carrying CONFLICTING labels removed
       entirely; 3) exact duplicates (same features, same label) reduced to one row.
       Row identity uses a 64-bit hash of the feature vector (collisions are possible but negligible)."""
    n_raw = len(df)
    ts = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else None
    labels = df[lab_col].map(norm_label)
    feat = [c for c in df.columns if not DROP_RE.search(c) and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat].to_numpy(dtype=np.float32)
    X = X + np.float32(0)                                         # writable copy; -0.0 -> 0.0 so hashing is value-based
    dupcols, seen = [], {}
    for j, f in enumerate(feat):                                  # exact duplicate (non-constant) columns
        col = X[:, j]
        if (col == col[0]).all():
            continue
        key = (float(np.nansum(col[:200000], dtype=np.float64)), float(col[:2000].std()))
        for i in seen.get(key, []):
            if np.array_equal(X[:, i], col, equal_nan=True):
                dupcols.append((f, feat[i]))
                break
        else:
            seen.setdefault(key, []).append(j)
    if dupcols:
        drop = {f for f, _ in dupcols}
        keepc = [j for j, f in enumerate(feat) if f not in drop]
        X, feat = np.ascontiguousarray(X[:, keepc]), [feat[j] for j in keepc]
        DUP_ALIAS.update({f: k for f, k in dupcols})
        print(f"  duplicate feature columns removed: {dupcols}")
    finite = np.isfinite(X).all(axis=1)
    h = pd.util.hash_pandas_object(pd.DataFrame(X, copy=False), index=False).to_numpy()
    codes, _ = pd.factorize(labels)
    fi = np.where(finite)[0]
    dfh = pd.DataFrame({"h": h[fi], "y": codes[fi]})
    u = dfh.drop_duplicates()
    conf_h = u["h"][u["h"].duplicated(keep=False)].unique()
    conflict = dfh["h"].isin(conf_h).to_numpy()
    first = ~dfh.duplicated(keep="first").to_numpy()
    keep_idx = fi[~conflict & first]
    lab_np = labels.to_numpy()
    Xc, yc = X[keep_idx], lab_np[keep_idx]
    tsc = ts.iloc[keep_idx].to_numpy() if ts is not None else None
    rep = pd.DataFrame({"raw": labels.value_counts(), "after_nonfinite_removal": pd.Series(lab_np[fi]).value_counts(),
                        "after_label_conflict_removal": pd.Series(lab_np[fi[~conflict]]).value_counts(),
                        "after_dedup": pd.Series(yc).value_counts()}).fillna(0).astype(int).sort_values("raw", ascending=False)
    rep.loc["TOTAL"] = [n_raw, len(fi), int((~conflict).sum()), len(yc)]
    save_table(rep.reset_index().rename(columns={"index": "class"}), out, "T_cleaning")
    print(f"  raw={n_raw:,}  non-finite removed={n_raw - len(fi):,}  label-conflict rows removed={int(conflict.sum()):,}  "
          f"duplicates removed={int((~conflict & ~first).sum()):,}  final={len(yc):,}")
    (out / "features_used.txt").write_text("\n".join(feat))
    (out / "duplicate_feature_columns.json").write_text(json.dumps(dupcols))
    return Xc, yc, feat, tsc


def _rowsplit(idx, rng, fr):
    n = len(idx)
    idx = idx.copy()
    rng.shuffle(idx)
    nte = max(1, int(round(n * fr[2])))
    nva = max(1, int(round(n * fr[1])))
    if nte + nva > n - 1:
        nva = max(1, n - 1 - nte)
    return idx[nte + nva:], idx[nte:nte + nva], idx[:nte]


def split(y, seed, fr=(0.6, 0.2, 0.2), groups=None):
    """Per-class split. Classes with <3 flows go entirely to train and are reported as unevaluated.
    groups (optional): all flows of one (class, 5-min burst) stay on the same side."""
    rng = np.random.default_rng(seed)
    tr, va, te, excluded, fallback = [], [], [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = len(idx)
        if n < 3:
            tr.extend(idx)
            excluded.append(int(c))
            continue
        if groups is not None:
            ug, inv = np.unique(groups[idx], return_inverse=True)
            if len(ug) >= 3:
                cnt = np.bincount(inv)
                tgt_te, tgt_va = max(1, round(n * fr[2])), max(1, round(n * fr[1]))
                assign = np.zeros(len(ug), int)
                c_te = c_va = 0
                for j in rng.permutation(len(ug)):
                    if c_te < tgt_te:
                        assign[j] = 2
                        c_te += cnt[j]
                    elif c_va < tgt_va:
                        assign[j] = 1
                        c_va += cnt[j]
                if (assign == 0).any() and (assign == 1).any() and (assign == 2).any():
                    a = assign[inv]
                    tr.extend(idx[a == 0]); va.extend(idx[a == 1]); te.extend(idx[a == 2])
                    continue
            fallback.append(int(c))
        a, b, d = _rowsplit(idx, rng, fr)
        tr.extend(a); va.extend(b); te.extend(d)
    return np.array(tr), np.array(va), np.array(te), excluded, fallback


# ----------------------------------------------------------------------------------------
# 3. Models, metrics, helpers
# ----------------------------------------------------------------------------------------
def sqrt_balanced_weights(y, K):
    cnt = np.bincount(y, minlength=K).astype(float)
    w_c = (len(y) / (K * np.maximum(cnt, 1))) ** 0.5
    return w_c[y] / w_c[y].mean(), {i: float(w_c[i] / w_c[y].mean()) for i in range(K)}


def make_models(K, seed, a):
    return {
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", batch_size=256,
                             learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=10, max_iter=a.mlp_max_iter, random_state=seed),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, tree_method="hist", objective="multi:softprob",
                                     early_stopping_rounds=20, eval_metric="mlogloss", n_jobs=-1, random_state=seed),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, min_child_samples=5,
                                       n_jobs=-1, random_state=seed, verbose=-1),
        "LightGBM (unweighted)": lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8,
                                                    subsample_freq=1, colsample_bytree=0.8, min_child_samples=5,
                                                    n_jobs=-1, random_state=seed, verbose=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, criterion="gini", min_impurity_decrease=1e-6,
                                               random_state=seed),
    }


def fit_model(name, m, Xtr, ytr, Xva, yva, w, cw, rapl):
    t0 = time.perf_counter()
    e0 = rapl.read()
    if name == "MLP":
        m.fit(Xtr, ytr)
    elif name == "XGBoost":
        m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
    elif name == "LightGBM":
        m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(20, verbose=False)])
    elif name == "LightGBM (unweighted)":
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(20, verbose=False)])
    else:
        m.set_params(class_weight=cw)
        m.fit(Xtr, ytr)
    t = time.perf_counter() - t0
    e = rapl.delta(e0, rapl.read())
    return t, (e / 1e6 if e is not None else None)


def safe_proba(m, X, K):
    p = m.predict_proba(X).astype(np.float32)
    if p.shape[1] != K:                       # a class missing in the model's training labels
        full = np.zeros((len(X), K), np.float32)
        full[:, m.classes_] = p
        p = full
    return p


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def metrics_all(y, p, proba, K, benign, auc=True):
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=np.arange(K), zero_division=0)
    pres = sup > 0
    ya, pa = y != benign, p != benign
    out = dict(accuracy=float((y == p).mean()), macro_f1=float(f1[pres].mean()),
               macro_precision=float(prec[pres].mean()), macro_recall=float(rec[pres].mean()),
               mcc=float(matthews_corrcoef(y, p)),
               attack_detection_recall=float((pa & ya).sum() / max(ya.sum(), 1)),
               attack_detection_precision=float((pa & ya).sum() / max(pa.sum(), 1)),
               fpr=float((pa & ~ya).sum() / max((~ya).sum(), 1)),
               false_alerts_per_million_benign=float((pa & ~ya).sum() / max((~ya).sum(), 1) * 1e6))
    a = np.nan
    if auc and proba is not None:
        v = [roc_auc_score(y == k, proba[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)]
        a = float(np.mean(v)) if v else np.nan
    out["auc_ovr_macro"] = a
    return out, (prec, rec, f1, sup)


def mcnemar_exact(y, p1, p2):
    b = int(((p1 == y) & (p2 != y)).sum())
    c = int(((p1 != y) & (p2 == y)).sum())
    pv = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, pv


def ci95(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def save_table(df, out, name):
    (out / "tables").mkdir(exist_ok=True, parents=True)
    df.to_csv(out / "tables" / f"{name}.csv", index=False)
    try:
        df.to_latex(out / "tables" / f"{name}.tex", index=False, float_format="%.4f", escape=True)
    except Exception as e:
        print("  [warn] latex export failed for", name, e)


class Rapl:
    """Intel RAPL package counters (needs readable /sys/class/powercap). Returns None when unavailable."""

    def __init__(self):
        self.pk = []
        for p in glob.glob("/sys/class/powercap/intel-rapl:*"):
            if re.fullmatch(r".*intel-rapl:\d+", p):
                try:
                    int(open(p + "/energy_uj").read())
                    self.pk.append((p, int(open(p + "/max_energy_range_uj").read())))
                except Exception:
                    pass
        self.ok = bool(self.pk)
        self.domain = "package (" + ", ".join(Path(p).name for p, _ in self.pk) + ")" if self.ok else "unavailable"

    def read(self):
        if not self.ok:
            return None
        try:
            return [int(open(p + "/energy_uj").read()) for p, _ in self.pk]
        except Exception:
            return None

    def delta(self, a, b):
        if a is None or b is None:
            return None
        return float(sum((y - x) % m for x, y, (_, m) in zip(a, b, self.pk)))     # micro-joules

    def measure(self, fn, n, reps):
        """energy per call (J): idle-corrected and raw, mean/sd over reps"""
        if not self.ok:
            return None
        cor, raw = [], []
        for _ in range(reps):
            e0, t0 = self.read(), time.perf_counter()
            for _ in range(n):
                fn()
            t1, e1 = time.perf_counter(), self.read()
            E, t = self.delta(e0, e1) / 1e6, t1 - t0
            i0 = self.read()
            time.sleep(t)
            i1 = self.read()
            Pidle = self.delta(i0, i1) / 1e6 / t
            cor.append((E - Pidle * t) / n)
            raw.append(E / n)
        return dict(J_per_call_idle_corrected_mean=float(np.mean(cor)), J_per_call_idle_corrected_sd=float(np.std(cor, ddof=1)) if reps > 1 else np.nan,
                    J_per_call_raw_mean=float(np.mean(raw)), reps=reps, calls_per_rep=n)


def cpu_model():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                return l.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ----------------------------------------------------------------------------------------
# 4. One full experiment (one seed): split -> fit -> val sweeps -> test
# ----------------------------------------------------------------------------------------
TAU_GRID = np.round(np.arange(0.10, 0.951, 0.05), 2)
WEIGHTED = ["MLP", "XGBoost", "LightGBM", "DecisionTree"]       # main comparison (LightGBM is the weighted one)


# Flow observables with a direct meaning in a network-connection ontology (UCO observable:NetworkConnection facets /
# ATT&CK "Network Traffic" data source). Learned rules may ONLY use these, so that every rule stays readable.
OBSERVABLES = ["destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets",
               "total_length_of_fwd_packets", "total_length_of_bwd_packets", "flow_bytes_s", "flow_packets_s",
               "fwd_packets_s", "bwd_packets_s", "fwd_psh_flags", "fin_flag_count", "rst_flag_count", "psh_flag_count",
               "ack_flag_count", "urg_flag_count", "init_win_bytes_forward", "init_win_bytes_backward",
               "flow_iat_mean", "flow_iat_max", "fwd_packet_length_max", "bwd_packet_length_max", "packet_length_mean",
               "min_seg_size_forward", "down_up_ratio", "idle_mean", "active_mean"]
CAT_ATTACK = {"DDoS": "T1498/T1499 (Network / Endpoint Denial of Service)", "PortScan": "T1046 (Network Service Discovery)",
              "BruteForce": "T1110 (Brute Force)", "Malware": "T1071 / T1584 (C2 over application protocol, botnet)"}


def learn_rules(Xtr, ytr, C, seed, per_cat=3, depth=4, min_precision=0.9, min_support=20):
    """Ontology-anchored, data-calibrated rules: for each ontology category, a shallow one-vs-rest decision tree is
    fitted on TRAINING data only, restricted to OBSERVABLES; its purest positive leaves become conjunctive rules.
    Nothing from validation or test is used here (validation is used afterwards for rule confidence only)."""
    rng = np.random.default_rng(seed + 7)
    feat = C["feat"]
    cols = [feat.index(f) for f in OBSERVABLES if f in feat]
    code = C["cls_code"][ytr]
    rules = []
    for ci, cat in enumerate(CATS):
        pos = np.where(code == ci)[0]
        neg = np.where(code != ci)[0]
        if len(pos) < min_support:
            continue
        sp = pos if len(pos) <= 150000 else rng.choice(pos, 150000, replace=False)
        sn = neg if len(neg) <= 450000 else rng.choice(neg, 450000, replace=False)
        idx = np.concatenate([sp, sn])
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_support, class_weight="balanced", random_state=seed)
        t.fit(Xtr[idx][:, cols], (code[idx] == ci).astype(int))
        tr_ = t.tree_
        cands = []

        def walk(node, conds):
            if tr_.children_left[node] == -1:
                if tr_.value[node][0].argmax() == 1:
                    cands.append(list(conds))
                return
            f, thr = feat[cols[tr_.feature[node]]], float(tr_.threshold[node])
            walk(tr_.children_left[node], conds + [(f, "<=", thr)])
            walk(tr_.children_right[node], conds + [(f, ">", thr)])
        walk(0, [])
        scored = []
        for conds in cands:
            tight = {}
            for f, op, thr in conds:                               # keep the tightest bound per (feature, direction)
                k = (f, op)
                tight[k] = min(tight[k], thr) if (k in tight and op == "<=") else (max(tight[k], thr) if k in tight else thr)
            conds = [(f, op, thr) for (f, op), thr in tight.items()]
            m = np.ones(len(Xtr), bool)
            for f, op, thr in conds:
                m &= vec_cond(Xtr[:, feat.index(f)], op, thr)
            n = int(m.sum())
            tp = int((m & (code == ci)).sum())
            if n >= min_support:
                scored.append((tp / n >= min_precision, tp, tp / n, n, conds))
        scored.sort(key=lambda z: (z[0], z[1]), reverse=True)
        for _, tp, prec, n, conds in scored[:per_cat]:
            rules.append(dict(id=f"L{len(rules) + 1:02d}", cat=cat,
                              desc=f"learned on train for {cat} [{CAT_ATTACK[cat]}]; train precision {prec:.3f}, covers {tp / max(len(pos), 1):.1%} of the category",
                              conds=conds))
    return rules


SPACE = {
    "XGBoost": dict(max_depth=[4, 6, 8, 10], learning_rate=[0.05, 0.1, 0.2], subsample=[0.7, 0.8, 1.0],
                    colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 3, 5]),
    "LightGBM": dict(num_leaves=[31, 63, 127], learning_rate=[0.05, 0.1, 0.2], min_child_samples=[5, 10, 20],
                     subsample=[0.7, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0]),
    "DecisionTree": dict(max_depth=[12, 20, 30, None], min_samples_leaf=[1, 2, 5]),
}


def tune(name, K, seed, a, Z, ytr, yva, w, cw, C, rapl):
    """Random search, selection on VALIDATION macro-F1; trained on a class-capped subsample of train."""
    rng = np.random.default_rng(seed)
    per = max(2000, a.tune_rows // K)
    idx = np.concatenate([(ci if len(ci) <= per else rng.choice(ci, per, replace=False))
                          for ci in (np.where(ytr == c)[0] for c in range(K))])
    best, rows = (-1.0, None), []
    for it in range(a.tune):
        p = {k: v[rng.integers(len(v))] for k, v in SPACE[name].items()}
        m = make_models(K, seed, a)[name]
        m.set_params(**p)
        nm = "LightGBM" if name == "LightGBM" else name
        fit_model(nm, m, Z["tr"][idx], ytr[idx], Z["va"], yva, w[idx], cw, rapl)
        pv = safe_proba(m, Z["va"], K).argmax(1)
        f1 = precision_recall_fscore_support(yva, pv, labels=np.arange(K), zero_division=0)[2]
        sc = float(f1[np.bincount(yva, minlength=K) > 0].mean())
        rows.append(dict(model=name, trial=it, val_macro_f1=sc, **{k: str(v) for k, v in p.items()}))
        if sc > best[0]:
            best = (sc, p)
    C["tuning_rows"].extend(rows)
    print(f"    tuned {name}: best val macro-F1={best[0]:.4f} with {best[1]}")
    return best[1]


def ensure_tuned(C, a, rapl, cache):
    """Hyper-parameters are selected once, on the PRIMARY seed's train/validation data, and reused for
    every seed (so only the primary seed is strictly free of hyper-parameter selection overlap)."""
    f = cache / "tuned.json" if cache else None
    if f is not None and f.exists():
        d = json.loads(f.read_text())
        C["tuned"], C["tuning_rows"] = d["tuned"], d["rows"]
        print(f"  loaded tuned hyper-parameters: {C['tuned']}")
        return
    X, y, K = C["X"], C["y"], C["K"]
    tr, va, _, _, _ = split(y, a.seed0)
    Xtr = X[tr]
    keep = np.where(Xtr.var(axis=0) > 0)[0]
    sc = StandardScaler().fit(Xtr[:, keep])
    Z = {"tr": sc.transform(Xtr[:, keep]).astype(np.float32), "va": sc.transform(X[va][:, keep]).astype(np.float32)}
    del Xtr
    w, cw = sqrt_balanced_weights(y[tr], K)
    for key in SPACE:
        t0 = time.time()
        C["tuned"][key] = tune(key, K, a.seed0, a, Z, y[tr], y[va], w, cw, C, rapl)
        print(f"    ({key} search: {(time.time() - t0) / 60:.1f} min)", flush=True)
    if f is not None:
        f.write_text(json.dumps({"tuned": C["tuned"], "rows": C["tuning_rows"]}, default=str))


def run_seed(seed, C, a, rapl, grouped=False):
    X, y, K, benign = C["X"], C["y"], C["K"], C["benign"]
    tr, va, te, excl, fb = split(y, seed, groups=C["groups"] if grouped else None)
    print(f"  seed {seed}: train={len(tr):,} val={len(va):,} test={len(te):,} "
          f"unevaluated classes (<3 flows)={[C['classes'][i] for i in excl]}"
          + (f"  group-fallback classes={[C['classes'][i] for i in fb]}" if grouped else ""))
    Xtr_raw = X[tr]
    keep = np.where(Xtr_raw.var(axis=0) > 0)[0]
    scaler = StandardScaler().fit(Xtr_raw[:, keep])
    Z = {k: scaler.transform(X[i][:, keep]).astype(np.float32) for k, i in (("tr", tr), ("va", va), ("te", te))}
    ytr, yva, yte = y[tr], y[va], y[te]
    rules_seed = learn_rules(Xtr_raw, ytr, C, seed) if a.learn_rules else C["rules"]
    onto = Ontology(rules_seed).fit(X[tr], C["feat"], ytr == benign)
    rule_table = onto.calibrate(X[va], C["cls_code"][yva])
    onto.prepare_scalar()
    w, cw = sqrt_balanced_weights(ytr, K)
    S = dict(rules_seed=rules_seed, seed=seed, tr=tr, va=va, te=te, excl=excl, keep=keep, scaler=scaler, onto=onto, rule_table=rule_table,
             Z=Z, models={}, P={}, rows=[], perclass={}, sweep=[], tau={}, fused={}, train_cost=[])
    ont_va = onto.infer(X[va])
    ont_te = onto.infer(X[te])
    S["ont_va"], S["ont_te"] = ont_va, ont_te
    for name, m in make_models(K, seed, a).items():
        if a.tune and name in ("XGBoost", "LightGBM", "LightGBM (unweighted)", "DecisionTree"):
            key = "LightGBM" if name.startswith("LightGBM") else name
            if key not in C["tuned"]:
                C["tuned"][key] = tune(key, K, seed, a, Z, ytr, yva, w, cw, C, rapl)
            m.set_params(**C["tuned"][key])
        t, e = fit_model(name, m, Z["tr"], ytr, Z["va"], yva, w, cw, rapl)
        S["models"][name] = m
        S["train_cost"].append(dict(seed=seed, model=name, train_seconds=t, train_joules=e))
        pv, pt = safe_proba(m, Z["va"], K), safe_proba(m, Z["te"], K)
        S["P"][name] = (pv, pt)
        for split_n, yy, pp in (("val", yva, pv), ("test", yte, pt)):
            mt, pc = metrics_all(yy, pp.argmax(1), pp, K, benign, auc=a.auc and split_n == "test")
            S["rows"].append(dict(seed=seed, config=name, split=split_n, **mt))
            if split_n == "test":
                S["perclass"][name] = pc
        print(f"    {name:22s} val macro-F1={S['rows'][-2]['macro_f1']:.4f}  test macro-F1={S['rows'][-1]['macro_f1']:.4f}  fit={t:.1f}s")
        if name == "LightGBM (unweighted)":
            continue
        # --- tau sweep on VALIDATION only; pick tau*; apply once to TEST ---
        best = (-1, None)
        for tau in TAU_GRID:
            f, st, _ = fuse(pv, ont_va[0], ont_va[1], tau, a.delta, C["cls_code"], "literal")
            mt, _ = metrics_all(yva, f, None, K, benign, auc=False)
            S["sweep"].append(dict(seed=seed, backend=name, tau=float(tau), **mt,
                                   share_consensus=float((st == 0).mean()), share_silent=float((st == 1).mean()),
                                   share_ontology_override=float((st == 2).mean()), share_ml_override=float((st == 3).mean()),
                                   share_conservative=float((st == 4).mean())))
            if mt["macro_f1"] > best[0] + 1e-12:
                best = (mt["macro_f1"], float(tau))
        S["tau"][name] = best[1]
        for variant in ("literal", "ml_trust"):
            f, st, cf = fuse(pt, ont_te[0], ont_te[1], best[1], a.delta, C["cls_code"], variant)
            mt, pc = metrics_all(yte, f, None, K, benign, auc=False)
            cfg = f"NOHI+{name} ({variant})"
            S["rows"].append(dict(seed=seed, config=cfg, split="test", tau=best[1], **mt))
            S["perclass"][cfg] = pc
            S["fused"][(name, variant)] = (f, st, cf)
    # ontology alone (binary view: it has no fine-grained labels)
    ya = yte != benign
    pa = ont_te[0] >= 0
    S["ontology_alone"] = dict(attack_detection_recall=float((pa & ya).sum() / max(ya.sum(), 1)),
                               attack_detection_precision=float((pa & ya).sum() / max(pa.sum(), 1)),
                               fpr=float((pa & ~ya).sum() / max((~ya).sum(), 1)),
                               category_accuracy_when_fired_on_covered_attacks=float(
                                   (ont_te[0][pa & ya & (C["cls_code"][yte] < 4)] == C["cls_code"][yte][pa & ya & (C["cls_code"][yte] < 4)]).mean())
                               if (pa & ya & (C["cls_code"][yte] < 4)).any() else np.nan)
    sw = pd.DataFrame([r for r in S["sweep"] if r["seed"] == seed])
    S["primary_backend"] = max(WEIGHTED, key=lambda b: next(r["macro_f1"] for r in S["rows"] if r["config"] == b and r["split"] == "val"))
    return S


# ----------------------------------------------------------------------------------------
# 5. End-to-end per-flow latency / energy of the COMPLETE pipeline
# ----------------------------------------------------------------------------------------
class Pipe:
    def __init__(self, S, C, name, a, variant="literal"):
        self.m, self.sc, self.keep, self.on = S["models"][name], S["scaler"], S["keep"], S["onto"]
        self.tau, self.delta, self.C, self.variant, self.name = S["tau"][name], a.delta, C, variant, name
        try:
            self.m.set_params(n_jobs=1)
            if name == "XGBoost":
                self.m.get_booster().set_param({"nthread": 1})
        except Exception:
            pass

    def run(self, x, timing=None, explain=True):
        pc = time.perf_counter
        t0 = pc()
        xs = self.sc.transform(x[self.keep].reshape(1, -1))
        t1 = pc()
        p = self.m.predict_proba(xs)[0]
        if len(p) != self.C["K"]:
            full = np.zeros(self.C["K"]); full[self.m.classes_] = p; p = full
        t2 = pc()
        oc, of, rk = self.on.infer_one(x)
        t3 = pc()
        fin, st, cf = fuse_one(p, oc, of, self.tau, self.delta, self.C["cls_code"], self.variant)
        t4 = pc()
        e = explain_one(st, fin, cf, p, oc, of, rk, x, self.C["classes"], self.on, self.tau, self.delta) if explain else ""
        t5 = pc()
        if timing is not None:
            timing.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4))
        return fin, p, oc, of, rk, e


def latency_profile(S, C, a, rapl, out):
    rng = np.random.default_rng(0)
    pick = rng.choice(len(S["te"]), size=min(a.n_latency, len(S["te"])), replace=False)
    Xte = C["X"][S["te"]]
    rows, mism = [], {}
    for name in WEIGHTED:
        pipe = Pipe(S, C, name, a)
        for i in pick[:30]:
            pipe.run(Xte[i])                                     # warm-up
        tm, fin_s, P_s, oc_s, of_s = [], [], [], [], []
        for i in pick:
            f, p, oc, of, rk, _ = pipe.run(Xte[i], tm)
            fin_s.append(f); P_s.append(p); oc_s.append(oc); of_s.append(of)
        tm = np.array(tm) * 1e6                                  # microseconds
        tot = tm.sum(1)
        # consistency check: scalar fusion == vectorised fusion on the same inputs
        fv, _, _ = fuse(np.array(P_s), np.array(oc_s), np.array(of_s), pipe.tau, a.delta, C["cls_code"], "literal")
        mism[name] = int((fv != np.array(fin_s)).sum())
        # batch throughput (ML only)
        Zb = S["Z"]["te"][pick]
        t0 = time.perf_counter(); S["models"][name].predict_proba(Zb); tb = time.perf_counter() - t0
        row = dict(backend=name, n_flows=len(pick), unit="microseconds per flow (single-flow path)")
        for k, nm in enumerate(["preprocess", "ml_inference", "ontology", "fusion", "explanation"]):
            row[f"{nm}_mean"] = tm[:, k].mean()
        row.update(total_mean=tot.mean(), total_median=np.median(tot), total_p95=np.percentile(tot, 95),
                   total_p99=np.percentile(tot, 99), total_sd=tot.std(ddof=1),
                   ml_only_mean=tm[:, 1].mean(), below_2ms_mean=bool(tot.mean() < 2000),
                   share_of_flows_below_2ms=float((tot < 2000).mean()),
                   batch_ml_throughput_flows_per_s=len(pick) / tb, fuse_vs_fuse_one_mismatches=mism[name])
        rows.append(row)
    df = pd.DataFrame(rows)
    save_table(df, out, "T_latency")
    return df


def energy_profile(S, C, a, rapl, out):
    rows = []
    rng = np.random.default_rng(1)
    pick = rng.choice(len(S["te"]), size=min(500, len(S["te"])), replace=False)
    Xte = C["X"][S["te"]]
    for name in WEIGHTED:
        pipe = Pipe(S, C, name, a)
        cnt = {"i": 0}

        def full():
            pipe.run(Xte[pick[cnt["i"] % len(pick)]]); cnt["i"] += 1

        def ml_only():
            xs = S["Z"]["te"][pick[cnt["i"] % len(pick)]].reshape(1, -1); cnt["i"] += 1
            pipe.m.predict_proba(xs)

        for scope, fn in (("ML inference only", ml_only), ("full pipeline (preprocess+ML+ontology+fusion+explanation)", full)):
            r = rapl.measure(fn, a.energy_n, a.energy_reps)
            base = dict(backend=name, scope=scope, cpu=cpu_model(), rapl_domain=rapl.domain)
            if r is None:
                rows.append(dict(base, status="NOT MEASURED: RAPL counters not readable on this host (no estimate reported)"))
            else:
                rows.append(dict(base, status="measured (RAPL, idle-corrected)", **r))
    df = pd.DataFrame(rows)
    save_table(df, out, "T_energy")
    return df


# ----------------------------------------------------------------------------------------
# 6. Explanation examples and fidelity vs SHAP
# ----------------------------------------------------------------------------------------
def explanation_examples(S, C, a, out):
    name = S["primary_backend"]
    pipe = Pipe(S, C, name, a)
    f, st, cf = S["fused"][(name, "literal")]
    yte = C["y"][S["te"]]
    Xte = C["X"][S["te"]]
    rng = np.random.default_rng(a.seed0)
    lines = [f"# Explanation examples (backend={name}, tau*={S['tau'][name]:.2f}, delta={a.delta}, test set)\n"]
    csv = []
    for s in range(5):
        for correct in (True, False):
            idx = np.where((st == s) & ((f == yte) == correct))[0]
            if len(idx) == 0:
                lines.append(f"- **{STRAT[s]} / {'correct' if correct else 'erroneous'}**: no such case in the test set.\n")
                continue
            i = int(rng.choice(idx))
            fin, p, oc, of, rk, e = pipe.run(Xte[i])
            lines.append(f"## {STRAT[s]} - {'correct' if correct else 'ERRONEOUS'} classification\n"
                         f"true label: **{C['classes'][yte[i]]}**; predicted: **{C['classes'][fin]}**\n\n> {e}\n")
            csv.append(dict(strategy=STRAT[s], correct=correct, true=C["classes"][yte[i]], predicted=C["classes"][fin], explanation=e))
    (out / "explanation_examples.md").write_text("\n".join(lines), encoding="utf-8")
    save_table(pd.DataFrame(csv), out, "T_explanation_examples")
    cnt = pd.Series(st).map(dict(enumerate(STRAT))).value_counts()
    save_table(cnt.rename_axis("strategy").reset_index(name="n_test_flows"), out, "T_strategy_counts")


def shap_fidelity(S, C, a, out):
    try:
        import shap
    except ImportError:
        print("  [skip] shap not installed")
        return
    m = S["models"]["XGBoost"]
    onto, K = S["onto"], C["K"]
    code, conf, rk, F = S["ont_te"]
    pt = S["P"]["XGBoost"][1]
    ml = pt.argmax(1)
    ok = np.where((rk >= 0) & (C["cls_code"][ml] == code) & (code >= 0))[0]
    if len(ok) == 0:
        print("  [skip] no rule-supported detections for fidelity analysis"); return
    rng = np.random.default_rng(0)
    ok = rng.choice(ok, size=min(a.shap_n, len(ok)), replace=False)
    Z = S["Z"]["te"][ok]
    sv = shap.TreeExplainer(m).shap_values(Z)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv
    if sv.shape[0] != len(ok):
        sv = np.transpose(sv, (1, 2, 0)) if sv.shape[0] == K else sv
    kept = [C["feat"][j] for j in S["keep"]]
    res = []
    for r_i, i in enumerate(ok):
        rule = onto.active[rk[i]]
        names = {C["feat"][j] for _, j, _, _ in rule["resolved"]}
        E = [kept.index(nm) for nm in names if nm in kept]
        if not E:
            continue
        k = ml[i]
        mm = len(E)
        top = np.argsort(-np.abs(sv[r_i, :, k]))[:mm]
        base = pt[i, k]

        def drop(idx):
            z = Z[r_i:r_i + 1].copy(); z[0, list(idx)] = 0.0       # 0 = training mean after standardisation
            return base - safe_proba(m, z, K)[0, k]
        rnd = np.mean([drop(rng.choice(len(kept), mm, replace=False)) for _ in range(5)])
        res.append(dict(m=mm, overlap_at_m=len(set(E) & set(top)) / mm, drop_ontology=drop(E), drop_shap=drop(top), drop_random=rnd))
    d = pd.DataFrame(res)
    rows = []
    for col, lab in (("overlap_at_m", "overlap@m between rule features and SHAP top-m"),
                     ("drop_ontology", "prob. drop when masking the rule's features"),
                     ("drop_shap", "prob. drop when masking SHAP top-m features"),
                     ("drop_random", "prob. drop when masking m random features")):
        mu, hw = ci95(d[col])
        rows.append(dict(quantity=lab, mean=mu, ci95_halfwidth=hw, n_flows=len(d)))
    save_table(pd.DataFrame(rows), out, "T_explanation_fidelity")



# ----------------------------------------------------------------------------------------
# 6b. Ablation suite: does the ONTOLOGY (and not just the arbitration logic) add anything?
# ----------------------------------------------------------------------------------------
def conf_from(y, p, K, w=None):
    return np.bincount(y * K + p, weights=w, minlength=K * K).reshape(K, K)


def mets_from_conf(cm, benign, present=None):
    sup, tp, pred = cm.sum(1), np.diag(cm), cm.sum(0)
    prec = np.divide(tp, pred, out=np.zeros(len(tp)), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros(len(tp)), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    pres = (sup > 0) if present is None else present
    nb, tot = sup[benign], cm.sum()
    natt = tot - nb
    att_pred = tot - cm[benign, :].sum() - cm[:, benign].sum() + cm[benign, benign]
    fpr = (nb - cm[benign, benign]) / nb if nb > 0 else np.nan
    return dict(accuracy=float(tp.sum() / tot), macro_f1=float(f1[pres].mean()), fpr=float(fpr),
                attack_recall=float(att_pred / natt) if natt > 0 else np.nan,
                false_alerts_per_million_benign=float(fpr * 1e6))


def fuse_score_only(P, tau, cls_code):
    """Control WITHOUT ontology: same conservative default, triggered by low ML confidence only."""
    ml = P.argmax(1)
    mlc = P[np.arange(len(P)), ml]
    final = ml.copy()
    att = np.where(cls_code != NORMAL_CODE)[0]
    low = np.where(mlc <= tau)[0]
    if len(low) and len(att):
        final[low] = att[np.argmax(P[np.ix_(low, att)], 1)]
    return final


def placebo_ont(onto, X_va, X_te, true_code_va, rng):
    """Placebo rules: same categories and per-rule firing rates as the real rules, but firing is
    independent of the features. Confidence calibrated on validation exactly like real rules."""
    Fv = onto.fired(X_va)
    rate = Fv.mean(0) if Fv.shape[1] else np.array([])
    Pv = rng.random(Fv.shape) < rate
    Pt = rng.random((len(X_te), Fv.shape[1])) < rate
    conf = np.full(len(rate), 0.5)
    for k in range(len(rate)):
        fired = Pv[:, k].sum()
        tp = (Pv[:, k] & (true_code_va == onto.rule_cat[k])).sum()
        conf[k] = (tp + 1) / (fired + 2)

    def inf(F):
        n = len(F)
        if F.shape[1] == 0:
            return np.full(n, -1), np.full(n, 0.5)
        sc = F * conf[None, :]
        best = sc.argmax(1)
        any_ = F.any(1)
        return np.where(any_, onto.rule_cat[best], -1), np.where(any_, sc[np.arange(n), best], 0.5)
    return inf(Pv), inf(Pt)


def prune(onto, rule_table, pmin, minsup):
    """Keep only rules whose VALIDATION precision >= pmin and that fired >= minsup times on validation."""
    n = len(onto.active)
    rt = rule_table.iloc[:n]
    keep = (rt["precision_val"].fillna(0).to_numpy() >= pmin) & (rt["fired_val"].to_numpy() >= minsup)
    o = copy.copy(onto)
    o.active = [r for r, k in zip(onto.active, keep) if k]
    o.rule_cat, o.conf = onto.rule_cat[keep], onto.conf[keep]
    o.prepare_scalar()
    return o, keep


ABL_SYSTEMS = [("ML_ONLY", None, None), ("SCORE_ONLY", "SCORE", None)] + \
              [(f"{o}_{v}", o, v) for o in ("ONTOLOGY", "PLACEBO", "PRUNED") for v in ("literal", "ml_trust")]


def _policy_pred(kind, variant, P, o, tau, a, cc):
    if kind is None:
        return P.argmax(1)
    if kind == "SCORE":
        return fuse_score_only(P, tau, cc)
    return fuse(P, o[0], o[1], tau, a.delta, cc, variant)[0]


def ablation_for_seed(S, C, a, keep_preds=False):
    K, benign, cc, X = C["K"], C["benign"], C["cls_code"], C["X"]
    yva, yte = C["y"][S["va"]], C["y"][S["te"]]
    rng = np.random.default_rng(S["seed"] + 1000)
    onto = S["onto"]
    pv_, pt_ = placebo_ont(onto, X[S["va"]], X[S["te"]], cc[yva], rng)
    pr, kept = prune(onto, S["rule_table"], a.prune_precision, a.prune_min_support)
    ont = {"ONTOLOGY": (S["ont_va"], S["ont_te"]), "PLACEBO": (pv_, pt_),
           "PRUNED": (pr.infer(X[S["va"]]), pr.infer(X[S["te"]]))}
    rows, preds = [], {}
    for backend in sorted({S["primary_backend"], "XGBoost"}):
        Pva, Pte = S["P"][backend]
        base = Pte.argmax(1)
        for name, o, v in ABL_SYSTEMS:
            kind = None if name == "ML_ONLY" else ("SCORE" if name == "SCORE_ONLY" else "FUSE")
            best = (-1.0, None)
            if kind is not None:
                for tau in TAU_GRID:
                    pv = _policy_pred(kind, v, Pva, ont[o][0] if o in ont else None, tau, a, cc)
                    f = mets_from_conf(conf_from(yva, pv, K), benign)["macro_f1"]
                    if f > best[0] + 1e-12:
                        best = (f, float(tau))
            tau = best[1]
            pt = _policy_pred(kind, v, Pte, ont[o][1] if o in ont else None, tau, a, cc)
            m = mets_from_conf(conf_from(yte, pt, K), benign)
            rows.append(dict(seed=S["seed"], backend=backend, policy=name, tau_star=tau,
                             rules_used=(len(pr.active) if o == "PRUNED" else (len(onto.active) if o in ("ONTOLOGY", "PLACEBO") else 0)),
                             modified_predictions=int((pt != base).sum()),
                             corrections=int(((base != yte) & (pt == yte)).sum()),
                             new_errors=int(((base == yte) & (pt != yte)).sum()), **m))
            if keep_preds:
                preds[(backend, name)] = pt
    rt = S["rule_table"].iloc[:len(onto.active)]
    prune_rows = [dict(seed=S["seed"], rule=r["rule"], category=r["category"], fired_val=r["fired_val"],
                       precision_val=r["precision_val"], kept=bool(k)) for (_, r), k in zip(rt.iterrows(), kept)]
    return rows, prune_rows, preds


def budget_table(S, C, a, out, abl_primary=None):
    """Attack recall at a fixed false-positive budget. The operating point is chosen on VALIDATION,
    then applied once to TEST; the budget is reported as met / violated on TEST."""
    K, benign, cc, X = C["K"], C["benign"], C["cls_code"], C["X"]
    yva, yte = C["y"][S["va"]], C["y"][S["te"]]
    rng = np.random.default_rng(S["seed"] + 1000)
    onto = S["onto"]
    pv_, pt_ = placebo_ont(onto, X[S["va"]], X[S["te"]], cc[yva], rng)
    pr, _ = prune(onto, S["rule_table"], a.prune_precision, a.prune_min_support)
    ont = {"ONTOLOGY": (S["ont_va"], S["ont_te"]), "PLACEBO": (pv_, pt_),
           "PRUNED": (pr.infer(X[S["va"]]), pr.infer(X[S["te"]]))}
    budgets = [float(b) for b in a.fpr_budgets.split(",")]
    rows = []
    for backend in sorted({S["primary_backend"], "XGBoost"}):
        Pva, Pte = S["P"][backend]
        for budget in budgets:
            # ML alone: threshold on P(attack) = 1 - P(Normal)
            sv, st = 1 - Pva[:, benign], 1 - Pte[:, benign]
            thr = float(np.quantile(sv[yva == benign], 1 - budget, method="higher"))
            for lab, sc, y in (("val", sv, yva), ("test", st, yte)):
                pa, ya = sc > thr, y != benign
                if lab == "val":
                    fv, rv = float((pa & ~ya).sum() / max((~ya).sum(), 1)), float((pa & ya).sum() / max(ya.sum(), 1))
                else:
                    ft, rt_ = float((pa & ~ya).sum() / max((~ya).sum(), 1)), float((pa & ya).sum() / max(ya.sum(), 1))
            rows.append(dict(backend=backend, fpr_budget=budget, system="ML_ONLY (threshold on 1-P(Normal))", param=f"theta={thr:.4g}",
                             feasible_on_validation=True, val_fpr=fv, val_attack_recall=rv, test_fpr=ft, test_attack_recall=rt_, budget_met_on_test=bool(ft <= budget)))
            for name, o, v in ABL_SYSTEMS[1:]:
                kind = "SCORE" if name == "SCORE_ONLY" else "FUSE"
                cand = []
                for tau in TAU_GRID:
                    pv = _policy_pred(kind, v, Pva, ont[o][0] if o in ont else None, tau, a, cc)
                    m = mets_from_conf(conf_from(yva, pv, K), benign)
                    cand.append((float(tau), m["fpr"], m["attack_recall"]))
                feas = [c for c in cand if c[1] <= budget]
                if not feas:
                    mn = min(cand, key=lambda c: c[1])
                    rows.append(dict(backend=backend, fpr_budget=budget, system=name, param="infeasible on validation",
                                     feasible_on_validation=False, val_fpr=mn[1], val_attack_recall=mn[2]))
                    continue
                tau, fv, rv = max(feas, key=lambda c: c[2])
                pt = _policy_pred(kind, v, Pte, ont[o][1] if o in ont else None, tau, a, cc)
                m = mets_from_conf(conf_from(yte, pt, K), benign)
                rows.append(dict(backend=backend, fpr_budget=budget, system=name, param=f"tau={tau:.2f}", feasible_on_validation=True,
                                 val_fpr=fv, val_attack_recall=rv, test_fpr=m["fpr"], test_attack_recall=m["attack_recall"],
                                 budget_met_on_test=bool(m["fpr"] <= budget)))
    save_table(pd.DataFrame(rows), out, "T_recall_at_fpr_budget")
    if abl_primary is None:
        return
    # ---- matched-FPR comparison: ML alone re-thresholded to the SAME validation FPR as each fusion system ----
    mrows = []
    for backend in sorted({S["primary_backend"], "XGBoost"}):
        Pva, Pte = S["P"][backend]
        sv, st = 1 - Pva[:, benign], 1 - Pte[:, benign]
        ya_v, ya_t = yva != benign, yte != benign
        for _, r in abl_primary[abl_primary.backend == backend].iterrows():
            if r["policy"] == "ML_ONLY":
                continue
            name = r["policy"]
            o = name.split("_")[0] if name != "SCORE_ONLY" else None
            kind = "SCORE" if name == "SCORE_ONLY" else "FUSE"
            v = None if kind == "SCORE" else name.split("_", 1)[1]
            tau = r["tau_star"]
            pv = _policy_pred(kind, v, Pva, ont[o][0] if o in ont else None, tau, a, cc)
            pt = _policy_pred(kind, v, Pte, ont[o][1] if o in ont else None, tau, a, cc)
            fv = float(((pv != benign) & ~ya_v).sum() / max((~ya_v).sum(), 1))
            rv = float(((pv != benign) & ya_v).sum() / max(ya_v.sum(), 1))
            ft = float(((pt != benign) & ~ya_t).sum() / max((~ya_t).sum(), 1))
            rt_ = float(((pt != benign) & ya_t).sum() / max(ya_t.sum(), 1))
            sb = sv[~ya_v]
            t1 = float(np.quantile(sb, 1 - fv, method="higher")) if fv > 0 else float(sb.max())
            cands = []
            for op in (">", ">="):
                f_ = float((sb > t1).mean()) if op == ">" else float((sb >= t1).mean())
                cands.append((abs(f_ - fv), op, f_))
            _, op, mfv = min(cands)
            pa = (st > t1) if op == ">" else (st >= t1)
            mft = float((pa & ~ya_t).sum() / max((~ya_t).sum(), 1))
            mrt = float((pa & ya_t).sum() / max(ya_t.sum(), 1))
            pa_arg = Pte.argmax(1) != benign                                   # ML argmax operating point (parameter-free)
            aft = float((pa_arg & ~ya_t).sum() / max((~ya_t).sum(), 1))
            art = float((pa_arg & ya_t).sum() / max(ya_t.sum(), 1))
            arg_match = bool(abs(aft - ft) <= max(0.3 * ft, 2e-4))
            if arg_match:                                                      # argmax is also a fair baseline at this FPR
                mrt = max(mrt, art)
            ok_match = bool((abs(mfv - fv) <= max(0.3 * fv, 2e-4) and abs(mft - ft) <= max(0.3 * ft, 2e-4)) or arg_match)
            mrows.append(dict(backend=backend, system=name, tau_star=tau, system_val_fpr=fv, system_test_fpr=ft, system_test_attack_recall=rt_,
                              ML_matched_val_fpr=mfv, ML_argmax_test_fpr=aft, ML_argmax_test_attack_recall=art, matching_reliable=ok_match, ML_matched_test_fpr=mft, ML_matched_test_attack_recall=mrt,
                              recall_gain_over_matched_ML=(rt_ - mrt) if ok_match else np.nan))
    save_table(pd.DataFrame(mrows), out, "T_recall_at_matched_fpr")


def bootstrap_table(S, C, a, preds, out):
    """Paired Poisson bootstrap (Bayesian-bootstrap weights) of Delta = system - ML_ONLY on the test set.
    Resampling unit = group (class, block / 5-min burst) when groups exist, else individual flows."""
    K, benign = C["K"], C["benign"]
    yte = C["y"][S["te"]]
    if C["groups"] is not None:
        gid = np.unique(C["groups"][S["te"]], return_inverse=True)[1]
        unit = "group (session proxy)"
    else:
        gid, unit = np.arange(len(yte)), "flow (iid assumption)"
    G = int(gid.max()) + 1
    present = np.bincount(yte, minlength=K) > 0
    rng = np.random.default_rng(a.seed0)
    rows = []
    for backend in sorted({S["primary_backend"], "XGBoost"}):
        codes = {n: yte * K + preds[(backend, n)] for n, _, _ in ABL_SYSTEMS}
        acc = {n: {"macro_f1": [], "fpr": [], "attack_recall": []} for n, _, _ in ABL_SYSTEMS[1:]}
        for _ in range(a.boot):
            w = rng.poisson(1.0, G)[gid].astype(np.float64)
            mb = mets_from_conf(np.bincount(codes["ML_ONLY"], weights=w, minlength=K * K).reshape(K, K), benign, present)
            for n, _, _ in ABL_SYSTEMS[1:]:
                m = mets_from_conf(np.bincount(codes[n], weights=w, minlength=K * K).reshape(K, K), benign, present)
                for k in acc[n]:
                    acc[n][k].append(m[k] - mb[k])
        for n in acc:
            for k, v in acc[n].items():
                v = np.array(v)
                lo, hi = np.percentile(v, [2.5, 97.5])
                rows.append(dict(backend=backend, system=n, metric=f"delta_{k}", mean=float(v.mean()), ci95_lo=float(lo), ci95_hi=float(hi),
                                 ci_includes_zero=bool(lo <= 0 <= hi), n_boot=a.boot, resampling_unit=unit))
    df = pd.DataFrame(rows)
    save_table(df, out, "T_bootstrap_delta_vs_ML")
    return df


def fig_ablation(bt, backend, out):
    d = bt[(bt.backend == backend)]
    sysn = [n for n, _, _ in ABL_SYSTEMS[1:]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a_, met, ttl in ((ax[0], "delta_macro_f1", "Delta macro-F1 vs ML alone"), (ax[1], "delta_fpr", "Delta false-positive rate vs ML alone")):
        x = d[d.metric == met].set_index("system").reindex(sysn)
        y = np.arange(len(sysn))
        a_.errorbar(x["mean"], y, xerr=[x["mean"] - x["ci95_lo"], x["ci95_hi"] - x["mean"]], fmt="o", capsize=3)
        a_.axvline(0, color="k", lw=.8, ls="--"); a_.set_title(ttl + f" ({backend})", fontsize=8)
        a_.set_yticks(y); a_.set_yticklabels(sysn, fontsize=7)
    savefig(fig, out, "fig7_ablation_delta_vs_ML")

# ----------------------------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------------------------
def savefig(fig, out, name):
    (out / "figures").mkdir(exist_ok=True, parents=True)
    fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / "figures" / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(8.2, 3.2)); ax.axis("off"); ax.set_xlim(0, 10.2); ax.set_ylim(0, 4)

    def box(x, y, w, h, t, c="#dce9f7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#33506f"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#33506f"))
    box(0.1, 1.5, 1.6, 1.0, "Network flow\n(CICFlowMeter\nfeatures)", "#eeeeee")
    box(2.3, 2.5, 2.3, 1.0, "Preprocessing\n(train-set scaler)\n+ ML backend\n-> class probs, C_ml")
    box(2.3, 0.3, 2.3, 1.0, "Ontology reasoner\n(rules on raw features)\n-> category, C_ont\n(abstains if no rule)", "#f7e8d0")
    box(5.3, 1.4, 2.5, 1.2, "Risk-averse fusion\n(Algorithm 1; tau, delta)\ncategory-level arbitration\n-> 15-class label", "#e3f3e0")
    box(8.4, 1.4, 1.6, 1.2, "Output:\nlabel + fused conf.\n+ structural\nexplanation e", "#f3e0f3")
    arr(1.7, 2.2, 2.3, 3.0); arr(1.7, 1.8, 2.3, 0.9)
    arr(4.6, 3.0, 5.3, 2.3); arr(4.6, 0.8, 5.3, 1.7); arr(7.8, 2.0, 8.4, 2.0)
    savefig(fig, out, "fig1_architecture")


def fig_performance(main, out):
    cfgs = ["MLP", "XGBoost", "LightGBM", "LightGBM (unweighted)", "DecisionTree", "NOHI+XGBoost (literal)"]
    cfgs = [c for c in cfgs if c in main.config.values]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.26
    for k, (met, col) in enumerate((("accuracy", "#3b78b5"), ("macro_f1", "#2e9e5b"), ("mcc", "#c8442f"))):
        mu = [main[(main.config == c)][f"{met}_mean"].iloc[0] for c in cfgs]
        er = [np.nan_to_num(main[(main.config == c)][f"{met}_ci95"].iloc[0]) for c in cfgs]
        ax.bar(np.arange(len(cfgs)) + (k - 1) * w, mu, w, yerr=er, capsize=2, label=met.replace("_", "-"), color=col)
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([c.replace(" (", "\n(") for c in cfgs], fontsize=7)
    ax.set_ylabel("score (test, mean +/- 95% CI over seeds)"); ax.legend(fontsize=7, ncol=3, loc="lower right"); ax.set_ylim(0, 1.02)
    savefig(fig, out, "fig2_performance")


def fig_confusion(y, p, classes, title, out, name, normalize=True):
    K = len(classes)
    cm = confusion_matrix(y, p, labels=np.arange(K)).astype(float)
    sup = cm.sum(1, keepdims=True)
    M = cm / np.maximum(sup, 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(classes, rotation=90, fontsize=7); ax.set_yticklabels([f"{c} (n={int(s)})" for c, s in zip(classes, sup[:, 0])], fontsize=7)
    for i in range(K):
        for j in range(K):
            v = M[i, j]
            if (normalize and v >= 0.005) or (not normalize and v > 0):
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > (0.5 if normalize else M.max() * .5) else "black")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class (test support)")
    ax.set_title(title + (" - row-normalised" if normalize else " - raw counts"))
    fig.colorbar(im, ax=ax, fraction=0.04)
    savefig(fig, out, name)


def fig_threshold(sw, backend, tau_star, out):
    d = sw[sw.backend == backend].groupby("tau").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for col, lab in (("macro_f1", "macro-F1"), ("macro_precision", "macro-precision"), ("macro_recall", "macro-recall"), ("accuracy", "accuracy")):
        ax[0].plot(d.tau, d[col], marker="o", ms=3, label=lab)
    ax[0].axvline(tau_star, color="k", ls="--", lw=.8, label=f"tau*={tau_star:.2f} (chosen on validation)")
    ax[0].set_xlabel("confidence threshold tau"); ax[0].set_ylabel("validation score"); ax[0].legend(fontsize=6); ax[0].set_title(f"(a) metrics vs tau - {backend}")
    ax[1].plot(d.tau, d.false_alerts_per_million_benign, "r-o", ms=3, label="false alerts / 1M benign flows")
    ax[1].set_xlabel("confidence threshold tau"); ax[1].set_ylabel("false alerts per 1M benign flows")
    a2 = ax[1].twinx()
    a2.plot(d.tau, d.share_conservative, "g--s", ms=3, label="share of conservative-default decisions")
    a2.set_ylabel("share of flows")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=6)
    ax[1].axvline(tau_star, color="k", ls="--", lw=.8); ax[1].set_title("(b) operational cost vs tau (validation)")
    savefig(fig, out, "fig3_threshold_sensitivity")


def fig_latency(lat, out):
    fig, ax = plt.subplots(figsize=(7.5, 3))
    stages = ["preprocess", "ml_inference", "ontology", "fusion", "explanation"]
    cols = ["#9ecae1", "#3182bd", "#fdae6b", "#a1d99b", "#cbb5e8"]
    left = np.zeros(len(lat))
    for s, c in zip(stages, cols):
        v = lat[f"{s}_mean"].to_numpy()
        ax.barh(lat.backend, v, left=left, color=c, label=s.replace("_", " "))
        left += v
    ax.axvline(2000, color="r", ls="--"); ax.text(2000, -0.45, " 2 ms", color="r", fontsize=7)
    ax.set_xlabel("mean end-to-end latency per flow (microseconds, single-flow path)"); ax.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.18))
    savefig(fig, out, "fig5_latency_breakdown")


def fig_perclass(pcw, out):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(pcw.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pcw.shape[1])); ax.set_xticklabels([c.replace(" (", "\n(") for c in pcw.columns], fontsize=6)
    ax.set_yticks(range(pcw.shape[0])); ax.set_yticklabels(pcw.index, fontsize=7)
    for i in range(pcw.shape[0]):
        for j in range(pcw.shape[1]):
            ax.text(j, i, f"{pcw.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if pcw.values[i, j] < .6 else "k")
    fig.colorbar(im, ax=ax, fraction=.04, label="per-class F1 (test)")
    savefig(fig, out, "fig6_perclass_f1")


# ----------------------------------------------------------------------------------------
# 7b. Cross-dataset (external) evaluation: train on CIC-IDS2017, test on another CICFlowMeter-based set
# ----------------------------------------------------------------------------------------
def external_eval(S, C, a, rapl, out):
    print(f"  loading external dataset: {a.external}")
    df = read_table(a.external)
    lab_col = next((c for c in df.columns if norm_col(c) in ("label", "attack_label", "attack", "class")), None)
    if lab_col is None:
        print("  [skip] external: no label column"); return
    featset, colmap = set(C["feat"]), {}
    for c in df.columns:
        for cand in (norm_col(c), canon(c)):
            if cand in featset and cand not in colmap.values():
                colmap[c] = cand
                break
    common = [f for f in C["feat"] if f in colmap.values()]
    if len(common) < 20:
        print(f"  [skip] external: only {len(common)} shared features (needs a CICFlowMeter-based dataset such as CSE-CIC-IDS2018)"); return
    inv = {v: k for k, v in colmap.items()}
    Xe = df[[inv[f] for f in common]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    le = df[lab_col].map(norm_label).to_numpy()
    del df
    ok = np.isfinite(Xe).all(axis=1)
    d = pd.DataFrame(Xe[ok], columns=common); d["__y"] = le[ok]
    first = ~d.duplicated().to_numpy()
    Xe, le = Xe[ok][first], le[ok][first]
    rng = np.random.default_rng(a.seed0)
    keep = []
    for l in np.unique(le):
        i = np.where(le == l)[0]
        cap = a.external_cap * (4 if ext_label_code(l) == NORMAL_CODE else 1)
        keep.append(i if len(i) <= cap else rng.choice(i, cap, replace=False))
    keep = np.sort(np.concatenate(keep)); Xe, le = Xe[keep], le[keep]
    ecode = np.array([ext_label_code(l) for l in le])
    cidx = [C["feat"].index(f) for f in common]
    X, y, K, benign = C["X"], C["y"], C["K"], C["benign"]
    tr, va = S["tr"], S["va"]
    kp = np.where(X[tr][:, cidx].var(axis=0) > 0)[0]
    sc = StandardScaler().fit(X[tr][:, cidx][:, kp])
    Ztr, Zva, Ze = (sc.transform(M[:, cidx][:, kp]).astype(np.float32) for M in (X[tr], X[va], Xe))
    w, _ = sqrt_balanced_weights(y[tr], K)
    m = xgb.XGBClassifier(**S["models"]["XGBoost"].get_params())
    m.fit(Ztr, y[tr], sample_weight=w, eval_set=[(Zva, y[va])], verbose=False)
    P = safe_proba(m, Ze, K)
    on = Ontology(C["rules"]).fit(X[tr][:, cidx], common, y[tr] == benign)
    on.calibrate(X[va][:, cidx], C["cls_code"][y[va]])
    oc, of, _, _ = on.infer(Xe)
    tau = S["tau"]["XGBoost"]
    systems = {"XGBoost alone": P.argmax(1)}
    for v in ("literal", "ml_trust"):
        systems[f"NOHI+XGBoost ({v})"] = fuse(P, oc, of, tau, a.delta, C["cls_code"], v)[0]
    ya = ecode != NORMAL_CODE
    summ, bylab = [], []
    for nm, pr in systems.items():
        pa = pr != benign
        summ.append(dict(system=nm, n_external_flows=len(le), n_attack=int(ya.sum()), shared_features=len(common), features_2017=len(C["feat"]),
                         attack_detection_recall=float((pa & ya).sum() / max(ya.sum(), 1)),
                         attack_detection_precision=float((pa & ya).sum() / max(pa.sum(), 1)),
                         fpr=float((pa & ~ya).sum() / max((~ya).sum(), 1)), tau=tau))
        for l in np.unique(le):
            mk = le == l
            cov = ecode[mk][0] < NORMAL_CODE
            bylab.append(dict(system=nm, external_label=l, family=("Normal" if ecode[mk][0] == NORMAL_CODE else ("not in ontology" if ecode[mk][0] == UNCOV_CODE else CATS[ecode[mk][0]])),
                              n=int(mk.sum()), flagged_as_attack=float(pa[mk].mean()),
                              family_accuracy=float((C["cls_code"][pr[mk]] == ecode[mk][0]).mean()) if cov else np.nan))
    save_table(pd.DataFrame(summ), out, "T_external_summary")
    save_table(pd.DataFrame(bylab), out, "T_external_by_label")


def write_paper_numbers(out, main, lat, P, a):
    """Single source of truth for every number quoted in the manuscript / response letter."""
    cl = pd.read_csv(out / "tables" / "T_cleaning.csv")
    tot = cl[cl["class"] == "TOTAL"].iloc[0]
    d = dict(cleaning={c_: int(tot[c_]) for c_ in cl.columns if c_ != "class"},
             primary_backend=P["primary_backend"], tau_star=P["tau"], delta=a.delta, n_seeds=a.seeds,
             main_test={r["config"]: {m: [float(r[f"{m}_mean"]), float(r[f"{m}_ci95"]) if pd.notna(r[f"{m}_ci95"]) else None]
                                      for m in ("accuracy", "macro_f1", "mcc", "fpr", "false_alerts_per_million_benign")}
                        for _, r in main.iterrows()},
             latency_us={r["backend"]: dict(total_mean=float(r["total_mean"]), total_p99=float(r["total_p99"]),
                                            share_below_2ms=float(r["share_of_flows_below_2ms"])) for _, r in lat.iterrows()})
    (out / "paper_numbers.json").write_text(json.dumps(d, indent=2, default=float))


# ----------------------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5, help="number of independent re-splits/re-trainings")
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=0, help="stratified subsample for development (0 = all)")
    ap.add_argument("--delta", type=float, default=0.9)
    ap.add_argument("--mlp-max-iter", type=int, default=60)
    ap.add_argument("--n-latency", type=int, default=1000)
    ap.add_argument("--energy-n", type=int, default=2000); ap.add_argument("--energy-reps", type=int, default=5)
    ap.add_argument("--tdp-watts", type=float, default=None)
    ap.add_argument("--shap", action="store_true"); ap.add_argument("--shap-n", type=int, default=300)
    ap.add_argument("--no-auc", dest="auc", action="store_false")
    ap.add_argument("--grouped-check", action="store_true", help="extra run with burst-grouped split (timestamp, or --block-split)")
    ap.add_argument("--block-split", type=int, default=0, help="proxy when no timestamp: consecutive blocks of N same-class flows (file order) stay together")
    ap.add_argument("--rules", help="JSON rule base (see rules_default.json); default = built-in reconstruction")
    ap.add_argument("--labelmap", help="JSON label->ontology category overrides")
    ap.add_argument("--prune-precision", type=float, default=0.5, help="rule pruning: min validation precision")
    ap.add_argument("--prune-min-support", type=int, default=20, help="rule pruning: min validation firings")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap replicates for paired deltas (0 = skip)")
    ap.add_argument("--fpr-budgets", default="0.001,0.0001", help="comma-separated FPR budgets")
    ap.add_argument("--learn-rules", action="store_true", help="learn 12 ontology-anchored rules on each seed's TRAINING data (see learn_rules)")
    ap.add_argument("--seed-cache", help="folder where finished non-primary seeds (and tuned hyper-parameters) are cached")
    ap.add_argument("--cache-only", action="store_true", help="compute + cache the non-primary seeds, then exit")
    ap.add_argument("--tune", type=int, default=0, help="random-search trials per tree model (validation macro-F1); 0 = off")
    ap.add_argument("--tune-rows", type=int, default=200000)
    ap.add_argument("--external", help="external CICFlowMeter-based dataset (e.g. CSE-CIC-IDS2018 CSV/parquet/folder)")
    ap.add_argument("--external-cap", type=int, default=50000, help="max flows per external label (benign: 4x)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if a.labelmap:
        load_labelmap(a.labelmap)
    rules = load_rules(a.rules) if a.rules else RULES
    (out / "rules_used.json").write_text(json.dumps(rules_to_json(rules), indent=2))
    print("== Loading and cleaning ==")
    df, lab_col = load_data(a)
    X, ylab, feat, ts = clean(df, lab_col, out)
    del df
    classes = np.array(sorted(set(ylab), key=lambda s: (s != "Normal", s)))      # Normal first
    y = np.searchsorted(classes, ylab) if False else pd.Series(ylab).map({c: i for i, c in enumerate(classes)}).to_numpy()
    K, benign = len(classes), 0
    if a.max_rows and len(y) > a.max_rows:                                       # stratified dev subsample
        rng = np.random.default_rng(a.seed0)
        keep = np.concatenate([rng.choice(np.where(y == c)[0], size=max(min((y == c).sum(), 5), int((y == c).sum() * a.max_rows / len(y))), replace=False) for c in range(K)])
        X, y = X[keep], y[keep]
        ts = ts[keep] if ts is not None else None
        print(f"  [dev] subsampled to {len(y):,} rows")
    cls_code = np.array([label_to_code(c) for c in classes])
    groups = None
    if ts is not None and not pd.isna(ts).all():
        bucket = pd.to_datetime(pd.Series(ts)).dt.floor("5min").astype("int64").astype(str).to_numpy()
        groups = np.char.add(y.astype(str), np.char.add("_", bucket))
    if groups is None and a.block_split > 0:
        pos = np.zeros(len(y), int)
        for c in range(K):
            i = np.where(y == c)[0]
            pos[i] = np.arange(len(i)) // a.block_split
        groups = np.char.add(y.astype(str), np.char.add("_", pos.astype(str)))
        print(f"  [info] no timestamp: using file-order blocks of {a.block_split} same-class flows as a proxy for sessions")
    C = dict(X=X, y=y, K=K, benign=benign, classes=classes, feat=feat, cls_code=cls_code, groups=groups,
             rules=rules, tuned={}, tuning_rows=[])
    rapl = Rapl()
    print(f"== Setup: K={K} classes, {X.shape[1]} candidate features, RAPL={'available' if rapl.ok else 'NOT available'}, CPU={cpu_model()}")

    print("== Experiments ==")
    cache = Path(a.seed_cache) if a.seed_cache else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    if a.tune:
        ensure_tuned(C, a, rapl, cache)
    sig = hashlib.sha256(json.dumps([len(y), feat, C["tuned"], rules_to_json(rules), a.delta, a.mlp_max_iter,
                                     a.prune_precision, a.prune_min_support, np.bincount(y).tolist(), bool(a.learn_rules)], default=str).encode()).hexdigest()
    LIGHT = ("seed", "rows", "sweep", "train_cost", "tau", "primary_backend", "excl", "ontology_alone", "perclass")
    runs, abl_rows, prune_rows, abl_preds = [], [], [], {}
    for s_ in (range(1, a.seeds) if a.cache_only else range(a.seeds)):
        seed, t0 = a.seed0 + s_, time.time()
        cf = cache / f"seed_{seed}.pkl" if cache is not None else None
        if s_ > 0 and cf is not None and cf.exists():
            d_ = pickle.load(open(cf, "rb"))
            if d_["sig"] == sig:
                runs.append(d_["S"]); abl_rows += d_["abl"]; prune_rows += d_["prune"]
                print(f"  seed {seed}: loaded from cache", flush=True)
                continue
            print(f"  seed {seed}: cache signature mismatch -> recomputed")
        Sx = run_seed(seed, C, a, rapl)
        r_, pr_, pd_ = ablation_for_seed(Sx, C, a, keep_preds=(s_ == 0))
        if s_ == 0:
            abl_preds = pd_
        else:                                                                   # keep only light results (memory)
            Sx = {k_: Sx[k_] for k_ in LIGHT}
            if cf is not None:
                pickle.dump({"sig": sig, "S": Sx, "abl": r_, "prune": pr_}, open(cf, "wb"))
        abl_rows += r_; prune_rows += pr_
        runs.append(Sx)
        print(f"  seed {seed}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.cache_only:
        print("cache-only: non-primary seeds cached; exiting")
        return
    P = runs[0]                                                                 # primary seed for figures/examples
    rows = pd.DataFrame([r for R in runs for r in R["rows"]])
    sweeps = pd.DataFrame([r for R in runs for r in R["sweep"]])
    cost = pd.DataFrame([r for R in runs for r in R["train_cost"]])

    if C["tuning_rows"]:
        save_table(pd.DataFrame(C["tuning_rows"]), out, "T_tuning_trials")
    # ---- Table: protocol / split counts ----
    yt = {"train": y[P["tr"]], "validation": y[P["va"]], "test": y[P["te"]]}
    proto = pd.DataFrame({"class": classes, "ontology_category": [("Normal" if c == NORMAL_CODE else ("not represented" if c == UNCOV_CODE else CATS[c])) for c in cls_code]})
    for k, v in yt.items():
        proto[k] = np.bincount(v, minlength=K)
    proto["evaluated"] = np.where(proto["test"] > 0, "yes", "NO (<3 flows)")
    save_table(proto, out, "T_protocol_split_counts")
    # ---- Table: hyper-parameters actually used ----
    hp = []
    for n, m in P["models"].items():                      # parameters of the models ACTUALLY fitted (incl. tuned ones)
        for k, v in m.get_params().items():
            if k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "num_leaves", "min_child_samples", "early_stopping_rounds",
                     "hidden_layer_sizes", "batch_size", "n_iter_no_change", "max_iter", "min_impurity_decrease", "criterion", "objective", "tree_method", "learning_rate_init", "min_child_weight", "min_samples_leaf"):
                hp.append(dict(model=n, parameter=k, value=str(v)))
    hp.append(dict(model="XGBoost / LightGBM / DecisionTree", parameter="imbalance handling", value="sqrt-balanced class weights (MLP: none, not supported by sklearn)"))
    hp.append(dict(model="fusion", parameter="delta (only scales reported confidence)", value=a.delta))
    hp.append(dict(model="fusion", parameter="tau grid / selection", value=f"{TAU_GRID[0]}..{TAU_GRID[-1]} step 0.05; argmax validation macro-F1"))
    save_table(pd.DataFrame(hp), out, "T_hyperparameters")
    if a.learn_rules:
        (out / "learned_rules_primary_seed.json").write_text(json.dumps(rules_to_json(P["rules_seed"]), indent=2))
    # ---- Table: rules ----
    save_table(P["rule_table"], out, "T_rules")

    # ---- Table: main results (mean +/- CI over seeds, TEST) ----
    te = rows[rows.split == "test"]
    mets = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "mcc", "auc_ovr_macro", "attack_detection_recall", "fpr", "false_alerts_per_million_benign"]
    main_rows = []
    for cfg in te.config.unique():
        d = te[te.config == cfg]
        r = dict(config=cfg, n_seeds=len(d))
        for m_ in mets:
            mu, hw = ci95(d[m_]); r[f"{m_}_mean"], r[f"{m_}_ci95"] = mu, hw
        if "tau" in d: r["tau_star_mean"] = d["tau"].mean() if d["tau"].notna().any() else np.nan
        main_rows.append(r)
    main = pd.DataFrame(main_rows)
    save_table(main, out, "T_main_test")
    val_main = rows[rows.split == "val"].groupby("config")[["accuracy", "macro_f1", "mcc"]].mean().reset_index()
    save_table(val_main, out, "T_main_validation")

    # ---- per-class (primary seed, test) ----
    long = []
    for cfg, (pr, rc, f1, sup) in P["perclass"].items():
        for k in range(K):
            lo, hi = clopper_pearson(int(round(rc[k] * sup[k])), int(sup[k]))
            long.append(dict(config=cfg, **{"class": classes[k]}, precision=pr[k], recall=rc[k], f1=f1[k], support_test=int(sup[k]), recall_ci95_lo=lo, recall_ci95_hi=hi))
    pcl = pd.DataFrame(long)
    save_table(pcl, out, "T_perclass_long")
    sel = [c for c in ["MLP", "XGBoost", "LightGBM", "LightGBM (unweighted)", "DecisionTree", "NOHI+XGBoost (literal)"] if c in pcl.config.values]
    wide = pcl[pcl.config.isin(sel)].pivot(index="class", columns="config", values="f1").reindex(classes)[sel]
    wide.insert(0, "support_test", pcl[pcl.config == sel[0]].set_index("class")["support_test"].reindex(classes).values)
    save_table(wide.reset_index(), out, "T_perclass_f1_wide")

    pcs = []
    for R in runs:
        for cfg, (pr_c, rc_c, f1_c, sup_c) in R["perclass"].items():
            for k in range(K):
                pcs.append({"seed": R["seed"], "config": cfg, "class": classes[k], "precision": pr_c[k], "recall": rc_c[k],
                            "f1": f1_c[k], "support_test": int(sup_c[k])})
    pcs = pd.DataFrame(pcs)
    g_ = pcs.groupby(["config", "class"], sort=False)
    pcs_agg = g_.agg(n_seeds=("f1", "size"), support_test_mean=("support_test", "mean"), precision_mean=("precision", "mean"),
                     recall_mean=("recall", "mean"), recall_sd=("recall", "std"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                     f1_min=("f1", "min"), f1_max=("f1", "max")).reset_index()
    save_table(pcs_agg, out, "T_perclass_over_seeds")
    # ---- threshold table (validation) ----
    tt = sweeps[sweeps.seed == P["seed"]].copy()
    save_table(tt, out, "T_threshold_sweep_validation")
    # ---- ablation (test, primary seed) ----
    pb = P["primary_backend"]
    ab = []
    yte = y[P["te"]]
    for b in sorted({pb, "XGBoost"}):
        base = te[(te.seed == P["seed"]) & (te.config == b)].iloc[0]
        ab.append(dict(system=f"{b} alone (ML only)", macro_f1=base.macro_f1, mcc=base.mcc, accuracy=base.accuracy, attack_recall=base.attack_detection_recall, fpr=base.fpr))
        for v in ("literal", "ml_trust"):
            r = te[(te.seed == P["seed"]) & (te.config == f"NOHI+{b} ({v})")].iloc[0]
            ab.append(dict(system=f"NOHI+{b} ({v}), tau*={P['tau'][b]:.2f}", macro_f1=r.macro_f1, mcc=r.mcc, accuracy=r.accuracy, attack_recall=r.attack_detection_recall, fpr=r.fpr))
    oa = P["ontology_alone"]
    ab.append(dict(system="Ontology alone (binary view only)", attack_recall=oa["attack_detection_recall"], fpr=oa["fpr"], precision=oa["attack_detection_precision"]))
    save_table(pd.DataFrame(ab), out, "T_ablation_test")
    # ---- McNemar (exact), primary seed ----
    mc = []
    for b in WEIGHTED:
        for v in ("literal", "ml_trust"):
            f = P["fused"][(b, v)][0]
            bb, cc, pv = mcnemar_exact(yte, P["P"][b][1].argmax(1), f)
            mc.append(dict(comparison=f"{b} vs NOHI+{b} ({v})", b_ml_right_fusion_wrong=bb, c_ml_wrong_fusion_right=cc, p_exact=pv, identical_predictions=bool((P["P"][b][1].argmax(1) == f).all())))
    for b1, b2 in (("XGBoost", "LightGBM"), ("XGBoost", "MLP"), ("XGBoost", "DecisionTree")):
        bb, cc, pv = mcnemar_exact(yte, P["P"][b1][1].argmax(1), P["P"][b2][1].argmax(1))
        mc.append(dict(comparison=f"{b1} vs {b2}", b_ml_right_fusion_wrong=bb, c_ml_wrong_fusion_right=cc, p_exact=pv))
    save_table(pd.DataFrame(mc), out, "T_mcnemar")
    # ---- ablation suite ----
    ab = pd.DataFrame(abl_rows)
    save_table(ab[ab.seed == P["seed"]], out, "T_ablation_policies_primary_seed")
    agg = []
    for (b_, pol), d in ab.groupby(["backend", "policy"], sort=False):
        r_ = dict(backend=b_, policy=pol, n_seeds=len(d), tau_star_mean=d["tau_star"].mean(), rules_used=int(d["rules_used"].iloc[0]),
                  modified_predictions_mean=d["modified_predictions"].mean(), corrections_mean=d["corrections"].mean(), new_errors_mean=d["new_errors"].mean())
        for m_ in ("macro_f1", "accuracy", "fpr", "attack_recall", "false_alerts_per_million_benign"):
            r_[f"{m_}_mean"], r_[f"{m_}_ci95"] = ci95(d[m_])
        agg.append(r_)
    save_table(pd.DataFrame(agg), out, "T_ablation_policies_over_seeds")
    save_table(pd.DataFrame(prune_rows), out, "T_rule_pruning")
    budget_table(P, C, a, out, ab[ab.seed == P["seed"]])
    bt = None
    if a.boot > 0:
        bt = bootstrap_table(P, C, a, abl_preds, out)
    # ---- training cost ----
    tc = cost.groupby("model").agg(train_seconds_mean=("train_seconds", "mean"), train_seconds_sd=("train_seconds", "std"),
                                   train_joules_mean=("train_joules", "mean")).reset_index()
    tc["energy_status"] = "measured (RAPL)" if rapl.ok else "NOT MEASURED (RAPL unavailable)"
    save_table(tc, out, "T_training_cost")
    sizes = []
    import io, joblib
    for n, m in P["models"].items():
        buf = io.BytesIO(); joblib.dump(m, buf); sizes.append(dict(model=n, serialized_size_MB=buf.tell() / 2 ** 20))
    save_table(pd.DataFrame(sizes), out, "T_model_size")

    # ---- latency / energy / explanations ----
    print("== Latency (complete single-flow pipeline) ==")
    lat = latency_profile(P, C, a, rapl, out)
    print(lat[["backend", "ml_inference_mean", "ontology_mean", "explanation_mean", "total_mean", "total_p99", "fuse_vs_fuse_one_mismatches"]].round(1).to_string(index=False))
    print("== Energy ==")
    en = energy_profile(P, C, a, rapl, out)
    print(en[["backend", "scope", "status"]].head(2).to_string(index=False))
    print("== Explanations ==")
    explanation_examples(P, C, a, out)
    if a.shap:
        print("== SHAP fidelity ==")
        shap_fidelity(P, C, a, out)

    if a.external:
        print("== External (cross-dataset) evaluation ==")
        external_eval(P, C, a, rapl, out)

    # ---- optional grouped-split robustness ----
    if a.grouped_check and groups is not None:
        print("== Burst-grouped split check ==")
        G = run_seed(a.seed0, C, a, rapl, grouped=True)
        gr = pd.DataFrame([r for r in G["rows"] if r["split"] == "test"])
        save_table(gr, out, "T_grouped_split_test")

    # ---- figures ----
    print("== Figures ==")
    fig_architecture(out)
    fig_performance(main, out)
    f_best = P["fused"][(pb, "literal")][0]
    fig_confusion(yte, f_best, classes, f"NOHI+{pb}", out, "fig4_confusion_normalised", True)
    fig_confusion(yte, f_best, classes, f"NOHI+{pb}", out, "fig4b_confusion_counts", False)
    for nm_, pred_ in ((f"confusion_{pb}_alone", P["P"][pb][1].argmax(1)), (f"confusion_NOHI_{pb}_literal", f_best)):
        pd.DataFrame(confusion_matrix(yte, pred_, labels=np.arange(K)), index=classes, columns=classes).to_csv(out / "tables" / f"{nm_}.csv")
    fig_threshold(sweeps, pb, P["tau"][pb], out)
    fig_latency(lat, out)
    fig_perclass(wide.drop(columns="support_test"), out)
    if bt is not None:
        fig_ablation(bt, pb, out)

    write_paper_numbers(out, main, lat, P, a)
    pn = json.loads((out / "paper_numbers.json").read_text())
    pn["ablation_over_seeds"] = pd.DataFrame(agg).to_dict("records")
    if bt is not None:
        pn["bootstrap_delta_vs_ML"] = bt.to_dict("records")
    (out / "paper_numbers.json").write_text(json.dumps(pn, indent=2, default=float))
    fp = hashlib.sha256(("|".join(feat) + str(np.bincount(y).tolist())).encode() + X[:500].tobytes() + X[-500:].tobytes()).hexdigest()
    summary = dict(data_fingerprint_sha256=fp, python=sys.version.split()[0], numpy=np.__version__, xgboost=xgb.__version__, lightgbm=lgb.__version__,
                   cpu=cpu_model(), rapl=rapl.ok, args=vars(a), n_flows_after_cleaning=int(len(y)), n_features_after_cleaning=int(X.shape[1]),
                   n_features_after_zero_variance_filter_primary_seed=int(len(P["keep"])), primary_backend=pb, tau_star=P["tau"],
                   active_rules=[r["id"] for r in P["onto"].active], runtime_minutes=(time.time() - t_start) / 60)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Outputs in {out}/ (tables/, figures/, explanation_examples.md, run_summary.json)")


if __name__ == "__main__":
    main()
