#!/usr/bin/env python3
"""Energy bench for NOHI, designed for the Green Metrics Tool (each sub-command is one measured flow).

  prepare      download the public CIC-IDS2017 flow export and keep a stratified random sample (run at image build time)
  train        train XGBoost and fit/calibrate the rule layer on the sample; store the state
  batch-ml     classifier only, vectorised, N flows          batch-nohi   classifier + rules + fusion, vectorised
  single-ml    classifier only, one flow at a time           single-nohi  full NOHI path with the decision trace

Energy per flow = energy of the flow reported by the bench / N (report it gross and net of the idle power)."""
import argparse, glob, os, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nohi_v2 as N

D = Path(os.environ.get("NOHI_BENCH_DIR", "/bench")); DELTA = 0.9


def make_sample(source, n, floor=500):
    """Stratified random sample of the public export, read batch by batch to keep memory low. Identifier-like columns are
    removed; every class keeps at least `floor` flows (or all it has)."""
    import pyarrow as pa, pyarrow.parquet as pq
    pf = pq.ParquetFile(source); names = pf.schema_arrow.names; lab = next(c for c in names if N.norm_col(c) in ("attack_label", "label"))
    labels = pf.read(columns=[lab]).column(0).to_pandas().astype(str); frac = n / len(labels); rng = np.random.default_rng(0); mask = np.zeros(len(labels), bool)
    for c, idx in labels.groupby(labels).indices.items():
        mask[rng.choice(idx, size=min(len(idx), max(floor, int(round(frac * len(idx))))), replace=False)] = True
    cols = [c for c in names if N.norm_col(c) not in ("flow_id", "source_port", "source_ip", "destination_ip", "timestamp")]; parts, off = [], 0
    for bt in pf.iter_batches(batch_size=200000, columns=cols):
        m = mask[off:off + bt.num_rows]; off += bt.num_rows
        if m.any(): parts.append(pa.Table.from_batches([bt]).filter(pa.array(m)))
    pq.write_table(pa.concat_tables(parts), D / "sample.parquet"); print(f"sample written: {int(mask.sum())} of {len(labels)} flows, {labels.nunique()} classes, {len(cols)} columns kept")


def prepare(n, source=None):
    """Public source: Hugging Face data set rdpahalavan/CIC-IDS2017, file Network-Flows/CICIDS_Flow.parquet (the export read by the
    nids-datasets package, about 370 MB). `source` may point to a local copy of that file."""
    if source is None:
        from huggingface_hub import hf_hub_download
        source = hf_hub_download(repo_id="rdpahalavan/CIC-IDS2017", subfolder="Network-Flows", filename="CICIDS_Flow.parquet", repo_type="dataset", local_dir=str(D / "raw"))
    make_sample(source, n)


def train(sample):
    a = SimpleNamespace(data=str(sample), synthetic=False, seed0=42, mlp_max_iter=60)
    df, lab = N.load_data(a); df = df.drop(columns=[c for c in df.columns if c in ("capture_file", "source_port", "flow_id")])
    X, ylab, feat, _ = N.clean(df, lab, D / "clean")
    classes = np.array(sorted(set(ylab), key=lambda s: (s != "Normal", s))); y = pd.Series(ylab).map({c: i for i, c in enumerate(classes)}).to_numpy(); K = len(classes)
    cc = np.array([N.label_to_code(c) for c in classes]); tr, va, te, _, _ = N.split(y, 42)
    from sklearn.preprocessing import StandardScaler
    keep = np.where(X[tr].var(axis=0) > 0)[0]; sc = StandardScaler().fit(X[tr][:, keep]); Z = lambda i: sc.transform(X[i][:, keep]).astype(np.float32)
    t0 = time.perf_counter(); m = N.make_models(K, 42, a)["XGBoost"]; w, cw = N.sqrt_balanced_weights(y[tr], K); N.fit_model("XGBoost", m, Z(tr), y[tr], Z(va), y[va], w, cw, N.Rapl())
    t1 = time.perf_counter(); onto = N.Ontology(N.RULES).fit(X[tr], feat, y[tr] == 0); onto.calibrate(X[va], cc[y[va]]); t2 = time.perf_counter()
    Pv = N.safe_proba(m, Z(va), K); ov = onto.infer(X[va])
    f1 = lambda p: N.metrics_all(y[va], p, None, K, 0, auc=False)[0]["macro_f1"]
    tau = max(((f1(N.fuse(Pv, ov[0], ov[1], t, DELTA, cc, "literal")[0]), -float(t)) for t in N.TAU_GRID))[1] * -1
    pickle.dump(dict(m=m, sc=sc, keep=keep, onto=onto, tau=tau, cc=cc, classes=classes, K=K, Xte=X[te]), open(D / "state.pkl", "wb"))
    print(f"trained on {len(tr)} flows, {len(keep)} features, {K} classes, tau={tau}, test flows kept={len(te)}, active rules={len(onto.active)}/{len(N.RULES)}")
    print(f"classifier training {t1 - t0:.1f} s, rule layer fit+calibration {t2 - t1:.2f} s")
    if len(onto.active) < len(N.RULES): print("WARNING: some rules could not be resolved on these column names:", sorted({r['id'] for r in N.RULES} - {r['id'] for r in onto.active}))


def load():
    S = pickle.load(open(D / "state.pkl", "rb")); return SimpleNamespace(**S)


def batch(mode, n, chunk=100000):
    S = load(); done, alerts = 0, 0; t0 = time.perf_counter()
    while done < n:
        X = S.Xte[np.arange(done, min(done + chunk, n)) % len(S.Xte)]; P = N.safe_proba(S.m, S.sc.transform(X[:, S.keep]).astype(np.float32), S.K)
        if mode == "nohi":
            code, conf, rk, _ = S.onto.infer(X); fin = N.fuse(P, code, conf, S.tau, DELTA, S.cc, "literal")[0]
        else: fin = P.argmax(1)
        alerts += int((fin != 0).sum()); done += len(X)
    dt = time.perf_counter() - t0; print(f"batch-{mode}: {done} flows in {dt:.2f} s ({done / dt:,.0f} flows/s), alerts={alerts}")


def single(mode, n):
    S = load(); S.onto.prepare_scalar(); alerts = 0
    try:
        S.m.set_params(n_jobs=1); S.m.get_booster().set_param({"nthread": 1})
    except Exception: pass
    t0 = time.perf_counter()
    for i in range(n):
        x = S.Xte[i % len(S.Xte)]; p = S.m.predict_proba(S.sc.transform(x[S.keep].reshape(1, -1)))[0]
        if len(p) != S.K:
            full = np.zeros(S.K); full[S.m.classes_] = p; p = full
        if mode == "nohi":
            oc, of, rk = S.onto.infer_one(x); fin, st, cf = N.fuse_one(p, oc, of, S.tau, DELTA, S.cc, "literal")
            N.explain_one(st, fin, cf, p, oc, of, rk, x, S.classes, S.onto, S.tau, DELTA)
        else: fin = int(np.argmax(p))
        alerts += int(fin != 0)
    dt = time.perf_counter() - t0; print(f"single-{mode}: {n} flows in {dt:.2f} s ({1e3 * dt / n:.3f} ms per flow), alerts={alerts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "train", "batch-ml", "batch-nohi", "single-ml", "single-nohi"])
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--sample", default=None); a = ap.parse_args(); D.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare": prepare(a.n or 300000, a.sample)
    elif a.cmd == "train": train(a.sample or D / "sample.parquet")
    elif a.cmd.startswith("batch"): batch(a.cmd.split("-")[1], a.n or 3000000)
    else: single(a.cmd.split("-")[1], a.n or 30000)
