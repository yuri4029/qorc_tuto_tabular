"""QORC zero-day intrusion detection on NSL-KDD via GPU-resident exact boson sampling.

Unlike compare/qorc_nslkdd_zeroday_clifford.py (which uses Perceval's CPU-only
CliffordClifford2017 *sampling* backend + a hashing trick because its circuit,
N=16 photons/M=256 modes, is too large for exact simulation), this script keeps
MerLin's *exact* permanent-based QuantumLayer, but runs it with device=cuda so the
exact probability computation itself executes on GPU. This lets circuit size scale
far beyond any exact-simulation script in this repo (previous max: 25 modes/4
photons = 12,650 features here run/tested at 40 modes/6 photons = 3,838,380 features)
while still using exact probabilities, not shot noise.

GPU-quantum feasibility was validated by qorc_gpu_phase0_smoketest.py: MerLin's
QuantumLayer genuinely executes on CUDA (not a silent CPU fallback) and matches the
CPU build to ~1e-8. qorc_gpu_memory_benchmark.py measured ~150 bytes of GPU memory
per (row, output-state) for this measurement strategy, used below to size batches.

Because the exact output dimension (C(n_modes, n_photons)) is enormous, a dense
(n_rows x n_features) quantum feature matrix cannot be materialized on host RAM
(e.g. 10k rows x 3.8M features x 4 bytes = 152 GB). So the two detector arms that
consume quantum features (SUP_q, NOV_q) are reimplemented as GPU-streaming
algorithms (mini-batch logistic regression; randomized block power-iteration SVD)
that never leave GPU tensors and never materialize the full quantum feature matrix.
The other arms (SUP_raw, NOV_raw, NOV_dials, IForest) operate on small (<=122-dim)
classical features and stay on sklearn/CPU, unchanged in spirit from
QRC/team/multiple_config_graph.ipynb's ablation design.

Zero-day scenario: family-level holdout (--held-out-categories), leakage-free split
ported from qorc_nslkdd_zeroday_clifford.py's build_zeroday_split.

Multi-GPU: every (held-out-category, seed) combination is one task; tasks are
distributed round-robin across the GPUs named in --gpu-ids (default: all visible
GPUs), one worker process per GPU, each additionally batching rows internally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import perceval as pcvl
import torch
import torch.nn as nn
import merlin as ML
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    MeasurementStrategy = ML.MeasurementStrategy
    ComputationSpace = ML.ComputationSpace
except AttributeError:
    from merlin.measurement.strategies import MeasurementStrategy, ComputationSpace

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NON_FEATURE_COLS = ("label", "category", "is_attack")
DETECTORS = ["SUP_raw", "SUP_q", "NOV_raw", "NOV_q", "NOV_dials", "IForest"]

# Standard NSL-KDD attack-name -> 5-class family taxonomy (Tavallaee et al. 2009).
LABEL_TO_CATEGORY = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "mailbomb": "dos", "worm": "dos",
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r", "perl": "u2r",
    "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default=str(here / "NSL_KDD_labeled.csv"))
    p.add_argument("--output-dir", default=str(here / "outdir_zeroday_gpu"))
    p.add_argument(
        "--held-out-categories", nargs="+", default=["u2r"],
        choices=["dos", "probe", "r2l", "u2r"],
        help="Attack families withheld from training, one scenario (task) per value.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.add_argument("--n-photons", type=int, default=6)
    p.add_argument("--n-modes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64,
                    help="Row batch size for GPU quantum feature streaming; "
                         "see qorc_gpu_memory_benchmark.py to recalibrate per GPU.")
    p.add_argument("--gpu-ids", nargs="+", type=int, default=None,
                    help="Defaults to all CUDA devices visible to this process.")
    p.add_argument("--n-nov-comps", type=int, default=32, help="Rank k for novelty detectors' PCA/SVD.")
    p.add_argument("--sup-q-epochs", type=int, default=3)
    p.add_argument("--sup-q-lr", type=float, default=1e-2)
    p.add_argument("--fa-budgets", nargs="+", type=float, default=[0.01, 0.05])
    p.add_argument("--max-train-rows", type=int, default=20000)
    p.add_argument("--max-test-rows", type=int, default=10000)
    return p.parse_args()


def configure_logging(log_file: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


# --------------------------------------------------------------------------- #
# Phase 1 - Data loading & leakage-free zero-day split (ported from
# qorc_nslkdd_zeroday_clifford.py's build_zeroday_split/stratified_subsample)
# --------------------------------------------------------------------------- #

def stratified_subsample(
    df: pd.DataFrame, max_rows: int, protect_mask: pd.Series | None, seed: int
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    if protect_mask is None:
        protect_mask = pd.Series(False, index=df.index)
    protected = df[protect_mask]
    pool = df[~protect_mask]
    n_remaining = max(max_rows - len(protected), 0)
    if n_remaining <= 0 or pool.empty:
        return protected
    if len(pool) <= n_remaining:
        sampled_pool = pool
    else:
        try:
            sampled_pool, _ = train_test_split(
                pool, train_size=n_remaining, stratify=pool["is_attack"], random_state=seed
            )
        except ValueError:
            sampled_pool = pool.sample(n=n_remaining, random_state=seed)
    return pd.concat([protected, sampled_pool], axis=0)


def build_zeroday_split(
    df: pd.DataFrame, held_out_categories: list[str], seed: int,
    max_train_rows: int, max_test_rows: int, logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    unmapped = sorted(set(df["label"].unique()) - set(LABEL_TO_CATEGORY))
    if unmapped:
        raise ValueError(f"Unmapped NSL-KDD labels found, extend LABEL_TO_CATEGORY: {unmapped}")
    df["category"] = df["label"].map(LABEL_TO_CATEGORY)
    df["is_attack"] = (df["label"] != "normal").astype(int)

    normal_df = df[df["category"] == "normal"]
    normal_train, normal_test = train_test_split(normal_df, test_size=0.25, random_state=seed)

    held_out_mask = df["category"].isin(held_out_categories)
    held_out_df = df[held_out_mask]
    seen_attack_df = df[(df["is_attack"] == 1) & ~held_out_mask]

    train_df = pd.concat([normal_train, seen_attack_df], axis=0)
    test_df = pd.concat([normal_test, held_out_df], axis=0)

    leaked = set(train_df.loc[train_df["category"].isin(held_out_categories), "label"])
    if leaked:
        raise AssertionError(f"Zero-day leakage detected: held-out labels found in train: {leaked}")
    logger.info("Zero-day split OK: 0 held-out labels leaked into train.")

    train_df = stratified_subsample(train_df, max_train_rows, protect_mask=None, seed=seed)
    test_protect_mask = test_df["category"].isin(held_out_categories)
    test_df = stratified_subsample(test_df, max_test_rows, protect_mask=test_protect_mask, seed=seed)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Phase 1b - Classical preprocessing: raw (scaled), PCA "dials", and phase encoding
# --------------------------------------------------------------------------- #

def to_phase(z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Smooth tanh squash, no clipping, no wrap-around -> (0, pi)."""
    return ((np.pi / 2) * (1 + np.tanh((z - mu) / (3 * sd)))).astype(np.float32)


def preprocess(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_modes: int, seed: int,
) -> dict:
    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_cat_train = ohe.fit_transform(train_df[CATEGORICAL_COLS])
    x_cat_test = ohe.transform(test_df[CATEGORICAL_COLS])
    x_num_train = train_df[numeric_cols].to_numpy(dtype=np.float32)
    x_num_test = test_df[numeric_cols].to_numpy(dtype=np.float32)
    x_raw_train_pre = np.hstack([x_num_train, x_cat_train]).astype(np.float32)
    x_raw_test_pre = np.hstack([x_num_test, x_cat_test]).astype(np.float32)

    scaler = StandardScaler().fit(x_raw_train_pre)
    x_raw_train = scaler.transform(x_raw_train_pre).astype(np.float32)
    x_raw_test = scaler.transform(x_raw_test_pre).astype(np.float32)

    pca = PCA(n_components=n_modes, random_state=seed).fit(x_raw_train)
    z_train = pca.transform(x_raw_train).astype(np.float32)
    z_test = pca.transform(x_raw_test).astype(np.float32)

    mu, sd = z_train.mean(0), z_train.std(0) + 1e-8
    phases_train = to_phase(z_train, mu, sd)
    phases_test = to_phase(z_test, mu, sd)

    return dict(
        raw_train=x_raw_train, raw_test=x_raw_test,
        dials_train=z_train, dials_test=z_test,
        phases_train=phases_train, phases_test=phases_test,
    )


# --------------------------------------------------------------------------- #
# Phase 2 - GPU-resident exact quantum layer (validated by qorc_gpu_phase0_smoketest.py)
# --------------------------------------------------------------------------- #

def build_quantum_layer(n_modes: int, n_photons: int, seed: int, device: torch.device):
    np.random.seed(seed)  # perceval's random_unitary uses the global numpy RNG
    unitary = pcvl.Matrix.random_unitary(n_modes)
    precircuit = pcvl.Unitary(unitary)
    reservoir = precircuit.copy()
    c_var = pcvl.Circuit(n_modes)
    for i in range(n_modes):
        c_var.add(i, pcvl.PS(pcvl.P(f"px{i + 1}")))
    circuit = precircuit // c_var // reservoir

    step = (n_modes - 1) / (n_photons - 1)
    input_state = [0] * n_modes
    for k in range(n_photons):
        input_state[int(round(k * step))] = 1

    ql = ML.QuantumLayer(
        input_size=n_modes,
        circuit=circuit,
        trainable_parameters=[],
        input_parameters=["px"],
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED),
        device=device,
    )
    ql.eval()
    return ql


def iter_quantum_batches(ql, phases: np.ndarray, batch_size: int, device: torch.device):
    """Yields (start, end, quantum_features) with quantum_features a live GPU tensor.
    Never materializes the full (n_rows x n_features) matrix on host or device."""
    n = phases.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(phases[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            z = ql(x)
        yield start, end, z


def quantum_output_dim(n_modes: int, n_photons: int) -> int:
    return math.comb(n_modes, n_photons)


# --------------------------------------------------------------------------- #
# Phase 4a - Small-dimension detectors (raw 122-dim / dials n_modes-dim): CPU/sklearn
# (ported from QRC/team/multiple_config_graph.ipynb)
# --------------------------------------------------------------------------- #

class NoveltyDetector:
    """Learns the shape of normal traffic; scores each row by how far it falls from
    it (Mahalanobis + reconstruction error). Never sees an attack during fit()."""

    def __init__(self, k: int = 32):
        self.k = k

    def fit(self, xn: np.ndarray) -> "NoveltyDetector":
        self.mu = xn.mean(0)
        z = xn - self.mu
        k = min(self.k, z.shape[1], z.shape[0] - 1)
        self.pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(z)
        p = self.pca.transform(z)
        self.sd = p.std(0) + 1e-8
        r = z - self.pca.inverse_transform(p)
        rnorm = np.sqrt((r ** 2).sum(1))
        self.rmu, self.rsd = rnorm.mean(), rnorm.std() + 1e-8
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = x - self.mu
        p = self.pca.transform(z)
        maha = np.sqrt(((p / self.sd) ** 2).sum(1))
        r = z - self.pca.inverse_transform(p)
        recon = (np.sqrt((r ** 2).sum(1)) - self.rmu) / self.rsd
        return maha + np.maximum(recon, 0)


def auc(s_normal: np.ndarray, s_attack: np.ndarray) -> float:
    y = np.r_[np.zeros(len(s_normal)), np.ones(len(s_attack))]
    return roc_auc_score(y, np.r_[s_normal, s_attack])


def detection_at_fa(s_normal: np.ndarray, s_attack: np.ndarray, fa: float) -> float:
    thr = np.quantile(s_normal, 1 - fa)
    return float((s_attack > thr).mean())


# --------------------------------------------------------------------------- #
# Phase 4b - GPU-streaming detectors for the quantum arm (D = C(n_modes,n_photons),
# too large to ever materialize as a dense (n_rows x D) array on host or device)
# --------------------------------------------------------------------------- #

class GPUStreamingLogisticRegression:
    """Logistic regression over quantum features, trained by mini-batch SGD directly
    on the QuantumLayer's streamed GPU output. The expensive quantum forward pass
    runs under torch.no_grad(); only the small nn.Linear(D,1) head is trained."""

    def __init__(self, output_dim: int, device: torch.device, lr: float = 1e-2, epochs: int = 3):
        self.device = device
        self.epochs = epochs
        self.model = nn.Linear(output_dim, 1).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, ql, phases: np.ndarray, y: np.ndarray, batch_size: int) -> "GPUStreamingLogisticRegression":
        n = phases.shape[0]
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        rng = np.random.default_rng(0)
        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                x = torch.tensor(phases[idx], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    z = ql(x)
                logits = self.model(z).squeeze(1)
                loss = self.loss_fn(logits, y_t[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, ql, phases: np.ndarray, batch_size: int) -> np.ndarray:
        out = np.empty(phases.shape[0], dtype=np.float32)
        for start, end, z in iter_quantum_batches(ql, phases, batch_size, self.device):
            with torch.no_grad():
                logits = self.model(z).squeeze(1)
                out[start:end] = torch.sigmoid(logits).cpu().numpy()
        return out


class GPUStreamingNoveltyDetector:
    """Novelty detector for the quantum arm: same Mahalanobis + reconstruction-error
    idea as NoveltyDetector, but the (D x k) principal subspace is estimated via a
    streaming randomized range-finder (Halko et al. style) so the full (n_rows x D)
    quantum feature matrix is never materialized. Reconstruction residual uses
    ||r||^2 = ||z||^2 - ||proj||^2 (Pythagorean identity for an orthonormal basis),
    so the D-dim reconstruction itself never needs to be formed either.
    """

    def __init__(self, output_dim: int, device: torch.device, k: int = 32):
        self.device = device
        self.k = k
        self.d = output_dim

    def fit(self, ql, phases: np.ndarray, batch_size: int) -> "GPUStreamingNoveltyDetector":
        n = phases.shape[0]
        device = self.device
        k = min(self.k, n - 1)

        # Pass 1: streaming mean.
        mu_sum = torch.zeros(self.d, device=device)
        for start, end, z in iter_quantum_batches(ql, phases, batch_size, device):
            mu_sum += z.sum(0)
        self.mu = mu_sum / n

        # Pass 2: Y = A_centered @ Omega  (N x k, small)
        omega = torch.randn(self.d, k, device=device)
        y_mat = torch.zeros(n, k, device=device)
        for start, end, z in iter_quantum_batches(ql, phases, batch_size, device):
            zc = z - self.mu
            y_mat[start:end] = zc @ omega
        q_mat, _ = torch.linalg.qr(y_mat)

        # Pass 3: B = A_centered^T @ Q  (D x k); orthonormalize -> range approx of A^T.
        b_mat = torch.zeros(self.d, k, device=device)
        for start, end, z in iter_quantum_batches(ql, phases, batch_size, device):
            zc = z - self.mu
            b_mat += zc.T @ q_mat[start:end]
        self.basis, _ = torch.linalg.qr(b_mat)  # (D, k) orthonormal columns

        # Pass 4: projection/residual stats on the (normal) training data itself.
        proj_all = torch.zeros(n, k, device=device)
        normsq_all = torch.zeros(n, device=device)
        for start, end, z in iter_quantum_batches(ql, phases, batch_size, device):
            zc = z - self.mu
            proj_all[start:end] = zc @ self.basis
            normsq_all[start:end] = (zc ** 2).sum(1)
        self.sd = proj_all.std(0) + 1e-8
        resid_sq = torch.clamp(normsq_all - (proj_all ** 2).sum(1), min=0)
        resid = torch.sqrt(resid_sq)
        self.rmu, self.rsd = resid.mean(), resid.std() + 1e-8
        return self

    def score(self, ql, phases: np.ndarray, batch_size: int) -> np.ndarray:
        out = np.empty(phases.shape[0], dtype=np.float32)
        for start, end, z in iter_quantum_batches(ql, phases, batch_size, self.device):
            zc = z - self.mu
            proj = zc @ self.basis
            maha = torch.sqrt(((proj / self.sd) ** 2).sum(1))
            normsq = (zc ** 2).sum(1)
            resid_sq = torch.clamp(normsq - (proj ** 2).sum(1), min=0)
            recon = (torch.sqrt(resid_sq) - self.rmu) / self.rsd
            score = maha + torch.clamp(recon, min=0)
            out[start:end] = score.cpu().numpy()
        return out


# --------------------------------------------------------------------------- #
# Phase 3/4 combined - one (held-out category, seed) task, run on one GPU
# --------------------------------------------------------------------------- #

@dataclass
class TaskResult:
    category: str
    seed: int
    gpu_id: int
    metrics: dict
    duration_s: float


def run_scenario_seed(
    input_csv: str, category: str, seed: int, gpu_id: int,
    n_modes: int, n_photons: int, batch_size: int, n_nov_comps: int,
    sup_q_epochs: int, sup_q_lr: float, fa_budgets: list[float],
    max_train_rows: int, max_test_rows: int, run_dir: Path,
) -> TaskResult:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    task_dir = run_dir / f"{category}_seed{seed}"
    task_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(task_dir / "run.log", name=f"gpu{gpu_id}_{category}_{seed}")
    t0 = time.time()

    df = pd.read_csv(input_csv)
    train_df, test_df = build_zeroday_split(
        df, [category], seed, max_train_rows, max_test_rows, logger
    )
    feats = preprocess(train_df, test_df, n_modes, seed)

    y_train = train_df["is_attack"].to_numpy()
    normal_mask = (test_df["category"] == "normal").to_numpy()
    zeroday_mask = test_df["category"].isin([category]).to_numpy()
    no = (y_train == 0)

    out_dim = quantum_output_dim(n_modes, n_photons)
    logger.info(
        "[gpu%d] scenario=%s seed=%d train_rows=%d test_rows=%d out_dim=%d",
        gpu_id, category, seed, len(train_df), len(test_df), out_dim,
    )
    ql = build_quantum_layer(n_modes, n_photons, seed, device)

    metrics: dict[str, float] = {}
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # SUP_raw / SUP_q -- supervised (trained on normal + seen attacks)
    clf_raw = LogisticRegression(max_iter=2000).fit(feats["raw_train"], y_train)
    s_n = clf_raw.predict_proba(feats["raw_test"][normal_mask])[:, 1]
    s_a = clf_raw.predict_proba(feats["raw_test"][zeroday_mask])[:, 1]
    metrics["SUP_raw"] = auc(s_n, s_a); scores["SUP_raw"] = (s_n, s_a)

    sup_q = GPUStreamingLogisticRegression(out_dim, device, lr=sup_q_lr, epochs=sup_q_epochs)
    sup_q.fit(ql, feats["phases_train"], y_train.astype(np.float32), batch_size)
    s_n = sup_q.predict_proba(ql, feats["phases_test"][normal_mask], batch_size)
    s_a = sup_q.predict_proba(ql, feats["phases_test"][zeroday_mask], batch_size)
    metrics["SUP_q"] = auc(s_n, s_a); scores["SUP_q"] = (s_n, s_a)

    # NOV_raw / NOV_dials -- novelty detectors on small classical features (CPU)
    nov_raw = NoveltyDetector(k=n_nov_comps).fit(feats["raw_train"][no])
    s_n = nov_raw.score(feats["raw_test"][normal_mask])
    s_a = nov_raw.score(feats["raw_test"][zeroday_mask])
    metrics["NOV_raw"] = auc(s_n, s_a); scores["NOV_raw"] = (s_n, s_a)

    nov_dials = NoveltyDetector(k=n_nov_comps).fit(feats["dials_train"][no])
    s_n = nov_dials.score(feats["dials_test"][normal_mask])
    s_a = nov_dials.score(feats["dials_test"][zeroday_mask])
    metrics["NOV_dials"] = auc(s_n, s_a); scores["NOV_dials"] = (s_n, s_a)

    # NOV_q -- novelty detector on quantum features (GPU streaming)
    nov_q = GPUStreamingNoveltyDetector(out_dim, device, k=n_nov_comps)
    nov_q.fit(ql, feats["phases_train"][no], batch_size)
    s_n = nov_q.score(ql, feats["phases_test"][normal_mask], batch_size)
    s_a = nov_q.score(ql, feats["phases_test"][zeroday_mask], batch_size)
    metrics["NOV_q"] = auc(s_n, s_a); scores["NOV_q"] = (s_n, s_a)

    # IForest -- classical baseline (CPU)
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(feats["raw_train"][no])
    s_n = -iso.score_samples(feats["raw_test"][normal_mask])
    s_a = -iso.score_samples(feats["raw_test"][zeroday_mask])
    metrics["IForest"] = auc(s_n, s_a); scores["IForest"] = (s_n, s_a)

    for fa in fa_budgets:
        for tag in DETECTORS:
            s_n, s_a = scores[tag]
            metrics[f"{tag}_det@fa{fa}"] = detection_at_fa(s_n, s_a, fa)

    duration = time.time() - t0
    logger.info("[gpu%d] scenario=%s seed=%d done in %.1fs metrics=%s", gpu_id, category, seed, duration, metrics)

    row = {"category": category, "seed": seed, "gpu_id": gpu_id, "duration_s": duration,
           "n_train_rows": len(train_df), "n_test_rows": len(test_df),
           "n_zeroday_rows": int(zeroday_mask.sum()), "out_dim": out_dim, **metrics}
    pd.DataFrame([row]).to_csv(task_dir / "summary_metrics.csv", index=False)

    return TaskResult(category=category, seed=seed, gpu_id=gpu_id, metrics=metrics, duration_s=duration)


# --------------------------------------------------------------------------- #
# Main - job scheduling across GPUs
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. Run compare/qorc_gpu_phase0_smoketest.py first "
            "to validate GPU quantum-layer support before using this script."
        )
    gpu_ids = args.gpu_ids or list(range(torch.cuda.device_count()))

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir / "main.log", name="main")
    logger.info("Using GPUs: %s", gpu_ids)
    pd.DataFrame([vars(args)]).to_csv(run_dir / "run_config.csv", index=False)

    tasks = list(itertools.product(args.held_out_categories, args.seeds))
    logger.info("Scheduling %d tasks across %d GPU(s): %s", len(tasks), len(gpu_ids), tasks)

    results: list[TaskResult] = []
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp.get_context("spawn")) as pool:
        futures = {}
        for i, (category, seed) in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            fut = pool.submit(
                run_scenario_seed, args.input_csv, category, seed, gpu_id,
                args.n_modes, args.n_photons, args.batch_size, args.n_nov_comps,
                args.sup_q_epochs, args.sup_q_lr, args.fa_budgets,
                args.max_train_rows, args.max_test_rows, run_dir,
            )
            futures[fut] = (category, seed, gpu_id)

        for fut in as_completed(futures):
            category, seed, gpu_id = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception("Task failed: category=%s seed=%d gpu=%d", category, seed, gpu_id)

    rows = [
        {"category": r.category, "seed": r.seed, "gpu_id": r.gpu_id,
         "duration_s": r.duration_s, **r.metrics}
        for r in results
    ]
    agg = pd.DataFrame(rows)
    agg.to_csv(run_dir / "aggregate_summary.csv", index=False)
    logger.info("All tasks complete. Aggregate summary:\n%s", agg.to_string())


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        logging.getLogger("main").critical(
            "Out of memory - reduce --batch-size, --n-modes, --n-photons, or --max-*-rows."
        )
        raise
    except Exception:
        logging.getLogger("main").exception("Fatal error during GPU QORC zero-day run.")
        raise
