"""QORC zero-day intrusion detection on NSL-KDD via Clifford2017 boson sampling.

Reproduces the Quantum Optical Reservoir Computing (QORC) architecture from
Sakurai et al., "Quantum optical reservoir computing powered by boson sampling"
(Optica Quantum, 2025) at a scale (N=16 photons, M=256 modes) where the Hilbert
space (~1.2e19 basis states) makes exact probability computation impossible.

Note: the installed Perceval/exqalibur build hard-caps BasicState/FockState at
256 modes (a compiled limit, not configurable), so M=256 is the maximum reachable
in this environment instead of the originally requested M=300 - still comfortably
above N^2=256 per the paper's M>N^2 guidance.

Unlike the MerLin-based scripts elsewhere in this repo (which rely on SLOS/exact
backends and cannot run at this scale), this script drives raw Perceval directly:
the pre-circuit and reservoir are precomputed once as Haar-random unitaries, the
data-dependent phase encoding is folded into a single combined unitary per row
(fast numpy matmul), and the CliffordClifford2017 backend draws a fixed number of
empirical sampling shots per row instead of computing the full distribution. Since
the output basis (C(300,16) ~ 2e22 states) cannot be enumerated, observed Fock
states are mapped into a fixed-size feature vector with a signed hashing trick
(Weinberger et al.) instead of a growing dictionary.

Zero-day scenario: all U2R attack rows are withheld from training and appear only
in the test set (leakage-free construction mirroring gialai-2026/QRC/data/zeroday).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import perceval as pcvl
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NON_FEATURE_COLS = ("label", "category", "is_attack")

# Standard NSL-KDD attack-name -> 5-class family taxonomy (Tavallaee et al. 2009),
# covering both KDDTrain+ and KDDTest+ label vocabularies.
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
    p.add_argument("--output-dir", default=str(here / "outdir_zeroday_clifford"))
    p.add_argument(
        "--held-out-categories", nargs="+", default=["u2r"],
        choices=["dos", "probe", "r2l", "u2r"],
        help="Attack families withheld from training and only present in test.",
    )
    p.add_argument("--n-photons", type=int, default=16)
    p.add_argument("--n-modes", type=int, default=256, help="Max 256: hard cap of the installed exqalibur BasicState backend.")
    p.add_argument("--n-shots", type=int, default=500, help="Clifford2017 sampling shots per row.")
    p.add_argument("--hash-dim", type=int, default=8192, help="Fixed hashing-trick feature dimension.")
    p.add_argument("--max-train-rows", type=int, default=3000)
    p.add_argument("--max-test-rows", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib worker count (-1 = all cores).")
    return p.parse_args()


def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("qorc_zeroday_clifford")
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
# Phase 1 - Data loading & leakage-free zero-day split
# --------------------------------------------------------------------------- #

def stratified_subsample(
    df: pd.DataFrame, max_rows: int, protect_mask: pd.Series | None, seed: int
) -> pd.DataFrame:
    """Subsample df down to max_rows, always keeping rows flagged by protect_mask."""
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
            # Falls back to a plain random sample if a class has too few members to stratify.
            sampled_pool = pool.sample(n=n_remaining, random_state=seed)
    return pd.concat([protected, sampled_pool], axis=0)


def build_zeroday_split(
    df: pd.DataFrame,
    held_out_categories: list[str],
    seed: int,
    max_train_rows: int,
    max_test_rows: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    unmapped = sorted(set(df["label"].unique()) - set(LABEL_TO_CATEGORY))
    if unmapped:
        raise ValueError(
            f"Unmapped NSL-KDD labels found, extend LABEL_TO_CATEGORY: {unmapped}"
        )
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
# Phase 2 - Classical preprocessing & phase encoding
# --------------------------------------------------------------------------- #

def preprocess_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_modes: int, logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    scaler = MinMaxScaler()
    x_num_train = scaler.fit_transform(train_df[numeric_cols].to_numpy(dtype=np.float64))
    x_num_test = scaler.transform(test_df[numeric_cols].to_numpy(dtype=np.float64))

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_cat_train = ohe.fit_transform(train_df[CATEGORICAL_COLS])
    x_cat_test = ohe.transform(test_df[CATEGORICAL_COLS])

    x_raw_train = np.hstack([x_num_train, x_cat_train]).astype(np.float32)
    x_raw_test = np.hstack([x_num_test, x_cat_test]).astype(np.float32)
    logger.info("Raw (pre-PCA) feature dim: %d", x_raw_train.shape[1])

    # NSL-KDD's raw feature space (~122 dims) is smaller than the requested M=300
    # modes, so PCA can only ever produce min(n_samples, n_features) components.
    # The remaining phase shifters are zero-padded (identity phase) below, keeping
    # the full 300-mode circuit while only ~n_pca of the phases carry data.
    n_pca = min(x_raw_train.shape[0], x_raw_train.shape[1], n_modes)
    pca = PCA(n_components=n_pca, random_state=0)
    z_train = pca.fit_transform(x_raw_train)
    z_test = pca.transform(x_raw_test)
    logger.info(
        "PCA reduced to %d components (of requested %d modes); remaining phases zero-padded",
        n_pca, n_modes,
    )

    lo, hi = float(z_train.min()), float(z_train.max())
    eps = 1e-8

    def to_phase(z: np.ndarray) -> np.ndarray:
        return np.clip((z - lo) / (hi - lo + eps), 0.0, 1.0) * np.pi

    theta_train = np.zeros((z_train.shape[0], n_modes), dtype=np.float64)
    theta_test = np.zeros((z_test.shape[0], n_modes), dtype=np.float64)
    theta_train[:, :n_pca] = to_phase(z_train)
    theta_test[:, :n_pca] = to_phase(z_test)

    return x_raw_train, x_raw_test, theta_train, theta_test, n_pca


# --------------------------------------------------------------------------- #
# Phase 3 - Photonic reservoir via Clifford2017 sampling + hashing trick
# --------------------------------------------------------------------------- #

def hash_token(token: tuple[int, ...], hash_dim: int) -> tuple[int, int]:
    """Deterministic (stable across processes/runs) signed hashing trick."""
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    h = int.from_bytes(digest, "big")
    index = h % hash_dim
    sign = 1 if (h // hash_dim) % 2 == 0 else -1
    return index, sign


def _quantum_feature_single_row(
    theta_row: np.ndarray,
    u_pre: np.ndarray,
    u_res: np.ndarray,
    input_state_occ: tuple[int, ...],
    n_shots: int,
    hash_dim: int,
) -> tuple[np.ndarray, int, str | None]:
    try:
        # diag(exp(i*theta)) @ U_res == broadcasting each row of U_res by exp(i*theta).
        u_total = u_pre @ (np.exp(1j * theta_row)[:, None] * u_res)
        circuit = pcvl.Unitary(u_total)
        backend = pcvl.BackendFactory().get_backend("CliffordClifford2017")
        backend.set_circuit(circuit)
        backend.set_input_state(pcvl.BasicState(list(input_state_occ)))
        samples = backend.samples(n_shots)
        counts = Counter(tuple(s) for s in samples)

        feature = np.zeros(hash_dim, dtype=np.float32)
        for token, count in counts.items():
            idx, sign = hash_token(token, hash_dim)
            feature[idx] += sign * (count / n_shots)
        return feature, len(counts), None
    except Exception as exc:  # memory-safety fallback: never let one row abort the batch
        return np.zeros(hash_dim, dtype=np.float32), 0, str(exc)


def features_stats_from_array(
    features: np.ndarray, duration: float, split_name: str, n_failed: int = 0,
    mean_unique_tokens: float = float("nan"),
) -> dict:
    return {
        "split": split_name,
        "n_rows": int(features.shape[0]),
        "n_failed_rows": n_failed,
        "mean_unique_tokens": mean_unique_tokens,
        "mean_nonzero_hashed_dims": float(np.mean((features != 0).sum(axis=1))) if len(features) else 0.0,
        "duration_s": duration,
        "rows_per_second": (features.shape[0] / duration) if duration > 0 else float("nan"),
    }


def compute_quantum_features(
    theta_matrix: np.ndarray,
    u_pre: np.ndarray,
    u_res: np.ndarray,
    input_state_occ: tuple[int, ...],
    n_shots: int,
    hash_dim: int,
    n_jobs: int,
    logger: logging.Logger,
    split_name: str,
) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_quantum_feature_single_row)(
            theta_matrix[i], u_pre, u_res, input_state_occ, n_shots, hash_dim
        )
        for i in range(theta_matrix.shape[0])
    )
    duration = time.time() - t0

    features = np.stack([r[0] for r in results]).astype(np.float32)
    n_unique_list = [r[1] for r in results]
    errors = [(i, r[2]) for i, r in enumerate(results) if r[2] is not None]
    if errors:
        logger.warning(
            "[%s] %d/%d rows failed quantum feature computation (zero-vector fallback). First error: %s",
            split_name, len(errors), len(results), errors[0][1],
        )

    stats = features_stats_from_array(
        features, duration, split_name, n_failed=len(errors),
        mean_unique_tokens=float(np.mean(n_unique_list)) if n_unique_list else 0.0,
    )
    logger.info(
        "[%s] quantum features: %d rows in %.2fs (%.2f rows/s), %d failed",
        split_name, stats["n_rows"], duration, stats["rows_per_second"], stats["n_failed_rows"],
    )
    return features, stats


def get_quantum_features(
    theta_matrix: np.ndarray,
    u_pre: np.ndarray,
    u_res: np.ndarray,
    input_state_occ: tuple[int, ...],
    n_shots: int,
    hash_dim: int,
    n_jobs: int,
    logger: logging.Logger,
    split_name: str,
    cache_file: Path,
) -> tuple[np.ndarray, dict]:
    if cache_file.exists():
        logger.info("[%s] loading cached quantum features from %s", split_name, cache_file)
        features = np.load(cache_file)["features"]
        stats = features_stats_from_array(features, 0.0, f"{split_name} (cached)")
        return features, stats

    features, stats = compute_quantum_features(
        theta_matrix, u_pre, u_res, input_state_occ, n_shots, hash_dim, n_jobs, logger, split_name
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, features=features)
    return features, stats


def config_hash(args: argparse.Namespace, n_train_rows: int, n_test_rows: int) -> str:
    payload = (
        f"{args.n_photons}-{args.n_modes}-{args.n_shots}-{args.hash_dim}-{args.seed}-"
        f"{n_train_rows}-{n_test_rows}-{','.join(sorted(args.held_out_categories))}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Phase 4 - Feature fusion, training, evaluation
# --------------------------------------------------------------------------- #

def train_and_evaluate(
    y_train_features: np.ndarray,
    y_train_target: np.ndarray,
    y_test_features: np.ndarray,
    y_test_target: np.ndarray,
    test_df: pd.DataFrame,
    held_out_categories: list[str],
    logger: logging.Logger,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, dict, dict, dict, float]:
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    t0 = time.time()
    clf.fit(y_train_features, y_train_target)
    duration_train = time.time() - t0
    logger.info("Classifier trained in %.2fs", duration_train)

    y_pred = clf.predict(y_test_features)
    y_proba = clf.predict_proba(y_test_features)[:, 1]

    overall = {
        "accuracy": accuracy_score(y_test_target, y_pred),
        "precision": precision_score(y_test_target, y_pred, zero_division=0),
        "recall": recall_score(y_test_target, y_pred, zero_division=0),
        "f1": f1_score(y_test_target, y_pred, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_test_target, y_proba) if len(set(y_test_target)) > 1 else float("nan")
        ),
    }

    zeroday_mask = test_df["category"].isin(held_out_categories).to_numpy()
    normal_mask = (test_df["category"] == "normal").to_numpy()
    zeroday_metrics = {
        "n_zeroday_rows": int(zeroday_mask.sum()),
        "detection_rate": float(np.mean(y_pred[zeroday_mask] == 1)) if zeroday_mask.any() else float("nan"),
        "false_alarm_rate": float(np.mean(y_pred[normal_mask] == 1)) if normal_mask.any() else float("nan"),
    }

    report_dict = classification_report(
        y_test_target, y_pred, target_names=["normal", "attack"], output_dict=True, zero_division=0
    )
    return clf, y_pred, y_proba, overall, zeroday_metrics, report_dict, duration_train


# --------------------------------------------------------------------------- #
# Phase 5 - Output
# --------------------------------------------------------------------------- #

def save_outputs(
    run_dir: Path,
    args: argparse.Namespace,
    train_stats: dict,
    test_stats: dict,
    overall: dict,
    zeroday_metrics: dict,
    report_dict: dict,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    duration_total: float,
) -> None:
    pd.DataFrame([vars(args)]).to_csv(run_dir / "run_config.csv", index=False)
    pd.DataFrame([train_stats, test_stats]).to_csv(run_dir / "quantum_feature_stats.csv", index=False)

    report_df = pd.DataFrame(report_dict).T
    zeroday_row = pd.DataFrame(
        [{
            "precision": np.nan,
            "recall": zeroday_metrics["detection_rate"],
            "f1-score": np.nan,
            "support": zeroday_metrics["n_zeroday_rows"],
        }],
        index=["zero_day_only"],
    )
    pd.concat([report_df, zeroday_row]).to_csv(run_dir / "classification_report.csv")

    pd.DataFrame({
        "row_index": test_df.index,
        "true_label": test_df["label"].to_numpy(),
        "true_category": test_df["category"].to_numpy(),
        "is_attack": test_df["is_attack"].to_numpy(),
        "predicted_is_attack": y_pred,
        "predicted_proba_attack": y_proba,
        "correct": test_df["is_attack"].to_numpy() == y_pred,
    }).to_csv(run_dir / "test_predictions.csv", index=False)

    summary = {
        **overall,
        **zeroday_metrics,
        "n_train_rows": train_stats["n_rows"],
        "n_test_rows": test_stats["n_rows"],
        "duration_quantum_features_train_s": train_stats["duration_s"],
        "duration_quantum_features_test_s": test_stats["duration_s"],
        "duration_total_s": duration_total,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary_metrics.csv", index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    if args.n_photons > args.n_modes:
        raise ValueError("n_photons must not exceed n_modes")
    if args.n_modes > 256:
        raise ValueError(
            "n_modes > 256 is not supported: the installed Perceval/exqalibur build "
            "hard-caps BasicState/FockState at 256 modes."
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(args.output_dir)
    run_dir = base_out / f"run_{timestamp}"
    cache_dir = base_out / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir / "run.log")

    t_start = time.time()
    logger.info("Loading NSL-KDD data from %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    train_df, test_df = build_zeroday_split(
        df, args.held_out_categories, args.seed, args.max_train_rows, args.max_test_rows, logger
    )
    logger.info(
        "Train rows: %d (attacks=%d) | Test rows: %d (attacks=%d, zero-day=%d)",
        len(train_df), int(train_df["is_attack"].sum()),
        len(test_df), int(test_df["is_attack"].sum()),
        int(test_df["category"].isin(args.held_out_categories).sum()),
    )

    x_raw_train, x_raw_test, theta_train, theta_test, n_pca = preprocess_features(
        train_df, test_df, args.n_modes, logger
    )

    u_pre = np.array(pcvl.Matrix.random_unitary(args.n_modes))
    u_res = np.array(pcvl.Matrix.random_unitary(args.n_modes))
    input_state_occ = tuple([1] * args.n_photons + [0] * (args.n_modes - args.n_photons))
    logger.info(
        "Photonic reservoir: %d photons, %d modes, %d shots/row, hash_dim=%d, %d/%d phases data-driven",
        args.n_photons, args.n_modes, args.n_shots, args.hash_dim, n_pca, args.n_modes,
    )

    cfg_key = config_hash(args, len(train_df), len(test_df))
    train_u, train_stats = get_quantum_features(
        theta_train, u_pre, u_res, input_state_occ, args.n_shots, args.hash_dim, args.n_jobs,
        logger, "train", cache_dir / f"qfeat_train_{cfg_key}.npz",
    )
    test_u, test_stats = get_quantum_features(
        theta_test, u_pre, u_res, input_state_occ, args.n_shots, args.hash_dim, args.n_jobs,
        logger, "test", cache_dir / f"qfeat_test_{cfg_key}.npz",
    )

    y_train_features = np.hstack([x_raw_train, train_u]).astype(np.float32)
    y_test_features = np.hstack([x_raw_test, test_u]).astype(np.float32)

    clf, y_pred, y_proba, overall, zeroday_metrics, report_dict, duration_train = train_and_evaluate(
        y_train_features, train_df["is_attack"].to_numpy(),
        y_test_features, test_df["is_attack"].to_numpy(),
        test_df, args.held_out_categories, logger,
    )
    logger.info("Overall test metrics: %s", overall)
    logger.info("Zero-day metrics: %s", zeroday_metrics)

    duration_total = time.time() - t_start
    save_outputs(
        run_dir, args, train_stats, test_stats, overall, zeroday_metrics, report_dict,
        test_df, y_pred, y_proba, duration_total,
    )
    logger.info("Run complete in %.2fs. Outputs written to %s", duration_total, run_dir)


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        logging.getLogger("qorc_zeroday_clifford").critical(
            "Out of memory - reduce --max-train-rows/--max-test-rows, --n-modes, or --hash-dim."
        )
        raise
    except Exception:
        logging.getLogger("qorc_zeroday_clifford").exception("Fatal error during QORC zero-day run.")
        raise
