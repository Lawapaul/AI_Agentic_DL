"""Run Adaptive Risk Fusion method comparison on processed IDS data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from risk_fusion import FusionFactory
from risk_fusion.utils import severity_to_index


DEFAULT_PROCESSED_PATH = "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed"
RNG_SEED = 42


def _flatten_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3 and X.shape[-1] == 1:
        return X[..., 0]
    if X.ndim == 2:
        return X
    raise ValueError(f"Unsupported feature shape: {X.shape}")


def _minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    vmin = float(v.min())
    vmax = float(v.max())
    if np.isclose(vmax, vmin):
        return np.full_like(v, 0.5, dtype=np.float32)
    return (v - vmin) / (vmax - vmin)


def load_processed_dataset(processed_path: str) -> Dict[str, np.ndarray]:
    """Load dataset from processed folder (expects .npy splits)."""
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed path not found: {processed_path}")

    files = {p.name for p in path.iterdir() if p.is_file()}
    expected = {"X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"}

    if not expected.issubset(files):
        missing = sorted(expected - files)
        raise FileNotFoundError(
            f"Missing required files in {processed_path}: {missing}. "
            f"Found: {sorted(files)}"
        )

    X_train = np.load(path / "X_train.npy")
    X_test = np.load(path / "X_test.npy")
    y_train = np.load(path / "y_train.npy").reshape(-1)
    y_test = np.load(path / "y_test.npy").reshape(-1)

    return {
        "X_train": _flatten_features(X_train),
        "X_test": _flatten_features(X_test),
        "y_train": np.asarray(y_train, dtype=np.int32),
        "y_test": np.asarray(y_test, dtype=np.int32),
    }


def simulate_pipeline_signals(X: np.ndarray, y: np.ndarray, seed: int = RNG_SEED) -> np.ndarray:
    """Create deterministic proxy signals [C, M, G, F] in [0, 1]."""
    rng = np.random.default_rng(seed)

    X = _flatten_features(X)
    y = np.asarray(y, dtype=np.int32)

    abs_mean = _minmax(np.mean(np.abs(X), axis=1))
    std = _minmax(np.std(X, axis=1))

    y_scale = y.astype(np.float32)
    y_norm = _minmax(y_scale)
    attack_flag = (y > 0).astype(np.float32)

    eps1 = rng.normal(0.0, 0.04, size=X.shape[0]).astype(np.float32)
    eps2 = rng.normal(0.0, 0.05, size=X.shape[0]).astype(np.float32)
    eps3 = rng.normal(0.0, 0.05, size=X.shape[0]).astype(np.float32)
    eps4 = rng.normal(0.0, 0.04, size=X.shape[0]).astype(np.float32)

    confidence = np.clip(0.50 * abs_mean + 0.35 * attack_flag + 0.15 * y_norm + eps1, 0.0, 1.0)
    memory_sim = np.clip(0.45 * confidence + 0.30 * y_norm + 0.25 * abs_mean + eps2, 0.0, 1.0)
    graph_sim = np.clip(0.40 * confidence + 0.35 * std + 0.25 * y_norm + eps3, 0.0, 1.0)
    fg_strength = np.clip(0.55 * abs_mean + 0.30 * std + 0.15 * confidence + eps4, 0.0, 1.0)

    return np.column_stack([confidence, memory_sim, graph_sim, fg_strength]).astype(np.float32)


def build_severity_targets(signals: np.ndarray, y: np.ndarray, seed: int = RNG_SEED) -> np.ndarray:
    """Generate reproducible severity targets from signals and labels."""
    rng = np.random.default_rng(seed + 1)
    y = np.asarray(y, dtype=np.int32)

    c, m, g, f = signals[:, 0], signals[:, 1], signals[:, 2], signals[:, 3]
    attack_bias = (y > 0).astype(np.float32)
    noise = rng.normal(0.0, 0.03, size=signals.shape[0]).astype(np.float32)

    risk = np.clip(0.42 * c + 0.24 * m + 0.20 * g + 0.14 * f + 0.08 * attack_bias + noise, 0.0, 1.0)

    # 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL
    sev = np.zeros_like(y, dtype=np.int32)
    sev[risk > 0.4] = 1
    sev[risk > 0.6] = 2
    sev[risk > 0.8] = 3
    return sev


def evaluate_method(name: str, fusion, X_test: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    preds = []
    for c, m, g, f in X_test:
        out = fusion.compute_risk(float(c), float(m), float(g), float(f))
        preds.append(severity_to_index(out["severity"]))

    y_pred = np.asarray(preds, dtype=np.int32)
    return {
        "Method": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_experiment(processed_path: str, output_dir: str) -> Tuple[pd.DataFrame, str]:
    np.random.seed(RNG_SEED)

    data = load_processed_dataset(processed_path)
    X_train_signals = simulate_pipeline_signals(data["X_train"], data["y_train"], seed=RNG_SEED)
    X_test_signals = simulate_pipeline_signals(data["X_test"], data["y_test"], seed=RNG_SEED + 10)

    y_train_sev = build_severity_targets(X_train_signals, data["y_train"], seed=RNG_SEED)
    y_test_sev = build_severity_targets(X_test_signals, data["y_test"], seed=RNG_SEED + 10)

    methods = [
        ("Weighted", "weighted"),
        ("Dynamic", "dynamic_weighted"),
        ("LogisticRegression", "logistic_regression"),
        ("RandomForest", "random_forest"),
        ("Fuzzy", "fuzzy_logic"),
        ("Attention", "attention"),
    ]

    rows = []
    for display_name, key in methods:
        fusion = FusionFactory.create(key)
        if hasattr(fusion, "fit"):
            fusion.fit(X_train_signals, y_train_sev)
        rows.append(evaluate_method(display_name, fusion, X_test_signals, y_test_sev))

    results = pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, "risk_fusion_comparison_results.csv")
    results.to_csv(table_path, index=False)

    fig_path = os.path.join(output_dir, "risk_fusion_f1_comparison.png")
    plt.figure(figsize=(10, 5))
    plt.bar(
        results["Method"],
        results["F1"],
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    )
    plt.ylim(0, 1)
    plt.ylabel("F1 Score (Macro)")
    plt.title("Adaptive Risk Fusion Method Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    return results, fig_path


def main():
    parser = argparse.ArgumentParser(description="Adaptive Risk Fusion experiment runner")
    parser.add_argument("--processed_path", type=str, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    results, fig_path = run_experiment(args.processed_path, args.output_dir)

    print("\nMethod Comparison")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved comparison CSV: {os.path.join(args.output_dir, 'risk_fusion_comparison_results.csv')}")
    print(f"Saved F1 chart: {fig_path}")


if __name__ == "__main__":
    main()
