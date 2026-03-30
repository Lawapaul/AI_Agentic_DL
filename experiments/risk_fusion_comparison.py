"""Colab-ready adaptive risk fusion comparison."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from risk_fusion import FusionFactory
from risk_fusion.utils import severity_to_idx


DEFAULT_PATH = "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed"
RNG_SEED = 42


def load_data(processed_path: str):
    X_train = np.load(os.path.join(processed_path, "X_train.npy"))
    X_test = np.load(os.path.join(processed_path, "X_test.npy"))
    y_train = np.load(os.path.join(processed_path, "y_train.npy")).reshape(-1)
    y_test = np.load(os.path.join(processed_path, "y_test.npy")).reshape(-1)
    if X_train.ndim == 3 and X_train.shape[-1] == 1:
        X_train = X_train[..., 0]
    if X_test.ndim == 3 and X_test.shape[-1] == 1:
        X_test = X_test[..., 0]
    return X_train, X_test, y_train.astype(np.int32), y_test.astype(np.int32)


def _minmax(v):
    v = np.asarray(v, dtype=np.float32)
    a, b = float(v.min()), float(v.max())
    if np.isclose(a, b):
        return np.full_like(v, 0.5)
    return (v - a) / (b - a)


def simulate_signals(X, y, seed=42):
    rng = np.random.default_rng(seed)
    abs_mean = _minmax(np.mean(np.abs(X), axis=1))
    std = _minmax(np.std(X, axis=1))
    y_norm = _minmax(y.astype(np.float32))
    attack = (y > 0).astype(np.float32)

    c = np.clip(0.5 * abs_mean + 0.35 * attack + 0.15 * y_norm + rng.normal(0, 0.04, len(y)), 0, 1)
    m = np.clip(0.45 * c + 0.3 * y_norm + 0.25 * abs_mean + rng.normal(0, 0.05, len(y)), 0, 1)
    g = np.clip(0.4 * c + 0.35 * std + 0.25 * y_norm + rng.normal(0, 0.05, len(y)), 0, 1)
    f = np.clip(0.55 * abs_mean + 0.3 * std + 0.15 * c + rng.normal(0, 0.04, len(y)), 0, 1)
    return np.column_stack([c, m, g, f]).astype(np.float32)


def build_targets(signals, y, seed=42):
    rng = np.random.default_rng(seed + 1)
    c, m, g, f = signals[:, 0], signals[:, 1], signals[:, 2], signals[:, 3]
    risk = np.clip(0.42 * c + 0.24 * m + 0.2 * g + 0.14 * f + 0.08 * (y > 0) + rng.normal(0, 0.03, len(y)), 0, 1)
    sev = np.zeros(len(y), dtype=np.int32)
    sev[risk > 0.4] = 1
    sev[risk > 0.6] = 2
    sev[risk > 0.8] = 3
    return sev


def evaluate(processed_path: str, output_dir: str):
    X_train, X_test, y_train, y_test = load_data(processed_path)
    Xtr = simulate_signals(X_train, y_train, RNG_SEED)
    Xte = simulate_signals(X_test, y_test, RNG_SEED + 10)
    ytr = build_targets(Xtr, y_train, RNG_SEED)
    yte = build_targets(Xte, y_test, RNG_SEED + 10)

    methods = [
        ("Weighted", "weighted"),
        ("Dynamic", "dynamic_weighted"),
        ("LogisticRegression", "logistic_regression"),
        ("RandomForest", "random_forest"),
        ("Fuzzy", "fuzzy_logic"),
        ("Attention", "attention"),
    ]

    rows = []
    for name, key in methods:
        model = FusionFactory.create(key)
        if hasattr(model, "fit"):
            model.fit(Xtr, ytr)
        preds = []
        for c, m, g, f in Xte:
            out = model.compute_risk(float(c), float(m), float(g), float(f))
            preds.append(severity_to_idx(out["severity"]))
        preds = np.asarray(preds, dtype=np.int32)
        rows.append(
            {
                "Method": name,
                "Accuracy": accuracy_score(yte, preds),
                "Precision": precision_score(yte, preds, average="macro", zero_division=0),
                "Recall": recall_score(yte, preds, average="macro", zero_division=0),
                "F1": f1_score(yte, preds, average="macro", zero_division=0),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "risk_fusion_comparison.csv"), index=False)

    plt.figure(figsize=(9, 4))
    plt.bar(df["Method"], df["F1"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    plt.ylim(0, 1)
    plt.title("Risk Fusion F1 Comparison")
    plt.ylabel("F1")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_fusion_f1.png"), dpi=160)
    plt.close()
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_path", type=str, default=DEFAULT_PATH)
    p.add_argument("--output_dir", type=str, default="experiments/results")
    args = p.parse_args()
    res = evaluate(args.processed_path, args.output_dir)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
