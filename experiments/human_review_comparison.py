"""Compare human-review trigger strategies."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_review.pipelines import run_human_review


DEFAULT_PROCESSED_PATH = "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed"
RNG_SEED = 42


def _load_y_test(processed_path: str) -> np.ndarray:
    return np.asarray(np.load(os.path.join(processed_path, "y_test.npy")).reshape(-1), dtype=np.int32)


def _build_samples(y_test: np.ndarray, n: int = 300) -> List[Dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(y_test), size=min(n, len(y_test)), replace=False)
    samples = []
    for i in idx:
        y = int(y_test[i])
        confidence = float(np.clip(rng.normal(0.85 if y > 0 else 0.62, 0.18), 0.0, 1.0))
        risk = float(np.clip(0.65 * confidence + 0.30 * (y > 0) + rng.normal(0, 0.1), 0.0, 1.0))
        graph_distance = float(np.clip(rng.beta(5, 2) if y > 0 else rng.beta(2, 5), 0.0, 1.0))
        samples.append({"confidence": confidence, "risk_score": risk, "graph_distance": graph_distance, "is_benign": y == 0})
    return samples


def evaluate(processed_path: str, output_dir: str) -> pd.DataFrame:
    y_test = _load_y_test(processed_path)
    samples = _build_samples(y_test)

    methods = {
        "ConfidenceTrigger": "confidence",
        "UncertaintyTrigger": "uncertainty",
        "NoveltyTrigger": "novelty",
    }

    rows = []
    for name, key in methods.items():
        outcomes = [run_human_review(s, method=key) for s in samples]
        review_flags = [bool(o["review_required"]) for o in outcomes]

        review_frequency = float(np.mean(review_flags))
        analyst_workload = int(np.sum(review_flags))

        baseline_fp = sum(1 for s in samples if s["is_benign"] and s["risk_score"] > 0.6)
        post_fp = sum(1 for s, r in zip(samples, review_flags) if s["is_benign"] and s["risk_score"] > 0.6 and not r)
        fpr_reduction = float((baseline_fp - post_fp) / baseline_fp) if baseline_fp > 0 else 0.0

        rows.append(
            {
                "Method": name,
                "ReviewFrequency": review_frequency,
                "FalsePositiveReduction": fpr_reduction,
                "AnalystWorkload": analyst_workload,
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["FalsePositiveReduction", "ReviewFrequency"], ascending=[False, True])
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "human_review_comparison.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["Method"], df["FalsePositiveReduction"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1)
    plt.title("Human Review False Positive Reduction")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "human_review_fp_reduction.png"), dpi=150)
    plt.close()

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", type=str, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    results = evaluate(args.processed_path, args.output_dir)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
