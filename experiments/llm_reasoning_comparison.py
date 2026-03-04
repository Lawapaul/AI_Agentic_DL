"""Compare LLM reasoning methods for explainability quality and latency."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_reasoning.pipelines import run_llm_reasoning


DEFAULT_PROCESSED_PATH = "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed"
RNG_SEED = 42


def _load_labels(processed_path: str) -> np.ndarray:
    y_test = np.load(os.path.join(processed_path, "y_test.npy")).reshape(-1)
    return np.asarray(y_test, dtype=np.int32)


def _build_samples(y_test: np.ndarray, n_samples: int = 200) -> List[Dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    attack_names = {0: "BENIGN", 1: "DOS", 2: "PORTSCAN", 3: "BRUTEFORCE", 4: "DDOS"}
    feature_pool = ["flow_duration", "packet_length", "syn_flag_count", "dst_port", "fwd_iat_mean", "payload_entropy"]
    graph_pool = ["DOS", "DDOS", "PORTSCAN", "INFILTRATION", "BRUTEFORCE"]

    idx = rng.choice(len(y_test), size=min(n_samples, len(y_test)), replace=False)
    samples = []
    for i in idx:
        label = int(y_test[i])
        attack_type = attack_names.get(label, f"ATTACK_{label}")
        confidence = float(np.clip(rng.normal(0.78 if label > 0 else 0.55, 0.12), 0.0, 1.0))
        risk = float(np.clip(0.55 * confidence + 0.35 * (label > 0) + rng.normal(0, 0.08), 0.0, 1.0))
        top_features = rng.choice(feature_pool, size=4, replace=False).tolist()
        neighbors = rng.choice(graph_pool, size=3, replace=False).tolist()
        samples.append(
            {
                "attack_type": attack_type,
                "confidence": confidence,
                "risk_score": risk,
                "top_features": top_features,
                "memory_context": {"summary": f"top historical match for {attack_type}"},
                "graph_neighbors": neighbors,
            }
        )
    return samples


def _faithfulness(outputs: List[Dict[str, str]], samples: List[Dict[str, object]]) -> float:
    scores = []
    for out, sample in zip(outputs, samples):
        text = (out.get("reasoning", "") + " " + out.get("attack_pattern", "")).lower()
        feats = [f.lower() for f in sample["top_features"]]
        hit = sum(1 for feat in feats if feat in text)
        scores.append(hit / max(len(feats), 1))
    return float(np.mean(scores))


def _consistency(method: str, samples: List[Dict[str, object]]) -> float:
    deterministic_hits = []
    for sample in samples[:50]:
        a = run_llm_reasoning(sample, method=method)
        b = run_llm_reasoning(sample, method=method)
        deterministic_hits.append(1.0 if a == b else 0.0)
    return float(np.mean(deterministic_hits))


def evaluate(processed_path: str, output_dir: str) -> pd.DataFrame:
    y_test = _load_labels(processed_path)
    samples = _build_samples(y_test)

    methods = {
        "RuleBased": "rule_based",
        "CoT": "cot",
        "RAG": "rag",
        "FlanT5": "flan_t5",
    }

    rows = []
    for name, key in methods.items():
        start = time.perf_counter()
        outputs = [run_llm_reasoning(s, method=key) for s in samples]
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "Method": name,
                "ExplanationConsistency": _consistency(key, samples),
                "FaithfulnessToFeatures": _faithfulness(outputs, samples),
                "RuntimeSec": elapsed,
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["FaithfulnessToFeatures", "ExplanationConsistency"], ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "llm_reasoning_comparison.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["Method"], df["FaithfulnessToFeatures"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.ylim(0, 1)
    plt.title("LLM Reasoning Faithfulness Comparison")
    plt.ylabel("Faithfulness")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "llm_reasoning_faithfulness.png"), dpi=150)
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
