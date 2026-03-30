"""Compare decision planner strategies for IDS response quality and latency."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision_planner.pipelines import run_decision_planner


DEFAULT_PROCESSED_PATH = "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed"
RNG_SEED = 42
ACTIONS = ("BLOCK_IP", "RATE_LIMIT", "ALERT_ADMIN", "ALLOW")


def _load_y_test(processed_path: str) -> np.ndarray:
    return np.asarray(np.load(os.path.join(processed_path, "y_test.npy")).reshape(-1), dtype=np.int32)


def _target_action(risk: float, attack: str) -> str:
    if risk > 0.85:
        return "BLOCK_IP"
    if risk > 0.65:
        return "RATE_LIMIT"
    if risk > 0.45 and attack != "BENIGN":
        return "ALERT_ADMIN"
    return "ALLOW"


def _build_samples(y_test: np.ndarray, n: int = 200) -> List[Dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    names = {0: "BENIGN", 1: "DOS", 2: "PORTSCAN", 3: "BRUTEFORCE", 4: "DDOS"}
    idx = rng.choice(len(y_test), size=min(n, len(y_test)), replace=False)
    samples = []
    for i in idx:
        y = int(y_test[i])
        attack = names.get(y, f"ATTACK_{y}")
        risk = float(np.clip(rng.normal(0.25 if y == 0 else 0.72, 0.15), 0.0, 1.0))
        samples.append(
            {
                "risk_score": risk,
                "attack_type": attack,
                "llm_explanation": {"summary": f"explain {attack}"},
                "target_action": _target_action(risk, attack),
                "is_benign": attack == "BENIGN",
            }
        )
    return samples


def _accuracy(pred: List[str], target: List[str]) -> float:
    return float(np.mean([p == t for p, t in zip(pred, target)]))


def _false_mitigation_rate(pred: List[str], benign_mask: List[bool]) -> float:
    benign_preds = [p for p, m in zip(pred, benign_mask) if m]
    if not benign_preds:
        return 0.0
    false_mit = sum(1 for p in benign_preds if p != "ALLOW")
    return float(false_mit / len(benign_preds))


def evaluate(processed_path: str, output_dir: str) -> pd.DataFrame:
    y_test = _load_y_test(processed_path)
    samples = _build_samples(y_test)

    methods = {
        "RiskThreshold": "risk_threshold",
        "RulePolicy": "rule_policy",
        "Probabilistic": "probabilistic",
        "RLAgent": "rl_agent",
    }

    rows = []
    for name, key in methods.items():
        preds = []
        t0 = time.perf_counter()
        for s in samples:
            out = run_decision_planner(s, method=key)
            preds.append(str(out["action"]))
        latency = (time.perf_counter() - t0) / max(len(samples), 1)

        rows.append(
            {
                "Method": name,
                "DecisionAccuracy": _accuracy(preds, [s["target_action"] for s in samples]),
                "FalseMitigationRate": _false_mitigation_rate(preds, [s["is_benign"] for s in samples]),
                "DecisionLatencySec": latency,
            }
        )

    df = pd.DataFrame(rows).sort_values(by=["DecisionAccuracy", "FalseMitigationRate"], ascending=[False, True])
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "decision_planner_comparison.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["Method"], df["DecisionAccuracy"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.ylim(0, 1)
    plt.title("Decision Planner Accuracy")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_planner_accuracy.png"), dpi=150)
    plt.close()

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", type=str, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    results = evaluate(args.processed_path, args.output_dir)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
