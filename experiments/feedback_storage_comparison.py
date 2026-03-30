"""Compare feedback storage backends for latency and storage efficiency."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feedback_storage.json_storage import JSONFeedbackStorage
from feedback_storage.sqlite_storage import SQLiteFeedbackStorage
from feedback_storage.vector_memory_storage import VectorMemoryFeedbackStorage


DEFAULT_PROCESSED_PATH = "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed"
RNG_SEED = 42


def _load_features(processed_path: str) -> np.ndarray:
    x = np.load(os.path.join(processed_path, "X_test.npy"))
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return np.asarray(x, dtype=np.float32)


def _build_records(X: np.ndarray, n: int = 400) -> List[Dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
    records = []
    for i in idx:
        feat = X[i][:20].astype(float).tolist()
        score = float(np.clip(np.mean(np.abs(feat)) + rng.normal(0, 0.05), 0, 1))
        records.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": feat,
                "predicted_attack": "DOS" if score > 0.6 else "BENIGN",
                "risk_score": score,
                "decision": "BLOCK_IP" if score > 0.8 else "ALERT_ADMIN" if score > 0.4 else "ALLOW",
                "human_feedback": "confirmed" if score > 0.6 else "benign",
                "final_label": "ATTACK" if score > 0.6 else "BENIGN",
            }
        )
    return records


def _size_bytes(paths: List[str]) -> int:
    total = 0
    for p in paths:
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total


def evaluate(processed_path: str, output_dir: str) -> pd.DataFrame:
    X = _load_features(processed_path)
    records = _build_records(X)

    os.makedirs(output_dir, exist_ok=True)
    bench_root = os.path.join(output_dir, "feedback_bench")
    if os.path.exists(bench_root):
        shutil.rmtree(bench_root)
    os.makedirs(bench_root, exist_ok=True)

    rows = []

    # JSON backend
    json_path = os.path.join(bench_root, "feedback.jsonl")
    json_store = JSONFeedbackStorage(json_path)
    t0 = time.perf_counter()
    for r in records:
        json_store.save(r)
    write_sec = time.perf_counter() - t0

    q0 = time.perf_counter()
    for _ in range(100):
        _ = json_store.query_recent(limit=50)
    query_sec = (time.perf_counter() - q0) / 100

    size_b = _size_bytes([json_path])
    rows.append(
        {
            "Method": "JSON",
            "RetrievalSpeedRecPerSec": len(records) / max(write_sec, 1e-6),
            "StorageBytesPerRecord": size_b / max(len(records), 1),
            "QueryLatencySec": query_sec,
        }
    )

    # SQLite backend
    db_path = os.path.join(bench_root, "feedback.db")
    sql_store = SQLiteFeedbackStorage(db_path)
    t0 = time.perf_counter()
    for r in records:
        sql_store.save(r)
    write_sec = time.perf_counter() - t0

    q0 = time.perf_counter()
    for _ in range(100):
        _ = sql_store.query_recent(limit=50)
    query_sec = (time.perf_counter() - q0) / 100

    size_b = _size_bytes([db_path])
    rows.append(
        {
            "Method": "SQLite",
            "RetrievalSpeedRecPerSec": len(records) / max(write_sec, 1e-6),
            "StorageBytesPerRecord": size_b / max(len(records), 1),
            "QueryLatencySec": query_sec,
        }
    )

    # Vector backend
    vec_prefix = os.path.join(bench_root, "feedback_vec")
    vec_store = VectorMemoryFeedbackStorage(vec_prefix)
    t0 = time.perf_counter()
    for r in records:
        vec_store.save(r)
    vec_store.flush()
    write_sec = time.perf_counter() - t0

    q0 = time.perf_counter()
    for _ in range(100):
        _ = vec_store.query_recent(limit=50)
    query_sec = (time.perf_counter() - q0) / 100

    size_b = _size_bytes([f"{vec_prefix}_vectors.npy", f"{vec_prefix}_meta.jsonl"])
    rows.append(
        {
            "Method": "VectorMemory",
            "RetrievalSpeedRecPerSec": len(records) / max(write_sec, 1e-6),
            "StorageBytesPerRecord": size_b / max(len(records), 1),
            "QueryLatencySec": query_sec,
        }
    )

    df = pd.DataFrame(rows).sort_values(by=["QueryLatencySec", "StorageBytesPerRecord"], ascending=[True, True])
    df.to_csv(os.path.join(output_dir, "feedback_storage_comparison.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["Method"], df["QueryLatencySec"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("Feedback Storage Query Latency")
    plt.ylabel("Latency (sec)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feedback_storage_latency.png"), dpi=150)
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
