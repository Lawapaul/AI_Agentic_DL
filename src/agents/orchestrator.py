"""Top-level orchestration helpers for full and demo pipeline execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.human_review.review_logic import simulate_human_review, trigger_human_review
from src.agents.human_review.schemas import ReviewRequest
from src.agents.memory.runtime_memory import RuntimeMemoryRetriever
from src.decision.rule_based import RuleBasedPlanner
from src.fusion.graph_correlation import (
    build_attack_graph,
    build_attack_profiles,
    get_top_correlated_classes,
)
from src.fusion.risk_fusion.logistic_regression_fusion import RiskFusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.json"
DEFAULT_SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "network_events.json"
DEFAULT_DEMO_OUTPUT_PATH = PROJECT_ROOT / "data" / "sample" / "demo_pipeline_summary.json"

ATTACK_LABELS = {
    0: "Normal Traffic",
    1: "DoS Attack",
    2: "Port Scan",
    3: "Brute Force Attack",
    4: "Web Attack",
    5: "Botnet",
    6: "Infiltration",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _default_config() -> dict[str, Any]:
    return {
        "demo": {
            "sample_count": 5,
            "graph_threshold": 0.35,
            "memory_min_similarity": 0.4,
            "risk_threshold": 0.75,
            "uncertainty_threshold": 0.35,
        }
    }


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = _default_config()
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config.update(_load_json(path))
    return config


def _full_pipeline_assets(processed_path: str | None, model_path: str | None) -> tuple[str | None, str | None]:
    if processed_path and model_path:
        return processed_path, model_path

    candidates = [
        (PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "saved_models" / "ids_model.keras"),
        (PROJECT_ROOT / "preprocessed_dataset", PROJECT_ROOT / "saved_models" / "hybrid_cnn_lstm_500k.keras"),
        (PROJECT_ROOT / "preprocessed_dataset", PROJECT_ROOT / "saved_models" / "hybrid_cnn_lstm_full_2_2m.keras"),
    ]

    required_arrays = {"X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"}
    for candidate_processed, candidate_model in candidates:
        if not candidate_processed.exists() or not candidate_model.exists():
            continue
        if required_arrays.issubset({path.name for path in candidate_processed.glob("*.npy")}):
            return str(candidate_processed), str(candidate_model)

    return processed_path, model_path


def _load_demo_samples(limit: int) -> list[dict[str, Any]]:
    payload = _load_json(DEFAULT_SAMPLE_DATA_PATH)
    records = payload.get("samples", [])
    return records[:limit]


def _build_demo_graph(features: np.ndarray, predictions: np.ndarray, threshold: float) -> tuple[Any, bool]:
    """Build graph correlation output when networkx is available."""

    try:
        profiles = build_attack_profiles(
            X_eval=features,
            y_eval=predictions,
            fg_importance=lambda sample: np.abs(np.asarray(sample, dtype=np.float32)).reshape(-1),
        )
        return build_attack_graph(profiles, threshold=threshold), True
    except Exception:
        return None, False


def _top_features(vector: np.ndarray, top_k: int = 5) -> list[str]:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    top_idx = np.argsort(np.abs(flat))[-top_k:][::-1]
    return [f"feature_{int(idx)} ({float(flat[int(idx)]):.4f})" for idx in top_idx]


def _target_action(attack: str, confidence: float) -> str:
    if attack == "Normal Traffic":
        return "No Action"
    if confidence >= 0.85:
        return "Block"
    if confidence >= 0.65:
        return "Alert"
    return "Monitor"


def _fallback_reasoning(
    attack: str,
    risk_score: float,
    top_features: list[str],
    memory_similarity: float,
    graph_weight: float,
    decision: str,
) -> str:
    feature_text = ", ".join(top_features[:3]) if top_features else "the strongest telemetry features"
    risk_band = "Critical" if risk_score >= 0.85 else "High" if risk_score >= 0.7 else "Medium" if risk_score >= 0.4 else "Low"
    return (
        f"{risk_band} risk identified for {attack.lower()} based on classifier confidence, "
        f"memory similarity ({memory_similarity:.2f}), and graph correlation ({graph_weight:.2f}). "
        f"The most influential signals were {feature_text}. Recommended action: {decision}."
    )


def _fallback_graph_neighbors(
    features: np.ndarray,
    predictions: np.ndarray,
    sample_index: int,
    target_prediction: int,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Approximate correlated neighbors when graph dependencies are unavailable."""

    sims = []
    current = np.asarray(features[sample_index], dtype=np.float32)
    for idx, (candidate, candidate_label) in enumerate(zip(features, predictions)):
        if idx == sample_index or int(candidate_label) == int(target_prediction):
            continue
        denom = float(np.linalg.norm(current) * np.linalg.norm(candidate))
        similarity = 0.0 if denom == 0.0 else float(np.dot(current, candidate) / denom)
        sims.append((int(candidate_label), similarity))
    sims.sort(key=lambda item: item[1], reverse=True)
    return sims[:top_k]


def _generate_reasoning(
    feature_vector: np.ndarray,
    prediction: int,
    confidence: float,
    risk_score: float,
    top_features: list[str],
    memory_similarity: float,
    graph_weight: float,
    decision: str,
) -> tuple[str | None, str]:
    try:
        from src.reasoning import LLMPipeline

        prompt, reasoning = LLMPipeline().run(
            feature_vector.tolist(),
            int(prediction),
            float(confidence),
            risk_score=float(risk_score),
            top_features=top_features,
            memory_similarity=float(memory_similarity),
            graph_weight=float(graph_weight),
            decision=decision,
        )
        return prompt, reasoning
    except Exception:
        return None, _fallback_reasoning(
            attack=ATTACK_LABELS.get(int(prediction), "Unknown Attack"),
            risk_score=float(risk_score),
            top_features=top_features,
            memory_similarity=float(memory_similarity),
            graph_weight=float(graph_weight),
            decision=decision,
        )


def run_showcase_demo(config_path: str | Path | None = None) -> dict[str, Any]:
    """Run a lightweight end-to-end demo using bundled synthetic samples."""

    config = _load_config(config_path)
    demo_config = config["demo"]
    samples = _load_demo_samples(int(demo_config["sample_count"]))
    if not samples:
        raise FileNotFoundError(f"Missing demo samples at {DEFAULT_SAMPLE_DATA_PATH}")

    features = np.asarray([sample["feature_vector"] for sample in samples], dtype=np.float32)
    predictions = np.asarray([sample["prediction"] for sample in samples], dtype=np.int32)
    confidences = np.asarray([sample["confidence"] for sample in samples], dtype=np.float32)

    feature_scale = np.maximum(np.max(np.abs(features), axis=1), 1e-6)
    fg_strengths = np.mean(np.abs(features), axis=1) / feature_scale

    attack_graph, graph_available = _build_demo_graph(
        features,
        predictions,
        threshold=float(demo_config["graph_threshold"]),
    )
    memory = RuntimeMemoryRetriever(min_similarity=float(demo_config["memory_min_similarity"]))
    fusion = RiskFusion(random_state=42)
    planner = RuleBasedPlanner().fit(
        attacks=[ATTACK_LABELS.get(int(pred), "Unknown Attack") for pred in predictions],
        confidences=confidences,
        targets=[_target_action(ATTACK_LABELS.get(int(pred), "Unknown Attack"), float(conf)) for pred, conf in zip(predictions, confidences)],
    )

    records: list[dict[str, Any]] = []
    for idx, (feature_vector, prediction, confidence, fg_strength) in enumerate(
        zip(features, predictions, confidences, fg_strengths),
        start=1,
    ):
        memory_context = memory.retrieve(query_embedding=feature_vector, predicted_class=int(prediction))
        if graph_available:
            correlated = get_top_correlated_classes(attack_graph, int(prediction), top_k=3)
        else:
            correlated = _fallback_graph_neighbors(features, predictions, idx - 1, int(prediction), top_k=3)
        graph_weight = float(correlated[0][1]) if correlated else 0.0
        risk = fusion.compute_risk(
            float(confidence),
            float(memory_context["top_similarity"]),
            graph_weight,
            float(fg_strength),
        )

        attack_name = ATTACK_LABELS.get(int(prediction), "Unknown Attack")
        top_features = _top_features(feature_vector)
        action = planner.decide(attack_name, float(confidence), str(risk["severity"]))
        prompt, reasoning = _generate_reasoning(
            feature_vector=feature_vector,
            prediction=int(prediction),
            confidence=float(confidence),
            risk_score=float(risk["risk_score"]),
            top_features=top_features,
            memory_similarity=float(memory_context["top_similarity"]),
            graph_weight=graph_weight,
            decision=action,
        )

        decision_output = {
            "event_id": f"demo-sample-{idx}",
            "risk_score": float(risk["risk_score"]),
            "uncertainty": float(1.0 - confidence),
            "explanation": reasoning,
            "suggested_action": action,
        }
        review_decision = trigger_human_review(
            decision_output,
            risk_threshold=float(demo_config["risk_threshold"]),
            uncertainty_threshold=float(demo_config["uncertainty_threshold"]),
        )
        if review_decision.requires_review:
            review_output = simulate_human_review(
                ReviewRequest(
                    event_id=decision_output["event_id"],
                    risk_score=decision_output["risk_score"],
                    explanation=decision_output["explanation"],
                    suggested_action=decision_output["suggested_action"],
                    uncertainty=decision_output["uncertainty"],
                )
            ).model_dump()
        else:
            review_output = {
                "approved": True,
                "correct_action": action,
                "feedback": "Review skipped because the sample stayed within configured thresholds.",
                "severity_adjustment": 0.0,
            }

        records.append(
            {
                "sample_id": decision_output["event_id"],
                "classification_layer": {
                    "prediction": int(prediction),
                    "attack_name": attack_name,
                    "confidence": float(confidence),
                },
                "feature_importance": top_features,
                "graph_correlation": {
                    "neighbors": [{"class_id": int(class_id), "weight": float(weight)} for class_id, weight in correlated],
                    "max_weight": graph_weight,
                },
                "memory_match": memory_context,
                "risk_fusion": risk,
                "llm_reasoning": reasoning,
                "llm_prompt": prompt,
                "decision_planner": {
                    "suggested_action": action,
                    "human_review_required": bool(review_decision.requires_review),
                    "review_reason": review_decision.reason,
                },
                "human_review": review_output,
            }
        )
        memory.add(feature_vector, int(prediction))

    summary = {
        "status": "completed",
        "mode": "demo",
        "output_path": str(DEFAULT_DEMO_OUTPUT_PATH),
        "records": records,
        "graph": {
            "node_count": int(attack_graph.number_of_nodes()) if graph_available else int(len(np.unique(predictions))),
            "edge_count": int(attack_graph.number_of_edges()) if graph_available else 0,
            "backend": "networkx" if graph_available else "fallback-cosine",
        },
    }
    DEFAULT_DEMO_OUTPUT_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_pipeline(
    mode: str = "auto",
    processed_path: str | None = None,
    model_path: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full pipeline when assets exist, otherwise fall back to the demo."""

    normalized_mode = mode.lower().strip()
    resolved_processed_path, resolved_model_path = _full_pipeline_assets(processed_path, model_path)

    if normalized_mode in {"auto", "full"} and resolved_processed_path and resolved_model_path:
        try:
            from src.agents.integration.final_pipeline import run_full_pipeline

            return run_full_pipeline(resolved_processed_path, resolved_model_path)
        except Exception:
            if normalized_mode == "full":
                raise

    return run_showcase_demo(config_path=config_path)
