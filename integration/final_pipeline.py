"""Full system orchestration for the existing IDS modules."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from decision_planner import (
    ConfidenceBasedPlanner,
    HybridPlanner,
    LLMPlanner,
    PolicyBasedPlanner,
    RLPlanner as DecisionRLPlanner,
    RiskBasedPlanner,
    RuleBasedPlanner,
)
from feedback_store.feedback_db import fetch_feedback, init_db
from feedback_store.feedback_logger import compute_reward, log_feedback
from graph_correlation import build_attack_graph, build_attack_profiles, get_top_correlated_classes
from graph_correlation import fallback_correlate_with_history
from human_review.review_logic import simulate_human_review, trigger_human_review
from human_review.schemas import ReviewRequest
from memory import RuntimeMemoryRetriever
from retraining import RLTrainer, SupervisedDecisionTrainer, load_feedback_dataset
from risk_fusion.logistic_regression_fusion import RiskFusion
from risk_fusion.utils import severity_from_risk

try:
    from experiments.phase4_memory_pipeline import CombinedMemory
except Exception:  # pragma: no cover
    CombinedMemory = None


LOGGER = logging.getLogger(__name__)
RNG = np.random.default_rng(42)
DEFAULT_LLM_MODEL = "google/flan-t5-small"
DEFAULT_HIGH_RISK_PERCENTILE = 0.15


def _setup_logging(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "final_pipeline.log"

    if not LOGGER.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)

    return log_path


def _safe_call(step: str, fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - error path is runtime-dependent
        LOGGER.exception("%s failed: %s", step, exc)
        return default


def _init_optional_llm(component_name: str, factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on transformers runtime
        LOGGER.exception("%s failed: %s", component_name, exc)
        return None


def _build_llm_pipeline(model_name: str):
    from llm_reasoning.llm_pipeline import LLMPipeline

    return LLMPipeline(model_name)


def _get_model_trainer_class():
    from models.trainer import IDSModelTrainer

    return IDSModelTrainer


def _get_feature_gradient_factory():
    from explainability.feature_gradient_explainer import create_feature_gradient_explainer

    return create_feature_gradient_explainer


def _load_model_with_fallback(model_path: str, x_train: np.ndarray, y_train: np.ndarray):
    IDSModelTrainer = _get_model_trainer_class()
    try:
        return IDSModelTrainer.load_model(model_path)
    except Exception as exc:
        LOGGER.warning("Falling back to rebuilt hybrid architecture for model load: %s", exc)
        from models.model_factory import get_model

        input_shape = (int(x_train.shape[1]), 1)
        num_classes = int(len(np.unique(y_train)))
        model = get_model("hybrid", input_shape=input_shape, num_classes=num_classes)
        model.load_weights(model_path)
        return model


def _available_ram_gb() -> float:
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return float(parts[1]) / (1024.0 * 1024.0)

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return float(page_size * pages) / (1024.0 ** 3)
        except (ValueError, OSError):
            pass

    return 8.0


def _choose_runtime_limits() -> dict[str, Any]:
    IDSModelTrainer = _get_model_trainer_class()
    ram_gb = _available_ram_gb()
    device = IDSModelTrainer.detect_device()
    if device == "CPU":
        if ram_gb < 8:
            max_samples = 24
        elif ram_gb < 16:
            max_samples = 50
        else:
            max_samples = 75
    else:
        if ram_gb < 8:
            max_samples = 2000
        elif ram_gb < 16:
            max_samples = 4000
        elif ram_gb < 24:
            max_samples = 6000
        elif ram_gb < 40:
            max_samples = 8000
        else:
            max_samples = 10000

    batch_size = 256 if device == "GPU" else 64
    return {
        "available_ram_gb": round(ram_gb, 2),
        "max_samples": int(max_samples),
        "batch_size": int(batch_size),
        "device": device,
    }


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")
    return arr


def _ensure_3d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], arr.shape[1], 1)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")


def _read_array(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _load_processed_subset(processed_path: str, max_samples: int) -> dict[str, np.ndarray]:
    base = Path(processed_path)
    required = {
        "X_train": base / "X_train.npy",
        "X_test": base / "X_test.npy",
        "y_train": base / "y_train.npy",
        "y_test": base / "y_test.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files: {missing}")

    x_train = _ensure_2d(_read_array(required["X_train"]))
    x_test = _ensure_2d(_read_array(required["X_test"]))
    y_train = np.asarray(_read_array(required["y_train"]), dtype=np.int32).reshape(-1)
    y_test = np.asarray(_read_array(required["y_test"]), dtype=np.int32).reshape(-1)

    train_limit = min(len(x_train), max_samples)
    test_limit = min(len(x_test), max_samples)

    train_idx = np.arange(train_limit)
    test_idx = np.arange(test_limit)

    return {
        "X_train": np.asarray(x_train[train_idx], dtype=np.float32),
        "X_test": np.asarray(x_test[test_idx], dtype=np.float32),
        "y_train": np.asarray(y_train[train_idx], dtype=np.int32),
        "y_test": np.asarray(y_test[test_idx], dtype=np.int32),
    }


def _batched_model_predict(model, x_2d: np.ndarray, batch_size: int) -> np.ndarray:
    x_3d = _ensure_3d(x_2d)
    outputs = []
    for start in range(0, len(x_3d), batch_size):
        stop = min(start + batch_size, len(x_3d))
        outputs.append(model.predict(x_3d[start:stop], batch_size=batch_size, verbose=0))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def _batched_embeddings(trainer: IDSModelTrainer, x_2d: np.ndarray, batch_size: int) -> np.ndarray:
    embeddings = []
    for start in range(0, len(x_2d), batch_size):
        stop = min(start + batch_size, len(x_2d))
        embeddings.append(trainer.extract_embeddings(x_2d[start:stop], batch_size=batch_size))
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _feature_names(num_features: int) -> list[str]:
    return [f"feature_{idx}" for idx in range(num_features)]


def _label_names(labels: np.ndarray) -> list[str]:
    return [str(label) for label in sorted(np.unique(labels).tolist())]


def _risk_rank(predictions: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    is_attack = (np.asarray(predictions) != 0).astype(np.float32)
    return confidences + (0.25 * is_attack)


def _select_high_risk_indices(predictions: np.ndarray, confidences: np.ndarray, percentile: float) -> np.ndarray:
    total = len(predictions)
    if total == 0:
        return np.asarray([], dtype=np.int32)
    top_k = max(1, int(math.ceil(total * percentile)))
    ranking = _risk_rank(predictions, confidences)
    return np.argsort(ranking)[-top_k:][::-1].astype(np.int32)


def _compute_fg_vectors(explainer, samples_2d: np.ndarray) -> np.ndarray:
    vectors = []
    for sample in samples_2d:
        vectors.append(np.asarray(explainer.feature_importance(sample), dtype=np.float32).reshape(-1))
    return np.asarray(vectors, dtype=np.float32)


def _fg_strengths(fg_vectors: np.ndarray) -> np.ndarray:
    if len(fg_vectors) == 0:
        return np.asarray([], dtype=np.float32)
    raw = np.mean(np.abs(fg_vectors), axis=1)
    max_value = float(np.max(raw)) if len(raw) else 0.0
    if max_value <= 0.0:
        return np.zeros_like(raw, dtype=np.float32)
    return (raw / max_value).astype(np.float32)


def _build_lime_explainer(x_train_2d: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("LIME is not installed. Add `lime` to the Colab environment.") from exc

    return LimeTabularExplainer(
        training_data=np.asarray(x_train_2d, dtype=np.float32),
        feature_names=feature_names,
        class_names=_label_names(y_train),
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )


def _run_lime_for_indices(
    lime_explainer,
    model,
    x_test_2d: np.ndarray,
    target_indices: np.ndarray,
    batch_size: int,
    top_k: int = 8,
) -> dict[int, list[str]]:
    outputs: dict[int, list[str]] = {}

    def predict_fn(batch_2d):
        batch_3d = _ensure_3d(np.asarray(batch_2d, dtype=np.float32))
        return model.predict(batch_3d, batch_size=batch_size, verbose=0)

    for idx in target_indices.tolist():
        explanation = lime_explainer.explain_instance(
            data_row=np.asarray(x_test_2d[idx], dtype=np.float32),
            predict_fn=predict_fn,
            num_features=top_k,
        )
        outputs[int(idx)] = [f"{feature} ({weight:.4f})" for feature, weight in explanation.as_list()]
    return outputs


def _fallback_top_features(sample: np.ndarray, feature_names: list[str], top_k: int = 5) -> list[str]:
    sample_arr = np.asarray(sample, dtype=np.float32).reshape(-1)
    top_idx = np.argsort(np.abs(sample_arr))[-top_k:][::-1]
    return [f"{feature_names[int(idx)]} ({float(sample_arr[int(idx)]):.4f})" for idx in top_idx]


def _graph_window_summary(
    graph,
    predicted_class: int,
    sample_features: np.ndarray,
    history_features: np.ndarray,
    ensure_neighbor: bool = False,
) -> dict[str, Any]:
    neighbors = get_top_correlated_classes(graph, int(predicted_class), top_k=3)
    if not neighbors:
        history_neighbors = fallback_correlate_with_history(
            current_features=sample_features,
            history_features=history_features,
            top_k=3,
            threshold=0.7,
            ensure_one=ensure_neighbor,
        )
        neighbors = [(idx, weight) for idx, weight in history_neighbors]
        return {
            "top_neighbors": [{"sample_id": f"sample_{int(idx)}", "weight": float(weight)} for idx, weight in neighbors],
            "max_weight": float(neighbors[0][1]) if neighbors else 0.0,
        }
    return {
        "top_neighbors": [{"class_id": int(cls), "weight": float(weight)} for cls, weight in neighbors],
        "max_weight": float(neighbors[0][1]) if neighbors else 0.0,
    }


def _build_graph_summaries(
    samples_2d: np.ndarray,
    predicted_labels: np.ndarray,
    explainer,
    window_size: int = 64,
    threshold: float = 0.7,
    ensure_neighbor: bool = False,
) -> dict[int, dict[str, Any]]:
    outputs: dict[int, dict[str, Any]] = {}
    samples_3d = _ensure_3d(samples_2d)

    for start in range(0, len(samples_2d), window_size):
        stop = min(start + window_size, len(samples_2d))
        window_x = samples_3d[start:stop]
        window_y = predicted_labels[start:stop]
        if len(np.unique(window_y)) == 0:
            continue

        profiles = build_attack_profiles(window_x, window_y, explainer.feature_importance)
        graph = build_attack_graph(profiles, threshold=threshold)

        for offset, predicted_class in enumerate(window_y.tolist()):
            current_idx = start + offset
            history_start = max(0, current_idx - 5)
            history_features = samples_2d[history_start:current_idx]
            outputs[current_idx] = _graph_window_summary(
                graph,
                int(predicted_class),
                samples_2d[current_idx],
                history_features,
                ensure_neighbor=ensure_neighbor,
            )

    return outputs


def _make_attack_name(prediction: int) -> str:
    return "Normal Traffic" if int(prediction) == 0 else "Attack"


def _dynamic_decision(severity: str, confidence: float, uncertainty: float) -> str:
    severity_upper = str(severity).upper()
    if severity_upper in {"CRITICAL", "HIGH"}:
        return "Block"
    if severity_upper == "MEDIUM":
        return "Monitor" if uncertainty > 0.2 or confidence < 0.8 else "Alert"
    return "No Action"


def _fallback_reasoning(
    top_features: list[str],
    risk_score: float,
    graph_output: dict[str, Any],
    memory_output: dict[str, Any],
    decision: str,
) -> str:
    feature_text = ", ".join(top_features[:3]) if top_features else "top feature values unavailable"
    graph_text = graph_output.get("top_neighbors", [])
    memory_text = memory_output.get("labels", [])
    return (
        f"Risk {risk_score:.2f}; key signals: {feature_text}; "
        f"graph={graph_text}; memory={memory_text}; action={decision}."
    )


def _confidence_action(attack: str, confidence: float) -> str:
    planner = ConfidenceBasedPlanner()
    return planner.decide(attack, float(confidence), "unused")


def _build_review_target(risk_score: float, uncertainty: float, attack: str, confidence: float) -> dict[str, Any]:
    suggested_action = _confidence_action(attack, confidence)
    request = ReviewRequest(
        event_id="planner_target",
        risk_score=float(risk_score),
        explanation="Planner comparison target generation",
        suggested_action=str(suggested_action),
        uncertainty=float(uncertainty),
    )
    review = simulate_human_review(request).model_dump()
    return {
        "target_action": str(review["correct_action"]),
        "severity_adjustment": float(review["severity_adjustment"]),
        "review": review,
    }


def _consistency_score(actions_once: list[str], actions_twice: list[str]) -> float:
    if not actions_once:
        return 0.0
    matches = [a == b for a, b in zip(actions_once, actions_twice)]
    return float(np.mean(matches))


def _safe_planner_fit(name: str, planner, attacks, confidences, severities, targets):
    if name == "rule_based":
        return planner.fit(attacks, confidences, targets)
    if name == "confidence_based":
        return planner.fit(confidences, targets)
    if name == "risk_based":
        return planner.fit(confidences, severities, targets)
    if name == "hybrid":
        return planner.fit(attacks, confidences, severities, targets)
    if name == "rl":
        return planner.fit(attacks, confidences, targets, epochs=10)
    if name == "llm":
        return planner.fit(attacks, confidences, severities, targets)
    if name == "policy":
        return planner.fit(attacks, severities, targets)
    raise ValueError(f"Unsupported planner: {name}")


def _compare_decision_planners(samples: list[dict[str, Any]]) -> dict[str, Any]:
    attacks = [sample["attack"] for sample in samples]
    confidences = [float(sample["confidence"]) for sample in samples]
    severities = [str(sample["severity"]) for sample in samples]
    targets = [str(sample["target_action"]) for sample in samples]
    severity_adjustments = [float(sample["severity_adjustment"]) for sample in samples]

    planners = {
        "rule_based": RuleBasedPlanner(),
        "confidence_based": ConfidenceBasedPlanner(),
        "risk_based": RiskBasedPlanner(),
        "hybrid": HybridPlanner(),
        "rl": DecisionRLPlanner(),
        "policy": PolicyBasedPlanner(),
    }

    llm_planner = _init_optional_llm("decision_planner.llm.init", LLMPlanner)
    if llm_planner is not None:
        planners["llm"] = llm_planner

    metrics: dict[str, dict[str, Any]] = {}
    best_name = None
    best_score = (-1.0, -1.0, -1.0)

    for name, planner in planners.items():
        fitted = _safe_call(f"decision_planner.{name}.fit", _safe_planner_fit, name, planner, attacks, confidences, severities, targets, default=None)
        if fitted is None:
            continue

        actions_once = [
            str(planner.decide(attack, confidence, severity))
            for attack, confidence, severity in zip(attacks, confidences, severities)
        ]
        actions_twice = [
            str(planner.decide(attack, confidence, severity))
            for attack, confidence, severity in zip(attacks, confidences, severities)
        ]

        accuracy = float(np.mean([pred == target for pred, target in zip(actions_once, targets)]))
        consistency = _consistency_score(actions_once, actions_twice)
        reward = float(
            np.mean(
                [
                    compute_reward(pred, target, adj)
                    for pred, target, adj in zip(actions_once, targets, severity_adjustments)
                ]
            )
        )
        majority_action = Counter(actions_once).most_common(1)[0][0] if actions_once else "Monitor"
        metrics[name] = {
            "accuracy": accuracy,
            "consistency": consistency,
            "reward": reward,
            "majority_action": majority_action,
        }
        score = (accuracy, consistency, reward)
        if score > best_score:
            best_score = score
            best_name = name

    return {
        "best_planner": best_name,
        "metrics": metrics,
    }


def _parse_feedback_state(state_json: str) -> dict[str, float]:
    state = json.loads(state_json)
    return {
        "risk_score": float(state.get("risk_score", 0.0)),
        "uncertainty": float(state.get("uncertainty", 0.0)),
    }


def _compare_retraining_methods(db_path: Path) -> dict[str, Any]:
    features, labels = load_feedback_dataset(str(db_path))
    records = fetch_feedback(str(db_path))
    if len(records) == 0:
        return {"best_method": None, "metrics": {}}

    label_to_action = {
        0: "No Action",
        1: "Monitor",
        2: "Alert",
        3: "Block",
    }
    targets = [str(record["human_action"]) for record in records]

    metrics: dict[str, dict[str, Any]] = {}

    rl = RLTrainer()
    for record in reversed(records):
        state = _parse_feedback_state(record["state"])
        rl.update_policy(
            risk_score=state["risk_score"],
            action=str(record["human_action"]),
            reward=float(record["reward"]),
            context_embedding=[state["uncertainty"]],
        )

    rl_actions_once = []
    rl_actions_twice = []
    for record in records:
        state = _parse_feedback_state(record["state"])
        rl_actions_once.append(rl.select_action(state["risk_score"], [state["uncertainty"]], epsilon=0.0))
        rl_actions_twice.append(rl.select_action(state["risk_score"], [state["uncertainty"]], epsilon=0.0))

    rl_accuracy = float(np.mean([pred == target for pred, target in zip(rl_actions_once, targets)]))
    rl_consistency = _consistency_score(rl_actions_once, rl_actions_twice)
    rl_reward = float(np.mean([compute_reward(pred, target, 0.0) for pred, target in zip(rl_actions_once, targets)]))
    metrics["rl_trainer"] = {
        "accuracy": rl_accuracy,
        "consistency": rl_consistency,
        "reward": rl_reward,
    }

    best_method = "rl_trainer"
    best_score = (rl_accuracy, rl_consistency, rl_reward)

    supervised = SupervisedDecisionTrainer()
    summary = _safe_call("retraining.supervised.train", supervised.train_model, features, labels, default=None)
    if summary is not None and len(features) > 0:
        sup_actions_once = []
        sup_actions_twice = []
        for idx in range(len(features)):
            prediction = supervised.predict_action(features[idx])
            sup_actions_once.append(prediction)
            sup_actions_twice.append(supervised.predict_action(features[idx]))

        sup_accuracy = float(np.mean([pred == target for pred, target in zip(sup_actions_once, targets)]))
        sup_consistency = _consistency_score(sup_actions_once, sup_actions_twice)
        sup_reward = float(np.mean([compute_reward(pred, target, 0.0) for pred, target in zip(sup_actions_once, targets)]))
        metrics["supervised_trainer"] = {
            "accuracy": sup_accuracy,
            "consistency": sup_consistency,
            "reward": sup_reward,
            "training_summary": summary.__dict__,
        }
        score = (sup_accuracy, sup_consistency, sup_reward)
        if score > best_score:
            best_score = score
            best_method = "supervised_trainer"

    return {
        "best_method": best_method,
        "metrics": metrics,
    }


def _default_project_root(processed_path: Path) -> Path:
    if processed_path.name == "processed" and processed_path.parent.name == "data":
        return processed_path.parent.parent
    return Path.cwd()


def _resolve_results_dir(processed_path: str) -> Path:
    project_root = _default_project_root(Path(processed_path).resolve())
    return project_root / "experiments" / "results" / "final_pipeline"


def _resolve_default_paths() -> tuple[str, str]:
    candidates = [
        (
            "/content/drive/MyDrive/Deep Learning Project/AI Agentic/data/processed",
            "/content/drive/MyDrive/Deep Learning Project/AI Agentic/saved_models/hybrid_cnn_lstm_full_2_2m.keras",
        ),
        (
            "/content/AI_Agentic_DL/data/processed",
            "/content/AI_Agentic_DL/saved_models/ids_model.keras",
        ),
        (
            str(Path.cwd() / "data" / "processed"),
            str(Path.cwd() / "saved_models" / "ids_model.keras"),
        ),
    ]

    for processed_path, model_path in candidates:
        if Path(processed_path).exists() and Path(model_path).exists():
            return processed_path, model_path

    raise FileNotFoundError("Could not auto-resolve processed/model paths for demo pipeline.")


def _core_pipeline(processed_path: str, model_path: str, sample_override: int | None = None, demo_mode: bool = False) -> dict[str, Any]:
    results_dir = _resolve_results_dir(processed_path)
    log_path = _setup_logging(results_dir)
    runtime = _choose_runtime_limits()
    if sample_override is not None:
        runtime["max_samples"] = int(sample_override)
    runtime["demo_mode"] = bool(demo_mode)
    runtime["platform"] = platform.platform()

    LOGGER.info("Starting final pipeline with runtime=%s", runtime)

    db_path = results_dir / "feedback.db"
    init_db(str(db_path))

    data = _safe_call("data.load", _load_processed_subset, processed_path, runtime["max_samples"], default=None)
    if data is None:
        return {"status": "failed", "runtime": runtime, "errors": ["data.load failed"], "log_path": str(log_path)}

    IDSModelTrainer = _get_model_trainer_class()
    trainer = IDSModelTrainer(model_type="hybrid", model_save_path=model_path)
    trainer.model = _safe_call("model.load", _load_model_with_fallback, model_path, data["X_train"], data["y_train"], default=None)
    if trainer.model is None:
        return {"status": "failed", "runtime": runtime, "errors": ["model.load failed"], "log_path": str(log_path)}

    x_train = data["X_train"]
    x_test = data["X_test"]

    probabilities = _safe_call("classifier.predict", _batched_model_predict, trainer.model, x_test, runtime["batch_size"], default=np.empty((0, 0), dtype=np.float32))
    if probabilities.size == 0:
        return {"status": "failed", "runtime": runtime, "errors": ["classifier.predict failed"], "log_path": str(log_path)}

    predicted_labels = np.argmax(probabilities, axis=1).astype(np.int32)
    confidences = np.max(probabilities, axis=1).astype(np.float32)
    high_risk_indices = np.arange(min(len(x_test), runtime["max_samples"]), dtype=np.int32)

    feature_names = _feature_names(x_train.shape[1])
    create_feature_gradient_explainer = _get_feature_gradient_factory()
    fg_explainer = _safe_call("feature_gradient.init", create_feature_gradient_explainer, trainer.model, feature_names, default=None)
    if fg_explainer is None:
        return {"status": "failed", "runtime": runtime, "errors": ["feature_gradient.init failed"], "log_path": str(log_path)}

    filtered_x = x_test[high_risk_indices]
    filtered_predictions = predicted_labels[high_risk_indices]
    filtered_confidences = confidences[high_risk_indices]

    lime_limit = min(len(high_risk_indices), len(high_risk_indices) if demo_mode else 8)
    lime_indices = high_risk_indices[:lime_limit]
    lime_results = {}
    lime_explainer = _safe_call(
        "lime.init",
        _build_lime_explainer,
        x_train[: min(len(x_train), 128)],
        data["y_train"][: min(len(data["y_train"]), 128)],
        feature_names,
        default=None,
    )
    if lime_explainer is not None and len(lime_indices) > 0:
        lime_results = _safe_call(
            "lime.run",
            _run_lime_for_indices,
            lime_explainer,
            trainer.model,
            x_test,
            lime_indices,
            runtime["batch_size"],
            default={},
        ) or {}

    filtered_fg = _safe_call("feature_gradient.filtered", _compute_fg_vectors, fg_explainer, filtered_x, default=np.empty((0, x_train.shape[1]), dtype=np.float32))
    graph_results = _safe_call(
        "graph.build",
        _build_graph_summaries,
        filtered_x,
        filtered_predictions,
        fg_explainer,
        ensure_neighbor=True,
        default={},
    ) or {}

    memory_results: dict[int, dict[str, Any]] = {}
    runtime_memory = RuntimeMemoryRetriever(top_k=1, min_similarity=0.5)
    bank_size = min(len(x_train), 64 if not demo_mode else 32)
    if CombinedMemory is not None and bank_size > 0 and len(filtered_x) > 0:
        train_embeddings = _safe_call("memory.train_embeddings", _batched_embeddings, trainer, x_train[:bank_size], runtime["batch_size"], default=None)
        train_fg = _safe_call("memory.train_fg", _compute_fg_vectors, fg_explainer, x_train[:bank_size], default=None)
        query_embeddings = _safe_call("memory.query_embeddings", _batched_embeddings, trainer, filtered_x, runtime["batch_size"], default=None)
        if train_embeddings is not None and train_fg is not None and query_embeddings is not None:
            memory_module = CombinedMemory(top_k=5, embedding_weight=0.5)
            fitted_memory = _safe_call(
                "memory.fit",
                memory_module.fit,
                embeddings=train_embeddings,
                fg_vectors=train_fg,
                labels=data["y_train"][:bank_size],
                default=None,
            )
            if fitted_memory is not None:
                for offset in range(len(filtered_x)):
                    context = _safe_call(
                        "memory.retrieve",
                        memory_module.retrieve,
                        query_embedding=query_embeddings[offset],
                        query_fg=filtered_fg[offset] if len(filtered_fg) > offset else np.zeros(x_train.shape[1], dtype=np.float32),
                        predicted_class=int(filtered_predictions[offset]),
                        default={},
                    ) or {}
                    memory_results[offset] = {
                        "labels": [int(value) for value in context.get("labels", [])] if context else [],
                        "similarities": [float(value) for value in context.get("similarities", [])] if context else [],
                        "top_similarity": float(context.get("top_similarity", 0.0)) if context else 0.0,
                    }

    fg_strength = _fg_strengths(filtered_fg)
    fusion = RiskFusion(random_state=42)

    records = []
    planner_samples = []
    llm_pipeline = _init_optional_llm("llm.init", _build_llm_pipeline, DEFAULT_LLM_MODEL)
    llm_limit = len(filtered_x)
    retraining_interval = 1 if demo_mode else min(10, max(1, runtime["batch_size"] // 2))
    latest_retraining_update = {"best_method": None, "metrics": {}}

    for offset, source_idx in enumerate(high_risk_indices.tolist()):
        sample_graph = graph_results.get(offset, {"top_neighbors": [], "max_weight": 0.0})
        sample_memory = memory_results.get(offset, {"labels": [], "similarities": [], "top_similarity": 0.0})
        if not sample_memory.get("labels"):
            runtime_context = runtime_memory.retrieve(
                query_embedding=filtered_x[offset],
                predicted_class=int(filtered_predictions[offset]),
            )
            sample_memory = {
                "labels": [int(value) for value in runtime_context.get("labels", [])],
                "similarities": [float(value) for value in runtime_context.get("similarities", [])],
                "top_similarity": float(runtime_context.get("top_similarity", 0.0)),
            }
        sample_fg_strength = float(fg_strength[offset]) if len(fg_strength) > offset else 0.0
        fusion_output = _safe_call(
            "risk_fusion.compute",
            fusion.compute_risk,
            float(filtered_confidences[offset]),
            float(sample_memory.get("top_similarity", 0.0)),
            float(sample_graph.get("max_weight", 0.0)),
            sample_fg_strength,
            default={"risk_score": float(filtered_confidences[offset]), "severity": severity_from_risk(float(filtered_confidences[offset]))},
        ) or {"risk_score": float(filtered_confidences[offset]), "severity": severity_from_risk(float(filtered_confidences[offset]))}

        attack = _make_attack_name(filtered_predictions[offset])
        uncertainty = float(1.0 - filtered_confidences[offset])
        top_features = lime_results.get(int(source_idx), [])
        if len(top_features) < 5:
            top_features = _fallback_top_features(filtered_x[offset], feature_names, top_k=5)
        planner_action = _dynamic_decision(str(fusion_output["severity"]), float(filtered_confidences[offset]), uncertainty)
        target_meta = _build_review_target(float(fusion_output["risk_score"]), uncertainty, attack, float(filtered_confidences[offset]))

        llm_reasoning = None
        llm_prompt = None
        if llm_pipeline is not None and offset < llm_limit:
            llm_output = _safe_call(
                "llm.run",
                llm_pipeline.run,
                filtered_x[offset].tolist(),
                int(filtered_predictions[offset]),
                float(filtered_confidences[offset]),
                default=None,
            )
            if llm_output is not None:
                llm_prompt, llm_reasoning = llm_output
        if not llm_reasoning:
            llm_reasoning = _fallback_reasoning(
                top_features=top_features,
                risk_score=float(fusion_output["risk_score"]),
                graph_output=sample_graph,
                memory_output=sample_memory,
                decision=planner_action,
            )

        planner_samples.append(
            {
                "attack": attack,
                "confidence": float(filtered_confidences[offset]),
                "severity": str(fusion_output["severity"]),
                "target_action": str(target_meta["target_action"]),
                "severity_adjustment": float(target_meta["severity_adjustment"]),
            }
        )

        records.append(
            {
                "sample_index": int(source_idx),
                "classifier_output": {
                    "prediction": int(filtered_predictions[offset]),
                    "confidence": float(filtered_confidences[offset]),
                },
                "lime_features": top_features,
                "graph_correlation": sample_graph,
                "memory_match": sample_memory,
                "fused_risk": {
                    "risk_score": float(fusion_output["risk_score"]),
                    "severity": str(fusion_output["severity"]),
                },
                "llm_reasoning": llm_reasoning,
                "llm_prompt": llm_prompt,
                "planner_target": target_meta,
            }
        )
        runtime_memory.add(filtered_x[offset], int(filtered_predictions[offset]))

    planner_comparison = _compare_decision_planners(planner_samples) if planner_samples else {"best_planner": None, "metrics": {}}
    best_planner_name = planner_comparison.get("best_planner")

    planner_instances = {
        "rule_based": RuleBasedPlanner(),
        "confidence_based": ConfidenceBasedPlanner(),
        "risk_based": RiskBasedPlanner(),
        "hybrid": HybridPlanner(),
        "rl": DecisionRLPlanner(),
        "policy": PolicyBasedPlanner(),
    }
    if best_planner_name == "llm":
        best_planner = _init_optional_llm("decision_planner.llm.reinit", LLMPlanner)
        if best_planner is not None:
            planner_instances["llm"] = best_planner

    if best_planner_name in planner_instances and planner_samples:
        _safe_call(
            "decision_planner.best.fit",
            _safe_planner_fit,
            best_planner_name,
            planner_instances[best_planner_name],
            [sample["attack"] for sample in planner_samples],
            [sample["confidence"] for sample in planner_samples],
            [sample["severity"] for sample in planner_samples],
            [sample["target_action"] for sample in planner_samples],
            default=None,
        )

    for offset, record in enumerate(records):
        risk_score = float(record["fused_risk"]["risk_score"])
        severity = str(record["fused_risk"]["severity"])
        confidence = float(record["classifier_output"]["confidence"])
        attack = _make_attack_name(int(record["classifier_output"]["prediction"]))
        uncertainty = float(1.0 - confidence)

        suggested_action = _dynamic_decision(severity, confidence, uncertainty)
        planner_action = None
        if best_planner_name in planner_instances:
            planner_action = _safe_call(
                "decision_planner.best.decide",
                planner_instances[best_planner_name].decide,
                attack,
                confidence,
                severity,
                default=suggested_action,
            )

        decision_output = {
            "event_id": f"sample_{record['sample_index']}",
            "risk_score": risk_score,
            "uncertainty": uncertainty,
            "explanation": record["llm_reasoning"],
            "suggested_action": suggested_action,
        }
        review_decision = trigger_human_review(decision_output, risk_threshold=0.75, uncertainty_threshold=0.35)
        if review_decision.requires_review:
            request = ReviewRequest(
                event_id=decision_output["event_id"],
                risk_score=risk_score,
                explanation=decision_output["explanation"],
                suggested_action=suggested_action,
                uncertainty=uncertainty,
            )
            review_output = _safe_call("human_review.simulate", simulate_human_review, request, default=None)
            review_output = review_output.model_dump() if review_output is not None else {
                "approved": False,
                "correct_action": suggested_action,
                "feedback": "Human review unavailable.",
                "severity_adjustment": 0.0,
            }
        else:
            review_output = {
                "approved": True,
                "correct_action": suggested_action,
                "feedback": "Auto-approved because risk and uncertainty stayed below review thresholds.",
                "severity_adjustment": 0.0,
            }
        review_output["severity_adjustment"] = 0.0

        feedback_record = _safe_call(
            "feedback.log",
            log_feedback,
            str(db_path),
            str(results_dir),
            decision_output,
            review_output,
            default=None,
        )

        record["decision"] = decision_output
        record["planner_action"] = planner_action
        record["human_review"] = review_output
        record["feedback_record"] = feedback_record
        if ((offset + 1) % retraining_interval == 0) or (offset + 1 == len(records)):
            latest_retraining_update = _compare_retraining_methods(db_path)
        record["retraining_update"] = latest_retraining_update

    retraining_comparison = latest_retraining_update

    summary = {
        "status": "completed",
        "runtime": runtime,
        "processed_path": processed_path,
        "model_path": model_path,
        "results_dir": str(results_dir),
        "db_path": str(db_path),
        "log_path": str(log_path),
        "classifier": {
            "evaluated_samples": int(len(x_test)),
            "high_risk_samples": int(len(high_risk_indices)),
        },
        "decision_planner": planner_comparison,
        "retraining": retraining_comparison,
        "records": records,
    }

    summary_path = results_dir / ("demo_pipeline_summary.json" if demo_mode else "full_pipeline_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_full_pipeline(processed_path: str, model_path: str) -> dict[str, Any]:
    """Run the full integrated pipeline with dynamic sample limits."""

    return _core_pipeline(processed_path=processed_path, model_path=model_path, sample_override=None, demo_mode=False)


def run_demo_pipeline() -> dict[str, Any]:
    """Run a small, presentation-friendly pipeline demo."""

    processed_path, model_path = _resolve_default_paths()
    summary = _core_pipeline(processed_path=processed_path, model_path=model_path, sample_override=8, demo_mode=True)

    for idx, record in enumerate(summary.get("records", []), start=1):
        print("---------------------------------")
        print(f" SAMPLE {idx}")
        print("---------------------------------")
        print(f"Classifier Output: {record['classifier_output']}")
        print(f"Top Features: {record.get('lime_features')}")
        print(f"Graph Correlation: {record.get('graph_correlation')}")
        print(f"Memory Match: {record.get('memory_match')}")
        print(f"Fused Risk Score: {record.get('fused_risk')}")
        print(f"LLM Reasoning: {record.get('llm_reasoning')}")
        print(f"Decision: {record.get('decision')}")
        print(f"Human Review: {record.get('human_review')}")
        print(f"Feedback Stored: {record.get('feedback_record')}")
        print(f"Retraining Update: {record.get('retraining_update')}")
        print("---------------------------------")

    return summary
