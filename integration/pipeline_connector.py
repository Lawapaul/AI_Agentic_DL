"""Connector that closes the loop between decisions, humans, storage, and learning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from feedback_store.feedback_db import init_db
from feedback_store.feedback_logger import log_feedback
from human_review.review_logic import simulate_human_review, trigger_human_review
from human_review.schemas import ReviewRequest
from retraining.data_loader import load_feedback_dataset
from retraining.rl_trainer import RLTrainer
from retraining.supervised_trainer import SupervisedDecisionTrainer


LOGGER = logging.getLogger(__name__)


class AgenticPipelineConnector:
    """Production-style connector for the decision -> review -> learn loop."""

    def __init__(
        self,
        db_path: str,
        results_dir: str,
        risk_threshold: float = 0.75,
        uncertainty_threshold: float = 0.35,
    ):
        self.db_path = db_path
        self.results_dir = results_dir
        self.risk_threshold = risk_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.rl_trainer = RLTrainer()
        self.supervised_trainer = SupervisedDecisionTrainer()

        init_db(self.db_path)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def _coerce_decision_output(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if "decision_output" in event:
            output = dict(event["decision_output"])
        else:
            output = {
                "event_id": str(event.get("event_id", "unknown")),
                "risk_score": float(event.get("risk_score", 0.0)),
                "uncertainty": float(event.get("uncertainty", 0.0)),
                "explanation": str(event.get("explanation", "No explanation provided.")),
                "suggested_action": str(event.get("suggested_action", "Monitor")),
            }
        output.setdefault("event_id", str(event.get("event_id", "unknown")))
        output.setdefault("risk_score", 0.0)
        output.setdefault("uncertainty", 0.0)
        output.setdefault("explanation", "No explanation provided.")
        output.setdefault("suggested_action", "Monitor")
        return output

    def run_full_pipeline(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Run decision review, feedback logging, and learning updates for one event."""

        decision_output = self._coerce_decision_output(event)
        LOGGER.info("Processing event %s", decision_output["event_id"])

        review_decision = trigger_human_review(
            decision_output,
            risk_threshold=self.risk_threshold,
            uncertainty_threshold=self.uncertainty_threshold,
        )

        if review_decision.requires_review:
            review_request = ReviewRequest(
                event_id=str(decision_output["event_id"]),
                risk_score=float(decision_output["risk_score"]),
                explanation=str(decision_output["explanation"]),
                suggested_action=str(decision_output["suggested_action"]),
                uncertainty=float(decision_output.get("uncertainty", 0.0)),
            )
            review_output = simulate_human_review(review_request).model_dump()
        else:
            review_output = {
                "approved": True,
                "correct_action": str(decision_output["suggested_action"]),
                "feedback": "Review skipped because thresholds were not exceeded.",
                "severity_adjustment": 0.0,
            }

        feedback_record = log_feedback(self.db_path, self.results_dir, decision_output, review_output)
        self.rl_trainer.update_policy(
            risk_score=float(decision_output["risk_score"]),
            action=str(feedback_record["human_action"]),
            reward=float(feedback_record["reward"]),
        )

        features, labels = load_feedback_dataset(self.db_path)
        training_summary = None
        if len(features) >= 2:
            training_summary = self.supervised_trainer.train_model(features, labels)

        policy_path = Path(self.results_dir) / "rl_policy.json"
        self.rl_trainer.dump_policy(str(policy_path))

        result = {
            "decision_output": decision_output,
            "review_decision": review_decision.model_dump(),
            "review_output": review_output,
            "feedback_record": feedback_record,
            "rl_policy_snapshot": self.rl_trainer.q_table,
            "supervised_training": training_summary.__dict__ if training_summary else None,
        }

        summary_path = Path(self.results_dir) / "latest_pipeline_run.json"
        summary_path.write_text(json.dumps(result, indent=2, default=str))
        return result


def run_full_pipeline(event: Mapping[str, Any], connector: AgenticPipelineConnector) -> dict[str, Any]:
    """Functional wrapper for notebook or pipeline usage."""

    return connector.run_full_pipeline(event)
