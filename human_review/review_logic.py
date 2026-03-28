"""Trigger logic and simulation helpers for human-in-the-loop review."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from human_review.schemas import ReviewDecision, ReviewRequest, ReviewResponse


LOGGER = logging.getLogger(__name__)


def trigger_human_review(
    decision_output: Mapping[str, Any],
    risk_threshold: float = 0.75,
    uncertainty_threshold: float = 0.35,
) -> ReviewDecision:
    """Return whether the event should be escalated to a human reviewer."""

    risk_score = float(decision_output.get("risk_score", 0.0))
    uncertainty = float(decision_output.get("uncertainty", 0.0))
    event_id = str(decision_output.get("event_id", "unknown"))

    if risk_score > risk_threshold:
        reason = f"risk_score>{risk_threshold:.2f}"
        LOGGER.info("Human review triggered for %s because %s", event_id, reason)
        return ReviewDecision(event_id=event_id, requires_review=True, reason=reason)

    if uncertainty > uncertainty_threshold:
        reason = f"uncertainty>{uncertainty_threshold:.2f}"
        LOGGER.info("Human review triggered for %s because %s", event_id, reason)
        return ReviewDecision(event_id=event_id, requires_review=True, reason=reason)

    LOGGER.info("Human review not required for %s", event_id)
    return ReviewDecision(event_id=event_id, requires_review=False, reason="below-threshold")


def simulate_human_review(request: ReviewRequest) -> ReviewResponse:
    """Deterministic stand-in reviewer for Colab demos."""

    approved = request.suggested_action in {"No Action", "Monitor"} and request.risk_score < 0.8
    if request.risk_score >= 0.85:
        correct_action = "Block"
    elif request.risk_score >= 0.65:
        correct_action = "Alert"
    elif request.suggested_action == "No Action" and request.uncertainty > 0.25:
        correct_action = "Monitor"
    else:
        correct_action = request.suggested_action

    feedback = (
        "Automated analyst review approved the current action."
        if approved and correct_action == request.suggested_action
        else f"Adjusted action to {correct_action} based on risk and uncertainty."
    )

    severity_adjustment = round((request.risk_score - 0.5) * 0.4, 3)
    response = ReviewResponse(
        approved=approved and correct_action == request.suggested_action,
        correct_action=correct_action,
        feedback=feedback,
        severity_adjustment=severity_adjustment,
    )
    LOGGER.info("Simulated review completed for %s: %s", request.event_id, response.correct_action)
    return response
