"""FastAPI review interface for human-in-the-loop approval."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from human_review.review_logic import simulate_human_review
from human_review.schemas import ReviewRequest, ReviewResponse


LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Human Review API", version="1.0.0")


@app.post("/review", response_model=ReviewResponse)
def review_event(payload: ReviewRequest) -> ReviewResponse:
    """Review a flagged decision event."""

    LOGGER.info("Received review request for %s", payload.event_id)
    return simulate_human_review(payload)


def create_app() -> FastAPI:
    """Factory for embedding the review API in other processes."""

    return app
