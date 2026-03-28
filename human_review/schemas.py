"""Pydantic schemas for human review requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)
    suggested_action: str = Field(..., min_length=1)
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)


class ReviewResponse(BaseModel):
    approved: bool
    correct_action: str
    feedback: str
    severity_adjustment: float = Field(..., ge=-1.0, le=1.0)
    reviewer_id: str = "simulated-analyst"
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


class ReviewDecision(BaseModel):
    event_id: str
    requires_review: bool
    reason: Optional[str] = None
