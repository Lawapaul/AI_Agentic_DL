"""Schema models for human review requests and responses."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime
from typing import Optional

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - fallback for lightweight local demos
    BaseModel = None
    Field = None


if BaseModel is not None:

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

else:

    @dataclass
    class _SchemaFallback:
        """Minimal BaseModel-like API used when Pydantic is unavailable."""

        def __init__(self, **kwargs):
            for schema_field in fields(self):
                if schema_field.name in kwargs:
                    value = kwargs[schema_field.name]
                elif schema_field.default is not MISSING:
                    value = schema_field.default
                elif schema_field.default_factory is not MISSING:
                    value = schema_field.default_factory()
                else:
                    raise TypeError(f"Missing required field: {schema_field.name}")
                setattr(self, schema_field.name, value)

        def model_dump(self):
            return asdict(self)


    @dataclass(init=False)
    class ReviewRequest(_SchemaFallback):
        event_id: str
        risk_score: float
        explanation: str
        suggested_action: str
        uncertainty: float = 0.0


    @dataclass(init=False)
    class ReviewResponse(_SchemaFallback):
        approved: bool
        correct_action: str
        feedback: str
        severity_adjustment: float
        reviewer_id: str = "simulated-analyst"
        reviewed_at: datetime = field(default_factory=datetime.utcnow)


    @dataclass(init=False)
    class ReviewDecision(_SchemaFallback):
        event_id: str
        requires_review: bool
        reason: Optional[str] = None
