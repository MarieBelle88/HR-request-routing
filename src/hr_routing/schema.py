from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Category = Literal[
    "leave",
    "payroll",
    "benefits",
    "onboarding",
    "offboarding",
    "contract",
    "workplace_issue",
    "policy",
    "training",
    "personal_data",
    "other",
]
Urgency = Literal["low", "medium", "high"]
Action = Literal["direct_answer", "human_review"]


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    urgency: Urgency
    action: Action
    escalation_reason: str | None = Field(default=None, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)

