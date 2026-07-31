from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .schema import Prediction


SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (
        r"harass|sexual comment|inappropriate touch|touched me inappropriate|"
        r"bully|mocks my accent|rassistische witze",
        "sensitive_workplace_complaint",
    ),
    (
        r"discriminat|racist|rassistisch|because of my religion|because of my gender",
        "possible_discrimination",
    ),
    (
        r"unsafe|threaten|hit me|violence|immediate danger",
        "possible_safety_risk",
    ),
    (
        r"medical condition|disability|pregnan|mental health|diagnosis|accommodation",
        "sensitive_health_or_accommodation_matter",
    ),
    (r"lawyer|legal action|wrongful|tribunal|court|illegal", "possible_legal_dispute"),
    (
        r"fired unfairly|dismissal dispute|contest my termination",
        "dismissal_dispute",
    ),
    (
        r"another employee.{0,35}(salary|address|medical|file)|"
        r"coworker.{0,35}(salary|address|medical|file)",
        "another_employee_confidential_data",
    ),
]

AMBIGUOUS_REQUESTS = {
    "help",
    "i need hr",
    "can hr call me?",
    "please help",
}


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    reasons: list[str]
    routing_latency_seconds: float


def sensitive_reasons(message: str) -> list[str]:
    normalized = " ".join(message.lower().split())
    return [reason for pattern, reason in SENSITIVE_PATTERNS if re.search(pattern, normalized)]


def is_ambiguous(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    return normalized in AMBIGUOUS_REQUESTS or len(normalized) < 8


def should_escalate(
    message: str,
    prediction: Prediction,
    validation_issues: list[str],
    escalation_config: dict,
) -> EscalationDecision:
    """Inspect the 3B result and decide whether the request proceeds to 7B.

    The rules never answer or classify an HR request. They only make the
    Topic-16 routing decision between the fast SLM and the larger LLM.
    """

    start = time.perf_counter()
    reasons: list[str] = []

    if escalation_config.get("escalate_on_validation_error", True) and validation_issues:
        reasons.append("small_model_validation_error")

    threshold = float(escalation_config.get("confidence_threshold", 0.75))
    if prediction.confidence < threshold:
        reasons.append("small_model_low_confidence")

    if (
        escalation_config.get("escalate_on_human_review_action", True)
        and prediction.action == "human_review"
    ):
        reasons.append("small_model_requested_human_review")

    if (
        escalation_config.get("escalate_on_high_urgency", True)
        and prediction.urgency == "high"
    ):
        reasons.append("high_urgency")

    if escalation_config.get("escalate_on_sensitive_topic", True):
        reasons.extend(sensitive_reasons(message))

    if escalation_config.get("escalate_on_ambiguous_request", True) and is_ambiguous(message):
        reasons.append("ambiguous_or_insufficient_request")

    reasons = list(dict.fromkeys(reasons))
    return EscalationDecision(
        escalate=bool(reasons),
        reasons=reasons,
        routing_latency_seconds=time.perf_counter() - start,
    )
