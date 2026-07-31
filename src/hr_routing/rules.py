from __future__ import annotations

import re

from .schema import Prediction


SENSITIVE_PATTERNS: list[tuple[str, str, str]] = [
    (r"harass|sexual comment|inappropriate touch|touched me inappropriate|bully|mocks my accent", "workplace_issue", "harassment or bullying allegation"),
    (r"discriminat|racist|rassistisch|because of my religion|because of my gender", "workplace_issue", "possible discrimination"),
    (r"unsafe|threaten|hit me|violence|immediate danger", "workplace_issue", "possible immediate safety risk"),
    (r"medical condition|disability|pregnan|mental health|diagnosis", "contract", "sensitive health or accommodation matter"),
    (r"lawyer|legal action|wrongful|tribunal|court", "contract", "possible legal dispute"),
    (r"fired unfairly|dismissal dispute|contest my termination", "offboarding", "dismissal dispute"),
    (r"another employee.{0,30}(salary|address|medical|file)|coworker.{0,30}(salary|address|medical|file)", "personal_data", "request for another employee's confidential data"),
]


FAQ_PATTERNS: list[tuple[str, str]] = [
    (r"^(where (can|do) i (find|download) my payslip|how do i access my payslip)", "payroll"),
    (r"^(how do i report sick leave|where do i submit (a )?sick note)", "leave"),
    (r"^(where can i see my (remaining )?(holiday|vacation) days|how do i check my leave balance)", "leave"),
    (r"^(how do i update my (home )?address|where can i change my bank details)", "personal_data"),
    (r"^(where (is|can i find) the remote[- ]work policy|where can i read the travel policy)", "policy"),
    (r"^(where do i find the mandatory training|how do i access the compliance course)", "training"),
    (r"^(what documents should i bring on my first day|where is the onboarding checklist)", "onboarding"),
    (r"^(where do i return (my )?(laptop|equipment)|how do i return company equipment)", "offboarding"),
    (r"^(where can i find information about the pension plan|how do i view my benefits portal)", "benefits"),
]


def infer_urgency(message: str) -> str:
    text = message.lower()
    if re.search(r"\b(urgent|immediately|today|within 24 hours|before my shift)\b", text):
        return "high"
    if re.search(r"\b(this week|tomorrow|soon|next monday|before friday)\b", text):
        return "medium"
    return "low"


def deterministic_route(message: str) -> tuple[str, Prediction, str] | None:
    """Return route, prediction, and matched rule, or None for Qwen."""
    normalized = " ".join(message.lower().split())

    for pattern, category, reason in SENSITIVE_PATTERNS:
        if re.search(pattern, normalized):
            urgency = "high" if re.search(r"unsafe|threaten|violence|immediate|urgent", normalized) else "medium"
            return (
                "human",
                Prediction(
                    category=category,
                    urgency=urgency,
                    action="human_review",
                    escalation_reason=reason,
                    confidence=0.99,
                ),
                pattern,
            )

    if len(normalized) < 18 or normalized in {"help", "i need hr", "can hr call me?"}:
        return (
            "human",
            Prediction(
                category="other",
                urgency="low",
                action="human_review",
                escalation_reason="insufficient information",
                confidence=0.99,
            ),
            "insufficient_information",
        )

    for pattern, category in FAQ_PATTERNS:
        if re.search(pattern, normalized):
            return (
                "rules",
                Prediction(
                    category=category,
                    urgency=infer_urgency(normalized),
                    action="direct_answer",
                    escalation_reason=None,
                    confidence=0.99,
                ),
                pattern,
            )
    return None
