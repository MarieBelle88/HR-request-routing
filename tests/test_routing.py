from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.rules import should_escalate
from hr_routing.schema import Prediction


CONFIG = {
    "confidence_threshold": 0.75,
    "escalate_on_validation_error": True,
    "escalate_on_human_review_action": True,
    "escalate_on_high_urgency": True,
    "escalate_on_sensitive_topic": True,
    "escalate_on_ambiguous_request": True,
}


def prediction(**updates) -> Prediction:
    values = {
        "category": "payroll",
        "urgency": "low",
        "action": "direct_answer",
        "escalation_reason": None,
        "confidence": 0.95,
    }
    values.update(updates)
    return Prediction(**values)


def test_reliable_ordinary_request_stays_at_small_model():
    decision = should_escalate(
        "Where can I download my payslip?", prediction(), [], CONFIG
    )
    assert decision.escalate is False
    assert decision.reasons == []


def test_sensitive_complaint_escalates_to_large_model():
    decision = should_escalate(
        "My manager keeps making sexual comments.",
        prediction(category="workplace_issue", action="human_review"),
        [],
        CONFIG,
    )
    assert decision.escalate is True
    assert "sensitive_workplace_complaint" in decision.reasons


def test_low_confidence_escalates_to_large_model():
    decision = should_escalate(
        "My overtime is missing.", prediction(confidence=0.40), [], CONFIG
    )
    assert decision.escalate is True
    assert "small_model_low_confidence" in decision.reasons


def test_validation_error_escalates_to_large_model():
    decision = should_escalate(
        "My overtime is missing.",
        prediction(),
        ["parse_or_schema_error:JSONDecodeError"],
        CONFIG,
    )
    assert decision.escalate is True
    assert "small_model_validation_error" in decision.reasons


def test_high_urgency_escalates_to_large_model():
    decision = should_escalate(
        "I need the payslip today.", prediction(urgency="high"), [], CONFIG
    )
    assert decision.escalate is True
    assert "high_urgency" in decision.reasons
