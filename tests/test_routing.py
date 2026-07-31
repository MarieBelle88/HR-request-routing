from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.rules import deterministic_route


def test_safe_faq_goes_to_rules():
    route, prediction, _ = deterministic_route("Where can I download my payslip?")
    assert route == "rules"
    assert prediction.category == "payroll"
    assert prediction.action == "direct_answer"


def test_sensitive_complaint_goes_to_human():
    route, prediction, _ = deterministic_route("My manager keeps making sexual comments.")
    assert route == "human"
    assert prediction.category == "workplace_issue"
    assert prediction.action == "human_review"


def test_ordinary_case_goes_to_qwen():
    assert deterministic_route("My overtime from June is missing from this month's salary.") is None


def test_short_ambiguous_request_goes_to_human():
    route, prediction, _ = deterministic_route("Help")
    assert route == "human"
    assert prediction.category == "other"

