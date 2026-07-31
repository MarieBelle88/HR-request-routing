from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.evaluation import summarize


def test_perfect_tiered_prediction_scores_one():
    gold = [
        {
            "request_id": "HR001",
            "gold_category": "payroll",
            "gold_urgency": "high",
            "gold_action": "direct_answer",
            "gold_escalate_to_large_model": True,
        }
    ]
    outputs = [
        {
            "request_id": "HR001",
            "route": "large_model",
            "prediction": {
                "category": "payroll",
                "urgency": "high",
                "action": "direct_answer",
            },
            "escalation": {"escalated": True, "reasons": ["high_urgency"]},
            "model_calls": {"small_model": 1, "large_model": 1},
            "metrics": {
                "total_latency_seconds": 2.0,
                "gpu_seconds": 1.9,
                "peak_vram_bytes": 100,
            },
        }
    ]
    metrics, rows = summarize(gold, outputs, seed=1, bootstrap_samples=100)
    assert metrics["exact_resolution_accuracy"] == 1.0
    assert metrics["large_model_escalation_f1"] == 1.0
    assert metrics["small_model_calls"] == 1
    assert metrics["large_model_calls"] == 1
    assert rows[0]["exact_resolution_correct"] is True


def test_duplicate_output_ids_are_rejected():
    gold = [
        {
            "request_id": "HR001",
            "gold_category": "payroll",
            "gold_urgency": "low",
            "gold_action": "direct_answer",
            "gold_escalate_to_large_model": False,
        }
    ]
    output = {
        "request_id": "HR001",
        "route": "small_model",
        "prediction": {
            "category": "payroll",
            "urgency": "low",
            "action": "direct_answer",
        },
        "model_calls": {"small_model": 1, "large_model": 0},
        "metrics": {},
    }
    try:
        summarize(gold, [output, output])
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Duplicate output IDs should fail evaluation.")
