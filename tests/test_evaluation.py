from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.evaluation import summarize


def test_perfect_prediction_scores_one():
    gold = [{"request_id": "HR001", "gold_category": "payroll", "gold_urgency": "low", "gold_action": "direct_answer", "gold_ideal_route": "qwen"}]
    outputs = [{"request_id": "HR001", "route": "qwen", "prediction": {"category": "payroll", "urgency": "low", "action": "direct_answer"}, "metrics": {"latency_seconds": 1.0}}]
    metrics, rows = summarize(gold, outputs, seed=1)
    assert metrics["exact_resolution_accuracy"] == 1.0
    assert metrics["route_accuracy"] == 1.0
    assert rows[0]["exact_resolution_correct"] is True

