import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_shape_split_and_new_labels():
    records = [
        json.loads(line)
        for line in (ROOT / "data" / "hr_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 60
    assert Counter(row["split"] for row in records) == {"test": 45, "dev": 15}
    assert len({row["request_id"] for row in records}) == 60
    assert sum(row["gold_escalate_to_large_model"] for row in records) == 17
    assert sum(row["gold_action"] == "human_review" for row in records) == 15
    assert all("gold_ideal_route" not in row for row in records)
