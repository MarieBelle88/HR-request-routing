import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_shape_and_split():
    records = [json.loads(line) for line in (ROOT / "data" / "hr_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 60
    assert Counter(row["split"] for row in records) == {"test": 45, "dev": 15}
    assert len({row["request_id"] for row in records}) == 60

