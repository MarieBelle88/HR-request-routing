#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "hr_requests.jsonl"
VALID_CATEGORIES = {
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
}


def main() -> None:
    records = [
        json.loads(line)
        for line in PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 60, f"Expected 60 records, got {len(records)}"
    assert len({record["request_id"] for record in records}) == 60
    assert Counter(record["split"] for record in records) == {"test": 45, "dev": 15}
    assert {record["gold_category"] for record in records} <= VALID_CATEGORIES
    assert {record["gold_urgency"] for record in records} <= {"low", "medium", "high"}
    assert {record["gold_action"] for record in records} <= {
        "direct_answer",
        "human_review",
    }
    assert all(isinstance(record["gold_escalate_to_large_model"], bool) for record in records)
    assert sum(record["gold_escalate_to_large_model"] for record in records) == 17
    assert sum(record["gold_action"] == "human_review" for record in records) == 15
    assert all(
        record["gold_action"] != "human_review"
        or record["gold_escalate_to_large_model"]
        for record in records
    )
    assert all(record["message"].strip() for record in records)
    assert all(record["source_type"] == "synthetic_controlled" for record in records)
    print(
        "Dataset verified: 60 records; 15 dev / 45 test; "
        "17 gold 3B-to-7B escalations; 15 final human-review actions."
    )


if __name__ == "__main__":
    main()
