#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.io_utils import load_json, load_jsonl, write_jsonl
from hr_routing.pipeline import compose_tiered, escalation_decisions


def fake_model_output(record: dict, model_role: str, system: str) -> dict:
    prediction = {
        "category": record["gold_category"],
        "urgency": record["gold_urgency"],
        "action": record["gold_action"],
        "escalation_reason": record["gold_escalation_reason"],
        "confidence": 0.95,
    }
    latency = 0.7 if model_role == "small_model" else 1.4
    return {
        "request_id": record["request_id"],
        "split": record["split"],
        "system": system,
        "route": model_role,
        "model_role": model_role,
        "model_name": f"offline_fixture_{model_role}",
        "prediction": prediction,
        "metrics": {
            "latency_seconds": latency,
            "gpu_seconds": latency,
            "prompt_tokens": 50,
            "output_tokens": 30,
            "peak_vram_bytes": 1,
        },
        "raw_output": json.dumps(prediction),
        "validation_issues": [],
        "model_calls": {
            "small_model": int(model_role == "small_model"),
            "large_model": int(model_role == "large_model"),
        },
        "official_run": False,
    }


def main() -> None:
    records = load_jsonl(ROOT / "data" / "hr_requests.jsonl")
    config = load_json(ROOT / "config" / "experiment.json")
    small_outputs = [
        fake_model_output(record, "small_model", "small_model_tier")
        for record in records
    ]
    decisions = escalation_decisions(records, small_outputs, config["escalation"])
    for record, decision in zip(records, decisions):
        assert decision["escalate_to_large_model"] == record[
            "gold_escalate_to_large_model"
        ], (record["request_id"], decision["reasons"])

    notebook = json.loads(
        (ROOT / "notebooks" / "HR_Routing_Study_Colab.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4

    test_records = [record for record in records if record["split"] == "test"]
    test_ids = {record["request_id"] for record in test_records}
    test_small = [row for row in small_outputs if row["request_id"] in test_ids]
    test_decisions = [row for row in decisions if row["request_id"] in test_ids]
    baseline = [
        fake_model_output(record, "large_model", "uniform_large_model_baseline")
        for record in test_records
    ]
    escalated_ids = {
        row["request_id"]
        for row in test_decisions
        if row["escalate_to_large_model"]
    }
    tiered_large = [row for row in baseline if row["request_id"] in escalated_ids]
    tiered = compose_tiered(test_records, test_small, test_decisions, tiered_large)

    with tempfile.TemporaryDirectory(prefix="hr-routing-check-") as temporary:
        temporary_path = Path(temporary)
        write_jsonl(temporary_path / "small.jsonl", test_small)
        write_jsonl(temporary_path / "baseline.jsonl", baseline)
        write_jsonl(temporary_path / "tiered.jsonl", tiered)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_results.py"),
                "--small-model",
                str(temporary_path / "small.jsonl"),
                "--baseline",
                str(temporary_path / "baseline.jsonl"),
                "--tiered",
                str(temporary_path / "tiered.jsonl"),
                "--output-dir",
                str(temporary_path / "analysis"),
                "--allow-unofficial",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads(
            (temporary_path / "analysis" / "experiment_summary.json").read_text()
        )
        assert summary["official_results"] is False
        assert summary["small_model_tier"]["exact_resolution_accuracy"] == 1.0
        assert summary["uniform_large_model_baseline"]["exact_resolution_accuracy"] == 1.0
        assert summary["tiered_system"]["exact_resolution_accuracy"] == 1.0
        assert summary["tiered_system"]["large_model_escalation_f1"] == 1.0
        assert (temporary_path / "analysis" / "figures" / "accuracy_comparison.png").exists()

    print(
        "Offline checks passed: Topic-16 routing, labels, notebook JSON, "
        "evaluation, and figures. Fixtures are explicitly unofficial."
    )


if __name__ == "__main__":
    main()
