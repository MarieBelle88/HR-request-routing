#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.io_utils import load_jsonl, write_jsonl
from hr_routing.rules import deterministic_route


def main() -> None:
    records = load_jsonl(ROOT / "data" / "hr_requests.jsonl")
    for record in records:
        routed = deterministic_route(record["message"])
        expected = record["gold_ideal_route"]
        actual = "qwen" if routed is None else routed[0]
        assert actual == expected, (record["request_id"], expected, actual)
        if routed is not None:
            prediction = routed[1].model_dump()
            for field in ("category", "urgency", "action"):
                assert prediction[field] == record[f"gold_{field}"], (
                    record["request_id"], field, prediction[field], record[f"gold_{field}"]
                )

    json.loads((ROOT / "notebooks" / "HR_Routing_Study_Colab.ipynb").read_text(encoding="utf-8"))

    test = [record for record in records if record["split"] == "test"]
    baseline = []
    tiered = []
    for record in test:
        prediction = {
            "category": record["gold_category"],
            "urgency": record["gold_urgency"],
            "action": record["gold_action"],
            "escalation_reason": record["gold_escalation_reason"],
            "confidence": 0.9,
        }
        model_metrics = {"latency_seconds": 1.0, "prompt_tokens": 50, "output_tokens": 30, "peak_vram_bytes": 1}
        baseline.append({"request_id": record["request_id"], "split": "test", "system": "baseline", "route": "qwen", "prediction": prediction, "metrics": model_metrics, "official_run": False})
        deterministic = deterministic_route(record["message"])
        route = "qwen" if deterministic is None else deterministic[0]
        tiered.append({"request_id": record["request_id"], "split": "test", "system": "tiered", "route": route, "prediction": prediction, "metrics": model_metrics if route == "qwen" else {"latency_seconds": 0.001, "prompt_tokens": 0, "output_tokens": 0, "peak_vram_bytes": 0}, "official_run": False})

    with tempfile.TemporaryDirectory(prefix="hr-routing-check-") as temporary:
        temporary_path = Path(temporary)
        write_jsonl(temporary_path / "baseline.jsonl", baseline)
        write_jsonl(temporary_path / "tiered.jsonl", tiered)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_results.py"),
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
        summary = json.loads((temporary_path / "analysis" / "experiment_summary.json").read_text())
        assert summary["official_results"] is False
        assert summary["baseline"]["exact_resolution_accuracy"] == 1.0
        assert summary["tiered"]["exact_resolution_accuracy"] == 1.0
        assert (temporary_path / "analysis" / "figures" / "accuracy_comparison.png").exists()

    print("Offline checks passed: routing, labels, notebook JSON, evaluation, and figures.")


if __name__ == "__main__":
    main()

