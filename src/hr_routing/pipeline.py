from __future__ import annotations

import time
from dataclasses import asdict

from .model_runner import QwenClassifier
from .rules import deterministic_route


def qwen_log(record: dict, classifier: QwenClassifier, system: str) -> dict:
    prediction, metrics, raw, issues = classifier.classify(record["message"])
    return {
        "request_id": record["request_id"],
        "split": record["split"],
        "system": system,
        "route": "qwen",
        "prediction": prediction.model_dump(),
        "metrics": asdict(metrics),
        "raw_output": raw,
        "validation_issues": issues,
        "official_run": True,
    }


def baseline(records: list[dict], classifier: QwenClassifier) -> list[dict]:
    return [qwen_log(record, classifier, "baseline") for record in records]


def tiered(records: list[dict], classifier: QwenClassifier | None) -> list[dict]:
    outputs = []
    for record in records:
        start = time.perf_counter()
        routed = deterministic_route(record["message"])
        if routed is None:
            if classifier is None:
                outputs.append(
                    {
                        "request_id": record["request_id"],
                        "split": record["split"],
                        "system": "tiered",
                        "route": "qwen_pending",
                        "prediction": None,
                        "metrics": {"latency_seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0, "peak_vram_bytes": 0},
                        "raw_output": None,
                        "validation_issues": ["rules_only_smoke"],
                        "official_run": False,
                    }
                )
            else:
                outputs.append(qwen_log(record, classifier, "tiered"))
            continue
        route, prediction, matched_rule = routed
        outputs.append(
            {
                "request_id": record["request_id"],
                "split": record["split"],
                "system": "tiered",
                "route": route,
                "prediction": prediction.model_dump(),
                "metrics": {
                    "latency_seconds": time.perf_counter() - start,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "peak_vram_bytes": 0,
                },
                "raw_output": None,
                "validation_issues": [],
                "matched_rule": matched_rule,
                "official_run": classifier is not None,
            }
        )
    return outputs

