from __future__ import annotations

from dataclasses import asdict

from .model_runner import QwenClassifier
from .rules import should_escalate


def model_log(record: dict, classifier: QwenClassifier, system: str, route: str) -> dict:
    prediction, metrics, raw, issues = classifier.classify(record["message"])
    return {
        "request_id": record["request_id"],
        "split": record["split"],
        "system": system,
        "route": route,
        "model_role": classifier.model_role,
        "model_name": classifier.model_name,
        "prediction": prediction.model_dump(),
        "metrics": asdict(metrics),
        "raw_output": raw,
        "validation_issues": issues,
        "model_calls": {
            "small_model": int(classifier.model_role == "small_model"),
            "large_model": int(classifier.model_role == "large_model"),
        },
        "official_run": True,
    }


def run_model(
    records: list[dict],
    classifier: QwenClassifier,
    system: str,
    route: str,
) -> list[dict]:
    return [model_log(record, classifier, system, route) for record in records]


def escalation_decisions(
    records: list[dict],
    small_outputs: list[dict],
    escalation_config: dict,
) -> list[dict]:
    records_by_id = {record["request_id"]: record for record in records}
    decisions = []
    for output in small_outputs:
        record = records_by_id[output["request_id"]]
        from .schema import Prediction

        prediction = Prediction.model_validate(output["prediction"])
        decision = should_escalate(
            record["message"],
            prediction,
            output.get("validation_issues", []),
            escalation_config,
        )
        decisions.append(
            {
                "request_id": output["request_id"],
                "split": output["split"],
                "escalate_to_large_model": decision.escalate,
                "reasons": decision.reasons,
                "routing_latency_seconds": decision.routing_latency_seconds,
            }
        )
    return decisions


def compose_tiered(
    records: list[dict],
    small_outputs: list[dict],
    decisions: list[dict],
    large_outputs: list[dict],
) -> list[dict]:
    records_by_id = {record["request_id"]: record for record in records}
    small_by_id = {row["request_id"]: row for row in small_outputs}
    decision_by_id = {row["request_id"]: row for row in decisions}
    large_by_id = {row["request_id"]: row for row in large_outputs}
    outputs = []

    for request_id in records_by_id:
        small = small_by_id[request_id]
        decision = decision_by_id[request_id]
        escalated = bool(decision["escalate_to_large_model"])
        large = large_by_id.get(request_id)
        if escalated and large is None:
            raise ValueError(f"Missing large-model output for escalated request {request_id}.")

        final = large if escalated else small
        small_metrics = small.get("metrics", {})
        large_metrics = large.get("metrics", {}) if large else {}
        routing_latency = float(decision.get("routing_latency_seconds") or 0.0)
        small_latency = float(small_metrics.get("latency_seconds") or 0.0)
        large_latency = float(large_metrics.get("latency_seconds") or 0.0)
        small_gpu = float(small_metrics.get("gpu_seconds") or 0.0)
        large_gpu = float(large_metrics.get("gpu_seconds") or 0.0)
        peak_values = [
            value
            for value in [
                small_metrics.get("peak_vram_bytes"),
                large_metrics.get("peak_vram_bytes"),
            ]
            if value is not None
        ]

        outputs.append(
            {
                "request_id": request_id,
                "split": small["split"],
                "system": "tiered_3b_rules_7b",
                "route": "large_model" if escalated else "small_model",
                "prediction": final["prediction"],
                "small_model_prediction": small["prediction"],
                "large_model_prediction": large["prediction"] if large else None,
                "escalation": {
                    "escalated": escalated,
                    "reasons": decision["reasons"],
                },
                "metrics": {
                    "total_latency_seconds": small_latency + routing_latency + large_latency,
                    "routing_latency_seconds": routing_latency,
                    "small_model_latency_seconds": small_latency,
                    "large_model_latency_seconds": large_latency,
                    "gpu_seconds": small_gpu + large_gpu,
                    "prompt_tokens": int(small_metrics.get("prompt_tokens") or 0)
                    + int(large_metrics.get("prompt_tokens") or 0),
                    "output_tokens": int(small_metrics.get("output_tokens") or 0)
                    + int(large_metrics.get("output_tokens") or 0),
                    "peak_vram_bytes": max(peak_values) if peak_values else None,
                },
                "model_calls": {"small_model": 1, "large_model": int(escalated)},
                "raw_output": {
                    "small_model": small.get("raw_output"),
                    "large_model": large.get("raw_output") if large else None,
                },
                "validation_issues": {
                    "small_model": small.get("validation_issues", []),
                    "large_model": large.get("validation_issues", []) if large else [],
                },
                "official_run": bool(
                    small.get("official_run") is True
                    and (not escalated or large.get("official_run") is True)
                ),
            }
        )
    return outputs
