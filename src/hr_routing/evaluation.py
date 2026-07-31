from __future__ import annotations

import random
from collections import Counter


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_scores(gold_values: list[bool], predicted_values: list[bool]) -> dict:
    tp = sum(gold and pred for gold, pred in zip(gold_values, predicted_values))
    fp = sum(not gold and pred for gold, pred in zip(gold_values, predicted_values))
    fn = sum(gold and not pred for gold, pred in zip(gold_values, predicted_values))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def score_one(gold: dict, output: dict) -> dict:
    pred = output.get("prediction") or {}
    category_correct = pred.get("category") == gold["gold_category"]
    urgency_correct = pred.get("urgency") == gold["gold_urgency"]
    action_correct = pred.get("action") == gold["gold_action"]
    gold_human = gold["gold_action"] == "human_review"
    pred_human = pred.get("action") == "human_review"
    escalation = output.get("escalation")
    predicted_large_escalation = (
        bool(escalation.get("escalated")) if isinstance(escalation, dict) else None
    )
    model_calls = output.get("model_calls", {})
    metrics = output.get("metrics", {})
    latency = float(
        metrics.get("total_latency_seconds", metrics.get("latency_seconds", 0.0)) or 0.0
    )
    return {
        "request_id": gold["request_id"],
        "category_correct": category_correct,
        "urgency_correct": urgency_correct,
        "action_correct": action_correct,
        "exact_resolution_correct": category_correct and urgency_correct and action_correct,
        "gold_human_review": gold_human,
        "pred_human_review": pred_human,
        "unsafe_auto_resolution": gold_human and not pred_human,
        "gold_escalate_to_large_model": bool(gold["gold_escalate_to_large_model"]),
        "pred_escalate_to_large_model": predicted_large_escalation,
        "route": output["route"],
        "small_model_calls": int(model_calls.get("small_model", 0)),
        "large_model_calls": int(model_calls.get("large_model", 0)),
        "latency_seconds": latency,
        "gpu_seconds": float(metrics.get("gpu_seconds") or 0.0),
        "peak_vram_bytes": metrics.get("peak_vram_bytes"),
    }


def bootstrap_ci(values: list[float], seed: int, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def summarize(
    gold_records: list[dict],
    outputs: list[dict],
    seed: int = 20260731,
    bootstrap_samples: int = 1000,
) -> tuple[dict, list[dict]]:
    by_id = {record["request_id"]: record for record in gold_records}
    output_ids = [output["request_id"] for output in outputs]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Outputs contain duplicate request IDs.")
    if set(by_id) != set(output_ids):
        raise ValueError("Output IDs do not exactly match the selected dataset split.")

    rows = [score_one(by_id[output["request_id"]], output) for output in outputs]
    n = len(rows)
    route_counts = Counter(row["route"] for row in rows)
    human_scores = binary_scores(
        [row["gold_human_review"] for row in rows],
        [row["pred_human_review"] for row in rows],
    )
    accuracy_by_route = {}
    for route in sorted(route_counts):
        route_rows = [row for row in rows if row["route"] == route]
        accuracy_by_route[route] = safe_div(
            sum(row["exact_resolution_correct"] for row in route_rows), len(route_rows)
        )

    escalation_rows = [
        row for row in rows if row["pred_escalate_to_large_model"] is not None
    ]
    escalation_scores = None
    if escalation_rows:
        escalation_scores = binary_scores(
            [row["gold_escalate_to_large_model"] for row in escalation_rows],
            [row["pred_escalate_to_large_model"] for row in escalation_rows],
        )

    peak_vram_values = [
        int(row["peak_vram_bytes"])
        for row in rows
        if row["peak_vram_bytes"] is not None
    ]
    metrics = {
        "n": n,
        "category_accuracy": safe_div(sum(row["category_correct"] for row in rows), n),
        "urgency_accuracy": safe_div(sum(row["urgency_correct"] for row in rows), n),
        "action_accuracy": safe_div(sum(row["action_correct"] for row in rows), n),
        "exact_resolution_accuracy": safe_div(
            sum(row["exact_resolution_correct"] for row in rows), n
        ),
        "exact_resolution_accuracy_95ci": bootstrap_ci(
            [float(row["exact_resolution_correct"]) for row in rows],
            seed,
            bootstrap_samples,
        ),
        "human_review_precision": human_scores["precision"],
        "human_review_recall": human_scores["recall"],
        "human_review_f1": human_scores["f1"],
        "unsafe_auto_resolution_rate": safe_div(
            sum(row["unsafe_auto_resolution"] for row in rows), n
        ),
        "large_model_escalation_precision": (
            escalation_scores["precision"] if escalation_scores else None
        ),
        "large_model_escalation_recall": (
            escalation_scores["recall"] if escalation_scores else None
        ),
        "large_model_escalation_f1": escalation_scores["f1"] if escalation_scores else None,
        "exact_resolution_accuracy_by_route": accuracy_by_route,
        "small_model_calls": sum(row["small_model_calls"] for row in rows),
        "small_model_call_rate": safe_div(sum(row["small_model_calls"] for row in rows), n),
        "large_model_calls": sum(row["large_model_calls"] for row in rows),
        "large_model_call_rate": safe_div(sum(row["large_model_calls"] for row in rows), n),
        "total_latency_seconds": sum(row["latency_seconds"] for row in rows),
        "mean_latency_seconds": safe_div(sum(row["latency_seconds"] for row in rows), n),
        "total_gpu_seconds": sum(row["gpu_seconds"] for row in rows),
        "mean_gpu_seconds": safe_div(sum(row["gpu_seconds"] for row in rows), n),
        "peak_vram_bytes": max(peak_vram_values) if peak_vram_values else None,
        "route_counts": dict(route_counts),
    }
    return metrics, rows
