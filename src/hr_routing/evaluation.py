from __future__ import annotations

import random
from collections import Counter


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def score_one(gold: dict, output: dict) -> dict:
    pred = output.get("prediction") or {}
    category_correct = pred.get("category") == gold["gold_category"]
    urgency_correct = pred.get("urgency") == gold["gold_urgency"]
    action_correct = pred.get("action") == gold["gold_action"]
    gold_escalate = gold["gold_action"] == "human_review"
    pred_escalate = pred.get("action") == "human_review"
    return {
        "request_id": gold["request_id"],
        "category_correct": category_correct,
        "urgency_correct": urgency_correct,
        "action_correct": action_correct,
        "exact_resolution_correct": category_correct and urgency_correct and action_correct,
        "gold_escalate": gold_escalate,
        "pred_escalate": pred_escalate,
        "unsafe_auto_resolution": gold_escalate and not pred_escalate,
        "route_correct": output["route"] == gold["gold_ideal_route"],
        "route": output["route"],
        "qwen_called": output["route"] == "qwen",
        "latency_seconds": float(output.get("metrics", {}).get("latency_seconds") or 0.0),
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


def summarize(gold_records: list[dict], outputs: list[dict], seed: int = 20260731) -> tuple[dict, list[dict]]:
    by_id = {record["request_id"]: record for record in gold_records}
    if set(by_id) != {output["request_id"] for output in outputs}:
        raise ValueError("Output IDs do not exactly match the selected dataset split.")
    rows = [score_one(by_id[output["request_id"]], output) for output in outputs]
    n = len(rows)
    tp = sum(row["gold_escalate"] and row["pred_escalate"] for row in rows)
    fp = sum(not row["gold_escalate"] and row["pred_escalate"] for row in rows)
    fn = sum(row["gold_escalate"] and not row["pred_escalate"] for row in rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    route_counts = Counter(row["route"] for row in rows)
    accuracy_by_route = {}
    for route in sorted(route_counts):
        route_rows = [row for row in rows if row["route"] == route]
        accuracy_by_route[route] = safe_div(
            sum(row["exact_resolution_correct"] for row in route_rows), len(route_rows)
        )
    non_human_rows = [row for row in rows if row["route"] != "human"]
    metrics = {
        "n": n,
        "category_accuracy": safe_div(sum(row["category_correct"] for row in rows), n),
        "urgency_accuracy": safe_div(sum(row["urgency_correct"] for row in rows), n),
        "action_accuracy": safe_div(sum(row["action_correct"] for row in rows), n),
        "exact_resolution_accuracy": safe_div(sum(row["exact_resolution_correct"] for row in rows), n),
        "exact_resolution_accuracy_95ci": bootstrap_ci(
            [float(row["exact_resolution_correct"]) for row in rows], seed
        ),
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": safe_div(2 * precision * recall, precision + recall),
        "unsafe_auto_resolution_rate": safe_div(sum(row["unsafe_auto_resolution"] for row in rows), n),
        "route_accuracy": safe_div(sum(row["route_correct"] for row in rows), n),
        "exact_resolution_accuracy_by_route": accuracy_by_route,
        "non_human_resolution_accuracy": safe_div(
            sum(row["exact_resolution_correct"] for row in non_human_rows), len(non_human_rows)
        ),
        "human_review_requests": route_counts.get("human", 0),
        "human_review_rate": safe_div(route_counts.get("human", 0), n),
        "qwen_calls": sum(row["qwen_called"] for row in rows),
        "qwen_call_rate": safe_div(sum(row["qwen_called"] for row in rows), n),
        "total_latency_seconds": sum(row["latency_seconds"] for row in rows),
        "mean_latency_seconds": safe_div(sum(row["latency_seconds"] for row in rows), n),
        "route_counts": dict(route_counts),
    }
    return metrics, rows
