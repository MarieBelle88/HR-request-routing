#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.evaluation import summarize
from hr_routing.io_utils import load_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tiered", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--allow-unofficial", action="store_true")
    return parser.parse_args()


def make_figures(output_dir: Path, baseline_metrics: dict, tiered_metrics: dict) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    labels = ["Category", "Urgency", "Action", "Exact resolution"]
    keys = ["category_accuracy", "urgency_accuracy", "action_accuracy", "exact_resolution_accuracy"]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - width / 2 for i in x], [baseline_metrics[k] for k in keys], width, label="Uniform Qwen")
    ax.bar([i + width / 2 for i in x], [tiered_metrics[k] for k in keys], width, label="Tiered")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(x), labels, rotation=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "accuracy_comparison.png", dpi=180)
    plt.close(fig)

    labels = ["Precision", "Recall", "F1"]
    keys = ["escalation_precision", "escalation_recall", "escalation_f1"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in range(3)], [baseline_metrics[k] for k in keys], width, label="Uniform Qwen")
    ax.bar([i + width / 2 for i in range(3)], [tiered_metrics[k] for k in keys], width, label="Tiered")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(range(3), labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "escalation_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Uniform Qwen", "Tiered"], [baseline_metrics["qwen_calls"], tiered_metrics["qwen_calls"]])
    ax.set_ylabel("Qwen calls")
    ax.set_title("Language-model use on the test set")
    fig.tight_layout()
    fig.savefig(figures / "efficiency_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_outputs = load_jsonl(args.baseline)
    tiered_outputs = load_jsonl(args.tiered)
    if not args.allow_unofficial:
        if not baseline_outputs or not tiered_outputs or not all(
            row.get("official_run") is True for row in baseline_outputs + tiered_outputs
        ):
            raise SystemExit("Refusing unofficial, incomplete, or smoke-test outputs.")
    split = baseline_outputs[0]["split"]
    if split != tiered_outputs[0]["split"]:
        raise SystemExit("Baseline and tiered outputs use different splits.")
    gold = [record for record in load_jsonl(ROOT / "data" / "hr_requests.jsonl") if record["split"] == split]
    baseline_metrics, baseline_rows = summarize(gold, baseline_outputs)
    tiered_metrics, tiered_rows = summarize(gold, tiered_outputs)
    baseline_calls = baseline_metrics["qwen_calls"]
    call_reduction = 1 - tiered_metrics["qwen_calls"] / baseline_calls if baseline_calls else 0.0
    latency_reduction = (
        1 - tiered_metrics["total_latency_seconds"] / baseline_metrics["total_latency_seconds"]
        if baseline_metrics["total_latency_seconds"]
        else 0.0
    )
    summary = {
        "experiment_id": "hr-routing-qwen3b-v1",
        "split": split,
        "official_results": all(row.get("official_run") is True for row in baseline_outputs + tiered_outputs),
        "baseline": baseline_metrics,
        "tiered": tiered_metrics,
        "comparison": {
            "qwen_call_reduction": call_reduction,
            "relative_inference_cost_reduction": call_reduction,
            "latency_reduction": latency_reduction,
            "exact_resolution_accuracy_difference": tiered_metrics["exact_resolution_accuracy"] - baseline_metrics["exact_resolution_accuracy"],
        },
        "cost_note": "Cost is a relative model-call proxy, not a monetary API estimate. Human-review workload is reported separately.",
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_summary.json", summary)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "uniform_qwen", "tiered"])
        for key in ["category_accuracy", "urgency_accuracy", "action_accuracy", "exact_resolution_accuracy", "non_human_resolution_accuracy", "route_accuracy", "escalation_precision", "escalation_recall", "escalation_f1", "unsafe_auto_resolution_rate", "human_review_requests", "human_review_rate", "qwen_calls", "qwen_call_rate", "total_latency_seconds", "mean_latency_seconds"]:
            writer.writerow([key, baseline_metrics[key], tiered_metrics[key]])
    by_id = {row["request_id"]: row for row in gold}
    error_rows = []
    for system, rows in [("baseline", baseline_rows), ("tiered", tiered_rows)]:
        for row in rows:
            if not row["exact_resolution_correct"]:
                error_rows.append({"system": system, "message": by_id[row["request_id"]]["message"], **row})
    with (output_dir / "error_analysis.jsonl").open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    make_figures(output_dir, baseline_metrics, tiered_metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
