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
from hr_routing.io_utils import load_json, load_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tiered", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--allow-unofficial", action="store_true")
    return parser.parse_args()


def make_figures(
    output_dir: Path,
    small_metrics: dict,
    baseline_metrics: dict,
    tiered_metrics: dict,
    comparison: dict,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    system_names = ["3B tier", "Uniform 7B", "Tiered final"]
    metric_names = ["Category", "Urgency", "Action", "Exact"]
    metric_keys = [
        "category_accuracy",
        "urgency_accuracy",
        "action_accuracy",
        "exact_resolution_accuracy",
    ]
    x = list(range(len(metric_names)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (name, metrics) in enumerate(
        zip(system_names, [small_metrics, baseline_metrics, tiered_metrics])
    ):
        positions = [value + (offset - 1) * width for value in x]
        ax.bar(positions, [metrics[key] for key in metric_keys], width, label=name)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, metric_names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    escalation_values = [
        tiered_metrics["large_model_escalation_precision"],
        tiered_metrics["large_model_escalation_recall"],
        tiered_metrics["large_model_escalation_f1"],
    ]
    ax.bar(["Precision", "Recall", "F1"], escalation_values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("3B-to-7B escalation performance")
    fig.tight_layout()
    fig.savefig(figures / "escalation_performance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(
        ["Uniform 7B", "Tiered"],
        [baseline_metrics["large_model_calls"], tiered_metrics["large_model_calls"]],
    )
    axes[0].set_ylabel("7B calls")
    axes[0].set_title("Large-model use")
    axes[1].bar(
        ["Uniform 7B", "Tiered"],
        [comparison["baseline_cost_proxy_units"], comparison["tiered_cost_proxy_units"]],
    )
    axes[1].set_ylabel("Relative compute units")
    axes[1].set_title("Parameter-weighted cost proxy")
    fig.tight_layout()
    fig.savefig(figures / "efficiency_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        ["Uniform 7B", "Tiered"],
        [baseline_metrics["total_gpu_seconds"], tiered_metrics["total_gpu_seconds"]],
    )
    ax.set_ylabel("GPU-seconds")
    ax.set_title("Measured inference time")
    fig.tight_layout()
    fig.savefig(figures / "gpu_time_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    small_outputs = load_jsonl(args.small_model)
    baseline_outputs = load_jsonl(args.baseline)
    tiered_outputs = load_jsonl(args.tiered)
    all_outputs = small_outputs + baseline_outputs + tiered_outputs
    if not args.allow_unofficial:
        if not all_outputs or not all(row.get("official_run") is True for row in all_outputs):
            raise SystemExit("Refusing unofficial, incomplete, or smoke-test outputs.")

    split_values = {
        rows[0]["split"] for rows in [small_outputs, baseline_outputs, tiered_outputs] if rows
    }
    if len(split_values) != 1:
        raise SystemExit("Small-model, baseline, and tiered outputs use different splits.")
    split = split_values.pop()
    gold = [
        record
        for record in load_jsonl(ROOT / "data" / "hr_requests.jsonl")
        if record["split"] == split
    ]
    config = load_json(ROOT / "config" / "experiment.json")
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["seed"])
    small_metrics, small_rows = summarize(gold, small_outputs, seed, samples)
    baseline_metrics, baseline_rows = summarize(gold, baseline_outputs, seed, samples)
    tiered_metrics, tiered_rows = summarize(gold, tiered_outputs, seed, samples)

    baseline_large_calls = baseline_metrics["large_model_calls"]
    large_call_reduction = (
        1 - tiered_metrics["large_model_calls"] / baseline_large_calls
        if baseline_large_calls
        else 0.0
    )
    small_unit = float(config["cost_proxy"]["small_model_call_units"])
    large_unit = float(config["cost_proxy"]["large_model_call_units"])
    baseline_cost = baseline_metrics["large_model_calls"] * large_unit
    tiered_cost = (
        tiered_metrics["small_model_calls"] * small_unit
        + tiered_metrics["large_model_calls"] * large_unit
    )
    gpu_seconds_reduction = (
        1 - tiered_metrics["total_gpu_seconds"] / baseline_metrics["total_gpu_seconds"]
        if baseline_metrics["total_gpu_seconds"]
        else 0.0
    )
    latency_reduction = (
        1 - tiered_metrics["total_latency_seconds"] / baseline_metrics["total_latency_seconds"]
        if baseline_metrics["total_latency_seconds"]
        else 0.0
    )
    comparison = {
        "large_model_call_reduction": large_call_reduction,
        "baseline_cost_proxy_units": baseline_cost,
        "tiered_cost_proxy_units": tiered_cost,
        "relative_cost_proxy_reduction": 1 - tiered_cost / baseline_cost if baseline_cost else 0.0,
        "gpu_seconds_reduction": gpu_seconds_reduction,
        "latency_reduction": latency_reduction,
        "exact_resolution_accuracy_difference": (
            tiered_metrics["exact_resolution_accuracy"]
            - baseline_metrics["exact_resolution_accuracy"]
        ),
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "split": split,
        "official_results": all(row.get("official_run") is True for row in all_outputs),
        "small_model_tier": small_metrics,
        "uniform_large_model_baseline": baseline_metrics,
        "tiered_system": tiered_metrics,
        "comparison": comparison,
        "cost_note": config["cost_proxy"]["description"],
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_summary.json", summary)
    metric_keys = [
        "category_accuracy",
        "urgency_accuracy",
        "action_accuracy",
        "exact_resolution_accuracy",
        "human_review_precision",
        "human_review_recall",
        "human_review_f1",
        "unsafe_auto_resolution_rate",
        "large_model_escalation_precision",
        "large_model_escalation_recall",
        "large_model_escalation_f1",
        "small_model_calls",
        "large_model_calls",
        "total_latency_seconds",
        "mean_latency_seconds",
        "total_gpu_seconds",
        "mean_gpu_seconds",
        "peak_vram_bytes",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "small_model_3b", "uniform_large_model_7b", "tiered_final"])
        for key in metric_keys:
            writer.writerow(
                [key, small_metrics.get(key), baseline_metrics.get(key), tiered_metrics.get(key)]
            )

    by_id = {row["request_id"]: row for row in gold}
    error_rows = []
    for system, rows in [
        ("small_model_3b", small_rows),
        ("uniform_large_model_7b", baseline_rows),
        ("tiered_final", tiered_rows),
    ]:
        for row in rows:
            if not row["exact_resolution_correct"]:
                error_rows.append(
                    {"system": system, "message": by_id[row["request_id"]]["message"], **row}
                )
    with (output_dir / "error_analysis.jsonl").open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    make_figures(output_dir, small_metrics, baseline_metrics, tiered_metrics, comparison)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
