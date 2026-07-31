#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.io_utils import load_json, load_jsonl, write_json, write_jsonl
from hr_routing.model_runner import QwenClassifier
from hr_routing.pipeline import compose_tiered, escalation_decisions, run_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--system", choices=["baseline", "tiered", "both"], default="both")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(config: dict, args: argparse.Namespace) -> dict:
    result = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit(),
        "split": args.split,
        "system": args.system,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "batch_size": config["generation"]["batch_size"],
        "small_model": config["models"]["small_model"],
        "large_model": config["models"]["large_model"],
    }
    try:
        import bitsandbytes
        import torch
        import transformers

        result.update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "bitsandbytes": bitsandbytes.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_memory_bytes": (
                    torch.cuda.get_device_properties(0).total_memory
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        result["model_packages_available"] = False
    return result


def load_classifier(config: dict, role: str, prompt: str) -> QwenClassifier:
    classifier = QwenClassifier(
        config["models"][role],
        config["generation"],
        prompt,
        model_role=role,
    )
    classifier.load()
    return classifier


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / "config" / "experiment.json")
    all_records = load_jsonl(ROOT / config["dataset_path"])
    records = [record for record in all_records if record["split"] == args.split]
    if not records:
        raise SystemExit(f"No records found for split {args.split!r}.")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")

    small_outputs: list[dict] = []
    decisions: list[dict] = []
    if args.system in {"tiered", "both"}:
        print("Loading fast SLM: Qwen2.5-3B-Instruct")
        small_classifier = load_classifier(config, "small_model", prompt)
        small_outputs = run_model(
            records, small_classifier, "small_model_tier", "small_model"
        )
        write_jsonl(output_dir / f"{args.split}_small_model.jsonl", small_outputs)
        decisions = escalation_decisions(records, small_outputs, config["escalation"])
        write_jsonl(output_dir / f"{args.split}_escalation_decisions.jsonl", decisions)
        small_classifier.unload()
        print(f"Small-model tier complete: {len(small_outputs)} requests.")

    print("Loading larger model: Qwen2.5-7B-Instruct")
    large_classifier = load_classifier(config, "large_model", prompt)
    baseline_outputs: list[dict] = []
    large_outputs_for_tiered: list[dict] = []

    if args.system in {"baseline", "both"}:
        baseline_outputs = run_model(
            records, large_classifier, "uniform_large_model_baseline", "large_model"
        )
        write_jsonl(output_dir / f"{args.split}_baseline.jsonl", baseline_outputs)
        print(f"Uniform 7B baseline complete: {len(baseline_outputs)} requests.")

    if args.system in {"tiered", "both"}:
        escalated_ids = {
            row["request_id"]
            for row in decisions
            if row["escalate_to_large_model"]
        }
        if args.system == "both":
            # Deterministic decoding allows the baseline's 7B output to be reused
            # for the same input when composing the tiered result. Metrics still
            # count a theoretical 7B call for every escalated tiered request.
            large_outputs_for_tiered = [
                row for row in baseline_outputs if row["request_id"] in escalated_ids
            ]
        else:
            escalated_records = [
                record for record in records if record["request_id"] in escalated_ids
            ]
            large_outputs_for_tiered = run_model(
                escalated_records,
                large_classifier,
                "tiered_large_model_stage",
                "large_model",
            )

        tiered_outputs = compose_tiered(
            records, small_outputs, decisions, large_outputs_for_tiered
        )
        write_jsonl(output_dir / f"{args.split}_tiered.jsonl", tiered_outputs)
        print(
            f"Tiered system complete: {len(tiered_outputs)} requests; "
            f"{len(escalated_ids)} escalated to 7B."
        )

    large_classifier.unload()
    write_json(output_dir / "runtime_environment.json", environment(config, args))


if __name__ == "__main__":
    main()
