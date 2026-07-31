#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_routing.io_utils import load_json, load_jsonl, write_json, write_jsonl
from hr_routing.model_runner import QwenClassifier
from hr_routing.pipeline import baseline, tiered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--system", choices=["baseline", "tiered", "both"], default="both")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--rules-only-smoke", action="store_true")
    return parser.parse_args()


def environment() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        import transformers

        result.update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        result["model_packages_available"] = False
    return result


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / "config" / "experiment.json")
    all_records = load_jsonl(ROOT / config["dataset_path"])
    records = [record for record in all_records if record["split"] == args.split]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rules_only_smoke:
        outputs = tiered(records, classifier=None)
        write_jsonl(output_dir / f"{args.split}_rules_smoke.jsonl", outputs)
        print(f"Rules-only smoke complete: {len(outputs)} records (not official results).")
        return

    classifier = QwenClassifier(
        config["model"],
        config["generation"],
        (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8"),
    )
    classifier.load()
    if args.system in {"baseline", "both"}:
        baseline_outputs = baseline(records, classifier)
        write_jsonl(output_dir / f"{args.split}_baseline.jsonl", baseline_outputs)
        print(f"Baseline complete: {len(baseline_outputs)} records.")
    if args.system in {"tiered", "both"}:
        tiered_outputs = tiered(records, classifier)
        write_jsonl(output_dir / f"{args.split}_tiered.jsonl", tiered_outputs)
        print(f"Tiered system complete: {len(tiered_outputs)} records.")
    write_json(output_dir / "runtime_environment.json", environment())


if __name__ == "__main__":
    main()

