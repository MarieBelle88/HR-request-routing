from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from .schema import Prediction


@dataclass
class ModelMetrics:
    latency_seconds: float
    prompt_tokens: int
    output_tokens: int
    peak_vram_bytes: int | None


def extract_json(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class QwenClassifier:
    def __init__(self, model_config: dict, generation_config: dict, system_prompt: str):
        self.model_config = model_config
        self.generation_config = generation_config
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self.torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["name"])
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.model_config.get("load_in_4bit", True):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.model_config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_config["name"], **kwargs)
        self.model.eval()

    def classify(self, message: str) -> tuple[Prediction, ModelMetrics, str, list[str]]:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Call load() before classify().")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"HR request:\n{message}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=int(self.generation_config["max_new_tokens"]),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        latency = time.perf_counter() - start
        new_tokens = generated[0, encoded["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        issues: list[str] = []
        try:
            prediction = Prediction.model_validate(extract_json(raw))
        except Exception as exc:
            issues.append(f"parse_or_schema_error:{type(exc).__name__}")
            prediction = Prediction(
                category="other",
                urgency="low",
                action="human_review",
                escalation_reason="model output could not be validated",
                confidence=0.0,
            )
        metrics = ModelMetrics(
            latency_seconds=latency,
            prompt_tokens=int(encoded["input_ids"].shape[-1]),
            output_tokens=int(new_tokens.shape[-1]),
            peak_vram_bytes=(
                int(self.torch.cuda.max_memory_allocated()) if self.torch.cuda.is_available() else None
            ),
        )
        return prediction, metrics, raw, issues

