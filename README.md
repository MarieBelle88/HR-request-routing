# Evaluating a Tiered Language-Model Routing System for HR Service Requests

This repository contains the reproducible experiment for an Applied AI term
paper on Topic 16, **Routing and Filtering Layers in Enterprise**.

## Required architecture and baseline

The tiered system follows the course architecture exactly:

```text
HR request
→ Tier 1: Qwen2.5-3B-Instruct (fast SLM)
→ Tier 2: fixed rule-based escalation decision
→ Tier 3: Qwen2.5-7B-Instruct for escalated requests
→ final action: direct answer or qualified human HR review
```

The baseline sends every request directly to the same 7B model. Human review is
a final operational action; it does not replace the required large-model tier.

## Research question

Can a 3B-to-7B tiered routing system reduce use of the larger model and
inference cost while maintaining resolution accuracy on synthetic HR service
requests compared with a uniform 7B deployment?

## Dataset

`data/hr_requests.jsonl` contains 60 synthetic HR requests:

- 15 development records for implementation checks and threshold selection;
- 45 held-out test records for the reported comparison;
- routine processes, ordinary casework, explicit urgency, sensitive workplace
  matters, legal/health/confidentiality cases, and an ambiguous request;
- mainly English with three German stress cases.

The data evaluate pretrained models; they are not used to train or fine-tune
Qwen. `gold_escalate_to_large_model` is separate from the final
`gold_action = human_review` label. See the dataset card and annotation guide.

## Escalation criteria

The 3B output is escalated to 7B when at least one fixed criterion applies:

- JSON parsing or schema validation fails;
- confidence is below `0.75`;
- the 3B model requests human review;
- urgency is high;
- the message contains a sensitive workplace, health, legal, safety, dismissal,
  discrimination, or confidentiality issue;
- the request is materially ambiguous or insufficient.

Rules make only the escalation decision. They do not classify requests or
answer FAQs.

## Evaluation

The evaluation reports:

- category, urgency, action, and exact-resolution accuracy for the 3B tier,
  uniform 7B baseline, and final tiered system;
- precision, recall, and F1 for 3B-to-7B escalation;
- precision, recall, F1, and unsafe-auto-resolution rate for human review;
- 3B and 7B call counts and percentage reduction in 7B calls;
- relative parameter-weighted compute units;
- latency, GPU-seconds, peak VRAM, batch size, model names, quantisation, package
  versions, and GPU information.

## Offline checks

These checks do not download or run either model and do not produce paper
results:

```bash
python scripts/build_dataset.py
python scripts/verify_dataset.py
python scripts/offline_checks.py
pytest -q
```

## Official Google Colab experiment

Open `notebooks/HR_Routing_Study_Colab.ipynb` in Google Colab, select a T4 GPU,
and run the cells in order. Both models use 4-bit NF4 weights with FP16 compute
and are loaded sequentially to fit the 15 GB environment.

Equivalent commands:

```bash
python scripts/run_experiment.py --split test --system both --output-dir results
python scripts/evaluate_results.py \
  --small-model results/test_small_model.jsonl \
  --baseline results/test_baseline.jsonl \
  --tiered results/test_tiered.jsonl \
  --output-dir results
```

With deterministic decoding, the official `both` run reuses each applicable 7B
baseline output when composing the tiered result for the identical request.
The evaluation still counts a conceptual 7B call for every escalated tiered
request. This avoids repeating identical GPU inference while preserving the two
system comparisons.

## Reproducibility and limitations

- Random seed: `20260731`.
- Decoding: deterministic (`do_sample=false`).
- Batch size: 1.
- Models: Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct.
- Quantisation: 4-bit NF4 with FP16 computation.
- Official evidence must be produced on the complete locked 45-record test set.
- The small, synthetic, single-annotator dataset limits generalisability and
  does not demonstrate production readiness.
