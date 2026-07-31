# Evaluating a Tiered Language-Model Routing System for HR Service Requests

This repository contains the reproducible experiment for an Applied AI term
paper. It compares two ways of processing employee HR requests:

1. **Uniform-Qwen baseline:** every request is classified by
   `Qwen/Qwen2.5-3B-Instruct`.
2. **Tiered system:** safe, highly recognisable FAQs are handled by deterministic
   rules; sensitive or clearly high-risk cases are routed to human review; the
   remaining requests are classified by Qwen2.5-3B.

The goal is to test whether routing reduces language-model use and latency while
maintaining acceptable resolution and escalation accuracy.

## Research question

Can a tiered routing system process HR service requests more efficiently than
sending every request to one language model, while maintaining acceptable
classification accuracy and safe escalation?

## Dataset

`data/hr_requests.jsonl` contains 60 synthetic HR requests:

- 15 development records used to check the implementation and freeze rules;
- 45 held-out test records used for the reported comparison;
- routine FAQs, ordinary casework, ambiguous messages, and sensitive requests;
- English, German, and a small number of code-switched stress cases.

No real employee messages or personal data are included. See the dataset card
and annotation guide before interpreting the results.

## Outputs and metrics

The experiment records category, urgency, action, escalation reason, route,
latency, tokens, and model calls. The evaluation reports:

- category and urgency accuracy;
- action accuracy and exact resolution accuracy;
- escalation precision, recall, and F1;
- unsafe auto-resolution rate;
- Qwen-call reduction and relative inference-cost reduction;
- total and per-request latency.

## Run locally without a model

```bash
python scripts/build_dataset.py
python scripts/verify_dataset.py
python scripts/offline_checks.py
pytest -q
python scripts/run_experiment.py --split test --rules-only-smoke
```

## Run the official experiment in Google Colab

Open `notebooks/HR_Routing_Study_Colab.ipynb`, select a T4 GPU, and run the
cells in order. The notebook installs the pinned dependencies, runs checks,
executes the development split, then executes the held-out test comparison.

Equivalent command-line run:

```bash
python scripts/run_experiment.py --split test --system both --output-dir results
python scripts/evaluate_results.py \
  --baseline results/test_baseline.jsonl \
  --tiered results/test_tiered.jsonl \
  --output-dir results
```

The official paper results must come from a genuine GPU run. Smoke outputs are
marked and are rejected by the evaluator as official evidence.

## Reproducibility notes

- Model: Qwen2.5-3B-Instruct, 4-bit NF4 quantisation.
- Decoding: deterministic (`do_sample=false`).
- Dataset split and rules are fixed in the repository.
- Runtime package versions and GPU information are saved with the results.
- Synthetic data limits external validity; results should not be presented as
  proof of production readiness.
