# Paper handoff

Complete the paper only after the official Colab results exist. Every reported
table value must trace to `results/metrics.csv` or
`results/experiment_summary.json`; use the generated figures without manually
changing their values.

## Working title

**Evaluating a Tiered Language-Model Routing System for HR Service Requests**

## Research question

**Can a 3B-to-7B tiered routing system reduce use of the larger model and
inference cost while maintaining resolution accuracy on synthetic HR service
requests compared with a uniform 7B deployment?**

## Architecture to describe

```text
Qwen2.5-3B fast SLM
→ rule-based escalation
→ Qwen2.5-7B for escalated requests
→ direct answer or human HR review
```

Baseline: every request goes to Qwen2.5-7B.

## Course checklist

- 8–14 pages, excluding references and appendix.
- 150–200-word abstract.
- At least 12 peer-reviewed references.
- Consistent IEEE or ACM citations.
- Methodology documents data, split, prompt, models, quantisation, rules,
  thresholds, batch size, hardware, and evaluation.
- Results include quantitative comparison and confidence intervals where
  applicable.
- Limitations section is explicit.
- Every figure has a caption and every table has a title.
- Code snippets appear only in the appendix.
- GitHub repository link appears in the paper header.

Do not report offline fixtures or development outputs as final test results.
