# Results

This folder contains no official metrics before the Google Colab run.

Expected generated test files:

- `test_small_model.jsonl` — Qwen2.5-3B output for every test request.
- `test_escalation_decisions.jsonl` — rule decision and reasons for every request.
- `test_baseline.jsonl` — uniform Qwen2.5-7B baseline output.
- `test_tiered.jsonl` — final 3B → rules → optional 7B output.
- `metrics.csv` — paper-traceable metric table.
- `experiment_summary.json` — full metric and comparison summary.
- `error_analysis.jsonl` — incorrect cases for each evaluated system.
- `runtime_environment.json` — GPU, package, batch-size, model, and run metadata.
- `figures/accuracy_comparison.png`
- `figures/escalation_performance.png`
- `figures/efficiency_comparison.png`
- `figures/gpu_time_comparison.png`

Only files marked `official_run: true` and produced from all 45 locked test
records may be cited as final results. Development and offline-fixture results
must not be reported as held-out evidence.
