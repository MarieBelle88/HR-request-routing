# Dataset Card: Synthetic HR Service Requests v2

## Purpose

This controlled dataset supports a proof-of-concept evaluation of the Topic 16
architecture: fast SLM, rule-based escalation, and a larger LLM. It is an
evaluation dataset, not a language-model training dataset.

Public HR ticket datasets were not used because suitable public records rarely
include both the processing-tier label and the final human-review label needed
for this experiment. The sourcing decision and synthetic construction are
therefore documented explicitly.

## Composition

- 60 newly written synthetic HR requests.
- Fixed split: 15 development and 45 held-out test records.
- Labels: category, urgency, final action, 3B-to-7B escalation, and escalation
  reason.
- Categories: leave, payroll, benefits, onboarding, offboarding, contract,
  workplace issue, policy, training, personal data, and other.
- Languages: mainly English with three German stress cases.
- Cases include routine requests, active casework, explicit urgency,
  harassment/discrimination, health accommodations, legal issues,
  confidentiality risks, and one materially ambiguous request.

## Gold-label distinction

`gold_escalate_to_large_model` indicates whether the fast 3B tier should be
checked by the 7B tier under the fixed routing policy. High-urgency, sensitive,
and insufficiently specified cases are positive.

`gold_action` indicates the final operational outcome. `human_review` is used
when a qualified HR person must handle the request. A request can therefore be
escalated to 7B and still end in human review; model escalation and human review
are not the same label.

## Provenance and privacy

The records were authored specifically for this study from common HR service
themes. They do not copy private employee messages, company records, or real
personal data. Names and situations are fictional.

## Intended use

The dataset is intended only for course research, reproducibility checks, and
software validation. It must not be used to make real employment decisions or
to claim production readiness.

## Limitations

The dataset is small, synthetic, and labelled by one researcher. Synthetic
messages are cleaner than real HR communication, and several sensitive cases
contain explicit keywords that can favour the escalation rules. The 45-record
test set also gives wide uncertainty: one error changes test accuracy by about
2.2 percentage points. A real deployment would require a larger independently
annotated dataset, organisational policy review, privacy and fairness analysis,
multilingual validation, and ongoing monitoring.
