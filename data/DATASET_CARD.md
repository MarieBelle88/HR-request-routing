# Dataset Card: Synthetic HR Service Requests v1

## Purpose

This controlled dataset supports a small comparison of uniform language-model
processing and tiered routing for HR service requests. Public ticket datasets
were not used as the primary evidence because they generally lack the required
gold labels for safe escalation, urgency, and the intended processing tier.

## Composition

- 60 synthetic requests written for this experiment.
- Fixed split: 15 development and 45 test records.
- Labels: category, urgency, action, ideal route, and escalation reason.
- Categories: leave, payroll, benefits, onboarding, offboarding, contract,
  workplace issue, policy, training, personal data, and other.
- Languages: mainly English, with several German and code-switched stress cases.

## Provenance and privacy

The records are newly authored examples. They do not copy private employee
messages, company records, or personal data. Names and situations are fictional.

## Intended use

The dataset is intended only for a course experiment and software validation.
It is not suitable for making real employment decisions or evaluating fairness
across demographic groups.

## Limitations

Synthetic examples are cleaner and less diverse than real HR communication.
The dataset is small, was labelled by one researcher, and shares wording
patterns with the routing rules. This may favour the tiered system. A real
deployment would require independent annotation, organisational policies,
privacy review, multilingual validation, and monitoring.

