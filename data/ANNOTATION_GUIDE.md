# Annotation Guide

Each request has one gold label for every evaluation field.

## Category

- `leave`: annual, parental, compassionate, or sick-leave processes.
- `payroll`: salary, payslip, tax, reimbursement, or payment issues.
- `benefits`: pension, insurance, allowances, and benefit enrolment.
- `onboarding`: starting work, accounts, documents, and orientation.
- `offboarding`: resignation, final day, equipment return, and final pay.
- `contract`: contract terms, working hours, renewals, and amendments.
- `workplace_issue`: complaints, harassment, discrimination, conflict, safety.
- `policy`: general questions about company rules and procedures.
- `training`: required training, course access, and learning records.
- `personal_data`: access to or correction of employee data.
- `other`: not enough information or no supported category.

## Urgency

- `high`: explicitly urgent, an immediate safety issue, or a deadline within
  24 hours.
- `medium`: time-sensitive but not immediate, or an active case requiring action.
- `low`: general information or no stated time pressure.

## Final action

- `direct_answer`: the request can continue through the automated service flow.
- `human_review`: a qualified HR person must handle the request because it is
  sensitive, legally consequential, confidential, threatening,
  discriminatory, medical/disability-related, or materially unclear.

## Escalation from 3B to 7B

`gold_escalate_to_large_model` is `true` when the request is high urgency,
sensitive, or materially ambiguous. It is independent of `gold_action`.

The implemented rules also escalate when the 3B result fails schema validation
or reports confidence below the threshold fixed in `config/experiment.json`.
Those dynamic failures cannot be known from the message-only gold label, so they
may appear as false-positive escalations during evaluation.

## Split policy

Development examples may be inspected while checking the prompt, parser, and
confidence threshold. Test labels are locked for final evaluation and must not
be used to tune the system after the official test run starts.
