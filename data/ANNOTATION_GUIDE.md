# Annotation Guide

Each request has one gold label for each field.

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

- `high`: explicitly urgent, immediate safety issue, or deadline within 24 hours.
- `medium`: time-sensitive but not immediate, or an active case requiring action.
- `low`: general information or no stated time pressure.

## Action

- `direct_answer`: safe to classify or answer through the automated workflow.
- `human_review`: sensitive, legally consequential, confidential, threatening,
  discriminatory, medical/disability-related, or too unclear to handle safely.

## Ideal route

- `rules`: narrow, recognisable, low-risk FAQ or process request.
- `qwen`: ordinary natural-language request needing semantic classification.
- `human`: sensitive/high-risk or materially ambiguous request.

Development examples may be inspected while designing rules. Test labels are
held out for final evaluation and must not be used to tune the system after the
official run begins.

