#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEV_IDS = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57}

# message, category, urgency, action, escalation_reason, language
EXAMPLES = [
    ("Where can I download my payslip?", "payroll", "low", "direct_answer", None, "en"),
    ("How do I report sick leave?", "leave", "low", "direct_answer", None, "en"),
    ("How do I check my leave balance?", "leave", "low", "direct_answer", None, "en"),
    ("How do I update my home address?", "personal_data", "low", "direct_answer", None, "en"),
    ("Where is the remote-work policy?", "policy", "low", "direct_answer", None, "en"),
    ("Where do I find the mandatory training?", "training", "low", "direct_answer", None, "en"),
    ("What documents should I bring on my first day?", "onboarding", "low", "direct_answer", None, "en"),
    ("Where do I return my laptop?", "offboarding", "low", "direct_answer", None, "en"),
    ("Where can I find information about the pension plan?", "benefits", "low", "direct_answer", None, "en"),
    ("How do I view my benefits portal?", "benefits", "low", "direct_answer", None, "en"),
    ("Where do I find my payslip? It is urgent today.", "payroll", "high", "direct_answer", None, "en"),
    ("Where do I submit a sick note tomorrow?", "leave", "medium", "direct_answer", None, "en"),
    ("Where can I see my remaining vacation days?", "leave", "low", "direct_answer", None, "en"),
    ("Where can I change my bank details today?", "personal_data", "high", "direct_answer", None, "en"),
    ("Where can I read the travel policy?", "policy", "low", "direct_answer", None, "en"),
    ("How do I access the compliance course?", "training", "low", "direct_answer", None, "en"),
    ("Where is the onboarding checklist?", "onboarding", "low", "direct_answer", None, "en"),
    ("How do I return company equipment?", "offboarding", "low", "direct_answer", None, "en"),
    ("How do I update my address this week?", "personal_data", "medium", "direct_answer", None, "en"),
    ("How do I access my payslip?", "payroll", "low", "direct_answer", None, "en"),
    ("My overtime from June is missing from this month's salary. Can payroll check it before Friday?", "payroll", "medium", "direct_answer", None, "en"),
    ("I booked three holiday days but the portal still shows the old balance.", "leave", "medium", "direct_answer", None, "en"),
    ("My public transport allowance disappeared after I changed departments.", "benefits", "medium", "direct_answer", None, "en"),
    ("I start next Monday and have not received login instructions or an office location.", "onboarding", "medium", "direct_answer", None, "en"),
    ("My final working day is next week. Please confirm when unused leave will be paid.", "offboarding", "medium", "direct_answer", None, "en"),
    ("Could someone explain whether the weekend hours in my new contract are optional?", "contract", "low", "direct_answer", None, "en"),
    ("I need a copy of the expense rule for client dinners because my claim was returned.", "policy", "low", "direct_answer", None, "en"),
    ("The security course shows incomplete even though I passed it yesterday.", "training", "medium", "direct_answer", None, "en"),
    ("My surname changed and I need it corrected on payroll and the employee directory.", "personal_data", "medium", "direct_answer", None, "en"),
    ("Can unused annual leave be carried into January if my manager approves it?", "leave", "low", "direct_answer", None, "en"),
    ("I received two different net salary amounts in the portal and bank transfer.", "payroll", "medium", "direct_answer", None, "en"),
    ("Does the company health insurance also cover dependants, and when can I enrol them?", "benefits", "low", "direct_answer", None, "en"),
    ("Bitte bestätigt, ob ich für meinen ersten Arbeitstag noch ein Führungszeugnis brauche.", "onboarding", "medium", "direct_answer", None, "de"),
    ("I resigned yesterday. Who confirms my remaining notice period and final handover date?", "offboarding", "medium", "direct_answer", None, "en"),
    ("My fixed-term agreement ends in six weeks. When should the renewal decision be communicated?", "contract", "medium", "direct_answer", None, "en"),
    ("The flex-time document and my team's instructions give different core working hours.", "policy", "medium", "direct_answer", None, "en"),
    ("My manager asked me to complete a course, but the learning portal says I am not eligible.", "training", "medium", "direct_answer", None, "en"),
    ("Please correct my emergency contact; the portal saves the old phone number.", "personal_data", "medium", "direct_answer", None, "en"),
    ("I requested parental leave two weeks ago and need to know whether the documents were received.", "leave", "medium", "direct_answer", None, "en"),
    ("The reimbursement was approved but has not appeared in this month's payment.", "payroll", "medium", "direct_answer", None, "en"),
    ("Kann ich meine betriebliche Altersvorsorge ändern, nachdem die Anmeldefrist abgelaufen ist?", "benefits", "low", "direct_answer", None, "de"),
    ("I begin remotely next week, but I only received the office-based onboarding schedule.", "onboarding", "medium", "direct_answer", None, "en"),
    ("Do I need to return my access card before or on my final day?", "offboarding", "low", "direct_answer", None, "en"),
    ("My offer letter says 35 hours while the contract draft says 40. Which document should be corrected?", "contract", "medium", "direct_answer", None, "en"),
    ("The leave policy was updated, but the German translation still shows the previous notice period.", "policy", "low", "direct_answer", None, "en"),
    ("My manager keeps making sexual comments and I need HR to intervene.", "workplace_issue", "medium", "human_review", "harassment or bullying allegation", "en"),
    ("I believe I was denied promotion because of my religion.", "workplace_issue", "medium", "human_review", "possible discrimination", "en"),
    ("A colleague threatened to hit me before my shift today. I feel unsafe.", "workplace_issue", "high", "human_review", "possible immediate safety risk", "en"),
    ("I need different working hours because of a medical condition. What proof must I send?", "contract", "medium", "human_review", "sensitive health or accommodation matter", "en"),
    ("My lawyer says the overtime clause may be illegal and I want the company to respond.", "contract", "medium", "human_review", "possible legal dispute", "en"),
    ("I was fired unfairly and want to contest my termination immediately.", "offboarding", "high", "human_review", "dismissal dispute", "en"),
    ("Send me my coworker's salary details because we have the same role.", "personal_data", "medium", "human_review", "request for another employee's confidential data", "en"),
    ("My supervisor repeatedly mocks my accent in team meetings.", "workplace_issue", "medium", "human_review", "harassment or bullying allegation", "en"),
    ("I am pregnant and need to discuss changes to duties and working hours.", "contract", "medium", "human_review", "sensitive health or accommodation matter", "en"),
    ("If this is not fixed, I will take legal action against the company.", "contract", "medium", "human_review", "possible legal dispute", "en"),
    ("Can you show me another employee's medical file? I am covering their work.", "personal_data", "medium", "human_review", "request for another employee's confidential data", "en"),
    ("Mein Vorgesetzter macht rassistische Witze über mich. Ich möchte das melden.", "workplace_issue", "medium", "human_review", "possible discrimination", "de"),
    ("I need an accommodation for my disability, but I do not want my manager to see my diagnosis.", "contract", "medium", "human_review", "sensitive health or accommodation matter", "en"),
    ("Help", "other", "low", "human_review", "insufficient information", "en"),
    ("My manager touched me inappropriately at the work event yesterday.", "workplace_issue", "medium", "human_review", "harassment or bullying allegation", "en"),
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for index, example in enumerate(EXAMPLES, start=1):
        message, category, urgency, action, reason, language = example
        escalate_to_large_model = action == "human_review" or urgency == "high"
        records.append(
            {
                "request_id": f"HR{index:03d}",
                "message": message,
                "language": language,
                "split": "dev" if index in DEV_IDS else "test",
                "gold_category": category,
                "gold_urgency": urgency,
                "gold_action": action,
                "gold_escalate_to_large_model": escalate_to_large_model,
                "gold_escalation_reason": reason,
                "source_type": "synthetic_controlled",
                "annotation_status": "gold_v1_single_reviewer",
            }
        )

    jsonl_path = DATA_DIR / "hr_requests.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = DATA_DIR / "hr_requests.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    summary = {
        "records": len(records),
        "dev": sum(record["split"] == "dev" for record in records),
        "test": sum(record["split"] == "test" for record in records),
        "escalate_to_large_model": {
            "true": sum(record["gold_escalate_to_large_model"] for record in records),
            "false": sum(not record["gold_escalate_to_large_model"] for record in records),
        },
        "human_review_actions": sum(
            record["gold_action"] == "human_review" for record in records
        ),
        "sha256_hr_requests_jsonl": checksum,
    }
    (DATA_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
