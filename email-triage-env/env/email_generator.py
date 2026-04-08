import random
import hashlib
from datetime import datetime, timedelta
from env.models import Email

TEMPLATES = {
    "triage_easy": [
        {
            "label": "urgent",
            "subjects": [
                "URGENT: Server down in production",
                "Critical security breach detected",
                "Payment processing failed - action required",
                "URGENT: Database corruption alert",
                "Critical: Customer data exposure risk",
            ],
            "bodies": [
                "Our production server has gone down. Customers cannot access the service. Immediate action needed.",
                "We have detected a critical security breach in our systems. All hands on deck.",
                "Payment processing has failed for the last 2 hours affecting hundreds of customers. Please fix ASAP.",
                "Database corruption detected on primary node. Backups may be affected. Urgent response required.",
                "Potential customer data exposure detected. Legal and security teams need to respond immediately.",
            ],
            "queue": "support",
            "priority": 3,
            "sla_hours": 1,
            "escalate": True,
        },
        {
            "label": "not_urgent",
            "subjects": [
                "Monthly team newsletter",
                "Team lunch next Friday",
                "FYI: Updated office policy",
                "Reminder: Quarterly review next month",
                "New blog post published",
            ],
            "bodies": [
                "Here is your monthly team digest with updates from all departments.",
                "We are organizing a team lunch next Friday. Please RSVP by Wednesday.",
                "Just a heads up that the office parking policy has been updated. See attached.",
                "A reminder that quarterly reviews are scheduled for next month. No action needed now.",
                "Our latest blog post on industry trends is now live. Check it out when you have time.",
            ],
            "queue": "general",
            "priority": 1,
            "sla_hours": 72,
            "escalate": False,
        },
    ],
    "triage_medium": [
        {
            "label": "urgent",
            "subjects": ["Contract renewal deadline tomorrow", "Invoice overdue - account suspension"],
            "bodies": [
                "The client contract expires tomorrow and we have not received a renewal signature.",
                "Your account invoice is 30 days overdue. Service will be suspended within 24 hours.",
            ],
            "queue": "billing",
            "priority": 3,
            "sla_hours": 4,
            "escalate": True,
        },
        {
            "label": "urgent",
            "subjects": ["Legal notice received", "Compliance audit request"],
            "bodies": [
                "We have received a legal notice requiring a response within 48 hours.",
                "Regulatory body has requested compliance documentation by end of week.",
            ],
            "queue": "legal",
            "priority": 3,
            "sla_hours": 8,
            "escalate": True,
        },
        {
            "label": "urgent",
            "subjects": ["Enterprise client escalation", "VIP customer complaint"],
            "bodies": [
                "Our largest enterprise client is threatening to cancel their contract due to ongoing issues.",
                "A VIP customer has escalated their complaint to executive level. Immediate response needed.",
            ],
            "queue": "sales",
            "priority": 3,
            "sla_hours": 2,
            "escalate": True,
        },
        {
            "label": "not_urgent",
            "subjects": ["Product feedback from customer", "Feature request submission"],
            "bodies": [
                "A customer has submitted detailed product feedback through our portal.",
                "New feature request received via the customer portal. Please review when possible.",
            ],
            "queue": "support",
            "priority": 2,
            "sla_hours": 48,
            "escalate": False,
        },
        {
            "label": "not_urgent",
            "subjects": ["Partnership inquiry", "New sales lead from website"],
            "bodies": [
                "A company has reached out about a potential partnership opportunity.",
                "New inbound lead from the website contact form. Low urgency, follow up this week.",
            ],
            "queue": "sales",
            "priority": 1,
            "sla_hours": 72,
            "escalate": False,
        },
    ],
    "triage_hard": [
        {
            "label": "urgent",
            "subjects": ["URGENT: Data breach notification required by law", "Critical system failure - SLA breach"],
            "bodies": [
                "Under GDPR Article 33, we must notify authorities within 72 hours of breach discovery.",
                "Core system has failed. Current downtime breaches our SLA with enterprise clients.",
            ],
            "queue": "legal",
            "priority": 3,
            "sla_hours": 2,
            "escalate": True,
            "duplicate_of": None,
        },
        {
            "label": "urgent",
            "subjects": ["Follow up: Server outage from this morning", "Re: Payment failure - same issue"],
            "bodies": [
                "This is a follow-up to the server outage reported earlier today. Still not resolved.",
                "Same payment failure as reported 2 hours ago. No fix yet. Escalating.",
            ],
            "queue": "support",
            "priority": 3,
            "sla_hours": 1,
            "escalate": True,
            "is_duplicate": True,
        },
        {
            "label": "not_urgent",
            "subjects": ["Routine system health report", "Weekly analytics summary"],
            "bodies": [
                "Automated weekly system health check report. All systems nominal.",
                "Your weekly analytics dashboard summary is ready for review.",
            ],
            "queue": "general",
            "priority": 1,
            "sla_hours": 168,
            "escalate": False,
            "duplicate_of": None,
        },
    ],
}


class EmailGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_episode(self, task_id: str, n: int, seed: int):
        rng = random.Random(seed)
        emails = []
        ground_truths = []

        task_key = task_id if task_id in TEMPLATES else "triage_easy"
        templates = TEMPLATES[task_key]

        duplicate_candidates = {}

        for i in range(n):
            tpl = rng.choice(templates)
            subject = rng.choice(tpl["subjects"])
            body = rng.choice(tpl["bodies"])
            email_id = hashlib.md5(f"{seed}-{i}".encode()).hexdigest()[:8]
            ts = (datetime(2024, 1, 1) + timedelta(hours=i * 3)).isoformat()

            emails.append(Email(
                id=email_id,
                subject=subject,
                body=body,
                sender=f"user{i}@example.com",
                timestamp=ts,
            ))

            is_dup = tpl.get("is_duplicate", False)
            dup_of = None
            if is_dup and duplicate_candidates:
                dup_of = rng.choice(list(duplicate_candidates.keys()))

            ground_truths.append({
                "label": tpl["label"],
                "queue": tpl.get("queue"),
                "priority": tpl.get("priority"),
                "sla_hours": tpl.get("sla_hours"),
                "duplicate_of": dup_of,
                "escalate": tpl.get("escalate", False),
            })

            if not is_dup:
                duplicate_candidates[email_id] = tpl["label"]

        return emails, ground_truths
