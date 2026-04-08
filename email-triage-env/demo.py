"""
Quick demo: run a keyword-based heuristic agent (NO API key needed).
Shows the full step()/reset()/state() API in action.

Usage:
    python demo.py
    python demo.py --task triage_medium
    python demo.py --task triage_hard
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority


URGENT_KEYWORDS = [
    "urgent", "critical", "breach", "failed", "failure", "down", "outage",
    "expire", "overdue", "suspension", "legal notice", "compliance", "escalat",
    "vip", "corruption", "exposure",
]

QUEUE_KEYWORDS = {
    "billing": ["invoice", "payment", "billing", "overdue", "subscription", "charge"],
    "legal": ["legal", "compliance", "contract", "regulation", "gdpr", "audit", "notice"],
    "sales": ["sales", "renewal", "lead", "partnership", "enterprise", "client", "deal"],
    "support": ["server", "outage", "bug", "error", "crash", "system", "technical", "database"],
}


def heuristic_agent(obs) -> Action:
    """Simple keyword-matching heuristic agent."""
    email = obs.email
    text = (email.subject + " " + email.body).lower()

    # Determine label
    label = "not_urgent"
    for kw in URGENT_KEYWORDS:
        if kw in text:
            label = "urgent"
            break

    # Determine queue
    queue = Queue.GENERAL
    for q, keywords in QUEUE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            queue = Queue(q)
            break

    # Determine priority
    priority = Priority.HIGH if label == "urgent" else Priority.LOW

    # SLA estimate
    sla_map = {Priority.HIGH: 4, Priority.MEDIUM: 24, Priority.LOW: 72}
    sla_hours = sla_map[priority]

    # Escalation: escalate if critical keywords AND urgent
    escalate = label == "urgent" and any(kw in text for kw in ["critical", "breach", "legal", "outage", "corruption"])

    return Action(
        email_id=email.id,
        label=label,
        queue=queue,
        priority=priority,
        sla_hours=sla_hours,
        duplicate_of=None,
        escalate=escalate,
    )


def run_demo(task_id: str = "triage_easy", seed: int = 42):
    print(f"\n{'='*65}")
    print(f"  EmailTriageEnv — Heuristic Agent Demo")
    print(f"  Task: {task_id} | Seed: {seed}")
    print(f"{'='*65}")

    env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    total_reward = 0.0
    step_count = 0

    print(f"\n  📬 Inbox: {obs.total_emails} emails to triage\n")

    while obs is not None:
        action = heuristic_agent(obs)
        result = env.step(action)

        step_count += 1
        total_reward += result.reward

        scores = result.info.get("partial_scores", {})
        penalties = result.info.get("penalties", {})
        scores_str = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
        pen_str = f" ⚠ {penalties}" if any(v != 0 for v in penalties.values()) else ""

        print(
            f"  [{step_count:02d}/{obs.total_emails}] "
            f"{obs.email.subject[:38]:<38} "
            f"→ {action.label:<10} "
            f"r={result.reward:.2f} "
            f"({scores_str}){pen_str}"
        )

        obs = result.observation

        if result.done:
            episode_score = result.info.get("episode_score", 0.0)

    final = env.state()

    print(f"\n{'='*65}")
    print(f"  📊 Final Episode Score : {episode_score:.4f}")
    print(f"  📊 Cumulative Reward   : {final['cumulative_reward']:.4f}")
    print(f"  📊 Steps Taken         : {step_count}")
    print(f"{'='*65}")

    print(f"\n  Full state snapshot:")
    print(json.dumps({
        "task_id": final["task_id"],
        "seed": final["seed"],
        "done": final["done"],
        "cumulative_reward": final["cumulative_reward"],
        "episode_score": round(episode_score, 4),
    }, indent=4))

    return episode_score


def main():
    parser = argparse.ArgumentParser(description="EmailTriageEnv Heuristic Demo (no API key needed)")
    parser.add_argument("--task", default="triage_easy",
                        choices=["triage_easy", "triage_medium", "triage_hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all 3 tasks")
    args = parser.parse_args()

    if args.all:
        scores = {}
        for task in ["triage_easy", "triage_medium", "triage_hard"]:
            scores[task] = run_demo(task, args.seed)
        print(f"\n{'='*65}")
        print("  ALL TASKS SUMMARY (Heuristic Agent)")
        print(f"{'='*65}")
        for task, score in scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {task:<20} [{bar}] {score:.4f}")
    else:
        run_demo(args.task, args.seed)


if __name__ == "__main__":
    main()
