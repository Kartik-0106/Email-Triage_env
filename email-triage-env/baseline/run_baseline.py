"""
Baseline inference script for EmailTriageEnv.
Uses OpenAI API with deterministic settings for reproducibility.

Usage:
    # Set your API key first:
    export OPENAI_API_KEY=sk-...

    # Run all tasks:
    python baseline/run_baseline.py

    # Run specific task(s):
    python baseline/run_baseline.py --tasks triage_easy triage_medium

    # Custom seed:
    python baseline/run_baseline.py --seed 42
"""

import os
import sys
import json
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import Action

TASK_IDS = ["triage_easy", "triage_medium", "triage_hard"]
SEED = 42

SYSTEM_PROMPT = """You are an expert email triage assistant working in a corporate environment.
You will receive emails one at a time and must classify each one precisely.

For EACH email, respond ONLY with a valid JSON object. No explanation, no markdown, just raw JSON.

JSON Schema (include only relevant fields for the task):
{
  "email_id": "<exact id from the email>",
  "label": "urgent" OR "not_urgent",
  "queue": "sales" OR "support" OR "billing" OR "legal" OR "general",
  "priority": 1 (low) OR 2 (medium) OR 3 (high),
  "sla_hours": <integer hours to resolve>,
  "duplicate_of": "<email_id of duplicate>" OR null,
  "escalate": true OR false
}

Guidelines:
- "urgent": requires action within 24 hours; involves outages, security, legal, payment failure, or VIP escalation
- "not_urgent": informational, newsletters, low-stakes requests, can wait 48+ hours
- Queue routing: sales=leads/renewals, support=technical/product, billing=invoices/payments, legal=compliance/contracts, general=everything else
- Priority 3=high/critical, 2=normal business, 1=low/informational
- SLA hours: estimate realistic resolution time (1=critical, 4-8=urgent, 24-72=normal, 168=low)
- duplicate_of: if this email covers same issue as a previous email in this session, reference its ID
- escalate: true only for C-suite visibility required, legal risk, or multi-day outages
"""


def build_user_message(obs) -> str:
    e = obs.email
    ctx = ""
    if obs.context_window:
        ctx = "\n\nRecent emails for context:\n"
        for prev in obs.context_window[-2:]:
            ctx += f"  [{prev.id}] {prev.subject}\n"
    return (
        f"Email ID: {e.id}\n"
        f"From: {e.sender}\n"
        f"Timestamp: {e.timestamp}\n"
        f"Subject: {e.subject}\n"
        f"Body:\n{e.body}\n"
        f"Position: {obs.inbox_position + 1} of {obs.total_emails}"
        f"{ctx}"
    )


def run_task(client, task_id: str, seed: int = SEED, verbose: bool = True) -> dict:
    env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    history = []
    episode_score = None
    step_log = []

    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()}  |  Emails: {obs.total_emails}  |  Seed: {seed}")
    print(f"{'='*60}")

    while obs is not None:
        user_msg = build_user_message(obs)
        history.append({"role": "user", "content": user_msg})

        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                temperature=0.0,
                seed=seed,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [API ERROR] {e}")
            raw = json.dumps({"email_id": obs.email.id, "label": "not_urgent"})

        history.append({"role": "assistant", "content": raw})

        # Parse action
        try:
            # Strip markdown code fences if present
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            action = Action(**parsed)
        except Exception as e:
            if verbose:
                print(f"  [PARSE ERROR] {e} | Raw: {raw[:80]}")
            action = Action(email_id=obs.email.id, label="not_urgent")

        result = env.step(action)

        if verbose:
            scores = result.info.get("partial_scores", {})
            scores_str = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
            print(
                f"  [{result.info['step']:02d}/{result.info['total_steps']}] "
                f"Email: {obs.email.subject[:40]:<40} "
                f"→ label={action.label:<10} "
                f"reward={result.reward:.2f} "
                f"({scores_str})"
            )

        step_log.append({
            "step": result.info["step"],
            "email_id": obs.email.id,
            "subject": obs.email.subject,
            "action": action.model_dump(),
            "reward": result.reward,
            "partial_scores": result.info.get("partial_scores", {}),
            "penalties": result.info.get("penalties", {}),
        })

        obs = result.observation
        if result.done:
            episode_score = result.info.get("episode_score", 0.0)

    print(f"\n  ✅ Episode Score: {episode_score:.4f}")
    return {
        "task_id": task_id,
        "seed": seed,
        "episode_score": round(episode_score, 4),
        "step_log": step_log,
        "final_state": env.state(),
    }


def main():
    parser = argparse.ArgumentParser(description="EmailTriageEnv Baseline (OpenAI GPT-4o)")
    parser.add_argument("--tasks", nargs="+", default=TASK_IDS, choices=TASK_IDS,
                        help="Tasks to run (default: all)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output file for results (default: baseline_results.json)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        print("❌ ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    print("\n🚀 EmailTriageEnv Baseline Evaluation")
    print(f"   Model: gpt-4o | Seed: {args.seed} | Tasks: {args.tasks}")

    all_results = {}
    start_time = time.time()

    for task_id in args.tasks:
        result = run_task(client, task_id, seed=args.seed, verbose=not args.quiet)
        all_results[task_id] = result

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for task_id, result in all_results.items():
        score = result["episode_score"]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<20} [{bar}] {score:.4f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {args.output}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
