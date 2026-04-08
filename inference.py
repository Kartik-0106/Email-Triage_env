"""
inference.py — OpenEnv baseline inference script (root of project).

Required env vars:
  API_BASE_URL   The base URL for the OpenAI-compatible LLM API endpoint.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key (used as the API key).

Optional:
  OPENAI_API_KEY Fallback API key if HF_TOKEN is not set.

Usage:
  API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o HF_TOKEN=sk-... python inference.py
  python inference.py --tasks triage_easy triage_medium triage_hard --seed 42
  python inference.py --quiet
"""

import os
import sys
import json
import argparse
import time

from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority

# ── Credentials & config from env vars (required by spec) ─────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN")     or os.environ.get("OPENAI_API_KEY")

TASK_IDS = ["triage_easy", "triage_medium", "triage_hard"]
SEED     = 42

SYSTEM_PROMPT = """You are an expert email triage assistant in a corporate environment.
For each email you receive, respond ONLY with a valid JSON object — no prose, no markdown fences.

Required JSON schema:
{
  "email_id":    "<exact id from the email — do not change it>",
  "label":       "urgent" | "not_urgent",
  "queue":       "sales" | "support" | "billing" | "legal" | "general",
  "priority":    1 (low) | 2 (medium) | 3 (high),
  "sla_hours":   <integer — estimated hours to resolve>,
  "duplicate_of": "<email_id of the earlier email this duplicates>" | null,
  "escalate":    true | false
}

Guidelines:
- urgent: needs action within 24h (outages, breaches, legal, payment failure, VIP escalation)
- not_urgent: informational, newsletters, low-stakes (can wait 48h+)
- queue: sales=leads/renewals, support=technical, billing=invoices, legal=compliance, general=other
- priority 3=critical, 2=normal, 1=low
- escalate true only when C-suite visibility is genuinely required
- duplicate_of: set only if a previous email in this session covers the same issue"""


def _build_user_msg(obs) -> str:
    e = obs.email
    ctx = ""
    if obs.context_window:
        ctx = "\n\nRecent context:\n" + "\n".join(
            f"  [{p.id}] {p.subject}" for p in obs.context_window[-2:]
        )
    return (
        f"Email ID: {e.id}\nFrom: {e.sender}\nTimestamp: {e.timestamp}\n"
        f"Subject: {e.subject}\nBody:\n{e.body}\n"
        f"Position: {obs.inbox_position + 1}/{obs.total_emails}{ctx}"
    )


def run_task(client, task_id: str, seed: int = SEED, verbose: bool = True) -> dict:
    env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    history = []
    episode_score = None
    step_log = []

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Task: {task_id.upper()}  |  Emails: {obs.total_emails}  |  Seed: {seed}")
        print(f"{'='*62}")

    while obs is not None:
        user_msg = _build_user_msg(obs)
        history.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                temperature=0.0,
                seed=seed,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            if verbose:
                print(f"  [API ERROR] {exc}")
            raw = json.dumps({"email_id": obs.email.id, "label": "not_urgent"})

        history.append({"role": "assistant", "content": raw})

        # Parse — strip accidental markdown fences
        clean = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            parsed = json.loads(clean)
            action = Action(
                email_id   = parsed.get("email_id", obs.email.id),
                label      = parsed.get("label", "not_urgent"),
                queue      = Queue(parsed["queue"]) if parsed.get("queue") else None,
                priority   = Priority(int(parsed["priority"])) if parsed.get("priority") else None,
                sla_hours  = parsed.get("sla_hours"),
                duplicate_of = parsed.get("duplicate_of"),
                escalate   = parsed.get("escalate", False),
            )
        except Exception as parse_err:
            if verbose:
                print(f"  [PARSE WARN] {parse_err} | raw={raw[:60]!r}")
            action = Action(email_id=obs.email.id, label="not_urgent")

        result = env.step(action)
        scores = result.info.get("partial_scores", {})

        if verbose:
            sc_str = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
            print(
                f"  [{result.info['step']:02d}/{result.info['total_steps']}] "
                f"{obs.email.subject[:36]:<36} "
                f"→ {action.label:<10} r={result.reward:.2f} ({sc_str})"
            )

        step_log.append({
            "step":          result.info["step"],
            "email_id":      obs.email.id,
            "subject":       obs.email.subject,
            "action":        action.model_dump() if hasattr(action, "model_dump") else str(action),
            "reward":        result.reward,
            "partial_scores": scores,
            "penalties":     result.info.get("penalties", {}),
        })

        obs = result.observation
        if result.done:
            episode_score = result.info.get("episode_score", 0.0)

    if verbose:
        print(f"\n  Episode score: {episode_score:.4f}")

    return {
        "task_id":       task_id,
        "model":         MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "seed":          seed,
        "episode_score": round(episode_score, 4),
        "step_log":      step_log,
        "final_state":   env.state(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="EmailTriageEnv inference script — reads API_BASE_URL, MODEL_NAME, HF_TOKEN"
    )
    parser.add_argument("--tasks",  nargs="+", default=TASK_IDS, choices=TASK_IDS)
    parser.add_argument("--seed",   type=int,  default=SEED)
    parser.add_argument("--quiet",  action="store_true")
    parser.add_argument("--output", default="inference_results.json")
    args = parser.parse_args()

    # ── Validate required credentials ─────────────────────────────────────────
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable not set.", file=sys.stderr)
        print("  export HF_TOKEN=your-api-key", file=sys.stderr)
        sys.exit(1)

    print(f"API_BASE_URL : {API_BASE_URL}")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"HF_TOKEN     : {'*' * 8}{HF_TOKEN[-4:] if len(HF_TOKEN) > 4 else '****'}")

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    # ── OpenAI client using spec-required variables ────────────────────────────
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    print(f"\nRunning {len(args.tasks)} task(s) with seed={args.seed} ...")
    start = time.time()

    all_results = {}
    for task_id in args.tasks:
        all_results[task_id] = run_task(client, task_id, seed=args.seed, verbose=not args.quiet)

    elapsed = time.time() - start

    print(f"\n{'='*62}")
    print("  INFERENCE RESULTS SUMMARY")
    print(f"{'='*62}")
    for task_id, r in all_results.items():
        score = r["episode_score"]
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<22} [{bar}] {score:.4f}")
    print(f"\n  Elapsed : {elapsed:.1f}s")
    print(f"  Saved   : {args.output}")

    with open(args.output, "w") as fh:
        json.dump(all_results, fh, indent=2)


if __name__ == "__main__":
    main()
