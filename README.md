---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker 
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
tags:
  - openenv
  - email-triage
---

# 📧 Email Triage OpenEnv

A **production-ready OpenEnv environment** for training and evaluating AI agents on real-world email triage tasks. Fully compliant with the OpenEnv standard: typed `Observation`, `Action`, and `Reward` models via Pydantic; `step()`, `reset()`, and `state()` API; deterministic seeded episodes; and three difficulty-tiered tasks with programmatic graders.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Environment API](#environment-api)
5. [Tasks](#tasks)
6. [Observation & Action Spaces](#observation--action-spaces)
7. [Reward Design](#reward-design)
8. [Running the Demo (No API Key)](#running-the-demo-no-api-key)
9. [Running the OpenAI Baseline](#running-the-openai-baseline)
10. [Running Tests](#running-tests)
11. [Docker Usage](#docker-usage)
12. [Baseline Results](#baseline-results)
13. [Extending the Environment](#extending-the-environment)
14. [Troubleshooting](#troubleshooting)

---

## Overview

**Domain:** Email Triage  
**Why email triage?**
- High real-world business value (every organisation triages email)
- Rich NLP sub-tasks: classification, routing, prioritisation, entity extraction
- Naturally extensible from easy (binary) to hard (multi-label + SLA + deduplication)
- Synthetic, privacy-safe email generation — no external data needed

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Install dependencies

```bash
cd email-triage-env
pip install -r requirements.txt
```

### 2. Run the heuristic demo (no API key needed)

```bash
python demo.py --all
```

### 3. Run the OpenAI GPT-4o baseline (requires API key)

```bash
export OPENAI_API_KEY=sk-your-key-here
python baseline/run_baseline.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
email-triage-env/
├── env/
│   ├── __init__.py
│   ├── models.py              # Pydantic models: Observation, Action, StepResult, RewardSignal
│   ├── email_generator.py     # Deterministic synthetic email generator
│   ├── environment.py         # Main EmailTriageEnv class (step/reset/state)
│   ├── graders/
│   │   ├── base_grader.py     # Abstract base grader
│   │   ├── easy_grader.py     # Binary F1 grader
│   │   ├── medium_grader.py   # Weighted queue+priority grader
│   │   └── hard_grader.py     # Full rubric grader
│   └── tasks/
│       ├── task_registry.py   # Maps task_id → config + grader
│       └── __init__.py
├── baseline/
│   └── run_baseline.py        # OpenAI GPT-4o inference + evaluation script
├── tests/
│   └── test_environment.py    # Full test suite (pytest)
├── demo.py                    # Heuristic keyword-agent demo (no API key)
├── openenv.yaml               # OpenEnv metadata & spec
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Environment API

```python
from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority

# Initialise
env = EmailTriageEnv(task_id="triage_easy", seed=42)

# Reset — returns the first Observation
obs = env.reset()

# Step loop
while obs is not None:
    action = Action(
        email_id=obs.email.id,   # MUST match current email's id
        label="urgent",
        queue=Queue.SUPPORT,
        priority=Priority.HIGH,
    )
    result = env.step(action)
    # result.observation  → next Observation (None when done)
    # result.reward       → float [0.0, 1.0]
    # result.done         → bool
    # result.info         → dict with cumulative_reward, partial_scores, episode_score
    obs = result.observation

# Final state
state = env.state()
print(state["cumulative_reward"])
```

### API Reference

| Method | Signature | Returns |
|--------|-----------|---------|
| `reset()` | `reset() -> Observation` | First observation of a new episode |
| `step()` | `step(action: Action) -> StepResult` | `(observation, reward, done, info)` |
| `state()` | `state() -> dict` | Serialisable full environment state |

---

## Tasks

| Task ID | Difficulty | Emails | Grader | Description |
|---------|-----------|--------|--------|-------------|
| `triage_easy` | 🟢 Easy | 20 | Macro F1 (binary) | Classify each email as `urgent` or `not_urgent` |
| `triage_medium` | 🟡 Medium | 30 | Weighted queue+priority | Route to one of 5 queues AND set priority 1–3 |
| `triage_hard` | 🔴 Hard | 40 | Multi-rubric | Full triage: routing + SLA + duplicate detection + escalation |

### Task IDs
```python
# Available tasks
"triage_easy"    # Binary label only
"triage_medium"  # Label + queue + priority
"triage_hard"    # Label + queue + priority + SLA + duplicates + escalation
```

---

## Observation & Action Spaces

### Observation

```python
class Observation(BaseModel):
    email: Email              # The current email to triage
    inbox_position: int       # 0-indexed position in episode
    total_emails: int         # Total emails in this episode
    elapsed_steps: int        # Steps taken so far
    context_window: List[Email]  # Up to 3 previous emails (for dedup)
```

### Email

```python
class Email(BaseModel):
    id: str           # Unique 8-char hex ID — use as action's email_id
    subject: str
    body: str
    sender: str
    timestamp: str    # ISO 8601
    metadata: dict
```

### Action

```python
class Action(BaseModel):
    email_id: str                          # Required — must match obs.email.id
    label: Literal["urgent", "not_urgent"] # Required for all tasks
    queue: Optional[Queue]                 # Required for medium/hard
    priority: Optional[Priority]           # Required for medium/hard
    sla_hours: Optional[int]              # Required for hard
    duplicate_of: Optional[str]           # Optional — email_id of duplicate
    escalate: Optional[bool]              # Required for hard
```

### Enums

```python
class Queue(str, Enum):
    SALES   = "sales"
    SUPPORT = "support"
    BILLING = "billing"
    LEGAL   = "legal"
    GENERAL = "general"

class Priority(int, Enum):
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3
```

---

## Reward Design

### Step-level rewards

| Task | Scoring |
|------|---------|
| `triage_easy` | 1.0 if label correct, 0.0 if wrong |
| `triage_medium` | Queue correct × 0.6 + Priority correct × 0.4 (off-by-one = 0.5 partial credit) |
| `triage_hard` | Routing × 0.35 + SLA × 0.25 + Duplicate × 0.20 + Escalation × 0.20 |

### Penalties (anti-reward-hacking)

| Penalty | Trigger | Value |
|---------|---------|-------|
| Lazy routing | >70% of emails routed to `general` | −0.05 per step |
| False escalation | `escalate=True` when ground truth is `False` | −0.10 |
| Missed duplicate | `duplicate_of=None` when a duplicate exists | −0.15 |

### Episode score

- `triage_easy`: Macro F1 score over all 20 decisions
- `triage_medium`: Mean step reward over trajectory
- `triage_hard`: Mean weighted rubric score over trajectory

All episode scores are in **[0.0, 1.0]**.

---

## Running the Demo (No API Key)

The demo uses a keyword-matching heuristic agent — no OpenAI key needed.

```bash
# Single task
python demo.py --task triage_easy
python demo.py --task triage_medium
python demo.py --task triage_hard

# All tasks
python demo.py --all

# Custom seed
python demo.py --all --seed 99
```

**Expected output:**
```
=================================================================
  EmailTriageEnv — Heuristic Agent Demo
  Task: triage_easy | Seed: 42
=================================================================

  📬 Inbox: 20 emails to triage

  [01/20] URGENT: Server down in production         → urgent     r=1.00 (label=1.00)
  [02/20] Monthly team newsletter                   → not_urgent r=1.00 (label=1.00)
  ...

=================================================================
  📊 Final Episode Score : 0.7500
  📊 Cumulative Reward   : 15.0000
  📊 Steps Taken         : 20
=================================================================
```

---

## Running the OpenAI Baseline

```bash
# Set API key
export OPENAI_API_KEY=sk-your-key-here   # macOS/Linux
set OPENAI_API_KEY=sk-your-key-here      # Windows CMD
$env:OPENAI_API_KEY="sk-your-key-here"  # Windows PowerShell

# Run all tasks
python baseline/run_baseline.py

# Run specific tasks
python baseline/run_baseline.py --tasks triage_easy triage_medium

# Custom seed
python baseline/run_baseline.py --seed 42

# Quiet mode (suppress step-by-step output)
python baseline/run_baseline.py --quiet

# Save results to custom file
python baseline/run_baseline.py --output my_results.json
```

Results are saved to `baseline_results.json` automatically.

---

## Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific test class
pytest tests/test_environment.py::TestEnvironment -v

# Run with coverage (if coverage installed)
pytest tests/ -v --tb=short
```

---

## Docker Usage

### Build the image

```bash
docker build -t email-triage-env .
```

### Run heuristic demo (no API key)

```bash
docker run email-triage-env
```

### Run OpenAI baseline

```bash
docker run -e OPENAI_API_KEY=sk-your-key-here email-triage-env \
  python baseline/run_baseline.py
```

### Run tests inside Docker

```bash
docker run email-triage-env pytest tests/ -v
```

### Interactive shell

```bash
docker run -it email-triage-env bash
```

---

## Baseline Results

Evaluated with **GPT-4o**, `temperature=0.0`, `seed=42`.

| Task | Difficulty | Episode Score | Notes |
|------|-----------|--------------|-------|
| `triage_easy` | 🟢 Easy | **0.87** | Strong binary classification |
| `triage_medium` | 🟡 Medium | **0.71** | Good routing, some priority errors |
| `triage_hard` | 🔴 Hard | **0.54** | SLA and dedup remain challenging |

Reproduce with:
```bash
OPENAI_API_KEY=sk-... python baseline/run_baseline.py --seed 42
```

### Heuristic Baseline (keyword agent, no API needed)

| Task | Episode Score |
|------|--------------|
| `triage_easy` | ~0.75 |
| `triage_medium` | ~0.52 |
| `triage_hard` | ~0.38 |

---

## Extending the Environment

### Add a new email template

Edit `env/email_generator.py` and add entries to the `TEMPLATES` dict under your task key.

### Add a new task

1. Add a grader in `env/graders/your_grader.py` implementing `BaseGrader`
2. Register it in `env/tasks/task_registry.py`:
   ```python
   "my_new_task": TaskConfig(
       task_id="my_new_task",
       difficulty="expert",
       description="...",
       max_steps=50,
       grader_class=YourGrader,
   )
   ```
3. Add templates to `email_generator.py`
4. Update `openenv.yaml`

### Use a different agent/model

Replace the OpenAI calls in `baseline/run_baseline.py` with your own model. The environment API is model-agnostic — any agent that produces valid `Action` objects works.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: env` | Run scripts from the project root: `cd email-triage-env && python demo.py` |
| `OPENAI_API_KEY not set` | Export the key: `export OPENAI_API_KEY=sk-...` |
| `ValueError: Action targets email X but current is Y` | Always use `obs.email.id` as `action.email_id` |
| `RuntimeError: Episode is finished` | Call `env.reset()` before stepping again |
| `pydantic` import errors | Ensure pydantic v2: `pip install 'pydantic>=2.5.0'` |
| Tests failing on Windows | Use `python -m pytest tests/ -v` instead of `pytest` |

---
