"""
FastAPI HTTP server exposing the OpenEnv-compliant interface.

Endpoints:
  POST /reset          → Observation
  POST /step           → StepResult (observation, reward, done, info)
  GET  /state          → current state dict
  GET  /tasks          → list available tasks
  GET  /health         → 200 OK liveness probe
  GET  /               → 200 OK (HF Space / automated ping)

Start:
  uvicorn server:app --host 0.0.0.0 --port 7860
"""
import os
import json
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as PydanticBase
except ImportError:
    raise RuntimeError("fastapi and uvicorn are required: pip install fastapi uvicorn")

from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority
from env.tasks.task_registry import TaskRegistry

app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv-compliant email triage environment API",
    version="1.0.0",
)

# ── In-memory session store (single-session for HF Space demo) ────────────────
_sessions: dict[str, EmailTriageEnv] = {}
DEFAULT_SESSION = "default"


def _get_env(session_id: str = DEFAULT_SESSION) -> EmailTriageEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


def _obs_to_dict(obs) -> Optional[dict]:
    if obs is None:
        return None
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    # dataclass fallback
    from dataclasses import asdict
    return asdict(obs)


# ── Request schemas ────────────────────────────────────────────────────────────

class ResetRequest(PydanticBase):
    task_id: str = "triage_easy"
    seed: int = 42
    session_id: str = DEFAULT_SESSION


class StepRequest(PydanticBase):
    email_id: str
    label: str
    queue: Optional[str]    = None
    priority: Optional[int] = None
    sla_hours: Optional[int] = None
    duplicate_of: Optional[str] = None
    escalate: Optional[bool] = False
    session_id: str = DEFAULT_SESSION


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "email-triage-openenv", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    tasks = []
    for tid in TaskRegistry.list_tasks():
        cfg = TaskRegistry.get(tid)
        tasks.append({
            "task_id": cfg.task_id,
            "difficulty": cfg.difficulty,
            "description": cfg.description,
            "max_steps": cfg.max_steps,
        })
    return {"tasks": tasks}


@app.post("/reset")
async def reset(req: ResetRequest):
    """Reset (or create) an environment session. Returns first Observation."""
    env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[req.session_id] = env
    return {"observation": _obs_to_dict(obs), "session_id": req.session_id}


@app.post("/step")
async def step(req: StepRequest):
    """Take one action. Returns StepResult."""
    env = _get_env(req.session_id)
    try:
        action = Action(
            email_id=req.email_id,
            label=req.label,
            queue=Queue(req.queue) if req.queue else None,
            priority=Priority(req.priority) if req.priority else None,
            sla_hours=req.sla_hours,
            duplicate_of=req.duplicate_of,
            escalate=req.escalate,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = env.step(action)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": _obs_to_dict(result.observation),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def state(session_id: str = DEFAULT_SESSION):
    """Return full current environment state."""
    env = _get_env(session_id)
    return env.state()
