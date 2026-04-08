"""
OpenEnv-compliant Pydantic models for EmailTriageEnv.
Falls back to stdlib dataclasses when pydantic is not installed
so the environment works in zero-dependency containers.
"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from typing import Optional, List, Literal
    from enum import Enum
    _PYDANTIC = True

    class Priority(int, Enum):
        LOW    = 1
        MEDIUM = 2
        HIGH   = 3

    class Queue(str, Enum):
        SALES   = "sales"
        SUPPORT = "support"
        BILLING = "billing"
        LEGAL   = "legal"
        GENERAL = "general"

    class Email(BaseModel):
        id: str
        subject: str
        body: str
        sender: str
        timestamp: str
        metadata: dict = Field(default_factory=dict)

    class Observation(BaseModel):
        email: Email
        inbox_position: int
        total_emails: int
        elapsed_steps: int
        context_window: List[Email] = Field(default_factory=list)

    class Action(BaseModel):
        email_id: str
        label: Literal["urgent", "not_urgent"]
        queue: Optional[Queue]       = None
        priority: Optional[Priority] = None
        sla_hours: Optional[int]     = None
        duplicate_of: Optional[str]  = None
        escalate: Optional[bool]     = False

    class RewardSignal(BaseModel):
        step_reward: float
        cumulative_reward: float
        partial_scores: dict
        penalties: dict
        episode_score: Optional[float] = None

    class StepResult(BaseModel):
        observation: Optional[Observation]
        reward: float
        done: bool
        info: dict

except ImportError:
    # ── stdlib fallback (zero dependencies) ──────────────────────────────────
    from dataclasses import dataclass, field
    from typing import Optional, List
    from enum import Enum
    _PYDANTIC = False

    class Priority(int, Enum):
        LOW    = 1
        MEDIUM = 2
        HIGH   = 3

    class Queue(str, Enum):
        SALES   = "sales"
        SUPPORT = "support"
        BILLING = "billing"
        LEGAL   = "legal"
        GENERAL = "general"

    @dataclass
    class Email:
        id: str
        subject: str
        body: str
        sender: str
        timestamp: str
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Observation:
        email: Email
        inbox_position: int
        total_emails: int
        elapsed_steps: int
        context_window: List[Email] = field(default_factory=list)

    @dataclass
    class Action:
        email_id: str
        label: str
        queue: Optional[Queue]       = None
        priority: Optional[Priority] = None
        sla_hours: Optional[int]     = None
        duplicate_of: Optional[str]  = None
        escalate: Optional[bool]     = False

        def __post_init__(self):
            if self.label not in {"urgent", "not_urgent"}:
                raise ValueError(f"label must be 'urgent' or 'not_urgent', got {self.label!r}")
            if self.queue is not None and not isinstance(self.queue, Queue):
                self.queue = Queue(self.queue)
            if self.priority is not None and not isinstance(self.priority, Priority):
                self.priority = Priority(int(self.priority))

        def model_dump(self):
            return {
                "email_id": self.email_id, "label": self.label,
                "queue": self.queue.value if self.queue else None,
                "priority": self.priority.value if self.priority else None,
                "sla_hours": self.sla_hours, "duplicate_of": self.duplicate_of,
                "escalate": self.escalate,
            }

    @dataclass
    class RewardSignal:
        step_reward: float
        cumulative_reward: float
        partial_scores: dict
        penalties: dict
        episode_score: Optional[float] = None

    @dataclass
    class StepResult:
        observation: Optional[Observation]
        reward: float
        done: bool
        info: dict
