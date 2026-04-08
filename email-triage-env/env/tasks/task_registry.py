from dataclasses import dataclass
from typing import Type
from env.graders.base_grader import BaseGrader
from env.graders.easy_grader import EasyGrader
from env.graders.medium_grader import MediumGrader
from env.graders.hard_grader import HardGrader


@dataclass
class TaskConfig:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    grader_class: Type[BaseGrader]

    def grader(self) -> BaseGrader:
        return self.grader_class()

    @property
    def config(self):
        return {"max_steps": self.max_steps}


TASKS = {
    "triage_easy": TaskConfig(
        task_id="triage_easy",
        difficulty="easy",
        description="Binary urgent/not_urgent classification of 20 emails.",
        max_steps=20,
        grader_class=EasyGrader,
    ),
    "triage_medium": TaskConfig(
        task_id="triage_medium",
        difficulty="medium",
        description="Queue routing + priority assignment for 30 emails.",
        max_steps=30,
        grader_class=MediumGrader,
    ),
    "triage_hard": TaskConfig(
        task_id="triage_hard",
        difficulty="hard",
        description="Full triage: routing, SLA, duplicate detection, escalation for 40 emails.",
        max_steps=40,
        grader_class=HardGrader,
    ),
}


class TaskRegistry:
    @staticmethod
    def get(task_id: str) -> TaskConfig:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id!r}. Available: {list(TASKS.keys())}")
        return TASKS[task_id]

    @staticmethod
    def list_tasks():
        return list(TASKS.keys())
