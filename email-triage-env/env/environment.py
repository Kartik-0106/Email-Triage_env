import random
from typing import Optional
from env.models import Observation, Action, StepResult, RewardSignal
from env.email_generator import EmailGenerator
from env.tasks.task_registry import TaskRegistry


class EmailTriageEnv:
    """
    OpenEnv-compliant email triage environment.

    API:
        reset()         -> Observation
        step(action)    -> StepResult(observation, reward, done, info)
        state()         -> dict (fully serializable current state)

    Tasks:
        triage_easy   (easy)   - binary label classification
        triage_medium (medium) - queue routing + priority
        triage_hard   (hard)   - full rubric triage
    """

    def __init__(self, task_id: str = "triage_easy", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._task = TaskRegistry.get(task_id)
        self._grader = self._task.grader()
        self._generator = EmailGenerator(seed=seed)

        # Episode state
        self._emails = []
        self._ground_truth = []
        self._current_index = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._step_rewards = []
        self._actions_taken = []

    def reset(self) -> Observation:
        """Reset the environment. Returns the first Observation."""
        config = self._task.config
        self._emails, self._ground_truth = self._generator.generate_episode(
            task_id=self.task_id,
            n=config["max_steps"],
            seed=self.seed,
        )
        self._current_index = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._step_rewards = []
        self._actions_taken = []
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process one action.
        Returns StepResult(observation, reward, done, info).
        observation is None when done=True.
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new episode.")

        current_email = self._emails[self._current_index]
        if action.email_id != current_email.id:
            raise ValueError(
                f"Action targets email '{action.email_id}' "
                f"but current email is '{current_email.id}'. "
                f"Always use obs.email.id as the action's email_id."
            )

        ground_truth = self._ground_truth[self._current_index]
        reward_signal: RewardSignal = self._grader.score(
            action=action,
            ground_truth=ground_truth,
            cumulative_reward=self._cumulative_reward,
            history=self._actions_taken,
        )

        self._cumulative_reward += reward_signal.step_reward
        self._step_rewards.append(reward_signal.step_reward)
        self._actions_taken.append(action)
        self._current_index += 1

        self._done = self._current_index >= len(self._emails)
        episode_score = None
        if self._done:
            episode_score = self._grader.finalize(
                actions=self._actions_taken,
                ground_truths=self._ground_truth,
            )

        next_obs = None if self._done else self._make_observation()

        return StepResult(
            observation=next_obs,
            reward=reward_signal.step_reward,
            done=self._done,
            info={
                "cumulative_reward": self._cumulative_reward,
                "partial_scores": reward_signal.partial_scores,
                "penalties": reward_signal.penalties,
                "episode_score": episode_score,
                "step": self._current_index,
                "total_steps": len(self._emails),
            },
        )

    def state(self) -> dict:
        """Return full current environment state as a serializable dict."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "current_index": self._current_index,
            "total_emails": len(self._emails),
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "step_rewards": [round(r, 4) for r in self._step_rewards],
            "actions_taken": [a.model_dump() for a in self._actions_taken],
        }

    def _make_observation(self) -> Observation:
        email = self._emails[self._current_index]
        context_start = max(0, self._current_index - 3)
        context = self._emails[context_start:self._current_index]
        return Observation(
            email=email,
            inbox_position=self._current_index,
            total_emails=len(self._emails),
            elapsed_steps=self._current_index,
            context_window=context,
        )
