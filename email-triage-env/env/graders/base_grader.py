from abc import ABC, abstractmethod
from env.models import Action, RewardSignal


class BaseGrader(ABC):

    @abstractmethod
    def score(
        self,
        action: Action,
        ground_truth: dict,
        cumulative_reward: float,
        history: list,
    ) -> RewardSignal:
        """Score a single step action against ground truth."""
        pass

    @abstractmethod
    def finalize(self, actions: list, ground_truths: list) -> float:
        """Compute final episode score in [0.0, 1.0]."""
        pass
