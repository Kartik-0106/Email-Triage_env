from env.models import Action, RewardSignal
from env.graders.base_grader import BaseGrader


class MediumGrader(BaseGrader):
    """
    Partial credit grader for triage_medium.
    Queue correct = 0.6 weight, Priority correct = 0.4 weight.
    Penalty for lazy routing (routing >70% to 'general').
    """

    QUEUE_WEIGHT = 0.6
    PRIORITY_WEIGHT = 0.4
    ROUTING_PENALTY_THRESHOLD = 0.70
    ROUTING_PENALTY = -0.05

    def score(
        self,
        action: Action,
        ground_truth: dict,
        cumulative_reward: float,
        history: list,
    ) -> RewardSignal:
        # Queue score
        queue_score = 1.0 if (action.queue and action.queue.value == ground_truth.get("queue")) else 0.0

        # Priority score (partial credit for off-by-one)
        priority_score = 0.0
        if action.priority is not None and ground_truth.get("priority") is not None:
            diff = abs(int(action.priority) - int(ground_truth["priority"]))
            if diff == 0:
                priority_score = 1.0
            elif diff == 1:
                priority_score = 0.5

        base_reward = self.QUEUE_WEIGHT * queue_score + self.PRIORITY_WEIGHT * priority_score

        # Anti-gaming penalty: penalize lazy routing to 'general'
        general_count = sum(
            1 for a in history if a.queue is not None and a.queue.value == "general"
        )
        total = len(history) + 1
        penalty = 0.0
        if total > 5 and (general_count / total) > self.ROUTING_PENALTY_THRESHOLD:
            penalty = self.ROUTING_PENALTY

        step_reward = max(0.0, base_reward + penalty)
        return RewardSignal(
            step_reward=step_reward,
            cumulative_reward=cumulative_reward + step_reward,
            partial_scores={"queue": queue_score, "priority": priority_score},
            penalties={"lazy_routing": penalty},
        )

    def finalize(self, actions: list, ground_truths: list) -> float:
        total = 0.0
        running_history = []
        for i, (action, gt) in enumerate(zip(actions, ground_truths)):
            result = self.score(action, gt, total, running_history)
            total += result.step_reward
            running_history.append(action)
        return min(1.0, total / max(len(actions), 1))
