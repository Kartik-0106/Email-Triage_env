from env.models import Action, RewardSignal
from env.graders.base_grader import BaseGrader


class HardGrader(BaseGrader):
    """
    Full rubric grader for triage_hard.
    Weighted scoring:
      - Routing (queue):      0.35
      - SLA accuracy:         0.25
      - Duplicate detection:  0.20
      - Escalation:           0.20
    Penalties:
      - False escalation:    -0.10
      - Missed duplicate:    -0.15
    """

    ROUTING_W = 0.35
    SLA_W = 0.25
    DUPLICATE_W = 0.20
    ESCALATION_W = 0.20

    FALSE_ESCALATION_PENALTY = -0.10
    MISSED_DUPLICATE_PENALTY = -0.15

    def score(
        self,
        action: Action,
        ground_truth: dict,
        cumulative_reward: float,
        history: list,
    ) -> RewardSignal:
        penalties = {}

        # Routing score
        routing_score = 1.0 if (
            action.queue and action.queue.value == ground_truth.get("queue")
        ) else 0.0

        # SLA score (within 50% of correct value = partial, exact = full)
        sla_score = 0.0
        if action.sla_hours is not None and ground_truth.get("sla_hours") is not None:
            gt_sla = ground_truth["sla_hours"]
            diff_ratio = abs(action.sla_hours - gt_sla) / max(gt_sla, 1)
            if diff_ratio == 0:
                sla_score = 1.0
            elif diff_ratio <= 0.5:
                sla_score = 0.5

        # Duplicate detection score
        gt_dup = ground_truth.get("duplicate_of")
        duplicate_score = 0.0
        if gt_dup is None and action.duplicate_of is None:
            duplicate_score = 1.0  # Correctly identified as not duplicate
        elif gt_dup is not None and action.duplicate_of == gt_dup:
            duplicate_score = 1.0  # Correctly identified duplicate
        elif gt_dup is not None and action.duplicate_of is not None and action.duplicate_of != gt_dup:
            duplicate_score = 0.3  # Tried to detect but wrong ID
        else:
            # gt_dup is not None but agent missed it
            penalties["missed_duplicate"] = self.MISSED_DUPLICATE_PENALTY

        # Escalation score
        gt_escalate = ground_truth.get("escalate", False)
        escalation_score = 0.0
        if action.escalate == gt_escalate:
            escalation_score = 1.0
        elif action.escalate and not gt_escalate:
            penalties["false_escalation"] = self.FALSE_ESCALATION_PENALTY

        base_reward = (
            self.ROUTING_W * routing_score
            + self.SLA_W * sla_score
            + self.DUPLICATE_W * duplicate_score
            + self.ESCALATION_W * escalation_score
        )

        total_penalty = sum(penalties.values())
        step_reward = max(0.0, base_reward + total_penalty)

        return RewardSignal(
            step_reward=step_reward,
            cumulative_reward=cumulative_reward + step_reward,
            partial_scores={
                "routing": routing_score,
                "sla": sla_score,
                "duplicate": duplicate_score,
                "escalation": escalation_score,
            },
            penalties=penalties,
        )

    def finalize(self, actions: list, ground_truths: list) -> float:
        total = 0.0
        history = []
        for action, gt in zip(actions, ground_truths):
            result = self.score(action, gt, total, history)
            total += result.step_reward
            history.append(action)
        return min(1.0, total / max(len(actions), 1))
