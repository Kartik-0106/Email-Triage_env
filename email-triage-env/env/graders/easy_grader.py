from env.models import Action, RewardSignal
from env.graders.base_grader import BaseGrader


class EasyGrader(BaseGrader):
    """
    Binary label grader for triage_easy.
    Step reward = 1.0 if correct, 0.0 if wrong.
    Episode score = macro F1 over all decisions.
    Edge case: if only one class appears in ground truth, falls back to accuracy.
    """

    def score(self, action, ground_truth, cumulative_reward, history):
        correct = action.label == ground_truth["label"]
        step_reward = 1.0 if correct else 0.0
        return RewardSignal(
            step_reward=step_reward,
            cumulative_reward=cumulative_reward + step_reward,
            partial_scores={"label": step_reward},
            penalties={},
        )

    def finalize(self, actions, ground_truths):
        preds  = [a.label for a in actions]
        truths = [g["label"] for g in ground_truths]

        classes = list(set(truths))          # only classes present in ground truth
        if len(classes) == 1:
            # Single-class episode: return accuracy
            correct = sum(p == t for p, t in zip(preds, truths))
            return correct / max(len(truths), 1)

        f1_scores = []
        for cls in classes:
            tp = sum(1 for p, t in zip(preds, truths) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(preds, truths) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(preds, truths) if p != cls and t == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)
