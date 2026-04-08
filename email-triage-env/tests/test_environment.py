"""
Tests for EmailTriageEnv.

Run with pytest (if installed):  pytest tests/ -v
Run with stdlib unittest:        python3 -m unittest discover tests/ -v
Run directly:                    python3 tests/test_environment.py
"""
import sys
import os
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority


class TestModels(unittest.TestCase):

    def test_action_basic(self):
        a = Action(email_id="abc123", label="urgent")
        self.assertEqual(a.label, "urgent")
        self.assertIsNone(a.queue)

    def test_action_full(self):
        a = Action(
            email_id="abc123",
            label="urgent",
            queue=Queue.SUPPORT,
            priority=Priority.HIGH,
            sla_hours=4,
            escalate=True,
        )
        self.assertEqual(a.queue, Queue.SUPPORT)
        self.assertEqual(a.priority, Priority.HIGH)

    def test_action_invalid_label(self):
        with self.assertRaises(ValueError):
            Action(email_id="abc", label="UNKNOWN_LABEL")

    def test_action_string_queue_coercion(self):
        a = Action(email_id="x", label="urgent", queue="sales")
        self.assertEqual(a.queue, Queue.SALES)

    def test_model_dump(self):
        a = Action(email_id="x", label="urgent", queue=Queue.BILLING)
        d = a.model_dump()
        self.assertEqual(d["label"], "urgent")
        self.assertEqual(d["queue"], "billing")


class TestEnvironmentReset(unittest.TestCase):

    def test_reset_returns_observation(self):
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs = env.reset()
        self.assertIsNotNone(obs)
        self.assertIsNotNone(obs.email)
        self.assertEqual(obs.total_emails, 20)
        self.assertEqual(obs.inbox_position, 0)

    def test_medium_reset_email_count(self):
        env = EmailTriageEnv(task_id="triage_medium", seed=42)
        obs = env.reset()
        self.assertEqual(obs.total_emails, 30)

    def test_hard_reset_email_count(self):
        env = EmailTriageEnv(task_id="triage_hard", seed=42)
        obs = env.reset()
        self.assertEqual(obs.total_emails, 40)


class TestEnvironmentStep(unittest.TestCase):

    def test_step_basic(self):
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs = env.reset()
        action = Action(email_id=obs.email.id, label="urgent")
        result = env.step(action)
        self.assertGreaterEqual(result.reward, 0.0)
        self.assertLessEqual(result.reward, 1.0)
        self.assertIsInstance(result.done, bool)
        self.assertIsNotNone(result.info)

    def test_wrong_email_id_raises(self):
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(Action(email_id="wrong_id_xyz", label="urgent"))

    def test_step_after_done_raises(self):
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs = env.reset()
        while obs is not None:
            result = env.step(Action(email_id=obs.email.id, label="not_urgent"))
            obs = result.observation
        with self.assertRaises(RuntimeError):
            env.step(Action(email_id="anything", label="urgent"))

    def test_observation_is_none_when_done(self):
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs = env.reset()
        result = None
        while obs is not None:
            result = env.step(Action(email_id=obs.email.id, label="not_urgent"))
            obs = result.observation
        self.assertTrue(result.done)
        self.assertIsNone(result.observation)


class TestFullEpisodes(unittest.TestCase):

    def _run_episode(self, task_id, label="not_urgent", queue=None, priority=None,
                     sla_hours=None, escalate=False):
        env = EmailTriageEnv(task_id=task_id, seed=42)
        obs = env.reset()
        episode_score = None
        while obs is not None:
            action = Action(
                email_id=obs.email.id,
                label=label,
                queue=Queue(queue) if queue else None,
                priority=Priority(priority) if priority else None,
                sla_hours=sla_hours,
                escalate=escalate,
            )
            result = env.step(action)
            obs = result.observation
            if result.done:
                episode_score = result.info["episode_score"]
        return episode_score

    def test_easy_episode_completes(self):
        score = self._run_episode("triage_easy")
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_medium_episode_completes(self):
        score = self._run_episode("triage_medium", queue="general", priority=1)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.0)

    def test_hard_episode_completes(self):
        score = self._run_episode("triage_hard", queue="general", priority=1, sla_hours=72)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0.0)

    def test_perfect_easy_score(self):
        """All-urgent agent on all-urgent episode should score 1.0 or near."""
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs = env.reset()
        ep_score = None
        while obs is not None:
            result = env.step(Action(email_id=obs.email.id, label="urgent"))
            obs = result.observation
            if result.done:
                ep_score = result.info["episode_score"]
        # Score should be > 0 (it won't be 0 as some emails are actually urgent)
        self.assertGreater(ep_score, 0.0)


class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_episode(self):
        env1 = EmailTriageEnv(task_id="triage_easy", seed=42)
        env2 = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        self.assertEqual(obs1.email.id, obs2.email.id)
        self.assertEqual(obs1.email.subject, obs2.email.subject)

    def test_different_seeds_differ(self):
        env1 = EmailTriageEnv(task_id="triage_easy", seed=42)
        env2 = EmailTriageEnv(task_id="triage_easy", seed=999)
        obs1 = env1.reset()
        obs2 = env2.reset()
        # At least the sequence will differ somewhere; IDs are hash-based on seed+index
        self.assertNotEqual(obs1.email.id, obs2.email.id)


class TestStateSerialisation(unittest.TestCase):

    def test_state_is_json_serialisable(self):
        env = EmailTriageEnv(task_id="triage_easy", seed=42)
        obs = env.reset()
        env.step(Action(email_id=obs.email.id, label="urgent"))
        state = env.state()
        json_str = json.dumps(state)   # must not raise
        loaded = json.loads(json_str)
        self.assertIn("task_id", loaded)
        self.assertIn("cumulative_reward", loaded)
        self.assertIn("done", loaded)

    def test_state_fields_present(self):
        env = EmailTriageEnv(task_id="triage_medium", seed=42)
        env.reset()
        state = env.state()
        for key in ["task_id", "seed", "current_index", "total_emails", "done",
                    "cumulative_reward", "step_rewards", "actions_taken"]:
            self.assertIn(key, state)


class TestGraders(unittest.TestCase):

    def test_easy_grader_correct(self):
        from env.graders.easy_grader import EasyGrader
        grader = EasyGrader()
        a = Action(email_id="e1", label="urgent")
        result = grader.score(a, {"label": "urgent"}, 0.0, [])
        self.assertEqual(result.step_reward, 1.0)

    def test_easy_grader_wrong(self):
        from env.graders.easy_grader import EasyGrader
        grader = EasyGrader()
        a = Action(email_id="e1", label="urgent")
        result = grader.score(a, {"label": "not_urgent"}, 0.0, [])
        self.assertEqual(result.step_reward, 0.0)

    def test_easy_grader_perfect_f1(self):
        from env.graders.easy_grader import EasyGrader
        grader = EasyGrader()
        actions = [Action(email_id=f"e{i}", label="urgent") for i in range(5)]
        truths  = [{"label": "urgent"} for _ in range(5)]
        score = grader.finalize(actions, truths)
        self.assertAlmostEqual(score, 1.0)

    def test_medium_grader_full_credit(self):
        from env.graders.medium_grader import MediumGrader
        grader = MediumGrader()
        a = Action(email_id="e1", label="urgent", queue=Queue.SUPPORT, priority=Priority.HIGH)
        result = grader.score(a, {"label": "urgent", "queue": "support", "priority": 3}, 0.0, [])
        self.assertAlmostEqual(result.step_reward, 1.0)

    def test_medium_grader_partial_credit(self):
        from env.graders.medium_grader import MediumGrader
        grader = MediumGrader()
        a = Action(email_id="e1", label="urgent", queue=Queue.SUPPORT, priority=Priority.MEDIUM)
        result = grader.score(a, {"label": "urgent", "queue": "support", "priority": 3}, 0.0, [])
        # Queue correct (0.6) + priority off-by-one partial (0.5 * 0.4 = 0.2) = 0.8
        self.assertAlmostEqual(result.step_reward, 0.8)

    def test_hard_grader_false_escalation_penalty(self):
        from env.graders.hard_grader import HardGrader
        grader = HardGrader()
        a = Action(email_id="e1", label="not_urgent", queue=Queue.GENERAL,
                   priority=Priority.LOW, sla_hours=72, escalate=True)
        gt = {"label": "not_urgent", "queue": "general", "priority": 1,
              "sla_hours": 72, "duplicate_of": None, "escalate": False}
        result = grader.score(a, gt, 0.0, [])
        self.assertIn("false_escalation", result.penalties)
        self.assertAlmostEqual(result.penalties["false_escalation"], -0.10)

    def test_reward_always_non_negative(self):
        from env.graders.hard_grader import HardGrader
        grader = HardGrader()
        # Worst-case: wrong on everything + false escalation + missed dup
        a = Action(email_id="e1", label="urgent", queue=Queue.SALES,
                   priority=Priority.LOW, sla_hours=1000, escalate=True)
        gt = {"label": "not_urgent", "queue": "general", "priority": 1,
              "sla_hours": 72, "duplicate_of": "some_id", "escalate": False}
        result = grader.score(a, gt, 0.0, [])
        self.assertGreaterEqual(result.step_reward, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
