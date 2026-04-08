"""
validate.py — pre-submission validation script.

Checks every gate from the pre-submission checklist:
  1. openenv.yaml present and valid
  2. Pydantic models importable (or stdlib fallback)
  3. step() / reset() / state() API works end-to-end
  4. All 3 tasks complete without error
  5. Episode scores in [0.0, 1.0]
  6. inference.py exists at root
  7. Required env var names referenced in inference.py
  8. Dockerfile present
  9. server.py present

Run:
  python validate.py
"""
import sys
import os
import importlib

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append((name, True, None))
    except Exception as exc:
        print(f"{FAIL}  {name}")
        print(f"       {exc}")
        results.append((name, False, str(exc)))

print("\n=== EmailTriageEnv Pre-Submission Validator ===\n")

# 1. openenv.yaml
def _yaml():
    import yaml
    data = yaml.safe_load(open("openenv.yaml"))
    assert "name" in data
    assert "tasks" in data and len(data["tasks"]) >= 3
    assert "http_endpoints" in data
    assert "observation_space" in data
    assert "action_space" in data
check("openenv.yaml — present and valid", _yaml)

# 2. Models importable
def _models():
    from env.models import Observation, Action, RewardSignal, StepResult, Queue, Priority
    a = Action(email_id="test", label="urgent", queue=Queue.SUPPORT, priority=Priority.HIGH)
    assert a.label == "urgent"
check("Pydantic/dataclass models — importable and constructable", _models)

# 3-5. Full episode for each task
from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority

for task_id, max_steps in [("triage_easy",20),("triage_medium",30),("triage_hard",40)]:
    def _task(tid=task_id, ms=max_steps):
        env = EmailTriageEnv(task_id=tid, seed=42)
        obs = env.reset()
        assert obs is not None
        assert obs.total_emails == ms
        ep_score = None
        steps = 0
        while obs is not None:
            act = Action(
                email_id=obs.email.id, label="urgent",
                queue=Queue.SUPPORT, priority=Priority.HIGH,
                sla_hours=4, escalate=False,
            )
            result = env.step(act)
            assert 0.0 <= result.reward <= 1.0, f"reward {result.reward} out of range"
            steps += 1
            obs = result.observation
            if result.done:
                ep_score = result.info["episode_score"]
        assert ep_score is not None
        assert 0.0 <= ep_score <= 1.0, f"episode_score {ep_score} out of range"
        state = env.state()
        assert state["done"] is True
    check(f"Task '{task_id}' — episode completes, scores in [0,1]", _task)

# 6. inference.py at root
def _inference_file():
    assert os.path.isfile("inference.py"), "inference.py not found at project root"
    src = open("inference.py").read()
    for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        assert var in src, f"Required env var {var!r} not referenced in inference.py"
check("inference.py — present at root, references required env vars", _inference_file)

# 7. Dockerfile
def _dockerfile():
    assert os.path.isfile("Dockerfile"), "Dockerfile not found"
    src = open("Dockerfile").read()
    assert "EXPOSE" in src
    assert "pip install" in src
check("Dockerfile — present with EXPOSE and pip install", _dockerfile)

# 8. server.py
def _server():
    assert os.path.isfile("server.py"), "server.py not found"
    src = open("server.py").read()
    assert "/reset" in src
    assert "/step"  in src
    assert "/state" in src
    assert "/health" in src
check("server.py — present with /reset /step /state /health", _server)

# 9. app.py (HF Space)
def _app():
    assert os.path.isfile("app.py"), "app.py not found"
    readme = open("README.md").read()
    assert "sdk: gradio" in readme, "HF Space YAML header missing from README.md"
check("app.py + README HF Space header", _app)

# 10. 27 unit tests
def _tests():
    import unittest
    loader = unittest.TestLoader()
    suite  = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull,"w"))
    result = runner.run(suite)
    assert result.wasSuccessful(), f"{len(result.failures)} failures, {len(result.errors)} errors"
check("Unit tests — all pass", _tests)

# ── Summary ──────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"\n{'='*48}")
print(f"  {passed}/{total} checks passed  |  {failed} failed")
print(f"{'='*48}")
if failed:
    print("\n  FAILED CHECKS:")
    for name, ok, err in results:
        if not ok:
            print(f"    - {name}")
    sys.exit(1)
else:
    print("\n  All checks passed. Ready to submit!")
    sys.exit(0)
