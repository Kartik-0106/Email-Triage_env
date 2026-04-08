"""
app.py — Hugging Face Space entrypoint (Gradio UI).

Exposes the EmailTriageEnv interactively AND embeds the FastAPI server
so the automated openenv validate ping to /reset and /health succeeds.

Start locally:
  python app.py
"""
import threading
import uvicorn
import gradio as gr
import json

from env.environment import EmailTriageEnv
from env.models import Action, Queue, Priority
from env.tasks.task_registry import TaskRegistry


# ── Launch the FastAPI server in a background thread ─────────────────────────
def _start_api():
    from server import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="warning")

api_thread = threading.Thread(target=_start_api, daemon=True)
api_thread.start()


# ── Heuristic keyword agent (demo — no API key) ───────────────────────────────
URGENT_KW = ["urgent","critical","breach","failed","failure","down","outage",
              "expire","overdue","suspension","legal","gdpr","vip","corruption"]
QUEUE_KW  = {
    "billing": ["invoice","payment","billing","overdue","charge","subscription"],
    "legal":   ["legal","compliance","contract","regulation","gdpr","audit"],
    "sales":   ["sales","renewal","lead","partnership","enterprise","client"],
    "support": ["server","outage","bug","error","crash","system","database"],
}

def _heuristic(obs):
    text = (obs.email.subject + " " + obs.email.body).lower()
    label = "urgent" if any(k in text for k in URGENT_KW) else "not_urgent"
    queue = Queue.GENERAL
    for q, kws in QUEUE_KW.items():
        if any(k in text for k in kws):
            queue = Queue(q); break
    priority = Priority.HIGH if label == "urgent" else Priority.LOW
    return Action(
        email_id=obs.email.id, label=label, queue=queue,
        priority=priority, sla_hours=4 if label=="urgent" else 72, escalate=False,
    )


# ── Gradio demo ───────────────────────────────────────────────────────────────
def run_demo(task_id: str, seed: int) -> tuple[str, float, str]:
    env = EmailTriageEnv(task_id=task_id, seed=int(seed))
    obs = env.reset()
    log, ep_score = [], 0.0
    while obs is not None:
        action = _heuristic(obs)
        result = env.step(action)
        log.append({
            "step":    result.info["step"],
            "subject": obs.email.subject[:50],
            "label":   action.label,
            "reward":  round(result.reward, 3),
            "scores":  {k: round(v,2) for k,v in result.info.get("partial_scores",{}).items()},
        })
        if result.done:
            ep_score = result.info.get("episode_score", 0.0)
        obs = result.observation

    state_json = json.dumps(env.state(), indent=2)
    log_text   = "\n".join(
        f"[{e['step']:02d}] {e['subject']:<45} → {e['label']:<10} r={e['reward']:.3f}"
        for e in log
    )
    return log_text, round(ep_score, 4), state_json


with gr.Blocks(title="Email Triage OpenEnv") as demo:
    gr.Markdown("## 📧 Email Triage OpenEnv\nHeuristic keyword agent demo — no API key required.")
    with gr.Row():
        task_dd = gr.Dropdown(
            choices=["triage_easy","triage_medium","triage_hard"],
            value="triage_easy", label="Task"
        )
        seed_sl = gr.Slider(minimum=0, maximum=100, value=42, step=1, label="Seed")
        run_btn = gr.Button("Run episode")
    log_box   = gr.Textbox(label="Step log", lines=22, interactive=False)
    score_box = gr.Number(label="Episode score [0.0 – 1.0]")
    state_box = gr.Code(label="Final state (JSON)", language="json")

    run_btn.click(fn=run_demo, inputs=[task_dd, seed_sl],
                  outputs=[log_box, score_box, state_box])

    gr.Markdown("""
---
**API endpoints** (same container, port 7860):
`POST /reset` · `POST /step` · `GET /state` · `GET /tasks` · `GET /health`
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
