FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .

# Expose single port used by both FastAPI (/reset /step /state /health)
# and Gradio UI (via background thread)
EXPOSE 7860

# Health check — automated ping must return 200
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s CMD curl -f http://localhost:7860/health || exit 1

# Default: FastAPI server (handles openenv validate pings)
# For Gradio Space UI: override with CMD ["python","app.py"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
