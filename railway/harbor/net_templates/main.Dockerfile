# Agent ("main") image. It holds no simulator, no timetable and no token: those live only in the sim image
# and in the task's tests/ folder, which Harbor uploads after the agent has finished.
FROM python:3.11-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates nodejs npm \
 && rm -rf /var/lib/apt/lists/*
# Pre-installed so Harbor skips OpenCode's install step, which the network allowlist blocks.
RUN npm i -g opencode-ai@1.19.5 && opencode --version
{{TEACHER}}# Exactly the sim image's pins (from uv.lock): the verifier replays here and must match the live server.
RUN pip install --no-cache-dir \
{{PINS}}
RUN useradd -m -u 1000 agent
WORKDIR /app
