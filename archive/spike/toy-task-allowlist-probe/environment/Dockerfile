# Agent ("main") image: python + MCP client libs for the oracle and verifier. Nothing from sim/.
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates nodejs npm \
 && rm -rf /var/lib/apt/lists/*
# Pre-installed so Harbor skips OpenCode's install step, which needs apt/npm hosts the allowlist blocks.
RUN npm i -g opencode-ai@1.19.5 && opencode --version
RUN pip install --no-cache-dir mcp==2.3.0 httpx==0.28.1 pydantic==2.14.3
RUN useradd -m -u 1000 agent
WORKDIR /app
