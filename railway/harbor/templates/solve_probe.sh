#!/usr/bin/env bash
# Isolation probe (export --probe; never in a judge dataset). Records what the agent's container can see and
# reach in /logs/agent/probe.json (verify.py --probe copies it into details.json), then runs the oracle.
set -uo pipefail
mkdir -p /logs/agent
export P_UID="$(id -u)"
if curl -s --max-time 5 -o /dev/null https://github.com; then P_GITHUB=reachable; else P_GITHUB=unreachable; fi
export P_GITHUB
export P_HEALTH="$(curl -s --max-time 5 http://sim:8000/health || true)"
export P_NO_TOKEN="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://sim:8000/episode_log || true)"
export P_WRONG_TOKEN="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 -H 'Authorization: Bearer wrong' http://sim:8000/episode_log || true)"
export P_SCENARIOS="$(find / -xdev -name 'scenario*.json' 2>/dev/null | tr '\n' ' ' | sed 's/ *$//')"
# From / and without PYTHONPATH: the vendored solution code must not count as "railway is importable".
if (unset PYTHONPATH; cd / && python3 -c "import railway") 2>/dev/null; then P_IMPORT=ok; else P_IMPORT=fails; fi
export P_IMPORT
python3 - <<'PY'
import json
import os

keys = {
    "uid": "P_UID",
    "github": "P_GITHUB",
    "sim_health": "P_HEALTH",
    "episode_log_no_token": "P_NO_TOKEN",
    "episode_log_wrong_token": "P_WRONG_TOKEN",
    "scenario_files": "P_SCENARIOS",
    "import_railway": "P_IMPORT",
}
with open("/logs/agent/probe.json", "w") as f:
    json.dump({k: os.environ.get(v, "") for k, v in keys.items()}, f, indent=2, sort_keys=True)
PY
set -e
PYTHONPATH="$(dirname "$0")" python3 -m railway.ref.oracle --url http://sim:8000/mcp
