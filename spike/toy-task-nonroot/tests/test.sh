#!/usr/bin/env bash
# Toy verifier: runs in the main container after the agent. Infra failure -> exit 1, no reward.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /logs/verifier
TOKEN="$(cat "$HERE/token")"
LOG=""
for _ in 1 2 3 4 5; do
  if LOG="$(curl -sf -H "Authorization: Bearer $TOKEN" http://sim:8000/episode_log)"; then break; fi
  LOG=""
  sleep 2
done
if [ -z "$LOG" ]; then
  echo "INFRA: sim unreachable"
  exit 1
fi
python3 - "$LOG" <<'EOF'
import json, os, sys
calls = json.loads(sys.argv[1])
pinged = any(c.get("tool") == "ping" for c in calls)
secret_visible = os.path.exists("/srv/secret.txt")
with open("/logs/verifier/details.json", "w") as f:
    json.dump({"calls": calls, "secret_visible": secret_visible}, f)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write("1.0\n" if pinged and not secret_visible else "0.0\n")
EOF
