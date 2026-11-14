#!/usr/bin/env bash
# Verifier (runs in the main container after the agent). verify.py writes reward.txt last and only after a
# clean replay; any failure exits non-zero with no reward, which Harbor records as a trial error.
set -euo pipefail
PYTHONPATH="$(dirname "$0")" python3 "$(dirname "$0")/verify.py" --logs-dir /logs/verifier{{VERIFY_EXTRA}}
