#!/usr/bin/env bash
# Verifier (runs in the main container after the agent). verify.py writes reward.txt last and only after a
# clean replay; any failure exits non-zero with no reward, which Harbor records as a trial error.
#
# Run as root (Harbor's default for tests/), this script first locks /logs/verifier against the agent
# (UID 1000): it kills every lingering agent process (e.g. a `(sleep 5 && echo 0.0 > reward.txt) &`
# left behind on exit), removes anything the agent planted in /logs/verifier, makes the directory
# root:root 0755 and, after scoring, leaves the outputs root-owned 0644, so the agent can neither race
# nor overwrite reward.txt. Run as any other user (local unit tests) it only runs verify.py.
set -euo pipefail

LOGS_DIR=/logs/verifier
AGENT_USER=agent
AGENT_UID=1000

agent_pids() {
    local status pid key ruid euid rest
    for status in /proc/[0-9]*/status; do
        pid=${status#/proc/}
        pid=${pid%/status}
        {
            while read -r key ruid euid rest; do
                if [ "$key" = "Uid:" ]; then
                    if [ "$ruid" = "$AGENT_UID" ] || [ "$euid" = "$AGENT_UID" ]; then
                        echo "$pid"
                    fi
                    break
                fi
            done <"$status"
        } 2>/dev/null || true
    done
}

kill_agent_processes() {
    local pass pids
    for pass in 1 2 3 4 5 6 7 8 9 10; do
        pkill -9 -u "$AGENT_USER" 2>/dev/null || true
        # kill(-1, SIGKILL) issued as the agent UID reaches every process of that UID in one call.
        if command -v setpriv >/dev/null 2>&1; then
            setpriv --reuid="$AGENT_UID" --regid="$AGENT_UID" --clear-groups \
                python3 -c 'import os, signal; os.kill(-1, signal.SIGKILL)' >/dev/null 2>&1 || true
        fi
        pids=$(agent_pids)
        [ -z "$pids" ] && return 0
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 0.2
    done
    echo "WARN: agent processes still alive after kill passes: $(agent_pids | tr '\n' ' ')" >&2
}

if [ "$(id -u)" = "0" ]; then
    kill_agent_processes
    mkdir -p "$LOGS_DIR"
    find "$LOGS_DIR" -mindepth 1 ! -user 0 -exec rm -rf {} + 2>/dev/null || true
    chown root:root "$LOGS_DIR" || echo "WARN: cannot chown $LOGS_DIR" >&2
    chmod 0755 "$LOGS_DIR" || echo "WARN: cannot chmod $LOGS_DIR" >&2
fi

rc=0
PYTHONPATH="$(dirname "$0")" python3 "$(dirname "$0")/verify.py" --logs-dir "$LOGS_DIR" || rc=$?

if [ "$(id -u)" = "0" ] && [ "$rc" = "0" ]; then
    chown root:root "$LOGS_DIR"/* 2>/dev/null || true
    chmod 0644 "$LOGS_DIR"/* || echo "WARN: cannot chmod $LOGS_DIR outputs" >&2
fi
exit "$rc"
