"""End-to-end check of exported tasks without Harbor: real MCP server, oracle / greedy / do-nothing over MCP,
then the task's own vendored verifier. Expect oracle 1.000000, greedy 0.000000, do-nothing 0.000000.

    python tests/v03/check_task.py datasets/t1-dev-v03/T1-...  [more task dirs]
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from railway.policies.baselines import DoNothing, Greedy
from railway.server.spawn import spawn
from railway.sim.model import Scenario
from railway.tools.client import McpClient


def play(task: Path, who: str) -> str:
    scenario = Scenario.model_validate_json((task / "tests" / "scenario.json").read_text(encoding="utf-8"))
    with spawn(scenario) as server, McpClient(server.url) as client:
        if who == "oracle":
            for call in json.loads((task / "solution" / "plan.json").read_text(encoding="utf-8")):
                out = client.call(call["tool"], call["args"])
                if "SHIFT ENDED" in out or "R-SHIFT-ENDED" in out:
                    break
            client.call("end_shift", {})
        else:
            {"greedy": Greedy, "nothing": DoNothing}[who]().run(client)
        with tempfile.TemporaryDirectory() as tmp:
            tests = Path(tmp) / "tests"
            shutil.copytree(task / "tests", tests)
            (tests / "token").write_text(server.token, encoding="utf-8")
            logs = Path(tmp) / "logs"
            done = subprocess.run(
                [sys.executable, str(tests / "verify.py"), "--logs-dir", str(logs), "--sim-url", server.base,
                 "--agent-logs-dir", str(Path(tmp) / "agent")],
                cwd=tests, capture_output=True, text=True, env={"PYTHONPATH": str(tests), "SYSTEMROOT": "C:\\Windows"},
            )  # fmt: skip
            out = done.stdout.strip() or done.stderr.strip()[-300:]
            details = json.loads((logs / "details.json").read_text(encoding="utf-8")) if (logs / "details.json").exists() else {}
            return f"{out} | cost {details.get('cost_total')} kpis {details.get('kpis')}"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        for who in ("oracle", "greedy", "nothing"):
            print(Path(arg).name, who, "->", play(Path(arg), who), flush=True)
