import re
import tomllib

from tests.helpers import REPO

TASK = REPO / "spike" / "toy-task"


def test_T01_16_task_layout():
    for rel in ["instruction.md", "task.toml", "environment/Dockerfile",
                "environment/docker-compose.yaml", "environment/sim/Dockerfile",
                "environment/sim/server.py", "tests/test.sh", "solution/solve.sh"]:
        assert (TASK / rel).is_file(), rel


def test_T01_17_task_toml_mcp():
    cfg = tomllib.loads((TASK / "task.toml").read_text())
    servers = cfg["environment"]["mcp_servers"]
    toy = [s for s in servers if s["name"] == "toy"]
    assert toy and toy[0]["transport"] == "streamable-http"
    assert toy[0]["url"] == "http://sim:8000/mcp"


def test_T01_18_main_dockerfile_has_no_sim():
    text = (TASK / "environment" / "Dockerfile").read_text()
    for line in text.splitlines():
        if re.match(r"\s*(COPY|ADD)\b", line, re.I):
            assert not re.search(r"sim|secret|token", line, re.I), line


def test_T01_19_verifier_never_writes_reward_json():
    assert "reward.json" not in (TASK / "tests" / "test.sh").read_text()


def test_T01_19b_verifier_and_sim_tokens_match():
    """A drifted token makes the verifier's fetch fail as a confusing infra error."""
    assert (TASK / "tests" / "token").read_bytes() == (TASK / "environment" / "sim" / "token").read_bytes()
