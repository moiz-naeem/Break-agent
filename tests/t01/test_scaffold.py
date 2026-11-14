import importlib
import importlib.metadata as md
import sys

import hypothesis
import pytest

SUBPACKAGES = ["sim", "gen", "tools", "server", "policies", "ref", "harbor", "loop", "eval", "train"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_T01_01_package_layout(name):
    importlib.import_module(f"railway.{name}")


def test_T01_02_python_and_deps():
    assert sys.version_info >= (3, 11)
    assert md.version("pydantic").startswith("2.")
    for dist in ["mcp", "httpx", "hypothesis", "pytest"]:
        assert md.version(dist)


def test_T01_03_markers_registered(pytestconfig):
    names = {m.split(":")[0].strip() for m in pytestconfig.getini("markers")}
    assert {"slow", "harbor", "model", "gpu"} <= names
    assert "--strict-markers" in pytestconfig.getini("addopts")


def test_T01_04_hypothesis_profile():
    assert hypothesis.settings.default.derandomize is True
