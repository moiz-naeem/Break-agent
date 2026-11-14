import subprocess
import sys

from railway.gen.generator import generate
from railway.gen.presets import PRESETS
from railway.util import canonical_json


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "railway.gen.cli", *args], capture_output=True, text=True)


def test_T04_28_sample_writes_canonical(tmp_path):
    output = tmp_path / "sample.json"
    result = run_cli("sample", "--config", "T1", "--seed", "5", "--out", str(output))
    assert result.returncode == 0
    expected = canonical_json(generate(PRESETS["T1"], 5).model_dump(mode="json")) + "\n"
    assert output.read_text(encoding="utf-8") == expected


def test_T04_29_describe_summarizes_counts(tmp_path):
    output = tmp_path / "sample.json"
    run_cli("sample", "--config", "T2", "--seed", "7", "--out", str(output)).check_returncode()
    result = run_cli("describe", str(output))
    assert result.returncode == 0
    assert "12 stations" in result.stdout
    assert "14 trains" in result.stdout
    assert "possession=" in result.stdout and "speed_restriction=" in result.stdout


def test_T04_30_bad_preset_lists_valid_names(tmp_path):
    result = run_cli("sample", "--config", "T9", "--seed", "5", "--out", str(tmp_path / "x"))
    assert result.returncode != 0
    assert "T1" in result.stderr and "T2-S14" in result.stderr


def test_T04_26_candidates_print_in_order():
    result = run_cli("candidates", "--set", "heldout", "--count", "3")
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["1000000", "1000001", "1000002"]
