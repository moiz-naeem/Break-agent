import os
import subprocess
import sys

from railway.util import canonical_json, content_hash


def test_T01_05_canonical_json_key_order():
    assert canonical_json({"b": 1, "a": [2, {"d": 1, "c": 2}]}) == '{"a":[2,{"c":2,"d":1}],"b":1}'


def test_T01_06_canonical_json_unicode_and_floats():
    s = canonical_json({"x": "é", "y": 0.1})
    assert '"é"' in s and "\\u00e9" not in s
    assert "0.1" in s


def test_T01_07_content_hash_stable():
    a = {"x": 1, "y": [1, 2]}
    b = {"y": [1, 2], "x": 1}
    assert content_hash(a) == content_hash(b)
    assert len(content_hash(a)) == 64 and int(content_hash(a), 16) >= 0
    assert content_hash(a) != content_hash({"x": 2, "y": [1, 2]})


def test_T01_08_hash_across_processes():
    code = "from railway.util import content_hash; print(content_hash({'k': ['a', 1, 2.5], 'z': {'q': None}}))"
    outs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                   text=True, check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0].strip()) == 64
