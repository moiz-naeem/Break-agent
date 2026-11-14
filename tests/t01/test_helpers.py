from tests.helpers import variant


def test_T01_09_variant_deep_copies():
    d = {"a": {"b": 1, "c": [1]}}
    v = variant(d, **{"a.b": 5})
    assert v["a"]["b"] == 5
    assert d["a"]["b"] == 1
    v["a"]["c"].append(2)
    assert d["a"]["c"] == [1]


def test_T01_10_variant_list_index():
    d = {"trains": [{"pax": 1}, {"pax": 2}]}
    v = variant(d, **{"trains.0.pax": 7})
    assert v["trains"][0]["pax"] == 7
    assert d["trains"][0]["pax"] == 1
