.PHONY: test lint ci oracle spike-variants

test:
	uv run pytest -m "not harbor and not model and not gpu"

lint:
	uv run ruff check railway tests spike scripts

spike-variants:
	uv run python spike/make_variants.py

# T07 section 2.10: unit gate, then the oracle on datasets/t1-dev-v03; T07-H1 asserts 5 trials, every reward exactly -c_best.
oracle:
	uv run pytest tests/t07/test_harbor_runs.py::test_T07_H1_oracle_all_tasks -m harbor

ci: lint test oracle
