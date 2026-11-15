.PHONY: test test-modal lint ci oracle spike-variants

test:
	uv run pytest -m "not harbor and not model and not gpu"

# The same suite sharded over Modal containers: ~3 min instead of ~48. Needs `modal setup` once.
test-modal:
	uv run --with modal modal run scripts/modal_tests.py

lint:
	uv run ruff check railway tests archive/spike scripts

spike-variants:
	uv run python archive/spike/make_variants.py

# T07 section 2.10: unit gate, then the oracle on archive/datasets/t1-dev-mini; T07-H1 asserts 3 trials, every reward exactly 1.0.
oracle:
	uv run pytest tests/t07/test_harbor_runs.py::test_T07_H1_oracle_all_tasks -m harbor

ci: lint test oracle
