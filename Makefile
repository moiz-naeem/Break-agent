.PHONY: test lint ci oracle redteam spike-variants

test:
	uv run pytest -m "not harbor and not model and not gpu"

lint:
	uv run ruff check railway tests spike scripts

spike-variants:
	uv run python spike/make_variants.py

# T07 section 2.10: unit gate, then the oracle on datasets/t1-dev-mini; T07-H1 asserts 3 trials, every reward exactly 1.0.
oracle:
	uv run pytest tests/t07/test_harbor_runs.py::test_T07_H1_oracle_all_tasks -m harbor

# docs/redteam-playbook.md: the strategic battery and the verifier tamper suite, into results/redteam/.
redteam:
	uv run python -m railway.redteam.panel --preset T1 --seeds 5

ci: lint test oracle
