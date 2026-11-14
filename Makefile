.PHONY: test lint ci spike-variants

test:
	uv run pytest -m "not harbor and not model and not gpu"

lint:
	uv run ruff check railway tests spike

spike-variants:
	uv run python spike/make_variants.py

ci: lint test
