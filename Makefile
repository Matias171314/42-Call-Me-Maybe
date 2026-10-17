install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find src -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src -type f -name "*.py[co]" -delete
	rm -rf .mypy_cache

clean-venv:
	rm -rf .venv

lint:
	uv run flake8 . --extend-exclude=.venv,llm_sdk
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --extend-exclude=.venv,llm_sdk
	uv run mypy . --strict

.PHONY: install run debug clean clean-venv lint lint-strict
