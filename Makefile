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

help:
	@printf '%s\n' \
		'make install      Install project dependencies with uv' \
		'make run          Process requests using the default paths' \
		'make debug        Run the CLI under the Python debugger' \
		'make lint         Check style and the required mypy flags' \
		'make lint-strict  Check style and strict typing' \
		'make clean        Remove application bytecode and mypy cache' \
		'make clean-venv   Remove the virtual environment' \
		'make help         Show these commands' \
		'' \
		'Custom paths: make run ARGS="--input prompts.json --output results.json"' \
		'The same ARGS option is supported by make debug.'

.PHONY: install run debug clean clean-venv lint lint-strict help
