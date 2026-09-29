.PHONY: lint format

lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff check --fix .
