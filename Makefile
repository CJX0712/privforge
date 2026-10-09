# PrivForge Makefile —— 一键复现
.PHONY: install test lint format demo ci clean

install:
	pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest -q -W ignore::UserWarning

demo:
	python -m examples.run_demo

ci: lint test demo

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ examples/benchmark.json
