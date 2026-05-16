# Development and benchmark commands.
# Every Python invocation runs through the project venv created by `make install`.

PY := ./.venv/bin/python
PYTEST := ./.venv/bin/pytest

.PHONY: install test test-e2e bench compare lint format clean

## install — create a local venv and install the package (with dev extras) in editable mode
install:
	python3 -m venv .venv
	$(PY) -m pip install -U pip wheel
	$(PY) -m pip install -e ".[dev]"

## test — fast tests with no API spend (unit + contract + integration with mocks)
test:
	$(PYTEST) tests/unit tests/integration -m "not e2e and not benchmark"

## test-e2e — real-OpenAI smoke (requires OPENAI_API_KEY)
test-e2e:
	$(PYTEST) tests/e2e -m e2e

## bench — full run of one pipeline version over the 12-question subset
##         usage: make bench PIPELINE=v4_1_promptfix
bench:
	$(PY) -m ramdocs_rag.eval.runner --pipeline $(PIPELINE)

## compare — summary table across frozen pipeline versions
##           usage: make compare VERSIONS="v3_3_analyzer_tuned v4_1_promptfix"
compare:
	$(PY) -m ramdocs_rag.eval.compare $(VERSIONS)

## lint — ruff check
lint:
	$(PY) -m ruff check src tests

## format — ruff format
format:
	$(PY) -m ruff format src tests

## clean — remove caches and build artefacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/*.egg-info build dist
