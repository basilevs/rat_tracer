.DEFAULT_GOAL := help

PYTHON ?= python3

.PHONY: install-dev  ## Install the package with development tooling
install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: format  ## Auto-format and auto-fix with Ruff
format:
	$(PYTHON) -m ruff format rat_tracer tests
	$(PYTHON) -m ruff check --fix rat_tracer tests

.PHONY: lint  ## Lint with Ruff (no changes)
lint:
	$(PYTHON) -m ruff check rat_tracer tests
	$(PYTHON) -m ruff format --check rat_tracer tests

.PHONY: typecheck  ## Static type-check with mypy
typecheck:
	$(PYTHON) -m mypy

.PHONY: test  ## Run the offline test suite
test:
	$(PYTHON) -m pytest -m "not network"

.PHONY: check  ## Run all checks (lint, typecheck, test)
check: lint typecheck test

.PHONY: help  ## Display this help message
help:
	@grep -E '^.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-15s\033[0m %s\n", $$2, $$3}'
