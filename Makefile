# ENGINE_DIR points at the engine source (git submodule). Before the submodule exists, use
#   make ENGINE_DIR=../iirabm <target>
ENGINE_DIR ?= engine
export ENGINE_DIR

.DEFAULT_GOAL := help
.PHONY: help sync test lint format cohort figure

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

sync:  ## Create the virtual environment and install dependencies
	uv sync

test: sync  ## Run the Python tests
	uv run pytest

lint: sync  ## Check lint and formatting
	uv run ruff check .
	uv run ruff format --check .

format: sync  ## Apply formatting and safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

cohort: sync  ## Build the development (40) and test (100) virtual-patient cohorts locally
	IIRABM_BIN=$${IIRABM_BIN:-scripts/iirabm_docker.sh} uv run python -m twin.patients

figure: sync  ## Plot docs/figures/example_patient.png (needs the cohorts)
	uv run python scripts/plot_example_patient.py
