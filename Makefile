.PHONY: *

VENV=.venv
PYTHON=$(VENV)/bin/python3

# ================== LOCAL WORKSPACE SETUP ==================
venv:
	@echo "=== Creating virtual environment ==="
	python -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

install:
	@echo "=== Installing requirements ==="
	pip install -r requirements.txt

install_dev:
	@echo "=== Installing requirements ==="
	pip install -r requirements-dev.txt

pre_commit_install:
	@echo "=== Installing pre-commit ==="
	$(PYTHON) -m pre_commit install

get_models:
	@echo "=== Downloading models ==="

# ================== RUN ==================
run:
	$(PYTHON) -m src.main

# ================== CONTINUOUS INTEGRATION =================
ci_static_code_analysis:
	$(PYTHON) -m pre_commit run --all-files
