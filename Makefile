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

get_weights_csm:
	@echo "=== Downloading CSM models ==="
	huggingface-cli download unsloth/Llama-3.2-1B --local-dir data/weights/csm/Llama-3.2-1B
	huggingface-cli download 'sesame/csm-1b' --local-dir data/weights/csm/csm-1b

get_weights_ditto:
	@echo "=== Downloading Ditto models ==="
	huggingface-cli download digital-avatar/ditto-talkinghead --local-dir data/weights/ditto

# ================== RUN ==================
run:
	$(PYTHON) -m src.main

# ================== CONTINUOUS INTEGRATION =================
ci_static_code_analysis:
	$(PYTHON) -m pre_commit run --all-files
