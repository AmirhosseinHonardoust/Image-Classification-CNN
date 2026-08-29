.PHONY: install install-dev lint format typecheck test check train evaluate

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check src tests

format:
	black src tests

typecheck:
	mypy src

test:
	pytest

check: lint
	black --check src tests
	$(MAKE) typecheck
	$(MAKE) test

train:
	train-cnn --config train_config.example.json

evaluate:
	evaluate-cnn --model outputs/best_cnn.pth --dataset mnist --outdir outputs
