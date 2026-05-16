.PHONY: train eval test clean lint format install

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Start training"
	@echo "  make eval       - Evaluate latest checkpoint"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Remove generated files"

install:
	pip install -r requirements.txt

train:
	python train.py

eval:
	python eval.py --checkpoint checkpoints/latest.pt

test:
	pytest -v --cov=.

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .

clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
