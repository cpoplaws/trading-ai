SHELL := /bin/bash

.PHONY: setup install install-dev test pipeline docker-build docker-up lint format type-check clean

setup:
	@echo "Running setup script..."
	./setup.sh

install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing dev dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	@echo "Running test suite..."
	python -m pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

pipeline:
	@echo "Running daily pipeline..."
	python src/execution/daily_retrain.py

docker-build:
	@echo "Building Docker images using docker-compose..."
	docker-compose build --pull --no-cache

docker-up:
	@echo "Starting services via docker-compose..."
	docker-compose up --build

docker-down:
	@echo "Stopping services..."
	docker-compose down

format:
	@echo "Formatting code with black and isort..."
	black src/ tests/
	isort src/ tests/

lint:
	@echo "Linting code with ruff..."
	ruff check src/ tests/

type-check:
	@echo "Type checking with mypy..."
	mypy src/

clean:
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage

help:
	@echo "Trading AI - Available Commands:"
	@echo ""
	@echo "  make setup          - Run initial setup script"
	@echo "  make install        - Install core dependencies"
	@echo "  make install-dev    - Install dev dependencies + pre-commit"
	@echo "  make test           - Run test suite"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make pipeline       - Run daily trading pipeline"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start services with Docker"
	@echo "  make docker-down    - Stop Docker services"
	@echo "  make format         - Format code (black + isort)"
	@echo "  make lint           - Lint code (ruff)"
	@echo "  make type-check     - Type check (mypy)"
	@echo "  make clean          - Remove generated files"
	@echo ""
