# Variables
APP_NAME := openchemie_rest
PORT := 8000

# Set default shell
SHELL := /bin/bash

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  make install        Install poetry and project dependencies"
	@echo "  make run            Run uvicorn server for dev"
	@echo "  make build          Build Docker image"
	@echo "  make run-container  Run Docker container"
	@echo "  make stop           Stop Docker container"
	@echo "  make test           Run tests"
	@echo "  make lint           Lint the code using flake8 or black"
	@echo "  make format         Format code using black or similar tool"
	@echo "  make clean          Remove build artifacts, dist files, and virtualenv"
	@echo "  make help           Show this help message"

install:
	pyenv install -s 3.9
	poetry env use 3.9
	poetry install

run:
	poetry run uvicorn app:app --reload --host 0.0.0.0 --port $(PORT)

build:
	docker build -t $(APP_NAME):latest .

run-container:
	docker run -p $(PORT):$(PORT) $(APP_NAME):latest

stop:
	docker stop $$(docker ps -q --filter ancestor=$(APP_NAME):latest)

test:
	poetry run pytest tests

lint:
	poetry run flake8 .

format:
	poetry run black .

clean:
	poetry env remove -q || true
	find . -type d -name '__pycache__' -exec rm -r {} +
	docker system prune -f
