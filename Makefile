.PHONY: help install lint format typecheck test check clean serve bench eval chart docker-cpu docker-cuda

PYTHON := uv run python
CONFIG ?= bench-smoke

help:
	@echo "Forge — common tasks"
	@echo ""
	@echo "  install      Install dependencies via uv"
	@echo "  lint         Ruff lint check"
	@echo "  format       Ruff format (writes changes)"
	@echo "  typecheck    mypy"
	@echo "  test         pytest"
	@echo "  check        lint + typecheck + test (mirrors CI)"
	@echo ""
	@echo "  serve        Launch vLLM OpenAI-compatible server"
	@echo "  bench        Run benchmark harness (CONFIG=bench-smoke|bench-full)"
	@echo "  eval         Run quality eval"
	@echo "  chart        Rebuild all charts from results/"
	@echo ""
	@echo "  docker-cpu   Build CPU vLLM image (Dockerfile.cpu) for M1 dev"
	@echo "  docker-cuda  Build production CUDA image (Dockerfile)"
	@echo ""
	@echo "  clean        Remove caches and build artefacts"

install:
	uv sync --all-extras --all-groups

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

serve:
	$(PYTHON) -m scripts.serve

bench:
	$(PYTHON) -m scripts.bench --config configs/$(CONFIG).yaml

eval:
	$(PYTHON) -m scripts.eval --config configs/$(CONFIG).yaml

chart:
	$(PYTHON) -m scripts.chart

docker-cpu:
	docker build -f Dockerfile.cpu -t forge:cpu .

docker-cuda:
	docker build -f Dockerfile -t forge:cuda .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
