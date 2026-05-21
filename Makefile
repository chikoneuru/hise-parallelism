# HISE testbed convenience targets

PY ?= python
PIP ?= pip

.PHONY: install dev test lint smoke up down logs exp01 exp02 exp03 clean

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e .[dev,rl,energy-api,docker]

test:
	pytest -ra

lint:
	ruff check hise tests experiments

smoke:
	$(PY) experiments/exp01_smoke_test.py

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f orchestrator

exp01:
	$(PY) experiments/exp01_smoke_test.py

exp02:
	$(PY) experiments/exp02_carbon_replay.py --trace traces/synthetic_solar.csv --hours 24

exp03:
	$(PY) experiments/exp03_elastic_reconfig.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
