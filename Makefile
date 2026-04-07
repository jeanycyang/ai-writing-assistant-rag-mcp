PYTHON ?= python3.12

.PHONY: venv install dev-install up down logs migrate ingest-sample test

venv:
	test -d venv || $(PYTHON) -m venv venv

install: venv
	source venv/bin/activate && pip install -r requirements.txt

dev-install: venv
	source venv/bin/activate && pip install -r requirements-dev.txt

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	source venv/bin/activate && alembic upgrade head

ingest-sample:
	source venv/bin/activate && python scripts/ingest_data.py --summary-dir data/sample/summaries --raw-dir data/sample/raw

test:
	source venv/bin/activate && pytest
