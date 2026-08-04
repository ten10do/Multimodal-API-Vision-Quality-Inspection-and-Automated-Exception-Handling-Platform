.PHONY: dev test lint format migrate demo smoke-real-api \
	backend-lint-typecheck backend-tests frontend-lint-typecheck-tests \
	frontend-build e2e

dev:
	docker compose up --build

test:
	cd apps/api && pytest --cov=app --cov-report=term-missing
	cd apps/web && npm test -- --run

lint:
	cd apps/api && ruff check . && mypy app
	cd apps/web && npm run lint && npm run typecheck

backend-lint-typecheck:
	cd apps/api && ruff check . && mypy app

backend-tests:
	cd apps/api && pytest --cov=app --cov-report=term-missing --cov-report=xml

frontend-lint-typecheck-tests:
	cd apps/web && npm run lint && npm run typecheck && npm run test:coverage

frontend-build:
	cd apps/web && npm run build

e2e:
	cd apps/web && npm run test:e2e

format:
	cd apps/api && ruff format . && ruff check --fix .
	cd apps/web && npm run format

migrate:
	cd apps/api && alembic upgrade head

demo:
	cd apps/api && python ../../scripts/create_sample_images.py
	docker compose up --build

smoke-real-api:
	cd apps/api && python -m app.real_api_smoke
