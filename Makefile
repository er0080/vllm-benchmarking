.DEFAULT_GOAL := help
SHELL := /bin/bash

# Everything here must work identically on native Linux and macOS with Colima.
# See CLAUDE.md, "Platform support".

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install Python and web dependencies
	uv sync --all-packages
	cd web && npm ci

.PHONY: hooks
hooks: ## Install pre-commit hooks
	uv run pre-commit install

# ---------------------------------------------------------------------------
# Quality — tier 1, no services required
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Lint and format-check
	uv run ruff check .
	uv run ruff format --check .

.PHONY: format
format: ## Apply formatting and safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: types
types: ## Type-check Python and TypeScript
	uv run pyright
	cd web && npx tsc --noEmit

.PHONY: versions
versions: ## Check package versions match VERSION
	python3 scripts/check_versions.py

.PHONY: test
test: ## Run unit tests (no services)
	uv run pytest -m "not integration and not vllm_cpu"

.PHONY: check
check: lint types versions test ## Everything CI tier 1 runs

# ---------------------------------------------------------------------------
# Quality — tier 2, requires services
# ---------------------------------------------------------------------------

.PHONY: test-db
test-db: ## Create and migrate vllmbench_test, which the integration suite empties
	docker compose exec -T postgres sh -c \
		'psql -U "$$POSTGRES_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '"'"'vllmbench_test'"'"'" \
		| grep -q 1 || createdb -U "$$POSTGRES_USER" vllmbench_test'
	docker compose run --rm migrate sh -c \
		'DATABASE_URL="$${DATABASE_URL%/*}/vllmbench_test" alembic upgrade head'

# DATABASE_URL is built here rather than left to the code default, which cannot know the
# password in .env. Pointed at vllmbench_test deliberately: this suite empties every table
# in whatever it is given.
.PHONY: test-integration
test-integration: test-db ## Run integration tests (empties vllmbench_test, never vllmbench)
	set -a; [ -f .env ] && . ./.env; set +a; \
	DATABASE_URL="postgresql+psycopg://$${POSTGRES_USER:-vllmbench}:$${POSTGRES_PASSWORD:-vllmbench}@localhost:$${POSTGRES_PORT:-5432}/vllmbench_test" \
	uv run pytest -m integration

.PHONY: test-vllm-cpu
test-vllm-cpu: ## Run tests against the real vLLM CPU backend container
	uv run pytest -m vllm_cpu

# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

.PHONY: up
up: ## Start the stack
	docker compose up -d --build

.PHONY: dev
dev: ## Start the stack with the mock agent, no GPU host needed
	docker compose --profile dev up -d --build

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean
clean: ## Stop the stack and delete its volumes
	docker compose down -v

.PHONY: logs
logs: ## Follow stack logs
	docker compose logs -f

.PHONY: migrate
migrate: ## Apply database migrations
	docker compose run --rm migrate
