# Shared image for the Python control-plane services. The service is selected with
# --build-arg PACKAGE, so api and orchestrator share every layer up to the final sync.
#
# Note this image is never used on the GPU host: the agent installs there via uv into the
# user's own vLLM environment, and invariant 3 keeps container runtimes off that machine.
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

ARG PACKAGE
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

# Workspace metadata first: dependency resolution is the slow layer and changes far less
# often than source does.
COPY pyproject.toml uv.lock VERSION ./
COPY packages/protocol/pyproject.toml ./packages/protocol/
COPY packages/db/pyproject.toml ./packages/db/
COPY packages/api/pyproject.toml ./packages/api/
COPY packages/orchestrator/pyproject.toml ./packages/orchestrator/
COPY packages/agent/pyproject.toml ./packages/agent/
COPY packages/mockagent/pyproject.toml ./packages/mockagent/

COPY packages ./packages
COPY scripts ./scripts

RUN uv sync --frozen --no-dev --package "${PACKAGE}"

ENV PATH="/app/.venv/bin:${PATH}"

# Non-root. Nothing here needs privileges, and the control plane is the piece most likely
# to be exposed on a network.
RUN useradd --create-home --uid 10001 vllmbench && chown -R vllmbench:vllmbench /app
USER vllmbench
