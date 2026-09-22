# Image Studio task runner. Run `just` with no arguments to list recipes.

# List available recipes (default action).
default:
    @just --list

# Synchronize frontend dependencies without changing the lockfile.
frontend-deps:
    pnpm --dir frontend install --frozen-lockfile

# Synchronize browser-test dependencies without changing the lockfile.
e2e-deps:
    pnpm --dir tests/e2e install --frozen-lockfile

# Install runtime and development dependencies and activate the Git hooks.
# No model weights, GPU execution, or background services.
setup: frontend-deps e2e-deps
    uv sync --locked
    prek install

# Synchronize dependencies and build into src/image_studio/web/static/.
frontend: frontend-deps
    pnpm --dir frontend run build

# Build the frontend, then the Python wheel and sdist into dist/.
build: frontend
    uv build

# Build the frontend and start the server with inference dependencies.
serve *args: frontend
    uv run --locked image-studio serve {{args}}

# Run end-to-end tests against a freshly built and installed wheel.
# Requires Playwright Chromium; all application data uses temporary directories.
e2e-check: build e2e-deps
    uv run --locked python scripts/run_integration_e2e.py
