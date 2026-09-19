# Image Studio task runner. Run `just` with no arguments to list recipes.

# List available recipes (default action).
default:
    @just --list

# Synchronize frontend dependencies without changing the lockfile.
frontend-deps:
    pnpm --dir frontend install --frozen-lockfile

# No inference extra, model weights, GPU setup, or background services.
# Install base and development dependencies and activate the Git hooks.
setup: frontend-deps
    uv sync --locked
    pnpm --dir tests/e2e install --frozen-lockfile
    prek install

# Synchronize dependencies and build into src/image_studio/web/static/.
frontend: frontend-deps
    pnpm --dir frontend run build

# Build the frontend, then the Python wheel and sdist into dist/.
build: frontend
    uv build

# Build the frontend and start the server with inference dependencies.
serve *args: frontend
    uv run --locked --extra inference image-studio serve {{args}}
