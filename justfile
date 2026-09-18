# Image Studio task runner. Run `just` with no arguments to list recipes.

# List available recipes (default action).
default:
    @just --list

# No inference extra, model weights, GPU setup, or background services.
# Install base and development dependencies and activate the Git hooks.
setup:
    uv sync --locked
    npm --prefix frontend ci
    npm --prefix tests/e2e ci
    prek install

# Build the frontend into src/image_studio/web/static/.
frontend:
    npm --prefix frontend run build

# Dependencies come from a prior `just setup`; no setup or npm ci runs here.
# Build the frontend, then the Python wheel and sdist into dist/.
build: frontend
    uv build

# Start the server with inference dependencies.
serve *args:
    uv run --locked --extra inference image-studio serve {{args}}
