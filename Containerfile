# Keep Node and frontend build dependencies out of the application image.
FROM docker.io/library/node:26.10.0-bookworm-slim AS frontend
WORKDIR /build/frontend
RUN npm install --global pnpm@12.4.2
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

# Build and run the Python application in the same stage.
FROM docker.io/library/python:3.12-slim-bookworm AS application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_CACHE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
RUN pip install --no-cache-dir uv==0.10.10

WORKDIR /opt/image-studio
COPY pyproject.toml uv.lock README.md LICENSE ./
# Cache inference dependencies independently of application source changes.
RUN uv sync --locked --no-dev --no-install-project
COPY src/ ./src/
COPY --from=frontend /build/src/image_studio/web/static/ ./src/image_studio/web/static/
RUN uv sync --locked --no-dev --no-editable

USER app
EXPOSE 7860
STOPSIGNAL SIGTERM
ENTRYPOINT ["image-studio", "serve"]
# Publish this port on host loopback when running the container.
CMD ["--host", "0.0.0.0", "--port", "7860"]
