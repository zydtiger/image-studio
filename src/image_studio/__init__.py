"""Image Studio backend: server, storage, Hub integration, and inference runtime.

The backend server, SQLite storage with migrations, Hugging Face discovery,
fixed-commit downloads, the generation queue with restart reconciliation,
artifact storage, the HTTP API, and the real supervisor/worker inference
runtime are implemented and covered by CPU-only tests. Inference dependencies
are installed by default; real generation additionally requires model weights
and a GPU. The server also serves the built frontend application.
"""
