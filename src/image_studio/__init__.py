"""Image Studio backend: server, storage, Hub integration, and inference runtime.

The backend server, SQLite storage with migrations, Hugging Face discovery,
fixed-commit downloads, the generation queue with restart reconciliation,
artifact storage, the HTTP API, and the real supervisor/worker inference
runtime are implemented and covered by CPU-only tests. Real generation
additionally requires the optional inference extra and a GPU; the frontend
application is not implemented yet.
"""
