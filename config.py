"""Central configuration for the AI Software Architect application.

This file exists so the app does not scatter paths, model names, sizing
settings, and file filters across the codebase. Any module that needs a shared
default should read it from here, which keeps the behavior consistent between
the UI, the indexing pipeline, and the retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration values.

    The fields below are deliberately grouped by concern so a reader can see
    which values affect the UI, which ones affect retrieval, and which ones are
    just filesystem defaults.
    """

    # Identity and model settings tell the UI what to display and which Ollama
    # model the backend should try first.
    app_name: str = "AI Software Architect"
    default_model: str = "qwen3.5:2b-q4_K_M"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval sizing controls the amount of text the parser creates and how
    # much of that text is sent back into the model prompt.
    chunk_size: int = 1600
    chunk_overlap: int = 250
    top_k: int = 8
    max_context_chunks: int = 12

    # These paths keep generated data inside the project folder so the app can
    # run locally without needing a separate deployment layout.
    database_path: Path = BASE_DIR / "database"
    chroma_path: Path = BASE_DIR / "database" / "chroma"
    upload_path: Path = BASE_DIR / "uploaded_projects"
    logs_path: Path = BASE_DIR / "logs"
    max_file_size_bytes: int = 2_000_000

    # These directories are intentionally skipped because they are typically
    # build artifacts, caches, or dependency folders that do not help code
    # understanding.
    ignored_dirs: tuple[str, ...] = (
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "build",
        "dist",
        "__pycache__",
        ".idea",
        ".vscode",
        ".cache",
    )
    # Binary and media files are skipped because the parser is designed for
    # source text, not opaque blobs.
    ignored_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
    )

    # These extensions are the common source and text formats that the parser
    # can safely break into retrieval chunks.
    supported_text_extensions: tuple[str, ...] = (
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".md",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
        ".txt",
        ".dockerfile",
    )

    # Some files matter more because their meaning depends on the filename even
    # when the extension is not enough to identify them.
    supported_special_names: tuple[str, ...] = (
        "dockerfile",
        "requirements.txt",
        "package.json",
        "pyproject.toml",
        "poetry.lock",
        "pdm.lock",
        "makefile",
        "readme.md",
        "readme",
    )

    # Answer-generation controls keep the model output short, stable, and less
    # likely to ramble.
    llm_temperature: float = 0.1
    reflection_max_rounds: int = 2

    # Security heuristics use these keywords to look for risky patterns in a
    # fast, deterministic way.
    security_keywords: tuple[str, ...] = (
        "secret",
        "password",
        "api_key",
        "apikey",
        "token",
        "jwt",
        "debug",
        "unsafe",
    )
    max_answer_chars: int = 16_000
    planner_max_tools: int = 6
    show_agent_logs_default: bool = True
    app_theme: str = "dark"

    # Metadata fields are preserved so retrieval can filter by repository and
    # also explain which file or symbol produced a chunk.
    metadata_fields: tuple[str, ...] = (
        "repository",
        "file_path",
        "language",
        "class_name",
        "function_name",
        "chunk_type",
        "extension",
    )
    extra_fields: dict[str, str] = field(default_factory=dict)


settings = AppConfig()
