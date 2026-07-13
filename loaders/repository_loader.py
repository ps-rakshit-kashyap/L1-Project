"""Repository discovery and indexing support.

The loader is responsible for finding readable files in an extracted
repository, filtering out things we do not want to index, and attaching basic
metadata such as language and file size. It does not read file contents; that
job belongs to the repository service and parser.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from models.schemas import FileRecord


LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "React",
    ".ts": "TypeScript",
    ".tsx": "React",
    ".md": "Markdown",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".txt": "Text",
}


@dataclass
class RepositorySummary:
    """Compact inventory of the files discovered in a repository."""

    repository: str
    root: Path
    files: list[FileRecord] = field(default_factory=list)


class RepositoryLoader:
    """Scan repositories and build file metadata."""

    def __init__(self, ignored_dirs: tuple[str, ...] | None = None) -> None:
        # Use the shared ignore list unless a caller wants to override it for tests.
        self.ignored_dirs = ignored_dirs or settings.ignored_dirs

    def scan(self, repository_root: Path) -> RepositorySummary:
        # Walk the repository and build a compact inventory of readable files.
        files: list[FileRecord] = []
        for path in self._iter_files(repository_root):
            try:
                # Skip directories and inaccessible entries before reading
                # metadata so a single bad file does not stop the scan.
                if not path.is_file():
                    continue
                if self._ignored(path):
                    continue
                if path.suffix.lower() in settings.ignored_extensions:
                    continue
                size = path.stat().st_size
                relative_path = str(path.relative_to(repository_root))
            except (OSError, PermissionError, ValueError):
                continue
            if size > settings.max_file_size_bytes:
                continue
            # Attach a language label so later stages can decide how to chunk
            # the file.
            language = self.detect_language(path)
            files.append(
                FileRecord(
                    file_name=path.name,
                    file_path=relative_path,
                    extension=path.suffix.lower(),
                    size=size,
                    language=language,
                )
            )
        return RepositorySummary(
            repository=repository_root.name,
            root=repository_root,
            files=files,
        )

    def detect_language(self, path: Path) -> str:
        # Special filenames matter even when they have no obvious extension.
        name = path.name.lower()
        if name in settings.supported_special_names:
            if name == "package.json":
                return "JSON"
            if name == "requirements.txt":
                return "Text"
            if name == "dockerfile":
                return "Docker"
            if name.endswith("readme"):
                return "Markdown"
        return LANGUAGE_MAP.get(path.suffix.lower(), "Unknown")

    def _ignored(self, path: Path) -> bool:
        # Ignore any path that contains a known excluded directory segment.
        return any(part in self.ignored_dirs for part in path.parts)

    def _iter_files(self, repository_root: Path) -> Iterator[Path]:
        # os.walk is more permission-tolerant on Windows than recursive Path.rglob.
        for dirpath, dirnames, filenames in os.walk(repository_root, topdown=True, onerror=lambda _: None):
            current = Path(dirpath)
            dirnames[:] = [name for name in dirnames if name not in self.ignored_dirs]
            for filename in filenames:
                yield current / filename
