"""Source code parser and chunker.

The parser turns raw file content into structured chunks that the vector store
can index. Its job is to preserve useful semantic boundaries such as classes,
functions, and configuration sections instead of treating every file like one
big blob of text.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from config import settings
from models.schemas import ChunkRecord


@dataclass
class ParseResult:
    """Container for the chunks produced by a parse operation."""

    chunks: list[ChunkRecord]


class CodeParser:
    """Split repository files into semantically meaningful chunks."""

    def parse_file(self, repository: str, path: Path, content: str) -> ParseResult:
        # Route each file to a parser based on its language or special filename.
        language = self._detect_language(path)
        if language == "Python":
            return ParseResult(self._parse_python(repository, path, content))
        if path.name.lower() == "package.json":
            return ParseResult(self._parse_package_json(repository, path, content))
        if path.suffix.lower() in {".md", ".json", ".yml", ".yaml", ".toml", ".txt"}:
            return ParseResult(self._parse_text(repository, path, content, language))
        return ParseResult(self._chunk_text(repository, path, content, language))

    def _detect_language(self, path: Path) -> str:
        return {
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
        }.get(path.suffix.lower(), "Text")

    def _parse_python(self, repository: str, path: Path, content: str) -> list[ChunkRecord]:
        # Python files get AST-aware chunks so classes and functions stay separate.
        chunks: list[ChunkRecord] = []
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # Capture the full class block first, then drill into methods
                    # so the retriever can answer both structural and local
                    # implementation questions.
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    class_source = "\n".join(lines[start:end])
                    chunks.append(self._chunk(repository, path, "Python", class_source, "class", node.name, None, node.lineno, end))
                    for inner in node.body:
                        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            istart = inner.lineno - 1
                            iend = getattr(inner, "end_lineno", inner.lineno)
                            fn_source = "\n".join(lines[istart:iend])
                            chunks.append(self._chunk(repository, path, "Python", fn_source, "method", node.name, inner.name, inner.lineno, iend))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Top-level functions become standalone retrieval chunks.
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    fn_source = "\n".join(lines[start:end])
                    chunks.append(self._chunk(repository, path, "Python", fn_source, "function", None, node.name, node.lineno, end))
            if not chunks:
                chunks.extend(self._chunk_text(repository, path, content, "Python"))
        except SyntaxError:
            chunks.extend(self._chunk_text(repository, path, content, "Python"))
        return chunks

    def _parse_package_json(self, repository: str, path: Path, content: str) -> list[ChunkRecord]:
        # package.json is stored as a configuration chunk plus dependency/script summaries.
        chunks = [self._chunk(repository, path, "JSON", content, "configuration", None, None)]
        try:
            data = json.loads(content)
            for key in ("dependencies", "devDependencies", "scripts"):
                if key in data:
                    chunks.append(self._chunk(repository, path, "JSON", json.dumps({key: data[key]}, indent=2), "configuration", None, key))
        except json.JSONDecodeError:
            pass
        return chunks

    def _parse_text(self, repository: str, path: Path, content: str, language: str) -> list[ChunkRecord]:
        return self._chunk_text(repository, path, content, language)

    def _chunk_text(self, repository: str, path: Path, content: str, language: str) -> list[ChunkRecord]:
        # Non-Python text is split into fixed-size windows for retrieval.
        if not content.strip():
            return []
        pieces = self._split_by_size(content, settings.chunk_size, settings.chunk_overlap)
        return [
            self._chunk(repository, path, language, piece, "text", None, None)
            for piece in pieces
        ]

    def _split_by_size(self, content: str, size: int, overlap: int) -> list[str]:
        # Sliding window splitting keeps nearby lines together for better context.
        if len(content) <= size:
            return [content]
        chunks: list[str] = []
        start = 0
        while start < len(content):
            end = min(len(content), start + size)
            chunks.append(content[start:end])
            if end == len(content):
                break
            start = max(0, end - overlap)
        return chunks

    def _chunk(
        self,
        repository: str,
        path: Path,
        language: str,
        content: str,
        chunk_type: str,
        class_name: str | None,
        function_name: str | None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ChunkRecord:
        # Wrap chunk metadata in a normalized schema for storage and retrieval.
        return ChunkRecord(
            repository=repository,
            file_path=str(path),
            language=language,
            class_name=class_name,
            function_name=function_name,
            chunk_type=chunk_type,
            content=content,
            start_line=start_line,
            end_line=end_line,
        )
