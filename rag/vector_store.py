"""ChromaDB-backed vector store.

This module is the repository memory of the app. It stores parsed chunks,
attaches metadata for filtering and citations, and provides semantic search so
the question-answering layer can retrieve the most relevant evidence.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from math import sqrt
from typing import Any

try:
    from chromadb import PersistentClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    PersistentClient = None  # type: ignore[assignment]

from config import settings
from models.schemas import ChunkRecord
from rag.embeddings import EmbeddingProvider


logger = logging.getLogger(__name__)


class _InMemoryCollection:
    """Small compatibility layer when ChromaDB is unavailable."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        # Keep embeddings in memory so retrieval still works even without Chroma.
        self.embedding_provider = embedding_provider
        self._items: list[dict[str, Any]] = []

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        # Replace duplicates by ID so re-indexing the same file stays stable.
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self._items = [item for item in self._items if item["id"] != item_id]
            self._items.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

    def delete(self, ids: list[str]) -> None:
        # Remove all chunks for a repository before rebuilding its index.
        self._items = [item for item in self._items if item["id"] not in ids]

    def get(self, where: dict | None = None, include: list[str] | None = None) -> dict[str, list[Any]]:
        # Chroma-like get() contract used by the rest of the app.
        ids = []
        documents = []
        metadatas = []
        for item in self._items:
            if where and any(item["metadata"].get(key) != value for key, value in where.items()):
                continue
            ids.append(item["id"])
            if include and "documents" in include:
                documents.append(item["document"])
            if include and "metadatas" in include:
                metadatas.append(item["metadata"])
        result: dict[str, list[Any]] = {"ids": ids}
        if include and "documents" in include:
            result["documents"] = documents
        if include and "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        # Support both query embeddings and raw query text to mirror Chroma's API shape.
        if query_embeddings:
            query_embedding = query_embeddings[0]
        elif query_texts:
            query_embedding = self.embedding_provider.embed_query(query_texts[0])
        else:
            query_embedding = []
        matches: list[tuple[float, dict[str, Any]]] = []
        for item in self._items:
            if where and any(item["metadata"].get(key) != value for key, value in where.items()):
                continue
            score = self._cosine_similarity(query_embedding, item["embedding"])
            matches.append((score, item))
        matches.sort(key=lambda pair: pair[0], reverse=True)
        top_matches = matches[:n_results]
        return {
            "documents": [[item["document"] for _, item in top_matches]],
            "metadatas": [[item["metadata"] for _, item in top_matches]],
            "distances": [[1.0 - score for score, _ in top_matches]],
        }

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        # Standard cosine similarity gives a simple nearest-neighbor ranking.
        if not left or not right:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)


class VectorStore:
    """Store and search repository chunks."""

    def __init__(self, db_path: Path | None = None, embedding_provider: EmbeddingProvider | None = None) -> None:
        # Create the on-disk database directory up front so Chroma can open it safely.
        self.db_path = db_path or settings.chroma_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        if PersistentClient is None:
            # Offline or partially installed environments use the in-memory fallback.
            logger.warning("chromadb is unavailable; using in-memory vector store fallback")
            self.client = None
            self.collection = _InMemoryCollection(self.embedding_provider)
        else:
            try:
                # PersistentClient stores vectors in the local Chroma database folder.
                self.client = PersistentClient(path=str(self.db_path))
                self.collection = self.client.get_or_create_collection(
                    name="ai_software_architect",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning("ChromaDB unavailable, using in-memory vector store fallback: %s", exc)
                self.client = None
                self.collection = _InMemoryCollection(self.embedding_provider)

    def reset_repository(self, repository: str) -> None:
        # Remove all previously indexed chunks for this repository before rebuilding.
        existing = self.collection.get(where={"repository": repository})
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        # Embed each chunk and store it alongside file metadata for later citation.
        if not chunks:
            return
        ids = [f"{chunk.repository}:{chunk.file_path}:{idx}" for idx, chunk in enumerate(chunks)]
        documents = [chunk.content for chunk in chunks]
        metadatas = [self._metadata(chunk) for chunk in chunks]
        embeddings = self.embedding_provider.embed_documents(documents)
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def repository_overview(self, repository: str, top_k: int = 20) -> list[ChunkRecord]:
        # Build a file-level summary so architecture questions can use structure
        # instead of random chunks.
        data = self.collection.get(where={"repository": repository}, include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        grouped: dict[str, dict[str, Any]] = {}
        for document, metadata in zip(documents, metadatas):
            file_path = metadata.get("file_path", "")
            if not file_path:
                continue
            entry = grouped.setdefault(
                file_path,
                {
                    "repository": metadata.get("repository", repository),
                    "file_path": file_path,
                    "language": metadata.get("language", ""),
                    "chunk_types": set(),
                    "symbols": set(),
                    "sample": document,
                },
            )
            entry["chunk_types"].add(metadata.get("chunk_type", "text"))
            symbol = metadata.get("function_name") or metadata.get("class_name")
            if symbol:
                entry["symbols"].add(symbol)
            if not entry.get("sample"):
                entry["sample"] = document

        def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
            # Put entrypoint-like files ahead of deep helper files.
            file_path, _ = item
            name = Path(file_path).name.lower()
            special = {
                "readme.md",
                "readme",
                "app.py",
                "main.py",
                "index.py",
                "package.json",
                "requirements.txt",
                "pyproject.toml",
                "dockerfile",
            }
            priority = 0 if name in special else 1
            depth = max(file_path.count("/"), file_path.count("\\"))
            return (priority, depth, file_path.lower())

        overview: list[ChunkRecord] = []
        for file_path, entry in sorted(grouped.items(), key=sort_key)[:top_k]:
            # Each summary chunk describes one file, its language, and the
            # symbols found in it.
            chunk_types = ", ".join(sorted(entry["chunk_types"])) or "text"
            symbols = ", ".join(sorted(entry["symbols"]))
            content_lines = [
                f"File: {file_path}",
                f"Language: {entry['language'] or 'Unknown'}",
                f"Chunk types: {chunk_types}",
            ]
            if symbols:
                content_lines.append(f"Symbols: {symbols}")
            sample = str(entry.get("sample", "")).strip()
            if sample:
                content_lines.append(f"Sample: {sample[:250]}")
            overview.append(
                ChunkRecord(
                    repository=entry["repository"],
                    file_path=file_path,
                    language=entry["language"] or "Unknown",
                    chunk_type="structure",
                    content="\n".join(content_lines),
                )
            )
        return overview

    def search(self, query: str, top_k: int = 5, metadata_filter: dict | None = None, boost_terms: list[str] | None = None) -> list[ChunkRecord]:
        # Use query embeddings so the query space always matches the stored
        # document space.
        query_embedding = self.embedding_provider.embed_query(query)
        fetch_k = max(top_k * 3, top_k + 5)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        terms = self._normalize_terms(query, boost_terms)
        ranked: list[tuple[float, float, dict[str, Any], str]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            base_score = float(1.0 - distance)
            adjusted_score = self._boost_score(base_score, document, metadata, terms)
            ranked.append((adjusted_score, base_score, metadata, document))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chunks: list[ChunkRecord] = []
        for adjusted_score, _, metadata, document in ranked[:top_k]:
            # Convert raw vector store rows back into the app's typed chunk
            # model.
            chunks.append(
                ChunkRecord(
                    repository=metadata.get("repository", ""),
                    file_path=metadata.get("file_path", ""),
                    language=metadata.get("language", ""),
                    class_name=metadata.get("class_name"),
                    function_name=metadata.get("function_name"),
                    chunk_type=metadata.get("chunk_type", "text"),
                    content=document,
                    score=float(adjusted_score),
                )
            )
        return chunks

    def _normalize_terms(self, query: str, boost_terms: list[str] | None = None) -> list[str]:
        # Normalize query text into a compact set of keywords for ranking.
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "be",
            "can",
            "could",
            "for",
            "from",
            "how",
            "is",
            "it",
            "me",
            "of",
            "on",
            "please",
            "show",
            "that",
            "the",
            "this",
            "to",
            "what",
            "when",
            "where",
            "which",
            "why",
            "with",
            "would",
            "you",
        }
        terms: list[str] = []
        seen: set[str] = set()
        for source in [query, *(boost_terms or [])]:
            for raw in re.findall(r"[A-Za-z0-9_]+", source.lower()):
                if len(raw) < 2 or raw in stop_words or raw in seen:
                    continue
                seen.add(raw)
                terms.append(raw)
        return terms

    def _boost_score(self, base_score: float, document: str, metadata: dict[str, Any], terms: list[str]) -> float:
        # Promote files and symbols that match the query more directly.
        path = str(metadata.get("file_path", "")).lower()
        class_name = str(metadata.get("class_name", "")).lower()
        function_name = str(metadata.get("function_name", "")).lower()
        chunk_type = str(metadata.get("chunk_type", "")).lower()
        content = document.lower()
        boost = 0.0
        for term in terms[:12]:
            if term in path:
                boost += 0.10
            elif term in class_name or term in function_name:
                boost += 0.08
            elif term in content:
                boost += 0.03
        if any(keyword in path for keyword in ("readme", "docs", "guide")) and any(
            term in terms for term in ("readme", "docs", "documentation", "guide", "overview", "setup", "usage", "install")
        ):
            boost += 0.14
        if any(keyword in path for keyword in ("config", "settings", "env", "package.json", "requirements", "pyproject", "poetry.lock")) and any(
            term in terms for term in ("config", "configuration", "settings", "environment", "env", "dependencies", "requirements", "package", "install")
        ):
            boost += 0.12
        if any(keyword in path for keyword in ("app.py", "main.py", "index.py", "server.py", "manage.py")) and any(
            term in terms for term in ("architecture", "structure", "flow", "entrypoint", "overview", "app", "main")
        ):
            boost += 0.10
        if chunk_type == "structure":
            boost += 0.05
        return min(0.999, base_score + boost)

    def _metadata(self, chunk: ChunkRecord) -> dict:
        # Keep only the fields that are useful for filtering and citations.
        return {
            "repository": chunk.repository,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "class_name": chunk.class_name or "",
            "function_name": chunk.function_name or "",
            "chunk_type": chunk.chunk_type,
        }
