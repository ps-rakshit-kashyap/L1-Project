"""LangChain-compatible search tools.

This module provides the retrieval helpers that the planner calls. The methods
here are intentionally small wrappers around the vector store so the higher
level orchestration code can treat them like tools rather than raw database
calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings
from models.schemas import ChunkRecord
from rag.vector_store import VectorStore


@dataclass
class SearchTools:
    """Bundle the repository search operations used by the planner."""

    store: VectorStore

    def search_code(self, query: str, repository: str | None = None):
        # General code search for implementation details.
        return self._multi_search(query, repository, kind="code")

    def search_readme(self, query: str, repository: str | None = None):
        # Documentation search for overview and setup notes.
        return self._multi_search(query, repository, kind="readme")

    def search_configuration(self, query: str, repository: str | None = None):
        # Configuration search helps answer environment and setup questions.
        return self._multi_search(query, repository, kind="configuration")

    def search_routes(self, query: str, repository: str | None = None):
        # Route search is mainly useful for auth and web API exploration.
        return self._multi_search(query, repository, kind="routes")

    def list_project_structure(self, repository: str):
        # Structure questions use file-level summaries instead of raw semantic hits.
        return self.store.repository_overview(repository, top_k=settings.top_k * 4)

    def find_dependencies(self, repository: str):
        # Pull dependency-related files and chunks for package analysis.
        return self._multi_search(
            "dependencies package requirements poetry package.json pyproject install",
            repository,
            kind="dependencies",
        )

    def retrieve_documentation(self, repository: str):
        # Locate README-like content and any other architecture notes.
        return self._multi_search(
            "architecture documentation readme guide overview setup usage install",
            repository,
            kind="readme",
        )

    def generate_architecture(self, repository: str):
        # Architecture questions need the summarized file map rather than arbitrary chunks.
        return self.store.repository_overview(repository, top_k=settings.top_k * 4)

    def find_security_issues(self, repository: str):
        # Security scanning starts with likely-secret and validation-related files.
        return self._multi_search(
            "security secret token password auth validate sanitize injection",
            repository,
            kind="security",
        )

    def summarize_module(self, repository: str, module_path: str):
        # Module summaries help explain one file at a time.
        return self._multi_search(module_path, repository, kind="module")

    def _multi_search(self, query: str, repository: str | None = None, kind: str = "code"):
        # Run a small set of query variants, merge overlaps, and keep the best hit per chunk.
        variants = self._query_variants(query, kind)
        ranked: dict[str, ChunkRecord] = {}
        where = self._where(repository)
        boost_terms = self._boost_terms(query, kind)
        for variant in variants:
            for chunk in self.store.search(variant, top_k=settings.top_k, metadata_filter=where, boost_terms=boost_terms):
                key = self._chunk_key(chunk)
                current = ranked.get(key)
                if current is None or chunk.score > current.score:
                    ranked[key] = chunk
        ordered = sorted(ranked.values(), key=lambda chunk: chunk.score, reverse=True)
        return ordered[: settings.top_k]

    def _query_variants(self, query: str, kind: str) -> list[str]:
        # Expand the query with a small number of targeted retrieval hints.
        kind_hints = {
            "code": "implementation code function class method logic",
            "readme": "readme documentation overview setup usage install",
            "configuration": "configuration environment variables settings secrets dependencies",
            "routes": "routes endpoints middleware auth login api",
            "dependencies": "dependencies package install requirements poetry package.json pyproject",
            "security": "security validation sanitize auth jwt token secret password injection",
            "module": "module file implementation overview",
        }
        hints = kind_hints.get(kind, "")
        variants = [query]
        if hints:
            variants.append(f"{query} {hints}")
        return list(dict.fromkeys(variant.strip() for variant in variants if variant.strip()))

    def _boost_terms(self, query: str, kind: str) -> list[str]:
        # Add focused terms so vector search can favor obvious file and symbol matches.
        kind_terms = {
            "code": ["code", "implementation", "function", "class", "method", "logic"],
            "readme": ["readme", "documentation", "overview", "setup", "usage", "install"],
            "configuration": ["config", "configuration", "settings", "environment", "env", "dependencies", "requirements"],
            "routes": ["route", "routes", "endpoint", "middleware", "auth", "login", "api"],
            "dependencies": ["dependencies", "package", "install", "requirements", "poetry", "pyproject", "package.json"],
            "security": ["security", "validation", "sanitize", "auth", "jwt", "token", "secret", "password", "injection"],
            "module": ["module", "file", "implementation", "overview"],
        }
        return [query, *kind_terms.get(kind, [])]

    def _chunk_key(self, chunk: ChunkRecord):
        # Use a stable signature so repeated semantic hits collapse cleanly.
        return (chunk.file_path, chunk.class_name, chunk.function_name, chunk.chunk_type, chunk.content)

    def _where(self, repository: str | None, file_path_contains: str | None = None) -> dict | None:
        # Metadata filters keep retrieval locked to the current uploaded repository.
        if not repository and not file_path_contains:
            return None
        where: dict = {}
        if repository:
            where["repository"] = repository
        return where
