"""Tests for retrieval query expansion and ranking."""

from dataclasses import dataclass

from models.schemas import ChunkRecord
from tools.search_tools import SearchTools


@dataclass
class FakeStore:
    calls: list[tuple] = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def search(self, query, top_k=5, metadata_filter=None, boost_terms=None):
        self.calls.append((query, top_k, metadata_filter, tuple(boost_terms or [])))
        lower = query.lower()
        if "readme" in lower or "documentation" in lower or "overview" in lower:
            return [
                ChunkRecord(repository="repo", file_path="README.md", language="Markdown", chunk_type="text", content="overview", score=0.94),
                ChunkRecord(repository="repo", file_path="src/app.py", language="Python", chunk_type="function", content="run", score=0.51),
            ]
        return [
            ChunkRecord(repository="repo", file_path="src/app.py", language="Python", chunk_type="function", content="run", score=0.83),
            ChunkRecord(repository="repo", file_path="README.md", language="Markdown", chunk_type="text", content="overview", score=0.42),
        ]


def test_search_readme_prioritizes_docs_hits():
    store = FakeStore()
    tools = SearchTools(store)

    results = tools.search_readme("project overview", repository="repo")

    assert len(store.calls) >= 2
    assert results
    assert results[0].file_path == "README.md"


def test_search_code_keeps_repository_filter():
    store = FakeStore()
    tools = SearchTools(store)

    tools.search_code("authentication flow", repository="repo")

    assert store.calls
    assert all(call[2] == {"repository": "repo"} for call in store.calls)
