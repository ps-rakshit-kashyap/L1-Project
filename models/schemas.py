"""Pydantic schemas used across the app.

These models define the shared data shapes that move between the loader,
parser, vector store, services, and UI. Keeping them in one place prevents the
project from silently drifting into mismatched dictionaries and ad hoc return
values.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileRecord(BaseModel):
    """Metadata captured for every discovered file."""

    file_name: str
    file_path: str
    extension: str
    size: int
    language: str


class ChunkRecord(BaseModel):
    """A retrieval chunk produced from a source file or document."""

    repository: str
    file_path: str
    language: str
    class_name: str | None = None
    function_name: str | None = None
    chunk_type: str
    content: str
    start_line: int | None = None
    end_line: int | None = None
    score: float = Field(default=0.0)


class RetrievalResult(BaseModel):
    """Structured answer payload returned by the question service."""

    answer: str
    confidence: float
    referenced_files: list[str]
    referenced_functions: list[str]
    agent_log: list[str] = []
    diagram: str | None = None
