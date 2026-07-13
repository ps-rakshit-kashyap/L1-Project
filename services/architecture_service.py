"""Mermaid architecture generation.

This service turns retrieved chunks into small Mermaid snippets that help a
reader visualize the shape of the repository. It does not attempt a perfect
dependency graph; instead it provides a useful, explainable summary.
"""

from __future__ import annotations

from models.schemas import ChunkRecord


class ArchitectureService:
    """Build simple but useful Mermaid diagrams."""

    def generate(self, diagram_type: str, chunks: list[ChunkRecord]) -> str:
        # Use the first few retrieved files as the basis for a lightweight diagram.
        files = [chunk.file_path for chunk in chunks[:8]]
        if diagram_type == "dependency":
            return self._dependency(files)
        if diagram_type == "auth":
            return self._auth(files)
        if diagram_type == "api":
            return self._api(files)
        return self._flow(files)

    def _flow(self, files: list[str]) -> str:
        # Sequential flow is the simplest helpful diagram for mixed repositories.
        lines = ["flowchart TD"]
        for idx, file in enumerate(files):
            lines.append(f'  A{idx}["{file}"]')
        for idx in range(len(files) - 1):
            lines.append(f"  A{idx} --> A{idx+1}")
        return "\n".join(lines)

    def _dependency(self, files: list[str]) -> str:
        # Dependency diagrams reuse the same nodes but render left-to-right.
        lines = ["graph LR"]
        for idx, file in enumerate(files):
            lines.append(f'  D{idx}["{file}"]')
        for idx in range(len(files) - 1):
            lines.append(f"  D{idx} --> D{idx+1}")
        return "\n".join(lines)

    def _auth(self, files: list[str]) -> str:
        # Fixed auth diagram is used when the user asks about authentication flow.
        return "flowchart TD\n  User --> Login\n  Login --> AuthMiddleware\n  AuthMiddleware --> ProtectedRoutes\n  ProtectedRoutes --> Response"

    def _api(self, files: list[str]) -> str:
        # This sequence diagram explains the app interaction loop at a high level.
        return "sequenceDiagram\n  actor User\n  participant UI\n  participant API\n  participant Store\n  User->>UI: Submit question\n  UI->>API: Retrieve context\n  API->>Store: Semantic search\n  Store-->>API: Chunks\n  API-->>UI: Answer with citations"
