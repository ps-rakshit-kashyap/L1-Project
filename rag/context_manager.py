"""Context merging and deduplication.

The retrieval pipeline can surface the same file chunk more than once, so this
module keeps only the first copy of each logical chunk while preserving the
order in which the evidence was retrieved.
"""

from __future__ import annotations

from collections import OrderedDict

from models.schemas import ChunkRecord


class ContextManager:
    """Merge retrieved chunks into a stable context package."""

    def merge(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        # Deduplicate repeated retrieval hits while preserving their original order.
        ordered: OrderedDict[str, ChunkRecord] = OrderedDict()
        for chunk in chunks:
            key = f"{chunk.file_path}:{chunk.class_name}:{chunk.function_name}:{chunk.chunk_type}"
            if key not in ordered:
                ordered[key] = chunk
        return list(ordered.values())
