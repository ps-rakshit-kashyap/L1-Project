"""Security review helpers.

The security service is a heuristic scanner that looks for common code smells
such as hardcoded secrets, debug mode, and suspicious SQL usage. It is meant
to highlight obvious risks quickly, not to replace a dedicated security audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from models.schemas import ChunkRecord


@dataclass
class SecurityIssue:
    file_path: str
    severity: str
    issue: str


class SecurityService:
    """Detect common repository security risks."""

    def review(self, chunks: list[ChunkRecord]) -> list[SecurityIssue]:
        # Scan each chunk for simple but high-value security smells.
        issues: list[SecurityIssue] = []
        for chunk in chunks:
            content = chunk.content.lower()
            # Hardcoded credentials are the highest-priority finding.
            if re.search(r"(api_key|secret|password)\s*=\s*['\"][^'\"]+['\"]", content):
                issues.append(SecurityIssue(chunk.file_path, "high", "Potential hardcoded secret"))
            # Debug mode in production can leak internals and stack traces.
            if "debug = true" in content or "app.debug = true" in content:
                issues.append(SecurityIssue(chunk.file_path, "medium", "Debug mode enabled"))
            # String-interpolated SQL often indicates injection risk.
            if "sql" in content and ("f-string" in content or "%" in content):
                issues.append(SecurityIssue(chunk.file_path, "medium", "Potential unsafe SQL construction"))
            # JWT code without visible secret handling deserves a closer look.
            if "jwt" in content and "secret" not in content:
                issues.append(SecurityIssue(chunk.file_path, "medium", "JWT usage without obvious secret handling"))
        return issues
