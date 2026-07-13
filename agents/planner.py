"""Planning agent for multi-step retrieval.

This module decides which retrieval actions should run before the LLM sees any
context. It is intentionally rule-based so the app stays predictable and easy
to explain during demos or walkthroughs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanStep:
    """One retrieval action that should be executed for the user's question."""

    tool_name: str
    query: str
    rationale: str


@dataclass
class PlanResult:
    """Container for the ordered retrieval steps chosen by the planner."""

    steps: list[PlanStep] = field(default_factory=list)


class PlannerAgent:
    """Heuristic planner that chooses retrieval tools before answering."""

    def plan(self, question: str) -> PlanResult:
        # Normalize the question once so we can do fast keyword matching.
        lower = question.lower()
        steps: list[PlanStep] = []

        # Auth questions need overview docs plus route/config clues.
        if any(word in lower for word in ["auth", "login", "jwt", "token", "refresh"]):
            steps.extend([
                PlanStep("search_readme", "authentication login jwt token session refresh overview setup", "Find overview and setup docs"),
                PlanStep("search_routes", "authentication routes middleware jwt login api", "Find auth endpoints"),
                PlanStep("search_code", "jwt token auth middleware session refresh implementation", "Find implementation details"),
                PlanStep("search_configuration", "environment variables auth jwt secret settings", "Find config and secrets"),
            ])
        # Security questions should first inspect obvious risky patterns.
        elif any(word in lower for word in ["security", "vulnerability", "unsafe"]):
            steps.extend([
                PlanStep("find_security_issues", "security secret token password auth validate sanitize injection", "Scan for risky patterns"),
                PlanStep("search_code", "validation sanitize sql injection hardcoded secret password token", "Confirm issue locations"),
            ])
        # Architecture questions use structural retrieval plus file layout.
        elif any(word in lower for word in ["architecture", "diagram", "flow"]):
            steps.extend([
                PlanStep("generate_architecture", "architecture structure flow dependencies entrypoint", "Build architecture view"),
                PlanStep("list_project_structure", "project structure files modules entrypoint", "Inspect project layout"),
                PlanStep("retrieve_documentation", "architecture documentation readme overview", "Find supporting documentation"),
                PlanStep("find_dependencies", "dependencies package requirements install", "Inspect project dependencies"),
            ])
        # Default path falls back to direct code and documentation search.
        else:
            steps.extend([
                PlanStep("search_code", question, "Retrieve relevant implementation"),
                PlanStep("search_readme", question, "Retrieve documentation"),
                PlanStep("search_configuration", question, "Check configuration"),
            ])
        return PlanResult(steps=steps[:6])
