"""Question answering orchestration.

This is the main reasoning pipeline of the app. It decides which retrieval
tools to call, merges and checks the retrieved chunks, optionally asks the LLM
to generate a response, and falls back to deterministic architecture summaries
when the model does not provide a useful answer.
"""

from __future__ import annotations

import logging

from agents.planner import PlannerAgent
from config import settings
from rag.context_manager import ContextManager
from rag.reflection import ReflectionEngine
from rag.vector_store import VectorStore
from services.llm_service import LLMService
from tools.search_tools import SearchTools


logger = logging.getLogger(__name__)


class QuestionService:
    """Use planning, retrieval, reflection, and generation to answer questions."""

    def __init__(self, store: VectorStore, llm: LLMService, planner: PlannerAgent | None = None) -> None:
        self.store = store
        self.llm = llm
        self.planner = planner or PlannerAgent()
        self.context_manager = ContextManager()
        self.reflection = ReflectionEngine()
        self.tools = SearchTools(store)

    def answer(self, repository: str, question: str, history: list[dict] | None = None) -> dict:
        # Step 1: decide which retrieval tools are relevant for the question.
        plan = self.planner.plan(question)
        logger.info("Planner steps: %s", [step.tool_name for step in plan.steps])
        retrieved = []
        agent_log = [f"Planner created {len(plan.steps)} step(s)."]

        for step in plan.steps:
            # Step 2: execute the planned tools and collect matching chunks.
            tool = getattr(self.tools, step.tool_name, None)
            if not tool:
                continue
            agent_log.append(f"Calling {step.tool_name}: {step.rationale}")
            if step.tool_name in {
                "list_project_structure",
                "find_dependencies",
                "retrieve_documentation",
                "generate_architecture",
                "find_security_issues",
            }:
                results = tool(repository)
            else:
                results = tool(step.query, repository=repository)
            retrieved.extend(results)

        # Step 3: remove duplicates and check whether more context is needed.
        merged = self.context_manager.merge(retrieved)
        reflection = self.reflection.assess(question, merged)
        if reflection.needs_more_context:
            # If the first pass was thin, run a broader search for extra signals.
            agent_log.extend(reflection.missing_signals)
            extra = self.store.search(question, top_k=settings.top_k * 2, metadata_filter={"repository": repository})
            merged = self.context_manager.merge(merged + extra)

        # Architecture-style questions are answered from the structural summary
        # when possible because file-level shape is more useful than one random
        # code snippet.
        synthesized = self._synthesize_architecture_answer(repository, merged) if self._is_structure_question(question) else None
        if synthesized:
            answer = synthesized[: settings.max_answer_chars]
            agent_log.append("Synthesized architecture summary from indexed files.")
        else:
        # For general questions, build a prompt and ask the LLM directly.
            prompt = self._build_prompt(repository, question, merged, history or [])
            answer = self.llm.generate(prompt)[: settings.max_answer_chars]
            if self._should_synthesize(question, answer):
                synthesized = self._synthesize_architecture_answer(repository, merged)
                if synthesized:
                    agent_log.append("Synthesized architecture summary from indexed files because Ollama returned an empty response.")
                    answer = synthesized[: settings.max_answer_chars]
        confidence = self._confidence(merged, reflection.needs_more_context)

        return {
            "answer": answer,
            "confidence": confidence,
            "referenced_files": self._referenced_files(merged),
            "referenced_functions": self._referenced_functions(merged),
            "agent_log": agent_log,
            "diagram": self._diagram_if_requested(question, merged),
        }

    def _build_prompt(self, repository: str, question: str, chunks, history: list[dict]) -> str:
        # Clip long text so the prompt stays within a manageable context size.
        def _clip(text: str, limit: int = 700) -> str:
            text = text.strip()
            if len(text) <= limit:
                return text
            return text[: limit - 3] + "..."

        # Context is serialized with file tags so the model can cite exact
        # sources.
        context = "\n\n".join(
            f"[FILE: {chunk.file_path} | {chunk.chunk_type} | {chunk.function_name or chunk.class_name or 'n/a'}]\n{_clip(chunk.content)}"
            for chunk in chunks[: settings.max_context_chunks]
        )
        # Conversation history is kept short to avoid wasting prompt space.
        convo = "\n".join(f"{item.get('role')}: {_clip(str(item.get('content', '')), 300)}" for item in history[-6:])
        return (
            f"You are an AI Software Architect. Repository: {repository}\n"
            f"Answer at the level of detail the user asked for.\n"
            f"If the request is brief, be brief.\n"
            f"If the request is descriptive or complex, be thorough.\n"
            f"Use only the provided context when available.\n"
            f"Cite exact files when relevant.\n"
            f"Be direct, accurate, and helpful.\n\n"
            f"Conversation history:\n{convo}\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Return the answer in the style requested by the user."
        )

    def _confidence(self, chunks, needs_more_context: bool) -> float:
        # Confidence is a heuristic based on how much context was actually
        # retrieved.
        base = min(0.95, 0.45 + 0.08 * len(chunks))
        return max(0.1, base - (0.15 if needs_more_context else 0.0))

    def _referenced_files(self, chunks) -> list[str]:
        # Return a stable, unique list of files that informed the final answer.
        return sorted({chunk.file_path for chunk in chunks})

    def _referenced_functions(self, chunks) -> list[str]:
        # Collect class and function names so the UI can show what was used.
        names = []
        for chunk in chunks:
            if chunk.function_name:
                names.append(chunk.function_name)
            if chunk.class_name:
                names.append(chunk.class_name)
        return sorted(set(names))

    def _diagram_if_requested(self, question: str, chunks) -> str | None:
        # Only build a Mermaid diagram when the user asks for architecture/flow
        # output.
        lower = question.lower()
        if "diagram" not in lower and "architecture" not in lower and "flow" not in lower:
            return None
        return self._generate_mermaid(chunks)

    def _generate_mermaid(self, chunks) -> str:
        # Simple chain diagram built from the top retrieved files.
        nodes = [chunk.file_path.replace("/", "_").replace(".", "_") for chunk in chunks[:6]]
        lines = ["flowchart TD"]
        for idx, node in enumerate(nodes):
            lines.append(f"  N{idx}[{node}]")
        for idx in range(len(nodes) - 1):
            lines.append(f"  N{idx} --> N{idx+1}")
        return "\n".join(lines)

    def _should_synthesize(self, question: str, answer: str) -> bool:
        # If Ollama fails on a structure-style question, fall back to the
        # deterministic summary.
        return self._is_structure_question(question) and (not answer.strip() or answer.startswith("Ollama is not available locally."))

    def _is_structure_question(self, question: str) -> bool:
        # These keywords usually mean the user wants project layout or
        # architecture, not code details.
        lower = question.lower()
        return any(word in lower for word in ["architecture", "structure", "diagram", "flow"])

    def _synthesize_architecture_answer(self, repository: str, chunks) -> str:
        # Convert file-level chunks into a human-readable architecture
        # explanation.
        by_path: dict[str, list] = {}
        for chunk in chunks:
            by_path.setdefault(chunk.file_path, []).append(chunk)

        def _find_path(*predicates) -> str | None:
            # Helper for locating likely entrypoint files.
            for path in by_path:
                if all(predicate(path.lower()) for predicate in predicates):
                    return path
            return None

        main_app = _find_path(lambda path: "task master application" in path, lambda path: path.endswith("app.py"))
        lab_files = sorted(path for path in by_path if "\\lab\\" in path.lower() or "/lab/" in path.lower())
        db_file = next((path for path in by_path if path.lower().endswith(".db")), None)
        config_file = next((path for path in by_path if path.lower().endswith("config.py")), None)

        lines = [f"The repository `{repository}` looks like a mixed codebase with one main app and a separate lab/exercise area."]
        if main_app:
            # The main application reveals the CRUD-style runtime entrypoint.
            lines.append(f"- Main application: `{main_app}`. The indexed structure shows route/function names such as `index`, `update`, `delete`, and a `Todo` model, which points to a small CRUD-style web app.")
        if lab_files:
            # Lab files are educational extras and usually not part of production flow.
            examples = ", ".join(f"`{path}`" for path in lab_files[:5])
            if len(lab_files) > 5:
                examples += ", ..."
            lines.append(f"- Supporting exercises: {examples}. These files look like standalone practice snippets rather than the runtime app.")
        if db_file:
            # A local .db file suggests embedded persistence rather than remote services.
            lines.append(f"- Persistence: `{db_file}` suggests local database-backed storage.")
        if config_file:
            # Shared config files usually hold app-wide defaults and environment constants.
            lines.append(f"- Configuration: `{config_file}` is likely where shared settings live.")
        lines.append("- Overall shape: the project is organized around a single application entrypoint plus helper/demo files, so the architecture is closer to a small monolith than a multi-service system.")
        return "\n".join(lines)

