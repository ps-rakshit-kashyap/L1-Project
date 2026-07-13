"""Streamlit entrypoint for AI Software Architect.

This file exists to connect the whole project together in one user-facing
place. It creates the Streamlit page, wires in the repository upload flow,
boots the indexing services, and exposes the question-answering tools that sit
on top of the retrieval pipeline.

If someone wants to understand the application from the outside in, this is
the best starting point because every major user action eventually passes
through here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

from agents.planner import PlannerAgent
from config import settings
from loaders.repository_loader import RepositoryLoader
from parsers.code_parser import CodeParser
from rag.vector_store import VectorStore
from services.architecture_service import ArchitectureService
from services.documentation_service import DocumentationService
from services.llm_service import LLMService
from services.question_service import QuestionService
from services.repository_service import RepositoryService
from services.security_service import SecurityService
from utils.logger import configure_logging
from utils.zip_handler import ZipHandler, ZipHandlerError


configure_logging()
st.set_page_config(page_title=settings.app_name, layout="wide", initial_sidebar_state="expanded")


def detect_runtime_warning() -> str | None:
    """Warn when Streamlit is launched from a different interpreter than the project venv."""
    expected_python = Path.cwd() / ".venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable)

    if expected_python.exists() and current_python.resolve() != expected_python.resolve():
        # This message helps explain why local dependencies such as Ollama
        # integrations may appear missing even when the code itself is correct.
        return (
            "This app is running from a different Python environment than the project `.venv`. "
            "If Ollama or other dependencies seem unavailable, launch Streamlit with:\n"
            f"`{expected_python} -m streamlit run app.py`"
        )
    return None


runtime_warning = detect_runtime_warning()
if runtime_warning:
    st.warning(runtime_warning)


@st.cache_resource
def build_services():
    """Create long-lived app services once per Streamlit session."""
    # VectorStore is shared because indexing and question answering both need
    # the same repository-backed storage.
    store = VectorStore()
    return {
        "zip": ZipHandler(),
        "repo": RepositoryService(RepositoryLoader(), CodeParser(), store),
        "questions": QuestionService(store, LLMService(), PlannerAgent()),
        "security": SecurityService(),
        "docs": DocumentationService(),
        "arch": ArchitectureService(),
        "store": store,
    }


services = build_services()

st.title("AI Software Architect")
st.caption("Local Agentic RAG for repository understanding, documentation, security review, and architecture reasoning.")

if "history" not in st.session_state:
    # Conversation history is kept so the assistant can answer follow-up
    # questions with some awareness of what was already discussed.
    st.session_state.history = []
if "repository_root" not in st.session_state:
    # The extracted folder path must survive reruns because Streamlit reruns
    # the script whenever a widget changes.
    st.session_state.repository_root = None
if "repository_name" not in st.session_state:
    # This is the label shown in the interface after indexing succeeds.
    st.session_state.repository_name = ""
if "indexed" not in st.session_state:
    # Prevent question tools from running before the repository has been loaded
    # and indexed into the vector store.
    st.session_state.indexed = False

with st.sidebar:
    st.header("Repository Upload")
    uploaded = st.file_uploader("Upload repository ZIP", type=["zip"])
    if uploaded and st.button("Extract and Index", width="stretch"):
        # Streamlit gives us an in-memory upload object, but the ZIP handler
        # expects a real file path on disk.
        temp_zip = settings.upload_path / uploaded.name
        temp_zip.write_bytes(uploaded.getbuffer())
        try:
            # Extraction and indexing happen as two separate steps: first we
            # unpack the archive, then we scan and index the extracted files.
            repo_root = services["zip"].extract_zip(temp_zip)
            summary = services["repo"].index_repository(repo_root)
            st.session_state.repository_root = str(repo_root)
            st.session_state.repository_name = summary.repository
            st.session_state.indexed = True
            st.success(f"Indexed {summary.repository} with {len(summary.files)} files.")
        except ZipHandlerError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    st.divider()
    st.header("Configuration")
    # Showing the model here makes the app's runtime expectation visible to the
    # user instead of hiding it inside the service layer.
    st.text_input("Ollama model", value=settings.default_model, key="ollama_model")
    if st.session_state.ollama_model != settings.default_model:
        st.info("Model change is session-scoped in this build. Restart to persist a different default.")

repository_root = Path(st.session_state.repository_root) if st.session_state.repository_root else None

if st.session_state.indexed and repository_root:
    # Re-scan the extracted folder so the UI can show the current file inventory
    # and repository metrics.
    summary = services["repo"].loader.scan(repository_root)
    col1, col2, col3 = st.columns(3)
    col1.metric("Files", len(summary.files))
    col2.metric("Repository", summary.repository)
    col3.metric("Top Folder", repository_root.name)

    with st.expander("Repository Explorer", expanded=False):
        # This table lets the user verify exactly which files were discovered.
        st.dataframe([file.model_dump() for file in summary.files], width="stretch")

    # This text field is the primary question entrypoint for the repository
    # assistant. Everything below is driven by the user's input here.
    question = st.text_input("Ask the architecture agent", placeholder="Explain authentication flow, security risks, or module dependencies.")
    ask_col, doc_col, sec_col, arch_col = st.columns(4)
    ask_clicked = ask_col.button("Ask", width="stretch")
    doc_clicked = doc_col.button("Generate Docs", width="stretch")
    sec_clicked = sec_col.button("Security Review", width="stretch")
    arch_clicked = arch_col.button("Architecture Diagram", width="stretch")

    if ask_clicked and question:
        # The question service handles the full retrieval and answer-generation
        # workflow, including planning, reflection, and fallback logic.
        result = services["questions"].answer(summary.repository, question, st.session_state.history)
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": result["answer"]})
        st.subheader("Answer")
        st.write(result["answer"])
        st.metric("Confidence", f"{result['confidence']:.2f}")
        if result.get("diagram"):
            st.code(result["diagram"], language="markdown")
        st.subheader("Referenced Files")
        st.write(result["referenced_files"])
        st.subheader("Referenced Functions")
        st.write(result["referenced_functions"])
        with st.expander("Agent Logs", expanded=False):
            # The agent log is a human-readable trace of the internal steps.
            st.write(result["agent_log"])

    if doc_clicked:
        # Documentation mode produces a compact Markdown summary of the indexed
        # repository instead of a conversational answer.
        st.subheader("Documentation")
        st.markdown(services["docs"].generate(summary.repository, services["store"].search(summary.repository, top_k=20, metadata_filter={"repository": summary.repository})))

    if sec_clicked:
        # Security review reuses the vector store and then applies heuristic
        # checks to the retrieved chunks.
        chunks = services["store"].search("security review", top_k=30, metadata_filter={"repository": summary.repository})
        issues = services["security"].review(chunks)
        st.subheader("Security Findings")
        st.dataframe([issue.__dict__ for issue in issues], width="stretch")

    if arch_clicked:
        # Architecture mode renders a lightweight Mermaid diagram so the user
        # can see how the application pieces connect.
        chunks = services["store"].search("architecture flow dependencies", top_k=20, metadata_filter={"repository": summary.repository})
        st.subheader("Mermaid Diagram")
        st.code(services["arch"].generate("flow", chunks), language="markdown")

    st.subheader("Conversation History")
    # We surface the recent turn history so the user can remember what was
    # already asked without scrolling through the whole session.
    for message in st.session_state.history[-10:]:
        st.write(f"**{message['role']}**: {message['content']}")
else:
    # Before a repository is indexed there is nothing to answer yet, so the UI
    # stays focused on the upload prompt.
    st.info("Upload a repository ZIP to begin indexing.")
