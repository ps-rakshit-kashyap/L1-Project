# AI Software Architect

Local Agentic RAG application for repository understanding, documentation, security review, and architecture reasoning.

## Features

- ZIP repository upload and safe extraction
- Recursive repository indexing with language detection
- Semantic retrieval with ChromaDB
- Ollama-powered question answering
- Planning, retrieval, reflection, and answer synthesis
- Mermaid architecture diagrams
- Security review and repository documentation generation
- Conversation memory and file citations

## Local Stack

- Python
- Streamlit
- LangChain
- Ollama
- ChromaDB
- PyMuPDF
- python-docx
- pandas
- GitPython
- Pydantic

## Installation

1. Install Python 3.10+.
2. Install and start Ollama locally.
3. Pull the default models:

```bash
ollama pull qwen3.5:2b-q4_K_M
ollama pull nomic-embed-text
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Architecture

- `app.py` orchestrates the Streamlit UI.
- `services/` contains orchestration for questions, indexing, security, docs, and diagrams.
- `rag/` contains embeddings, retrieval, context merging, and reflection.
- `agents/` contains the planner.
- `tools/` exposes retrieval helpers used by the planner.
- `loaders/` scans repositories.
- `parsers/` chunks source files semantically.

## Screenshots

Add screenshots of:

- Upload flow
- Repository explorer
- Answer view with citations
- Mermaid diagram output
- Security review

## Tests

```bash
pytest
```

## Future Improvements

- Tree-sitter-based parsing for deeper code awareness
- Smarter dependency graph extraction
- Richer file-by-file architecture map
- Exportable PDF/Word documentation

