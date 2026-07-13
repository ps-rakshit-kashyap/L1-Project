# AI Software Architect

Local Agentic RAG application for repository understanding, documentation, security review, and architecture reasoning.

## Features

- **Upload & Index** — Upload a ZIP of any repository. The app extracts it safely, detects file types, and indexes code chunks into a local ChromaDB vector store.
- **Question Answering** — Ask questions about the codebase. A planner selects relevant retrieval tools, gathers context, reflects on sufficiency, and generates an answer via Ollama (with graceful fallback).
- **Security Review** — Heuristic scan for hardcoded secrets, debug mode, unsafe SQL patterns, and JWT without secret handling.
- **Documentation** — Auto-generate a Markdown overview of the indexed repository.
- **Architecture Diagrams** — Render Mermaid flow / dependency / auth / API diagrams based on the retrieved file structure.
- **Conversation Memory** — Session-aware follow-up questions with file and function citations.

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### Setup

```bash
# Clone and enter the project
cd ai-software-architect

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen3.5:2b-q4_K_M
ollama pull nomic-embed-text

# Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a `.zip` of any repository to begin.

## Project Structure

```
├── app.py                     # Streamlit entrypoint (UI + orchestration)
├── config.py                  # Centralized settings (paths, models, sizing)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules (venv, uploads, caches, …)
│
├── agents/
│   └── planner.py             # Keyword-based retrieval planner
│
├── loaders/
│   └── repository_loader.py   # File discovery & metadata extraction
│
├── models/
│   └── schemas.py             # Pydantic schemas (FileRecord, ChunkRecord, …)
│
├── parsers/
│   └── code_parser.py         # AST-aware Python chunker + text splitter
│
├── rag/
│   ├── embeddings.py          # Ollama embeddings with fallback
│   ├── vector_store.py        # ChromaDB client + in-memory fallback
│   ├── context_manager.py     # Deduplication and merging
│   └── reflection.py          # Context sufficiency checks
│
├── services/
│   ├── architecture_service.py  # Mermaid diagram generation
│   ├── documentation_service.py # Markdown report generation
│   ├── llm_service.py           # Ollama LLM with fallback
│   ├── question_service.py      # Full Q&A pipeline orchestration
│   ├── repository_service.py    # Scan → parse → index coordination
│   └── security_service.py      # Heuristic security scanner
│
├── tests/
│   ├── test_parser.py
│   ├── test_planner.py
│   ├── test_search_tools.py
│   └── test_zip_handler.py
│
├── tools/
│   └── search_tools.py        # LangChain-compatible search helpers
│
└── utils/
    ├── file_utils.py           # Hashing & path helpers
    ├── logger.py               # Logging configuration
    └── zip_handler.py          # Safe ZIP extraction
```

## Architecture Overview

The flow is:

1. **User uploads a ZIP** → `ZipHandler` extracts it safely to `uploaded_projects/`.
2. **RepositoryLoader** scans the extracted folder and builds a file inventory.
3. **CodeParser** chunks each file — Python files are parsed with `ast` to preserve class/method boundaries; other text files use sliding-window splits.
4. **VectorStore** embeds chunks via Ollama (`nomic-embed-text`) and stores them in ChromaDB.
5. **QuestionService** orchestrates the answer pipeline:
   - **Planner** selects retrieval tools based on keywords (auth, security, architecture, general).
   - **SearchTools** runs multi-variant queries with boosting.
   - **ReflectionEngine** checks if enough context was retrieved.
   - **LLMService** generates the final answer (or falls back deterministically).
6. **UI** displays the answer with confidence score, citations, conversation history, and optionally a Mermaid diagram.

## Tests

```bash
pytest
```

## Configuration

Edit `config.py` to customise:

| Setting | Default | Description |
|---|---|---|
| `default_model` | `qwen3.5:2b-q4_K_M` | Ollama model for Q&A |
| `embedding_model` | `nomic-embed-text` | Ollama model for embeddings |
| `chunk_size` | `1600` | Max characters per chunk |
| `chunk_overlap` | `250` | Overlap between adjacent chunks |
| `top_k` | `8` | Default retrieval count |
| `max_context_chunks` | `12` | Chunks fed into the LLM prompt |

## Future Improvements

- Tree-sitter-based parsing for deeper code awareness
- Smarter dependency graph extraction
- Richer file-by-file architecture map
- Exportable PDF/Word documentation
