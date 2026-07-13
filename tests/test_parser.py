"""Parser tests for the semantic chunker.

The parser should identify top-level Python functions as standalone retrieval
chunks so the vector store can cite them later.
"""

from pathlib import Path

from parsers.code_parser import CodeParser


def test_python_parse_function():
    parser = CodeParser()
    result = parser.parse_file("repo", Path("app.py"), "def hello():\n    return 1\n")
    assert result.chunks
    assert result.chunks[0].function_name == "hello"
