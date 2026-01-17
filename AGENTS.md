# 📜 Memoirr Coding Guidelines for Claude Sonnet

> **Purpose:** Enable Claude Sonnet to generate code that matches this project's patterns exactly on the first try.

---

## 🏗️ 1. Architecture: The Three-Layer Pattern

**Never put business logic in Haystack components.** Follow this strict separation:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Component** | `src/components/<name>/<name>.py` | Haystack wiring only: `@component`, `run()`, config loading |
| **Orchestrator** | `src/components/<name>/utilities/<name>/orchestrate_*.py` | Coordinates utilities, handles errors, does logging |
| **Utilities** | `src/components/<name>/utilities/<name>/*.py` | Pure functions, no side effects, easy to test |

### Component Template (Copy-Paste Ready)
```python
"""Haystack component for [description].

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import List, Dict, Any, Optional

from haystack import component

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.components.<name>.utilities.<name>.orchestrate_<action> import orchestrate_<action>


@component
class MyComponent:
    """[One-line description].

    Args:
        param1: Description of param1.
    """

    def __init__(self, *, param1: Optional[int] = None) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        
        self.param1 = param1 if param1 is not None else settings.default_param1
        
        self._logger.info(
            "MyComponent initialized",
            param1=self.param1,
            component="my_component"
        )

    @component.output_types(result=List[str], stats=Dict[str, Any])
    def run(self, input_data: List[str]) -> Dict[str, object]:  # type: ignore[override]
        """Process input data.

        Args:
            input_data: List of strings to process.

        Returns:
            Dict with:
            - result: Processed output
            - stats: Processing statistics
        """
        with LoggedOperation("my_operation", self._logger, input_count=len(input_data)) as op:
            result, stats = orchestrate_<action>(input_data, param1=self.param1)
            
            op.add_context(output_count=len(result))
            self._metrics.counter("items_processed_total", len(input_data), component="my_component")
            
            return {"result": result, "stats": stats}
```

---

## ⚠️ 2. Haystack 2.x Critical Rules

These break pipeline connections if violated:

| Rule | Why |
|------|-----|
| **❌ NEVER** use `from __future__ import annotations` in component files | Haystack introspects types at runtime |
| **✅ ALWAYS** use `typing` module: `List`, `Dict`, `Optional`, `Any` | Not `list`, `dict` — Haystack compatibility |
| **✅ ALWAYS** define `@component.output_types(...)` decorator | Pipeline graph validation |
| **✅ ALWAYS** match return dict keys to `output_types` exactly | Runtime type checking |
| **✅ Use** `Dict[str, object]` as run() return type with `# type: ignore[override]` | Silences mypy, matches Haystack pattern |

---

## 📊 3. Data Types: Frozen Dataclasses

**Use `@dataclass(frozen=True)` for internal data transfer. Use `dict` only at I/O boundaries.**

```python
# In types.py - define all shared types here
from __future__ import annotations  # OK in utility files, NOT in components

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ChunkWithTime:
    """Chunk span mapped back to time.
    
    Attributes:
        text: The chunk text content.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.
        token_count: Number of tokens in chunk.
        caption_indices: Original caption indices (optional).
    """
    text: str
    start_ms: int
    end_ms: int
    token_count: int
    caption_indices: Optional[List[int]] = None
```

---

## 🔧 4. Configuration & Logging

### Configuration (via Pydantic Settings)
```python
from src.core.config import get_settings

settings = get_settings()
# Access: settings.embedding_model_name, settings.qdrant_url, etc.
# Add new settings to src/core/config.py with Field(default=..., alias="ENV_VAR_NAME")
```

### Structured Logging (via structlog)
```python
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger

logger = get_logger(__name__)
metrics = MetricsLogger(logger)

# Simple logging with structured context
logger.info("Processing started", component="chunker", file_count=5)

# Operation tracking with automatic timing
with LoggedOperation("semantic_chunking", logger, input_count=100) as op:
    # ... do work ...
    op.add_context(output_count=25, avg_tokens=150)
# Auto-logs: semantic_chunking_started, semantic_chunking_completed with duration_ms

# Metrics for Prometheus
metrics.counter("documents_processed", 1, component="retriever")
metrics.histogram("processing_duration_ms", 1500, operation="embedding")
```

---

## 🐍 5. Python Style Rules

### Type Hints (Required Everywhere)
```python
from typing import List, Dict, Any, Optional, Iterable, Iterator, Tuple

def process_chunks(
    chunks: Iterable[ChunkWithTime],
    *,  # Force keyword-only args after this
    include_params: bool,
    params: ChunkerParams,
) -> Iterator[str]:
    """Yields JSONL strings for each chunk."""
    ...
```

### Exception Handling (Chain Everything)
```python
try:
    data = json.loads(content)
except json.JSONDecodeError as e:
    logger.error("JSON parse failed", context_sample=content[:50])
    raise ValueError(f"Invalid JSON content: {str(e)}") from e  # Always chain!
```

### Path Handling (pathlib Only)
```python
from pathlib import Path

path = Path("models") / model_name / "config.json"  # ✅
# NOT: os.path.join("models", model_name, "config.json")  # ❌
```

### Imports (Absolute, Grouped)
```python
# 1. Standard library
import json
from pathlib import Path
from typing import List, Dict, Any

# 2. Third-party
from haystack import component
from haystack.dataclasses import Document

# 3. Local application (absolute imports only)
from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.components.chunker.utilities.semantic_chunker.types import ChunkWithTime
```

### Docstrings (Google Style)
```python
def orchestrate_retrieval(
    query: str,
    document_store: QdrantDocumentStore,
    top_k: int,
    score_threshold: float,
    filters: Dict[str, Any],
) -> List[Document]:
    """Retrieve similar documents from Qdrant.

    Args:
        query: The search query text.
        document_store: Configured Qdrant document store.
        top_k: Maximum documents to retrieve.
        score_threshold: Minimum similarity score (0.0-1.0).
        filters: Metadata filters for search.

    Returns:
        List of Document objects ranked by similarity.

    Raises:
        ConnectionError: If Qdrant is unreachable.
        ValueError: If query is empty.
    """
```

---

## 🧪 6. Testing Patterns

### The Import Path Rule
**Patch where imported, not where defined:**
```python
# Testing orchestrate_chunking.py which imports run_semantic_chunker
@patch('src.components.chunker.utilities.semantic_chunker.orchestrate_chunking.run_semantic_chunker')  # ✅
# NOT: @patch('src.components.chunker.utilities.semantic_chunker.run_semantic_chunker.run_semantic_chunker')  # ❌
```

### LoggedOperation Mock (Copy-Paste)
```python
import time
from unittest.mock import Mock, patch, MagicMock

@patch('src.components.chunker.utilities.semantic_chunker.orchestrate_chunking.LoggedOperation')
def test_with_logged_operation(mock_logged_op):
    mock_op_instance = Mock()
    mock_op_instance.start_time = time.time()  # CRITICAL: prevents math errors
    mock_logged_op.return_value.__enter__.return_value = mock_op_instance
    
    # ... test code ...
```

### Haystack Pipeline Mocks
```python
# Mock component results, not Haystack internals
mock_pipeline = Mock()
mock_pipeline.run.return_value = {
    "retriever": {"documents": [Document(content="test", meta={})]},
    "generator": {"replies": ["Generated answer"], "meta": [{}]}
}
```

### Groq API Mock Pattern
```python
def create_mock_groq_response(content="Test response"):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content), finish_reason="stop")]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=25, total_tokens=125)
    return response

with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = create_mock_groq_response()
    mock_groq.return_value = mock_client
    # ... test code ...
```

### File System (Use tmp_path)
```python
def test_file_processing(tmp_path):
    test_file = tmp_path / "test.srt"
    test_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    # ... test with test_file ...
```

---

## 📁 7. File Organization Reference

```
src/components/<name>/
├── __init__.py           # Exports component class
├── <name>.py             # Haystack component (Layer 1)
├── README.md             # Component documentation
└── utilities/
    └── <name>/
        ├── __init__.py
        ├── orchestrate_<action>.py  # Orchestrator (Layer 2)
        ├── types.py                  # Frozen dataclasses
        └── <utility>.py              # Pure functions (Layer 3)

test/components/<name>/
├── test_<name>_component.py          # Component integration tests
└── utilities/
    └── <name>/
        └── test_<utility>.py         # Unit tests for utilities
```

---

## ✅ 8. Pre-Commit Checklist

Before generating code, verify:

1. **Architecture:** Logic in utilities, not components?
2. **Types:** Using `List`, `Dict` from `typing` (not builtins)?
3. **Component:** Has `@component.output_types`? Return keys match?
4. **No `from __future__ import annotations`** in component files?
5. **Dataclasses:** Using `@dataclass(frozen=True)` for internal data?
6. **Exceptions:** Chained with `from e`?
7. **Paths:** Using `pathlib.Path`?
8. **Docstrings:** Google-style with Args/Returns/Raises?
9. **Logging:** Using `get_logger(__name__)`, not `print()`?
10. **Tests:** Mocking at import location? LoggedOperation properly mocked?