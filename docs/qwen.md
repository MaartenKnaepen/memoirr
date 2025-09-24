### **LLM-Optimized Memoirr Coding Standards**

## **Quick Reference for LLMs**

**When implementing code:**
1. Use functional approach → Pure functions, immutable data
2. Follow repository structure → `src/components/type/component.py`
3. Add comprehensive logging → Use `LoggedOperation` for timing
4. **For tests:** → **Always refer to `docs/testing_guide.md`**

## **Critical Rules**

| Aspect | Rule | Example |
|--------|------|---------|
| Functions | Pure functions only | No side effects, predictable outputs |
| Imports | Absolute imports | `from src.core.config import get_settings` |
| Structure | Component-based | `src/components/chunker/semantic_chunker.py` |
| Testing | Use testing guide | See `docs/testing_guide.md` for patterns |
| Logging | LoggedOperation | All major operations must be logged |

## **Code Style Templates**

### **Naming Convention Algorithm:**
```
IF variable/function/module → snake_case
IF class → PascalCase  
IF constant → UPPER_SNAKE_CASE
```

### **Function Template:**
```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """Brief description.
    
    Args:
        arg1: Description.
        arg2: Description.
        
    Returns:
        Description of return value.
    """
    logger = get_logger(__name__)
    
    with LoggedOperation("operation_name", logger) as op:
        # Implementation
        result = process_data(arg1, arg2)
        
        op.add_context(processed_items=len(result))
        return result
```

### **Import Template:**
```python
# Standard library
from typing import Dict, List, Optional

# Third party
from haystack import Pipeline, Document

# Local imports
from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation
```

---

## **Repository Structure Algorithm**

### **File Placement Rules:**
```
IF shared/reusable logic → src/core/
IF pipeline component → src/components/type/component.py
IF orchestration → src/pipelines/pipeline_name.py
IF utilities for component → src/components/type/utilities/component/
IF tests → test/ (mirror src/ structure)
```

### **Directory Structure Template:**
```
src/
├── core/                    # Shared utilities
│   ├── config.py           # Settings management
│   ├── logging_config.py   # Logging setup
│   └── model_utils.py      # Model utilities
├── components/
│   ├── chunker/
│   │   ├── semantic_chunker.py
│   │   └── utilities/semantic_chunker/
│   ├── embedder/
│   ├── generator/
│   ├── preprocessor/
│   ├── retriever/
│   └── writer/
└── pipelines/
    ├── srt_to_qdrant.py
    └── rag_pipeline.py
```

## **Testing Standards**

### **CRITICAL: Always refer to `docs/testing_guide.md` for testing patterns**

### **Test Structure Algorithm:**
```
IF testing function in module X → Mock imports as 'X.imported_function'
IF testing with LoggedOperation → Add mock_instance.start_time = time.time() - 0.1
IF testing pipeline results → Use explicit key validation
IF need test templates → Use testing_guide.md copy-paste patterns
```

### **Quick Test Template:**
```python
@patch('src.module.imported_function')
def test_function_name(mock_func):
    # Arrange
    mock_func.return_value = expected_value
    
    # Act  
    result = function_under_test(input_data)
    
    # Assert
    assert result.expected_field == expected_value
    mock_func.assert_called_once()
```

---

### **Pure Function Template:**
```python
def pure_function(input_data: InputType) -> OutputType:
    """Pure function - no side effects, predictable output."""
    # Only data transformation, no I/O
    return transform_data(input_data)
```

### **Factory Pattern Template:**
```python
def get_component() -> ComponentType:
    """Factory for consistent object creation."""
    settings = get_settings()
    return ComponentType(
        param1=settings.param1,
        param2=settings.param2
    )
```

### **Haystack Component Template:**
```python
from haystack import component, Document
from typing import List, Dict, Any

@component
class ComponentName:
    """Component description."""
    
    @component.output_types(output_name=List[Document])
    def run(self, input_data: Any) -> Dict[str, Any]:
        """Component run method."""
        from .utilities.component_name.orchestrate_operation import orchestrate_operation
        
        result = orchestrate_operation(input_data)
        
        return {"output_name": result}
```

---

## **LLM Implementation Checklist**

### **Before Implementing Any Feature:**
1. ✅ Check repository structure → Use correct `src/components/type/` path
2. ✅ Reference testing guide → See `docs/testing_guide.md` for test patterns  
3. ✅ Use templates above → Copy-paste from this guide
4. ✅ Add logging → Wrap operations in `LoggedOperation`
5. ✅ Follow naming → `snake_case` functions, `PascalCase` classes

### **Implementation Priority:**
1. Use **templates** from this guide
2. Follow **file placement rules** algorithm
3. Add **comprehensive tests** using testing guide
4. Ensure **pure functions** where possible
5. Add **proper logging** and error handling

---

## **Legacy Documentation** (For Reference)

1.  **Testing:**
    *   **Unit Tests are Non-Negotiable:** Every new piece of logic (a function or a class method) must be accompanied by a `pytest` unit test.
    *   **Focus on Interfaces:** Tests should verify the public "contract" of a function: given a specific input, does it produce the expected output?
    *   **Mock Extensively:** Unit tests **must not** make real network calls to Plex, Qdrant, or Groq. Use `pytest-mock` to simulate these external services. This makes tests fast, reliable, and able to run anywhere.

2.  **Configuration & Secrets:**
    *   **No Hardcoding:** API keys, URLs, model names, or file paths must never be written directly in the Python code.
    *   **Source of Truth:** All configuration must be loaded from environment variables via the Pydantic `Settings` object in `core/config.py`.
    *   **Secrets Management:** The `.env` file is for local development only and **must** be in `.gitignore`. In a future production deployment, secrets would be injected into the Docker container via a secure mechanism (e.g., Docker Swarm/Kubernetes secrets, or the hosting platform's secret manager).

3.  **Error Handling:**
    *   **Be Specific:** Catch specific exceptions (`plexapi.exceptions.NotFound`), not generic `Exception`.
    *   **Graceful Failure:** The application should handle expected errors gracefully. For example, if the `PlexFetcher` cannot find a media item for a given path, it should log a clear warning and exit cleanly, not crash the entire indexing process.
    *   **User-Facing Errors:** The Gradio UI should display user-friendly error messages ("Could not connect to the search engine. Please try again later.") instead of raw Python tracebacks.

---

### **4. File & Folder Structure (The "Where Things Live")**

The structure of the repository is just as important as the code itself. To maintain clarity and modularity, we enforce strict rules for how files and folders are organized:

1. **One Function per File (Ultra-Modular Design):**
   * Each **pure function** lives in its own `.py` file.  
   * The filename **must** match the function name (in `snake_case`).  
     Example:
     ```
     ├── clean_lines.py   -> def clean_lines()
     ├── deduplicate.py   -> def deduplicate()
     ├── drop_empty.py    -> def drop_empty()
     ```

2. **Folder-per-Component:**
   * Each **Haystack component** gets its own folder under `src/components/`.  
   * Inside the folder, the top-level file defines the component itself (e.g., `srt_preprocessor.py`), while **all helper logic** is pushed down into a `utilities/` subfolder.

   Example:
   ```
    src/components/chunker/
    ├── preprocessor/
    │ ├── srt_preprocessor.py # The Haystack component
    │ └── utilities/srt_preprocessor/
    │ ├── apply_cleaning.py
    │ ├── clean_lines.py
    │ ├── deduplicate.py
    │ ├── drop_empty.py
    │ ├── exceptions.py
    │ └── language_filter.py
    ```


3. **Utilities Subfolders:**
* Every component folder **must contain a `utilities/` subfolder**.
* Each utility is a single-purpose function in its own file (enforcing DRY).
* Utilities should remain **stateless** and **pure functions** whenever possible.

4. **Imports & Usage:**
* Components import their helper functions **only from their own `utilities/` subfolder** (not from sibling components).
* Shared, cross-component utilities should live in `src/core/utils/`.

5. **Naming Rules:**
* Component files: `snake_case_component.py` (e.g., `srt_preprocessor.py`).
* Utility function files: same name as function (e.g., `drop_empty.py` → `def drop_empty()`).
* Test files mirror their targets (`test_drop_empty.py`, `test_srt_preprocessor.py`).

---

### 5. Memoirr-Specific Conventions and Updates

1. Imports and module paths
   - Use absolute imports rooted at `src.*` everywhere (code, tests, and notebooks). Examples:
     - `from src.components.preprocessor.srt_preprocessor import SRTPreprocessor`
     - `from src.components.chunker.semantic_chunker import SemanticChunker`
     - `from src.core.config import get_settings`

2. Haystack component pattern (orchestrator + thin wrapper)
   - Place end-to-end logic in a single orchestration function inside the component's `utilities/<component_name>/` subfolder (e.g., `orchestrate_chunking.py`).
   - The Haystack component file (e.g., `semantic_chunker.py`) is a thin `@component` wrapper that:
     - Declares sockets using `@component.input_types(...)` and `@component.output_types(...)`.
     - Calls the orchestrator and returns a dict matching declared outputs.
   - Prefer simple, JSON-serializable socket types (e.g., `list`, `dict`, `str`). If Haystack version compatibility is a concern, declare input sockets as `Any` to avoid strict type mismatches.

2a. Component typing and future annotations (Option A - standard)
   - Do not use `from __future__ import annotations` in any module that defines a Haystack `@component`.
   - Use PEP 585 built-in generics (e.g., `list[str]`, `dict[str, Any]`) in component signatures.
   - Declare sockets explicitly where supported by your Haystack version:
     - If available, use `@component.input_types(...)` for inputs; otherwise rely on `run()` type hints.
     - Always use `@component.output_types(...)` for outputs.
   - Prefer simple, JSON-serializable socket types. If compatibility issues arise, relax input sockets to `Any`. 


3. Configuration and environment
   - Centralize configuration in `src/core/config.py` using `pydantic-settings`:
     - `.env` at repo root is the single source of truth.
     - Example variables:
       - `EMBEDDING_MODEL_NAME` (local model folder under `models/`)
       - `EMBEDDING_DEVICE` (optional; e.g., `cuda:0` or `cpu`)
     - Use `Field(..., alias="ENV_VAR_NAME")` in `Settings` to map environment names.

4. Self-hosted model layout (embeddings)
   - Store models locally under `models/<MODEL_NAME>/` with a sentence-transformers-compatible layout (`model.safetensors`, tokenizer/config files).
   - The embedding loader uses sentence-transformers in-process. It resolves the model folder by:
     - First checking `models/<EMBEDDING_MODEL_NAME>`.
     - Falling back to a case-insensitive search for the terminal folder name under `models/` (e.g., `qwen3-embedding-0.6b`).

5. Preprocessor → Chunker I/O contracts (JSONL)
   - Preprocessor output JSONL per caption (canonical intermediate):
     - `{ "text": str, "start_ms": int, "end_ms": int, "caption_index": int }`
   - Chunker input: list[str] of these JSONL lines.
   - Chunker output JSONL per chunk:
     - Required: `text`, `start_ms`, `end_ms`, `token_count`
     - Optional (enabled by default): `caption_indices` (list[int]), `chunker_params` (dict of parameters)

6. Time-aware mapping in the chunker
   - Concatenate caption texts with a single space; track `[start, end)` character spans per caption.
   - For each chunk span `(start_index, end_index)`, overlap with caption spans to compute:
     - `start_ms = min(start_ms of overlaps)`
     - `end_ms = max(end_ms of overlaps)`
     - `caption_indices = sorted unique indices of overlaps` (optional)

7. Testing structure
   - Tests mirror `src/` under `test/` using `src.*` imports.
   - Unit-test each pure utility (parse, build spans, map to time, emit JSONL) in isolation.
   - Component smoke tests can monkeypatch the orchestrator to avoid heavy dependencies and model loading.

8. Notebooks
   - Use `src.*` imports and include a small pipeline smoke test (SRT preprocessor → semantic chunker) with a tiny SRT sample.
   - Print settings and verify model folder existence to help users debug local setups.

9. Pipeline connectivity tests (required when a component is completed)
   - Add connectivity tests that validate components can be wired in a Haystack `Pipeline`.
   - Example tests to add under `test/pipelines/`:
     - `test_pipeline_connect_explicit_output_to_input`:
       - Build a pipeline with the producer and consumer components.
       - Call `pipe.connect("producer.output_name", "consumer.input_name")`.
     - `test_pipeline_connect_shorthand_receiver`:
       - `pipe.connect("producer.output_name", "consumer")` if the consumer has a single input.
   - Keep these tests failing initially if the sockets are not aligned yet. They act as a contract and a reminder to finalize socket signatures. Once the component is stable, align socket types and names so that these tests pass.

### **6. CRITICAL: Haystack Component Type System Guidelines**

**⚠️ MANDATORY for all AI coding assistants working on Haystack components ⚠️**

#### **The Problem**
Haystack's pipeline connection system requires **EXACT type consistency** between components. Mixing type annotation styles causes pipeline connection failures with cryptic errors like:
```
Cannot connect 'component1.output' with 'component2.input': their declared input and output types do not match.
'component1': output_name: List[str]
'component2': input_name: 'List[str]' (available)
```

Note the quotes around `'List[str]'` - this indicates inconsistent type evaluation.

#### **Root Cause**
The `from __future__ import annotations` import changes how Python evaluates type annotations:
- **Without it**: `List[str]` becomes the actual `typing.List[str]` type object
- **With it**: `List[str]` becomes the string `"List[str]"` for deferred evaluation

Haystack sees these as different types even though they're semantically identical.

#### **MANDATORY Rules for Haystack Components**

1. **NEVER use `from __future__ import annotations` in Haystack component files**
   ```python
   # ❌ WRONG - This will break pipeline connections
   from __future__ import annotations
   from haystack import component
   
   # ✅ CORRECT - No future annotations import
   from haystack import component
   from typing import List, Dict
   ```

2. **Always use consistent typing imports across ALL component files**
   ```python
   # ✅ CORRECT - Use this pattern in every component
   from typing import List, Dict, Any, Optional
   ```

3. **Use typing module types consistently in @component.output_types**
   ```python
   # ✅ CORRECT
   @component.output_types(texts=List[str], metas=List[Dict[str, Any]])
   
   # ❌ WRONG - Mixing styles
   @component.output_types(texts=list[str], metas=List[dict])
   ```

4. **Match parameter types exactly with output types**
   ```python
   # Component A outputs List[str]
   @component.output_types(output_data=List[str])
   
   # Component B must accept exactly List[str]
   def run(self, input_data: List[str]) -> Dict[str, object]:
   ```

5. **Before connecting components, verify type annotations match exactly**
   - Check imports: same typing style across files
   - Check output_types: use typing module consistently  
   - Check parameter types: match exactly with connecting outputs

#### **Quick Debugging Checklist**

When you get pipeline connection errors:

1. **Check for `from __future__ import annotations`** - remove it from component files
2. **Verify imports are identical** across connected components
3. **Compare exact type spellings**: `List[str]` vs `list[str]` vs `typing.List[str]`
4. **Look for quotes in error messages** - indicates string vs object type mismatch

#### **Example: Correct Component Pattern**
```python
# ✅ CORRECT Haystack component pattern
from typing import List, Dict, Any  # Consistent imports
from haystack import component

@component
class MyComponent:
    @component.output_types(data=List[str], meta=List[Dict[str, Any]])
    def run(self, input_data: List[str]) -> Dict[str, object]:
        return {"data": input_data, "meta": []}
```

**Remember**: Haystack's type system is strict. Even semantically identical types must be syntactically identical for pipeline connections to work.

---

### **7. Context7 MCP Server Best Practices for Code Development**

**⚠️ ESSENTIAL for AI assistants using MCP tools in coding workflows ⚠️**

#### **Philosophy: Smart Tool Usage for Efficient Development**

The Context7 MCP server provides powerful tools for workspace interaction, but using them efficiently requires strategic thinking. Follow these practices to minimize iterations and maximize code quality.

#### **🎯 Core Principles**

1. **Plan Before Acting**: Understand the problem completely before making changes
2. **Batch Operations**: Use simultaneous tool calls when possible
3. **Target Testing**: Only test relevant components, not the entire codebase
4. **Clean as You Go**: Remove temporary files created during development

#### **📋 Pre-Development Analysis Workflow**

**Before writing any code:**

1. **Explore the workspace structure**
   ```python
   # Use expand_folder to understand project layout
   expand_folder("src/")
   expand_folder("test/")
   ```

2. **Read existing patterns and conventions**
   ```python
   # Check for local guidelines
   open_files(["AGENTS.md", "AGENTS.local.md", "README.md"])
   ```

3. **Understand the codebase context**
   ```python
   # Look at similar existing components
   grep(content_pattern="class.*Component", path_glob="src/components/**/*.py")
   ```

#### **🔧 Efficient Code Investigation**

**Use targeted exploration:**

```python
# ✅ GOOD: Simultaneous investigation
open_files([
    "src/components/target_component.py",
    "test/components/test_target_component.py",
    "src/core/config.py"
])

# ❌ AVOID: Sequential single-file opens
open_files(["src/components/target_component.py"])
open_files(["test/components/test_target_component.py"])  # Wasteful iteration
```

**Expand code strategically:**

```python
# ✅ GOOD: Expand specific functions you need to understand
expand_code_chunks(
    file_path="src/components/large_file.py",
    patterns=["def target_function", "class TargetClass"]
)

# ❌ AVOID: Opening entire large files if you only need specific parts
open_files(["src/components/large_file.py"])  # May show collapsed view
```

#### **⚡ Development Best Practices**

**1. Simultaneous Operations**
```python
# ✅ EXCELLENT: Make related changes together
find_and_replace_code(file_path="file1.py", find="old", replace="new")
find_and_replace_code(file_path="file2.py", find="old", replace="new")
find_and_replace_code(file_path="file3.py", find="old", replace="new")
```

**2. Smart Testing Strategy**
```python
# ✅ GOOD: Test only what you changed
bash("python -m pytest test/components/test_modified_component.py -v")

# ❌ AVOID: Running entire test suite for small changes
bash("python -m pytest")  # Wasteful unless doing major refactoring
```

**3. Incremental Verification**
```python
# ✅ GOOD: Verify changes step by step
bash("python -c 'from src.components.new_component import NewComponent; print(\"Import successful\")'")
bash("python -m pytest test/components/test_new_component.py::TestSpecificMethod -v")
```

#### **🧹 Cleanup and Organization**

**Temporary File Management:**
```python
# ✅ ALWAYS: Prefix temporary files for easy cleanup
create_file("tmp_rovodev_test_script.py", content="...")

# At end of development:
delete_file("tmp_rovodev_test_script.py")
```

**Documentation Updates:**
```python
# ✅ GOOD: Update documentation as you develop
find_and_replace_code(
    file_path="README.md",
    find="## Components",
    replace="## Components\n\n### NewComponent\nDescription of new component..."
)
```

#### **🚫 Common Anti-Patterns to Avoid**

**1. Excessive File Opening**
```python
# ❌ WASTEFUL: Don't re-open files you've already seen
open_files(["config.py"])
# ... make changes ...
open_files(["config.py"])  # You already know the content!
```

**2. Redundant Exploration**
```python
# ❌ INEFFICIENT: Don't re-expand code you've already reviewed
expand_code_chunks(file_path="file.py", patterns=["function_a"])
# ... later in same session ...
expand_code_chunks(file_path="file.py", patterns=["function_a"])  # Wasteful
```

**3. Unfocused Testing**
```python
# ❌ AVOID: Testing unrelated components for localized changes
bash("python -m pytest test/")  # Too broad for a single component fix
```

#### **📊 Iteration Budget Guidelines**

**Target iteration counts by task complexity:**

- **Simple tasks** (bug fix, documentation): ~10 iterations
- **Medium tasks** (new feature, refactoring): ~20 iterations  
- **Complex tasks** (major changes, integrations): ~30 iterations

**Optimization strategies:**
- Use `grep` to find patterns before making changes
- Batch multiple `find_and_replace_code` calls
- Create helper scripts for repetitive tasks
- Only test what's relevant to your changes

#### **🎯 Example: Efficient Bug Fix Workflow**

```python
# 1. Investigate the issue (1-2 iterations)
grep(content_pattern="error_pattern", path_glob="**/*.py")
open_files(["src/problematic_file.py", "test/test_problematic_file.py"])

# 2. Make the fix (1 iteration)
find_and_replace_code(
    file_path="src/problematic_file.py",
    find="buggy_code",
    replace="fixed_code"
)

# 3. Test the fix (1 iteration)
bash("python -m pytest test/test_problematic_file.py::test_specific_case -v")

# 4. Update related files if needed (1 iteration)
find_and_replace_code(
    file_path="docs/api.md",
    find="old_behavior_description",
    replace="new_behavior_description"
)
```

**Total: 4-5 iterations for a complete bug fix cycle**

#### **🏆 Success Metrics**

**You're using MCP tools efficiently when:**
- You complete tasks within expected iteration budgets
- You make simultaneous related changes rather than sequential ones
- You test only what you've modified
- You clean up temporary files
- You understand the codebase before making changes

**Remember**: The goal is not to minimize tool calls, but to make each tool call as valuable and purposeful as possible.
