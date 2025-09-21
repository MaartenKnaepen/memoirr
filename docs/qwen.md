### **Memoirr: Python Coding Standards & Architectural Guide**

This document outlines the style guides, architectural principles, and best practices to be followed throughout the development of the Memoirr project. The primary goals of these standards are to ensure the codebase is **Readable, Maintainable, Modular, and Extensible.** Adherence to these principles is mandatory for all new code.

### **1. Code Style & Formatting (The "Look and Feel")**

The foundation of readable code is consistent style.

1.  **Linter & Formatter:**
    *   **Tooling:** We will use **Ruff** for both linting and formatting. It is incredibly fast and integrates the functionality of dozens of tools (like Black, isort, Flake8) into one.
    *   **Configuration:** A `pyproject.toml` or `ruff.toml` file will be configured at the root of the project. It will enforce a line length of **88 characters**, automatically sort imports, and enforce a strict set of linting rules.
    *   **Workflow:** Code should be automatically formatted on save in the developer's IDE. A pre-commit hook can be configured to run Ruff before any code is committed to the repository.

2.  **Naming Conventions:**
    *   **`snake_case`:** For all variables, functions, methods, and modules (e.g., `media_processor.py`, `def process_media()`).
    *   **`PascalCase`:** For all classes (e.g., `PlexFetcher`, `Settings`).
    *   **`UPPER_SNAKE_CASE`:** For constants (e.g., `VECTOR_SIZE = 384`).
    *   **Clarity over Brevity:** Variable names should be descriptive. `def process_media_file(file_path: str)` is better than `def proc(fpath)`.

3.  **Type Hinting:**
    *   **Mandatory:** All function and method signatures **must** include type hints for all arguments and the return value.
    *   **Standard:** Use standard types from the `typing` module (`List`, `Dict`, `Optional`, `Tuple`).
    *   **Benefit:** Type hints are critical for maintainability, IDE support, and static analysis. They serve as a form of documentation and make the code's intent clear.

4.  **Docstrings:**
    *   **Mandatory:** Every public module, class, and function must have a docstring.
    *   **Format:** We will use the **Google Python Style Guide** for docstrings. It is readable and well-supported by documentation generators.
    ```python
    def my_function(arg1: int, arg2: str) -> bool:
        """Summarizes what this function does.

        A more detailed explanation of the function's behavior,
        its side effects, and any complex logic.

        Args:
            arg1: Description of the first argument.
            arg2: Description of the second argument.

        Returns:
            A boolean indicating success or failure.
        """
        # ... function code ...
    ```

---

### **2. Architectural Principles (The "Philosophy")**

These are the high-level rules that guide how we structure the application.

1.  **Modularity & Single Responsibility Principle (SRP):**
    *   **Rule:** Every file (module) and every class should have one, and only one, reason to change.
    *   **Implementation:**
        *   **`core/`:** Contains project-wide, reusable, and abstract logic (e.g., `config.py`, `hardware.py`). These files know nothing about Plex or pictures.
        *   **`components/`:** Contains our custom, specialized Haystack components. Each file is one component (e.g., `plex_fetcher.py`, `chonky_splitter.py`). A component's job is to perform one specific step in a pipeline.
        *   **`pipelines/`:** Contains the orchestration logic. These files import components and wire them together to form a `Pipeline` object.
        *   **`ui.py`:** Contains only the code related to the Gradio user interface. It should know as little as possible about the internal workings of the pipelines it calls.

2.  **Don't Repeat Yourself (DRY):**
    *   **Rule:** If you find yourself writing the same block of code in two or more places, abstract it into a function or a class.
    *   **Implementation:**
        *   **Factories:** We use factory functions like `get_document_store()` and `get_groq_generator()`. This ensures that these critical objects are always instantiated in the exact same way everywhere in the application, using the central configuration.
        *   **Shared Utilities:** Common, pure functions (e.g., a function to clean text) should be placed in a `utils.py` file to be imported where needed.

3.  **Dependency Inversion & Abstraction (The Key to Reusability):**
    *   **Rule:** Our core logic should not depend on concrete implementations; it should depend on abstractions. This is the most important principle for making the embedding and storage reusable.
    *   **Implementation:**
        *   **Haystack is our Abstraction Layer:** The primary reason we use Haystack is that it provides these abstractions. Our `indexing_pipeline` does not know it is talking to `PlexFetcher`. It just knows it's talking to a component that *produces* `Document` objects.
        *   **Designing for the Future:** When we build the "Picture Module," we will not change the core indexing pipeline. We will create a **new pipeline**, `picture_indexing_pipeline.py`. This new pipeline will reuse the same `ChonkySplitter` (if we're chunking VLM descriptions), the same `FastembedDocumentEmbedder`, and the same `QdrantDocumentStore` writer component.
        *   **The `Document` is our Universal Data Format:** The Haystack `Document` object (with its `content` and `meta` fields) is the standardized "shipping container" for data in our system. Any future data source (PDFs, web pages) must have a "Fetcher" component whose job is to transform that source data into this standard `Document` format. As long as it does that, it can be plugged into the rest of our reusable embedding and storage machinery.

---

### **3. Development Workflow & Best Practices**

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
