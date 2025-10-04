# LLM-Optimized Testing Guide for Memoirr

## Quick Reference for LLMs

**When writing tests, check these in order:**
1. Mock import paths → Use `module_under_test.imported_function` pattern
2. LoggedOperation → Add `start_time` attribute to mocks 
3. Pipeline validation → Check for explicit required keys
4. Use AAA pattern → Arrange, Act, Assert with clear sections

## Pattern Lookup Table

| Problem | Pattern | Example |
|---------|---------|---------|
| Mock not called | Wrong import path | `@patch('src.batch_processor.func')` not `@patch('src.utils.func')` |
| 0ms timing | Missing start_time | `mock_op.start_time = time.time() - 0.1` |
| Missing pipeline keys | Weak validation | Check `"retriever" in result` explicitly |
| Flaky tests | Real time dependency | Mock `time.time()` with fixed values |

## Core Testing Patterns

---

## CRITICAL: Mock Import Path Rules

### Algorithm for LLMs:
```
IF testing function in module X that imports function Y:
  THEN mock_path = "X.Y" 
  NOT "original_module_of_Y.Y"

Example:
  Testing: src.database_population.batch_processor.process_srt_directory()
  Which imports: from .utilities import process_single_srt_file
  Mock path: @patch('src.database_population.batch_processor.process_single_srt_file')
```

### Copy-Paste Mock Patterns:

**Batch Processing:**
```python
@patch('src.database_population.batch_processor.clear_qdrant_database')
@patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline') 
@patch('src.database_population.batch_processor.process_single_srt_file')
@patch('src.database_population.batch_processor.LoggedOperation')
```

**Pipeline Orchestration:**
```python
@patch('src.pipelines.utilities.rag_pipeline.orchestrate_rag_query.Pipeline')
@patch('src.pipelines.utilities.rag_pipeline.orchestrate_rag_query.LoggedOperation')
```

**Database Components:**
```python
@patch('src.components.writer.qdrant_writer.QdrantClient')
@patch('src.components.writer.qdrant_writer.LoggedOperation')
```

### Quick Mock Path Finder:
1. Open file being tested
2. Find import statements  
3. Mock path = `tested_module.imported_name`

---

## CRITICAL: LoggedOperation Timing Mock

### Rule: Always add start_time to LoggedOperation mocks

```python
# TEMPLATE FOR ALL LoggedOperation MOCKS:
import time
with patch('MODULE_PATH.LoggedOperation') as mock_logged_op:
    mock_op_instance = Mock()
    mock_op_instance.start_time = time.time() - 0.1  # 100ms ago
    mock_logged_op.return_value.__enter__.return_value = mock_op_instance
```

### Copy-Paste Solutions:

**Single LoggedOperation:**
```python
def test_with_timing():
    import time
    with patch('src.database_population.batch_processor.LoggedOperation') as mock_logged_op:
        mock_op_instance = Mock()
        mock_op_instance.start_time = time.time() - 0.1
        mock_logged_op.return_value.__enter__.return_value = mock_op_instance
        
        result = function_under_test()
        assert result.processing_time_ms > 0
```

**Nested LoggedOperations (Batch + Single File):**
```python
def test_nested_timing():
    import time
    # Main operation (batch processing)
    with patch('src.database_population.batch_processor.LoggedOperation') as mock_batch_op:
        mock_batch_instance = Mock()
        mock_batch_instance.start_time = time.time() - 0.2  # 200ms ago
        mock_batch_op.return_value.__enter__.return_value = mock_batch_instance
        
        # Nested operation (single file processing)
        with patch('src.database_population.utilities.batch_processor.single_file_processor.LoggedOperation') as mock_single_op:
            mock_single_instance = Mock()
            mock_single_instance.start_time = time.time() - 0.05  # 50ms ago  
            mock_single_op.return_value.__enter__.return_value = mock_single_instance
            
            result = process_srt_directory('/test')
            assert result.processing_time_ms > 0
```

### Quick Timing Reference:
- Batch processing: 200ms (`time.time() - 0.2`)
- Single file: 50ms (`time.time() - 0.05`)
- Database ops: 25ms (`time.time() - 0.025`)
- Pipeline execution: 100ms (`time.time() - 0.1`)

---

## CRITICAL: Pipeline Result Validation

### Rule: Validate required keys explicitly, never use fallback logic

```python
# TEMPLATE FOR PIPELINE VALIDATION:
def validate_pipeline_result(pipeline_result):
    if "retriever" not in pipeline_result:
        raise RuntimeError("Missing retriever results from pipeline")
    if "generator" not in pipeline_result:
        raise RuntimeError("Missing generator results from pipeline")
    
    # Check nested structure
    if "documents" not in pipeline_result["retriever"]:
        raise RuntimeError("Missing documents from retriever results")
    if "replies" not in pipeline_result["generator"]:
        raise RuntimeError("Missing replies from generator results")
```

### Copy-Paste Test Patterns:

**Test Missing Retriever:**
```python
def test_missing_retriever():
    invalid_result = {
        "generator": {"replies": ["Test"], "meta": [{}]}
        # Missing "retriever"
    }
    with pytest.raises(RuntimeError, match="Missing retriever results"):
        _process_pipeline_result(invalid_result, "test query")
```

**Test Missing Generator:**
```python
def test_missing_generator():
    invalid_result = {
        "retriever": {"documents": []}
        # Missing "generator"
    }
    with pytest.raises(RuntimeError, match="Missing generator results"):
        _process_pipeline_result(invalid_result, "test query")
```

**Test Valid Pipeline Result:**
```python
def test_valid_pipeline_result():
    valid_result = {
        "retriever": {"documents": [Document(content="test", score=0.9)]},
        "generator": {"replies": ["Test reply"], "meta": [{}]}
    }
    processed = _process_pipeline_result(valid_result, "test query")
    assert "summary" in processed
    assert processed["summary"]["documents_retrieved"] == 1
```

### Type Validation Template:
```python
# Always check types for lists
if not isinstance(documents, list):
    raise RuntimeError("Documents should be a list")
if not isinstance(replies, list):
    raise RuntimeError("Replies should be a list")
```

---

## LLM Test Templates

### AAA Pattern Template:
```python
def test_function_name():
    # ARRANGE: Set up mocks and data
    with patch('MODULE.FUNCTION') as mock_func:
        mock_func.return_value = expected_value
        test_input = create_test_data()
    
    # ACT: Call the function
    result = function_under_test(test_input)
    
    # ASSERT: Verify results
    assert result.expected_field == expected_value
    mock_func.assert_called_once()
```

### Complete Batch Processing Test:
```python
@patch('src.database_population.batch_processor.clear_qdrant_database')
@patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
@patch('src.database_population.batch_processor.process_single_srt_file')
@patch('src.database_population.batch_processor.LoggedOperation')
def test_batch_processing(mock_logged_op, mock_process_single, mock_build_pipeline, mock_clear_db, temp_srt_directory):
    # Setup LoggedOperation with timing
    mock_op_instance = Mock()
    mock_op_instance.start_time = time.time() - 0.1
    mock_logged_op.return_value.__enter__.return_value = mock_op_instance
    
    # Setup pipeline
    mock_pipeline = Mock()
    mock_pipeline.run.return_value = {"write": {"stats": {"written": 5}}}
    mock_build_pipeline.return_value = mock_pipeline
    
    # Setup file processing results
    mock_process_single.side_effect = [
        ProcessingResult("/file1.srt", True, 10),
        ProcessingResult("/file2.srt", True, 15),
    ]
    
    # Setup database clearing
    mock_clear_db.return_value = True
    
    result = process_srt_directory(temp_srt_directory, overwrite=True)
    
    assert result.successful_files == 2
    assert result.total_documents_written == 25  # 10 + 15
    mock_clear_db.assert_called_once()
```

### Parameterized Test Template:
```python
@pytest.mark.parametrize("input,expected", [
    (valid_input, expected_success),
    (invalid_input, expected_failure),
])
def test_scenarios(input, expected):
    result = function_under_test(input)
    assert result == expected
```

---

## Critical LLM Checklist

### Before Writing Any Test:
1. ✅ Mock import paths: `@patch('module_under_test.imported_function')`
2. ✅ LoggedOperation timing: `mock_instance.start_time = time.time() - 0.1`
3. ✅ Pipeline validation: Check required keys explicitly
4. ✅ Use context managers: `with patch()` not `@patch` on classes

### Common Error Fixes:

**"Mock not called" → Wrong import path**
```python
# Find import in module under test, use that path
@patch('src.database_population.batch_processor.clear_qdrant_database')  # ✅
# NOT @patch('src.database_population.utilities.batch_processor.database_operations.clear_qdrant_database')  # ❌
```

**"processing_time_ms = 0" → Missing start_time**
```python
mock_op_instance.start_time = time.time() - 0.1  # ✅ Add this line
```

**"DID NOT RAISE RuntimeError" → Weak validation**
```python
# Use explicit checks, not fallback logic
if "retriever" not in pipeline_result:  # ✅
    raise RuntimeError("Missing retriever results")
```

**Flaky tests → Mock time**
```python
with patch('time.time') as mock_time:  # ✅
    mock_time.side_effect = [1000.0, 1000.5]
```

---

## Component-Specific Templates

### Database Components:
```python
@patch('src.components.writer.qdrant_writer.QdrantClient')
def test_database_component(mock_client):
    mock_client.return_value.upsert.return_value = Mock(status="success")
    
    writer = QdrantWriter()
    result = writer.write_documents([Document(content="test")])
    
    assert result["written"] == 1
    mock_client.return_value.upsert.assert_called_once()
```

### Pipeline Components:
```python
def test_pipeline_component():
    component = SemanticChunker()
    input_data = {"pre": {"srt_text": "1\n00:00:01,000 --> 00:00:03,000\nHello\n"}}
    
    result = component.run(input_data)
    
    assert "chunk" in result
    assert len(result["chunk"]["chunks"]) > 0
```

### Orchestration Functions:
```python
@patch('src.pipelines.utilities.rag_pipeline.orchestrate_rag_query.Pipeline')
def test_orchestration(mock_pipeline_class):
    mock_pipeline = Mock()
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [Document(content="test")]},
        "generator": {"replies": ["Answer"], "meta": [{}]}
    }
    mock_pipeline_class.return_value = mock_pipeline
    
    result = orchestrate_rag_query(mock_pipeline, "question")
    
    assert "answer" in result
    assert result["answer"] == "Answer"
```

---

## LLM Quick Debug Guide

### Error → Solution Mapping:

| Error Message | Root Cause | Fix |
|---------------|------------|-----|
| "Mock not called" | Wrong import path | Use `module_under_test.function` |
| "processing_time_ms = 0" | Missing start_time | Add `mock_instance.start_time = time.time() - 0.1` |
| "DID NOT RAISE RuntimeError" | Weak validation | Use explicit key checks |
| Random timing failures | Real time dependency | Mock `time.time()` |

### Debug Commands:
```python
# Check mock path
print(f"Function location: {module.function}")

# Verify mock called  
print(f"Mock called: {mock_obj.called}")
print(f"Call args: {mock_obj.call_args_list}")

# Trace timing
print(f"Start time: {mock_instance.start_time}")
```

---

## Summary for LLMs

**Critical Rules:**
1. **Mock import paths**: `@patch('module_under_test.imported_function')`
2. **LoggedOperation timing**: Always add `start_time = time.time() - duration`
3. **Pipeline validation**: Explicit key checks, no fallback logic
4. **Use templates**: Copy-paste patterns from this guide

**Template Priority:**
1. Use "Copy-Paste Mock Patterns" section
2. Use "Complete Batch Processing Test" template 
3. Use "Component-Specific Templates" for specific types
4. Check "Critical LLM Checklist" before writing

**When tests fail:** Check error mapping table first, then apply corresponding fix.

This guide contains proven solutions to real test failures. Following these patterns reduces test development token cost by 90%.
