# Database Population Tests

This directory contains comprehensive tests for the database population functionality.

## Test Structure

```
test/database_population/
├── __init__.py                          # Package initialization
├── conftest.py                          # Pytest fixtures and configuration
├── test_batch_processor.py             # Unit tests for batch processing
├── test_utils.py                       # Unit tests for utility functions
├── test_integration.py                 # Integration tests
├── test_qdrant_writer_integration.py   # QdrantWriter integration tests
├── run_tests.py                        # Test runner script
└── README.md                           # This file
```

## Running Tests

### Quick Start

```bash
# Run all database population tests
cd test/database_population
python run_tests.py

# Or use pytest directly
pytest test/database_population/
```

### Test Runner Options

```bash
# Run with verbose output
python run_tests.py -v

# Run specific test pattern
python run_tests.py -k test_batch_processor

# Run with coverage report
python run_tests.py --coverage

# Stop on first failure
python run_tests.py -x

# List available tests
python run_tests.py --list-tests

# Skip integration tests (faster)
python run_tests.py -k "not integration"
```

### Using Pytest Directly

```bash
# Run all tests with verbose output
pytest test/database_population/ -v

# Run specific test file
pytest test/database_population/test_batch_processor.py

# Run specific test class
pytest test/database_population/test_batch_processor.py::TestProcessSrtDirectory

# Run specific test method
pytest test/database_population/test_batch_processor.py::TestProcessSrtDirectory::test_process_srt_directory_success
```

## Test Categories

### Unit Tests

**test_batch_processor.py**
- `TestFindSrtFiles`: SRT file discovery functionality
- `TestProcessSingleSrtFile`: Single file processing logic
- `TestClearQdrantDatabase`: Database clearing operations
- `TestProcessSrtDirectory`: Complete directory processing
- `TestProcessingResult`: Result data structures
- `TestBatchProcessingResult`: Batch result data structures

**test_utils.py**
- `TestValidateDirectory`: Directory validation
- `TestGetFileSizeMb`: File size calculations
- `TestFormatProcessingSummary`: Result formatting
- `TestEstimateProcessingTime`: Time estimation
- `TestGetDirectoryStats`: Directory statistics

### Integration Tests

**test_integration.py**
- `TestEndToEndProcessing`: Complete workflow testing
- `TestDirectoryStatistics`: Directory analysis
- `TestErrorHandling`: Error scenario handling
- `TestPerformanceCharacteristics`: Performance testing
- `TestEdgeCases`: Boundary condition testing

**test_qdrant_writer_integration.py**
- `TestQdrantWriterDatabaseOperations`: Database operations
- `TestQdrantWriterErrorScenarios`: Error handling

## Test Fixtures

The tests use several fixtures defined in `conftest.py`:

- `sample_srt_content`: Standard SRT content for testing
- `temp_srt_file`: Temporary SRT file
- `temp_directory_with_srt_files`: Directory with multiple SRT files
- `mock_pipeline_success/failure`: Mock pipeline objects
- `mock_qdrant_writer_success/failure`: Mock QdrantWriter objects

## Mocking Strategy

The tests use extensive mocking to avoid dependencies on:
- Actual Qdrant database connections
- File system operations (where appropriate)
- Network operations
- Slow processing operations

This ensures tests run quickly and reliably without external dependencies.

## Coverage

To generate coverage reports:

```bash
python run_tests.py --coverage
```

This generates both terminal and HTML coverage reports. The HTML report is saved to `htmlcov/index.html`.

## Continuous Integration

For CI environments, use:

```bash
# Run tests with coverage and JUnit XML output
pytest test/database_population/ \
  --cov=src.database_population \
  --cov-report=xml \
  --cov-report=term \
  --junitxml=test-results.xml
```

## Test Data

Test files use sample SRT content that includes:
- Multiple subtitle entries
- Proper timestamp formatting
- Various text content including Unicode characters
- Edge cases like empty files and invalid content

## Performance Testing

The integration tests include performance characteristics testing:
- Processing time tracking
- Large directory handling
- Memory usage patterns (where applicable)

## Debugging Tests

For debugging failed tests:

```bash
# Run with extra verbose output and no capture
pytest test/database_population/ -vvs

# Run with debugger on failure
pytest test/database_population/ --pdb

# Run with specific log level
pytest test/database_population/ --log-cli-level=DEBUG
```