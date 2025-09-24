# Batch Processor Utilities

This directory contains the utility functions and data types for batch processing SRT files into the Qdrant database.

## Structure

- `types.py` - Data classes for processing results
- `file_operations.py` - File discovery and validation utilities
- `single_file_processor.py` - Processing logic for individual SRT files
- `database_operations.py` - Database management utilities (clearing, etc.)

## Usage

These utilities are automatically imported by the main `batch_processor.py` module in the parent directory. You can also import them directly:

```python
from src.database_population.utilities.batch_processor.types import ProcessingResult, BatchProcessingResult
from src.database_population.utilities.batch_processor.file_operations import find_srt_files
from src.database_population.utilities.batch_processor.single_file_processor import process_single_srt_file
from src.database_population.utilities.batch_processor.database_operations import clear_qdrant_database
```

Or use the main interface:

```python
from src.database_population import process_srt_directory, BatchProcessingResult, ProcessingResult
```