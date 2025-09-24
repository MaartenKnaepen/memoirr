# Database Population

This module provides batch processing functionality to populate the Qdrant database with SRT files.

## Features

- Recursive processing of SRT files in directories
- Batch processing with configurable overwrite behavior
- Error handling and logging for individual file failures
- Integration with existing SRT-to-Qdrant pipeline

## Usage

```python
from src.database_population.batch_processor import process_srt_directory

# Process all SRT files in a directory, adding to existing database
process_srt_directory("/path/to/srt/files", overwrite=False)

# Process all SRT files in a directory, overwriting existing database
process_srt_directory("/path/to/srt/files", overwrite=True)
```