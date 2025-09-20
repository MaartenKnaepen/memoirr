# SRT Preprocessor Utilities

This directory contains the pure utility functions that implement the SRT preprocessing pipeline. These functions are orchestrated by the SRTPreprocessor component.

## Modules

### srt_processing.py
The main orchestration function that composes all the preprocessing utilities into a complete pipeline.

### parse_srt.py
Parses raw SRT text into structured caption records with timing information.

### clean_lines.py
Cleans individual caption lines by removing HTML tags, speaker cues, sound descriptions, and other noise.

### drop_empty.py
Filters out empty or near-empty captions that don't contain meaningful content.

### language_filter.py
Filters captions by language using langdetect or ASCII-based heuristics for English text.

### deduplicate.py
Removes near-duplicate captions that occur within a specified time window.

### apply_cleaning.py
Applies all cleaning steps in the correct order to produce final cleaned captions.

### to_jsonl.py
Converts cleaned caption records to JSONL format for downstream processing.

### types.py
Type definitions and dataclasses for the preprocessing pipeline.

### exceptions.py
Custom exceptions for handling preprocessing errors.