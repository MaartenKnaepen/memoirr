# Logging Guide for Memoirr

This guide explains how to use the structured logging system in Memoirr, which is designed for observability with Grafana, Loki, and Prometheus.

## Quick Start

```python
from src.core.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Processing started", component="chunker", items=42)
```

## Configuration

Set logging behavior via environment variables in `.env`:

```bash
# Development: Human-readable console output
LOG_FORMAT=console
LOG_LEVEL=INFO

# Production: JSON output for Loki/Grafana
LOG_FORMAT=json
LOG_LEVEL=WARNING
```

## Logging Patterns

### 1. Basic Structured Logging

```python
logger = get_logger(__name__)

# Good: Structured with context
logger.info(
    "Document processed successfully",
    document_id="doc_123",
    processing_time_ms=150,
    component="preprocessor"
)

# Avoid: Unstructured strings
logger.info("Document doc_123 processed in 150ms")
```

### 2. Error Logging

```python
try:
    process_document()
except Exception as e:
    logger.error(
        "Document processing failed",
        document_id="doc_123",
        error_type=type(e).__name__,
        error_message=str(e),
        component="preprocessor"
    )
```

### 3. Operation Tracking

```python
from src.core.logging_config import LoggedOperation

with LoggedOperation("srt_preprocessing", logger, file_count=5) as op:
    # Do work
    captions = process_srt(content)
    op.add_context(captions_processed=len(captions))
    # Automatically logs completion with timing
```

### 4. Performance Metrics

```python
from src.core.logging_config import MetricsLogger

metrics = MetricsLogger(logger)
metrics.counter("documents_processed_total", 1, component="chunker")
metrics.histogram("processing_duration_seconds", 1.5, operation="embedding")
metrics.gauge("memory_usage_mb", 256.5, component="embedder")
```

## Integration with Observability Stack

### Grafana Dashboards

The JSON log format includes standardized fields for dashboard creation:

- `service`: Service name (e.g., "memoirr")
- `environment`: Environment (development/staging/production)
- `component`: Component name (preprocessor/chunker/embedder/writer)
- `operation`: Operation name for tracking
- `duration_ms`: Timing data for performance monitoring

### Loki Queries

Common LogQL queries for Loki:

```logql
# All errors from chunker component
{service="memoirr"} |= "error" | json | component="chunker"

# Performance metrics
{service="memoirr"} |= "metric_histogram" | json

# Operation tracking
{service="memoirr"} |= "operation" | json | operation="srt_preprocessing"
```

### Prometheus Metrics

Metrics logged with `MetricsLogger` can be parsed and exposed for Prometheus:

```yaml
# Example metric extraction config
- metric_name: memoirr_documents_processed_total
  log_line_filter: 'metric_counter'
  value_field: 'value'
  labels: ['component', 'status']
```

## Migration from Print Statements

Replace existing print statements with structured logging:

```python
# Before
print(f"Warning: Failed to embed text {i}: {e}")

# After  
logger.warning(
    "Failed to embed text",
    text_index=i,
    error=str(e),
    component="embedder",
    fallback_action="using_zero_vector"
)
```

## Best Practices

1. **Always include component context**: Identify which part of the system generated the log
2. **Use structured fields**: Prefer separate fields over formatted strings
3. **Include timing data**: Use `duration_ms` for performance tracking
4. **Log business metrics**: Use `MetricsLogger` for counters, histograms, and gauges
5. **Consistent field names**: Use standard field names across components
6. **Log levels**:
   - `DEBUG`: Detailed debugging information
   - `INFO`: General information, normal operations
   - `WARNING`: Something unexpected but system continues
   - `ERROR`: Error conditions that need attention
   - `CRITICAL`: Serious errors requiring immediate attention

## Environment-Specific Behavior

### Development (`LOG_FORMAT=console`)
- Human-readable output with colors
- Full context displayed
- Suitable for local development

### Production (`LOG_FORMAT=json`)  
- JSON output for machine parsing
- Optimized for log aggregation
- Compatible with Loki, Fluentd, etc.

## Troubleshooting

### Logging Not Working
1. Ensure `structlog` is installed: `pip install structlog>=24.1.0`
2. Check that `src.core` is imported early in your application
3. Verify `.env` configuration is loaded

### Performance Concerns
- JSON logging has minimal overhead in production
- Use appropriate log levels to reduce noise
- Consider log sampling for high-volume operations

### Testing
- Use console format during development: `LOG_FORMAT=console`
- Mock loggers in unit tests to verify log calls
- Test log parsing with actual JSON output