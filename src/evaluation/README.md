# Haystack Native RAG Evaluation Framework

This evaluation framework uses Haystack's built-in evaluator components to measure RAG system performance on LOTR subtitle data. It provides comprehensive baseline measurements, missing feature tracking, and visualization tools.

## Overview

The evaluation system leverages Haystack's native evaluation capabilities instead of custom metrics, providing:

- **Baseline Performance Measurement**: Comprehensive evaluation using industry-standard metrics
- **Missing Feature Tracking**: Failing tests that track unimplemented capabilities
- **Visualization Tools**: DataFrames and charts for analysis and reporting
- **Content-Agnostic Design**: Works with any media collection, not just LOTR

## Architecture

```
src/evaluation/
├── haystack_evaluator.py           # Main evaluation orchestrator
├── pipelines/
│   ├── evaluation_pipeline.py      # Haystack evaluators pipeline
│   └── baseline_pipeline.py        # RAG + evaluation combined
├── test_data/
│   ├── ground_truth_builder.py     # Generate test data from Qdrant
│   ├── qdrant_query_generator.py   # Extract queries from existing data
│   └── missing_features_tests.py   # Track unimplemented features
└── visualization/
    ├── metrics_dataframe.py        # Convert results to DataFrames
    └── evaluation_charts.py        # Create charts and dashboards
```

## Haystack Evaluators Used

### Core Evaluation Components
- **`FaithfulnessEvaluator`**: Measures if generated answers are faithful to retrieved context
- **`ContextRelevanceEvaluator`**: Evaluates relevance of retrieved context to queries
- **`AnswerExactMatchEvaluator`**: Tests exact quote finding capabilities
- **`DocumentRecallEvaluator`**: Measures retrieval quality (relevant docs found)
- **`DocumentNDCGEvaluator`**: Ranking quality using Normalized Discounted Cumulative Gain

### Missing Feature Detection
- **`LLMEvaluator`**: Custom evaluator to detect missing capabilities like speaker attribution

## Quick Start

### Basic Evaluation

```python
from src.evaluation.haystack_evaluator import HaystackRAGEvaluator
from src.pipelines.rag_pipeline import RAGPipeline

# Initialize with existing Qdrant LOTR data
evaluator = HaystackRAGEvaluator(qdrant_collection_name="lotr_evaluation")
rag_pipeline = RAGPipeline()

# Run comprehensive baseline evaluation
baseline_results = evaluator.run_baseline_evaluation(num_test_queries=30)

print(f"Faithfulness score: {baseline_results['faithfulness']:.2f}")
print(f"Context relevance: {baseline_results['context_relevance']:.2f}")
print(f"Exact match accuracy: {baseline_results['exact_match']:.2f}")
print(f"Average response time: {baseline_results['avg_latency_ms']:.0f}ms")
```

### Generate Visualization Dashboard

```python
from src.evaluation.visualization.evaluation_charts import export_evaluation_dashboard

# Create comprehensive dashboard with charts
dashboard_path = export_evaluation_dashboard(baseline_results)
print(f"Dashboard saved to: {dashboard_path}")

# Generate DataFrames for custom analysis
metrics_df = evaluator.get_results_dataframe()
print(metrics_df.head())
```

### Track Missing Features

```python
# Test for unimplemented capabilities
missing_features = evaluator.test_missing_features(rag_pipeline)

for feature, status in missing_features.items():
    print(f"❌ {feature}: {status}")

# Example output:
# ❌ speaker_attribution: FAIL - No speaker information in responses
# ❌ hybrid_search: FAIL - Only semantic search, no BM25 integration
# ❌ conversation_threading: FAIL - No dialogue context awareness
```

## Evaluation Types

### 1. Faithfulness Evaluation
Tests whether generated answers can be inferred from the retrieved context.

```python
# Uses FaithfulnessEvaluator
from haystack.components.evaluators import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator()
questions = ["What did Gandalf say about the ring?"]
contexts = [["Gandalf warned about the ring's power"]]
predicted_answers = ["Gandalf mentioned the ring was dangerous"]

result = evaluator.run(
    questions=questions, 
    contexts=contexts, 
    predicted_answers=predicted_answers
)
print(result["score"])  # Overall faithfulness score
print(result["individual_scores"])  # Per-question scores
```

### 2. Context Relevance Evaluation
Measures how relevant retrieved context is to the input query.

```python
# Uses ContextRelevanceEvaluator
from haystack.components.evaluators import ContextRelevanceEvaluator

evaluator = ContextRelevanceEvaluator()
questions = ["Tell me about Gandalf's staff"]
contexts = [["Gandalf carried a wooden staff", "Frodo left the Shire"]]

result = evaluator.run(questions=questions, contexts=contexts)
print(result["score"])  # Overall relevance score
print(result["individual_scores"])  # Per-question scores
```

### 3. Exact Match Evaluation
Tests ability to find exact quotes in the retrieved documents.

```python
# Uses AnswerExactMatchEvaluator
from haystack.components.evaluators import AnswerExactMatchEvaluator

evaluator = AnswerExactMatchEvaluator()
predicted_answers = ["One does not simply walk into Mordor"]
ground_truth_answers = ["One does not simply walk into Mordor"]

result = evaluator.run(
    predicted_answers=predicted_answers,
    ground_truth_answers=ground_truth_answers
)
print(result["score"])  # Overall exact match accuracy
print(result["individual_scores"])  # Per-answer scores (1 or 0)
```

## Test Data Generation

### From Existing Qdrant Collection

```python
from src.evaluation.test_data.ground_truth_builder import build_evaluation_dataset

# Generate test queries from processed LOTR subtitle data
dataset = build_evaluation_dataset(
    collection_name="lotr_evaluation",
    num_queries=30,
    query_types=["exact_match", "faithfulness", "context_relevance"]
)

# Queries are automatically extracted from actual chunk content
for data_point in dataset[:3]:
    print(f"Query: {data_point.query}")
    print(f"Type: {data_point.evaluation_type}")
    print(f"Expected: {data_point.expected_answer}")
    print("---")
```

### Custom Query Generation

```python
from src.evaluation.test_data.qdrant_query_generator import generate_queries_from_qdrant

# Generate queries with specific distribution
queries = generate_queries_from_qdrant(
    collection_name="lotr_evaluation",
    num_queries=25,
    query_distribution={
        "exact_quote": 0.4,    # 40% exact quote queries
        "plot_summary": 0.3,   # 30% plot questions
        "temporal": 0.3        # 30% timeline questions
    }
)
```

## Visualization and Analysis

### Create Performance Charts

```python
from src.evaluation.visualization.evaluation_charts import (
    create_metrics_overview_chart,
    create_missing_features_timeline
)

# Overview of all metrics
metrics_df = evaluator.get_results_dataframe()
overview_chart = create_metrics_overview_chart(metrics_df)
overview_chart.savefig("metrics_overview.png")

# Missing features roadmap
missing_df = build_missing_features_dataframe(missing_features)
timeline_chart = create_missing_features_timeline(missing_df)
timeline_chart.savefig("feature_roadmap.png")
```

### DataFrame Analysis

```python
from src.evaluation.visualization.metrics_dataframe import (
    build_evaluation_dataframe,
    build_query_performance_dataframe
)

# Convert results to structured DataFrames
metrics_df = build_evaluation_dataframe(baseline_results)
query_df = build_query_performance_dataframe(individual_query_results)

# Analyze performance by evaluation type
performance_by_type = metrics_df.groupby('evaluation_type')['value'].agg(['mean', 'std'])
print(performance_by_type)

# Find slowest queries
slowest_queries = query_df.nlargest(5, 'latency_ms')[['query_text', 'latency_ms']]
print(slowest_queries)
```

## Missing Features Framework

The evaluation system tracks features not yet implemented:

### Tracked Missing Features

1. **Speaker Attribution**: "Who said X?" queries
2. **Hybrid Search**: BM25 + semantic search combination
3. **Conversation Threading**: Grouping related dialogue
4. **Content Type Detection**: Dialogue vs narration classification

### Roadmap Tracking

```python
from src.evaluation.test_data.missing_features_tests import (
    track_missing_feature_duration,
    generate_feature_roadmap_data
)

# Track how long features have been missing
duration = track_missing_feature_duration("speaker_attribution")
print(f"Speaker attribution missing for {duration['days_missing']} days")

# Get roadmap data for planning
roadmap = generate_feature_roadmap_data()
for feature in roadmap:
    print(f"{feature['name']}: {feature['estimated_effort_weeks']} weeks")
```

## Integration with Existing System

### Pipeline Integration

```python
from src.evaluation.pipelines.baseline_pipeline import build_rag_with_evaluation_pipeline

# Combine existing RAG with evaluation
combined_pipeline = build_rag_with_evaluation_pipeline(
    enable_evaluation=True
)

# Single pipeline run gets both response and evaluation
result = combined_pipeline.run({
    "query": "What did Gandalf say about the ring?"
})

answer = result["generator"]["replies"][0]
faithfulness = result["faithfulness_eval"]["score"]
```

### Logging Integration

The evaluation framework integrates with the existing `MetricsLogger`:

```python
# Evaluation results are automatically logged with consistent format
# Metrics appear in structured logs for Prometheus/Grafana integration
```

## Configuration

### Evaluator Configuration

```python
# Configure LLM for evaluation (uses same providers as generation)
evaluator = HaystackRAGEvaluator(
    qdrant_collection_name="lotr_evaluation"
)

# Custom evaluation pipeline
from src.evaluation.pipelines.evaluation_pipeline import build_evaluation_pipeline

eval_pipeline = build_evaluation_pipeline(
    llm_api_key="your_groq_key",  # For LLMEvaluator
    evaluator_config={
        "faithfulness": {"threshold": 0.8},
        "context_relevance": {"threshold": 0.7}
    }
)
```

## Best Practices

### 1. Content-Agnostic Design
- All evaluations work with any media collection
- No hardcoded LOTR-specific logic
- Universal query patterns and metrics

### 2. Baseline Before Optimization
- Establish baseline performance first
- Track improvements after each feature addition
- Document performance regressions

### 3. Failing Tests for Missing Features
- Use `@pytest.mark.xfail` for unimplemented features
- Clear failure reasons and timelines
- Convert to passing tests as features are implemented

### 4. Regular Evaluation Runs
- Automate evaluation after code changes
- Track metric trends over time
- Alert on performance degradation

## Troubleshooting

### Common Issues

1. **No Qdrant Connection**: Ensure LOTR data is processed and available
2. **Missing API Keys**: LLMEvaluator requires API access for some evaluations
3. **Empty DataFrames**: Check that evaluation results contain valid data
4. **Chart Generation Errors**: Verify matplotlib dependencies and output directories

### Debug Mode

```python
# Enable detailed logging
import logging
logging.getLogger("src.evaluation").setLevel(logging.DEBUG)

# Check evaluation pipeline status
pipeline_status = evaluator._validate_pipeline_components()
print(f"Pipeline validation: {pipeline_status}")
```

## Future Enhancements

As features are implemented, the evaluation framework will:

1. **Convert failing tests to passing**: Missing feature tests become regression tests
2. **Add new evaluation types**: Conversation quality, speaker accuracy metrics
3. **Expand test datasets**: Beyond LOTR to other media collections
4. **Performance benchmarking**: Latency/throughput optimization tracking

## Contributing

When adding new evaluation capabilities:

1. Follow existing Haystack evaluator patterns
2. Add comprehensive test coverage
3. Update visualization components
4. Document new metrics and their interpretation
5. Ensure content-agnostic design for universal applicability