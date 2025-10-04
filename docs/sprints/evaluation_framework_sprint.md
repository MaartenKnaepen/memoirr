# Sprint Plan: Haystack Native RAG Evaluation Framework

**Sprint Goal**: Build a comprehensive evaluation system using Haystack's native evaluators to measure RAG performance, with LOTR as sample dataset.

**Duration**: 1 week (5 days)  
**Target**: Phase 0 from `docs/future_development.md` - Measurement Foundation

## LLM Assistant Implementation Guide

This sprint plan leverages Haystack's built-in evaluation components for faster, more reliable implementation. Each task includes specific file paths, implementation details, and validation steps.

---

## Day 1: Core Framework Structure

### Task 1.1: Create Directory Structure
**Objective**: Set up the evaluation module structure following existing patterns.

**Files to Create**:
```
src/evaluation/
├── __init__.py
├── haystack_evaluator.py        # Main evaluation orchestrator using Haystack evaluators
├── pipelines/
│   ├── __init__.py
│   ├── evaluation_pipeline.py   # Haystack pipeline with native evaluators
│   └── baseline_pipeline.py     # Current RAG + evaluation components
├── test_data/
│   ├── __init__.py
│   ├── qdrant_query_generator.py
│   ├── ground_truth_builder.py  # Build evaluation datasets from Qdrant
│   └── missing_features_tests.py
├── visualization/
│   ├── __init__.py
│   ├── metrics_dataframe.py
│   └── evaluation_charts.py
└── README.md
```

**Implementation Notes**:
- Leverage existing Haystack evaluator components (no custom metrics needed)
- Follow existing component patterns in `src/components/`
- Use same import structure and docstring style
- Integration with existing `MetricsLogger` and pipeline architecture

### Task 1.2: Haystack Evaluation Orchestrator
**File**: `src/evaluation/haystack_evaluator.py`

**Requirements**:
- Use Haystack's native evaluator components instead of custom metrics
- Integrate with existing `MetricsLogger` from `src/core/logging_config.py`
- Support evaluation of existing RAG pipeline with LOTR data
- Return structured metrics compatible with existing logging infrastructure
- **Generate DataFrame output for visualization and charting**
- **Include failing tests for missing features (speaker attribution, etc.)**

**Key Methods to Implement**:
```python
class HaystackRAGEvaluator:
    def __init__(self, qdrant_collection_name: str = "lotr_evaluation")
    def evaluate_rag_pipeline(self, rag_pipeline) -> Dict[str, float]
    def run_faithfulness_evaluation(self, questions, contexts, answers) -> float
    def run_context_relevance_evaluation(self, questions, contexts) -> float
    def run_exact_match_evaluation(self, predicted_answers, ground_truth) -> float
    def run_document_recall_evaluation(self, retrieved_docs, relevant_docs) -> float
    def measure_pipeline_latency(self, rag_pipeline, queries) -> Dict[str, float]
    def get_results_dataframe(self) -> pd.DataFrame  # For visualization
    def test_missing_features(self, rag_pipeline) -> Dict[str, str]  # Planned failures
```

**Haystack Evaluators to Use**:
```python
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator, 
    AnswerExactMatchEvaluator,
    DocumentRecallEvaluator,
    DocumentNDCGEvaluator,
    LLMEvaluator
)
```

**Integration Points**:
- Import existing `RAGPipeline` from `src/pipelines/rag_pipeline.py`
- Use `MetricsLogger` for consistent metric reporting
- Follow error handling patterns from existing components
- **Connect to existing Qdrant collection with LOTR data**
- **Generate pandas DataFrames for metrics visualization**

---

## Day 2: Haystack Evaluation Pipeline Setup

### Task 2.1: Evaluation Pipeline Construction
**File**: `src/evaluation/pipelines/evaluation_pipeline.py`

**Requirements**:
- Build Haystack pipeline that combines RAG + evaluation components
- Use native Haystack evaluators instead of custom metrics
- Support multiple evaluation scenarios (faithfulness, context relevance, etc.)

**Pipeline Components to Add**:
```python
from haystack import Pipeline
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    AnswerExactMatchEvaluator,
    DocumentRecallEvaluator,
    DocumentNDCGEvaluator
)

def build_evaluation_pipeline() -> Pipeline:
    """Build pipeline with RAG + Haystack evaluators."""
    eval_pipeline = Pipeline()
    
    # Add evaluation components
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator())
    eval_pipeline.add_component("exact_match", AnswerExactMatchEvaluator())
    eval_pipeline.add_component("doc_recall", DocumentRecallEvaluator())
    eval_pipeline.add_component("doc_ndcg", DocumentNDCGEvaluator())
    
    return eval_pipeline
```

### Task 2.2: Baseline RAG + Evaluation Pipeline
**File**: `src/evaluation/pipelines/baseline_pipeline.py`

**Requirements**:
- Combine existing RAG pipeline with evaluation components
- Enable end-to-end evaluation in single pipeline run
- Follow existing pipeline patterns from `src/pipelines/`

**Integration Pattern**:
```python
def build_rag_with_evaluation_pipeline() -> Pipeline:
    """Combine existing RAG with Haystack evaluators."""
    # Import existing RAG components
    from src.pipelines.rag_pipeline import build_rag_pipeline
    
    # Build base RAG pipeline
    rag_pipeline = build_rag_pipeline()
    
    # Add evaluation components
    rag_pipeline.add_component("faithfulness_eval", FaithfulnessEvaluator())
    rag_pipeline.add_component("context_eval", ContextRelevanceEvaluator())
    
    # Connect RAG outputs to evaluation inputs
    rag_pipeline.connect("generator", "faithfulness_eval")
    rag_pipeline.connect("retriever", "context_eval")
    
    return rag_pipeline
```

---

## Day 3: Ground Truth Data Generation

### Task 3.1: Ground Truth Builder from Qdrant Data
**File**: `src/evaluation/test_data/ground_truth_builder.py`

**Requirements**:
- Extract realistic test queries from existing LOTR Qdrant collection
- Build ground truth labels for evaluation datasets
- Support multiple evaluation types (exact quotes, document relevance, etc.)

**Key Functions**:
```python
def extract_quote_queries_from_qdrant(collection_name: str, num_queries: int = 20) -> List[Dict]
def build_document_relevance_labels(queries: List[str], collection_name: str) -> List[Dict]
def create_faithfulness_test_set(collection_name: str) -> List[Dict]
def generate_context_relevance_test_set(collection_name: str) -> List[Dict]
```

**Ground Truth Data Structure**:
```python
@dataclass
class EvaluationDataPoint:
    query: str
    expected_answer: Optional[str]
    relevant_document_ids: List[str]
    ground_truth_contexts: List[str]
    evaluation_type: str  # "exact_match", "faithfulness", "context_relevance"
    metadata: Dict[str, Any]
```

### Task 3.2: Qdrant Query Generator Integration
**File**: `src/evaluation/test_data/qdrant_query_generator.py`

**Requirements**:
- Connect to existing Qdrant LOTR collection
- Extract actual chunks for realistic query generation
- Create evaluation datasets compatible with Haystack evaluators

**Key Functions**:
```python
def connect_to_qdrant_collection(collection_name: str) -> QdrantClient
def sample_chunks_for_evaluation(client: QdrantClient, num_samples: int = 100) -> List[Dict]
def generate_exact_quote_queries(chunks: List[Dict]) -> List[EvaluationDataPoint]
def generate_plot_summary_queries(chunks: List[Dict]) -> List[EvaluationDataPoint]
def validate_generated_queries(queries: List[EvaluationDataPoint]) -> bool
```

---

## Day 4: Baseline Evaluation & Missing Features

### Task 4.1: Baseline RAG System Evaluation
**File**: `src/evaluation/haystack_evaluator.py` (extend)

**Requirements**:
- Run comprehensive evaluation on current RAG system using Haystack evaluators
- Capture baseline performance metrics across all evaluation types
- Generate structured results for comparison with future improvements

**Key Evaluation Functions**:
```python
def run_baseline_evaluation(self, num_test_queries: int = 30) -> Dict[str, float]:
    """Run complete baseline evaluation using generated LOTR test data."""
    
def evaluate_faithfulness_baseline(self, test_data: List[EvaluationDataPoint]) -> float:
    """Evaluate how faithful current answers are to retrieved context."""
    
def evaluate_context_relevance_baseline(self, test_data: List[EvaluationDataPoint]) -> float:
    """Evaluate how relevant retrieved context is to queries."""
    
def evaluate_exact_match_baseline(self, test_data: List[EvaluationDataPoint]) -> float:
    """Evaluate exact quote finding capabilities."""
    
def measure_latency_baseline(self, test_queries: List[str]) -> Dict[str, float]:
    """Measure current system performance."""
```

### Task 4.2: Missing Features Test Framework
**File**: `src/evaluation/test_data/missing_features_tests.py`

**Requirements**:
- Create failing tests for unimplemented features using Haystack's LLMEvaluator
- Track how long features have been missing
- Provide clear roadmap for future development

**Missing Features to Test**:
```python
def create_speaker_attribution_tests() -> List[EvaluationDataPoint]:
    """Tests that will fail because speaker detection isn't implemented."""
    
def create_hybrid_search_tests() -> List[EvaluationDataPoint]:
    """Tests for BM25 + semantic search combination."""
    
def create_conversation_threading_tests() -> List[EvaluationDataPoint]:
    """Tests for dialogue context understanding."""
    
def track_missing_feature_duration(feature_name: str) -> Dict[str, Any]:
    """Track how long a feature has been missing/requested."""
```

**Integration with Haystack LLMEvaluator**:
```python
from haystack.components.evaluators import LLMEvaluator

# Create custom evaluator for missing features
speaker_evaluator = LLMEvaluator(
    instructions="Evaluate if the system can identify who said a specific quote. Return FAIL if speaker information is missing.",
    inputs=[("predicted_answer", str), ("query", str)],
    outputs=["score"]
)
```

### Task 4.3: Visualization DataFrame Builder
**File**: `src/evaluation/visualization/metrics_dataframe.py`

**Purpose**: Convert evaluation results into pandas DataFrames for analysis and charting.

**Key Functions**:
```python
def build_evaluation_dataframe(evaluation_results: Dict[str, Any]) -> pd.DataFrame
def build_query_performance_dataframe(query_results: List[Dict]) -> pd.DataFrame
def build_latency_dataframe(latency_measurements: List[Dict]) -> pd.DataFrame
def build_missing_features_dataframe(missing_features: Dict[str, str]) -> pd.DataFrame
```

**DataFrame Structures**:
```python
# Main evaluation metrics DataFrame
columns = ["metric_name", "value", "timestamp", "query_type", "status"]

# Query-level performance DataFrame  
columns = ["query_id", "query_text", "query_type", "latency_ms", "recall_at_5", 
           "precision_at_5", "found_exact_phrase", "qdrant_hits"]

# Missing features tracking DataFrame
columns = ["feature_name", "status", "expected_completion", "test_count", "failure_reason"]
```

### Task 4.4: Missing Features Test Framework
**File**: `src/evaluation/test_data/missing_features_queries.py`

**Purpose**: Create failing tests that track unimplemented features and measure how long they've been missing.

**Missing Features to Track**:
1. **Speaker Attribution**: "Who said 'One does not simply walk into Mordor'?"
2. **Speaker Filtering**: "Show me all quotes by Gandalf"
3. **Conversation Threading**: "What was the conversation between Frodo and Sam about the ring?"
4. **Content Type Detection**: "Find dialogue vs narration segments"
5. **Hybrid Search**: "Exact phrase + semantic similarity combined ranking"

**Function Signatures**:
```python
def create_speaker_attribution_tests() -> List[TestQuery]
def create_conversation_threading_tests() -> List[TestQuery] 
def create_hybrid_search_tests() -> List[TestQuery]
def track_feature_missing_duration(feature_name: str) -> timedelta
def generate_feature_roadmap_data() -> pd.DataFrame
```

---

## Day 5: Integration and Visualization

### Task 5.1: Evaluation Charts and Visualization
**File**: `src/evaluation/visualization/evaluation_charts.py`

**Requirements**:
- Create visualization functions for evaluation metrics
- Generate charts showing RAG performance over time
- Track missing feature timelines and roadmap progress
- Support export to common formats (PNG, SVG, HTML)

**Key Functions**:
```python
def create_metrics_overview_chart(df: pd.DataFrame) -> matplotlib.figure.Figure
def create_latency_distribution_chart(df: pd.DataFrame) -> matplotlib.figure.Figure
def create_missing_features_timeline(df: pd.DataFrame) -> matplotlib.figure.Figure
def create_query_type_performance_chart(df: pd.DataFrame) -> matplotlib.figure.Figure
def export_evaluation_dashboard(evaluation_results: Dict) -> str  # Returns HTML path
```

**Chart Types to Implement**:
1. **Metrics Dashboard**: Overview of all evaluation metrics
2. **Latency Analysis**: Response time distribution and trends  
3. **Missing Features Timeline**: Track how long features have been missing
4. **Query Performance Breakdown**: Success rate by query type
5. **Feature Roadmap**: Visual progress tracking for planned improvements

### Task 5.2: Integration Tests with Missing Features
**File**: `test/evaluation/test_universal_evaluator.py`

**Test Requirements**:
- Test evaluation pipeline end-to-end with **actual Qdrant LOTR data**
- Validate all metric calculations with known inputs
- **Include failing tests for missing features with clear tracking**
- Test error handling for malformed queries or missing data
- Ensure metrics are content-agnostic (could work with different sample data)
- **Test DataFrame generation and visualization functions**

**Key Test Cases**:
```python
def test_evaluate_retrieval_quality_with_qdrant_data()
def test_exact_phrase_matching_with_lotr_quotes()
def test_metric_calculation_accuracy()
def test_performance_measurement_consistency()
def test_dataframe_generation()
def test_visualization_chart_creation()

# Missing feature tests (expected to fail)
@pytest.mark.xfail(reason="Speaker attribution not implemented", strict=True)
def test_speaker_attribution_queries()

@pytest.mark.xfail(reason="Hybrid search not implemented", strict=True) 
def test_hybrid_search_performance()

@pytest.mark.xfail(reason="Conversation threading not implemented", strict=True)
def test_conversation_threading()
```

### Task 5.3: Documentation and Usage Examples
**File**: `src/evaluation/README.md`

**Content Requirements**:
- Overview of evaluation framework architecture
- Usage examples for different evaluation scenarios
- Explanation of each metric and its interpretation
- Guidelines for creating new test datasets
- Integration instructions for existing RAG systems

**Usage Example to Include**:
```python
from src.evaluation.haystack_evaluator import HaystackRAGEvaluator
from src.pipelines.rag_pipeline import RAGPipeline
from src.evaluation.visualization.evaluation_charts import export_evaluation_dashboard

# Initialize evaluation with existing Qdrant LOTR data
evaluator = HaystackRAGEvaluator(qdrant_collection_name="lotr_evaluation")
rag_pipeline = RAGPipeline()

# Run comprehensive baseline evaluation using Haystack evaluators
baseline_results = evaluator.run_baseline_evaluation(num_test_queries=30)
print(f"Faithfulness score: {baseline_results['faithfulness']:.2f}")
print(f"Context relevance: {baseline_results['context_relevance']:.2f}")
print(f"Exact match accuracy: {baseline_results['exact_match']:.2f}")
print(f"Average response time: {baseline_results['avg_latency_ms']:.0f}ms")

# Generate DataFrames for analysis and visualization
metrics_df = evaluator.get_results_dataframe()
print(f"Generated DataFrame with {len(metrics_df)} evaluation records")

# Create evaluation dashboard with charts
dashboard_path = export_evaluation_dashboard(baseline_results)
print(f"Evaluation dashboard saved to: {dashboard_path}")

# Track missing features with failing tests
missing_features = evaluator.test_missing_features(rag_pipeline)
for feature, status in missing_features.items():
    print(f"❌ {feature}: {status}")

# Example output:
# ❌ speaker_attribution: FAIL - No speaker information in responses
# ❌ hybrid_search: FAIL - Only semantic search, no BM25 integration
# ❌ conversation_threading: FAIL - No dialogue context awareness
```

---

## Validation Criteria

### Sprint Success Metrics:
- [ ] **Haystack native evaluators integrated and running on existing Qdrant LOTR data**
- [ ] Baseline performance measurements captured using FaithfulnessEvaluator, ContextRelevanceEvaluator, AnswerExactMatchEvaluator
- [ ] At least 3 different evaluation types successfully running (faithfulness, context relevance, exact match)
- [ ] **pandas DataFrames generated for all Haystack evaluation results**
- [ ] **Visualization charts created and exported successfully**
- [ ] Content-agnostic validation confirmed (Haystack evaluators work universally)
- [ ] Integration with existing MetricsLogger working
- [ ] Response time measurement shows current system performance
- [ ] **Missing feature tests implemented using LLMEvaluator and failing as expected**
- [ ] **Feature timeline tracking operational**
- [ ] Documentation complete with Haystack integration examples and chart examples

### Code Quality Checks:
- [ ] Follows existing codebase patterns and style
- [ ] Type hints on all function signatures
- [ ] Google-style docstrings on all classes and methods
- [ ] Error handling matches existing component patterns
- [ ] Unit tests achieve >90% coverage
- [ ] Integration tests pass with mocked external dependencies

### Performance Requirements:
- [ ] Evaluation suite runs in under 60 seconds for 30 test queries
- [ ] Memory usage remains reasonable for test dataset size
- [ ] No degradation of existing RAG pipeline performance

---

## Implementation Notes for LLM Assistant

**Code Style Consistency**:
- Study existing files in `src/components/` for patterns
- Use same import organization and error handling
- Follow the logging patterns from `src/core/logging_config.py`
- Match docstring style from existing components

**Testing Patterns**:
- Follow test structure from `test/pipelines/test_rag_integration.py`
- Use pytest fixtures from `test/conftest.py`
- Mock external dependencies (don't require real LOTR data files)

**Integration Points**:
- Evaluation system should work with existing `RAGPipeline` class
- Use `MetricsLogger` for consistent metric reporting
- Support both programmatic usage and potential CLI interface

**Error Handling**:
- Handle missing test data files gracefully
- Provide informative error messages for malformed queries
- Fail fast with clear error descriptions

This sprint plan provides the measurement foundation needed before implementing retrieval improvements, following the enterprise RAG principle of "measure before optimizing."