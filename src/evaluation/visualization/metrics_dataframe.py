"""Convert Haystack evaluation results into pandas DataFrames.

This module provides functions to transform evaluation results from Haystack
evaluators into structured DataFrames for analysis and visualization.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from src.core.logging_config import get_logger


def build_evaluation_dataframe(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """Convert evaluation results into structured DataFrame.
    
    Args:
        evaluation_results: Results from HaystackRAGEvaluator.run_baseline_evaluation()
        
    Returns:
        DataFrame with evaluation metrics and metadata
        
    Example:
        results = evaluator.run_baseline_evaluation()
        df = build_evaluation_dataframe(results)
        print(df.groupby('metric_name')['value'].mean())
    """
    logger = get_logger(__name__)
    
    try:
        records = []
        timestamp = datetime.now().isoformat()
        
        # Convert flat results dictionary to records
        for metric_name, value in evaluation_results.items():
            records.append({
                "metric_name": metric_name,
                "value": value,
                "timestamp": timestamp,
                "evaluation_type": _get_evaluation_type(metric_name),
                "status": "completed" if value is not None else "failed"
            })
        
        if not records:
            # Return empty DataFrame with correct columns
            df = pd.DataFrame(columns=["metric_name", "value", "timestamp", "evaluation_type", "status"])
        else:
            df = pd.DataFrame(records)
        
        logger.info(
            "Built evaluation DataFrame",
            num_records=len(df),
            metrics=list(evaluation_results.keys()),
            component="metrics_dataframe"
        )
        
        return df
        
    except Exception as e:
        logger.error(
            "Failed to build evaluation DataFrame",
            error=str(e),
            error_type=type(e).__name__,
            component="metrics_dataframe"
        )
        raise


def build_query_performance_dataframe(query_results: List[Dict]) -> pd.DataFrame:
    """Build DataFrame from individual query evaluation results.
    
    Args:
        query_results: List of individual query evaluation results
        
    Returns:
        DataFrame with query-level performance metrics
        
    Columns:
        - query_id: Unique identifier for the query
        - query_text: The actual query text
        - query_type: Type of query (exact_match, faithfulness, etc.)
        - latency_ms: Response time in milliseconds
        - faithfulness_score: Faithfulness evaluation score
        - context_relevance_score: Context relevance score
        - exact_match_score: Exact match accuracy
        - qdrant_hits: Number of documents retrieved from Qdrant
    """
    logger = get_logger(__name__)
    
    if not query_results:
        logger.warning("No query results provided for DataFrame")
        return pd.DataFrame()
    
    try:
        records = []
        
        for i, result in enumerate(query_results):
            record = {
                "query_id": result.get("query_id", f"query_{i}"),
                "query_text": result.get("query", ""),
                "query_type": result.get("evaluation_type", "unknown"),
                "latency_ms": result.get("latency_ms", 0.0),
                "faithfulness_score": result.get("faithfulness", None),
                "context_relevance_score": result.get("context_relevance", None), 
                "exact_match_score": result.get("exact_match", None),
                "qdrant_hits": result.get("num_retrieved_docs", 0),
                "timestamp": result.get("timestamp", datetime.now().isoformat())
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        logger.info(
            "Built query performance DataFrame",
            num_queries=len(df),
            avg_latency=df["latency_ms"].mean(),
            component="metrics_dataframe"
        )
        
        return df
        
    except Exception as e:
        logger.error(
            "Failed to build query performance DataFrame",
            error=str(e),
            component="metrics_dataframe"
        )
        raise


def build_latency_dataframe(latency_measurements: List[Dict]) -> pd.DataFrame:
    """Build DataFrame from latency measurement data.
    
    Args:
        latency_measurements: List of latency measurement results
        
    Returns:
        DataFrame with latency statistics and percentiles
    """
    logger = get_logger(__name__)
    
    try:
        if not latency_measurements:
            return pd.DataFrame(columns=["measurement_id", "latency_ms", "percentile", "measurement_type"])
        
        records = []
        
        for i, measurement in enumerate(latency_measurements):
            # Handle different latency measurement formats
            if isinstance(measurement, dict):
                if "latencies" in measurement:
                    # Individual latency measurements
                    for j, latency in enumerate(measurement["latencies"]):
                        records.append({
                            "measurement_id": f"{i}_{j}",
                            "latency_ms": latency,
                            "percentile": None,
                            "measurement_type": "individual",
                            "query": measurement.get("query", ""),
                            "timestamp": measurement.get("timestamp", datetime.now().isoformat())
                        })
                else:
                    # Summary statistics
                    for stat_name, value in measurement.items():
                        if "latency" in stat_name or "ms" in stat_name:
                            records.append({
                                "measurement_id": f"summary_{i}",
                                "latency_ms": value,
                                "percentile": _extract_percentile(stat_name),
                                "measurement_type": "summary",
                                "query": "",
                                "timestamp": datetime.now().isoformat()
                            })
        
        df = pd.DataFrame(records)
        
        logger.info(
            "Built latency DataFrame",
            num_measurements=len(df),
            avg_latency=df["latency_ms"].mean() if len(df) > 0 else 0,
            component="metrics_dataframe"
        )
        
        return df
        
    except Exception as e:
        logger.error(
            "Failed to build latency DataFrame",
            error=str(e),
            component="metrics_dataframe"
        )
        raise


def build_missing_features_dataframe(missing_features: Dict[str, str]) -> pd.DataFrame:
    """Build DataFrame for missing features tracking.
    
    Args:
        missing_features: Dictionary mapping feature names to failure reasons
        
    Returns:
        DataFrame with missing feature information and timeline data
        
    Columns:
        - feature_name: Name of the missing feature
        - status: Current status (not_implemented, in_progress, etc.)
        - failure_reason: Why the feature test failed
        - priority: Feature priority (high, medium, low)
        - estimated_effort: Effort estimate (low, medium, high)
        - days_missing: Number of days since feature was identified as missing
        - expected_completion: Expected completion date
    """
    logger = get_logger(__name__)
    
    try:
        records = []
        
        # Feature metadata for roadmap tracking
        feature_metadata = {
            "speaker_attribution": {
                "priority": "high",
                "estimated_effort": "medium",
                "expected_completion": "2024-04-01",
                "dependencies": ["audio_processing"]
            },
            "hybrid_search": {
                "priority": "high", 
                "estimated_effort": "high",
                "expected_completion": "2024-03-15",
                "dependencies": ["bm25_integration"]
            },
            "conversation_threading": {
                "priority": "medium",
                "estimated_effort": "high", 
                "expected_completion": "2024-05-01",
                "dependencies": ["speaker_attribution"]
            },
            "content_type_detection": {
                "priority": "medium",
                "estimated_effort": "low",
                "expected_completion": "2024-02-15",
                "dependencies": []
            }
        }
        
        for feature_name, failure_reason in missing_features.items():
            metadata = feature_metadata.get(feature_name, {})
            
            record = {
                "feature_name": feature_name,
                "status": "not_implemented",
                "failure_reason": failure_reason,
                "priority": metadata.get("priority", "unknown"),
                "estimated_effort": metadata.get("estimated_effort", "unknown"),
                "expected_completion": metadata.get("expected_completion", "TBD"),
                "dependencies": ",".join(metadata.get("dependencies", [])),
                "test_count": 1,  # Number of tests for this feature
                "last_tested": datetime.now().isoformat(),
                "days_missing": _calculate_days_missing(feature_name)
            }
            records.append(record)
        
        if not records:
            # Return empty DataFrame with correct columns
            columns = ["feature_name", "status", "failure_reason", "priority", 
                      "estimated_effort", "expected_completion", "dependencies", 
                      "test_count", "last_tested", "days_missing"]
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(records)
        
        logger.info(
            "Built missing features DataFrame",
            num_features=len(df),
            high_priority_count=len(df[df["priority"] == "high"]),
            component="metrics_dataframe"
        )
        
        return df
        
    except Exception as e:
        logger.error(
            "Failed to build missing features DataFrame",
            error=str(e),
            component="metrics_dataframe"
        )
        raise


def _get_evaluation_type(metric_name: str) -> str:
    """Determine evaluation type from metric name."""
    if "faithfulness" in metric_name.lower():
        return "faithfulness"
    elif "context" in metric_name.lower():
        return "context_relevance"
    elif "exact" in metric_name.lower() or "match" in metric_name.lower():
        return "exact_match"
    elif "latency" in metric_name.lower() or "ms" in metric_name.lower():
        return "performance"
    elif "recall" in metric_name.lower() or "ndcg" in metric_name.lower():
        return "retrieval_quality"
    else:
        return "other"


def _extract_percentile(stat_name: str) -> Optional[float]:
    """Extract percentile value from statistic name."""
    if "p95" in stat_name.lower():
        return 95.0
    elif "p99" in stat_name.lower():
        return 99.0
    elif "avg" in stat_name.lower() or "mean" in stat_name.lower():
        return 50.0
    else:
        return None


def _calculate_days_missing(feature_name: str) -> int:
    """Calculate how many days a feature has been missing."""
    # TODO: Implement persistent tracking
    # For now, assume baseline missing since project start
    baseline_date = datetime(2024, 1, 1)
    days_missing = (datetime.now() - baseline_date).days
    return days_missing


def combine_evaluation_dataframes(
    evaluation_df: pd.DataFrame,
    query_df: pd.DataFrame,
    missing_features_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Combine all evaluation DataFrames for comprehensive analysis.
    
    Args:
        evaluation_df: Main evaluation metrics DataFrame
        query_df: Query-level performance DataFrame
        missing_features_df: Missing features tracking DataFrame
        
    Returns:
        Dictionary of combined and summary DataFrames
    """
    logger = get_logger(__name__)
    
    try:
        combined_data = {
            "evaluation_metrics": evaluation_df,
            "query_performance": query_df, 
            "missing_features": missing_features_df
        }
        
        # Create summary DataFrame
        summary_records = []
        
        if not evaluation_df.empty:
            summary_records.append({
                "category": "Overall Performance",
                "metric": "Average Score",
                "value": evaluation_df["value"].mean(),
                "count": len(evaluation_df)
            })
        
        if not query_df.empty:
            summary_records.append({
                "category": "Query Performance", 
                "metric": "Average Latency (ms)",
                "value": query_df["latency_ms"].mean(),
                "count": len(query_df)
            })
        
        if not missing_features_df.empty:
            summary_records.append({
                "category": "Missing Features",
                "metric": "High Priority Count",
                "value": len(missing_features_df[missing_features_df["priority"] == "high"]),
                "count": len(missing_features_df)
            })
        
        combined_data["summary"] = pd.DataFrame(summary_records)
        
        logger.info(
            "Combined evaluation DataFrames",
            total_categories=len(combined_data),
            component="metrics_dataframe"
        )
        
        return combined_data
        
    except Exception as e:
        logger.error(
            "Failed to combine evaluation DataFrames",
            error=str(e),
            component="metrics_dataframe"
        )
        raise