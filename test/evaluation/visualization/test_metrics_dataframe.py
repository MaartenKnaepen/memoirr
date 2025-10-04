"""Tests for metrics DataFrame generation following Memoirr standards.

Tests the DataFrame conversion utilities using AAA pattern, comprehensive
mocking, and error handling validation.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from src.evaluation.visualization.metrics_dataframe import (
    build_evaluation_dataframe,
    build_query_performance_dataframe,
    build_latency_dataframe,
    build_missing_features_dataframe,
    combine_evaluation_dataframes,
    _get_evaluation_type,
    _extract_percentile,
    _calculate_days_missing
)


class TestBuildEvaluationDataframe:
    """Test cases for build_evaluation_dataframe function."""

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_evaluation_dataframe_with_valid_results(self, mock_get_logger):
        """Test DataFrame creation with valid evaluation results."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        evaluation_results = {
            "faithfulness": 0.85,
            "context_relevance": 0.92,
            "exact_match": 0.5,
            "avg_latency_ms": 1250.0
        }
        
        # ACT
        result_df = build_evaluation_dataframe(evaluation_results)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(evaluation_results)
        
        # Verify required columns
        expected_columns = {"metric_name", "value", "timestamp", "evaluation_type", "status"}
        assert set(result_df.columns) == expected_columns
        
        # Verify data content
        assert set(result_df["metric_name"]) == set(evaluation_results.keys())
        assert all(result_df["status"] == "completed")
        
        # Verify logging
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[1]
        assert log_call_args["num_records"] == len(result_df)

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_evaluation_dataframe_with_none_values(self, mock_get_logger):
        """Test DataFrame creation handles None values in results."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        evaluation_results = {
            "faithfulness": 0.85,
            "context_relevance": None,
            "exact_match": 0.5
        }
        
        # ACT
        result_df = build_evaluation_dataframe(evaluation_results)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        
        # Verify status handling for None values
        none_rows = result_df[result_df["metric_name"] == "context_relevance"]
        assert len(none_rows) == 1
        assert none_rows.iloc[0]["status"] == "failed"
        assert pd.isna(none_rows.iloc[0]["value"])

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_evaluation_dataframe_handles_exceptions(self, mock_get_logger):
        """Test DataFrame creation handles and logs exceptions."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Force an exception during DataFrame creation
        with patch('pandas.DataFrame', side_effect=ValueError("DataFrame creation failed")):
            # ACT & ASSERT
            with pytest.raises(ValueError, match="DataFrame creation failed"):
                build_evaluation_dataframe({"test": 0.5})
            
            # Verify error logging
            mock_logger.error.assert_called()
            error_call_args = mock_logger.error.call_args[1]
            assert error_call_args["error"] == "DataFrame creation failed"
            assert error_call_args["error_type"] == "ValueError"

    def test_build_evaluation_dataframe_with_empty_results(self):
        """Test DataFrame creation with empty results dictionary."""
        # ARRANGE & ACT
        result_df = build_evaluation_dataframe({})
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
        # Should still have the expected columns structure
        expected_columns = {"metric_name", "value", "timestamp", "evaluation_type", "status"}
        assert set(result_df.columns) == expected_columns


class TestBuildQueryPerformanceDataframe:
    """Test cases for build_query_performance_dataframe function."""

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_query_performance_dataframe_with_valid_data(self, mock_get_logger):
        """Test query performance DataFrame creation with valid data."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        query_results = [
            {
                "query_id": "q1",
                "query": "What is Python?",
                "evaluation_type": "faithfulness",
                "latency_ms": 1200.0,
                "faithfulness": 0.85,
                "context_relevance": 0.9,
                "num_retrieved_docs": 5
            },
            {
                "query_id": "q2",
                "query": "Who created Python?",
                "evaluation_type": "exact_match",
                "latency_ms": 800.0,
                "exact_match": 1.0,
                "num_retrieved_docs": 3
            }
        ]
        
        # ACT
        result_df = build_query_performance_dataframe(query_results)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(query_results)
        
        # Verify required columns
        expected_columns = {
            "query_id", "query_text", "query_type", "latency_ms",
            "faithfulness_score", "context_relevance_score", "exact_match_score",
            "qdrant_hits", "timestamp"
        }
        assert set(result_df.columns) == expected_columns
        
        # Verify data content
        assert result_df.iloc[0]["query_id"] == "q1"
        assert result_df.iloc[0]["faithfulness_score"] == 0.85
        assert result_df.iloc[1]["exact_match_score"] == 1.0
        
        # Verify logging
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[1]
        assert log_call_args["num_queries"] == len(result_df)
        assert "avg_latency" in log_call_args

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_query_performance_dataframe_with_empty_data(self, mock_get_logger):
        """Test query performance DataFrame creation with empty data."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # ACT
        result_df = build_query_performance_dataframe([])
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
        mock_logger.warning.assert_called_with("No query results provided for DataFrame")

    def test_build_query_performance_dataframe_handles_missing_fields(self):
        """Test DataFrame creation handles missing optional fields."""
        # ARRANGE
        query_results = [
            {
                "query": "Basic query",
                "latency_ms": 1000.0
                # Missing other optional fields
            }
        ]
        
        # ACT
        result_df = build_query_performance_dataframe(query_results)
        
        # ASSERT
        assert len(result_df) == 1
        
        # Verify default values for missing fields
        row = result_df.iloc[0]
        assert row["query_id"] == "query_0"  # Default generated ID
        assert row["query_type"] == "unknown"
        assert row["qdrant_hits"] == 0
        assert pd.isna(row["faithfulness_score"])


class TestBuildLatencyDataframe:
    """Test cases for build_latency_dataframe function."""

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_latency_dataframe_with_individual_measurements(self, mock_get_logger):
        """Test latency DataFrame creation with individual measurements."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        latency_measurements = [
            {
                "latencies": [1200.0, 1350.0, 1100.0],
                "query": "Test query 1",
                "timestamp": "2024-01-15T10:00:00"
            },
            {
                "latencies": [800.0, 950.0],
                "query": "Test query 2"
            }
        ]
        
        # ACT
        result_df = build_latency_dataframe(latency_measurements)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 5  # 3 + 2 individual measurements
        
        # Verify columns
        expected_columns = {"measurement_id", "latency_ms", "percentile", "measurement_type", "query", "timestamp"}
        assert set(result_df.columns) == expected_columns
        
        # Verify individual measurement data
        individual_rows = result_df[result_df["measurement_type"] == "individual"]
        assert len(individual_rows) == 5
        assert all(pd.isna(individual_rows["percentile"]))

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_latency_dataframe_with_summary_statistics(self, mock_get_logger):
        """Test latency DataFrame creation with summary statistics."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        latency_measurements = [
            {
                "avg_latency_ms": 1200.0,
                "p95_latency_ms": 1500.0,
                "p99_latency_ms": 1800.0
            }
        ]
        
        # ACT
        result_df = build_latency_dataframe(latency_measurements)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        
        # Verify summary measurement data
        summary_rows = result_df[result_df["measurement_type"] == "summary"]
        assert len(summary_rows) == 3  # avg, p95, p99
        
        # Verify percentile extraction
        percentiles = summary_rows["percentile"].dropna().tolist()
        assert 50.0 in percentiles  # avg should map to 50th percentile
        assert 95.0 in percentiles
        assert 99.0 in percentiles

    def test_build_latency_dataframe_with_empty_measurements(self):
        """Test latency DataFrame creation with empty measurements."""
        # ARRANGE & ACT
        result_df = build_latency_dataframe([])
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
        # Should have expected columns
        expected_columns = {"measurement_id", "latency_ms", "percentile", "measurement_type"}
        assert expected_columns.issubset(set(result_df.columns))


class TestBuildMissingFeaturesDataframe:
    """Test cases for build_missing_features_dataframe function."""

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_build_missing_features_dataframe_with_valid_data(self, mock_get_logger):
        """Test missing features DataFrame creation with valid data."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        missing_features = {
            "speaker_attribution": "FAIL - No speaker information in responses",
            "hybrid_search": "FAIL - Only semantic search, no BM25 integration",
            "conversation_threading": "FAIL - No dialogue context awareness"
        }
        
        # ACT
        result_df = build_missing_features_dataframe(missing_features)
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(missing_features)
        
        # Verify required columns
        expected_columns = {
            "feature_name", "status", "failure_reason", "priority",
            "estimated_effort", "expected_completion", "dependencies",
            "test_count", "last_tested", "days_missing"
        }
        assert set(result_df.columns) == expected_columns
        
        # Verify data content
        feature_names = set(result_df["feature_name"])
        assert feature_names == set(missing_features.keys())
        assert all(result_df["status"] == "not_implemented")
        
        # Verify logging
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[1]
        assert log_call_args["num_features"] == len(result_df)
        assert "high_priority_count" in log_call_args

    def test_build_missing_features_dataframe_includes_metadata(self):
        """Test that missing features DataFrame includes proper metadata."""
        # ARRANGE
        missing_features = {"speaker_attribution": "FAIL - Test failure"}
        
        # ACT
        result_df = build_missing_features_dataframe(missing_features)
        
        # ASSERT
        row = result_df.iloc[0]
        
        # Verify metadata fields are populated
        assert row["priority"] == "high"  # Speaker attribution should be high priority
        assert row["estimated_effort"] == "medium"
        assert row["expected_completion"] == "2024-04-01"
        assert "audio_processing" in row["dependencies"]
        assert row["test_count"] == 1
        assert isinstance(row["days_missing"], (int, np.integer))
        assert row["days_missing"] >= 0

    def test_build_missing_features_dataframe_with_empty_features(self):
        """Test missing features DataFrame creation with empty features."""
        # ARRANGE & ACT
        result_df = build_missing_features_dataframe({})
        
        # ASSERT
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_get_evaluation_type_with_known_metrics(self):
        """Test evaluation type detection for known metric names."""
        # ARRANGE & ACT & ASSERT
        assert _get_evaluation_type("faithfulness") == "faithfulness"
        assert _get_evaluation_type("context_relevance") == "context_relevance"
        assert _get_evaluation_type("exact_match") == "exact_match"
        assert _get_evaluation_type("avg_latency_ms") == "performance"
        assert _get_evaluation_type("doc_recall") == "retrieval_quality"
        assert _get_evaluation_type("unknown_metric") == "other"

    def test_get_evaluation_type_case_insensitive(self):
        """Test that evaluation type detection is case insensitive."""
        # ARRANGE & ACT & ASSERT
        assert _get_evaluation_type("FAITHFULNESS") == "faithfulness"
        assert _get_evaluation_type("Context_Relevance") == "context_relevance"
        assert _get_evaluation_type("EXACT_MATCH_SCORE") == "exact_match"

    def test_extract_percentile_with_known_statistics(self):
        """Test percentile extraction from statistic names."""
        # ARRANGE & ACT & ASSERT
        assert _extract_percentile("p95_latency_ms") == 95.0
        assert _extract_percentile("p99_latency_ms") == 99.0
        assert _extract_percentile("avg_latency_ms") == 50.0
        assert _extract_percentile("mean_response_time") == 50.0
        assert _extract_percentile("unknown_stat") is None

    def test_extract_percentile_case_insensitive(self):
        """Test that percentile extraction is case insensitive."""
        # ARRANGE & ACT & ASSERT
        assert _extract_percentile("P95_LATENCY") == 95.0
        assert _extract_percentile("Avg_Response_Time") == 50.0

    @patch('src.evaluation.visualization.metrics_dataframe.datetime')
    def test_calculate_days_missing_returns_realistic_values(self, mock_datetime):
        """Test that days missing calculation returns realistic values."""
        # ARRANGE
        # Mock current date as Jan 15, 2024
        mock_datetime.now.return_value = datetime(2024, 1, 15)
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        
        # ACT
        days_missing = _calculate_days_missing("speaker_attribution")
        
        # ASSERT
        assert isinstance(days_missing, int)
        assert days_missing >= 0
        assert days_missing <= 365 * 2  # Reasonable upper bound


class TestCombineEvaluationDataframes:
    """Test cases for combine_evaluation_dataframes function."""

    @patch('src.evaluation.visualization.metrics_dataframe.get_logger')
    def test_combine_evaluation_dataframes_with_all_dataframes(self, mock_get_logger):
        """Test combining all evaluation DataFrames."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        evaluation_df = pd.DataFrame([{"metric_name": "faithfulness", "value": 0.85}])
        query_df = pd.DataFrame([{"query_id": "q1", "latency_ms": 1200.0}])
        missing_features_df = pd.DataFrame([{"feature_name": "speaker_attribution", "priority": "high"}])
        
        # ACT
        result = combine_evaluation_dataframes(evaluation_df, query_df, missing_features_df)
        
        # ASSERT
        assert isinstance(result, dict)
        assert "evaluation_metrics" in result
        assert "query_performance" in result
        assert "missing_features" in result
        assert "summary" in result
        
        # Verify DataFrames are preserved
        assert result["evaluation_metrics"].equals(evaluation_df)
        assert result["query_performance"].equals(query_df)
        assert result["missing_features"].equals(missing_features_df)
        
        # Verify summary DataFrame exists
        summary_df = result["summary"]
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) > 0

    def test_combine_evaluation_dataframes_with_empty_dataframes(self):
        """Test combining empty DataFrames."""
        # ARRANGE
        empty_df = pd.DataFrame()
        
        # ACT
        result = combine_evaluation_dataframes(empty_df, empty_df, empty_df)
        
        # ASSERT
        assert isinstance(result, dict)
        assert "summary" in result
        
        # Summary should handle empty DataFrames gracefully
        summary_df = result["summary"]
        assert isinstance(summary_df, pd.DataFrame)

    def test_combine_evaluation_dataframes_creates_proper_summary(self):
        """Test that combined DataFrames create proper summary statistics."""
        # ARRANGE
        evaluation_df = pd.DataFrame([
            {"metric_name": "faithfulness", "value": 0.85},
            {"metric_name": "context_relevance", "value": 0.92}
        ])
        query_df = pd.DataFrame([
            {"query_id": "q1", "latency_ms": 1200.0},
            {"query_id": "q2", "latency_ms": 800.0}
        ])
        missing_features_df = pd.DataFrame([
            {"feature_name": "speaker_attribution", "priority": "high"},
            {"feature_name": "hybrid_search", "priority": "medium"}
        ])
        
        # ACT
        result = combine_evaluation_dataframes(evaluation_df, query_df, missing_features_df)
        
        # ASSERT
        summary_df = result["summary"]
        
        # Verify summary contains expected categories
        categories = set(summary_df["category"])
        expected_categories = {"Overall Performance", "Query Performance", "Missing Features"}
        assert categories == expected_categories
        
        # Verify summary calculations
        overall_row = summary_df[summary_df["category"] == "Overall Performance"].iloc[0]
        assert overall_row["value"] == 0.885  # (0.85 + 0.92) / 2
        
        query_row = summary_df[summary_df["category"] == "Query Performance"].iloc[0]
        assert query_row["value"] == 1000.0  # (1200 + 800) / 2
        
        missing_row = summary_df[summary_df["category"] == "Missing Features"].iloc[0]
        assert missing_row["value"] == 1  # Count of high priority features