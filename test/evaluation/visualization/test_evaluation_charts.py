"""Tests for evaluation charts and visualization following Memoirr standards.

Tests the chart generation components using AAA pattern, comprehensive
mocking, and error handling validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import os
from datetime import datetime

from src.evaluation.visualization.evaluation_charts import (
    create_metrics_overview_chart,
    create_latency_distribution_chart,
    create_missing_features_timeline,
    create_query_type_performance_chart,
    export_evaluation_dashboard,
    _create_dashboard_html
)


class TestCreateMetricsOverviewChart:
    """Test cases for create_metrics_overview_chart function."""

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    def test_create_metrics_overview_chart_with_valid_data(self, mock_get_logger):
        """Test metrics overview chart creation with valid DataFrame."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        sample_df = pd.DataFrame([
            {"metric_name": "faithfulness", "value": 0.85, "evaluation_type": "faithfulness", "status": "completed"},
            {"metric_name": "context_relevance", "value": 0.92, "evaluation_type": "context_relevance", "status": "completed"},
            {"metric_name": "exact_match", "value": 0.5, "evaluation_type": "exact_match", "status": "completed"}
        ])
        
        # ACT
        result = create_metrics_overview_chart(sample_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        
        # Verify chart has subplots
        axes = result.get_axes()
        assert len(axes) == 4  # 2x2 subplot grid
        
        # Verify logging
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[1]
        assert log_call_args["num_metrics"] == len(sample_df)
        assert log_call_args["component"] == "evaluation_charts"
        
        # Clean up
        plt.close(result)

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    def test_create_metrics_overview_chart_with_empty_dataframe(self, mock_get_logger):
        """Test metrics overview chart handles empty DataFrame."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        empty_df = pd.DataFrame()
        
        # ACT
        result = create_metrics_overview_chart(empty_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        # Should create chart with "No data available" message
        plt.close(result)

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    def test_create_metrics_overview_chart_handles_exceptions(self, mock_get_logger):
        """Test that chart creation handles and logs exceptions."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        sample_df = pd.DataFrame([{"metric_name": "test", "value": 0.5, "evaluation_type": "test", "status": "completed"}])
        
        # Force an exception during plotting
        with patch('matplotlib.pyplot.subplots', side_effect=RuntimeError("Plotting failed")):
            # ACT & ASSERT
            with pytest.raises(RuntimeError, match="Plotting failed"):
                create_metrics_overview_chart(sample_df)
            
            # Verify error logging
            mock_logger.error.assert_called()
            error_call_args = mock_logger.error.call_args[1]
            assert error_call_args["error"] == "Plotting failed"
            assert error_call_args["error_type"] == "RuntimeError"

    def test_create_metrics_overview_chart_figure_properties(self, sample_metrics_dataframe):
        """Test that created chart has proper figure properties."""
        # ARRANGE & ACT
        result = create_metrics_overview_chart(sample_metrics_dataframe)
        
        # ASSERT
        # Verify figure size
        assert result.get_size_inches()[0] == 12  # Width
        assert result.get_size_inches()[1] == 8   # Height
        
        # Verify title
        assert "RAG Evaluation Metrics Overview" in result._suptitle.get_text()
        
        # Clean up
        plt.close(result)


class TestCreateLatencyDistributionChart:
    """Test cases for create_latency_distribution_chart function."""

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    def test_create_latency_distribution_chart_with_valid_data(self, mock_get_logger):
        """Test latency distribution chart creation with valid data."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        latency_df = pd.DataFrame([
            {"measurement_id": "1_0", "latency_ms": 1200.0, "measurement_type": "individual", "percentile": None},
            {"measurement_id": "1_1", "latency_ms": 1350.0, "measurement_type": "individual", "percentile": None},
            {"measurement_id": "summary_0", "latency_ms": 1275.0, "measurement_type": "summary", "percentile": 50.0},
            {"measurement_id": "summary_1", "latency_ms": 1400.0, "measurement_type": "summary", "percentile": 95.0}
        ])
        
        # ACT
        result = create_latency_distribution_chart(latency_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        
        # Verify chart has subplots
        axes = result.get_axes()
        assert len(axes) == 2  # 1x2 subplot grid
        
        # Clean up
        plt.close(result)

    def test_create_latency_distribution_chart_with_empty_dataframe(self):
        """Test latency chart handles empty DataFrame."""
        # ARRANGE
        empty_df = pd.DataFrame()
        
        # ACT
        result = create_latency_distribution_chart(empty_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close(result)

    def test_create_latency_distribution_chart_figure_properties(self):
        """Test that latency chart has proper figure properties."""
        # ARRANGE
        latency_df = pd.DataFrame([
            {"latency_ms": 1200.0, "measurement_type": "individual", "percentile": None}
        ])
        
        # ACT
        result = create_latency_distribution_chart(latency_df)
        
        # ASSERT
        # Verify figure size
        assert result.get_size_inches()[0] == 12  # Width
        assert result.get_size_inches()[1] == 5   # Height
        
        # Verify title
        assert "RAG System Latency Analysis" in result._suptitle.get_text()
        
        plt.close(result)


class TestCreateMissingFeaturesTimeline:
    """Test cases for create_missing_features_timeline function."""

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    def test_create_missing_features_timeline_with_valid_data(self, mock_get_logger):
        """Test missing features timeline chart creation."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        missing_features_df = pd.DataFrame([
            {"feature_name": "speaker_attribution", "priority": "high", "estimated_effort": "medium", "days_missing": 30},
            {"feature_name": "hybrid_search", "priority": "high", "estimated_effort": "high", "days_missing": 25},
            {"feature_name": "content_type_detection", "priority": "low", "estimated_effort": "low", "days_missing": 20}
        ])
        
        # ACT
        result = create_missing_features_timeline(missing_features_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        
        # Verify chart has subplots
        axes = result.get_axes()
        assert len(axes) == 2  # 2x1 subplot grid
        
        plt.close(result)

    def test_create_missing_features_timeline_with_empty_dataframe(self):
        """Test missing features timeline handles empty DataFrame."""
        # ARRANGE
        empty_df = pd.DataFrame()
        
        # ACT
        result = create_missing_features_timeline(empty_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close(result)

    def test_create_missing_features_timeline_color_mapping(self):
        """Test that missing features timeline uses proper color mapping."""
        # ARRANGE
        missing_features_df = pd.DataFrame([
            {"feature_name": "test_feature", "priority": "high", "estimated_effort": "low", "days_missing": 10}
        ])
        
        # ACT
        result = create_missing_features_timeline(missing_features_df)
        
        # ASSERT
        # Chart should be created successfully with color mapping
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close(result)


class TestCreateQueryTypePerformanceChart:
    """Test cases for create_query_type_performance_chart function."""

    def test_create_query_type_performance_chart_with_valid_data(self):
        """Test query type performance chart creation."""
        # ARRANGE
        query_performance_df = pd.DataFrame([
            {"query_type": "faithfulness", "faithfulness_score": 0.85, "context_relevance_score": 0.9, "exact_match_score": None},
            {"query_type": "exact_match", "faithfulness_score": None, "context_relevance_score": 0.8, "exact_match_score": 1.0},
            {"query_type": "context_relevance", "faithfulness_score": 0.7, "context_relevance_score": 0.95, "exact_match_score": None}
        ])
        
        # ACT
        result = create_query_type_performance_chart(query_performance_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        
        # Verify chart has subplots
        axes = result.get_axes()
        assert len(axes) == 3  # 1x3 subplot grid
        
        plt.close(result)

    def test_create_query_type_performance_chart_with_empty_dataframe(self):
        """Test query performance chart handles empty DataFrame."""
        # ARRANGE
        empty_df = pd.DataFrame()
        
        # ACT
        result = create_query_type_performance_chart(empty_df)
        
        # ASSERT
        assert isinstance(result, matplotlib.figure.Figure)
        plt.close(result)


class TestExportEvaluationDashboard:
    """Test cases for export_evaluation_dashboard function."""

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    @patch('src.evaluation.visualization.evaluation_charts.os.makedirs')
    def test_export_evaluation_dashboard_creates_output_directory(self, mock_makedirs, mock_get_logger):
        """Test that dashboard export creates output directory."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        evaluation_results = {"faithfulness": 0.85, "context_relevance": 0.92}
        output_dir = "test_output"
        
        with patch('src.evaluation.visualization.metrics_dataframe.build_evaluation_dataframe') as mock_build_df:
            with patch('src.evaluation.visualization.metrics_dataframe.build_missing_features_dataframe') as mock_build_missing:
                with patch('builtins.open', create=True) as mock_open:
                    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                        mock_build_df.return_value = pd.DataFrame()
                        mock_build_missing.return_value = pd.DataFrame()
                        
                        # ACT
                        result = export_evaluation_dashboard(evaluation_results, output_dir)
        
        # ASSERT
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        assert result.endswith("evaluation_dashboard.html")

    @patch('src.evaluation.visualization.evaluation_charts.get_logger')
    def test_export_evaluation_dashboard_handles_exceptions(self, mock_get_logger):
        """Test that dashboard export handles and logs exceptions."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Force an exception during export
        with patch('src.evaluation.visualization.evaluation_charts.os.makedirs', side_effect=OSError("Directory creation failed")):
            # ACT & ASSERT
            with pytest.raises(OSError, match="Directory creation failed"):
                export_evaluation_dashboard({"test": 0.5})
            
            # Verify error logging
            mock_logger.error.assert_called()

    def test_export_evaluation_dashboard_returns_html_path(self, temporary_output_dir, sample_baseline_results):
        """Test that dashboard export returns correct HTML path."""
        # ARRANGE & ACT
        with patch('src.evaluation.visualization.metrics_dataframe.build_evaluation_dataframe') as mock_build_df:
            with patch('src.evaluation.visualization.metrics_dataframe.build_missing_features_dataframe') as mock_build_missing:
                with patch('builtins.open', create=True):
                    mock_build_df.return_value = pd.DataFrame()
                    mock_build_missing.return_value = pd.DataFrame()
                    
                    result = export_evaluation_dashboard(sample_baseline_results, temporary_output_dir)
        
        # ASSERT
        expected_path = os.path.join(temporary_output_dir, "evaluation_dashboard.html")
        assert result == expected_path


class TestCreateDashboardHtml:
    """Test cases for _create_dashboard_html function."""

    def test_create_dashboard_html_with_valid_data(self, sample_baseline_results):
        """Test HTML dashboard creation with valid data."""
        # ARRANGE
        metrics_df = pd.DataFrame([{"metric_name": "faithfulness", "value": 0.85}])
        missing_df = pd.DataFrame([{"feature_name": "speaker_attribution", "status": "not_implemented", "priority": "high", "estimated_effort": "medium", "expected_completion": "2024-04-01"}])
        overview_chart_path = "/path/to/overview.png"
        missing_chart_path = "/path/to/missing.png"
        
        # ACT
        result = _create_dashboard_html(
            sample_baseline_results,
            metrics_df,
            missing_df,
            overview_chart_path,
            missing_chart_path
        )
        
        # ASSERT
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
        assert "RAG System Evaluation Dashboard" in result
        assert "Baseline Performance Metrics" in result
        assert "Missing Features" in result
        
        # Verify metrics are included
        assert str(sample_baseline_results["faithfulness"]) in result
        assert str(sample_baseline_results["context_relevance"]) in result

    def test_create_dashboard_html_includes_timestamp(self, sample_baseline_results):
        """Test that HTML dashboard includes generation timestamp."""
        # ARRANGE
        metrics_df = pd.DataFrame()
        missing_df = pd.DataFrame()
        
        # ACT
        result = _create_dashboard_html(sample_baseline_results, metrics_df, missing_df, "", "")
        
        # ASSERT
        # Should include a timestamp
        current_year = str(datetime.now().year)
        assert current_year in result
        assert "Generated:" in result

    def test_create_dashboard_html_includes_missing_features_table(self):
        """Test that HTML dashboard includes missing features table."""
        # ARRANGE
        results = {"faithfulness": 0.85}
        metrics_df = pd.DataFrame()
        missing_df = pd.DataFrame([
            {"feature_name": "speaker_attribution", "status": "not_implemented", "priority": "high", "estimated_effort": "medium", "expected_completion": "2024-04-01"}
        ])
        
        # ACT
        result = _create_dashboard_html(results, metrics_df, missing_df, "", "")
        
        # ASSERT
        assert "<table>" in result
        assert "speaker_attribution" in result
        assert "not_implemented" in result
        assert "high" in result

    def test_create_dashboard_html_includes_chart_references(self):
        """Test that HTML dashboard includes chart image references."""
        # ARRANGE
        results = {"faithfulness": 0.85}
        metrics_df = pd.DataFrame()
        missing_df = pd.DataFrame()
        overview_chart = "overview_chart.png"
        missing_chart = "missing_chart.png"
        
        # ACT
        result = _create_dashboard_html(results, metrics_df, missing_df, overview_chart, missing_chart)
        
        # ASSERT
        assert overview_chart in result
        assert missing_chart in result
        assert '<img src=' in result


class TestEvaluationChartsIntegration:
    """Integration tests for evaluation charts components."""

    def test_end_to_end_dashboard_creation_workflow(self, sample_baseline_results, temporary_output_dir):
        """Test complete dashboard creation workflow."""
        # ARRANGE & ACT
        with patch('src.evaluation.visualization.metrics_dataframe.build_evaluation_dataframe') as mock_build_eval:
            with patch('src.evaluation.visualization.metrics_dataframe.build_missing_features_dataframe') as mock_build_missing:
                # Mock DataFrames with required columns
                mock_build_eval.return_value = pd.DataFrame([
                    {"metric_name": "faithfulness", "value": 0.85, "evaluation_type": "faithfulness", "status": "completed"}
                ])
                mock_build_missing.return_value = pd.DataFrame([
                    {"feature_name": "speaker_attribution", "priority": "high", "status": "not_implemented", "estimated_effort": "medium", "expected_completion": "2024-04-01"}
                ])
                
                # Create dashboard
                dashboard_path = export_evaluation_dashboard(sample_baseline_results, temporary_output_dir)
        
        # ASSERT
        assert dashboard_path.endswith(".html")
        assert temporary_output_dir in dashboard_path

    def test_chart_creation_with_realistic_data(self):
        """Test that all chart types can be created with realistic data."""
        # ARRANGE
        metrics_df = pd.DataFrame([
            {"metric_name": "faithfulness", "value": 0.85, "evaluation_type": "faithfulness", "status": "completed"},
            {"metric_name": "context_relevance", "value": 0.92, "evaluation_type": "context_relevance", "status": "completed"}
        ])
        
        latency_df = pd.DataFrame([
            {"latency_ms": 1200.0, "measurement_type": "individual", "percentile": None},
            {"latency_ms": 1500.0, "measurement_type": "summary", "percentile": 95.0}
        ])
        
        missing_df = pd.DataFrame([
            {"feature_name": "speaker_attribution", "priority": "high", "estimated_effort": "medium", "days_missing": 30}
        ])
        
        query_df = pd.DataFrame([
            {"query_type": "faithfulness", "faithfulness_score": 0.85, "context_relevance_score": 0.9, "exact_match_score": None}
        ])
        
        # ACT & ASSERT - All charts should be created successfully
        overview_chart = create_metrics_overview_chart(metrics_df)
        assert isinstance(overview_chart, matplotlib.figure.Figure)
        plt.close(overview_chart)
        
        latency_chart = create_latency_distribution_chart(latency_df)
        assert isinstance(latency_chart, matplotlib.figure.Figure)
        plt.close(latency_chart)
        
        missing_chart = create_missing_features_timeline(missing_df)
        assert isinstance(missing_chart, matplotlib.figure.Figure)
        plt.close(missing_chart)
        
        query_chart = create_query_type_performance_chart(query_df)
        assert isinstance(query_chart, matplotlib.figure.Figure)
        plt.close(query_chart)