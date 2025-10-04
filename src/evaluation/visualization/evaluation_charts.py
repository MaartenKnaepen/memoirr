"""Create visualization charts for RAG evaluation results.

This module provides functions to generate charts and dashboards from
evaluation DataFrames for analysis and reporting.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
from datetime import datetime
import os

from src.core.logging_config import get_logger


def create_metrics_overview_chart(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Create overview chart of all evaluation metrics.
    
    Args:
        df: DataFrame from build_evaluation_dataframe()
        
    Returns:
        Matplotlib figure with metrics overview
        
    Example:
        df = build_evaluation_dataframe(results)
        fig = create_metrics_overview_chart(df)
        fig.savefig("metrics_overview.png")
    """
    logger = get_logger(__name__)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("RAG Evaluation Metrics Overview", fontsize=16)
        
        if df.empty:
            fig.text(0.5, 0.5, "No evaluation data available", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Metrics by type
        eval_types = df.groupby('evaluation_type')['value'].mean()
        if not eval_types.empty:
            axes[0, 0].bar(eval_types.index, eval_types.values)
            axes[0, 0].set_title("Average Score by Evaluation Type")
            axes[0, 0].set_ylabel("Score")
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Individual metrics
        if len(df) > 0:
            axes[0, 1].bar(df['metric_name'], df['value'])
            axes[0, 1].set_title("Individual Metric Scores")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Status distribution
        status_counts = df['status'].value_counts()
        if not status_counts.empty:
            axes[1, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title("Evaluation Status Distribution")
        
        # Performance vs Quality scatter (if latency data available)
        performance_metrics = df[df['evaluation_type'] == 'performance']
        quality_metrics = df[df['evaluation_type'].isin(['faithfulness', 'context_relevance', 'exact_match'])]
        
        if not performance_metrics.empty and not quality_metrics.empty:
            avg_latency = performance_metrics['value'].mean()
            avg_quality = quality_metrics['value'].mean()
            axes[1, 1].scatter([avg_latency], [avg_quality], s=100, alpha=0.7)
            axes[1, 1].set_xlabel("Average Latency (ms)")
            axes[1, 1].set_ylabel("Average Quality Score")
            axes[1, 1].set_title("Performance vs Quality")
        else:
            axes[1, 1].text(0.5, 0.5, "Insufficient data for\nperformance vs quality",
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        logger.info(
            "Created metrics overview chart",
            num_metrics=len(df),
            component="evaluation_charts"
        )
        
        return fig
        
    except Exception as e:
        logger.error(
            "Failed to create metrics overview chart",
            error=str(e),
            error_type=type(e).__name__,
            component="evaluation_charts"
        )
        raise


def create_latency_distribution_chart(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Create latency analysis chart.
    
    Args:
        df: DataFrame from build_latency_dataframe()
        
    Returns:
        Matplotlib figure with latency distribution analysis
    """
    logger = get_logger(__name__)
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("RAG System Latency Analysis", fontsize=16)
        
        if df.empty:
            fig.text(0.5, 0.5, "No latency data available",
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Latency distribution histogram
        latency_data = df[df['measurement_type'] == 'individual']['latency_ms']
        if not latency_data.empty:
            axes[0].hist(latency_data, bins=20, alpha=0.7, edgecolor='black')
            axes[0].set_xlabel("Latency (ms)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Latency Distribution")
            axes[0].axvline(latency_data.mean(), color='red', linestyle='--', 
                           label=f'Mean: {latency_data.mean():.1f}ms')
            axes[0].legend()
        
        # Percentile comparison
        percentile_data = df[df['measurement_type'] == 'summary']
        if not percentile_data.empty:
            percentiles = percentile_data['percentile'].dropna()
            latencies = percentile_data.loc[percentile_data['percentile'].notna(), 'latency_ms']
            
            if len(percentiles) > 0:
                axes[1].bar([f"P{int(p)}" for p in percentiles], latencies)
                axes[1].set_ylabel("Latency (ms)")
                axes[1].set_title("Latency Percentiles")
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(
            "Failed to create latency distribution chart",
            error=str(e),
            component="evaluation_charts"
        )
        raise


def create_missing_features_timeline(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Create timeline chart for missing features.
    
    Args:
        df: DataFrame from build_missing_features_dataframe()
        
    Returns:
        Matplotlib figure with missing features timeline
    """
    logger = get_logger(__name__)
    
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Missing Features Roadmap", fontsize=16)
        
        if df.empty:
            fig.text(0.5, 0.5, "No missing features data available",
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Features by priority
        priority_counts = df['priority'].value_counts()
        colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        priority_colors = [colors.get(p, 'gray') for p in priority_counts.index]
        
        axes[0].bar(priority_counts.index, priority_counts.values, color=priority_colors)
        axes[0].set_title("Missing Features by Priority")
        axes[0].set_ylabel("Number of Features")
        
        # Days missing vs effort
        if 'days_missing' in df.columns and 'estimated_effort' in df.columns:
            effort_mapping = {'low': 1, 'medium': 2, 'high': 3}
            effort_numeric = df['estimated_effort'].map(effort_mapping)
            
            scatter = axes[1].scatter(df['days_missing'], effort_numeric, 
                                    c=df['priority'].map({'high': 'red', 'medium': 'orange', 'low': 'green'}),
                                    s=100, alpha=0.7)
            
            # Add feature labels
            for i, row in df.iterrows():
                axes[1].annotate(row['feature_name'], 
                               (row['days_missing'], effort_mapping.get(row['estimated_effort'], 2)),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1].set_xlabel("Days Missing")
            axes[1].set_ylabel("Estimated Effort")
            axes[1].set_yticks([1, 2, 3])
            axes[1].set_yticklabels(['Low', 'Medium', 'High'])
            axes[1].set_title("Missing Duration vs Implementation Effort")
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(
            "Failed to create missing features timeline",
            error=str(e),
            component="evaluation_charts"
        )
        raise


def create_query_type_performance_chart(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Create performance breakdown by query type.
    
    Args:
        df: DataFrame from build_query_performance_dataframe()
        
    Returns:
        Matplotlib figure with query type performance analysis
    """
    logger = get_logger(__name__)
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Performance by Query Type", fontsize=16)
        
        if df.empty:
            fig.text(0.5, 0.5, "No query performance data available",
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Average scores by query type
        score_columns = ['faithfulness_score', 'context_relevance_score', 'exact_match_score']
        query_types = df['query_type'].unique()
        
        for i, score_col in enumerate(score_columns):
            if score_col in df.columns:
                avg_scores = df.groupby('query_type')[score_col].mean().dropna()
                if not avg_scores.empty:
                    axes[i].bar(avg_scores.index, avg_scores.values)
                    axes[i].set_title(score_col.replace('_', ' ').title())
                    axes[i].set_ylabel("Average Score")
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(
            "Failed to create query type performance chart",
            error=str(e),
            component="evaluation_charts"
        )
        raise


def export_evaluation_dashboard(
    evaluation_results: Dict[str, Any],
    output_dir: str = "evaluation_output"
) -> str:
    """Export comprehensive evaluation dashboard.
    
    Args:
        evaluation_results: Complete evaluation results from HaystackRAGEvaluator
        output_dir: Directory to save dashboard files
        
    Returns:
        Path to the main dashboard HTML file
        
    Example:
        results = evaluator.run_baseline_evaluation()
        dashboard_path = export_evaluation_dashboard(results)
        print(f"Dashboard available at: {dashboard_path}")
    """
    logger = get_logger(__name__)
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Import DataFrame builders
        from src.evaluation.visualization.metrics_dataframe import (
            build_evaluation_dataframe,
            build_missing_features_dataframe
        )
        
        # Build DataFrames
        metrics_df = build_evaluation_dataframe(evaluation_results)
        
        # Create missing features data (placeholder)
        missing_features = {
            "speaker_attribution": "FAIL - No speaker information in responses",
            "hybrid_search": "FAIL - Only semantic search, no BM25 integration", 
            "conversation_threading": "FAIL - No dialogue context awareness"
        }
        missing_df = build_missing_features_dataframe(missing_features)
        
        # Generate charts
        overview_fig = create_metrics_overview_chart(metrics_df)
        missing_features_fig = create_missing_features_timeline(missing_df)
        
        # Save charts
        overview_path = os.path.join(output_dir, "metrics_overview.png")
        missing_path = os.path.join(output_dir, "missing_features.png")
        
        overview_fig.savefig(overview_path, dpi=300, bbox_inches='tight')
        missing_features_fig.savefig(missing_path, dpi=300, bbox_inches='tight')
        
        # Create HTML dashboard
        dashboard_html = _create_dashboard_html(
            evaluation_results, 
            metrics_df,
            missing_df,
            overview_path,
            missing_path
        )
        
        dashboard_path = os.path.join(output_dir, "evaluation_dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        # Save DataFrames as CSV
        metrics_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
        missing_df.to_csv(os.path.join(output_dir, "missing_features.csv"), index=False)
        
        logger.info(
            "Evaluation dashboard exported",
            dashboard_path=dashboard_path,
            output_dir=output_dir,
            component="evaluation_charts"
        )
        
        return dashboard_path
        
    except Exception as e:
        logger.error(
            "Failed to export evaluation dashboard",
            error=str(e),
            component="evaluation_charts"
        )
        raise


def _create_dashboard_html(
    results: Dict[str, Any],
    metrics_df: pd.DataFrame, 
    missing_df: pd.DataFrame,
    overview_chart_path: str,
    missing_chart_path: str
) -> str:
    """Create HTML dashboard content."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Evaluation Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e7f3ff; border-radius: 3px; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .fail {{ color: red; font-weight: bold; }}
            .success {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RAG System Evaluation Dashboard</h1>
            <p>Generated: {timestamp}</p>
            <p>Evaluation using Haystack native evaluators on LOTR dataset</p>
        </div>
        
        <div class="section">
            <h2>Baseline Performance Metrics</h2>
            <div class="metric">
                <strong>Faithfulness:</strong> {results.get('faithfulness', 'N/A')}
            </div>
            <div class="metric">
                <strong>Context Relevance:</strong> {results.get('context_relevance', 'N/A')}
            </div>
            <div class="metric">
                <strong>Exact Match:</strong> {results.get('exact_match', 'N/A')}
            </div>
            <div class="metric">
                <strong>Avg Latency:</strong> {results.get('avg_latency_ms', 'N/A')} ms
            </div>
        </div>
        
        <div class="section">
            <h2>Evaluation Overview</h2>
            <div class="chart">
                <img src="{os.path.basename(overview_chart_path)}" alt="Metrics Overview" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="section">
            <h2>Missing Features</h2>
            <div class="chart">
                <img src="{os.path.basename(missing_chart_path)}" alt="Missing Features Timeline" style="max-width: 100%;">
            </div>
            
            <h3>Feature Status</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Status</th>
                    <th>Priority</th>
                    <th>Estimated Effort</th>
                    <th>Expected Completion</th>
                </tr>
    """
    
    # Add missing features table rows
    for _, row in missing_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['feature_name']}</td>
                    <td class="fail">{row['status']}</td>
                    <td>{row['priority']}</td>
                    <td>{row['estimated_effort']}</td>
                    <td>{row['expected_completion']}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Next Steps</h2>
            <ul>
                <li>Implement high-priority missing features (speaker attribution, hybrid search)</li>
                <li>Run comparative evaluation after each feature implementation</li>
                <li>Monitor performance regression with new features</li>
                <li>Expand test dataset for more comprehensive evaluation</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html_content