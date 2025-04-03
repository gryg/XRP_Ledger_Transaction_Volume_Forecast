"""Utility functions for XRP forecasting project."""

import os
import logging
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies(include_deepar: bool = True, include_prophet: bool = True) -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Args:
        include_deepar: Whether to check DeepAR dependencies
        include_prophet: Whether to check Prophet dependencies
        
    Returns:
        Dictionary with dependency availability status
    """
    dependencies = {
        'basic': False,
        'deepar': False,
        'prophet': False
    }
    
    # Check basic dependencies
    try:
        import pandas
        import numpy
        import matplotlib
        import sklearn
        import xgboost
        import tensorflow
        dependencies['basic'] = True
        logger.info("Basic dependencies are installed.")
    except ImportError as e:
        logger.error(f"Missing basic dependency: {e}")
    
    # Check DeepAR dependencies if requested
    if include_deepar:
        try:
            import torch
            import gluonts
            dependencies['deepar'] = True
            logger.info("DeepAR dependencies are installed.")
        except ImportError as e:
            logger.warning(f"DeepAR dependencies not installed: {e}")
            logger.warning("Install with: pip install gluonts torch")
    
    # Check Prophet dependencies if requested
    if include_prophet:
        try:
            import prophet
            dependencies['prophet'] = True
            logger.info("Prophet dependencies are installed.")
        except ImportError as e:
            logger.warning(f"Prophet dependencies not installed: {e}")
            logger.warning("Install with: pip install prophet")
    
    return dependencies


def create_model_comparison(models_metrics: Dict[str, Dict[str, float]], output_dir: str) -> pd.DataFrame:
    """
    Create and save model comparison dataframe.
    
    Args:
        models_metrics: Dictionary with model metrics
        output_dir: Directory to save comparison
        
    Returns:
        DataFrame with model comparison
    """
    comparison_df = pd.DataFrame(models_metrics).T
    
    # Save to CSV
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Create comparison plot
    try:
        plt.figure(figsize=(10, 6))
        comparison_df['RMSE'].plot(kind='bar')
        plt.title('Model Comparison - RMSE')
        plt.ylabel('RMSE (Root Mean Squared Error)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison_rmse.png'))
        plt.close()
        logger.info(f"Model comparison plot saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
    
    return comparison_df


def plot_forecasts(forecasts: Dict[str, np.ndarray], actual: pd.Series, 
                   output_path: str, title: str = 'XRP Transaction Volume Forecast') -> None:
    """
    Plot actual values vs forecasts from multiple models.
    
    Args:
        forecasts: Dictionary mapping model names to forecast arrays
        actual: Series with actual values (with DatetimeIndex)
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(actual.index, actual.values, 'k-', label='Actual')
    
    # Plot forecasts
    colors = ['b', 'r', 'g', 'm', 'c']
    for i, (model, forecast) in enumerate(forecasts.items()):
        try:
            # Handle case where forecast length doesn't match actual length
            # Use the minimum length of both for plotting
            min_len = min(len(actual), len(forecast))
            if min_len == 0:
                logger.warning(f"Cannot plot {model} forecast - no data points")
                continue
                
            # Get corresponding indices
            plot_indices = actual.index[:min_len]
            plot_forecast = forecast[:min_len]
            
            plt.plot(plot_indices, plot_forecast, f'{colors[i % len(colors)]}-', label=f'{model} Forecast')
        except Exception as e:
            logger.error(f"Error plotting {model} forecast: {e}")
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Transaction Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Forecast comparison plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot: {e}")


def generate_report(output_dir: str, report_path: str, 
                    include_deepar: bool = True, include_prophet: bool = True) -> None:
    """
    Generate an HTML report of pipeline results.
    
    Args:
        output_dir: Root directory with all outputs
        report_path: Path to save the HTML report
        include_deepar: Whether to include DeepAR results
        include_prophet: Whether to include Prophet results
    """
    processed_dir = os.path.join(output_dir, 'processed_data')
    forecast_dir = os.path.join(output_dir, 'forecasts')
    deepar_dir = os.path.join(output_dir, 'deepar') if include_deepar else None
    prophet_dir = os.path.join(output_dir, 'prophet') if include_prophet else None
    
    # Get current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # HTML Header
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XRP Transaction Volume Forecasting Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333366; }}
            .section {{ margin-top: 30px; }}
            .chart {{ margin: 20px 0; max-width: 800px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>XRP Transaction Volume Forecasting Report</h1>
        <p>Generated on: {current_time}</p>
    """
    
    # Data Overview Section
    data_section = """        
        <div class="section">
            <h2>Data Overview</h2>
    """
    
    # Check if these visualizations exist
    if os.path.exists(os.path.join(processed_dir, 'daily_transactions.png')):
        data_section += """
            <div class="chart">
                <h3>Daily Transaction Volume</h3>
                <img src="processed_data/daily_transactions.png" alt="Daily Transactions" style="max-width: 100%;">
            </div>
        """
    
    if os.path.exists(os.path.join(processed_dir, 'tx_by_day_of_week.png')):
        data_section += """
            <div class="chart">
                <h3>Transaction Volume by Day of Week</h3>
                <img src="processed_data/tx_by_day_of_week.png" alt="Transactions by Day of Week" style="max-width: 100%;">
            </div>
        """
    
    if os.path.exists(os.path.join(processed_dir, 'tx_by_hour.png')):
        data_section += """
            <div class="chart">
                <h3>Transaction Volume by Hour of Day</h3>
                <img src="processed_data/tx_by_hour.png" alt="Transactions by Hour" style="max-width: 100%;">
            </div>
        """
    
    data_section += """
        </div>
    """
    
    html_content += data_section
    
    # Model Evaluation Section
    eval_section = """
        <div class="section">
            <h2>Model Evaluation</h2>
    """
    
    # Hourly and Daily Forecasts
    for freq in ['hourly', 'daily']:
        if os.path.exists(os.path.join(forecast_dir, freq, 'forecast_comparison.png')):
            eval_section += f"""
            <h3>{freq.capitalize()} Forecasting</h3>
            <div class="chart">
                <img src="forecasts/{freq}/forecast_comparison.png" alt="{freq.capitalize()} Forecast Comparison" style="max-width: 100%;">
            </div>
            """
    
    # DeepAR Section
    if include_deepar and deepar_dir:
        for freq in ['hourly', 'daily']:
            deepar_img = f"deepar/{freq}/results/deepar_forecast.png"
            if os.path.exists(os.path.join(output_dir, deepar_img)):
                eval_section += f"""
                <h3>{freq.capitalize()} DeepAR Forecast</h3>
                <div class="chart">
                    <img src="{deepar_img}" alt="{freq.capitalize()} DeepAR Forecast" style="max-width: 100%;">
                </div>
                """
    
    # Prophet Section
    if include_prophet and prophet_dir:
        prophet_results = os.path.join(prophet_dir, 'results')
        if os.path.exists(os.path.join(prophet_results, 'actual_vs_predicted.png')):
            eval_section += f"""
            <h3>Prophet Forecasting</h3>
            <div class="chart">
                <img src="prophet/results/actual_vs_predicted.png" alt="Prophet Forecast" style="max-width: 100%;">
            </div>
            """
        
        if os.path.exists(os.path.join(prophet_results, 'forecast_components.png')):
            eval_section += f"""
            <div class="chart">
                <h4>Prophet Forecast Components</h4>
                <img src="prophet/results/forecast_components.png" alt="Prophet Components" style="max-width: 100%;">
            </div>
            """
    
    eval_section += """
        </div>
    """
    
    html_content += eval_section
    
    # Future Forecasts Section
    future_section = """
        <div class="section">
            <h2>Future Forecasts</h2>
    """
    
    # Hourly and Daily Future Forecasts
    for freq in ['hourly', 'daily']:
        future_img = f"forecasts/{freq}/future_forecast.png"
        if os.path.exists(os.path.join(output_dir, future_img)):
            future_section += f"""
            <h3>{freq.capitalize()} Future Forecast</h3>
            <div class="chart">
                <img src="{future_img}" alt="{freq.capitalize()} Future Forecast" style="max-width: 100%;">
            </div>
            """
    
    # DeepAR Future
    if include_deepar and deepar_dir:
        for freq in ['hourly', 'daily']:
            deepar_future = f"deepar/{freq}/results/future_forecast.png"
            if os.path.exists(os.path.join(output_dir, deepar_future)):
                future_section += f"""
                <h3>{freq.capitalize()} DeepAR Future Forecast</h3>
                <div class="chart">
                    <img src="{deepar_future}" alt="{freq.capitalize()} DeepAR Future" style="max-width: 100%;">
                </div>
                """
    
    # Prophet Future
    if include_prophet and prophet_dir:
        prophet_future = "prophet/results/future_forecast.png"
        if os.path.exists(os.path.join(output_dir, prophet_future)):
            future_section += f"""
            <h3>Prophet Future Forecast</h3>
            <div class="chart">
                <img src="{prophet_future}" alt="Prophet Future Forecast" style="max-width: 100%;">
            </div>
            """
    
    future_section += """
        </div>
    """
    
    html_content += future_section
    
    # Model Comparison Section
    comparison_img = "processed_data/model_comparison_rmse.png"
    if os.path.exists(os.path.join(output_dir, comparison_img)):
        html_content += f"""
        <div class="section">
            <h2>Model Comparison</h2>
            <div class="chart">
                <img src="{comparison_img}" alt="Model Comparison" style="max-width: 100%;">
            </div>
        </div>
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {report_path}")