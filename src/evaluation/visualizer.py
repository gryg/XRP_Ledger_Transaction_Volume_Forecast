"""Visualization utilities for XRP forecasting."""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Visualizer:
    """Visualizer for time series data and forecasts."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_series(self, data: pd.DataFrame, column: str = 'tx_count', 
                        title: str = 'XRP Transaction Volume', filename: str = 'time_series.png'):
        """
        Plot a time series.
        
        Args:
            data: DataFrame with time series data
            column: Column to plot
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[column])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Time series plot saved to {output_path}")
    
    def plot_forecast_comparison(self, forecasts: Dict[str, np.ndarray], actual: pd.Series,
                               title: str = 'Model Comparison', filename: str = 'forecast_comparison.png'):
        """
        Plot actual values vs forecasts from multiple models.
        
        Args:
            forecasts: Dictionary mapping model names to forecast arrays
            actual: Series with actual values (with DatetimeIndex)
            title: Plot title
            filename: Output filename
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
                
                plt.plot(plot_indices, plot_forecast, f'{colors[i % len(colors)]}-', label=f'{model}')
            except Exception as e:
                logger.error(f"Error plotting {model} forecast: {e}")
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Forecast comparison plot saved to {output_path}")
    
    def plot_future_forecast(self, forecast: pd.DataFrame, historical_data: pd.DataFrame,
                           title: str = 'Future Forecast', filename: str = 'future_forecast.png'):
        """
        Plot future forecast along with historical data.
        
        Args:
            forecast: DataFrame with future forecasts (with DatetimeIndex)
            historical_data: DataFrame with historical data (with DatetimeIndex)
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 7 days or 30 days)
        if 'hour' in historical_data.columns:
            # Hourly data - show last 7 days
            history_size = min(7 * 24, len(historical_data))
        else:
            # Daily data - show last 30 days
            history_size = min(30, len(historical_data))
            
        historical_values = historical_data.iloc[-history_size:]['tx_count'].values
        historical_dates = historical_data.index[-history_size:]
        
        plt.plot(historical_dates, historical_values, 'k-', label='Historical')
        
        # Plot forecasts
        colors = ['b', 'r', 'g', 'm', 'c']
        for i, column in enumerate(forecast.columns):
            plt.plot(forecast.index, forecast[column].values, 
                    f'{colors[i % len(colors)]}-', label=f'{column}')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Future forecast plot saved to {output_path}")
    
    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]], 
                            metric: str = 'RMSE', title: str = 'Model Comparison',
                            filename: str = 'model_comparison.png'):
        """
        Plot model comparison based on evaluation metrics.
        
        Args:
            metrics: Dictionary mapping model names to metric dictionaries
            metric: Metric to compare ('RMSE', 'MAE', 'MAPE', 'R2')
            title: Plot title
            filename: Output filename
        """
        # Create DataFrame from metrics
        df = pd.DataFrame(metrics).T
        
        plt.figure(figsize=(10, 6))
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # Create bar plot
        ax = sns.barplot(x=df.index, y=df[metric])
        
        # Add value labels on top of bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.title(f"{title} - {metric}")
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Model comparison plot saved to {output_path}")
    
    def plot_seasonal_decomposition(self, decomposition, title: str = 'Seasonal Decomposition',
                                  filename: str = 'seasonal_decomposition.png'):
        """
        Plot seasonal decomposition of time series.
        
        Args:
            decomposition: Seasonal decomposition result from statsmodels
            title: Plot title
            filename: Output filename
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        ax1.set_xlabel('')
        
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        ax2.set_xlabel('')
        
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonality')
        ax3.set_xlabel('')
        
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residuals')
        
        plt.suptitle(title, y=0.92, fontsize=15)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Seasonal decomposition plot saved to {output_path}")
    
    def create_feature_importance_plot(self, feature_importance: pd.DataFrame, 
                                     top_n: int = 10, title: str = 'Feature Importance',
                                     filename: str = 'feature_importance.png'):
        """
        Create feature importance plot for tree-based models.
        
        Args:
            feature_importance: DataFrame with features and importance values
            top_n: Number of top features to show
            title: Plot title
            filename: Output filename
        """
        # Sort features by importance
        top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar plot
        sns.barplot(x='importance', y='feature', data=top_features)
        
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {output_path}")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = 'Training History',
                            filename: str = 'training_history.png'):
        """
        Plot training history for neural network models.
        
        Args:
            history: Dictionary with training metrics (e.g., from Keras history)
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        
        for metric, values in history.items():
            plt.plot(values, label=metric)
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Training history plot saved to {output_path}")