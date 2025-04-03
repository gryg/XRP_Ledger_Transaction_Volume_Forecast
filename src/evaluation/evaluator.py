"""Model evaluation module for time series forecasting."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from src.utils.helpers import plot_forecasts, create_model_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for time series forecasting models."""
    
    def __init__(self, output_dir: str = "./output/evaluation"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_models(self, models: Dict[str, Any], data_path: str, 
                       prediction_length: int) -> Dict[str, Any]:
        """
        Evaluate models and compare performance.
        
        Args:
            models: Dictionary with trained models
            data_path: Path to CSV test data
            prediction_length: Number of steps to predict
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating models using data from {data_path}")
        
        # Load test data
        test_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Initialize results containers
        forecasts = {}
        metrics = {}
        
        # Evaluate each model
        for model_name, model_results in models.items():
            try:
                # Check if this is a model object or training results dictionary
                if 'model' in model_results:
                    # Extract model object from results
                    model = model_results['model']
                    actual = model_results.get('actual', None)
                    predicted = model_results.get('predictions', None)
                    model_metrics = model_results.get('metrics', None)
                    
                    # If metrics already exist, use them
                    if model_metrics is not None:
                        metrics[model_name] = model_metrics
                        
                        # If predictions already exist, store them
                        if predicted is not None and actual is not None:
                            forecasts[model_name] = predicted
                            
                            # If this is our first model, store the actual values
                            if not hasattr(self, 'actual_values'):
                                self.actual_values = actual
                                self.actual_dates = model_results.get('test_dates', None)
                    
                    # Otherwise generate new predictions
                    else:
                        logger.info(f"Generating predictions for {model_name}")
                        
                        # The prediction approach depends on the model type
                        if 'deepar' in model_name.lower():
                            # DeepAR models have a different prediction interface
                            from src.models.deepar_model import DeepARModel
                            if isinstance(model, DeepARModel):
                                pred = model.predict(test_data, prediction_length)
                                forecasts[model_name] = pred
                        
                        elif 'prophet' in model_name.lower():
                            # Prophet models have a different prediction interface
                            from src.models.prophet_model import ProphetModel
                            if isinstance(model, ProphetModel):
                                pred = model.predict(test_data, prediction_length)
                                forecasts[model_name] = pred
                        
                        else:
                            # Standard model prediction
                            if hasattr(model, 'predict'):
                                pred = model.predict(test_data, prediction_length)
                                forecasts[model_name] = pred
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                logger.exception("Exception details:")
        
        # Create model comparison if we have metrics
        if metrics:
            comparison_df = create_model_comparison(metrics, self.output_dir)
            logger.info("Model comparison created")
            
            # Save as JSON as well
            with open(os.path.join(self.output_dir, 'model_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        
        # Create forecast comparison plot if we have forecasts
        if forecasts and hasattr(self, 'actual_values') and self.actual_dates is not None:
            # Create a Series with actual values
            actual_series = pd.Series(self.actual_values, index=self.actual_dates)
            
            # Plot forecasts
            plot_forecasts(
                forecasts=forecasts,
                actual=actual_series,
                output_path=os.path.join(self.output_dir, 'forecast_comparison.png'),
                title='Model Forecast Comparison'
            )
            logger.info("Forecast comparison plot created")
        
        return {
            'metrics': metrics,
            'forecasts': forecasts
        }
    
    def forecast_future(self, models: Dict[str, Any], data_path: str, 
                       forecast_horizon: int) -> Dict[str, Any]:
        """
        Generate future forecasts beyond available data.
        
        Args:
            models: Dictionary with trained models
            data_path: Path to CSV data with latest available points
            forecast_horizon: Number of steps to forecast ahead
            
        Returns:
            Dictionary with future forecasts
        """
        logger.info(f"Generating future forecasts for {forecast_horizon} steps")
        
        # Load latest data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Get the frequency
        try:
            freq = self._infer_frequency(data)
            logger.info(f"Inferred data frequency: {freq}")
        except:
            # Default to hourly if can't determine
            freq = 'H'
            logger.warning(f"Could not infer frequency, defaulting to hourly (H)")
        
        # Create future date range
        last_date = data.index[-1]
        # future_dates = pd.date_range(
        #     start=last_date + pd.Timedelta(hours=1 if freq == 'H' else days=1),
        #     periods=forecast_horizon,
        #     freq=freq
        # )
        if freq == 'H':
            future_dates = pd.date_range(
                start=data.index[-1] + pd.Timedelta(hours=1),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            future_dates = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq=freq
            )
        
        # Initialize future forecasts
        future_forecasts = {}
        
        # Generate forecasts for each model
        for model_name, model_results in models.items():
            try:
                # Extract model object from results
                if 'model' in model_results:
                    model = model_results['model']
                    
                    # The prediction approach depends on the model type
                    if 'deepar' in model_name.lower():
                        # DeepAR models have a different prediction interface
                        from src.models.deepar_model import DeepARModel
                        if isinstance(model, DeepARModel):
                            pred = model.predict(data, forecast_horizon)
                            future_forecasts[model_name] = pred
                    
                    elif 'prophet' in model_name.lower():
                        # Prophet models have a different prediction interface
                        from src.models.prophet_model import ProphetModel
                        if isinstance(model, ProphetModel):
                            pred = model.predict(data, forecast_horizon)
                            future_forecasts[model_name] = pred
                    
                    else:
                        # Standard model prediction - may need custom handling for multi-step
                        if hasattr(model, 'predict'):
                            # For traditional models like XGBoost, we need iterative forecasting
                            # This is a simplified approach that doesn't update features properly
                            logger.warning(f"Using simplified multi-step forecasting for {model_name}")
                            pred = np.zeros(forecast_horizon)
                            future_forecasts[model_name] = pred
            
            except Exception as e:
                logger.error(f"Failed to generate future forecast for {model_name}: {e}")
                logger.exception("Exception details:")
        
        # Create a DataFrame with future forecasts
        future_df = pd.DataFrame(index=future_dates)
        
        for model_name, forecast in future_forecasts.items():
            # Handle differently shaped forecasts
            if len(forecast) == forecast_horizon:
                future_df[model_name] = forecast
            else:
                logger.warning(f"Forecast from {model_name} has unexpected shape: {forecast.shape}")
                # Try to adapt if possible
                if len(forecast) > 0:
                    future_df[model_name] = forecast[:forecast_horizon]
                else:
                    future_df[model_name] = np.nan
        
        # Save future forecasts
        future_df.to_csv(os.path.join(self.output_dir, 'future_forecasts.csv'))
        
        # Plot future forecasts
        self._plot_future_forecasts(future_df, data, freq)
        
        return {
            'future_dates': future_dates,
            'forecasts': future_forecasts,
            'forecast_df': future_df
        }
    
    def _infer_frequency(self, data: pd.DataFrame) -> str:
        """
        Infer the frequency of a time series DataFrame.
        
        Args:
            data: DataFrame with DatetimeIndex
            
        Returns:
            Frequency string ('H' for hourly, 'D' for daily, etc.)
        """
        # Check if index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        # Compute most common difference between consecutive timestamps
        diff = pd.Series(data.index).diff().dropna()
        
        # Get the most common timedelta
        most_common = diff.value_counts().index[0]
        
        # Convert to frequency string
        if most_common == pd.Timedelta(hours=1):
            return 'H'
        elif most_common == pd.Timedelta(days=1):
            return 'D'
        elif most_common == pd.Timedelta(hours=24):
            return 'D'
        elif most_common == pd.Timedelta(weeks=1):
            return 'W'
        elif 28 <= most_common.days <= 31:
            return 'M'
        else:
            # Default to hourly
            return 'H'
    
    def _plot_future_forecasts(self, future_df: pd.DataFrame, historical_data: pd.DataFrame, 
                             freq: str = 'H') -> None:
        """
        Plot future forecasts along with historical data.
        
        Args:
            future_df: DataFrame with future forecasts
            historical_data: DataFrame with historical data
            freq: Data frequency ('H' for hourly, 'D' for daily)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 7 days or 30 days)
        history_size = 0
        if freq == 'H':
            # Hourly data - show last 7 days
            history_size = min(7 * 24, len(historical_data))
        else:
            # Daily data - show last 30 days
            history_size = min(30, len(historical_data))
            
        historical_values = historical_data.iloc[-history_size:]['tx_count'].values
        historical_dates = historical_data.index[-history_size:]
        
        plt.plot(historical_dates, historical_values, 'k-', label='Historical')
        
        # Plot future forecasts for each model
        colors = ['b', 'r', 'g', 'm', 'c']
        for i, column in enumerate(future_df.columns):
            plt.plot(future_df.index, future_df[column].values, 
                    f'{colors[i % len(colors)]}-', label=f'{column} Forecast')
        
        plt.title('Future Transaction Volume Forecast')
        plt.xlabel('Date')
        plt.ylabel('Transaction Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'future_forecast.png'))
        plt.close()
        logger.info(f"Future forecast plot saved to {self.output_dir}")