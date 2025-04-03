"""Prophet model implementation for time series forecasting."""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from src.models.base import TimeSeriesModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProphetModel(TimeSeriesModel):
    """Prophet model implementation for time series forecasting."""
    
    def __init__(self, output_dir: str = "./output/prophet"):
        """
        Initialize the Prophet model.
        
        Args:
            output_dir: Directory to save model and results
        """
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'models')
        self.results_dir = os.path.join(output_dir, 'results')
        
        # Create output directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize model placeholder
        self.model = None
    
    def train(self, data: pd.DataFrame, target_column: str = 'tx_count',
              prediction_length: int = 24, test_size: float = 0.2,
              cross_val: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train a Prophet model for time series forecasting.
        
        Args:
            data: DataFrame with time series data
            target_column: Name of the target column
            prediction_length: Number of steps to predict
            test_size: Proportion of data to use for testing
            cross_val: Whether to perform cross-validation
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with training results
        """
        try:
            from prophet import Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
            
            logger.info(f"Training Prophet model on data with shape {data.shape}")
            
            # Prepare data for Prophet (requires 'ds' for dates and 'y' for values)
            # First, reset the index to get the datetime as a column
            prophet_df = data.reset_index()
            
            # Print column names to debug
            logger.debug(f"Available columns: {prophet_df.columns.tolist()}")
            
            # Rename the columns correctly based on the actual column names
            # The index column might have a different name than 'index'
            date_col = prophet_df.columns[0]  # First column should be the datetime index
            prophet_df = prophet_df.rename(columns={date_col: 'ds', target_column: 'y'})
            
            # Extract only the necessary columns
            prophet_df = prophet_df[['ds', 'y']]
            
            # Split into train and test sets
            train_size = int(len(prophet_df) * (1 - test_size))
            train_data = prophet_df.iloc[:train_size]
            test_data = prophet_df.iloc[train_size:]
            
            logger.info(f"Train data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            
            # Create and fit the Prophet model
            logger.info("Training Prophet model...")
            self.model = Prophet(
                changepoint_prior_scale=kwargs.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=kwargs.get('seasonality_prior_scale', 10.0),
                seasonality_mode=kwargs.get('seasonality_mode', 'multiplicative'),
                daily_seasonality=kwargs.get('daily_seasonality', True),
                weekly_seasonality=kwargs.get('weekly_seasonality', True),
                yearly_seasonality=kwargs.get('yearly_seasonality', False)
            )
            
            # Add hourly seasonality if appropriate
            if 'hour' in data.columns:
                self.model.add_seasonality(
                    name='hourly',
                    period=24,
                    fourier_order=kwargs.get('fourier_order', 5)
                )
            
            # Fit the model
            self.model.fit(train_data)
            
            # Save the model
            model_path = os.path.join(self.model_dir, 'prophet_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {model_path}")
            
            # Make predictions for the test period
            future = self.model.make_future_dataframe(periods=len(test_data), freq='H')
            forecast = self.model.predict(future)
            
            # Extract predictions for the test period
            predictions = forecast.iloc[-len(test_data):]['yhat'].values
            actuals = test_data['y'].values
            test_dates = test_data['ds']
            
            # Calculate metrics
            metrics = self.evaluate(actuals, predictions)
            
            logger.info(f"Prophet Model Performance:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Save metrics
            with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Optional: Perform cross validation
            if cross_val:
                logger.info("Performing cross validation (this may take a while)...")
                
                # Determine horizon based on data frequency
                try:
                    if len(train_data) > 100:  # More than 100 data points
                        time_diff = (train_data['ds'].iloc[1] - train_data['ds'].iloc[0]).total_seconds()
                        
                        if time_diff <= 3600:  # Hourly or finer
                            cv_horizon = f'{prediction_length} hours'
                            initial = '30 days'
                            period = '7 days'
                        else:  # Daily or coarser
                            cv_horizon = f'{prediction_length} days'
                            initial = '90 days'
                            period = '30 days'
                    else:
                        # Default
                        cv_horizon = f'{prediction_length} hours'
                        initial = '7 days'
                        period = '1 days'
                except:
                    # Default
                    cv_horizon = f'{prediction_length} hours'
                    initial = '7 days'
                    period = '1 days'
                
                logger.info(f"Cross validation with horizon={cv_horizon}, initial={initial}, period={period}")
                
                df_cv = cross_validation(
                    self.model,
                    initial=initial,
                    period=period,
                    horizon=cv_horizon
                )
                
                # Calculate performance metrics from cross validation
                cv_metrics = performance_metrics(df_cv)
                logger.info("Cross validation metrics:")
                logger.info(cv_metrics[['horizon', 'mae', 'rmse', 'mape']].tail().to_string())
                
                # Save cross validation metrics
                cv_metrics.to_csv(os.path.join(self.results_dir, 'cv_metrics.csv'))
            
            # Create plots
            
            # 1. Actual vs Predicted
            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, actuals, 'b-', label='Actual')
            plt.plot(test_dates, predictions, 'r-', label='Predicted')
            plt.title('Prophet: Actual vs Predicted Transaction Volume')
            plt.xlabel('Date')
            plt.ylabel('Transaction Count')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'actual_vs_predicted.png'))
            plt.close()
            
            # 2. Forecast components
            fig = self.model.plot_components(forecast)
            fig.savefig(os.path.join(self.results_dir, 'forecast_components.png'))
            plt.close(fig)
            
            # 3. Future predictions
            self._plot_future_forecast(prediction_length)
            
            return {
                'model': self.model,
                'predictions': predictions,
                'actual': actuals,
                'metrics': metrics,
                'test_dates': pd.to_datetime(test_dates)
            }
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("Prophet is required. Install with: pip install prophet")
        except Exception as e:
            logger.error(f"Failed to train Prophet model: {e}")
            raise e
    
    def _plot_future_forecast(self, future_periods: int):
        """
        Plot future forecast beyond available data.
        
        Args:
            future_periods: Number of periods to forecast
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Make future dataframe
            future = self.model.make_future_dataframe(periods=future_periods, freq='H')
            future_forecast = self.model.predict(future)
            
            # Plot future forecast
            plt.figure(figsize=(12, 6))
            
            # Get historical data from model
            historical_data = self.model.history
            
            # Plot historical data (last week)
            historical_days = 7
            if len(historical_data) >= historical_days * 24:  # At least a week of hourly data
                historical_hours = historical_days * 24
                historical_data_subset = historical_data.iloc[-historical_hours:]
            else:
                # Use all available data
                historical_data_subset = historical_data
            
            plt.plot(historical_data_subset['ds'], historical_data_subset['y'], 'b-', label='Historical Data')
            
            # Plot future forecast
            future_dates = future_forecast.iloc[-future_periods:]['ds']
            future_values = future_forecast.iloc[-future_periods:]['yhat'].values
            future_lower = future_forecast.iloc[-future_periods:]['yhat_lower'].values
            future_upper = future_forecast.iloc[-future_periods:]['yhat_upper'].values
            
            plt.plot(future_dates, future_values, 'r-', label='Forecast')
            plt.fill_between(
                future_dates,
                future_lower,
                future_upper,
                color='r',
                alpha=0.2,
                label='95% Confidence Interval'
            )
            
            plt.title('Prophet: Future Transaction Volume Forecast')
            plt.xlabel('Date')
            plt.ylabel('Transaction Count')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'future_forecast.png'))
            plt.close()
            
            logger.info(f"Future forecast plot saved to {self.results_dir}")
        except Exception as e:
            logger.error(f"Failed to plot future forecast: {e}")
            
    def predict(self, data: pd.DataFrame, forecast_horizon: int, **kwargs) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            data: DataFrame containing data for prediction
            forecast_horizon: Number of steps to forecast ahead
            **kwargs: Additional parameters
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=forecast_horizon)
            forecast = self.model.predict(future)
            
            # Extract forecasted values
            predictions = forecast.iloc[-forecast_horizon:]['yhat'].values
            
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.model, f)
            
        logger.info(f"Model saved to {path}.pkl")
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path from where to load the model
        """
        try:
            # Load the model
            with open(f"{path}.pkl", 'rb') as f:
                self.model = pickle.load(f)
                
            logger.info(f"Model loaded from {path}.pkl")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e