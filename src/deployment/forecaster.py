"""Forecaster module for deploying trained models."""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Forecaster:
    """Forecaster for deploying trained time series models."""
    
    def __init__(self, model_dir: str, output_dir: str = "./forecasts"):
        """
        Initialize the forecaster.
        
        Args:
            model_dir: Directory containing trained models
            output_dir: Directory to save forecasts
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = {}
    
    def load_models(self, model_type: str = "xgboost", freq: str = "hourly"):
        """
        Load trained models from disk.
        
        Args:
            model_type: Type of model to load ('xgboost', 'lstm', 'deepar', 'prophet')
            freq: Frequency of data ('hourly', 'daily')
            
        Returns:
            Loaded model
        """
        try:
            if model_type == "xgboost":
                from src.models import XGBoostModel
                model = XGBoostModel(output_dir=self.output_dir)
                model.load(os.path.join(self.model_dir, freq, "xgboost_model"))
                
            elif model_type == "lstm":
                from src.models import LSTMModel
                model = LSTMModel(output_dir=self.output_dir)
                model.load(os.path.join(self.model_dir, freq, "lstm_model"))
                
            elif model_type == "deepar":
                from src.models import DeepARModel
                model = DeepARModel(
                    freq='H' if freq == 'hourly' else 'D',
                    output_dir=os.path.join(self.output_dir, 'deepar', freq)
                )
                model.load(os.path.join(self.model_dir, 'deepar', freq, 'models', f"deepar_{'H' if freq == 'hourly' else 'D'}_model"))
                
            elif model_type == "prophet":
                from src.models import ProphetModel
                model = ProphetModel(output_dir=os.path.join(self.output_dir, 'prophet', freq))
                model.load(os.path.join(self.model_dir, 'prophet', freq, 'models', "prophet_model"))
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Store the loaded model
            self.models[f"{model_type}_{freq}"] = model
            logger.info(f"Loaded {model_type} model for {freq} data")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model for {freq} data: {e}")
            logger.exception("Exception details:")
            return None
    
    def generate_forecast(self, data_path: str, forecast_horizon: int = 24, 
                         model_type: str = "xgboost", freq: str = "hourly", **kwargs):
        """
        Generate forecasts using a loaded model.
        
        Args:
            data_path: Path to input data CSV
            forecast_horizon: Number of steps to forecast ahead
            model_type: Type of model to use
            freq: Frequency of data ('hourly', 'daily')
            **kwargs: Additional parameters for the model
            
        Returns:
            DataFrame with forecasts
        """
        model_key = f"{model_type}_{freq}"
        
        # Load model if not already loaded
        if model_key not in self.models:
            self.load_models(model_type, freq)
        
        if model_key not in self.models:
            logger.error(f"Model {model_key} could not be loaded")
            return None
        
        try:
            # Load the data
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {data_path} with shape {data.shape}")
            
            # Generate forecast
            model = self.models[model_key]
            predictions = model.predict(data, forecast_horizon, **kwargs)
            
            # Create future date range
            last_date = data.index[-1]
            if freq == 'hourly':
                future_dates = pd.date_range(
                    start=last_date + timedelta(hours=1),
                    periods=forecast_horizon,
                    freq='H'
                )
            else:
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_horizon,
                    freq='D'
                )
            
            # Create DataFrame with forecasts
            forecast_df = pd.DataFrame({
                'forecast': predictions
            }, index=future_dates)
            
            # Save forecast
            output_path = os.path.join(
                self.output_dir, 
                f"{model_type}_{freq}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            forecast_df.to_csv(output_path)
            logger.info(f"Forecast saved to {output_path}")
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            logger.exception("Exception details:")
            return None
    
    def generate_ensemble_forecast(self, data_path: str, forecast_horizon: int = 24,
                                  model_types: List[str] = None, freq: str = "hourly", 
                                  weights: Dict[str, float] = None):
        """
        Generate an ensemble forecast using multiple models.
        
        Args:
            data_path: Path to input data CSV
            forecast_horizon: Number of steps to forecast ahead
            model_types: List of model types to include in ensemble
            freq: Frequency of data ('hourly', 'daily')
            weights: Dictionary mapping model types to weights
            
        Returns:
            DataFrame with ensemble forecast
        """
        if model_types is None:
            model_types = ["xgboost", "lstm", "deepar", "prophet"]
        
        # Generate individual forecasts
        forecasts = {}
        for model_type in model_types:
            try:
                forecast_df = self.generate_forecast(
                    data_path=data_path,
                    forecast_horizon=forecast_horizon,
                    model_type=model_type,
                    freq=freq
                )
                
                if forecast_df is not None:
                    forecasts[model_type] = forecast_df['forecast'].values
                    
            except Exception as e:
                logger.error(f"Failed to generate {model_type} forecast: {e}")
        
        if not forecasts:
            logger.error("No forecasts could be generated")
            return None
        
        # Get dates from one of the forecasts
        future_dates = next(iter(forecasts.values())).index
        
        # Create ensemble forecast
        if weights is None:
            # Equal weights
            weights = {model_type: 1/len(forecasts) for model_type in forecasts}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Create weighted average
        ensemble = np.zeros(forecast_horizon)
        for model_type, forecast in forecasts.items():
            weight = weights.get(model_type, 0)
            if len(forecast) == forecast_horizon:
                ensemble += weight * forecast
        
        # Create DataFrame with ensemble forecast
        ensemble_df = pd.DataFrame({
            'forecast': ensemble
        }, index=future_dates)
        
        # Save ensemble forecast
        output_path = os.path.join(
            self.output_dir, 
            f"ensemble_{freq}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        ensemble_df.to_csv(output_path)
        logger.info(f"Ensemble forecast saved to {output_path}")
        
        return ensemble_df