"""Base model interface for time series forecasting models."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union


class TimeSeriesModel(ABC):
    """Abstract base class for all time series forecasting models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame, target_column: str, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            data: DataFrame containing training data
            target_column: Name of the target column to predict
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, forecast_horizon: int, **kwargs) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            data: DataFrame containing data for prediction
            forecast_horizon: Number of steps to forecast ahead
            **kwargs: Additional model-specific parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path from where to load the model
        """
        pass
    
    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Handle cases where arrays might have different shapes
        min_len = min(len(actual), len(predicted))
        if min_len == 0:
            return {
                'MAE': float('nan'),
                'RMSE': float('nan'),
                'R2': float('nan'),
                'MAPE': float('nan')
            }
            
        # Trim to common length
        actual_trimmed = actual[:min_len]
        predicted_trimmed = predicted[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(actual_trimmed, predicted_trimmed)
        rmse = np.sqrt(mean_squared_error(actual_trimmed, predicted_trimmed))
        r2 = r2_score(actual_trimmed, predicted_trimmed)
        
        # Calculate MAPE with handling for zeros in actual values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual_trimmed - predicted_trimmed) / np.where(actual_trimmed != 0, actual_trimmed, np.nan))) * 100
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape) if not np.isnan(mape) else 0.0
        }
    
    @staticmethod
    def prepare_train_test_split(data: pd.DataFrame, test_size: float = 0.2, 
                                target_column: str = 'tx_count') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets, respecting time order.
        
        Args:
            data: DataFrame containing time series data
            test_size: Proportion of data to use for testing
            target_column: Name of target column
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Ensure data is sorted by index (datetime)
        data = data.sort_index()
        
        # Split data ensuring time order is preserved
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Prepare features and target
        X_train = train_data.drop(target_column, axis=1)
        y_train = train_data[target_column]
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
        
        return X_train, X_test, y_train, y_test