"""Metric calculation utilities for evaluating forecasts."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value (as percentage)
    """
    # Handle division by zero by replacing zeros with a small value
    with np.errstate(divide='ignore', invalid='ignore'):
        # Use a masked array to handle zeros properly
        masked_actual = np.ma.masked_where(actual == 0, actual)
        mape = np.mean(np.abs((masked_actual - predicted) / masked_actual)) * 100
        
    return float(mape) if not np.isnan(mape) else 0.0


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate R² score.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        R² value
    """
    # Calculate mean of actual values
    mean_actual = np.mean(actual)
    
    # Calculate total sum of squares
    ss_total = np.sum((actual - mean_actual) ** 2)
    
    # Calculate residual sum of squares
    ss_residual = np.sum((actual - predicted) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    return r2


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        SMAPE value (as percentage)
    """
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = np.abs(actual) + np.abs(predicted)
        # Mask when denominator is zero
        masked_denominator = np.ma.masked_where(denominator == 0, denominator)
        smape = np.mean(2.0 * np.abs(predicted - actual) / masked_denominator) * 100
        
    return float(smape) if not np.isnan(smape) else 0.0


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Dictionary with all metrics
    """
    # Handle case where arrays might have different shapes
    min_len = min(len(actual), len(predicted))
    if min_len == 0:
        return {
            'MAE': float('nan'),
            'RMSE': float('nan'),
            'MAPE': float('nan'),
            'SMAPE': float('nan'),
            'R2': float('nan')
        }
        
    # Trim to common length
    actual_trimmed = actual[:min_len]
    predicted_trimmed = predicted[:min_len]
    
    # Calculate all metrics
    metrics = {
        'MAE': calculate_mae(actual_trimmed, predicted_trimmed),
        'RMSE': calculate_rmse(actual_trimmed, predicted_trimmed),
        'MAPE': calculate_mape(actual_trimmed, predicted_trimmed),
        'SMAPE': calculate_smape(actual_trimmed, predicted_trimmed),
        'R2': calculate_r2(actual_trimmed, predicted_trimmed)
    }
    
    return metrics