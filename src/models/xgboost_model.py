"""XGBoost model implementation for time series forecasting."""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from src.models.base import TimeSeriesModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostModel(TimeSeriesModel):
    """XGBoost implementation for time series forecasting."""
    
    def __init__(self, output_dir: str = "./output/models"):
        """
        Initialize the XGBoost model.
        
        Args:
            output_dir: Directory to save model and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model placeholders
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def train(self, data: pd.DataFrame, target_column: str = 'tx_count', 
              forecast_horizon: int = 24, test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Train an XGBoost model for time series forecasting.
        
        Args:
            data: DataFrame with time series data
            target_column: Name of the target column
            forecast_horizon: How many steps ahead to forecast
            test_size: Proportion of data to use for testing
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with training results
        """
        try:
            import xgboost as xgb
            
            logger.info(f"Training XGBoost model on data with shape {data.shape}")
            
            # Drop NaN values
            data = data.dropna()
            
            # Split data using base class method
            X_train, X_test, y_train, y_test = self.prepare_train_test_split(
                data, test_size=test_size, target_column=target_column
            )
            
            # Scale numerical features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Save feature names for later use
            self.feature_names = X_train.columns.tolist()
            
            # Create XGBoost model with parameters
            model_params = {
                'n_estimators': kwargs.get('n_estimators', 1000),
                'learning_rate': kwargs.get('learning_rate', 0.01),
                'max_depth': kwargs.get('max_depth', 5),
                'subsample': kwargs.get('subsample', 0.8),
                'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'early_stopping_rounds': 50,
                'random_state': 42
            }
            
            self.model = xgb.XGBRegressor(**model_params)
            
            # Train with eval set
            eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=kwargs.get('verbose', True)
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics using base class method
            metrics = self.evaluate(y_test.values, y_pred)
            
            logger.info(f"XGBoost Model Performance:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
            # Save model and metadata
            self.save(os.path.join(self.output_dir, "xgboost_model"))
            
            # Save feature importance
            feature_importance.to_csv(
                os.path.join(self.output_dir, "feature_importance.csv"), 
                index=False
            )
            
            # Visualize actual vs predicted
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.index, y_test.values, label='Actual')
            plt.plot(y_test.index, y_pred, label='Predicted')
            plt.title('XGBoost: Actual vs Predicted Transaction Volume')
            plt.xlabel('Date')
            plt.ylabel('Transaction Count')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "xgboost_predictions.png"))
            plt.close()
            
            return {
                'model': self.model,
                'predictions': y_pred,
                'actual': y_test.values,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'test_dates': y_test.index
            }
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
    
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
        
        # Handle case where we're doing one-step prediction using existing data
        if kwargs.get('use_existing_data', True):
            # Prepare features
            if 'target_column' in kwargs:
                X = data.drop(kwargs['target_column'], axis=1)
            else:
                X = data
                
            # Ensure we have all required features
            missing_features = set(self.feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0
                
            # Ensure correct feature order
            X = X[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Generate predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
        
        # Handle multi-step forecasting (more complex, requires iterative prediction)
        else:
            # This is a simplified approach - real implementation would need to:
            # 1. Create the initial feature set from the last available data points
            # 2. Generate one prediction
            # 3. Append the prediction to the history
            # 4. Update the features (lags, rolling stats, etc.)
            # 5. Repeat steps 2-4 for each step in the forecast horizon
            
            logger.warning("Multi-step forecasting not fully implemented for XGBoost")
            # Return dummy forecasts
            return np.zeros(forecast_horizon)
    
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
        
        # Save the scaler
        with open(f"{path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        with open(f"{path}_feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
            
        logger.info(f"Model saved to {path}.pkl")
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path from where to load the model
        """
        # Load the model
        with open(f"{path}.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the scaler
        with open(f"{path}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(f"{path}_feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
            
        logger.info(f"Model loaded from {path}.pkl")