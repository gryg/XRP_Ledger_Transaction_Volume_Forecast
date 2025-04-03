"""LSTM model implementation for time series forecasting."""

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMModel(TimeSeriesModel):
    """LSTM model implementation for time series forecasting."""
    
    def __init__(self, output_dir: str = "./output/models"):
        """
        Initialize the LSTM model.
        
        Args:
            output_dir: Directory to save model and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model placeholders
        self.model = None
        self.scaler = None
        self.seq_length = None
        self.forecast_horizon = None
    
    def _prepare_sequences(self, df: pd.DataFrame, target_col: str, 
                          seq_length: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            seq_length: Length of input sequence
            forecast_horizon: How many steps ahead to forecast
            
        Returns:
            X and y for training
        """
        X, y = [], []
        
        for i in range(len(df) - seq_length - forecast_horizon + 1):
            X.append(df.iloc[i:(i + seq_length)].values)
            
            if forecast_horizon == 1:
                y.append(df.iloc[i + seq_length][target_col])
            else:
                y.append(df.iloc[i + seq_length:i + seq_length + forecast_horizon][target_col].values)
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, target_column: str = 'tx_count',
             seq_length: int = 24, forecast_horizon: int = 1, 
             test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Train an LSTM model for time series forecasting.
        
        Args:
            data: DataFrame with time series data
            target_column: Name of the target column
            seq_length: Number of time steps to use as input
            forecast_horizon: How many steps ahead to forecast
            test_size: Proportion of data to use for testing
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with training results
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            
            logger.info(f"Training LSTM model on data with shape {data.shape}")
            
            # Store parameters for later use
            self.seq_length = seq_length
            self.forecast_horizon = forecast_horizon
            
            # Drop NaN values
            data = data.dropna()
            
            # Scale all features for deep learning
            self.scaler = StandardScaler()
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Save target scaler for later inverse transformation
            self.target_scaler = StandardScaler()
            self.target_scaler.fit_transform(data[[target_column]])
            
            # Prepare sequences
            X, y = self._prepare_sequences(df_scaled, target_column, seq_length, forecast_horizon)
            
            logger.info(f"Prepared sequences with shapes: X {X.shape}, y {y.shape}")
            
            # Split data ensuring time order is preserved
            train_size = int(len(X) * (1 - test_size))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM model
            lstm_type = kwargs.get('lstm_type', 'bidirectional')
            units = kwargs.get('units', [100, 50])
            dropout = kwargs.get('dropout', 0.2)
            
            model = Sequential()
            
            if lstm_type == 'bidirectional':
                # Bidirectional LSTM
                model.add(Bidirectional(LSTM(units[0], return_sequences=True), input_shape=(seq_length, X.shape[2])))
                model.add(Dropout(dropout))
                model.add(Bidirectional(LSTM(units[1])))
                model.add(Dropout(dropout))
            else:
                # Standard LSTM
                model.add(LSTM(units[0], return_sequences=True, input_shape=(seq_length, X.shape[2])))
                model.add(Dropout(dropout))
                model.add(LSTM(units[1]))
                model.add(Dropout(dropout))
            
            # Output layer
            model.add(Dense(forecast_horizon))
            
            # Compile model
            optimizer = kwargs.get('optimizer', 'adam')
            loss = kwargs.get('loss', 'mse')
            
            model.compile(optimizer=optimizer, loss=loss)
            
            # Create callbacks
            model_path = os.path.join(self.output_dir, "lstm_model.keras")
            
            checkpoint = ModelCheckpoint(
                model_path,
                save_best_only=True,
                monitor='val_loss'
            )
            
            early_stopping = EarlyStopping(
                patience=kwargs.get('patience', 20),
                restore_best_weights=True,
                monitor='val_loss'
            )
            
            # Train model
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, checkpoint],
                verbose=kwargs.get('verbose', 1)
            )
            
            # Store model
            self.model = model
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Get test dates
            test_dates = data.index[train_size + seq_length:train_size + seq_length + len(y_test)]
            
            # Inverse transform predictions and actual values
            y_test_inv, y_pred_inv = self._inverse_transform(y_test, y_pred, data, target_column)
            
            # Calculate metrics
            metrics = self.evaluate(y_test_inv, y_pred_inv)
            
            logger.info(f"LSTM Model Performance:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "lstm_training_history.png"))
            plt.close()
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, y_test_inv, label='Actual')
            plt.plot(test_dates, y_pred_inv, label='Predicted')
            plt.title('LSTM: Actual vs Predicted Transaction Volume')
            plt.xlabel('Date')
            plt.ylabel('Transaction Count')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "lstm_predictions.png"))
            plt.close()
            
            # Save training metrics
            with open(os.path.join(self.output_dir, "lstm_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return {
                'model': self.model,
                'predictions': y_pred_inv,
                'actual': y_test_inv,
                'metrics': metrics,
                'history': history.history,
                'test_dates': test_dates
            }
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    def _inverse_transform(self, y_test, y_pred, original_data, target_column):
        """
        Inverse transform scaled predictions to original scale.
        
        Args:
            y_test: Test targets
            y_pred: Predicted values
            original_data: Original dataframe with correct scale
            target_column: Name of the target column
            
        Returns:
            Tuple of inverse-transformed actual and predicted values
        """
        # Get the number of samples
        n_samples = y_test.shape[0]
        
        # Reshape for inverse transformation if needed
        if self.forecast_horizon == 1:
            y_test_2d = y_test.reshape(-1, 1)
            y_pred_2d = y_pred.reshape(-1, 1)
        else:
            # For multi-step forecasting, handle properly
            y_test_2d = y_test
            y_pred_2d = y_pred
        
        # Create dummy dataframes with all zeros except target column
        dummy_test = np.zeros((n_samples, len(original_data.columns)))
        dummy_pred = np.zeros((n_samples, len(original_data.columns)))
        
        # Get the index of the target column
        target_idx = original_data.columns.get_loc(target_column)
        
        # Fill in the target values
        if self.forecast_horizon == 1:
            dummy_test[:, target_idx] = y_test_2d.ravel()
            dummy_pred[:, target_idx] = y_pred_2d.ravel()
        else:
            # For multi-step, use only the first prediction or all steps
            # This depends on how you want to evaluate
            dummy_test[:, target_idx] = y_test_2d[:, 0]  # First step only
            dummy_pred[:, target_idx] = y_pred_2d[:, 0]  # First step only
        
        # Inverse transform
        y_test_inv = self.scaler.inverse_transform(dummy_test)[:, target_idx]
        y_pred_inv = self.scaler.inverse_transform(dummy_pred)[:, target_idx]
        
        return y_test_inv, y_pred_inv
    
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
        
        # Scale input data
        data_scaled = self.scaler.transform(data)
        
        # Check if we have enough data for the sequence
        if len(data) < self.seq_length:
            raise ValueError(f"Not enough data points for sequence length {self.seq_length}")
        
        # Prepare input sequence (use last seq_length points)
        input_seq = data_scaled[-self.seq_length:].reshape(1, self.seq_length, data.shape[1])
        
        # Generate predictions
        scaled_pred = self.model.predict(input_seq)
        
        # Create dummy array for inverse transformation
        dummy_pred = np.zeros((1, data.shape[1]))
        
        # Get the index of the target column
        target_idx = data.columns.get_loc(kwargs.get('target_column', 'tx_count'))
        
        # Fill in the predicted value(s)
        if self.forecast_horizon == 1:
            dummy_pred[:, target_idx] = scaled_pred.ravel()
        else:
            # For multi-step, use only the first prediction
            dummy_pred[:, target_idx] = scaled_pred[0, 0]
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(dummy_pred)[:, target_idx]
        
        return predictions
    
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
        
        # Save the Keras model
        self.model.save(f"{path}.keras")
        
        # Save the scaler
        with open(f"{path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save target scaler
        with open(f"{path}_target_scaler.pkl", 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        # Save configuration
        config = {
            'seq_length': self.seq_length,
            'forecast_horizon': self.forecast_horizon
        }
        
        with open(f"{path}_config.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        logger.info(f"Model saved to {path}.keras")
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path from where to load the model
        """
        try:
            import tensorflow as tf
            
            # Load the Keras model
            self.model = tf.keras.models.load_model(f"{path}.keras")
            
            # Load the scaler
            with open(f"{path}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load target scaler
            with open(f"{path}_target_scaler.pkl", 'rb') as f:
                self.target_scaler = pickle.load(f)
            
            # Load configuration
            with open(f"{path}_config.json", 'r') as f:
                config = json.load(f)
                self.seq_length = config['seq_length']
                self.forecast_horizon = config['forecast_horizon']
                
            logger.info(f"Model loaded from {path}.keras")
            
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow: {e}")
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e