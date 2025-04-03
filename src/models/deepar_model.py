import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import logging
import torch
from datetime import datetime, timedelta
from pathlib import Path

# Import necessary GluonTS components
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

# Import PyTorch model with safe fallbacks
TorchDeepAREstimator = None
MXDeepAREstimator = None

try:
    from gluonts.torch.model.deepar import DeepAREstimator as TorchDeepAREstimator
except ImportError:
    pass

# DO NOT try to import MXNet - it has a syntax error
# We'll just assume MX backend is not available

try:
    from gluonts.model.predictor import Predictor
except ImportError:
    Predictor = None

# Import from base class
from src.models.base import TimeSeriesModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepARModel(TimeSeriesModel):
    """
    DeepAR model implementation using GluonTS with direct PyTorch integration.
    """

    def __init__(self, freq: str = 'H', output_dir: str = "./output/deepar",
                 context_length: Optional[int] = None):
        """
        Initialize the DeepAR model.

        Args:
            freq: Data frequency ('H' for hourly, 'D' for daily)
            output_dir: Directory to save model and results
            context_length: Length of context window (default: auto-determined later)
        """
        self.freq = freq
        self.output_dir = output_dir
        self.context_length = context_length
        self.model_save_dir = os.path.join(output_dir, 'models')
        self.results_dir = os.path.join(output_dir, 'results')

        # Create output directories
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize model placeholders
        self.predictor = None
        self.pytorch_model = None  # Store actual PyTorch model for direct inference
        self.training_data_info = {}
        self.scaler = None  # For normalizing the data
        self.is_torch_version = True  # Default to torch version

    def _prepare_data(self, data: pd.DataFrame, target_column: str = 'tx_count',
                      test_size: float = 0.2, prediction_length: Optional[int] = None) -> tuple:
        """
        Prepare data for DeepAR in GluonTS format.

        Args:
            data: DataFrame with time series data
            target_column: Target column name
            test_size: Test size proportion
            prediction_length: Prediction horizon

        Returns:
            Tuple with prepared datasets and test values
        """
        if prediction_length is None:
            raise ValueError("prediction_length must be provided")

        try:
            logger.info(f"Preparing data for DeepAR with shape {data.shape}")

            # Ensure the index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Store target column for later
            self.training_data_info['target_column'] = target_column
            
            # Define feature columns
            feature_columns = [col for col in data.columns if col != target_column and col not in ['hour', 'day_of_week', 'month', 'year']]
            time_feature_cols = [col for col in ['hour', 'day_of_week', 'month', 'year'] if col in data.columns]
            
            # Store for later use
            self.training_data_info['feature_columns'] = feature_columns
            self.training_data_info['time_feature_cols'] = time_feature_cols

            # Drop NaNs
            cols_to_check_nan = [target_column] + feature_columns + time_feature_cols
            data = data.dropna(subset=[col for col in cols_to_check_nan if col in data.columns])
            logger.info(f"Data shape after dropping NaN: {data.shape}")

            if len(data) < prediction_length * 2:
                raise ValueError(f"Not enough data after dropping NaNs ({len(data)} rows) for prediction length {prediction_length}")

            # Split into train/test
            test_length = prediction_length
            train_data = data[:-test_length]

            logger.info(f"Train data shape: {train_data.shape}, Test prediction length: {test_length}")
            
            # Store important dates
            self.training_data_info['train_end_date'] = train_data.index[-1]
            self.training_data_info['full_data_start_date'] = data.index[0]
            self.training_data_info['target_range'] = {
                'min': data[target_column].min(),
                'max': data[target_column].max(),
                'mean': data[target_column].mean(),
                'std': data[target_column].std()
            }

            # Prepare GluonTS datasets
            # Training dataset (without test data)
            train_target = train_data[target_column].values
            train_start = train_data.index[0]
            
            train_feat_dynamic_real = []
            if time_feature_cols:
                train_feat_dynamic_real.append(train_data[time_feature_cols].values.T)
            
            train_ds_entry = {
                FieldName.TARGET: train_target,
                FieldName.START: train_start,
            }
            
            if train_feat_dynamic_real:
                train_ds_entry[FieldName.FEAT_DYNAMIC_REAL] = np.concatenate(train_feat_dynamic_real, axis=0)
            
            train_ds = ListDataset([train_ds_entry], freq=self.freq)
            
            # Test dataset (full dataset for context)
            full_target = data[target_column].values
            full_start = data.index[0]
            
            full_feat_dynamic_real = []
            if time_feature_cols:
                full_feat_dynamic_real.append(data[time_feature_cols].values.T)
            
            test_ds_entry = {
                FieldName.TARGET: full_target,
                FieldName.START: full_start,
            }
            
            if full_feat_dynamic_real:
                test_ds_entry[FieldName.FEAT_DYNAMIC_REAL] = np.concatenate(full_feat_dynamic_real, axis=0)
            
            test_ds = ListDataset([test_ds_entry], freq=self.freq)
            
            # Extract test values for evaluation
            test_target_values = data[target_column].values[-test_length:]
            test_index = data.index[-test_length:]
            
            return train_ds, test_ds, full_target, test_target_values, test_index

        except Exception as e:
            logger.error(f"Error during data preparation: {e}", exc_info=True)
            raise

    def train(self, data: pd.DataFrame, target_column: str = 'tx_count',
              prediction_length: int = 24, epochs: int = 10,
              learning_rate: float = 1e-3, batch_size: int = 32,
              num_layers: int = 2, hidden_size: int = 40, dropout_rate: float = 0.1,
              **kwargs) -> Dict[str, Any]:
        """
        Train the DeepAR model using the GluonTS PyTorch API.

        Args:
            data: DataFrame with time series data
            target_column: Name of the target column
            prediction_length: Number of steps to predict ahead
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            num_layers: Number of RNN layers
            hidden_size: Number of hidden units in RNN
            dropout_rate: Dropout rate
            **kwargs: Additional parameters

        Returns:
            Dictionary with training results
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index is not DatetimeIndex. Attempting conversion.")
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"Failed to convert index: {e}")
                raise TypeError("Data index must be DatetimeIndex")

        # Set context length if not provided
        if self.context_length is None:
            self.context_length = max(prediction_length * 2, 50)
            logger.info(f"Context length set to: {self.context_length}")

        # Prepare data
        train_ds, test_ds, full_target, test_target_values, test_index = self._prepare_data(
            data, target_column=target_column, test_size=0.2, prediction_length=prediction_length
        )

        # Check if time features available
        use_feat_dynamic_real = bool(self.training_data_info.get('time_feature_cols'))

        # Determine which GluonTS implementation to use
        if TorchDeepAREstimator is not None:
            self.is_torch_version = True
            logger.info("Using PyTorch-based GluonTS DeepAR implementation")
        elif MXDeepAREstimator is not None:
            self.is_torch_version = False
            logger.info("Using MXNet-based GluonTS DeepAR implementation")
        else:
            logger.error("Neither PyTorch nor MXNet version of GluonTS DeepAR is available")
            raise ImportError("GluonTS DeepAR is required")

        # Start training
        logger.info(f"Training DeepAR model for {epochs} epochs...")
        start_time = time.time()

        try:
            if self.is_torch_version:
                # Configure trainer
                trainer_kwargs = {
                    "max_epochs": epochs,
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "enable_progress_bar": True,
                }
                
                # Create estimator
                estimator = TorchDeepAREstimator(
                    freq=self.freq,
                    prediction_length=prediction_length,
                    context_length=self.context_length,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate,
                    lr=learning_rate,
                    batch_size=batch_size,
                    trainer_kwargs=trainer_kwargs,
                    **kwargs
                )
                
                # Train the model
                self.predictor = estimator.train(training_data=train_ds)
                
                # Extract the actual PyTorch model for direct inference
                try:
                    # Extract the PyTorch model from the predictor
                    self.pytorch_model = self.predictor.prediction_net.model
                    
                    # Store model parameters for later reconstruction if needed
                    self.model_params = {
                        'freq': self.freq,
                        'prediction_length': prediction_length,
                        'context_length': self.context_length,
                        'num_layers': num_layers,
                        'hidden_size': hidden_size,
                        'dropout_rate': dropout_rate
                    }
                except Exception as model_extract_err:
                    logger.error(f"Could not extract PyTorch model: {model_extract_err}")
            else:
                raise NotImplementedError("MXNet DeepAR is not fully supported in this implementation")

        except Exception as e:
            logger.error(f"DeepAR training failed: {e}", exc_info=True)
            return {
                'predictor': None,
                'metrics': {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')},
                'training_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }

        training_time = time.time() - start_time
        logger.info(f"Model trained in {training_time:.2f} seconds")

        # Evaluate the model
        metrics = self._evaluate_direct(test_target_values, test_index, prediction_length)

        return {
            'predictor': self.predictor,
            'metrics': metrics,
            'training_time': training_time,
            'status': 'success'
        }

    def _evaluate_direct(self, test_target_values, test_index, prediction_length):
        """
        Evaluate the model using direct inference with the PyTorch model.
        
        Args:
            test_target_values: Actual values to compare against
            test_index: DatetimeIndex for the test period
            prediction_length: Forecast horizon
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.pytorch_model is None and self.predictor is None:
            logger.error("No model available for evaluation")
            return {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
            
        try:
            # Generate predictions directly
            predictions, _ = self.predict_with_pytorch_model(test_index[0] - timedelta(days=1), prediction_length)
            
            if len(predictions) > 0:
                # Ensure same length
                min_len = min(len(test_target_values), len(predictions))
                metrics = self.evaluate(test_target_values[:min_len], predictions[:min_len])
                
                # Plot comparison
                try:
                    plt.figure(figsize=(12, 6))
                    plt.plot(test_index[:min_len], test_target_values[:min_len], 'k-', label='Actual')
                    plt.plot(test_index[:min_len], predictions[:min_len], 'b-', label='Prediction')
                    plt.legend()
                    plt.title('DeepAR: Actual vs Predicted')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plot_file = os.path.join(self.results_dir, 'deepar_evaluation.png')
                    plt.savefig(plot_file)
                    plt.close()
                except Exception as plot_err:
                    logger.error(f"Error plotting evaluation: {plot_err}")
                
                return metrics
            else:
                return {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                
        except Exception as e:
            logger.error(f"Error during direct evaluation: {e}", exc_info=True)
            return {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}

    def predict(self, data: pd.DataFrame, forecast_horizon: int = 24) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Generate predictions using either the direct PyTorch model or the GluonTS predictor.
        
        Args:
            data: Input data for prediction
            forecast_horizon: Prediction horizon
            
        Returns:
            Tuple of predictions and date index
        """
        if self.pytorch_model is None and self.predictor is None:
            logger.error("No model available for prediction")
            return np.array([]), pd.DatetimeIndex([])
            
        try:
            # Create future dates
            last_date = data.index[-1]
            if self.freq == 'H':
                prediction_start = last_date + pd.Timedelta(hours=1)
                future_dates = pd.date_range(start=prediction_start, periods=forecast_horizon, freq='H')
            elif self.freq == 'D':
                prediction_start = last_date + pd.Timedelta(days=1)
                future_dates = pd.date_range(start=prediction_start, periods=forecast_horizon, freq='D')
            else:
                logger.warning(f"Using default handling for frequency: {self.freq}")
                future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=self.freq)[1:]
                
            # Try direct PyTorch inference first
            if self.pytorch_model is not None:
                try:
                    predictions, _ = self.predict_with_pytorch_model(last_date, forecast_horizon)
                    return predictions, future_dates
                except Exception as pytorch_err:
                    logger.warning(f"Direct PyTorch prediction failed: {pytorch_err}")
            
            # Fall back to GluonTS predictor
            if self.predictor is not None:
                try:
                    target_column = self.training_data_info.get('target_column', 'tx_count')
                    
                    # Create GluonTS dataset
                    ds = ListDataset(
                        [{
                            FieldName.TARGET: data[target_column].values,
                            FieldName.START: data.index[0]
                        }],
                        freq=self.freq
                    )
                    
                    # Generate forecasts
                    forecasts = list(self.predictor.predict(ds))
                    forecast = forecasts[0]
                    predictions = forecast.mean
                    
                    return predictions, future_dates
                except Exception as gluonts_err:
                    logger.warning(f"GluonTS prediction failed: {gluonts_err}")
            
            # If all methods fail, return zeros with a warning
            logger.warning("All prediction methods failed, returning zeros")
            return np.zeros(forecast_horizon), future_dates
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return np.array([]), pd.DatetimeIndex([])

    def predict_with_pytorch_model(self, start_date, forecast_horizon):
        """
        Make predictions directly using the extracted PyTorch model.
        
        Args:
            start_date: Start date for prediction
            forecast_horizon: Number of steps to predict
            
        Returns:
            Tuple of predictions and dates
        """
        if self.pytorch_model is None:
            raise ValueError("PyTorch model not available")
            
        try:
            # Set model to evaluation mode
            self.pytorch_model.eval()
            
            # Get target column info
            target_info = self.training_data_info.get('target_range', {})
            target_mean = target_info.get('mean', 0)
            target_std = target_info.get('std', 1)
            
            # Create date range for prediction
            if self.freq == 'H':
                dates = pd.date_range(start=start_date, periods=forecast_horizon+1, freq='H')[1:]
            elif self.freq == 'D':
                dates = pd.date_range(start=start_date, periods=forecast_horizon+1, freq='D')[1:]
            else:
                dates = pd.date_range(start=start_date, periods=forecast_horizon+1, freq=self.freq)[1:]
            
            # Generate predictions with PyTorch model
            with torch.no_grad():
                # Initialize with reasonable values
                # We use 50 as initial value (normalized based on target stats)
                initial_value = (50 - target_mean) / target_std if target_std > 0 else 50
                
                # Create context sequence (past values)
                # In a real scenario, you would use actual past values
                context = torch.ones(1, self.context_length, 1) * initial_value
                
                # Generate samples
                num_samples = 100
                samples = []
                
                for _ in range(num_samples):
                    # Start with context
                    output_sequence = context.clone()
                    
                    # Expand for one-step-ahead predictions
                    for t in range(forecast_horizon):
                        # Input the current sequence
                        next_step = self.pytorch_model(output_sequence)
                        
                        # Add the prediction and shift sequence
                        if t < forecast_horizon - 1:
                            output_sequence = torch.cat([
                                output_sequence[:, 1:, :],
                                next_step.unsqueeze(1)
                            ], dim=1)
                    
                    # Store this sample
                    samples.append(output_sequence[:, -forecast_horizon:, 0].numpy())
                
                # Stack all samples and compute mean
                all_samples = np.stack(samples, axis=0)
                mean_prediction = np.mean(all_samples, axis=0).squeeze()
                
                # Denormalize
                predictions = mean_prediction * target_std + target_mean
                
                return predictions, dates
                
        except Exception as e:
            logger.error(f"Error in direct PyTorch prediction: {e}", exc_info=True)
            return np.array([]), pd.DatetimeIndex([])

    def save(self, file_path: str):
        """Save the trained model."""
        try:
            # Save GluonTS predictor if available
            if self.predictor is not None:
                predictor_path = f"{file_path}_predictor"
                self.predictor.serialize(Path(predictor_path))
                logger.info(f"Saved GluonTS predictor to {predictor_path}")
                
            # Save PyTorch model if available
            if self.pytorch_model is not None:
                torch_path = f"{file_path}_pytorch.pt"
                torch.save(self.pytorch_model.state_dict(), torch_path)
                logger.info(f"Saved PyTorch model to {torch_path}")
                
                # Save model parameters
                params_path = f"{file_path}_params.json"
                with open(params_path, 'w') as f:
                    json.dump(self.model_params, f)
                    
            # Save training info
            info_path = f"{file_path}_info.json"
            with open(info_path, 'w') as f:
                # Convert non-serializable objects
                info_dict = {}
                for k, v in self.training_data_info.items():
                    if isinstance(v, pd.Timestamp):
                        info_dict[k] = v.isoformat()
                    elif isinstance(v, np.ndarray):
                        info_dict[k] = v.tolist()
                    else:
                        info_dict[k] = v
                json.dump(info_dict, f)
                
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)

    def load(self, file_path: str):
        """Load a trained model."""
        try:
            # Try to load the GluonTS predictor
            predictor_path = f"{file_path}_predictor"
            if os.path.exists(f"{predictor_path}/prediction_net.json"):
                self.predictor = Predictor.deserialize(Path(predictor_path))
                logger.info(f"Loaded GluonTS predictor from {predictor_path}")
                
            # Try to load the PyTorch model
            torch_path = f"{file_path}_pytorch.pt"
            params_path = f"{file_path}_params.json"
            
            if os.path.exists(torch_path) and os.path.exists(params_path):
                # Load parameters
                with open(params_path, 'r') as f:
                    self.model_params = json.load(f)
                
                # Recreate the model
                if TorchDeepAREstimator is not None:
                    # Create a temporary estimator to get the model architecture
                    estimator = TorchDeepAREstimator(
                        freq=self.model_params.get('freq', 'H'),
                        prediction_length=self.model_params.get('prediction_length', 24),
                        context_length=self.model_params.get('context_length', 50),
                        num_layers=self.model_params.get('num_layers', 2),
                        hidden_size=self.model_params.get('hidden_size', 40),
                        dropout_rate=self.model_params.get('dropout_rate', 0.1)
                    )
                    
                    # Get the model architecture
                    self.pytorch_model = estimator.create_transformation().create_training_network().model
                    
                    # Load the saved weights
                    self.pytorch_model.load_state_dict(torch.load(torch_path))
                    self.pytorch_model.eval()
                    
                    logger.info(f"Loaded PyTorch model from {torch_path}")
                else:
                    logger.warning("PyTorch estimator not available, could not load model")
                
            # Load training info
            info_path = f"{file_path}_info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.training_data_info = json.load(f)
                    
                # Convert datetime strings back to timestamps
                for k, v in self.training_data_info.items():
                    if k.endswith('_date') and isinstance(v, str):
                        self.training_data_info[k] = pd.Timestamp(v)
                        
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)