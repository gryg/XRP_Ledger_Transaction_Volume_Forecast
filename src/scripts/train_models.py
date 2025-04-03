#!/usr/bin/env python3
"""
Train forecasting models on processed XRP data.

Usage:
    python train_models.py --data_dir ./output --hourly_forecast 24 --daily_forecast 7
"""

import os
import argparse
import sys
import logging
from datetime import datetime

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import check_dependencies
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_xgboost(data_path, output_dir, forecast_horizon, **kwargs):
    """Train XGBoost model."""
    logger.info(f"Training XGBoost model on {data_path}")
    
    # Create model
    model = XGBoostModel(output_dir=output_dir)
    
    # Train model
    results = model.train(
        data=data_path,
        target_column='tx_count',
        forecast_horizon=forecast_horizon,
        **kwargs
    )
    
    logger.info(f"XGBoost training complete. Model saved to {output_dir}")
    return results


def train_lstm(data_path, output_dir, forecast_horizon, **kwargs):
    """Train LSTM model."""
    logger.info(f"Training LSTM model on {data_path}")
    
    # Create model
    model = LSTMModel(output_dir=output_dir)
    
    # Set sequence length based on frequency
    if 'hourly' in data_path:
        seq_length = 48  # 2 days of hourly data
    else:
        seq_length = 30  # 1 month of daily data
    
    # Train model
    results = model.train(
        data=data_path,
        target_column='tx_count',
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        **kwargs
    )
    
    logger.info(f"LSTM training complete. Model saved to {output_dir}")
    return results


def train_deepar(data_path, output_dir, freq, prediction_length, **kwargs):
    """Train DeepAR model."""
    logger.info(f"Training DeepAR model on {data_path}")
    
    try:
        from src.models.deepar_model import DeepARModel
        
        # Create model
        model = DeepARModel(
            freq=freq,
            output_dir=output_dir
        )
        
        # Train model
        results = model.train(
            data=data_path,
            target_column='tx_count',
            prediction_length=prediction_length,
            **kwargs
        )
        
        logger.info(f"DeepAR training complete. Model saved to {output_dir}")
        return results
        
    except ImportError as e:
        logger.error(f"DeepAR dependencies not available: {e}")
        logger.error("Install with: pip install 'xrp-forecasting[deepar]'")
        return None


def train_prophet(data_path, output_dir, prediction_length, **kwargs):
    """Train Prophet model."""
    logger.info(f"Training Prophet model on {data_path}")
    
    try:
        from src.models.prophet_model import ProphetModel
        
        # Create model
        model = ProphetModel(output_dir=output_dir)
        
        # Train model
        results = model.train(
            data=data_path,
            target_column='tx_count',
            prediction_length=prediction_length,
            **kwargs
        )
        
        logger.info(f"Prophet training complete. Model saved to {output_dir}")
        return results
        
    except ImportError as e:
        logger.error(f"Prophet dependencies not available: {e}")
        logger.error("Install with: pip install 'xrp-forecasting[prophet]'")
        return None


def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description='Train models for XRP transaction forecasting')
    parser.add_argument('--data_dir', default='./output', help='Directory with prepared data')
    parser.add_argument('--output_dir', default='./models', help='Directory to save trained models')
    parser.add_argument('--hourly_file', default='hourly.csv', help='Filename for hourly data')
    parser.add_argument('--daily_file', default='daily.csv', help='Filename for daily data')
    parser.add_argument('--hourly_forecast', type=int, default=24, help='How many hours to forecast')
    parser.add_argument('--daily_forecast', type=int, default=7, help='How many days to forecast')
    parser.add_argument('--skip_xgboost', action='store_true', help='Skip XGBoost training')
    parser.add_argument('--skip_lstm', action='store_true', help='Skip LSTM training')
    parser.add_argument('--skip_deepar', action='store_true', help='Skip DeepAR training')
    parser.add_argument('--skip_prophet', action='store_true', help='Skip Prophet training')
    parser.add_argument('--deepar_epochs', type=int, default=100, help='Number of epochs for DeepAR')
    parser.add_argument('--lstm_epochs', type=int, default=100, help='Number of epochs for LSTM')
    parser.add_argument('--enable_cross_val', action='store_true', help='Enable cross validation for Prophet')
    
    args = parser.parse_args()
    
    # Log start time
    logger.info(f"Starting model training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths for each frequency
    hourly_data_path = os.path.join(args.data_dir, args.hourly_file)
    daily_data_path = os.path.join(args.data_dir, args.daily_file)
    
    # Hourly data model training
    logger.info("=" * 80)
    logger.info("Training models on hourly data")
    logger.info("=" * 80)
    
    hourly_output_dir = os.path.join(args.output_dir, 'hourly')
    os.makedirs(hourly_output_dir, exist_ok=True)
    
    # Check if hourly data exists
    if os.path.exists(hourly_data_path):
        # Train XGBoost
        if not args.skip_xgboost:
            hourly_xgb_dir = os.path.join(hourly_output_dir, 'xgboost')
            os.makedirs(hourly_xgb_dir, exist_ok=True)
            
            xgb_results = train_xgboost(
                data_path=hourly_data_path,
                output_dir=hourly_xgb_dir,
                forecast_horizon=args.hourly_forecast
            )
        else:
            logger.info("Skipping XGBoost training (--skip_xgboost flag set)")
        
        # Train LSTM
        if not args.skip_lstm and dependencies.get('basic', False):
            hourly_lstm_dir = os.path.join(hourly_output_dir, 'lstm')
            os.makedirs(hourly_lstm_dir, exist_ok=True)
            
            lstm_results = train_lstm(
                data_path=hourly_data_path,
                output_dir=hourly_lstm_dir,
                forecast_horizon=args.hourly_forecast,
                epochs=args.lstm_epochs
            )
        else:
            if args.skip_lstm:
                logger.info("Skipping LSTM training (--skip_lstm flag set)")
            else:
                logger.error("TensorFlow dependencies not available, skipping LSTM training")
        
        # Train DeepAR
        if not args.skip_deepar and dependencies.get('deepar', False):
            hourly_deepar_dir = os.path.join(args.output_dir, 'deepar', 'hourly')
            os.makedirs(hourly_deepar_dir, exist_ok=True)
            
            deepar_results = train_deepar(
                data_path=hourly_data_path,
                output_dir=hourly_deepar_dir,
                freq='H',
                prediction_length=args.hourly_forecast,
                epochs=args.deepar_epochs
            )
        else:
            if args.skip_deepar:
                logger.info("Skipping DeepAR training (--skip_deepar flag set)")
            else:
                logger.warning("DeepAR dependencies not available, skipping DeepAR training")
        
        # Train Prophet
        if not args.skip_prophet and dependencies.get('prophet', False):
            hourly_prophet_dir = os.path.join(args.output_dir, 'prophet', 'hourly')
            os.makedirs(hourly_prophet_dir, exist_ok=True)
            
            prophet_results = train_prophet(
                data_path=hourly_data_path,
                output_dir=hourly_prophet_dir,
                prediction_length=args.hourly_forecast,
                cross_val=args.enable_cross_val
            )
        else:
            if args.skip_prophet:
                logger.info("Skipping Prophet training (--skip_prophet flag set)")
            else:
                logger.warning("Prophet dependencies not available, skipping Prophet training")
    else:
        logger.error(f"Hourly data file not found: {hourly_data_path}")
    
    # Daily data model training
    logger.info("=" * 80)
    logger.info("Training models on daily data")
    logger.info("=" * 80)
    
    daily_output_dir = os.path.join(args.output_dir, 'daily')
    os.makedirs(daily_output_dir, exist_ok=True)
    
    # Check if daily data exists
    if os.path.exists(daily_data_path):
        # Train XGBoost
        if not args.skip_xgboost:
            daily_xgb_dir = os.path.join(daily_output_dir, 'xgboost')
            os.makedirs(daily_xgb_dir, exist_ok=True)
            
            xgb_results = train_xgboost(
                data_path=daily_data_path,
                output_dir=daily_xgb_dir,
                forecast_horizon=args.daily_forecast
            )
        else:
            logger.info("Skipping XGBoost training (--skip_xgboost flag set)")
        
        # Train LSTM
        if not args.skip_lstm and dependencies.get('basic', False):
            daily_lstm_dir = os.path.join(daily_output_dir, 'lstm')
            os.makedirs(daily_lstm_dir, exist_ok=True)
            
            lstm_results = train_lstm(
                data_path=daily_data_path,
                output_dir=daily_lstm_dir,
                forecast_horizon=args.daily_forecast,
                epochs=args.lstm_epochs
            )
        else:
            if args.skip_lstm:
                logger.info("Skipping LSTM training (--skip_lstm flag set)")
            else:
                logger.error("TensorFlow dependencies not available, skipping LSTM training")
        
        # Train DeepAR
        if not args.skip_deepar and dependencies.get('deepar', False):
            daily_deepar_dir = os.path.join(args.output_dir, 'deepar', 'daily')
            os.makedirs(daily_deepar_dir, exist_ok=True)
            
            deepar_results = train_deepar(
                data_path=daily_data_path,
                output_dir=daily_deepar_dir,
                freq='D',
                prediction_length=args.daily_forecast,
                epochs=args.deepar_epochs
            )
        else:
            if args.skip_deepar:
                logger.info("Skipping DeepAR training (--skip_deepar flag set)")
            else:
                logger.warning("DeepAR dependencies not available, skipping DeepAR training")
        
        # Train Prophet
        if not args.skip_prophet and dependencies.get('prophet', False):
            daily_prophet_dir = os.path.join(args.output_dir, 'prophet', 'daily')
            os.makedirs(daily_prophet_dir, exist_ok=True)
            
            prophet_results = train_prophet(
                data_path=daily_data_path,
                output_dir=daily_prophet_dir,
                prediction_length=args.daily_forecast,
                cross_val=args.enable_cross_val
            )
        else:
            if args.skip_prophet:
                logger.info("Skipping Prophet training (--skip_prophet flag set)")
            else:
                logger.warning("Prophet dependencies not available, skipping Prophet training")
    else:
        logger.error(f"Daily data file not found: {daily_data_path}")
    
    logger.info("Model training completed")
    

if __name__ == "__main__":
    main()