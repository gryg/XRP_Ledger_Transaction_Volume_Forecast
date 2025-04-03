#!/usr/bin/env python3
"""
Generate forecasts using trained models.

Usage:
    python generate_forecasts.py --model_dir ./models --data_file ./data/latest.csv --output_dir ./forecasts
"""

import os
import argparse
import sys
import logging
from datetime import datetime

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deployment.forecaster import Forecaster
from src.utils.helpers import check_dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for generating forecasts."""
    parser = argparse.ArgumentParser(description='Generate forecasts using trained models')
    parser.add_argument('--model_dir', default='./models', help='Directory containing trained models')
    parser.add_argument('--data_file', required=True, help='Path to input data CSV')
    parser.add_argument('--output_dir', default='./forecasts', help='Directory to save forecasts')
    parser.add_argument('--forecast_horizon', type=int, default=24, help='Number of steps to forecast ahead')
    parser.add_argument('--model_type', default='all', help='Type of model to use (xgboost, lstm, deepar, prophet, ensemble, all)')
    parser.add_argument('--freq', choices=['hourly', 'daily'], default='hourly', help='Data frequency')
    
    args = parser.parse_args()
    
    # Log start time
    logger.info(f"Starting forecast generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies based on model type
    include_deepar = args.model_type in ['deepar', 'ensemble', 'all']
    include_prophet = args.model_type in ['prophet', 'ensemble', 'all']
    dependencies = check_dependencies(
        include_deepar=include_deepar, 
        include_prophet=include_prophet
    )
    
    # Create forecaster
    forecaster = Forecaster(model_dir=args.model_dir, output_dir=args.output_dir)
    
    # Generate forecasts based on model type
    if args.model_type == 'xgboost' or args.model_type == 'all':
        logger.info("Generating XGBoost forecast")
        xgb_forecast = forecaster.generate_forecast(
            data_path=args.data_file,
            forecast_horizon=args.forecast_horizon,
            model_type='xgboost',
            freq=args.freq
        )
    
    if (args.model_type == 'lstm' or args.model_type == 'all') and dependencies.get('basic', False):
        logger.info("Generating LSTM forecast")
        lstm_forecast = forecaster.generate_forecast(
            data_path=args.data_file,
            forecast_horizon=args.forecast_horizon,
            model_type='lstm',
            freq=args.freq
        )
    
    if (args.model_type == 'deepar' or args.model_type == 'all') and dependencies.get('deepar', False):
        logger.info("Generating DeepAR forecast")
        deepar_forecast = forecaster.generate_forecast(
            data_path=args.data_file,
            forecast_horizon=args.forecast_horizon,
            model_type='deepar',
            freq=args.freq
        )
    
    if (args.model_type == 'prophet' or args.model_type == 'all') and dependencies.get('prophet', False):
        logger.info("Generating Prophet forecast")
        prophet_forecast = forecaster.generate_forecast(
            data_path=args.data_file,
            forecast_horizon=args.forecast_horizon,
            model_type='prophet',
            freq=args.freq
        )
    
    if args.model_type == 'ensemble' or args.model_type == 'all':
        logger.info("Generating ensemble forecast")
        
        # Determine which models to include based on available dependencies
        model_types = ['xgboost']
        
        if dependencies.get('basic', False):
            model_types.append('lstm')
        
        if dependencies.get('deepar', False):
            model_types.append('deepar')
        
        if dependencies.get('prophet', False):
            model_types.append('prophet')
        
        ensemble_forecast = forecaster.generate_ensemble_forecast(
            data_path=args.data_file,
            forecast_horizon=args.forecast_horizon,
            model_types=model_types,
            freq=args.freq
        )
    
    logger.info("Forecast generation completed")


if __name__ == "__main__":
    main()