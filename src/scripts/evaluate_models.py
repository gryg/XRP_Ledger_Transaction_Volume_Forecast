#!/usr/bin/env python3
"""
Evaluate trained forecasting models and generate forecasts.

Usage:
    python evaluate_models.py --model_dir ./models --data_dir ./output --output_dir ./evaluation
"""

import os
import argparse
import sys
import logging
import pandas as pd
from datetime import datetime

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import Visualizer
from src.utils.helpers import check_dependencies, generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models(model_dir, freq):
    """Load all available models for the given frequency."""
    models = {}
    
    # Check for XGBoost model
    xgb_path = os.path.join(model_dir, freq, 'xgboost', 'xgboost_model.pkl')
    if os.path.exists(xgb_path):
        try:
            from src.models.xgboost_model import XGBoostModel
            model = XGBoostModel(output_dir=os.path.join(model_dir, freq, 'xgboost'))
            model.load(os.path.join(model_dir, freq, 'xgboost', 'xgboost_model'))
            models['xgboost'] = model
            logger.info(f"Loaded XGBoost model from {xgb_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
    
    # Check for LSTM model
    lstm_path = os.path.join(model_dir, freq, 'lstm', 'lstm_model.keras')
    if os.path.exists(lstm_path):
        try:
            from src.models.lstm_model import LSTMModel
            model = LSTMModel(output_dir=os.path.join(model_dir, freq, 'lstm'))
            model.load(os.path.join(model_dir, freq, 'lstm', 'lstm_model'))
            models['lstm'] = model
            logger.info(f"Loaded LSTM model from {lstm_path}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
    
    # Check for DeepAR model
    deepar_path = os.path.join(model_dir, 'deepar', freq, 'models')
    if os.path.exists(deepar_path):
        try:
            from src.models.deepar_model import DeepARModel
            model = DeepARModel(
                freq='H' if freq == 'hourly' else 'D',
                output_dir=os.path.join(model_dir, 'deepar', freq)
            )
            model.load(os.path.join(deepar_path, f"deepar_{'h' if freq == 'hourly' else 'd'}_model"))
            models['deepar'] = model
            logger.info(f"Loaded DeepAR model from {deepar_path}")
        except Exception as e:
            logger.error(f"Failed to load DeepAR model: {e}")
    
    # Check for Prophet model
    prophet_path = os.path.join(model_dir, 'prophet', freq, 'models', 'prophet_model.pkl')
    if os.path.exists(prophet_path):
        try:
            from src.models.prophet_model import ProphetModel
            model = ProphetModel(output_dir=os.path.join(model_dir, 'prophet', freq))
            model.load(os.path.join(model_dir, 'prophet', freq, 'models', 'prophet_model'))
            models['prophet'] = model
            logger.info(f"Loaded Prophet model from {prophet_path}")
        except Exception as e:
            logger.error(f"Failed to load Prophet model: {e}")
    
    return models


def evaluate_models(models, data_path, output_dir, forecast_horizon):
    """Evaluate models and generate forecasts."""
    # Create evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Evaluate models
    evaluation_results = evaluator.evaluate_models(
        models=models,
        data_path=data_path,
        prediction_length=forecast_horizon
    )
    
    # Generate future forecasts
    future_results = evaluator.forecast_future(
        models=models,
        data_path=data_path,
        forecast_horizon=forecast_horizon
    )
    
    return {
        'evaluation': evaluation_results,
        'forecasts': future_results
    }


def create_visualizations(results, data_path, output_dir):
    """Create visualizations of evaluation results."""
    # Create visualizer
    visualizer = Visualizer(output_dir=output_dir)
    
    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Plot time series
    visualizer.plot_time_series(
        data=data,
        column='tx_count',
        title='XRP Transaction Volume',
        filename='time_series.png'
    )
    
    # Plot model comparison if metrics available
    if 'metrics' in results['evaluation']:
        visualizer.plot_model_comparison(
            metrics=results['evaluation']['metrics'],
            metric='RMSE',
            title='Model Comparison',
            filename='model_comparison_rmse.png'
        )
        
        visualizer.plot_model_comparison(
            metrics=results['evaluation']['metrics'],
            metric='MAE',
            title='Model Comparison',
            filename='model_comparison_mae.png'
        )
    
    # Plot future forecasts if available
    if 'forecast_df' in results['forecasts']:
        visualizer.plot_future_forecast(
            forecast=results['forecasts']['forecast_df'],
            historical_data=data,
            title='Future Transaction Volume Forecast',
            filename='future_forecast.png'
        )


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained models and generate forecasts')
    parser.add_argument('--model_dir', default='./models', help='Directory containing trained models')
    parser.add_argument('--data_dir', default='./output', help='Directory containing processed data')
    parser.add_argument('--output_dir', default='./evaluation', help='Directory to save evaluation results')
    parser.add_argument('--hourly_file', default='hourly.csv', help='Filename for hourly data')
    parser.add_argument('--daily_file', default='daily.csv', help='Filename for daily data')
    parser.add_argument('--hourly_forecast', type=int, default=24, help='How many hours to forecast')
    parser.add_argument('--daily_forecast', type=int, default=7, help='How many days to forecast')
    parser.add_argument('--generate_report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Log start time
    logger.info(f"Starting model evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    hourly_output_dir = os.path.join(args.output_dir, 'hourly')
    daily_output_dir = os.path.join(args.output_dir, 'daily')
    os.makedirs(hourly_output_dir, exist_ok=True)
    os.makedirs(daily_output_dir, exist_ok=True)
    
    # Define data paths
    hourly_data_path = os.path.join(args.data_dir, args.hourly_file)
    daily_data_path = os.path.join(args.data_dir, args.daily_file)
    
    results = {}
    
    # Evaluate hourly models
    if os.path.exists(hourly_data_path):
        logger.info("=" * 80)
        logger.info("Evaluating hourly models")
        logger.info("=" * 80)
        
        # Load models
        hourly_models = load_models(args.model_dir, 'hourly')
        
        if hourly_models:
            # Evaluate models
            hourly_results = evaluate_models(
                models=hourly_models,
                data_path=hourly_data_path,
                output_dir=hourly_output_dir,
                forecast_horizon=args.hourly_forecast
            )
            
            # Create visualizations
            create_visualizations(
                results=hourly_results,
                data_path=hourly_data_path,
                output_dir=hourly_output_dir
            )
            
            # Store results
            results['hourly'] = hourly_results
        else:
            logger.warning("No hourly models found")
    else:
        logger.error(f"Hourly data file not found: {hourly_data_path}")
    
    # Evaluate daily models
    if os.path.exists(daily_data_path):
        logger.info("=" * 80)
        logger.info("Evaluating daily models")
        logger.info("=" * 80)
        
        # Load models
        daily_models = load_models(args.model_dir, 'daily')
        
        if daily_models:
            # Evaluate models
            daily_results = evaluate_models(
                models=daily_models,
                data_path=daily_data_path,
                output_dir=daily_output_dir,
                forecast_horizon=args.daily_forecast
            )
            
            # Create visualizations
            create_visualizations(
                results=daily_results,
                data_path=daily_data_path,
                output_dir=daily_output_dir
            )
            
            # Store results
            results['daily'] = daily_results
        else:
            logger.warning("No daily models found")
    else:
        logger.error(f"Daily data file not found: {daily_data_path}")
    
    # Generate HTML report if requested
    if args.generate_report:
        logger.info("Generating HTML report")
        
        report_path = os.path.join(args.output_dir, 'evaluation_report.html')
        generate_report(
            args.output_dir,
            report_path,
            include_deepar=dependencies.get('deepar', False),
            include_prophet=dependencies.get('prophet', False)
        )
        logger.info(f"Report generated at {report_path}")
    
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()