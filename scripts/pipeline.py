#!/usr/bin/env python3
"""
XRP Transaction Volume Forecasting Pipeline

This script provides an end-to-end pipeline for:
1. Processing raw XRP ledger data
2. Creating time series features
3. Training multiple forecasting models (XGBoost, LSTM, DeepAR, Prophet)
4. Evaluating model performance
5. Generating forecasts

Usage:
    python pipeline.py --data_path /path/to/xrp_data.csv --output_dir ./xrp_results
"""

import os
import argparse
import time
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import XRPDataProcessor
from src.utils.helpers import check_dependencies, generate_report
from src.models.deepar_model import DeepARModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_pipeline(args):
    """
    Set up the pipeline structure.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with directories
    """
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories
    directories = {
        'processed_data': os.path.join(args.output_dir, 'processed_data'),
        'models': os.path.join(args.output_dir, 'models'),
        'forecasts': os.path.join(args.output_dir, 'forecasts'),
        'deepar': os.path.join(args.output_dir, 'deepar'),
        'prophet': os.path.join(args.output_dir, 'prophet'),
    }
    
    # Create each directory
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    # Set up log file
    log_file = os.path.join(args.output_dir, f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add the file handler to the root logger
    logging.getLogger().addHandler(file_handler)
    
    return directories


def run_data_processing(args, directories):
    """
    Run data processing step.
    
    Args:
        args: Command line arguments
        directories: Dictionary with output directories
        
    Returns:
        Dictionary with processed data
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Data Processing")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Initialize processor
    processor = XRPDataProcessor(output_dir=directories['processed_data'])
    
    try:
        # Process data
        processed_data = processor.process(args.data_path)
        
        process_time = time.time() - start_time
        logger.info(f"Data processing completed in {process_time:.2f} seconds")
        logger.info(f"Processed data saved to {directories['processed_data']}")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        logger.exception("Exception details:")
        return None


def run_model_training(args, directories, processed_data):
    """
    Run model training step.
    
    Args:
        args: Command line arguments
        directories: Dictionary with output directories
        processed_data: Dictionary with processed data
        
    Returns:
        Dictionary with trained models and metrics
    """
    if processed_data is None:
        logger.error("Skipping model training due to failed data processing")
        return None
    
    logger.info("=" * 80)
    logger.info("STEP 2: Model Training")
    logger.info("=" * 80)
    
    # Check if ML modules are available
    dependencies = check_dependencies()
    if not dependencies['basic']:
        logger.error("Basic ML dependencies not available. Cannot train models.")
        return None
    
    results = {}
    
    # Determine which models to run
    # If use_all is set, run all models
    # Otherwise, only run models that are explicitly selected with use_* flags
    # If no use_* flags are set, fall back to legacy behavior using skip_* flags
    run_all = args.use_all
    explicit_selection = args.use_deepar or args.use_prophet or args.use_xgboost or args.use_lstm
    
    # Determine which models to run
    run_deepar = run_all or args.use_deepar or (not explicit_selection and not args.skip_deepar)
    run_prophet = run_all or args.use_prophet or (not explicit_selection and not args.skip_prophet)
    run_xgboost = run_all or args.use_xgboost or (not explicit_selection)
    run_lstm = run_all or args.use_lstm or (not explicit_selection)
    
    # Setup for both hourly and daily forecasting
    for freq in ['hourly', 'daily']:
        logger.info(f"Training models for {freq} data")
        
        # Forecasting horizon (24 hours or 7 days)
        prediction_length = 24 if freq == 'hourly' else 7
        
        # Create frequency-specific output directories
        freq_forecast_dir = os.path.join(directories['forecasts'], freq)
        os.makedirs(freq_forecast_dir, exist_ok=True)
        
        # Get the appropriate dataset
        data = processed_data.get(f'xgb_{freq}')
        if data is None:
            logger.error(f"No {freq} data available for training")
            continue
        
        # Train DeepAR model if selected and dependencies are available
        if run_deepar and dependencies['deepar']:
            try:
                logger.info(f"Training DeepAR model for {freq} data")
                deepar_output_dir = os.path.join(directories['deepar'], freq)
                
                deepar_model = DeepARModel(
                    freq='H' if freq == 'hourly' else 'D', 
                    output_dir=deepar_output_dir
                )
                
                deepar_results = deepar_model.train(
                    data=data,
                    target_column='tx_count',
                    prediction_length=prediction_length,
                    epochs=args.deepar_epochs
                )
                
                # Store results
                results[f'deepar_{freq}'] = deepar_results
                logger.info(f"DeepAR {freq} training complete")
                
            except Exception as e:
                logger.error(f"DeepAR {freq} training failed: {e}")
                logger.exception("Exception details:")
        else:
            if not run_deepar:
                logger.info("Skipping DeepAR training (not selected)")
            elif not dependencies['deepar']:
                logger.warning("DeepAR dependencies not available, skipping DeepAR training")
        
        # Train Prophet model if selected and dependencies are available
        if run_prophet and dependencies['prophet']:
            try:
                logger.info(f"Training Prophet model for {freq} data")
                # Import here to avoid dependency issues if not installed
                from src.models.prophet_model import ProphetModel
                
                prophet_output_dir = os.path.join(directories['prophet'], freq)
                prophet_model = ProphetModel(output_dir=prophet_output_dir)
                
                prophet_results = prophet_model.train(
                    data=data,
                    target_column='tx_count',
                    prediction_length=prediction_length,
                    cross_val=args.enable_cross_val
                )
                
                # Store results
                results[f'prophet_{freq}'] = prophet_results
                logger.info(f"Prophet {freq} training complete")
                
            except Exception as e:
                logger.error(f"Prophet {freq} training failed: {e}")
                logger.exception("Exception details:")
        else:
            if not run_prophet:
                logger.info("Skipping Prophet training (not selected)")
            elif not dependencies['prophet']:
                logger.warning("Prophet dependencies not available, skipping Prophet training")
                
        # Train traditional ML models
        if run_xgboost or run_lstm:
            logger.info(f"Training traditional ML models for {freq} data")
            
            # Train XGBoost model if selected
            if run_xgboost:
                try:
                    # Import here to maintain modularity
                    from src.models.xgboost_model import XGBoostModel
                    
                    # Train XGBoost model
                    xgb_output_dir = os.path.join(directories['models'], freq)
                    os.makedirs(xgb_output_dir, exist_ok=True)
                    
                    xgb_model = XGBoostModel(output_dir=xgb_output_dir)
                    
                    xgb_results = xgb_model.train(
                        data=data,
                        target_column='tx_count',
                        forecast_horizon=prediction_length
                    )
                    
                    # Store results
                    results[f'xgboost_{freq}'] = xgb_results
                    logger.info(f"XGBoost {freq} training complete")
                except Exception as e:
                    logger.error(f"XGBoost model training for {freq} failed: {e}")
                    logger.exception("Exception details:")
            else:
                logger.info("Skipping XGBoost training (not selected)")
            
            # Train LSTM model if selected
            if run_lstm:
                try:
                    # Import here to maintain modularity
                    from src.models.lstm_model import LSTMModel
                    
                    # Train LSTM model
                    lstm_output_dir = os.path.join(directories['models'], freq)
                    lstm_model = LSTMModel(output_dir=lstm_output_dir)
                    
                    # Sequence length based on frequency
                    seq_length = 48 if freq == 'hourly' else 30
                    
                    lstm_results = lstm_model.train(
                        data=data,
                        target_column='tx_count',
                        seq_length=seq_length,
                        forecast_horizon=prediction_length
                    )
                    
                    # Store results
                    results[f'lstm_{freq}'] = lstm_results
                    logger.info(f"LSTM {freq} training complete")
                except Exception as e:
                    logger.error(f"LSTM model training for {freq} failed: {e}")
                    logger.exception("Exception details:")
            else:
                logger.info("Skipping LSTM training (not selected)")
        else:
            logger.info("Skipping traditional ML models (not selected)")
    
    return results


def run_model_evaluation(args, directories, training_results):
    """
    Run model evaluation and generate forecasts.
    
    Args:
        args: Command line arguments
        directories: Dictionary with output directories
        training_results: Dictionary with training results
        
    Returns:
        Dictionary with evaluation results
    """
    if training_results is None:
        logger.error("Skipping model evaluation due to failed model training")
        return None
    
    logger.info("=" * 80)
    logger.info("STEP 3: Model Evaluation and Forecasting")
    logger.info("=" * 80)
    
    # Import necessary modules
    from src.evaluation.evaluator import ModelEvaluator
    
    evaluation_results = {}
    
    # Evaluate models for both hourly and daily forecasting
    for freq in ['hourly', 'daily']:
        logger.info(f"Evaluating models for {freq} data")
        
        # Create frequency-specific output directory
        freq_forecast_dir = os.path.join(directories['forecasts'], freq)
        os.makedirs(freq_forecast_dir, exist_ok=True)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(output_dir=freq_forecast_dir)
        
        # Collect models for this frequency
        freq_models = {k: v for k, v in training_results.items() if k.endswith(freq)}
        
        if not freq_models:
            logger.warning(f"No trained models found for {freq} data")
            continue
        
        try:
            # Run evaluation
            eval_results = evaluator.evaluate_models(
                models=freq_models,
                data_path=os.path.join(directories['processed_data'], f"{freq}.csv"),
                prediction_length=24 if freq == 'hourly' else 7
            )
            
            # Generate future forecasts
            future_results = evaluator.forecast_future(
                models=freq_models,
                data_path=os.path.join(directories['processed_data'], f"{freq}.csv"),
                forecast_horizon=24 if freq == 'hourly' else 7
            )
            
            # Store results
            evaluation_results[freq] = {
                'evaluation': eval_results,
                'forecasts': future_results
            }
            
            logger.info(f"{freq.capitalize()} model evaluation complete")
            
        except Exception as e:
            logger.error(f"Model evaluation for {freq} failed: {e}")
            logger.exception("Exception details:")
    
    return evaluation_results


def main():
    """
    Main pipeline function
    """
    parser = argparse.ArgumentParser(description='XRP Transaction Volume Forecasting Pipeline')
    parser.add_argument('--data_path', required=True, help='Path to raw XRP ledger CSV file')
    parser.add_argument('--output_dir', default='./xrp_results', help='Output directory for all results')
    parser.add_argument('--skip_processing', action='store_true', help='Skip data processing step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation step')
    parser.add_argument('--skip_deepar', action='store_true', help='Skip DeepAR forecasting step')
    parser.add_argument('--skip_prophet', action='store_true', help='Skip Prophet forecasting step')
    parser.add_argument('--hourly_forecast', type=int, default=24, help='Hours to forecast ahead')
    parser.add_argument('--daily_forecast', type=int, default=7, help='Days to forecast ahead')
    parser.add_argument('--deepar_epochs', type=int, default=100, help='Number of epochs for DeepAR training')
    parser.add_argument('--enable_cross_val', action='store_true', help='Enable cross validation for Prophet')
    # Add individual model selection flags  
    parser.add_argument('--use_deepar', action='store_true', help='Run DeepAR forecasting model')
    parser.add_argument('--use_prophet', action='store_true', help='Run Prophet forecasting model')
    parser.add_argument('--use_xgboost', action='store_true', help='Run XGBoost forecasting model')
    parser.add_argument('--use_lstm', action='store_true', help='Run LSTM forecasting model')
    parser.add_argument('--use_all', action='store_true', help='Run all available models')
    
    args = parser.parse_args()
    
    # Start time for the entire pipeline
    pipeline_start_time = time.time()
    
    # Log start time
    logger.info(f"Starting XRP Forecasting Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input data: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set up pipeline structure
    directories = setup_pipeline(args)
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Step 1: Data Processing
    processed_data = None
    if not args.skip_processing:
        processed_data = run_data_processing(args, directories)
    else:
        logger.info("Skipping data processing step (--skip_processing flag set)")
        # Try to load processed data
        try:
            from src.data.processor import XRPDataProcessor
            processor = XRPDataProcessor(output_dir=directories['processed_data'])
            
            # Check if processed files exist
            hourly_path = os.path.join(directories['processed_data'], 'hourly.csv')
            daily_path = os.path.join(directories['processed_data'], 'daily.csv')
            
            if os.path.exists(hourly_path) and os.path.exists(daily_path):
                # Load the data
                hourly_df = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
                daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
                
                # Recreate the processed data dictionary
                processed_data = {
                    'hourly': hourly_df,
                    'daily': daily_df
                }
                
                logger.info(f"Loaded processed data from {directories['processed_data']}")
            else:
                logger.error("Could not find processed data files. Please run data processing first.")
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
    
    # Step 2: Model Training
    training_results = None
    if not args.skip_training and processed_data is not None:
        training_results = run_model_training(args, directories, processed_data)
    else:
        if args.skip_training:
            logger.info("Skipping model training step (--skip_training flag set)")
        else:
            logger.error("Cannot train models without processed data")
    
    # Step 3: Model Evaluation
    evaluation_results = None
    if not args.skip_evaluation and training_results is not None:
        evaluation_results = run_model_evaluation(args, directories, training_results)
    else:
        if args.skip_evaluation:
            logger.info("Skipping model evaluation step (--skip_evaluation flag set)")
        else:
            logger.error("Cannot evaluate models without training results")
    
    # Calculate total run time
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    # Generate report
    try:
        report_path = os.path.join(args.output_dir, 'pipeline_report.html')
        generate_report(
            args.output_dir, 
            report_path,
            include_deepar=dependencies['deepar'] and not args.skip_deepar,
            include_prophet=dependencies['prophet'] and not args.skip_prophet
        )
        logger.info(f"Generated HTML report at {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
    
    # Display summary
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Pipeline completed in {total_time:.2f} seconds")
    logger.info(f"Output files are located in: {args.output_dir}")
    logger.info(f"- Processed data: {directories['processed_data']}")
    logger.info(f"- Trained models: {directories['models']}")
    logger.info(f"- Forecasts: {directories['forecasts']}")
    if dependencies['deepar'] and not args.skip_deepar:
        logger.info(f"- DeepAR forecasts: {directories['deepar']}")
    if dependencies['prophet'] and not args.skip_prophet:
        logger.info(f"- Prophet forecasts: {directories['prophet']}")
    logger.info(f"- HTML report: {os.path.join(args.output_dir, 'pipeline_report.html')}")


if __name__ == "__main__":
    main()