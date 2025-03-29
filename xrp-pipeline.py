#!/usr/bin/env python3
"""
XRP Transaction Volume Forecasting Pipeline

This script provides an end-to-end pipeline for:
1. Processing raw XRP ledger data
2. Creating time series features
3. Training multiple forecasting models
4. Evaluating model performance
5. Generating forecasts

Usage:
    python xrp_pipeline.py --data_path /path/to/xrp_data.csv --output_dir ./xrp_results
"""

import os
import argparse
import subprocess
import time
from datetime import datetime

def run_step(script_path, args, step_name):
    """
    Run a script with given arguments
    
    Args:
        script_path: Path to the script
        args: List of arguments
        step_name: Name of the step for logging
    
    Returns:
        Success status (True/False)
    """
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}")
    
    # Construct command
    cmd = ['python', script_path] + args
    print(f"Running: {' '.join(cmd)}")
    
    # Execute command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    # Check result
    if result.returncode == 0:
        print(f"✓ {step_name} completed successfully in {end_time - start_time:.2f} seconds")
        print(result.stdout)
        return True
    else:
        print(f"✗ {step_name} failed with exit code {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        return False

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
    parser.add_argument('--hourly_forecast', type=int, default=24, help='Hours to forecast ahead')
    parser.add_argument('--daily_forecast', type=int, default=7, help='Days to forecast ahead')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up log file
    log_file = os.path.join(args.output_dir, f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Create paths for intermediate files
    processed_dir = os.path.join(args.output_dir, 'processed_data')
    model_dir = os.path.join(args.output_dir, 'models')
    forecast_dir = os.path.join(args.output_dir, 'forecasts')
    
    # Create directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(forecast_dir, exist_ok=True)
    
    # Log start time
    start_time = time.time()
    print(f"Starting XRP Forecasting Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input data: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Set up script names - make sure these match your actual file names
    data_processor_script = 'xrp-data-processor.py'
    model_training_script = 'xrp-ml-forecasting.py'
    forecast_script = 'xrp-forecast-deployment.py'
    
    # STEP 1: Data Processing
    if not args.skip_processing:
        processing_success = run_step(
            data_processor_script,
            [
                '--filepath', args.data_path,
                '--output', processed_dir
            ],
            'Data Processing'
        )
        
        if not processing_success:
            print("Data processing failed. Pipeline stopped.")
            return
    else:
        print("Skipping data processing step...")
    
    # STEP 2: Model Training
    if not args.skip_training:
        training_success = run_step(
            model_training_script,
            [
                '--data_dir', processed_dir,
                '--hourly_file', 'xgboost_hourly.csv',
                '--daily_file', 'xgboost_daily.csv',
                '--forecast_horizon_hourly', str(args.hourly_forecast),
                '--forecast_horizon_daily', str(args.daily_forecast)
            ],
            'Model Training'
        )
        
        if not training_success:
            print("Model training failed. Pipeline stopped.")
            return
    else:
        print("Skipping model training step...")
    
    # STEP 3: Model Evaluation & Forecasting
    if not args.skip_evaluation:
        # Adjust the script paths to match your file names
        forecast_script = 'xrp-forecast-deployment.py'
        
        # Evaluate hourly models
        hourly_eval_success = run_step(
            forecast_script,
            [
                '--model_dir', model_dir,
                '--data_file', f"{processed_dir}/hourly_transactions.csv",
                '--output_dir', f"{forecast_dir}/hourly"
            ],
            'Hourly Model Evaluation'
        )
        
        # Evaluate daily models
        daily_eval_success = run_step(
            forecast_script,
            [
                '--model_dir', model_dir,
                '--data_file', f"{processed_dir}/daily_transactions.csv",
                '--output_dir', f"{forecast_dir}/daily"
            ],
            'Daily Model Evaluation'
        )
        
        if not hourly_eval_success or not daily_eval_success:
            print("Model evaluation partially failed.")
    else:
        print("Skipping model evaluation step...")
    
    # Calculate total run time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Pipeline completed in {total_time:.2f} seconds")
    print(f"Output files are located in: {args.output_dir}")
    print(f"- Processed data: {processed_dir}")
    print(f"- Trained models: {model_dir}")
    print(f"- Forecasts: {forecast_dir}")
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'pipeline_report.html')
    generate_report(args.output_dir, report_path)
    print(f"- HTML report: {report_path}")

def generate_report(output_dir, report_path):
    """
    Generate an HTML report of pipeline results
    
    Args:
        output_dir: Root directory with all outputs
        report_path: Path to save the HTML report
    """
    processed_dir = os.path.join(output_dir, 'processed_data')
    forecast_dir = os.path.join(output_dir, 'forecasts')
    
    # Get current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create HTML report - with simple string concatenation to avoid formatting issues
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XRP Transaction Volume Forecasting Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333366; }}
            .section {{ margin-top: 30px; }}
            .chart {{ margin: 20px 0; max-width: 800px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>XRP Transaction Volume Forecasting Report</h1>
        <p>Generated on: {current_time}</p>
        
        <div class="section">
            <h2>Data Overview</h2>
            <div class="chart">
                <h3>Daily Transaction Volume</h3>
                <img src="processed_data/daily_transactions.png" alt="Daily Transactions" style="max-width: 100%;">
            </div>
            
            <div class="chart">
                <h3>Transaction Volume by Day of Week</h3>
                <img src="processed_data/tx_by_day_of_week.png" alt="Transactions by Day of Week" style="max-width: 100%;">
            </div>
            
            <div class="chart">
                <h3>Transaction Volume by Hour of Day</h3>
                <img src="processed_data/tx_by_hour.png" alt="Transactions by Hour" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="section">
            <h2>Model Evaluation</h2>
            
            <h3>Hourly Forecasting</h3>
            <div class="chart">
                <img src="forecasts/hourly/forecast_comparison.png" alt="Hourly Forecast Comparison" style="max-width: 100%;">
            </div>
            
            <h3>Daily Forecasting</h3>
            <div class="chart">
                <img src="forecasts/daily/forecast_comparison.png" alt="Daily Forecast Comparison" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="section">
            <h2>Future Forecasts</h2>
            
            <h3>Hourly Future Forecast</h3>
            <div class="chart">
                <img src="forecasts/hourly/future_forecast.png" alt="Hourly Future Forecast" style="max-width: 100%;">
            </div>
            
            <h3>Daily Future Forecast</h3>
            <div class="chart">
                <img src="forecasts/daily/future_forecast.png" alt="Daily Future Forecast" style="max-width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {report_path}")
    
    print(f"HTML report generated at {report_path}")
    
    print(f"HTML report generated at {report_path}")

if __name__ == "__main__":
    main()