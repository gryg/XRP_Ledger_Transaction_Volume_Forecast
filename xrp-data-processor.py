import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import gc
from tqdm import tqdm

# File processing for large CSV
def process_large_csv(filepath, chunksize=100000):
    """
    Process a large CSV file in chunks to avoid memory issues.
    
    Args:
        filepath: Path to the CSV file
        chunksize: Number of rows to process at once
    
    Returns:
        Processed DataFrame with time-based features
    """
    # List to store processed chunks
    processed_chunks = []
    
    # Process the file in chunks
    for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize), desc="Processing chunks"):
        # Convert ClosingTime to datetime (assuming it's Unix timestamp)
        chunk['ClosingTime'] = pd.to_datetime(chunk['ClosingTime'], unit='s')
        
        # Extract relevant columns for transaction volume analysis
        processed_chunk = chunk[['LedgerSeq', 'ClosingTime', 'TransSetHash']]
        
        # Store processed chunk
        processed_chunks.append(processed_chunk)
        
        # Force garbage collection to free memory
        gc.collect()
    
    # Combine all processed chunks
    df = pd.concat(processed_chunks, ignore_index=True)
    
    return df

def create_time_series_features(df):
    """
    Create time-based features from the processed data
    
    Args:
        df: DataFrame with LedgerSeq, ClosingTime, and TransSetHash
    
    Returns:
        DataFrame with aggregated time series features
    """
    # Sort by closing time
    df = df.sort_values('ClosingTime')
    
    # Create multiple time-based aggregations for different ML models
    
    # 1. Hourly aggregation
    hourly_df = df.set_index('ClosingTime').groupby(pd.Grouper(freq='H')).count()
    hourly_df = hourly_df.rename(columns={'TransSetHash': 'tx_count'})
    hourly_df = hourly_df[['tx_count']]  # Keep only transaction count
    
    # 2. Daily aggregation
    daily_df = df.set_index('ClosingTime').groupby(pd.Grouper(freq='D')).count()
    daily_df = daily_df.rename(columns={'TransSetHash': 'tx_count'})
    daily_df = daily_df[['tx_count']]  # Keep only transaction count
    
    # Add time-based features that might help with forecasting
    for time_df in [hourly_df, daily_df]:
        # Add hour of day
        time_df['hour'] = time_df.index.hour
        
        # Add day of week
        time_df['day_of_week'] = time_df.index.dayofweek
        
        # Add month
        time_df['month'] = time_df.index.month
        
        # Add quarter
        time_df['quarter'] = time_df.index.quarter
        
        # Add year
        time_df['year'] = time_df.index.year
        
        # Add weekend flag
        time_df['is_weekend'] = (time_df['day_of_week'] >= 5).astype(int)
    
    return hourly_df, daily_df

def analyze_and_visualize(hourly_df, daily_df, output_dir="./output"):
    """
    Perform basic analysis and create visualizations
    
    Args:
        hourly_df: Hourly aggregated DataFrame
        daily_df: Daily aggregated DataFrame
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    print("Hourly Transaction Statistics:")
    print(hourly_df['tx_count'].describe())
    
    print("\nDaily Transaction Statistics:")
    print(daily_df['tx_count'].describe())
    
    # Visualize daily transactions over time
    plt.figure(figsize=(15, 7))
    plt.plot(daily_df.index, daily_df['tx_count'])
    plt.title('XRP Daily Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_transactions.png")
    
    # Visualize transactions by day of week
    plt.figure(figsize=(10, 6))
    daily_df.groupby('day_of_week')['tx_count'].mean().plot(kind='bar')
    plt.title('Average Transaction Volume by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Transactions')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tx_by_day_of_week.png")
    
    # Visualize transactions by hour of day
    plt.figure(figsize=(12, 6))
    hourly_df.groupby('hour')['tx_count'].mean().plot(kind='bar')
    plt.title('Average Transaction Volume by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Transactions')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tx_by_hour.png")
    
    # Check for seasonality with decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Decompose the daily time series (if we have enough data)
    if len(daily_df) >= 2 * 365:  # At least 2 years of data
        decomposition = seasonal_decompose(daily_df['tx_count'], model='additive', period=365)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonality')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residuals')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/seasonal_decomposition.png")
    
    # Save processed data for ML
    hourly_df.to_csv(f"{output_dir}/hourly_transactions.csv")
    daily_df.to_csv(f"{output_dir}/daily_transactions.csv")

def prepare_for_ml(hourly_df, daily_df, output_dir="./output"):
    """
    Prepare data for ML models (DeepAR, XGBoost)
    
    Args:
        hourly_df: Hourly aggregated DataFrame
        daily_df: Daily aggregated DataFrame
        output_dir: Directory to save prepared data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For DeepAR (requires specific format)
    # DeepAR Format for Amazon SageMaker: {"start": ..., "target": [...]}
    
    # 1. Prepare for DeepAR - hourly data
    deepar_hourly = {
        "start": hourly_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        "target": hourly_df['tx_count'].tolist()
    }
    
    # 2. Prepare for DeepAR - daily data
    deepar_daily = {
        "start": daily_df.index[0].strftime('%Y-%m-%d'),
        "target": daily_df['tx_count'].tolist()
    }
    
    # 3. Prepare for XGBoost and general ML - requires explicit features
    # For XGBoost, we'll use a more traditional tabular format with all features
    # We'll also add lag features which are crucial for time series forecasting
    
    # Add lag features (previous days/hours transactions)
    def add_lag_features(df, lags, column='tx_count'):
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        return df
    
    # Add rolling statistics
    def add_rolling_features(df, windows, column='tx_count'):
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        return df
    
    # Hourly data preparation
    xgb_hourly = hourly_df.copy()
    xgb_hourly = add_lag_features(xgb_hourly, [1, 2, 3, 6, 12, 24, 24*7])
    xgb_hourly = add_rolling_features(xgb_hourly, [6, 12, 24, 24*7])
    
    # Daily data preparation
    xgb_daily = daily_df.copy()
    xgb_daily = add_lag_features(xgb_daily, [1, 2, 3, 7, 14, 30])
    xgb_daily = add_rolling_features(xgb_daily, [7, 14, 30, 90])
    
    # Save all prepared datasets
    import json
    
    # Save DeepAR format
    with open(f"{output_dir}/deepar_hourly.json", 'w') as f:
        json.dump(deepar_hourly, f)
    
    with open(f"{output_dir}/deepar_daily.json", 'w') as f:
        json.dump(deepar_daily, f)
    
    # Save XGBoost format
    xgb_hourly.to_csv(f"{output_dir}/xgboost_hourly.csv")
    xgb_daily.to_csv(f"{output_dir}/xgboost_daily.csv")
    
    return {
        'deepar_hourly': deepar_hourly,
        'deepar_daily': deepar_daily,
        'xgb_hourly': xgb_hourly,
        'xgb_daily': xgb_daily
    }

def main(filepath, output_dir="./output"):
    """
    Main function to process the data
    
    Args:
        filepath: Path to the large CSV file
        output_dir: Directory to save output files
    """
    print(f"Processing {filepath}...")
    
    # Step 1: Process the large CSV file
    df = process_large_csv(filepath)
    
    # Step 2: Create time series features
    print("Creating time series features...")
    hourly_df, daily_df = create_time_series_features(df)
    
    # Step 3: Analyze and visualize the data
    print("Analyzing and visualizing data...")
    analyze_and_visualize(hourly_df, daily_df, output_dir)
    
    # Step 4: Prepare data for ML models
    print("Preparing data for ML models...")
    prepared_data = prepare_for_ml(hourly_df, daily_df, output_dir)
    
    print(f"Processing complete. Results saved to {output_dir}")
    return prepared_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process XRP ledger data for ML forecasting')
    parser.add_argument('--filepath', required=True, help='Path to the XRP ledger CSV file')
    parser.add_argument('--output', default='./output', help='Output directory for processed files')
    
    args = parser.parse_args()
    
    main(args.filepath, args.output)