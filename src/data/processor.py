"""Data processing module for XRP transaction data."""

import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, Dict, Any, List, Optional
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XRPDataProcessor:
    """Processor for XRP ledger transaction data."""
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the data processor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def process_large_csv(self, filepath: str, chunksize: int = 100000) -> pd.DataFrame:
        """
        Process a large CSV file in chunks to avoid memory issues.
        
        Args:
            filepath: Path to the CSV file
            chunksize: Number of rows to process at once
        
        Returns:
            Processed DataFrame with time-based features
        """
        logger.info(f"Processing {filepath} in chunks of {chunksize} rows")
        
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
        logger.info(f"Processed {len(df)} rows from {filepath}")
        
        return df
    
    def create_time_series_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create time-based features from the processed data.
        
        Args:
            df: DataFrame with LedgerSeq, ClosingTime, and TransSetHash
        
        Returns:
            Tuple of DataFrames with hourly and daily aggregated time series features
        """
        logger.info("Creating time series features")
        
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
        
        logger.info(f"Created hourly features with shape {hourly_df.shape}")
        logger.info(f"Created daily features with shape {daily_df.shape}")
        
        return hourly_df, daily_df
    
    def prepare_for_ml(self, hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for ML models (XGBoost, LSTM, DeepAR, Prophet).
        
        Args:
            hourly_df: Hourly aggregated DataFrame
            daily_df: Daily aggregated DataFrame
            
        Returns:
            Dictionary with prepared datasets
        """
        logger.info("Preparing data for ML models")
        
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
        
        # For DeepAR format
        deepar_hourly = {
            "start": hourly_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            "target": hourly_df['tx_count'].tolist()
        }
        
        deepar_daily = {
            "start": daily_df.index[0].strftime('%Y-%m-%d'),
            "target": daily_df['tx_count'].tolist()
        }
        
        # Save all prepared datasets
        self._save_prepared_data({
            'hourly': hourly_df,
            'daily': daily_df,
            'xgb_hourly': xgb_hourly,
            'xgb_daily': xgb_daily,
            'deepar_hourly': deepar_hourly,
            'deepar_daily': deepar_daily
        })
        
        return {
            'hourly': hourly_df,
            'daily': daily_df, 
            'xgb_hourly': xgb_hourly,
            'xgb_daily': xgb_daily,
            'deepar_hourly': deepar_hourly,
            'deepar_daily': deepar_daily
        }
    
    def _save_prepared_data(self, data_dict: Dict[str, Any]) -> None:
        """
        Save prepared datasets to disk.
        
        Args:
            data_dict: Dictionary of prepared datasets
        """
        # Save dataframes
        for name, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                output_path = os.path.join(self.output_dir, f"{name}.csv")
                df.to_csv(output_path)
                logger.info(f"Saved {name} to {output_path}")
        
        # Save DeepAR JSON format
        for name, data in data_dict.items():
            if name.startswith('deepar') and isinstance(data, dict):
                output_path = os.path.join(self.output_dir, f"{name}.json")
                with open(output_path, 'w') as f:
                    json.dump(data, f)
                logger.info(f"Saved {name} to {output_path}")
    
    def process(self, filepath: str) -> Dict[str, Any]:
        """
        Process the full pipeline from raw data to prepared ML features.
        
        Args:
            filepath: Path to the raw XRP ledger CSV file
            
        Returns:
            Dictionary with prepared datasets
        """
        # Step 1: Process the large CSV file
        df = self.process_large_csv(filepath)
        
        # Step 2: Create time series features
        hourly_df, daily_df = self.create_time_series_features(df)
        
        # Step 3: Analyze data and create visualizations
        self.analyze_and_visualize(hourly_df, daily_df)
        
        # Step 4: Prepare data for ML models
        prepared_data = self.prepare_for_ml(hourly_df, daily_df)
        
        logger.info(f"Processing complete. Results saved to {self.output_dir}")
        return prepared_data
    
    def analyze_and_visualize(self, hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
        """
        Perform basic analysis and create visualizations.
        
        Args:
            hourly_df: Hourly aggregated DataFrame
            daily_df: Daily aggregated DataFrame
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            logger.info("Generating visualizations")
            
            # Basic statistics
            hourly_stats = hourly_df['tx_count'].describe()
            daily_stats = daily_df['tx_count'].describe()
            
            with open(os.path.join(self.output_dir, 'statistics.txt'), 'w') as f:
                f.write("Hourly Transaction Statistics:\n")
                f.write(str(hourly_stats))
                f.write("\n\nDaily Transaction Statistics:\n")
                f.write(str(daily_stats))
            
            # Visualize daily transactions over time
            plt.figure(figsize=(15, 7))
            plt.plot(daily_df.index, daily_df['tx_count'])
            plt.title('XRP Daily Transaction Volume')
            plt.xlabel('Date')
            plt.ylabel('Number of Transactions')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'daily_transactions.png'))
            plt.close()
            
            # Visualize transactions by day of week
            plt.figure(figsize=(10, 6))
            daily_df.groupby('day_of_week')['tx_count'].mean().plot(kind='bar')
            plt.title('Average Transaction Volume by Day of Week')
            plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
            plt.ylabel('Average Transactions')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tx_by_day_of_week.png'))
            plt.close()
            
            # Visualize transactions by hour of day
            plt.figure(figsize=(12, 6))
            hourly_df.groupby('hour')['tx_count'].mean().plot(kind='bar')
            plt.title('Average Transaction Volume by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Transactions')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tx_by_hour.png'))
            plt.close()
            
            # Check for seasonality with decomposition
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
                plt.savefig(os.path.join(self.output_dir, 'seasonal_decomposition.png'))
                plt.close()
            
            logger.info(f"Visualizations saved to {self.output_dir}")
        
        except ImportError as e:
            logger.warning(f"Visualization libraries not available: {e}")
            logger.warning("Skipping visualization generation")