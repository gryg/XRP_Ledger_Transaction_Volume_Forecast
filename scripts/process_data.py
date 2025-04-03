#!/usr/bin/env python3
"""
Process XRP ledger data to create time series features.

Usage:
    python process_data.py --filepath /path/to/xrp_data.csv --output ./output
"""

import os
import argparse
import sys
import logging
from datetime import datetime

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import XRPDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the data processor."""
    parser = argparse.ArgumentParser(description='Process XRP ledger data for ML forecasting')
    parser.add_argument('--filepath', required=True, help='Path to the XRP ledger CSV file')
    parser.add_argument('--output', default='./output', help='Output directory for processed files')
    parser.add_argument('--chunksize', type=int, default=100000, help='Chunksize for processing large files')
    
    args = parser.parse_args()
    
    logger.info(f"Starting data processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input data: {args.filepath}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Initialize and run the processor
        processor = XRPDataProcessor(output_dir=args.output)
        processed_data = processor.process(args.filepath)
        
        logger.info(f"Data processing completed successfully")
        logger.info(f"Processed data saved to {args.output}")
        
        # Print some statistics
        for key, df in processed_data.items():
            if hasattr(df, 'shape'):
                logger.info(f"{key} data shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        logger.exception("Exception details:")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())