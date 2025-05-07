#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update data before retraining.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline.data_collector import BinancePerpetualDataCollector
from modeling.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_data():
    """Update data for model retraining."""
    try:
        # Initialize data collector
        collector = BinancePerpetualDataCollector(
            symbol='BTCUSDT',
            timeframe='15m'
        )
        
        # Calculate date range
        start_date = '2023-01-01'  # Start from 2023 for retraining
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Updating data from {start_date} to {end_date}")
        
        # Collect data with all available metrics
        data = collector.collect_extended_data(
            start_year=2023,
            end_year=datetime.now().year,
            include_funding=True,
            include_oi=True,
            include_ls=True,
            include_liq=True,
            force_download=True  # Force fresh download
        )
        
        # Process the data
        feature_engineer = FeatureEngineer()
        
        # Calculate features and save processed data
        feature_engineer.calculate_features()
        feature_engineer.save_processed_data()
        
        logger.info("Data update completed successfully!")
        
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        raise

if __name__ == '__main__':
    update_data() 