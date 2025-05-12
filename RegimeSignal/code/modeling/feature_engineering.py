#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for loading and processing data.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, List
import ta
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles data loading and feature engineering."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.data = None
        self.feature_names = None
        self.raw_data = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load and combine all raw data files.
        
        Returns:
            DataFrame with combined raw data
        """
        try:
            # Get all CSV files from the perpetual directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(project_root, 'data', 'perpetual')
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            # Load and combine all files
            dfs = []
            for file in csv_files:
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            # Combine all dataframes
            self.raw_data = pd.concat(dfs, ignore_index=True)
            
            # Convert timestamp to datetime
            if 'timestamp' in self.raw_data.columns:
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
                self.raw_data.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            self.raw_data.sort_index(inplace=True)
            
            # Remove duplicates
            self.raw_data = self.raw_data[~self.raw_data.index.duplicated(keep='first')]
            
            logger.info(f"Loaded raw data with shape: {self.raw_data.shape}")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate technical indicators and features.
        
        Returns:
            DataFrame with calculated features
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        try:
            # Create a copy of raw data
            df = self.raw_data.copy()
            
            # Save timestamp
            df = df.reset_index()
            
            # Rename columns to match new data format
            column_mapping = {
                'sumOpenInterest': 'open_interest',
                'sumOpenInterestValue': 'open_interest_value',
                'longShortRatio': 'long_short_ratio',
                'longAccount': 'long_account',
                'shortAccount': 'short_account',
                'buySellRatio': 'buy_sell_ratio',
                'buyVol': 'buy_volume',
                'sellVol': 'sell_volume',
                'totalVol': 'total_volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Calculate trend indicators
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            df['trend_strength'] = df['adx'] * np.where(df['close'] > df['ema_50'], 1, -1)
            
            # Calculate momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
            df['awesome_oscillator'] = ta.momentum.awesome_oscillator(df['high'], df['low'], window1=5, window2=34)
            df['rocr_48'] = ta.momentum.roc(df['close'], window=48)
            df['ultimate_osc'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'], window1=7, window2=14, window3=28)
            
            # Calculate volatility indicators
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            # Calculate Ulcer Index
            close_series = df['close']
            roll_max = close_series.rolling(window=14).max()
            pct_drawdown = ((close_series - roll_max) / roll_max) * 100
            df['ulcer_index'] = np.sqrt((pct_drawdown ** 2).rolling(window=14).mean())
            
            # Calculate volatility regime
            bb_width_ma = df['bb_width'].rolling(window=20).mean()
            df['volatility_regime'] = np.where(
                df['bb_width'] > bb_width_ma * 1.2,
                1,  # High volatility
                np.where(
                    df['bb_width'] < bb_width_ma * 0.8,
                    -1,  # Low volatility
                    0   # Normal volatility
                )
            )
            
            # Calculate volume indicators
            df['volume_normalized'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
            df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'], window=14)
            
            # Calculate funding rate indicators
            if 'funding_rate' in df.columns:
                df['funding_rate_ma'] = df['funding_rate'].rolling(window=20).mean()
                df['funding_rate_std'] = df['funding_rate'].rolling(window=20).std()
                df['funding_rate_zscore'] = (df['funding_rate'] - df['funding_rate_ma']) / df['funding_rate_std']
            
            # Calculate open interest indicators
            if 'open_interest' in df.columns:
                df['oi_ma'] = df['open_interest'].rolling(window=20).mean()
                df['oi_std'] = df['open_interest'].rolling(window=20).std()
                df['oi_zscore'] = (df['open_interest'] - df['oi_ma']) / df['oi_std']
                df['oi_change'] = df['open_interest'].pct_change()
                df['oi_value_change'] = df['open_interest_value'].pct_change()
            
            # Calculate long/short ratio indicators
            if 'long_short_ratio' in df.columns:
                df['lsr_ma'] = df['long_short_ratio'].rolling(window=20).mean()
                df['lsr_std'] = df['long_short_ratio'].rolling(window=20).std()
                df['lsr_zscore'] = (df['long_short_ratio'] - df['lsr_ma']) / df['lsr_std']
                df['long_account_change'] = df['long_account'].pct_change()
                df['short_account_change'] = df['short_account'].pct_change()
            
            # Calculate liquidation volume indicators
            if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
                df['liq_ratio'] = df['buy_volume'] / df['sell_volume']
                df['liq_ratio_ma'] = df['liq_ratio'].rolling(window=20).mean()
                df['liq_ratio_std'] = df['liq_ratio'].rolling(window=20).std()
                df['liq_ratio_zscore'] = (df['liq_ratio'] - df['liq_ratio_ma']) / df['liq_ratio_std']
                df['total_volume_change'] = df['total_volume'].pct_change()
            
            # Calculate market regime based on trend and momentum
            trend_threshold = 25  # ADX threshold for trend strength
            df['regime'] = np.where(
                df['trend_strength'] > trend_threshold, 1,  # Bullish
                np.where(
                    df['trend_strength'] < -trend_threshold, -1,  # Bearish
                    0  # Neutral
                )
            )
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Store the processed data
            self.data = df
            self.feature_names = df.columns.tolist()
            
            logger.info(f"Calculated features with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise
    
    def save_processed_data(self) -> None:
        """Save the processed data to CSV."""
        if self.data is None:
            raise ValueError("No processed data to save")
        
        try:
            # Create processed directory if it doesn't exist
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            processed_dir = os.path.join(project_root, 'RegimeSignal', 'data', 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            
            # Save to CSV
            output_path = os.path.join(processed_dir, 'processed_data.csv')
            self.data.to_csv(output_path)
            logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the processed data.
        
        Returns:
            DataFrame with features and regime labels
        """
        try:
            # Try to load existing processed data
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            processed_path = os.path.join(project_root, 'RegimeSignal', 'data', 'processed', 'processed_data.csv')
            
            if os.path.exists(processed_path):
                self.data = pd.read_csv(processed_path)
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data.set_index('timestamp', inplace=True)
                logger.info(f"Loaded existing processed data with shape: {self.data.shape}")
                return self.data
            
            # If no processed data exists, calculate features
            logger.info("No processed data found. Calculating features...")
            self.calculate_features()
            self.save_processed_data()
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_feature_categories(self) -> dict:
        """
        Get features grouped by category.
        
        Returns:
            Dictionary of feature categories and their features
        """
        categories = {
            'price': ['open', 'high', 'low', 'close'],
            'trend': ['ema_50', 'trend_strength', 'adx'],
            'momentum': ['rsi', 'mfi', 'awesome_oscillator', 'rocr_48', 'ultimate_osc'],
            'volatility': ['ulcer_index', 'bb_width', 'volatility_regime'],
            'volume': ['volume', 'volume_normalized', 'cmf', 'eom'],
            'regime': ['regime']
        }
        
        return categories
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        Get basic statistics for all features.
        
        Returns:
            DataFrame with feature statistics
        """
        if self.data is None:
            raise ValueError("Data must be loaded before getting feature stats")
        
        return self.data.describe()
    
    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the data.
        
        Returns:
            Series with missing value counts
        """
        if self.data is None:
            raise ValueError("Data must be loaded before checking missing values")
        
        return self.data.isnull().sum()
    
    def get_feature_correlations(self) -> pd.DataFrame:
        """
        Get correlation matrix for all features.
        
        Returns:
            DataFrame with feature correlations
        """
        if self.data is None:
            raise ValueError("Data must be loaded before getting correlations")
        
        return self.data.corr()

def main():
    """Main function to test feature engineering."""
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Load data
        data = feature_engineer.load_data()
        
        # Get feature categories
        categories = feature_engineer.get_feature_categories()
        logger.info("Feature categories:")
        for category, features in categories.items():
            logger.info(f"{category}: {features}")
        
        # Get feature statistics
        stats = feature_engineer.get_feature_stats()
        logger.info("\nFeature statistics:")
        logger.info(stats)
        
        # Check missing values
        missing = feature_engineer.check_missing_values()
        logger.info("\nMissing values:")
        logger.info(missing)
        
        # Get correlations
        correlations = feature_engineer.get_feature_correlations()
        logger.info("\nFeature correlations:")
        logger.info(correlations)
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

if __name__ == '__main__':
    main() 