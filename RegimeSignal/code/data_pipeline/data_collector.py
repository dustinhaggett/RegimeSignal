#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data collection module for fetching Binance perpetual futures data.
"""

import os
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
from calendar import monthrange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinancePerpetualDataCollector:
    """Class for collecting Binance perpetual futures data."""
    
    def __init__(self, symbol='BTCUSDT', timeframe='15m'):
        """
        Initialize the BinancePerpetualDataCollector.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe for the data (e.g., '15m', '1h', '1d')
        """
        # Remove the slash if present in the symbol
        self.symbol = symbol.replace('/', '') 
        self.timeframe = timeframe
        
        # Base URLs (try different endpoints if one fails)
        self.base_urls = [
            "https://fapi.binance.com",  # Global
            "https://dapi.binance.com",  # COIN-M Futures (alternative)
            "https://fapi.binance.us"    # US-specific (may not exist but worth trying)
        ]
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create subdirectory for perpetual data
        self.perpetual_dir = os.path.join(self.data_dir, 'perpetual')
        os.makedirs(self.perpetual_dir, exist_ok=True)
        
        # Configure timeframe mapping for API
        self.timeframe_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h', 
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        
        logger.info(f"Initialized BinancePerpetualDataCollector for {symbol} perpetual with {timeframe} timeframe")
    
    def fetch_data(self, start_date, end_date=None, include_funding=True, force_download=False):
        """
        Fetch perpetual futures data for the specified period.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format. If None, uses current date.
            include_funding (bool): Whether to include funding rate data
            force_download (bool): Whether to force download even if local file exists
        
        Returns:
            pd.DataFrame: DataFrame containing the fetched data
        """
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime.now()
        
        logger.info(f"Fetching {self.symbol} perpetual futures data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        # Check if data already exists locally
        filename = f"Binance_{self.symbol}_perpetual_{self.timeframe}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.perpetual_dir, filename)
        
        if os.path.exists(filepath) and not force_download:
            logger.info(f"Loading data from local file: {filepath}")
            return pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        
        # Try different base URLs until one works
        for base_url in self.base_urls:
            try:
                logger.info(f"Attempting to use API endpoint: {base_url}")
                
                # Fetch price data
                price_data = self._fetch_price_data(start_dt, end_dt, base_url)
                logger.info(f"Successfully fetched {len(price_data)} price data points")
                
                # Fetch funding rate data if requested
                if include_funding:
                    try:
                        funding_data = self._fetch_funding_rate_data(start_dt, end_dt, base_url)
                        logger.info(f"Successfully fetched {len(funding_data)} funding rate data points")
                        
                        # Merge price and funding data
                        final_data = self._merge_price_funding_data(price_data, funding_data)
                        logger.info(f"Merged price and funding data: {len(final_data)} data points")
                    except Exception as e:
                        logger.warning(f"Failed to fetch funding data: {e}")
                        logger.info("Proceeding with price data only")
                        final_data = price_data
                else:
                    final_data = price_data
                
                # Save data to file
                final_data.to_csv(filepath)
                logger.info(f"Saved data to {filepath}")
                
                return final_data
                
            except Exception as e:
                logger.warning(f"Failed with endpoint {base_url}: {e}")
        
        # If we get here, all endpoints failed
        raise Exception("All Binance API endpoints failed. You may need to use a VPN to access Binance API from your region.")
    
    def _fetch_price_data(self, start_dt, end_dt, base_url):
        """
        Fetch price data from Binance futures API.
        
        Args:
            start_dt (datetime): Start datetime
            end_dt (datetime): End datetime
            base_url (str): Base URL for the API
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        # Convert timeframe to milliseconds
        tf_ms = self._timeframe_to_milliseconds(self.timeframe)
        
        # Convert datetimes to timestamps
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        # Fetch data in chunks due to API limitations (typically 1000 candles per request)
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            # Construct API URL (using Binance futures API)
            tf_param = self.timeframe_map.get(self.timeframe, self.timeframe)
            url = f"{base_url}/fapi/v1/klines?symbol={self.symbol}&interval={tf_param}&startTime={current_ts}&limit=1000"
            
            logger.info(f"Fetching price data from {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                candles = response.json()
                if not candles:
                    logger.info("No more data available")
                    break
                
                if len(candles) == 0:
                    logger.info("Empty response, possibly reached end of available data")
                    break
                
                all_candles.extend(candles)
                
                # Update timestamp for next chunk
                # Add 1ms to avoid duplicates
                current_ts = candles[-1][0] + 1
                
                # Respect API rate limits
                time.sleep(0.5)
                
            except requests.exceptions.HTTPError as e:
                # Check for specific error codes
                try:
                    error_data = response.json()
                    if 'code' in error_data:
                        if error_data['code'] == -1121:
                            raise Exception(f"Invalid symbol: {self.symbol}")
                        elif error_data['code'] == -1003:
                            logger.warning("Rate limit exceeded, waiting longer")
                            time.sleep(60)  # Wait a minute
                            continue  # Try again
                        elif error_data['code'] == 0 and 'msg' in error_data and 'restricted location' in error_data['msg']:
                            raise Exception("Service unavailable from your location. Consider using a VPN.")
                except json.JSONDecodeError:
                    pass
                
                # Re-raise the exception
                raise
        
        # If we got data, convert to DataFrame
        if all_candles:
            # Binance klines format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            df = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only relevant columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        else:
            raise Exception("No data returned from Binance API")
    
    def _fetch_funding_rate_data(self, start_dt, end_dt, base_url):
        """
        Fetch funding rate data for the specified period.
        
        Args:
            start_dt (datetime): Start datetime
            end_dt (datetime): End datetime
            base_url (str): Base URL for the API
            
        Returns:
            pd.DataFrame: DataFrame with funding rate data
        """
        # Convert datetimes to timestamps
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        # Fetch funding rate data in chunks
        all_funding_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            # Construct API URL for funding rate
            url = f"{base_url}/fapi/v1/fundingRate?symbol={self.symbol}&startTime={current_ts}&limit=1000"
            
            logger.info(f"Fetching funding data from {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                funding_data = response.json()
                if not funding_data:
                    logger.info("No more funding rate data available")
                    break
                
                if len(funding_data) == 0:
                    logger.info("Empty funding rate response, possibly reached end of available data")
                    break
                
                all_funding_data.extend(funding_data)
                
                # Update timestamp for next chunk - add 1ms to avoid duplicates
                current_ts = int(funding_data[-1]['fundingTime']) + 1
                
                # Respect API rate limits
                time.sleep(0.5)
                
            except requests.exceptions.HTTPError as e:
                # Check for specific error codes
                try:
                    error_data = response.json()
                    if 'code' in error_data:
                        if error_data['code'] == -1121:
                            raise Exception(f"Invalid symbol: {self.symbol}")
                        elif error_data['code'] == -1003:
                            logger.warning("Rate limit exceeded, waiting longer")
                            time.sleep(60)  # Wait a minute
                            continue  # Try again
                except json.JSONDecodeError:
                    pass
                
                # Re-raise the exception
                raise
        
        # If we got data, convert to DataFrame
        if all_funding_data:
            df = pd.DataFrame(all_funding_data)
            
            # Convert timestamps and format data
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = pd.to_numeric(df['fundingRate'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only funding rate column
            df = df[['funding_rate']]
            
            return df
        else:
            raise Exception("No funding rate data returned from Binance API")
    
    def _merge_price_funding_data(self, price_df, funding_df):
        """
        Merge price and funding rate data.
        
        Args:
            price_df (pd.DataFrame): Price data
            funding_df (pd.DataFrame): Funding rate data
            
        Returns:
            pd.DataFrame: Merged DataFrame
        """
        # Ensure both DataFrames have datetime index
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("Price data must have a datetime index")
        if not isinstance(funding_df.index, pd.DatetimeIndex):
            raise ValueError("Funding rate data must have a datetime index")
        
        # Forward fill funding rates to match price candles
        # First, reindex funding_df to match price_df's index
        funding_reindexed = funding_df.reindex(price_df.index, method='ffill')
        
        # Then merge the DataFrames
        merged_df = price_df.copy()
        # Avoid the pandas warning by not using inplace=True with fillna
        funding_column = funding_reindexed['funding_rate'].fillna(0)
        merged_df['funding_rate'] = funding_column
        
        return merged_df
    
    def _timeframe_to_milliseconds(self, timeframe):
        """
        Convert timeframe string to milliseconds.
        
        Args:
            timeframe (str): Timeframe string (e.g., '15m', '1h', '1d')
            
        Returns:
            int: Timeframe in milliseconds
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        elif unit == 'M':
            return value * 30 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    def collect_extended_data(self, start_year=2022, end_year=2024, include_funding=True, force_download=False):
        """
        Collect data for multiple years, broken down by quarters for manageable downloads.
        
        Args:
            start_year (int): Starting year
            end_year (int): Ending year (inclusive)
            include_funding (bool): Whether to include funding rate data
            force_download (bool): Whether to force download even if local file exists
            
        Returns:
            dict: Dictionary mapping periods to DataFrames with the collected data
        """
        collected_data = {}
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        for year in range(start_year, end_year + 1):
            # Skip future years
            if year > current_year:
                logger.info(f"Skipping future year: {year}")
                continue
                
            for quarter in range(1, 5):
                # Calculate quarter months
                start_month = (quarter - 1) * 3 + 1
                end_month = quarter * 3
                
                # Skip future quarters
                if year == current_year and start_month > current_month:
                    logger.info(f"Skipping future quarter: Q{quarter} {year}")
                    continue
                
                # Calculate end day (last day of the month)
                end_day = monthrange(year, end_month)[1]
                
                # Define date range
                start_date = f'{year}-{start_month:02d}-01'
                end_date = f'{year}-{end_month:02d}-{end_day:02d}'
                
                # Adjust end date if it's in the future
                if year == current_year and end_month > current_month:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                try:
                    logger.info(f"Collecting data for Q{quarter} {year} ({start_date} to {end_date})")
                    data = self.fetch_data(start_date, end_date, include_funding, force_download)
                    period_key = f"{year}_Q{quarter}"
                    collected_data[period_key] = data
                    logger.info(f"Successfully collected {len(data)} data points for {period_key}")
                    
                    # Add a delay between quarters to respect API limits
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Failed to collect data for Q{quarter} {year}: {e}")
        
        return collected_data
    
    def combine_all_data(self, save_combined=True):
        """
        Combine all downloaded data into a single DataFrame.
        
        Args:
            save_combined (bool): Whether to save the combined data to a file
            
        Returns:
            pd.DataFrame: Combined DataFrame with all data
        """
        # Get all CSV files in the perpetual directory
        csv_files = [f for f in os.listdir(self.perpetual_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise Exception("No data files found to combine")
        
        # Initialize an empty list to store DataFrames
        dfs = []
        
        # Read each CSV file
        for file in csv_files:
            file_path = os.path.join(self.perpetual_dir, file)
            logger.info(f"Reading file: {file}")
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            dfs.append(df)
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs)
        
        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        logger.info(f"Combined {len(dfs)} files, resulting in {len(combined_df)} data points")
        
        # Save combined data if requested
        if save_combined:
            combined_path = os.path.join(self.perpetual_dir, f"Combined_{self.symbol}_{self.timeframe}.csv")
            combined_df.to_csv(combined_path)
            logger.info(f"Saved combined data to {combined_path}")
        
        return combined_df

if __name__ == "__main__":
    # Create collector instance
    collector = BinancePerpetualDataCollector(symbol='BTCUSDT', timeframe='15m')
    
    try:
        # Option 1: Collect a single month of data
        # data = collector.fetch_data('2023-01-01', '2023-01-31', include_funding=True)
        
        # Option 2: Collect extended data (multiple years)
        collected_data = collector.collect_extended_data(
            start_year=2022,
            end_year=2024,
            include_funding=True
        )
        
        # Combine all downloaded data
        combined_data = collector.combine_all_data(save_combined=True)
        
        print(f"Successfully collected and combined data:")
        print(f"Total data points: {len(combined_data)}")
        print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        print("\nSample data:")
        print(combined_data.head())
        
        if 'funding_rate' in combined_data.columns:
            print("\nFunding rate statistics:")
            print(f"Mean funding rate: {combined_data['funding_rate'].mean():.6f}")
            print(f"Max funding rate: {combined_data['funding_rate'].max():.6f}")
            print(f"Min funding rate: {combined_data['funding_rate'].min():.6f}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nIf you're in a region with Binance restrictions, you have these options:")
        print("1. Use a VPN to access Binance API")
        print("2. Download from CryptoDataDownload: https://www.cryptodatadownload.com/data/binance/")
        print("3. Use an alternative exchange API like Bybit or FTX")