#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diagnostic script to analyze data distributions and generate training statistics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from scipy import stats
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from modeling.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_regime_distribution(data: pd.DataFrame, window_size: int = 30) -> None:
    """Analyze regime distribution over time."""
    logger.info("\nRegime Distribution Analysis:")
    
    # Overall distribution
    regime_counts = data['market_regime'].value_counts()
    total = len(data)
    logger.info("\nOverall Regime Distribution:")
    for regime in sorted(regime_counts.index):
        count = regime_counts[regime]
        percentage = (count / total) * 100
        regime_name = "Bearish" if regime == -1 else "Neutral" if regime == 0 else "Bullish"
        logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
    
    # Rolling distribution
    rolling_regimes = pd.DataFrame()
    for regime in [-1, 0, 1]:
        rolling_regimes[f'Regime_{regime}'] = (data['market_regime'] == regime).rolling(
            window=window_size*24*4,  # Convert days to 15-min periods
            min_periods=1
        ).mean()
    
    # Plot regime distribution over time
    plt.figure(figsize=(15, 6))
    for regime in [-1, 0, 1]:
        plt.plot(rolling_regimes.index, 
                rolling_regimes[f'Regime_{regime}'] * 100,
                label=f"{'Bearish' if regime == -1 else 'Neutral' if regime == 0 else 'Bullish'}")
    
    plt.title('Regime Distribution Over Time (30-day Rolling Window)')
    plt.xlabel('Date')
    plt.ylabel('Percentage of Periods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(__file__).parent.parent.parent / 'results' / 'diagnostics' / 'regime_distribution.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"\nRegime distribution plot saved to: {plot_path}")

def generate_training_stats(data: pd.DataFrame, features: list) -> None:
    """Generate and save training statistics."""
    logger.info("\nGenerating Training Statistics:")
    
    stats_out = {}
    for feature in features:
        stats_out[feature] = {
            'mean': float(data[feature].mean()),
            'std': float(data[feature].std()),
            'skew': float(stats.skew(data[feature].dropna())),
            'kurtosis': float(stats.kurtosis(data[feature].dropna())),
            'q1': float(data[feature].quantile(0.25)),
            'q3': float(data[feature].quantile(0.75))
        }
        
        # Plot feature distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=feature, bins=50)
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Count')
        
        # Save plot
        plot_path = Path(__file__).parent.parent.parent / 'results' / 'diagnostics' / f'{feature}_distribution.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
    
    # Save statistics
    stats_path = Path(__file__).parent.parent.parent / 'models' / 'lstm' / 'training_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_out, f, indent=4)
    
    logger.info(f"Training statistics saved to: {stats_path}")
    logger.info(f"Feature distribution plots saved in: {plot_path.parent}")

def analyze_feature_correlations(data: pd.DataFrame, features: list) -> None:
    """Analyze and plot feature correlations."""
    logger.info("\nAnalyzing Feature Correlations:")
    
    # Calculate correlation matrix
    corr_matrix = data[features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # Save plot
    plot_path = Path(__file__).parent.parent.parent / 'results' / 'diagnostics' / 'feature_correlations.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Correlation matrix plot saved to: {plot_path}")

def main():
    """Main function to run diagnostics."""
    try:
        # Load feature engineering
        feature_engineer = FeatureEngineer()
        data = feature_engineer.load_data()
        
        # Load feature names
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        feature_names_path = project_root / 'models' / 'lstm' / 'feature_names.txt'
        with open(feature_names_path, 'r') as f:
            features = f.read().splitlines()
        
        logger.info(f"Analyzing data from {data.index.min()} to {data.index.max()}")
        logger.info(f"Total periods: {len(data)}")
        
        # Run analyses
        analyze_regime_distribution(data)
        generate_training_stats(data, features)
        analyze_feature_correlations(data, features)
        
        # Print recent regime distribution
        recent_cutoff = data.index.max() - timedelta(days=30)
        recent_data = data[data.index > recent_cutoff]
        
        logger.info("\nRecent Data Regime Distribution (Last 30 Days):")
        regime_counts = recent_data['market_regime'].value_counts()
        total = len(recent_data)
        for regime in sorted(regime_counts.index):
            count = regime_counts[regime]
            percentage = (count / total) * 100
            regime_name = "Bearish" if regime == -1 else "Neutral" if regime == 0 else "Bullish"
            logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error in diagnostics: {str(e)}")
        raise

if __name__ == '__main__':
    main() 