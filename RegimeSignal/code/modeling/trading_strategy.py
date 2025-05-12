#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading strategy implementation for RegimeSignal.
Generates trading signals based on regime predictions and confidence scores.
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def generate_trading_signals(
    labels: List[int],
    preds: List[int],
    probs: List[List[float]],
    prices: pd.Series,
    threshold: float = 0.85,
    base_size: float = 0.3,
    min_holding_period: int = 10,
    max_position_size: float = 0.8,
    volatility_lookback: int = 20,
    momentum_lookback: int = 5,
    regime_persistence: int = 5,
    transition_threshold: float = 0.75
) -> pd.DataFrame:
    """
    Generate trading signals based on model predictions and confidence scores.
    
    Args:
        labels: True regime labels
        preds: Model predictions
        probs: Prediction probabilities for each class
        prices: Price series for volatility calculation
        threshold: Confidence threshold for generating signals
        base_size: Base position size
        min_holding_period: Minimum number of days to hold a position
        max_position_size: Maximum position size multiplier
        volatility_lookback: Number of days for volatility calculation
        momentum_lookback: Number of days for momentum calculation
        regime_persistence: Number of days to confirm regime persistence
        transition_threshold: Minimum probability for regime transition
        
    Returns:
        DataFrame with trading signals and position sizes
    """
    try:
        logger.info(f"Generating trading signals with threshold={threshold}, base_size={base_size}")
        
        # Convert inputs to numpy arrays
        labels = np.array(labels)
        preds = np.array(preds)
        probs = np.array(probs)
        
        # Calculate volatility and momentum
        returns = prices.pct_change()
        volatility = returns.rolling(window=volatility_lookback).std()
        momentum = returns.rolling(window=momentum_lookback).mean()
        
        # Calculate dynamic volatility bands with wider range
        volatility_width = 2.5 * volatility.rolling(window=volatility_lookback).mean()
        upper_band = prices * (1 + volatility_width)
        lower_band = prices * (1 - volatility_width)
        
        # Calculate regime strength and transition metrics
        regime_strength = np.max(probs, axis=1) - np.partition(probs, -2, axis=1)[:, -2]
        regime_transition = np.abs(np.diff(probs, axis=0, prepend=probs[0]))
        
        # Calculate regime persistence with stricter requirements
        regime_persistence_mask = np.zeros_like(preds, dtype=bool)
        current_regime = preds[0]
        days_in_regime = 0
        
        for i in range(len(preds)):
            if preds[i] == current_regime:
                days_in_regime += 1
            else:
                current_regime = preds[i]
                days_in_regime = 1
            regime_persistence_mask[i] = days_in_regime >= regime_persistence and regime_strength[i] > 0.4
        
        # Calculate regime momentum with trend confirmation
        regime_momentum = np.zeros_like(preds, dtype=float)
        for i in range(momentum_lookback, len(preds)):
            window_returns = returns[i-momentum_lookback:i]
            if preds[i] == 2:  # Bullish
                regime_momentum[i] = np.mean(window_returns) if np.all(window_returns > 0) else 0
            elif preds[i] == 0:  # Bearish
                regime_momentum[i] = -np.mean(window_returns) if np.all(window_returns < 0) else 0
        
        # Initialize signals DataFrame
        signals = pd.DataFrame({
            'true_regime': labels,
            'pred_regime': preds,
            'confidence': np.max(probs, axis=1),
            'trading_signal': 'STAY_OUT',
            'position_size': 0.0,
            'volatility': volatility,
            'momentum': momentum,
            'days_held': 0,
            'regime_strength': regime_strength,
            'regime_persistence': regime_persistence_mask,
            'regime_momentum': regime_momentum
        })
        
        # Generate trading signals with position sizing
        current_position = 'STAY_OUT'
        days_held = 0
        
        for i in range(volatility_lookback, len(signals)):
            # Skip if not enough history
            if i < max(volatility_lookback, momentum_lookback):
                continue
                
            # Update days held
            if current_position != 'STAY_OUT':
                days_held += 1
                signals.loc[signals.index[i], 'days_held'] = days_held
            
            # Check for minimum holding period
            if days_held < min_holding_period and current_position != 'STAY_OUT':
                signals.loc[signals.index[i], 'trading_signal'] = current_position
                continue
            
            # Get current state
            pred = preds[i]
            conf = probs[i]
            vol = volatility.iloc[i]
            mom = momentum.iloc[i]
            regime_pers = regime_persistence_mask[i]
            r_strength = regime_strength[i]
            r_momentum = regime_momentum[i]
            
            # Skip if volatility is too high
            if vol > 0.005:
                logger.info(f"Skipping trade due to high volatility: {vol:.4f}")
                continue
            
            # Calculate position size factors with more conservative approach
            vol_factor = 1.0 - (vol / volatility.iloc[i-volatility_lookback:i].quantile(0.75))
            vol_factor = np.clip(vol_factor, 0.3, 1.0)
            
            momentum_factor = np.clip(abs(mom) / vol, 0.0, 1.0)
            persistence_factor = min(days_held / min_holding_period, 1.0) if days_held > 0 else 0.0
            
            # Base position size calculation with additional safety factors
            position_size = base_size * vol_factor * momentum_factor * (1 + persistence_factor) * (1 - vol)
            position_size = min(position_size, max_position_size)
            
            # Generate signals based on regime and conditions with stricter filters
            if pred == 2 and conf[2] > transition_threshold and regime_pers and r_strength > 0.4:  # Bullish
                if r_momentum > 0 and prices.iloc[i] > lower_band.iloc[i] and (current_position == 'STAY_OUT' or days_held >= min_holding_period):
                    signals.loc[signals.index[i], 'trading_signal'] = 'LONG'
                    signals.loc[signals.index[i], 'position_size'] = position_size
                    current_position = 'LONG'
                    days_held = 0 if current_position != 'LONG' else days_held
                    
            elif pred == 0 and conf[0] > transition_threshold and regime_pers and r_strength > 0.4:  # Bearish
                if r_momentum < 0 and prices.iloc[i] < upper_band.iloc[i] and (current_position == 'STAY_OUT' or days_held >= min_holding_period):
                    signals.loc[signals.index[i], 'trading_signal'] = 'SHORT'
                    signals.loc[signals.index[i], 'position_size'] = position_size
                    current_position = 'SHORT'
                    days_held = 0 if current_position != 'SHORT' else days_held
                    
            else:  # Neutral or low confidence
                if days_held >= min_holding_period:
                    signals.loc[signals.index[i], 'trading_signal'] = 'STAY_OUT'
                    signals.loc[signals.index[i], 'position_size'] = 0.0
                    current_position = 'STAY_OUT'
                    days_held = 0
        
        logger.info(f"Generated {len(signals[signals['trading_signal'] != 'STAY_OUT'])} trading signals")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        raise

def save_signals_to_csv(
    signals_df: pd.DataFrame,
    output_dir: str = 'results/signals'
) -> None:
    """Save trading signals to CSV file.
    
    Args:
        signals_df: DataFrame with trading signals
        output_dir: Output directory path
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, 'predicted_signals.csv')
        signals_df.to_csv(output_path, index=False)
        logger.info(f"Trading signals saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving signals to CSV: {str(e)}")
        raise

def analyze_signals(signals: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trading signal performance and statistics.
    
    Args:
        signals: DataFrame with trading signals
        
    Returns:
        Dictionary with signal analysis metrics
    """
    try:
        logger.info("Analyzing trading signals")
        
        # Calculate signal accuracy
        active_signals = signals['trading_signal'].isin(['LONG', 'SHORT'])
        correct_long = (signals['trading_signal'] == 'LONG') & (signals['true_regime'] == 2)
        correct_short = (signals['trading_signal'] == 'SHORT') & (signals['true_regime'] == 0)
        
        total_active = active_signals.sum()
        total_correct = (correct_long | correct_short).sum()
        
        signal_accuracy = total_correct / total_active if total_active > 0 else 0.0
        
        # Calculate mean position size for active signals
        mean_position = np.abs(signals.loc[active_signals, 'position_size']).mean()
        
        # Get signal distribution
        signal_dist = signals['trading_signal'].value_counts().to_dict()
        
        # Compile results
        results = {
            'signal_accuracy': signal_accuracy,
            'mean_position_size': mean_position,
            'active_signals': total_active,
            'total_signals': len(signals),
            'signal_dist': signal_dist
        }
        
        logger.info(f"Signal analysis complete: accuracy={signal_accuracy:.2f}, active_signals={total_active}")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing trading signals: {str(e)}")
        raise

def plot_regime_transitions(
    signals_df: pd.DataFrame,
    output_path: str
) -> None:
    """Plot regime transitions and trading signals over time.
    
    Args:
        signals_df: DataFrame with trading signals
        output_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Plot true regimes
        plt.subplot(2, 1, 1)
        plt.plot(signals_df['timestamp'], signals_df['true_label'], 
                label='True Regime', alpha=0.7)
        plt.plot(signals_df['timestamp'], signals_df['predicted_class'], 
                label='Predicted Regime', alpha=0.7)
        plt.title('Regime Transitions')
        plt.ylabel('Regime')
        plt.legend()
        
        # Plot trading signals
        plt.subplot(2, 1, 2)
        signal_colors = {'LONG': 'green', 'SHORT': 'red', 'STAY_OUT': 'gray'}
        for signal in ['LONG', 'SHORT', 'STAY_OUT']:
            mask = signals_df['trading_signal'] == signal
            plt.scatter(signals_df.loc[mask, 'timestamp'],
                       signals_df.loc[mask, 'position_size'],
                       c=signal_colors[signal],
                       label=signal,
                       alpha=0.6)
        
        plt.title('Trading Signals and Position Sizes')
        plt.ylabel('Position Size')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Regime transition plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting regime transitions: {str(e)}")
        raise

def plot_signal_distribution(
    signals_df: pd.DataFrame,
    output_path: str
) -> None:
    """Plot distribution of trading signals and confidence levels.
    
    Args:
        signals_df: DataFrame with trading signals
        output_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Signal distribution
        plt.subplot(2, 2, 1)
        signals_df['trading_signal'].value_counts().plot(kind='bar')
        plt.title('Trading Signal Distribution')
        plt.ylabel('Count')
        
        # Plot 2: Confidence distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=signals_df, x='confidence', hue='trading_signal',
                    multiple='stack', bins=20)
        plt.title('Confidence Distribution by Signal')
        
        # Plot 3: Position size distribution
        plt.subplot(2, 2, 3)
        sns.boxplot(data=signals_df, x='trading_signal', y='position_size')
        plt.title('Position Size Distribution by Signal')
        
        # Plot 4: Regime accuracy
        plt.subplot(2, 2, 4)
        accuracy = signals_df.groupby('true_label').apply(
            lambda x: (x['true_label'] == x['predicted_class']).mean()
        )
        accuracy.plot(kind='bar')
        plt.title('Accuracy by Regime')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Signal distribution plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting signal distribution: {str(e)}")
        raise

def plot_regime_heatmap(
    signals_df: pd.DataFrame,
    output_path: str
) -> None:
    """Plot heatmap of regime transitions and signal accuracy.
    
    Args:
        signals_df: DataFrame with trading signals
        output_path: Path to save the plot
    """
    try:
        # Create confusion matrix for regimes
        confusion_matrix = pd.crosstab(
            signals_df['true_label'],
            signals_df['predicted_class'],
            normalize='index'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('Regime Transition Matrix')
        plt.xlabel('Predicted Regime')
        plt.ylabel('True Regime')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Regime heatmap saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting regime heatmap: {str(e)}")
        raise

def main():
    """Main function to generate and analyze trading signals."""
    try:
        # Load evaluation results
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'results',
            'evaluation_results.json'
        )
        
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        preds = np.array(results['predictions'])
        labels = np.array(results['labels'])
        probs = np.array(results['probabilities'])
        
        # Generate trading signals
        logger.info("Generating trading signals...")
        signals_df = generate_trading_signals(
            labels=labels,
            preds=preds,
            probs=probs,
            prices=results['prices'],
            threshold=0.85,
            base_size=0.3,
            min_holding_period=10,
            max_position_size=0.8,
            volatility_lookback=20,
            momentum_lookback=5,
            regime_persistence=5,
            transition_threshold=0.75
        )
        
        # Save signals to CSV
        save_signals_to_csv(signals_df)
        
        # Analyze signals
        stats = analyze_signals(signals_df)
        
        # Print statistics
        logger.info("\nTrading Signal Statistics:")
        logger.info(f"Total Samples: {stats['total_signals']}")
        logger.info(f"Active Signals: {stats['active_signals']}")
        logger.info("\nSignal Distribution:")
        for signal, count in stats['signal_dist'].items():
            logger.info(f"{signal}: {count} ({count/stats['total_signals']*100:.2f}%)")
        logger.info(f"\nMean Position Size: {stats['mean_position_size']:.4f}")
        logger.info(f"Signal Accuracy: {stats['signal_accuracy']:.4f}")
        
        # Create visualizations
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'results',
            'visualizations'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        plot_regime_transitions(
            signals_df,
            os.path.join(output_dir, 'regime_transitions.png')
        )
        
        plot_signal_distribution(
            signals_df,
            os.path.join(output_dir, 'signal_distribution.png')
        )
        
        plot_regime_heatmap(
            signals_df,
            os.path.join(output_dir, 'regime_heatmap.png')
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 