#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting module for evaluating trading strategy performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.results = None
        self.trades = []
        self.performance_metrics = {}
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                    transaction_cost: float = 0.001) -> Dict:
        """
        Run backtest on trading signals.
        
        Args:
            data: DataFrame with price data
            signals: DataFrame with trading signals and position sizes
            transaction_cost: Cost per trade as fraction of trade value
            
        Returns:
            Dictionary of performance metrics
        """
        # Initialize portfolio tracking
        portfolio = pd.DataFrame(index=data.index)
        portfolio['close'] = data['close']
        portfolio['signal'] = signals['signal']
        portfolio['position_size'] = signals['position_size']
        portfolio['regime'] = signals['market_regime']
        
        # Calculate position values and returns
        portfolio['position_value'] = self.initial_capital * portfolio['position_size']
        portfolio['returns'] = portfolio['position_value'].pct_change()
        
        # Track trades
        self.trades = self._track_trades(portfolio, transaction_cost)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_metrics(portfolio)
        
        # Store results
        self.results = portfolio
        
        return self.performance_metrics
    
    def _track_trades(self, portfolio: pd.DataFrame, transaction_cost: float) -> List[Dict]:
        """Track individual trades and calculate trade-specific metrics."""
        trades = []
        current_position = 0
        entry_price = 0
        entry_time = None
        
        for idx, row in portfolio.iterrows():
            # Check for position changes
            if row['signal'] != current_position and row['signal'] != 0:
                # Close existing position
                if current_position != 0:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) * current_position
                    pnl -= abs(entry_price * current_position) * transaction_cost  # Entry cost
                    pnl -= abs(exit_price * current_position) * transaction_cost   # Exit cost
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': current_position,
                        'pnl': pnl,
                        'return': (pnl / (entry_price * abs(current_position))) - transaction_cost * 2,
                        'regime': portfolio.loc[entry_time, 'regime']
                    })
                
                # Open new position
                current_position = row['signal']
                entry_price = row['close']
                entry_time = idx
        
        return trades
    
    def _calculate_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = ((1 + portfolio['returns']).prod() - 1) * 100
        metrics['annual_return'] = ((1 + portfolio['returns']).prod() ** (252 / len(portfolio)) - 1) * 100
        metrics['volatility'] = portfolio['returns'].std() * np.sqrt(252) * 100
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility'] if metrics['volatility'] != 0 else 0
        
        # Drawdown analysis
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod()
        portfolio['rolling_max'] = portfolio['cumulative_returns'].expanding().max()
        portfolio['drawdown'] = (portfolio['cumulative_returns'] - portfolio['rolling_max']) / portfolio['rolling_max']
        metrics['max_drawdown'] = portfolio['drawdown'].min() * 100
        
        # Trade metrics
        if self.trades:
            metrics['total_trades'] = len(self.trades)
            metrics['winning_trades'] = len([t for t in self.trades if t['pnl'] > 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] * 100
            metrics['avg_trade_return'] = np.mean([t['return'] for t in self.trades]) * 100
            metrics['avg_trade_duration'] = np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades])
        
        # Regime-specific metrics
        for regime in [-1, 1]:
            regime_mask = portfolio['regime'] == regime
            regime_returns = portfolio.loc[regime_mask, 'returns']
            regime_name = 'bearish' if regime == -1 else 'bullish'
            
            if len(regime_returns) > 0:
                metrics[f'{regime_name}_return'] = ((1 + regime_returns).prod() - 1) * 100
                metrics[f'{regime_name}_sharpe'] = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() != 0 else 0
                regime_trades = [t for t in self.trades if t['regime'] == regime]
                if regime_trades:
                    metrics[f'{regime_name}_trades'] = len(regime_trades)
                    metrics[f'{regime_name}_win_rate'] = len([t for t in regime_trades if t['pnl'] > 0]) / len(regime_trades) * 100
        
        return metrics
    
    def plot_results(self, save_dir: str) -> None:
        """
        Generate and save performance visualization plots.
        
        Args:
            save_dir: Directory to save plots
        """
        if self.results is None:
            logger.warning("No results to plot. Run backtest first.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Equity curve
        plt.figure(figsize=(12, 6))
        self.results['cumulative_returns'].plot()
        plt.title('Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'equity_curve.png'))
        plt.close()
        
        # 2. Drawdown plot
        plt.figure(figsize=(12, 6))
        self.results['drawdown'].plot()
        plt.title('Strategy Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'drawdown.png'))
        plt.close()
        
        # 3. Regime-specific returns distribution
        plt.figure(figsize=(12, 6))
        for regime in [-1, 1]:
            regime_mask = self.results['regime'] == regime
            regime_returns = self.results.loc[regime_mask, 'returns']
            if len(regime_returns) > 0:
                sns.histplot(regime_returns, label='Bearish' if regime == -1 else 'Bullish',
                           alpha=0.5, bins=50)
        plt.title('Returns Distribution by Regime')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'returns_distribution.png'))
        plt.close()
        
        # 4. Trade analysis
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            
            # Trade PnL distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(trade_df['pnl'], bins=50)
            plt.title('Trade PnL Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'trade_pnl_distribution.png'))
            plt.close()
            
            # Trade duration vs. return scatter
            plt.figure(figsize=(12, 6))
            plt.scatter(trade_df['exit_time'] - trade_df['entry_time'],
                       trade_df['return'],
                       alpha=0.5)
            plt.title('Trade Duration vs. Return')
            plt.xlabel('Duration')
            plt.ylabel('Return')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'trade_duration_return.png'))
            plt.close()
    
    def save_results(self, save_dir: str) -> None:
        """
        Save backtest results to files.
        
        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save performance metrics
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'))
        
        # Save trade log
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(os.path.join(save_dir, 'trade_log.csv'))
        
        # Save portfolio results
        if self.results is not None:
            self.results.to_csv(os.path.join(save_dir, 'portfolio_results.csv'))

def main():
    """Main function to run backtest."""
    try:
        # Load data and strategy results
        from feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()
        data = feature_engineer.load_data()
        
        results_dir = '/Users/berlin/Desktop/RegimeAlpha/RegimeAlpha/data/results'
        signals = pd.read_csv(os.path.join(results_dir, 'trading_signals.csv'), index_col=0)
        signals.index = pd.to_datetime(signals.index)
        
        # Initialize and run backtest
        backtest = BacktestEngine()
        metrics = backtest.run_backtest(data, signals)
        
        # Log performance metrics
        logger.info("\nBacktest Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.2f}")
        
        # Generate and save visualizations
        plots_dir = os.path.join(results_dir, 'plots')
        backtest.plot_results(plots_dir)
        
        # Save detailed results
        backtest.save_results(results_dir)
        
        logger.info(f"\nResults saved to {results_dir}")
        logger.info(f"Plots saved to {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        raise

if __name__ == '__main__':
    main() 