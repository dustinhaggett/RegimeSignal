#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Walk-forward testing module for RegimeSignal.
Implements walk-forward analysis to validate model performance on unseen data.
"""

import numpy as np
import pandas as pd
import torch
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from scipy.stats import entropy
import matplotlib.dates as mdates

from lstm_trainer import RegimeAwareLSTM
from trading_strategy import generate_trading_signals, analyze_signals
from backtesting import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WalkForwardTester:
    """Class for performing walk-forward testing."""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        confidence_threshold: float = 0.85,
        base_size: float = 1.0,
        sequence_length: int = 60
    ):
        """
        Initialize walk-forward tester.
        
        Args:
            model_path: Path to trained model
            data_path: Path to data directory
            initial_capital: Starting capital for backtesting
            transaction_cost: Cost per trade as fraction of trade value
            confidence_threshold: Minimum confidence for trading signals
            base_size: Base position size
            sequence_length: Length of input sequences for LSTM
        """
        self.model_path = model_path
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.confidence_threshold = confidence_threshold
        self.base_size = base_size
        self.sequence_length = sequence_length
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(initial_capital=initial_capital)
        
    def _load_model(self) -> RegimeAwareLSTM:
        """Load trained model."""
        try:
            model = RegimeAwareLSTM(input_size=23, hidden_size=64, num_layers=2)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            logger.info(f"Model loaded successfully on device: {self.device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def _prepare_data(self, start_date: str, sequence_length: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare data for walk-forward testing."""
        try:
            # Load and preprocess data
            data = pd.read_csv(self.data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Filter data for the current window
            window_data = data[data['timestamp'] >= start_date].copy()
            if len(window_data) < sequence_length:
                logger.warning(f"Insufficient data points for window starting at {start_date}")
                return None, None, None
            
            # Get all features except 'regime' and 'timestamp'
            features = window_data.drop(['regime', 'timestamp'], axis=1)
            close_prices = window_data['close']
            
            # Map regime values from [-1, 0, 1] to [0, 1, 2]
            regime_map = {-1: 0, 0: 1, 1: 2}
            targets = window_data['regime'].map(regime_map)
            
            return features, targets, close_prices
        
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None, None
        
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs: np.ndarray,
        signals: pd.DataFrame,
        close_prices: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for a single window."""
        # Log sample confidence vectors
        logger.info(f"Sample confidence vectors (first 5):\n{probs[:5]}")
        logger.info(f"Confidence statistics:\n{probs.mean(axis=0)}")
        
        # Log signal distribution
        from collections import Counter
        signal_counts = Counter(signals['trading_signal'])
        logger.info(f"Signal distribution: {dict(signal_counts)}")
        
        # Overall metrics with zero_division parameter
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Log prediction distribution
        pred_counts = Counter(y_pred)
        true_counts = Counter(y_true)
        logger.info(f"Prediction distribution: {dict(pred_counts)}")
        logger.info(f"True distribution: {dict(true_counts)}")
        
        # Calculate trading metrics
        trading_metrics = self._calculate_win_rate(signals, close_prices)
        metrics.update(trading_metrics)
        
        # Log trading metrics
        logger.info(f"Trading metrics:")
        logger.info(f"Win rate: {metrics['win_rate']:.2%}")
        logger.info(f"Total trades: {metrics['total_trades']}")
        logger.info(f"Total P&L: {metrics['total_pnl']:.2f}")
        logger.info(f"Average trade duration: {metrics['avg_trade_duration']:.1f}")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2f}")
        logger.info(f"Profit factor: {metrics['profit_factor']:.2f}")
        
        return metrics
        
    def _calculate_win_rate(self, signals: pd.DataFrame, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate trading metrics including win rate, P&L, and risk metrics.
        
        Args:
            signals: DataFrame with trading signals
            prices: Price series for P&L calculation
            
        Returns:
            Dictionary with trading metrics
        """
        try:
            # Risk management parameters
            stop_loss_pct = 0.02
            take_profit_pct = 0.04
            max_daily_loss = 0.02
            trailing_stop_pct = 0.015
            max_drawdown_limit = 0.05
            
            # Initialize metrics
            total_trades = 0
            winning_trades = 0
            total_pnl = 0
            daily_pnl = 0
            current_drawdown = 0
            peak_equity = 1.0
            current_equity = 1.0
            
            # Track current position
            current_position = None
            entry_price = None
            position_size = 0
            days_held = 0
            trailing_stop_price = None
            
            # Calculate daily returns and volatility
            daily_returns = prices.pct_change()
            volatility = daily_returns.rolling(window=20).std()
            
            # Trading loop
            for i in range(len(prices)):
                current_price = prices.iloc[i]
                current_signal = signals.iloc[i]['trading_signal']
                current_vol = volatility.iloc[i]
                
                # Update equity and drawdown
                if current_position:
                    pnl = 0
                    if current_position == 'LONG':
                        pnl = (current_price - entry_price) / entry_price * position_size
                    else:
                        pnl = (entry_price - current_price) / entry_price * position_size
                        
                    current_equity = current_equity * (1 + pnl)
                    current_drawdown = (peak_equity - current_equity) / peak_equity
                    peak_equity = max(peak_equity, current_equity)
                    
                    # Check stop conditions
                    stop_triggered = False
                    
                    # Basic stop loss
                    if pnl < -stop_loss_pct:
                        stop_triggered = True
                        logging.info(f"Stop loss triggered at {current_price}")
                    
                    # Take profit
                    elif pnl > take_profit_pct:
                        stop_triggered = True
                        logging.info(f"Take profit triggered at {current_price}")
                    
                    # Trailing stop
                    if trailing_stop_price:
                        if current_position == 'LONG' and current_price < trailing_stop_price:
                            stop_triggered = True
                            logging.info(f"Trailing stop triggered at {current_price}")
                        elif current_position == 'SHORT' and current_price > trailing_stop_price:
                            stop_triggered = True
                            logging.info(f"Trailing stop triggered at {current_price}")
                    
                    # Maximum drawdown
                    if current_drawdown > max_drawdown_limit:
                        stop_triggered = True
                        logging.info(f"Max drawdown limit triggered at {current_price}")
                    
                    # Exit position if stop triggered
                    if stop_triggered:
                        total_pnl += pnl
                        if pnl > 0:
                            winning_trades += 1
                        total_trades += 1
                        
                        logging.info(f"Exited {current_position} position: P&L={pnl:.2f}, Reason=stop_triggered")
                        current_position = None
                        entry_price = None
                        position_size = 0
                        days_held = 0
                        trailing_stop_price = None
                
                # Check for new trade entry
                if not current_position:
                    # Skip if volatility is too high
                    if current_vol > 0.004:
                        logging.info(f"Skipping trade due to high volatility: {current_vol:.4f}")
                        continue
                    
                    if current_signal == 'LONG':
                        current_position = 'LONG'
                        entry_price = current_price
                        position_size = signals.iloc[i]['position_size']
                        trailing_stop_price = current_price * (1 - trailing_stop_pct)
                        logging.info(f"Entered LONG position at {entry_price:.2f}")
                        
                    elif current_signal == 'SHORT':
                        current_position = 'SHORT'
                        entry_price = current_price
                        position_size = signals.iloc[i]['position_size']
                        trailing_stop_price = current_price * (1 + trailing_stop_pct)
                        logging.info(f"Entered SHORT position at {entry_price:.2f}")
                
                # Update trailing stop
                if current_position == 'LONG':
                    trailing_stop_price = max(trailing_stop_price, current_price * (1 - trailing_stop_pct))
                elif current_position == 'SHORT':
                    trailing_stop_price = min(trailing_stop_price, current_price * (1 + trailing_stop_pct))
                
                days_held += 1
            
            # Calculate final metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 else 0
            profit_factor = winning_trades / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
            
            # Log metrics
            logger.info("Trading metrics:")
            logger.info(f"Win rate: {win_rate:.2%}")
            logger.info(f"Total trades: {total_trades}")
            logger.info(f"Total P&L: {total_pnl:.2f}")
            logger.info(f"Average trade duration: {days_held/total_trades if total_trades > 0 else 0:.1f}")
            logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
            logger.info(f"Max drawdown: {max(current_drawdown, 0):.2f}")
            logger.info(f"Profit factor: {profit_factor:.2f}")
            
            # Log price and return statistics
            logger.info("Sample prices (first 10):")
            logger.info(prices.head(10))
            logger.info("Price statistics:")
            logger.info(prices.describe())
            logger.info("Sample daily market returns (first 10):")
            logger.info(daily_returns.head(10))
            logger.info("Daily returns statistics:")
            logger.info(daily_returns.describe())
            
            return {
                'win_rate': win_rate,
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'avg_trade_duration': days_held/total_trades if total_trades > 0 else 0.0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max(current_drawdown, 0),
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            raise
        
    def _simulate_equity(
        self, 
        signals: List[dict], 
        prices: pd.Series,
        initial_equity: float = 1.0
    ) -> Dict[str, float]:
        """
        Simulate equity curve from signals.
        
        Args:
            signals: List of trading signals
            prices: Series of close prices
            initial_equity: Initial equity value
            
        Returns:
            Dictionary containing window return, cumulative return, max drawdown, and Sharpe ratio
        """
        try:
            # Log sample prices
            logger.info("Sample prices (first 10):")
            logger.info(prices.head(10))
            logger.info("Price statistics:")
            logger.info(prices.describe())
            
            # Calculate daily market returns
            daily_returns = prices.pct_change().fillna(0)
            
            # Log daily returns
            logger.info("Sample daily market returns (first 10):")
            logger.info(daily_returns.head(10))
            logger.info("Daily returns statistics:")
            logger.info(daily_returns.describe())
            
            # Initialize variables
            equity = initial_equity
            equity_curve = [equity]
            position = 0
            win_trades = 0
            total_trades = 0
            
            # Simulate trading
            for i in range(1, len(signals)):
                signal = signals[i]['trading_signal']
                size = signals[i]['position_size']
                market_return = daily_returns.iloc[i]
                
                # Calculate position return
                if signal == "LONG":
                    position_return = market_return * size
                    total_trades += 1
                    if market_return > 0:
                        win_trades += 1
                elif signal == "SHORT":
                    position_return = -market_return * size
                    total_trades += 1
                    if market_return < 0:
                        win_trades += 1
                else:
                    position_return = 0
                    
                # Update equity
                equity *= (1 + position_return - abs(position_return) * self.transaction_cost)
                equity_curve.append(equity)
            
            # Calculate metrics
            window_return = (equity_curve[-1] / equity_curve[0]) - 1
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
            
            # Log trading statistics
            logger.info(f"Total trades: {total_trades}")
            logger.info(f"Win trades: {win_trades}")
            logger.info(f"Win rate: {(win_trades/total_trades*100):.2f}%" if total_trades > 0 else "No trades")
            
            return {
                'window_return': window_return,
                'cumulative_return': equity_curve[-1],
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"Error simulating equity: {str(e)}")
            return {
                'window_return': 0.0,
                'cumulative_return': initial_equity,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve[1:]:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio from equity curve."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) < 2:
            return 0.0
            
        # Annualize Sharpe ratio (assuming daily returns)
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        return sharpe
        
    def _plot_equity_curve(self, equity_curve: pd.DataFrame, output_path: str) -> None:
        """Plot and save equity curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve['date'], equity_curve['equity'])
        plt.title('Walk-Forward Test Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def _plot_rolling_metrics(
        self,
        results: pd.DataFrame,
        output_path: str,
        window: int = 30
    ) -> None:
        """Plot rolling accuracy and win rate."""
        plt.figure(figsize=(12, 6))
        results['rolling_accuracy'] = results['accuracy'].rolling(window).mean()
        results['rolling_win_rate'] = results['win_rate'].rolling(window).mean()
        
        plt.plot(results['window_end'], results['rolling_accuracy'], label='Accuracy')
        plt.plot(results['window_end'], results['rolling_win_rate'], label='Win Rate')
        plt.title('Rolling Accuracy and Win Rate')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def _plot_regime_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str
    ) -> None:
        """Plot regime prediction heatmap."""
        confusion_matrix = pd.crosstab(
            pd.Series(y_true, name='True'),
            pd.Series(y_pred, name='Predicted')
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Regime Prediction Heatmap')
        plt.savefig(output_path)
        plt.close()
        
    def _plot_confidence_histogram(
        self,
        probs: np.ndarray,
        output_path: str
    ) -> None:
        """Plot confidence histogram per regime."""
        plt.figure(figsize=(12, 6))
        for i, regime in enumerate(['bearish', 'neutral', 'bullish']):
            plt.hist(probs[:, i], bins=20, alpha=0.5, label=regime)
        plt.title('Confidence Distribution by Regime')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def run_walk_forward(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        step_size: int = 30,
        sequence_length: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run walk-forward test.
        
        Args:
            train_start: Training start date
            train_end: Training end date
            test_start: Test start date
            test_end: Test end date
            step_size: Number of days in each window
            sequence_length: Length of input sequences
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        results = []
        current_test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)
        
        while current_test_start <= test_end:
            logger.info(f"Processing window starting at {current_test_start}")
            
            # Prepare data for current window
            features, targets, close_prices = self._prepare_data(
                start_date=current_test_start,
                sequence_length=sequence_length
            )
            
            if features is None or len(features) < sequence_length:
                logger.warning(f"Insufficient data for window starting at {current_test_start}")
                current_test_start += timedelta(days=step_size)
                continue
                
            # Create sequences
            sequences = []
            for i in range(len(features) - sequence_length + 1):
                seq = features.iloc[i:i + sequence_length].values
                sequences.append(seq)
            
            # Convert to tensor and get predictions
            inputs = torch.FloatTensor(sequences).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Get confidence scores
                confidence_scores = torch.max(probabilities, dim=1)[0]
                
            # Convert to numpy
            predictions = predictions.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            confidence_scores = confidence_scores.cpu().numpy()
            
            # Generate trading signals
            signals = []
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                if conf >= self.confidence_threshold:
                    if pred == 0:  # Bearish
                        signal = "SHORT"
                        size = self.base_size * conf
                    elif pred == 2:  # Bullish
                        signal = "LONG"
                        size = self.base_size * conf
                    else:  # Neutral
                        signal = "STAY_OUT"
                        size = 0.0
                else:
                    signal = "STAY_OUT"
                    size = 0.0
                    
                signals.append({
                    'trading_signal': signal,
                    'position_size': size,
                    'confidence': conf
                })
            
            # Calculate metrics
            window_metrics = self._calculate_metrics(
                y_true=targets.iloc[sequence_length-1:].values,
                y_pred=predictions,
                probs=probabilities,
                signals=pd.DataFrame(signals),
                close_prices=close_prices.iloc[sequence_length-1:]
            )
            
            # Calculate returns
            window_returns = self._simulate_equity(
                signals=signals,
                prices=close_prices.iloc[sequence_length-1:],
                initial_equity=1.0 if not results else results[-1]['cumulative_return']
            )
            
            # Store results
            window_results = {
                'window_start': current_test_start.strftime('%Y-%m-%d'),
                'window_end': (current_test_start + timedelta(days=step_size)).strftime('%Y-%m-%d'),
                'true_regime': targets.iloc[sequence_length-1:].values.tolist(),
                'pred_regime': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'signals': signals,
                'window_return': window_returns['window_return'],
                'cumulative_return': window_returns['cumulative_return'],
                'max_drawdown': window_returns['max_drawdown'],
                'sharpe_ratio': window_returns['sharpe_ratio'],
                **window_metrics
            }
            
            results.append(window_results)
            current_test_start += timedelta(days=step_size)
            
        return pd.DataFrame(results)
        
    def save_results(
        self,
        results: pd.DataFrame,
        output_dir: str,
        plot_equity: bool = True
    ) -> None:
        """Save walk-forward test results."""
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'walk_forward_{timestamp}')
        os.makedirs(output_path, exist_ok=True)
        
        # Save results
        results.to_csv(os.path.join(output_path, 'results.csv'), index=False)
        
        # Save summary statistics
        summary = {
            'total_windows': len(results),
            'avg_accuracy': results['accuracy'].mean(),
            'avg_precision': results['precision'].mean(),
            'avg_recall': results['recall'].mean(),
            'avg_f1': results['f1'].mean(),
            'avg_win_rate': results['win_rate'].mean(),
            'total_return': results['cumulative_return'].iloc[-1],
            'max_drawdown': results['max_drawdown'].max(),
            'avg_sharpe': results['sharpe_ratio'].mean()
        }
        with open(os.path.join(output_path, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Generate and save plots
        if plot_equity:
            self._plot_equity_curve(
                equity_curve=results['equity_curve'].apply(lambda x: pd.Series(x, index=['date', 'equity'])),
                output_path=os.path.join(output_path, 'equity_curve.png')
            )
            self._plot_rolling_metrics(
                results=results,
                output_path=os.path.join(output_path, 'rolling_metrics.png')
            )
            self._plot_regime_heatmap(
                y_true=results['true_regime'].values,
                y_pred=results['pred_regime'].values,
                output_path=os.path.join(output_path, 'regime_heatmap.png')
            )
            self._plot_confidence_histogram(
                probs=results['probabilities'].values,
                output_path=os.path.join(output_path, 'confidence_histogram.png')
            )
            
        logger.info(f"Results saved to {output_path}")
        
    def plot_equity_curve(self, results: pd.DataFrame, output_path: str):
        """Plot equity curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(results['cumulative_return'], label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def plot_rolling_metrics(self, results: pd.DataFrame, output_path: str):
        """Plot rolling metrics."""
        plt.figure(figsize=(12, 8))
        
        # Plot rolling accuracy
        plt.subplot(2, 1, 1)
        plt.plot(results['accuracy'].rolling(window=30).mean(), label='Rolling Accuracy')
        plt.title('30-Day Rolling Accuracy')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot rolling win rate
        plt.subplot(2, 1, 2)
        plt.plot(results['win_rate'].rolling(window=30).mean(), label='Rolling Win Rate')
        plt.title('30-Day Rolling Win Rate')
        plt.xlabel('Time')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_regime_heatmap(self, results: pd.DataFrame, output_path: str):
        """Plot regime prediction heatmap."""
        plt.figure(figsize=(10, 8))
        
        # Create confusion matrix
        y_true = np.concatenate(results['true_regime'].values)
        y_pred = np.concatenate(results['pred_regime'].values)
        conf_matrix = pd.crosstab(y_true, y_pred, normalize='index')
        
        # Plot heatmap
        sns.heatmap(conf_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Regime Prediction Heatmap')
        plt.xlabel('Predicted Regime')
        plt.ylabel('True Regime')
        
        plt.savefig(output_path)
        plt.close()
        
    def plot_confidence_histogram(self, results: pd.DataFrame, output_path: str):
        """Plot confidence score histogram."""
        plt.figure(figsize=(10, 6))
        
        # Plot histogram for each regime
        confidence_scores = np.concatenate(results['confidence_scores'].values)
        pred_regimes = np.concatenate(results['pred_regime'].values)
        
        for regime, label in [(0, 'Bearish'), (1, 'Neutral'), (2, 'Bullish')]:
            mask = pred_regimes == regime
            plt.hist(confidence_scores[mask], bins=30, alpha=0.5, label=label)
        
        plt.title('Prediction Confidence Distribution by Regime')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_path)
        plt.close()
        
    def save_results(self, results: pd.DataFrame, output_dir: str):
        """Save test results."""
        # Save detailed results
        results_path = os.path.join(output_dir, 'walk_forward_results.csv')
        results.to_csv(results_path, index=False)
        
        # Calculate and save summary metrics
        summary = {
            'total_windows': len(results),
            'avg_accuracy': results['accuracy'].mean(),
            'avg_win_rate': results['win_rate'].mean(),
            'total_return': results['cumulative_return'].iloc[-1],
            'max_drawdown': results['max_drawdown'].min(),
            'sharpe_ratio': results['sharpe_ratio'].mean(),
            'total_trades': results['total_trades'].sum(),
            'win_trades': results['winning_trades'].sum()
        }
        
        summary_path = os.path.join(output_dir, 'summary_metrics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Generate plots
        self.plot_equity_curve(results, os.path.join(output_dir, 'equity_curve.png'))
        self.plot_rolling_metrics(results, os.path.join(output_dir, 'rolling_metrics.png'))
        self.plot_regime_heatmap(results, os.path.join(output_dir, 'regime_heatmap.png'))
        self.plot_confidence_histogram(results, os.path.join(output_dir, 'confidence_histogram.png'))
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Summary metrics: {json.dumps(summary, indent=2)}")

def main():
    """Run walk-forward test."""
    # Configuration
    config = {
        'model_path': os.path.join('..', 'models', 'lstm', 'regime_lstm.pth'),
        'data_path': os.path.join('..', 'data'),
        'initial_capital': 100000.0,
        'transaction_cost': 0.001,
        'confidence_threshold': 0.85,
        'base_size': 1.0,
        'train_start': '2024-01-01',
        'train_end': '2024-06-30',
        'test_start': '2024-07-01',
        'test_end': '2024-12-31',
        'step_size': 30
    }
    
    # Initialize walk-forward tester
    tester = WalkForwardTester(
        model_path=config['model_path'],
        data_path=config['data_path'],
        initial_capital=config['initial_capital'],
        transaction_cost=config['transaction_cost'],
        confidence_threshold=config['confidence_threshold'],
        base_size=config['base_size']
    )
    
    # Run walk-forward test
    results = tester.run_walk_forward(
        train_start=config['train_start'],
        train_end=config['train_end'],
        test_start=config['test_start'],
        test_end=config['test_end'],
        step_size=config['step_size']
    )
    
    # Save results
    output_dir = os.path.join('..', 'results', 'walk_forward')
    tester.save_results(results, output_dir, plot_equity=True)
    
if __name__ == '__main__':
    main() 