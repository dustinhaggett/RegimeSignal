#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forward testing script for regime identification.
Generates a visualization of market regimes with confidence levels.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import sys
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import json
from scipy import stats

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from modeling.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegimeVisualizer:
    def __init__(self):
        self.model = None
        self.scalers = None
        self.feature_names = None
        self.sequence_length = 60  # Same as training
        self.temperature = 1.0  # Default temperature for scaling
        self.training_stats = None
        
    def load_model(self):
        """Load the saved LSTM model and scalers."""
        try:
            # Get paths
            project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            model_dir = project_root / 'models' / 'lstm'
            
            # Load model
            model_path = model_dir / 'regime_lstm.pth'
            scaler_path = model_dir / 'scaler.pkl'
            feature_names_path = model_dir / 'feature_names.txt'
            stats_path = model_dir / 'training_stats.json'
            
            # Load feature names
            with open(feature_names_path, 'r') as f:
                self.feature_names = f.read().splitlines()
            
            # Load training statistics if available
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
            else:
                logger.warning("Training statistics not found. Feature drift detection will be limited.")
            
            # Load scalers
            self.scalers = joblib.load(scaler_path)
            
            # Initialize and load model
            from modeling.lstm_trainer import RegimeAwareLSTM
            self.model = RegimeAwareLSTM(
                input_size=len(self.feature_names),
                hidden_size=64,
                num_layers=2
            )
            self.model.load_state_dict(torch.load(model_path, weights_only=False))
            self.model.eval()
            
            logger.info("Successfully loaded model and scalers")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def prepare_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Prepare data for model prediction."""
        try:
            # Debug logging for feature names
            logger.info("\nFeature Names Diagnostic:")
            logger.info(f"Expected features: {self.feature_names}")
            logger.info("\nAvailable columns in data:")
            logger.info(f"{data.columns.tolist()}")
            
            # Check if all features are present
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Debug logging for feature data
            logger.info("\nFeature Data Sample (first 5 rows):")
            logger.info(f"\n{data[self.feature_names].head()}")
            
            # Extract features
            X = data[self.feature_names].values
            
            # Debug logging for scaling
            logger.info("\nScaling Diagnostic:")
            logger.info(f"Number of scalers: {len(self.scalers)}")
            logger.info(f"Scaler keys: {list(self.scalers.keys())}")
            
            # Scale features by regime
            X_scaled = np.zeros_like(X)
            regimes = data['market_regime'].values
            unique_regimes = np.unique(regimes)
            logger.info(f"Unique regimes in data: {unique_regimes}")
            
            for regime, scaler in self.scalers.items():
                mask = regimes == regime
                if mask.any():
                    logger.info(f"\nScaling for regime {regime}:")
                    logger.info(f"Number of samples: {np.sum(mask)}")
                    try:
                        X_scaled[mask] = scaler.transform(X[mask])
                    except Exception as e:
                        logger.error(f"Error scaling regime {regime}: {str(e)}")
                        logger.error(f"Input shape: {X[mask].shape}")
                        logger.error(f"Scaler feature names: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'No feature names'}")
                        raise
            
            # Create sequences
            X_sequences = []
            for i in range(len(X_scaled) - self.sequence_length):
                X_sequences.append(X_scaled[i:i + self.sequence_length])
            
            X_sequences = np.array(X_sequences)
            
            # Debug logging for sequence shape
            logger.info(f"\nFinal sequence shape: {X_sequences.shape}")
            
            # Convert to tensor
            return torch.FloatTensor(X_sequences)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def predict_regimes(self, data: pd.DataFrame) -> tuple:
        """Predict regimes using the LSTM model."""
        try:
            if self.model is None:
                self.load_model()
            
            # Monitor feature distributions
            drift_report = self.monitor_feature_distributions(data)
            if drift_report:
                logger.warning("Feature drift detected. Model predictions may be less reliable.")
            
            # Prepare data
            X = self.prepare_data(data)
            
            # Create dataloader
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            
            # Make predictions
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for batch in dataloader:
                    features = batch[0].to(device)
                    # Use neutral regime (1) as initial regime for prediction
                    regimes = torch.ones(features.size(0), dtype=torch.long, device=device)
                    
                    outputs = self.model(features, regimes)
                    # Apply temperature scaling to logits
                    probs = self.calibrate_predictions(outputs)
                    predicted = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            # Convert predictions from model output (0,1,2) to regime values (-1,0,1)
            predictions = np.array(all_preds)
            probabilities = np.array(all_probs)
            
            # Calculate confidence as max probability
            confidence = np.max(probabilities, axis=1)
            
            # Apply minimum confidence threshold and smoothing
            MIN_CONFIDENCE = 0.9  # Increased from 0.8 to be more conservative
            MIN_DURATION = 4  # Minimum 1 hour (4 x 15min periods)
            
            # Map model outputs (0,1,2) to regime values (-1,0,1)
            predictions = predictions.astype(int) - 1  # Convert 0,1,2 to -1,0,1
            
            # Apply smoothing to predictions
            smoothed_predictions = predictions.copy()
            smoothed_confidence = confidence.copy()
            
            for i in range(len(predictions)):
                if confidence[i] < MIN_CONFIDENCE:
                    # If confidence is low, maintain previous regime
                    if i > 0:
                        smoothed_predictions[i] = smoothed_predictions[i-1]
                        smoothed_confidence[i] = smoothed_confidence[i-1]
                
                # Ensure minimum duration
                if i >= MIN_DURATION:
                    recent_regimes = smoothed_predictions[i-MIN_DURATION:i]
                    if len(set(recent_regimes)) > 1:  # If there's variation in recent regimes
                        # Keep the most common regime
                        from collections import Counter
                        most_common = Counter(recent_regimes).most_common(1)[0][0]
                        smoothed_predictions[i-MIN_DURATION:i] = most_common
            
            # Log debug information
            logger.info(f"\nDebug Information:")
            logger.info(f"Average confidence: {np.mean(confidence):.3f}")
            logger.info(f"Confidence threshold: {MIN_CONFIDENCE}")
            logger.info(f"Number of high confidence predictions: {np.sum(confidence >= MIN_CONFIDENCE)}")
            logger.info(f"Raw prediction distribution: {np.unique(all_preds, return_counts=True)}")
            logger.info(f"Mapped regime distribution: {np.unique(predictions, return_counts=True)}")
            
            # Log feature drift summary if any
            if drift_report:
                logger.info("\nFeature Drift Summary:")
                logger.info(f"Number of features with drift: {len(drift_report)}")
                logger.info("Affected features: " + ", ".join(drift_report.keys()))
            
            return smoothed_predictions, smoothed_confidence
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            raise
            
    def generate_tradingview_indicator(self, data: pd.DataFrame, results_dir: Path) -> None:
        """Generate TradingView Pine Script indicator."""
        try:
            # Get predictions and confidence
            predictions, confidence = self.predict_regimes(data)
            
            # Create regime changes list
            regime_changes = []
            prev_regime = None
            
            # Skip first sequence_length periods due to LSTM requirements
            for idx, (timestamp, regime, conf) in enumerate(zip(
                data.index[self.sequence_length:],
                predictions,
                confidence
            )):
                # Only include regime changes with high confidence
                if regime != prev_regime and conf >= 0.7:
                    regime_changes.append((timestamp, int(regime), float(conf)))
                    prev_regime = regime
            
            # Generate Pine Script for background coloring
            pine_script = '''
//@version=5
indicator("Market Regimes", overlay=true)

// Constants
var BULLISH = 1
var NEUTRAL = 0
var BEARISH = -1

// Arrays to store regime changes
var times = array.new_int()
var regimes = array.new_int()
var confs = array.new_float()

// Initialize arrays with regime changes
if barstate.isfirst
'''
            
            # Add regime changes to Pine Script
            for timestamp, regime, conf in zip(
                data.index[self.sequence_length:],
                predictions,
                confidence
            ):
                pine_script += f"    array.push(times, {int(timestamp.timestamp())})\n"
                pine_script += f"    array.push(regimes, {int(regime)})\n"
                pine_script += f"    array.push(confs, {float(conf)})\n"
            
            # Add the regime detection and coloring logic
            pine_script += '''
// Function to get color with opacity based on confidence
get_regime_color(regime, conf) =>
    transparency = 90 - math.round(conf * 60)  // Higher confidence = more opaque
    if regime == BULLISH
        color.new(color.green, transparency)
    else if regime == BEARISH
        color.new(color.red, transparency)
    else  // NEUTRAL
        color.new(color.gray, transparency)

// Get current regime based on timestamp
get_current_regime() =>
    var int regime = NEUTRAL
    var float conf = 0.5
    var int found_idx = -1
    
    if array.size(times) > 0
        for i = array.size(times) - 1 to 0
            if time >= array.get(times, i)
                found_idx := i
                break
        
        if found_idx >= 0
            regime := array.get(regimes, found_idx)
            conf := array.get(confs, found_idx)
    [regime, conf]

// Get current regime and confidence
[regime, confidence] = get_current_regime()

// Color the background based on regime
bgcolor(get_regime_color(regime, confidence))

// Add labels for regime changes
if ta.change(regime) != 0
    label.new(
        bar_index, 
        high, 
        text = regime == BULLISH ? "↑" : regime == BEARISH ? "↓" : "→",
        color = get_regime_color(regime, confidence),
        style = label.style_label_down,
        textcolor = color.white,
        size = size.small
    )
'''
            
            # Save Pine Script
            pine_script_path = results_dir / 'regime_indicator.pine'
            with open(pine_script_path, 'w') as f:
                f.write(pine_script)
            
            # Create HTML with TradingView widget
            html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Market Regime Analysis</title>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; background: #1e222d; }}
        .container {{ width: 100%; height: 100vh; }}
        .chart-container {{ width: 100%; height: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="chart-container" id="tradingview_chart"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "width": "100%",
            "height": "100%",
            "symbol": "BINANCE:BTCUSDTPERP",
            "interval": "15",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": false,
            "save_image": false,
            "container_id": "tradingview_chart",
            "loading_screen": {{ "backgroundColor": "#1e222d" }},
            "studies": [
                {{
                    "id": "Background@tv-basicstudies",
                    "inputs": {{
                        "text": "Market Regimes",
                        "color": "#2196F3"
                    }}
                }}
            ],
            "overrides": {{
                "paneProperties.background": "#1e222d",
                "paneProperties.vertGridProperties.color": "#363c4e",
                "paneProperties.horzGridProperties.color": "#363c4e",
                "symbolWatermarkProperties.transparency": 90,
                "scalesProperties.textColor" : "#AAA"
            }},
            "studies_overrides": {{
                "volume.volume.color.0": "#eb4d5c",
                "volume.volume.color.1": "#53b987",
                "volume.volume.transparency": 70
            }},
            "custom_css_url": "https://s3.tradingview.com/chart.css",
            "drawings_access": {{ "type": "all" }},
            "enabled_features": ["study_templates"],
            "disabled_features": [
                "header_symbol_search",
                "header_screenshot",
                "header_compare",
                "header_saveload"
            ],
            "time_frames": [
                {{ "text": "1D", "resolution": "15" }},
                {{ "text": "5D", "resolution": "15" }},
                {{ "text": "1M", "resolution": "15" }}
            ]
        }});
        </script>
    </div>
    <script>
        // Add regime data
        const regimeData = [
'''

            # Add regime data points
            for timestamp, regime, conf in zip(
                data.index[self.sequence_length:],
                predictions,
                confidence
            ):
                color = '#ff4444' if regime == -1 else '#888888' if regime == 0 else '#44ff44'
                opacity = min(1.0, conf)
                html_content += f"            {{time: {int(timestamp.timestamp()*1000)}, regime: {regime}, confidence: {conf}, color: '{color}'}},\n"

            html_content += '''
        ];

        // Wait for TradingView widget to load
        setTimeout(() => {
            const chart = document.querySelector('#tradingview_chart iframe').contentWindow.chart;
            if (chart) {
                // Add background colors for regimes
                regimeData.forEach(data => {
                    chart.createMultipointShape([
                        { time: data.time, price: 0 },
                        { time: data.time + 900000, price: 1000000 }  // 15 minutes in milliseconds
                    ], {
                        backgroundColor: data.color + Math.round(data.confidence * 40).toString(16),
                        backgroundVisible: true,
                        disableSelection: true,
                        disableSave: true
                    });
                });
            }
        }, 3000);
    </script>
</body>
</html>
'''
            
            # Save HTML file
            html_path = results_dir / 'regime_analysis.html'
            with open(html_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"Generated TradingView indicator at: {pine_script_path}")
            logger.info(f"Generated HTML chart at: {html_path}")
            
            # Log regime distribution
            regime_counts = pd.Series(predictions).value_counts()
            total = len(predictions)
            
            logger.info("\nRegime Distribution:")
            for regime in [-1, 0, 1]:
                count = regime_counts.get(regime, 0)
                percentage = (count / total) * 100
                regime_name = "Bearish" if regime == -1 else "Neutral" if regime == 0 else "Bullish"
                logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
            
            # Log high-confidence regime changes
            logger.info(f"\nTotal high-confidence regime changes: {len(regime_changes)}")
            
        except Exception as e:
            logger.error(f"Error generating TradingView indicator: {str(e)}")
            raise

    def generate_tradingview_url(self, data: pd.DataFrame) -> str:
        """Generate a TradingView chart URL with regime data."""
        try:
            # Get predictions and confidence
            predictions, confidence = self.predict_regimes(data)
            
            # Create a DataFrame with the results
            results_df = pd.DataFrame({
                'timestamp': data.index[self.sequence_length:],
                'regime': predictions,
                'confidence': confidence,
                'close': data['close'].values[self.sequence_length:]
            })
            
            # Save to CSV in results directory
            project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            results_dir = project_root / 'results' / 'forward_test'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = results_dir / 'regime_data.csv'
            results_df.to_csv(csv_path)
            
            # Generate TradingView URL
            base_url = "https://www.tradingview.com/chart/"
            symbol = "BINANCE:BTCUSDTPERP"
            timeframe = "15"  # 15 minute timeframe
            
            # Create URL with overlay indicators
            url = f"{base_url}?symbol={symbol}&interval={timeframe}"
            
            logger.info(f"Generated TradingView URL: {url}")
            logger.info(f"Regime data saved to: {csv_path}")
            
            # Log regime distribution
            regime_counts = pd.Series(predictions).value_counts()
            total = len(predictions)
            
            logger.info("\nRegime Distribution:")
            for regime in [-1, 0, 1]:
                count = regime_counts.get(regime, 0)
                percentage = (count / total) * 100
                regime_name = "Bearish" if regime == -1 else "Neutral" if regime == 0 else "Bullish"
                logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
            
            return url, csv_path
            
        except Exception as e:
            logger.error(f"Error generating TradingView URL: {str(e)}")
            raise

    def generate_html_visualization(self, data: pd.DataFrame, results_dir: Path) -> None:
        """Generate interactive HTML visualization with regime backgrounds."""
        try:
            # Get predictions and confidence
            predictions, confidence = self.predict_regimes(data)
            
            # Create figure with secondary y-axis
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
            
            # Add candlestick chart with improved styling
            fig.add_trace(
                go.Candlestick(
                    x=data.index[self.sequence_length:],
                    open=data['open'][self.sequence_length:],
                    high=data['high'][self.sequence_length:],
                    low=data['low'][self.sequence_length:],
                    close=data['close'][self.sequence_length:],
                    name="BTCUSDT",
                    increasing=dict(line=dict(color='#53b987', width=1), fillcolor='#53b987'),
                    decreasing=dict(line=dict(color='#eb4d5c', width=1), fillcolor='#eb4d5c'),
                    hoverinfo='none'  # We'll use a custom hover template
                )
            )
            
            # Add regime background colors with improved opacity
            for i, (timestamp, regime, conf) in enumerate(zip(
                data.index[self.sequence_length:],
                predictions,
                confidence
            )):
                # Calculate end time
                end_time = data.index[self.sequence_length + i + 1] if i < len(predictions) - 1 else timestamp + pd.Timedelta(minutes=15)
                
                # Set color based on regime with improved opacity
                if regime == 1:  # Bullish
                    color = f'rgba(83, 185, 135, {min(conf * 0.4, 0.4)})'  # Green
                elif regime == -1:  # Bearish
                    color = f'rgba(235, 77, 92, {min(conf * 0.4, 0.4)})'  # Red
                else:  # Neutral
                    color = f'rgba(128, 128, 128, {min(conf * 0.3, 0.3)})'  # Gray
                
                # Add background shape
                fig.add_shape(
                    type="rect",
                    x0=timestamp,
                    x1=end_time,
                    y0=data['low'][self.sequence_length:].min() * 0.999,  # Slight padding
                    y1=data['high'][self.sequence_length:].max() * 1.001,  # Slight padding
                    fillcolor=color,
                    opacity=1,
                    layer="below",
                    line_width=0,
                )
                
                # Add regime change markers
                if i == 0 or predictions[i] != predictions[i-1]:
                    marker = "↑" if regime == 1 else "↓" if regime == -1 else "→"
                    marker_color = '#53b987' if regime == 1 else '#eb4d5c' if regime == -1 else '#808080'
                    if conf >= 0.8:  # Only show markers for high confidence changes
                        fig.add_annotation(
                            x=timestamp,
                            y=data['high'][self.sequence_length:].max() * 1.001,
                            text=marker,
                            showarrow=False,
                            font=dict(size=16, color=marker_color),
                            bgcolor='rgba(30, 34, 45, 0.9)',
                            bordercolor=marker_color,
                            borderwidth=1,
                            borderpad=4
                        )
            
            # Add Bollinger Bands with improved styling
            bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
            fig.add_trace(
                go.Scatter(
                    x=data.index[self.sequence_length:],
                    y=bb.bollinger_mavg()[self.sequence_length:],
                    name="BB Middle",
                    line=dict(color='rgba(255, 255, 255, 0.8)', width=1, dash='dot'),
                    hoverinfo='none'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index[self.sequence_length:],
                    y=bb.bollinger_hband()[self.sequence_length:],
                    name="BB Upper",
                    line=dict(color='rgba(255, 255, 255, 0.4)', width=1, dash='dot'),
                    hoverinfo='none'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index[self.sequence_length:],
                    y=bb.bollinger_lband()[self.sequence_length:],
                    name="BB Lower",
                    line=dict(color='rgba(255, 255, 255, 0.4)', width=1, dash='dot'),
                    hoverinfo='none'
                )
            )
            
            # Update layout with improved styling
            fig.update_layout(
                title=dict(
                    text="Market Regime Analysis",
                    font=dict(size=24, color='white'),
                    x=0.5,
                    y=0.95
                ),
                template="plotly_dark",
                plot_bgcolor='#1e222d',
                paper_bgcolor='#1e222d',
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    type="date",
                    gridcolor='rgba(54, 60, 78, 0.5)',
                    showgrid=True,
                    gridwidth=1,
                    title="",
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="Price (USDT)",
                    tickformat=",.0f",
                    gridcolor='rgba(54, 60, 78, 0.5)',
                    showgrid=True,
                    gridwidth=1,
                    tickfont=dict(size=12),
                    title_font=dict(size=14)
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(30, 34, 45, 0.9)',
                    bordercolor='rgba(54, 60, 78, 0.5)',
                    borderwidth=1,
                    font=dict(size=12)
                ),
                margin=dict(l=50, r=50, t=50, b=50),
                hoverlabel=dict(
                    bgcolor='rgba(30, 34, 45, 0.9)',
                    font_size=12,
                    font_family="monospace"
                ),
                hovermode='x unified'
            )
            
            # Add regime distribution info with improved styling
            predictions_df = pd.DataFrame({
                'regime': predictions,
                'confidence': confidence
            }, index=data.index[self.sequence_length:])
            
            # Calculate regime counts for all predictions
            regime_counts = predictions_df['regime'].value_counts()
            total = len(predictions_df)
            
            # Calculate regime counts for high confidence predictions
            high_conf_df = predictions_df[predictions_df['confidence'] >= 0.8]
            high_conf_counts = high_conf_df['regime'].value_counts()
            high_conf_total = len(high_conf_df)
            
            # Ensure all regime values are present in counts
            for regime in [-1, 0, 1]:
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                if regime not in high_conf_counts:
                    high_conf_counts[regime] = 0
            
            # Sort the counts
            regime_counts = regime_counts.sort_index()
            high_conf_counts = high_conf_counts.sort_index()
            
            regime_info = "<br>".join([
                "<b>All Predictions:</b>",
                f"Bearish: {regime_counts.get(-1, 0)} periods ({regime_counts.get(-1, 0)/total*100:.1f}%)",
                f"Neutral: {regime_counts.get(0, 0)} periods ({regime_counts.get(0, 0)/total*100:.1f}%)",
                f"Bullish: {regime_counts.get(1, 0)} periods ({regime_counts.get(1, 0)/total*100:.1f}%)",
                "",
                "<b>High Confidence (>80%):</b>",
                f"Bearish: {high_conf_counts.get(-1, 0)} periods ({high_conf_counts.get(-1, 0)/high_conf_total*100:.1f}% of high conf)" if high_conf_total > 0 else "Bearish: 0 periods (0.0%)",
                f"Neutral: {high_conf_counts.get(0, 0)} periods ({high_conf_counts.get(0, 0)/high_conf_total*100:.1f}% of high conf)" if high_conf_total > 0 else "Neutral: 0 periods (0.0%)",
                f"Bullish: {high_conf_counts.get(1, 0)} periods ({high_conf_counts.get(1, 0)/high_conf_total*100:.1f}% of high conf)" if high_conf_total > 0 else "Bullish: 0 periods (0.0%)"
            ])
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.99,
                text=regime_info,
                showarrow=False,
                font=dict(size=12, color="white", family="monospace"),
                bgcolor='rgba(30, 34, 45, 0.9)',
                bordercolor='rgba(54, 60, 78, 0.5)',
                borderwidth=1,
                align="right",
                borderpad=4
            )
            
            # Save HTML file with config options
            html_path = results_dir / 'regime_analysis.html'
            fig.write_html(
                html_path,
                config={
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'displaylogo': False
                }
            )
            
            logger.info(f"Generated interactive visualization at: {html_path}")
            
            # Log regime distribution
            logger.info("\nRegime Distribution (All Predictions):")
            for regime in [-1, 0, 1]:
                count = regime_counts.get(regime, 0)
                percentage = (count / total) * 100
                regime_name = "Bearish" if regime == -1 else "Neutral" if regime == 0 else "Bullish"
                logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
            
            if high_conf_total > 0:
                logger.info("\nRegime Distribution (High Confidence >80%):")
                for regime in [-1, 0, 1]:
                    count = high_conf_counts.get(regime, 0)
                    percentage = (count / high_conf_total) * 100
                    regime_name = "Bearish" if regime == -1 else "Neutral" if regime == 0 else "Bullish"
                    logger.info(f"{regime_name}: {count} periods ({percentage:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error generating HTML visualization: {str(e)}")
            raise

    def monitor_feature_distributions(self, data: pd.DataFrame) -> dict:
        """
        Monitor feature distributions for drift compared to training data.
        Returns a dictionary of features with significant drift.
        """
        from scipy import stats
        drift_report = {}
        
        if self.training_stats is None:
            logger.warning("No training statistics available for drift detection.")
            return drift_report
        
        for feature in self.feature_names:
            current_stats = {
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std()),
                'skew': float(stats.skew(data[feature].dropna())),
                'kurtosis': float(stats.kurtosis(data[feature].dropna())),
                'q1': float(data[feature].quantile(0.25)),
                'q3': float(data[feature].quantile(0.75))
            }
            
            # Compare with training statistics
            if feature in self.training_stats:
                train_stats = self.training_stats[feature]
                
                # Calculate z-score for mean shift
                mean_zscore = abs(current_stats['mean'] - train_stats['mean']) / train_stats['std']
                
                # Calculate distribution overlap using IQR
                iqr_overlap = min(current_stats['q3'], train_stats['q3']) - max(current_stats['q1'], train_stats['q1'])
                iqr_total = max(current_stats['q3'], train_stats['q3']) - min(current_stats['q1'], train_stats['q1'])
                dist_overlap = max(0, iqr_overlap / iqr_total) if iqr_total > 0 else 0
                
                if mean_zscore > 2 or dist_overlap < 0.5:  # Significant drift thresholds
                    drift_report[feature] = {
                        'mean_shift': mean_zscore,
                        'distribution_overlap': dist_overlap,
                        'current': current_stats,
                        'training': train_stats
                    }
        
        if drift_report:
            logger.warning(f"Detected feature drift in {len(drift_report)} features")
            for feature, stats in drift_report.items():
                logger.warning(f"Feature '{feature}' shows significant drift:")
                logger.warning(f"  Mean Z-score: {stats['mean_shift']:.2f}")
                logger.warning(f"  Distribution overlap: {stats['distribution_overlap']:.2f}")
        
        return drift_report

    def calibrate_predictions(self, logits: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """
        Apply temperature scaling to model predictions.
        A higher temperature makes the model more conservative (reduces confidence).
        """
        if temperature is None:
            temperature = self.temperature
        
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=1)

def calibration_check(y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray, 
                     n_bins: int = 10, plot_title: str = "Model Calibration") -> None:
    """
    Evaluate how well the model's prediction confidence matches its actual accuracy.
    
    Args:
        y_true: Ground truth labels (-1, 0, 1)
        y_pred: Predicted labels (-1, 0, 1)
        y_conf: Confidence scores for predictions (0-1)
        n_bins: Number of bins to use for confidence ranges
        plot_title: Title for the calibration plot
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        # Create confidence bins
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize arrays for storing bin statistics
        accuracies = []
        avg_confidences = []
        bin_counts = []
        
        # Calculate statistics for each bin
        print("\nCalibration Statistics:")
        print(f"{'Conf Range':^15} | {'Accuracy':^10} | {'Avg Conf':^10} | {'Count':^8}")
        print("-" * 47)
        
        for i in range(n_bins):
            bin_mask = (y_conf >= bin_edges[i]) & (y_conf < bin_edges[i+1])
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(y_pred[bin_mask] == y_true[bin_mask])
                bin_conf = np.mean(y_conf[bin_mask])
                bin_count = np.sum(bin_mask)
                
                accuracies.append(bin_acc)
                avg_confidences.append(bin_conf)
                bin_counts.append(bin_count)
                
                print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}".center(15) + 
                      f" | {bin_acc:8.3f} | {bin_conf:8.3f} | {bin_count:6d}")
        
        # Plot calibration curve
        plt.figure(figsize=(10, 6))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot actual calibration curve
        plt.plot(avg_confidences, accuracies, 'bo-', label='Model Calibration')
        
        # Add error bars based on bin counts
        errors = [np.sqrt((acc * (1 - acc)) / count) if count > 0 else 0 
                 for acc, count in zip(accuracies, bin_counts)]
        plt.errorbar(avg_confidences, accuracies, yerr=errors, fmt='none', color='b', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Empirical Accuracy')
        plt.title(plot_title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = Path(__file__).parent.parent.parent / 'results' / 'forward_test' / 'calibration_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nCalibration plot saved to: {plot_path}")
        
        # Calculate overall statistics
        overall_acc = np.mean(y_pred == y_true)
        overall_conf = np.mean(y_conf)
        ece = np.sum([count * np.abs(acc - conf) / len(y_true) 
                     for acc, conf, count in zip(accuracies, avg_confidences, bin_counts)])
        
        print("\nOverall Statistics:")
        print(f"Overall Accuracy: {overall_acc:.3f}")
        print(f"Average Confidence: {overall_conf:.3f}")
        print(f"Expected Calibration Error: {ece:.3f}")
        
    except Exception as e:
        logger.error(f"Error in calibration check: {str(e)}")
        raise

def main():
    """Main function to run regime analysis."""
    try:
        # Initialize regime visualizer
        visualizer = RegimeVisualizer()
        
        # Load feature engineering
        feature_engineer = FeatureEngineer()
        data = feature_engineer.load_data()
        
        # Get last 30 days of data
        cutoff_date = data.index.max() - timedelta(days=30)
        recent_data = data[data.index > cutoff_date].copy()
        
        logger.info(f"Analyzing data from {recent_data.index.min()} to {recent_data.index.max()}")
        logger.info(f"Total periods: {len(recent_data)}")
        
        # Create results directory
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        results_dir = project_root / 'results' / 'forward_test'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Try different temperature values for calibration
        temperatures = [1.0, 1.5, 2.0, 2.5, 3.0]
        best_temp = 1.0
        best_ece = float('inf')
        
        logger.info("\nCalibration Temperature Search:")
        for temp in temperatures:
            visualizer.temperature = temp
            predictions, confidence = visualizer.predict_regimes(recent_data)
            
            # Calculate calibration metrics
            y_true = recent_data['market_regime'].values[visualizer.sequence_length:]
            
            # Calculate ECE directly here
            n_bins = 10
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            total_samples = len(y_true)
            
            for i in range(n_bins):
                bin_mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i+1])
                if np.sum(bin_mask) > 0:
                    bin_acc = np.mean(predictions[bin_mask] == y_true[bin_mask])
                    bin_conf = np.mean(confidence[bin_mask])
                    bin_count = np.sum(bin_mask)
                    ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
            
            logger.info(f"Temperature {temp:.1f} - ECE: {ece:.3f}")
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
                
            # Generate calibration plot for this temperature
            calibration_check(y_true, predictions, confidence, 
                            plot_title=f"Calibration (T={temp:.1f})")
        
        logger.info(f"\nBest temperature found: {best_temp:.1f} (ECE: {best_ece:.3f})")
        
        # Use best temperature for final predictions
        visualizer.temperature = best_temp
        predictions, confidence = visualizer.predict_regimes(recent_data)
        
        # Generate visualization
        visualizer.generate_html_visualization(recent_data, results_dir)
        
        # Save current feature statistics for future drift detection
        feature_stats = {}
        for feature in visualizer.feature_names:
            feature_stats[feature] = {
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std()),
                'skew': float(stats.skew(data[feature].dropna())),
                'kurtosis': float(stats.kurtosis(data[feature].dropna())),
                'q1': float(data[feature].quantile(0.25)),
                'q3': float(data[feature].quantile(0.75))
            }
        
        stats_path = project_root / 'models' / 'lstm' / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f, indent=4)
        
        logger.info("\nTo view the analysis:")
        logger.info(f"Open the HTML file: {results_dir / 'regime_analysis.html'}")
        logger.info(f"Check calibration plot: {results_dir / 'calibration_curve.png'}")
        
    except Exception as e:
        logger.error(f"Error in regime analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 