# RegimeSignal: Market Regime Detection in Cryptocurrency Markets

**Author:** Dustin M. Haggett  
**Course:** CpE646 – Pattern Recognition and Classification  
**Project Type:** Individual Research Project  
**Model:** LSTM with Multi-Head Attention  
**Target Asset:** BTC/USDT Perpetual Futures (15-minute intervals)

---

## Overview

RegimeSignal is a deep learning pipeline for classifying market regimes—Bullish, Bearish, and Neutral—in the BTC/USDT perpetual futures market. This project combines classical pattern recognition techniques (e.g., PCA, mutual information, random forest) with modern sequence models (LSTM + Attention). It aims to generate adaptive trading signals based on high-frequency technical indicators and dynamic market conditions.

The final model achieved **89.1% classification accuracy**, with strong results in bearish and bullish regimes.

---

## Key Features

- Feature engineering with 23 technical indicators across trend, momentum, volatility, volume, and risk
- Dimensionality reduction using PCA (96.6% variance explained)
- Bi-directional LSTM model with multi-head attention mechanism
- Class imbalance handling via Focal Loss and class weighting
- Walk-forward validation and regime transition matrix analysis
- Confidence-aware trading strategy with position sizing

---

## Project Structure

```
.
├── code/
│   ├── modeling/          # Model architecture, training, backtesting, strategy
│   ├── analysis/          # Data diagnostics, forward testing, visualization
│   └── data_pipeline/     # Raw data collection and processing
├── data/
│   └── processed/         # Cleaned and labeled data for training/testing
├── models/
│   └── lstm/              # Saved models, scalers, training history
├── environment.yml        # Conda environment file
└── README.md
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate regime-trader
```

---

## How to Run

1. **Collect latest market data:**
   ```bash
   python code/data_pipeline/data_collector.py
   ```
2. **Generate features and regime labels:**
   ```bash
   python code/modeling/feature_engineering.py
   ```
3. **Train the LSTM model:**
   ```bash
   python code/modeling/lstm_trainer.py
   ```
4. **Evaluate performance:**
   ```bash
   python code/modeling/walk_forward_test.py
   # or
   python code/modeling/backtesting.py
   ```
5. **Deploy the trading strategy:**
   ```bash
   python code/modeling/trading_strategy.py
   ```

---

## Model Artifacts

| File                           | Description                                 |
|--------------------------------|---------------------------------------------|
| regime_lstm_calibrated.pth     | Trained LSTM model weights                  |
| scaler.pkl                     | Saved StandardScaler for input normalization|
| training_history_calibrated.json | Training log and evaluation metrics        |
| feature_names.txt              | List of input features used during training |
| processed_data.csv             | Preprocessed dataset for training           |
| walk_forward_data.csv          | Labeled splits for walk-forward testing     |

---

## Evaluation Summary

- **Test Accuracy:** 89.1%
- **Bearish Accuracy:** 91.64%
- **Neutral Accuracy:** 85.02%
- **Bullish Accuracy:** 91.17%

Most classification errors occurred within the neutral regime. Confidence scores were useful for filtering uncertain predictions and improving risk-adjusted decision making.

---

## Theoretical Context

This project applies and extends several core topics from CpE646: Pattern Recognition and Classification:

- Supervised learning with softmax-based decision boundaries
- Dimensionality reduction using Principal Component Analysis
- Feature importance via mutual information and random forest
- Evaluation with confusion matrices and regime transition analysis
- Handling class imbalance with Focal Loss and weighting
- Time-series classification using sequential deep learning models

---

## Future Work

- Expand to multi-asset regime detection (e.g., ETH, SOL)
- Integrate sentiment and news-based features
- Test transformer-based sequence models
- Build real-time inference and alerting dashboard

---

## License

This project is intended for academic, research, and non-commercial use.