# RegimeAlpha

A regime-aware algorithmic trading system for cryptocurrency markets using time series analysis and greedy decision algorithms. This project identifies different market regimes (trending, ranging, volatile) and applies optimized trading strategies for each condition with a focus on high-leverage trading scenarios.

## Project Team
- Dustin Haggett
- Lambert Kongnyuy
- Mateusz Marciniak

## Project Overview

RegimeAlpha implements a machine learning-based trading system that:
1. Identifies market regimes using clustering/HMM techniques
2. Predicts price movements using LSTM/TCN neural networks
3. Makes trading decisions using a greedy algorithm optimized for each regime
4. Backtests performance on historical cryptocurrency data

The system is specifically designed for high-leverage trading (30x-50x) in the BTC/USDT market using 15-minute time frames.

## Repository Structure

```
RegimeAlpha/
├── code/
│   ├── data_pipeline/      # Data collection and preprocessing
│   ├── regime_detection/   # Market regime identification
│   ├── modeling/           # ML models for price prediction
│   ├── trading_logic/      # Trading strategy implementation
│   ├── backtesting/        # Backtesting engine
│   ├── visualization/      # Results visualization
│   └── main.py             # Main execution script
├── documents/              # Paper and presentation materials
└── references/             # Related research papers and resources
```

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies listed in `environment.yml`

### Installation

Clone this repository:
```bash
git clone https://github.com/[username]/RegimeAlpha.git
cd RegimeAlpha
```

Create the Conda environment:
```bash
conda env create -f environment.yml
conda activate regime-alpha
```

### Running the Project

Data collection:
```bash
python RegimeAlpha/code/main.py --mode data_collection
```

Train regime detection model:
```bash
python RegimeAlpha/code/main.py --mode train_regime
```

Train price prediction model:
```bash
python RegimeAlpha/code/main.py --mode train_predictor
```

Run backtesting:
```bash
python RegimeAlpha/code/main.py --mode backtest
```

Generate visualizations:
```bash
python RegimeAlpha/code/main.py --mode visualize
```

## Project Timeline

- Week 1: Setup and Data Pipeline
- Week 2: Modeling and Regime Detection
- Week 3: Trading Engine and Backtesting
- Week 4: Testing, Visualization, and Paper Finalization

## Course Requirements

This project is being completed as the final project for CPE 593: Applied Data Structures and Algorithms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
