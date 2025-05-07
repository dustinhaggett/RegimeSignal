# RegimeAlpha

Final project for CPE 593: Applied Data Structures and Algorithms

**Team Members:**
- Dustin Haggett
- Lambert Kongnyuy
- Mateusz Marciniak

## Project Overview

RegimeAlpha is a sophisticated crypto trading system that:
1. Identifies different market regimes (trending, ranging, volatile)
2. Uses ML models to predict price movements
3. Adapts trading strategies based on the detected regime
4. Optimizes for high-leverage (30x-50x) trading on BTC/USDT

## Directory Structure

- `code/`: Implementation of all system components
  - `data_pipeline/`: Data collection and preprocessing
  - `regime_detection/`: Market regime identification
  - `modeling/`: ML models for price prediction
  - `trading_logic/`: Trading strategy implementation
  - `backtesting/`: Backtesting engine
  - `visualization/`: Results visualization
  - `main.py`: Main execution script
- `documents/`: Paper and presentation materials
- `references/`: Related research papers and resources

## Data Structures & Algorithms Application

RegimeAlpha demonstrates several key DSA concepts:
- Sliding window algorithms for feature engineering
- Clustering algorithms for regime classification
- Time-series data processing with efficient data structures
- Greedy algorithm approach for trading decisions
- Graph-based state transitions for regime modeling

## Getting Started

1. **Setup Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate regime-alpha
   ```

2. **Configure the Project:**
   - Edit `config.json` to set your data parameters, model configurations, and trading settings

3. **Run the Pipeline:**
   ```bash
   # Collect data
   python code/main.py --mode data_collection
   
   # Train regime detection model
   python code/main.py --mode train_regime
   
   # Train price prediction model
   python code/main.py --mode train_predictor
   
   # Run backtesting
   python code/main.py --mode backtest
   
   # Generate visualizations
   python code/main.py --mode visualize
   
   # Or run the full pipeline
   python code/main.py --mode full_pipeline
   ```

## Project Milestones

- **Milestone 1 (Due: April 8th):**
  - [x] Form team
  - [x] Select project topic
  - [x] Setup GitHub repository
  - [x] Create project plan

- **Milestone 2 (Due: April 29th):**
  - [ ] Complete research on regime detection methods
  - [ ] Implement data pipeline
  - [ ] Implement regime detection
  - [ ] Start ML model implementation

- **Final Submission:**
  - [ ] Complete implementation of all components
  - [ ] Run comprehensive backtests
  - [ ] Finalize IEEE-format paper
  - [ ] Prepare presentation

## Research Focus

Our key research questions include:
1. How different market regimes affect trading performance
2. Which ML models perform best for each regime
3. How to optimize trading parameters for high-leverage scenarios
4. Quantitative comparison of regime-aware vs. regime-agnostic strategies

## Paper Structure

Our final paper will follow the IEEE conference format and include:
- Abstract
- Introduction
- Background and related work
- System design
- Implementation details
- Experimental results
- Analysis and discussion
- Conclusion
- References
