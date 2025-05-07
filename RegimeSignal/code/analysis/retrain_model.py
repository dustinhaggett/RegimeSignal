#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model retraining script with feature stability analysis and improved monitoring.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import sys
from tqdm import tqdm

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from modeling.feature_engineering import FeatureEngineer
from modeling.lstm_trainer import RegimeAwareLSTM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, start_date: str = '2023-01-01'):
        self.start_date = pd.to_datetime(start_date)
        self.sequence_length = 60
        self.batch_size = 128
        self.learning_rate = 0.001
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2  # Added dropout for better generalization
        
    def analyze_feature_stability(self, data: pd.DataFrame, features: list) -> dict:
        """Analyze feature stability and importance."""
        logger.info("\nAnalyzing Feature Stability:")
        
        stability_scores = {}
        for feature in features:
            # Calculate rolling statistics
            rolling_mean = data[feature].rolling(window=24*4*7).mean()  # 7-day window
            rolling_std = data[feature].rolling(window=24*4*7).std()
            
            # Calculate stability metrics
            cv = rolling_std / rolling_mean.abs()  # Coefficient of variation
            stability_score = 1 - cv.mean()
            
            stability_scores[feature] = {
                'stability_score': float(stability_score),
                'mean_cv': float(cv.mean()),
                'cv_std': float(cv.std())
            }
            
            # Plot stability over time
            plt.figure(figsize=(12, 6))
            cv.plot()
            plt.title(f'{feature} Stability Over Time')
            plt.xlabel('Date')
            plt.ylabel('Coefficient of Variation')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = Path(__file__).parent.parent.parent / 'results' / 'retraining' / 'stability' / f'{feature}_stability.png'
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
        
        # Save stability scores
        scores_path = Path(__file__).parent.parent.parent / 'results' / 'retraining' / 'feature_stability.json'
        with open(scores_path, 'w') as f:
            json.dump(stability_scores, f, indent=4)
        
        return stability_scores
    
    def prepare_data(self, data: pd.DataFrame, features: list) -> tuple:
        """Prepare data for training with feature stability weighting."""
        logger.info("\nPreparing Training Data:")
        
        # Filter data by date
        data = data[data.index >= self.start_date].copy()
        logger.info(f"Using data from {data.index.min()} to {data.index.max()}")
        logger.info(f"Total periods: {len(data)}")
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[features].values[i:i + self.sequence_length])
            y.append(data['market_regime'].values[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/val sets
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features by regime
        scalers = {}
        X_train_scaled = np.zeros_like(X_train)
        X_val_scaled = np.zeros_like(X_val)
        
        for regime in [-1, 0, 1]:
            mask = y_train == regime
            if mask.any():
                scaler = StandardScaler()
                # Reshape to 2D for scaling
                X_regime = X_train[mask].reshape(-1, X_train.shape[-1])
                scaler.fit(X_regime)
                scalers[regime] = scaler
                
                # Scale training data
                X_train_scaled[mask] = scaler.transform(X_train[mask].reshape(-1, X_train.shape[-1])).reshape(
                    -1, self.sequence_length, X_train.shape[-1])
                
                # Scale validation data for this regime
                val_mask = y_val == regime
                if val_mask.any():
                    X_val_scaled[val_mask] = scaler.transform(X_val[val_mask].reshape(-1, X_val.shape[-1])).reshape(
                        -1, self.sequence_length, X_val.shape[-1])
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), scalers
    
    def train_model(self, train_data: tuple, val_data: tuple, features: list) -> tuple:
        """Train model with improved monitoring and early stopping."""
        logger.info("\nTraining Model:")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train + 1)  # Convert -1,0,1 to 0,1,2
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val + 1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        model = RegimeAwareLSTM(
            input_size=len(features),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        epochs = 100
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                features, regimes = batch
                features, regimes = features.to(device), regimes.to(device)
                
                optimizer.zero_grad()
                outputs = model(features, regimes)
                loss = criterion(outputs, regimes)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += regimes.size(0)
                train_correct += (predicted == regimes).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, regimes in val_loader:
                    features, regimes = features.to(device), regimes.to(device)
                    outputs = model(features, regimes)
                    loss = criterion(outputs, regimes)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += regimes.size(0)
                    val_correct += (predicted == regimes).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Log metrics
            logger.info(f'Epoch {epoch + 1}/{epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            # Save metrics for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                         Path(__file__).parent.parent.parent / 'models' / 'lstm' / 'regime_lstm.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return model, (train_losses, val_losses, train_accuracies, val_accuracies)
    
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(train_accuracies, label='Train Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(__file__).parent.parent.parent / 'results' / 'retraining' / 'training_curves.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training curves saved to: {plot_path}")

def main():
    """Main function to retrain model."""
    try:
        # Initialize retrainer
        retrainer = ModelRetrainer(start_date='2023-01-01')
        
        # Load feature engineering
        feature_engineer = FeatureEngineer()
        data = feature_engineer.load_data()
        
        # Load feature names
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        feature_names_path = project_root / 'models' / 'lstm' / 'feature_names.txt'
        with open(feature_names_path, 'r') as f:
            features = f.read().splitlines()
        
        # Analyze feature stability
        stability_scores = retrainer.analyze_feature_stability(data, features)
        
        # Prepare data
        train_data, val_data, scalers = retrainer.prepare_data(data, features)
        
        # Train model
        model, metrics = retrainer.train_model(train_data, val_data, features)
        
        # Save scalers
        scaler_path = project_root / 'models' / 'lstm' / 'scaler.pkl'
        import joblib
        joblib.dump(scalers, scaler_path)
        
        # Generate and save training statistics
        stats_out = {}
        for feature in features:
            stats_out[feature] = {
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std()),
                'skew': float(stats.skew(data[feature].dropna())),
                'kurtosis': float(stats.kurtosis(data[feature].dropna())),
                'q1': float(data[feature].quantile(0.25)),
                'q3': float(data[feature].quantile(0.75)),
                'stability_score': stability_scores[feature]['stability_score']
            }
        
        stats_path = project_root / 'models' / 'lstm' / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_out, f, indent=4)
        
        logger.info("\nRetraining completed successfully!")
        logger.info(f"Model saved to: {project_root / 'models' / 'lstm' / 'regime_lstm.pth'}")
        logger.info(f"Scalers saved to: {scaler_path}")
        logger.info(f"Training statistics saved to: {stats_path}")
        
    except Exception as e:
        logger.error(f"Error in retraining: {str(e)}")
        raise

if __name__ == '__main__':
    main() 