#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM model training with regime-aware architecture.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import joblib  # Added for saving scalers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """Custom dataset for cryptocurrency data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature array of shape (n_samples, sequence_length, n_features)
            targets: Target array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]

class RegimeAwareLSTM(nn.Module):
    """Regime-aware LSTM model with attention."""
    
    def __init__(self, input_size: int = 11, hidden_size: int = 64, 
                 num_layers: int = 2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature encoder with dropout
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)  # Increased dropout
        )
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3  # Increased dropout
        )
        
        # Attention mechanism with dropout
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.3  # Increased dropout
        )
        
        # Output layers with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size = x.size(0)
        
        # Feature encoding
        x = self.feature_encoder(x)  # (batch_size, sequence_length, hidden_size)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)
        
        # Apply attention
        # Reshape for attention: (seq_len, batch_size, hidden_size)
        lstm_out_permuted = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(
            lstm_out_permuted,
            lstm_out_permuted,
            lstm_out_permuted
        )
        
        # Reshape back: (batch_size, seq_len, hidden_size)
        attn_out = attn_out.permute(1, 0, 2)
        
        # Use last hidden state for classification
        last_hidden = attn_out[:, -1, :]
        
        # Classification with temperature scaling
        logits = self.classifier(last_hidden)
        scaled_logits = logits / self.temperature
        
        return scaled_logits

class LSTMTrainer:
    """Trainer for the RegimeAwareLSTM model."""
    
    def __init__(self, data_path: str = None, sequence_length: int = 60):
        """
        Initialize trainer.
        
        Args:
            data_path: Path to processed data
            sequence_length: Length of input sequences
        """
        # Set default data path if not provided
        if data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            data_path = os.path.join(project_root, 'RegimeSignal', 'data', 'processed', 'processed_data.csv')
            
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Feature sets with importance weights
        self.feature_sets = {
            'primary': [
                'volume', 'bb_width', 'rocr_48', 'ulcer_index',
                'awesome_oscillator', 'volume_normalized'
            ],
            'secondary': [
                'ema_50', 'adx', 'trend_strength', 'cmf', 'mfi'
            ]
        }
        
        # Regime-specific feature weights
        self.feature_weights = {
            -1: torch.tensor([1.5, 1.3, 1.2, 0.8, 1.3, 0.7]),  # Bearish
            1: torch.tensor([1.0, 0.9, 0.8, 1.5, 1.3, 1.4])    # Bullish
        }
        
        # Initialize data
        self.load_data()
        
        # Initialize model
        self.model = RegimeAwareLSTM(
            input_size=len(self.feature_names),
            hidden_size=64,
            num_layers=2
        ).to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Early stopping parameters
        self.patience = 5
        self.best_val_loss = float('inf')
        self.counter = 0
        
        # Initialize class weights
        self.class_weights = None
        
    def load_data(self) -> None:
        """Load and prepare data with regime-aware preprocessing."""
        try:
            logger.info("Loading data...")
            self.data = pd.read_csv(self.data_path)
            
            # Get feature names (excluding target and timestamp columns)
            self.feature_names = [col for col in self.data.columns if col not in ['regime', 'timestamp']]
            
            # Split data by regime
            bearish_data = self.data[self.data['regime'] == -1]
            bullish_data = self.data[self.data['regime'] == 1]
            
            # Create regime-specific scalers
            self.scalers = {}
            for regime, data in [(-1, bearish_data), (1, bullish_data)]:
                scaler = StandardScaler()
                self.scalers[regime] = scaler.fit(data[self.feature_names])
            
            # Scale features by regime
            X = np.zeros_like(self.data[self.feature_names].values)
            for regime, data in [(-1, bearish_data), (1, bullish_data)]:
                mask = self.data['regime'] == regime
                X[mask] = self.scalers[regime].transform(data[self.feature_names])
            
            # Get target and adjust labels
            y = self.data['regime'].values
            y = y + 1  # Convert -1, 0, 1 to 0, 1, 2
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            for i in range(len(X) - self.sequence_length):
                X_sequences.append(X[i:i + self.sequence_length])
                y_sequences.append(y[i + self.sequence_length])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train = X_sequences[:split_idx]
            X_val = X_sequences[split_idx:]
            y_train = y_sequences[:split_idx]
            y_val = y_sequences[split_idx:]
            
            # Create datasets
            self.train_dataset = CryptoDataset(X_train, y_train)
            self.val_dataset = CryptoDataset(X_val, y_val)
            
            # Create dataloaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=4
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4
            )
            
            # Log data shapes and distributions
            logger.info(f"Training set shape: {X_train.shape}")
            logger.info(f"Validation set shape: {X_val.shape}")
            
            # Log class distribution
            for split_name, y_split in [('Train', y_train), ('Val', y_val)]:
                unique, counts = np.unique(y_split, return_counts=True)
                logger.info(f"\n{split_name} set class distribution:")
                for cls, count in zip(unique, counts):
                    regime = "Bearish" if cls == 0 else "Neutral" if cls == 1 else "Bullish"
                    logger.info(f"Class {regime} ({cls}): {count} samples ({count/len(y_split)*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _calculate_class_weights(self, train_loader: DataLoader) -> None:
        """Calculate class weights from training data."""
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        # Calculate class frequencies
        class_counts = np.bincount(all_labels, minlength=3)
        total_samples = len(all_labels)
        
        # Calculate weights (inverse of class frequency)
        weights = total_samples / (3 * class_counts)
        weights = torch.FloatTensor(weights).to(self.device)
        
        self.class_weights = weights
        logger.info(f"Class weights: {weights}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculate loss with class weights
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                acc = 100 * correct / total
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
        
        epoch_acc = 100 * correct / total
        epoch_loss = total_loss / len(train_loader)
        logger.info(f"Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        return epoch_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
        
        val_acc = 100 * correct / total
        val_loss = total_loss / len(val_loader)
        logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        return val_loss, val_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100) -> None:
        """Train the model."""
        self.model.train()
        self.best_val_loss = float('inf')
        self.counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
                
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info("Early stopping triggered")
                    break
            
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Calibrate temperature on validation set
        self._calibrate_temperature(val_loader)
    
    def _calibrate_temperature(self, val_loader: DataLoader) -> None:
        """Calibrate temperature parameter on validation set."""
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Get raw logits (before temperature scaling)
                logits = self.model(features)  # This already includes the classifier
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Optimize temperature using NLL loss
        temperature = nn.Parameter(torch.ones(1).to(self.device))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(all_logits / temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        # Update model temperature
        self.model.temperature.data = temperature.data
        
        logger.info(f"Calibrated temperature: {temperature.item():.4f}")
    
    def save_model(self, history: Dict) -> None:
        """
        Save model and training history.
        
        Args:
            history: Training history
        """
        try:
            # Create output directory using relative path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            output_dir = os.path.join(project_root, 'RegimeSignal', 'models', 'lstm')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(output_dir, 'regime_lstm.pth')
            torch.save(self.model.state_dict(), model_path)
            
            # Save scaler
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scalers, scaler_path)
            
            # Save feature names
            features_path = os.path.join(output_dir, 'feature_names.txt')
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.feature_names))
            
            # Plot training history
            self.plot_history(history, output_dir)
            
            logger.info(f"Saved model and artifacts to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def plot_history(self, history: Dict, output_dir: str) -> None:
        """
        Plot training history.
        
        Args:
            history: Training history
            output_dir: Output directory
        """
        try:
            # Plot loss
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'loss.png'))
            plt.close()
            
            # Plot accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'accuracy.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting history: {str(e)}")
            raise

def main():
    """Main function to train the model."""
    try:
        # Initialize trainer
        trainer = LSTMTrainer()
        
        # Train model
        trainer.train(trainer.train_loader, trainer.val_loader)
        
        logger.info("Completed model training")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 