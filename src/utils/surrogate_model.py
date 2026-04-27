"""
Phase 2: Surrogate Model - Neural Network for Aerodynamic Prediction
Trains on LBM simulation data to predict Cd, Cl, St instantly.

Usage:
    # Train model
    python surrogate_model.py --train --data training_data.h5
    
    # Load and predict
    python surrogate_model.py --predict --re 40 --radius 13 --ux 0.1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import argparse
from pathlib import Path
import json
from datetime import datetime


class AerodynamicSurrogate(nn.Module):
    """
    Neural Network Surrogate for LBM Aerodynamic Metrics.
    
    Input: [Re, cylinder_radius, Ux] (3 features)
    Output: [Cd, Cl_rms, St] (3 targets)
    
    Architecture:
    Input (3) → Dense(64) → ReLU → Dropout(0.2) → 
               Dense(128) → ReLU → Dropout(0.2) →
               Dense(64) → ReLU → Dense(3)
    """
    
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 64], output_dim=3, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation - regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def predict(self, inputs):
        """
        Predict aerodynamic metrics.
        
        Args:
            inputs: [Re, radius, Ux] or batch of these
        
        Returns:
            [Cd, Cl_rms, St] predictions
        """
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, (list, np.ndarray)):
                if len(inputs) == 3 and not isinstance(inputs[0], (list, np.ndarray)):
                    # Single prediction
                    inputs = torch.FloatTensor([inputs])
                else:
                    inputs = torch.FloatTensor(inputs)
            
            outputs = self.forward(inputs)
            return outputs.numpy()


class SurrogateTrainer:
    """Train and evaluate surrogate models."""
    
    def __init__(self, model, learning_rate=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def normalize_data(self, X_train, X_val, y_train, y_val):
        """
        Normalize inputs and outputs to [0, 1] range.
        
        Returns:
            Normalized data and normalization parameters
        """
        # Input normalization
        X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
        X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
        X_val_norm = (X_val - X_min) / (X_max - X_min + 1e-8)
        
        # Output normalization
        y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)
        y_train_norm = (y_train - y_min) / (y_max - y_min + 1e-8)
        y_val_norm = (y_val - y_min) / (y_max - y_min + 1e-8)
        
        # Save for later denormalization
        self.norm_params = {
            'X_min': X_min, 'X_max': X_max,
            'y_min': y_min, 'y_max': y_max
        }
        
        return X_train_norm, X_val_norm, y_train_norm, y_val_norm
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=8):
        """
        Train the surrogate model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Training epochs
            batch_size: Batch size
        """
        print("\nNormalizing data...")
        X_train_norm, X_val_norm, y_train_norm, y_val_norm = self.normalize_data(
            X_train, X_val, y_train, y_val
        )
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_data = (torch.FloatTensor(X_val_norm), torch.FloatTensor(y_val_norm))
        
        print(f"Training for {epochs} epochs, batch size {batch_size}...")
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_t, y_val_t = val_data
                X_val_t, y_val_t = X_val_t.to(self.device), y_val_t.to(self.device)
                y_val_pred = self.model(X_val_t)
                val_loss = self.criterion(y_val_pred, y_val_t).item()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_surrogate_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_surrogate_model.pth'))
        print("✓ Training complete. Best model loaded.")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set.
        
        Returns:
            Metrics dict with MAE, RMSE, R²
        """
        self.model.eval()
        with torch.no_grad():
            X_test_norm = (X_test - self.norm_params['X_min']) / (
                self.norm_params['X_max'] - self.norm_params['X_min'] + 1e-8
            )
            X_test_t = torch.FloatTensor(X_test_norm).to(self.device)
            y_pred_norm = self.model(X_test_t).cpu().numpy()
            
            # Denormalize
            y_pred = y_pred_norm * (self.norm_params['y_max'] - self.norm_params['y_min']) + self.norm_params['y_min']
        
        # Metrics
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        
        # R² score
        ss_res = np.sum((y_test - y_pred)**2)
        ss_tot = np.sum((y_test - y_test.mean(axis=0))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(np.mean(r2)),
            'predictions': y_pred,
            'actual': y_test
        }
        
        return metrics
    
    def predict(self, X):
        """Predict on new data (returns denormalized values)."""
        self.model.eval()
        with torch.no_grad():
            X_norm = (X - self.norm_params['X_min']) / (
                self.norm_params['X_max'] - self.norm_params['X_min'] + 1e-8
            )
            X_t = torch.FloatTensor(X_norm).to(self.device)
            y_pred_norm = self.model(X_t).cpu().numpy()
            
            # Denormalize
            y_pred = y_pred_norm * (self.norm_params['y_max'] - self.norm_params['y_min']) + self.norm_params['y_min']
        
        return y_pred


def load_training_data(h5_path, test_split=0.2):
    """Load and split training data."""
    with h5py.File(h5_path, 'r') as f:
        X = np.column_stack([f['Re'][:], f['radius'][:], f['Ux'][:]])
        y = np.column_stack([f['Cd'][:], f['Cl_rms'][:], f['St'][:]])
    
    # Remove NaN rows
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X, y = X[valid_idx], y[valid_idx]
    
    # Split
    n_test = int(len(X) * test_split)
    indices = np.random.permutation(len(X))
    
    X_train, y_train = X[indices[:-n_test]], y[indices[:-n_test]]
    X_test, y_test = X[indices[-n_test:]], y[indices[-n_test:]]
    
    # Further split training into train/val
    n_val = int(len(X_train) * 0.2)
    val_indices = np.random.choice(len(X_train), n_val, replace=False)
    train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)
    
    X_val, y_val = X_train[val_indices], y_train[val_indices]
    X_train, y_train = X_train[train_indices], y_train[train_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Surrogate Model Training')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--data', type=str, default='training_data.h5', help='Training data HDF5')
    parser.add_argument('--model', type=str, default='surrogate_model.pth', help='Model save path')
    parser.add_argument('--re', type=float, help='Reynolds number for prediction')
    parser.add_argument('--radius', type=float, help='Cylinder radius for prediction')
    parser.add_argument('--ux', type=float, help='Inlet velocity for prediction')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.train:
        print("\n" + "="*60)
        print("Phase 2: Training Surrogate Model")
        print("="*60)
        
        # Load data
        print(f"\nLoading data from {args.data}...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_training_data(args.data)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
        
        # Create and train model
        model = AerodynamicSurrogate(input_dim=3, output_dim=3)
        trainer = SurrogateTrainer(model, device=device)
        
        trainer.train(X_train, y_train, X_val, y_val, epochs=args.epochs)
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = trainer.evaluate(X_test, y_test)
        print(f"Test MAE: {metrics['MAE']:.6f}")
        print(f"Test RMSE: {metrics['RMSE']:.6f}")
        print(f"Test R²: {metrics['R2']:.4f}")
        
        # Save
        torch.save(model.state_dict(), args.model)
        with open('surrogate_metadata.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_path': args.model,
                'data_path': args.data,
                'test_metrics': metrics,
                'norm_params': {k: v.tolist() for k, v in trainer.norm_params.items()}
            }, f, indent=2)
        
        print(f"\n✓ Model saved to {args.model}")
        print("="*60)
    
    elif args.predict:
        if not args.re or not args.radius or not args.ux:
            print("Error: --predict requires --re, --radius, --ux")
            return
        
        print(f"\nLoading model from {args.model}...")
        model = AerodynamicSurrogate()
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        # Load normalization params
        with open('surrogate_metadata.json', 'r') as f:
            metadata = json.load(f)
            norm_params = metadata['norm_params']
        
        # Make prediction
        X_input = np.array([[args.re, args.radius, args.ux]])
        X_norm = (X_input - np.array(norm_params['X_min'])) / (
            np.array(norm_params['X_max']) - np.array(norm_params['X_min']) + 1e-8
        )
        
        model.eval()
        with torch.no_grad():
            y_pred_norm = model(torch.FloatTensor(X_norm)).numpy()
            y_pred = y_pred_norm * (np.array(norm_params['y_max']) - np.array(norm_params['y_min'])) + np.array(norm_params['y_min'])
        
        Cd, Cl_rms, St = y_pred[0]
        
        print(f"\n" + "="*60)
        print("Surrogate Model Prediction")
        print("="*60)
        print(f"Input: Re={args.re}, radius={args.radius}, Ux={args.ux}")
        print(f"Output:")
        print(f"  Cd       = {Cd:.4f}")
        print(f"  Cl (RMS) = {Cl_rms:.4f}")
        print(f"  St       = {St:.4f}")
        print("="*60)
    
    else:
        print("Use --train or --predict")


if __name__ == '__main__':
    main()
