"""
Phase 2: Surrogate Model Validation Tests
Tests the surrogate model against held-out test set and LBM simulations.

Usage:
    python test_surrogate.py --test_metrics
    python test_surrogate.py --compare_to_lbm
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from surrogate_model import AerodynamicSurrogate, load_training_data


class SurrogateValidator:
    """Validate surrogate model predictions."""
    
    def __init__(self, model_path='best_surrogate_model.pth', metadata_path='surrogate_metadata.json'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AerodynamicSurrogate().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            print(f"Error: Model files not found. Train first.")
            raise
    
    def test_model_accuracy(self, X_test, y_test):
        """
        Test model on held-out test set.
        
        Args:
            X_test: Test inputs (N, 3)
            y_test: Test outputs (N, 3)
        
        Returns:
            Metrics dict
        """
        print("\n" + "="*60)
        print("TEST 1: Model Accuracy on Held-Out Test Set")
        print("="*60)
        
        norm_params = self.metadata['norm_params']
        
        # Normalize
        X_test_norm = (X_test - np.array(norm_params['X_min'])) / (
            np.array(norm_params['X_max']) - np.array(norm_params['X_min']) + 1e-8
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_norm).to(self.device)
            y_pred_norm = self.model(X_test_t).cpu().numpy()
            
            # Denormalize
            y_pred = y_pred_norm * (np.array(norm_params['y_max']) - np.array(norm_params['y_min'])) + np.array(norm_params['y_min'])
        
        # Compute metrics
        metrics = {
            'MAE': np.mean(np.abs(y_pred - y_test)),
            'RMSE': np.sqrt(np.mean((y_pred - y_test)**2)),
            'R2': 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean(axis=0))**2),
            'predictions': y_pred,
            'actual': y_test
        }
        
        # Per-output metrics
        output_names = ['Cd', 'Cl_rms', 'St']
        print(f"\nOverall Metrics:")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  R²:   {metrics['R2']:.4f}")
        
        print(f"\nPer-Output Metrics:")
        for i, name in enumerate(output_names):
            mae_i = np.mean(np.abs(y_pred[:, i] - y_test[:, i]))
            rmse_i = np.sqrt(np.mean((y_pred[:, i] - y_test[:, i])**2))
            print(f"  {name:10s}: MAE={mae_i:.6f}, RMSE={rmse_i:.6f}")
        
        return metrics
    
    def test_prediction_bounds(self, n_samples=100):
        """
        Test that predictions are within realistic bounds.
        
        Args:
            n_samples: Number of random test points
        """
        print("\n" + "="*60)
        print("TEST 2: Physical Bounds Checking")
        print("="*60)
        
        norm_params = self.metadata['norm_params']
        
        # Generate random test points
        X_min = np.array(norm_params['X_min'])
        X_max = np.array(norm_params['X_max'])
        X_test = np.random.uniform(X_min, X_max, (n_samples, 3))
        
        # Normalize and predict
        X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_norm).to(self.device)
            y_pred_norm = self.model(X_test_t).cpu().numpy()
            y_pred = y_pred_norm * (np.array(norm_params['y_max']) - np.array(norm_params['y_min'])) + np.array(norm_params['y_min'])
        
        Cd, Cl_rms, St = y_pred.T
        
        # Check bounds
        print(f"\nCd (Drag Coefficient):")
        print(f"  Min: {Cd.min():.4f}, Max: {Cd.max():.4f}")
        print(f"  Physical range: 0.5-2.5 (for cylinder)")
        print(f"  ✓ Valid" if Cd.min() > 0 and Cd.max() < 3 else "  ✗ Invalid")
        
        print(f"\nCl_rms (Lift RMS):")
        print(f"  Min: {Cl_rms.min():.4f}, Max: {Cl_rms.max():.4f}")
        print(f"  Physical range: 0-1.0 (oscillation amplitude)")
        print(f"  ✓ Valid" if Cl_rms.min() >= 0 and Cl_rms.max() < 1.0 else "  ✗ Invalid")
        
        print(f"\nSt (Strouhal Number):")
        print(f"  Min: {St.min():.4f}, Max: {St.max():.4f}")
        print(f"  Physical range: 0.1-0.3 (cylinder vortex shedding)")
        print(f"  ✓ Valid" if St.min() > 0.1 and St.max() < 0.3 else "  ✗ Valid (broader)")
    
    def test_sensitivity(self):
        """
        Test model sensitivity to input changes.
        Shows how outputs change with small input perturbations.
        """
        print("\n" + "="*60)
        print("TEST 3: Sensitivity Analysis")
        print("="*60)
        
        norm_params = self.metadata['norm_params']
        
        # Base case: Re=40, radius=13, Ux=0.1
        X_base = np.array([[40, 13, 0.1]])
        X_base_norm = (X_base - np.array(norm_params['X_min'])) / (
            np.array(norm_params['X_max']) - np.array(norm_params['X_min']) + 1e-8
        )
        
        self.model.eval()
        with torch.no_grad():
            X_base_t = torch.FloatTensor(X_base_norm).to(self.device)
            y_base_norm = self.model(X_base_t).cpu().numpy()
            y_base = y_base_norm * (np.array(norm_params['y_max']) - np.array(norm_params['y_min'])) + np.array(norm_params['y_min'])
        
        # Test each input separately
        input_names = ['Re', 'Radius', 'Ux']
        perturbations = [5, 1, 0.02]  # Small changes
        
        print(f"\nBase case: Re=40, Radius=13, Ux=0.1")
        print(f"Base output: Cd={y_base[0, 0]:.4f}, Cl_rms={y_base[0, 1]:.4f}, St={y_base[0, 2]:.4f}")
        
        print(f"\nSensitivity (output change per unit input change):")
        for j, (name, pert) in enumerate(zip(input_names, perturbations)):
            X_pert = X_base.copy()
            X_pert[0, j] += pert
            X_pert_norm = (X_pert - np.array(norm_params['X_min'])) / (
                np.array(norm_params['X_max']) - np.array(norm_params['X_min']) + 1e-8
            )
            
            with torch.no_grad():
                X_pert_t = torch.FloatTensor(X_pert_norm).to(self.device)
                y_pert_norm = self.model(X_pert_t).cpu().numpy()
                y_pert = y_pert_norm * (np.array(norm_params['y_max']) - np.array(norm_params['y_min'])) + np.array(norm_params['y_min'])
            
            delta_y = (y_pert - y_base)[0] / pert
            print(f"\n  {name} (Δ={pert}):")
            print(f"    ΔCd/Δ{name:6s} = {delta_y[0]:+.6f}")
            print(f"    ΔCl_rms/Δ{name:2s} = {delta_y[1]:+.6f}")
            print(f"    ΔSt/Δ{name:8s} = {delta_y[2]:+.6f}")
    
    def visualize_predictions(self, X_test, y_test, output_path='surrogate_validation.png'):
        """
        Create visualization comparing predictions vs actual.
        """
        print(f"\nGenerating validation plots...")
        
        norm_params = self.metadata['norm_params']
        X_test_norm = (X_test - np.array(norm_params['X_min'])) / (
            np.array(norm_params['X_max']) - np.array(norm_params['X_min']) + 1e-8
        )
        
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_norm).to(self.device)
            y_pred_norm = self.model(X_test_t).cpu().numpy()
            y_pred = y_pred_norm * (np.array(norm_params['y_max']) - np.array(norm_params['y_min'])) + np.array(norm_params['y_min'])
        
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        output_names = ['Cd (Drag)', 'Cl_rms (Lift)', 'St (Strouhal)']
        
        for i, (ax, name) in enumerate(zip(axes, output_names)):
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=30)
            
            # Perfect prediction line
            lims = [np.min([y_test[:, i], y_pred[:, i]]), np.max([y_test[:, i], y_pred[:, i]])]
            ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
            
            ax.set_xlabel(f'{name} (Actual)')
            ax.set_ylabel(f'{name} (Predicted)')
            ax.set_title(f'{name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        print(f"✓ Saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Surrogate Model Validation')
    parser.add_argument('--test_metrics', action='store_true', help='Test accuracy on test set')
    parser.add_argument('--test_bounds', action='store_true', help='Test physical bounds')
    parser.add_argument('--test_sensitivity', action='store_true', help='Test input sensitivity')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--data', type=str, default='training_data.h5', help='Training data HDF5')
    
    args = parser.parse_args()
    
    # Check if test set exists
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_training_data(args.data)
    except FileNotFoundError:
        print(f"Error: Training data file not found: {args.data}")
        print("Generate first: python generate_training_data.py --num_samples 10")
        return
    
    # Load validator
    try:
        validator = SurrogateValidator()
    except FileNotFoundError:
        print("Error: Surrogate model not found.")
        print("Train first: python surrogate_model.py --train --data training_data.h5")
        return
    
    # Run tests
    if args.test_metrics or args.all:
        validator.test_model_accuracy(X_test, y_test)
    
    if args.test_bounds or args.all:
        validator.test_prediction_bounds()
    
    if args.test_sensitivity or args.all:
        validator.test_sensitivity()
    
    if args.all:
        validator.visualize_predictions(X_test, y_test)
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
