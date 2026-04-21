"""
Phase 2: Training Data Generation Pipeline
Systematically generates LBM simulation data for ML surrogate model training.

Usage:
    python generate_training_data.py --num_samples 50 --output training_data.h5
"""

import numpy as np
import h5py
import json
import argparse
from pathlib import Path
from datetime import datetime
from main import run_simulation, Phase1Metrics
import matplotlib
matplotlib.use('Agg')  # Headless mode - no GUI visualization


class TrainingDataGenerator:
    """Generate and store training data for surrogate models."""
    
    def __init__(self, output_path='training_data.h5', seed=42):
        """
        Initialize generator.
        
        Args:
            output_path: HDF5 file to store training data
            seed: Random seed for reproducibility
        """
        self.output_path = Path(output_path)
        self.seed = seed
        np.random.seed(seed)
        self.metrics_collector = Phase1Metrics()
        self.data = []
        
    def generate_parameter_sweep(self, num_samples=50, strategy='latin_hypercube'):
        """
        Generate parameter combinations for training.
        
        Args:
            num_samples: Number of parameter sets to generate
            strategy: 'latin_hypercube' (uniform coverage) or 'random'
        
        Returns:
            List of parameter dicts: {Re, cylinder_radius, tau, Ux, Ny}
        """
        # Parameter ranges (based on physical validity)
        param_ranges = {
            'Re': (20, 100),              # Reynolds number
            'cylinder_radius': (8, 18),   # Lattice units
            'Ux': (0.05, 0.15),           # Inlet velocity
        }
        
        if strategy == 'latin_hypercube':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=3, seed=self.seed)
            samples = sampler.random(n=num_samples)
            
            # Scale to parameter ranges
            params = []
            for sample in samples:
                Re = param_ranges['Re'][0] + sample[0] * (param_ranges['Re'][1] - param_ranges['Re'][0])
                radius = param_ranges['cylinder_radius'][0] + sample[1] * (param_ranges['cylinder_radius'][1] - param_ranges['cylinder_radius'][0])
                Ux = param_ranges['Ux'][0] + sample[2] * (param_ranges['Ux'][1] - param_ranges['Ux'][0])
                
                params.append({
                    'Re': float(Re),
                    'cylinder_radius': float(radius),
                    'Ux': float(Ux)
                })
        else:  # random
            params = []
            for _ in range(num_samples):
                params.append({
                    'Re': np.random.uniform(*param_ranges['Re']),
                    'cylinder_radius': np.random.uniform(*param_ranges['cylinder_radius']),
                    'Ux': np.random.uniform(*param_ranges['Ux']),
                })
        
        return params
    
    def compute_tau_from_re(self, Re, Ux, cylinder_radius, cs=1/np.sqrt(3)):
        """
        Compute relaxation time from Reynolds number.
        
        Re = rho * U * D / mu = U * D / (tau - 0.5)
        tau = 0.5 + D / (Re * cs²)
        
        Args:
            Re: Reynolds number
            Ux: Inlet velocity
            cylinder_radius: Cylinder diameter (2 * radius)
            cs: Speed of sound (lattice units)
        
        Returns:
            tau: Relaxation time parameter
        """
        D = 2 * cylinder_radius  # Diameter
        nu = Ux * D / Re  # Kinematic viscosity
        tau = 0.5 + nu / (cs**2)
        
        # Clamp to physical range
        tau = np.clip(tau, 0.52, 1.99)
        return float(tau)
    
    def run_single_simulation(self, params, iterations=50000, verbose=False):
        """
        Run a single LBM simulation with given parameters.
        
        Args:
            params: Dict with Re, cylinder_radius, Ux
            iterations: Number of timesteps
            verbose: Print progress
        
        Returns:
            Dict with inputs and outputs {Re, radius, Cd, Cl, St, quality, ...}
        """
        Re = params['Re']
        radius = params['cylinder_radius']
        Ux = params['Ux']
        
        # Compute tau from Re
        tau = self.compute_tau_from_re(Re, Ux, radius)
        
        if verbose:
            print(f"Running: Re={Re:.1f}, radius={radius:.1f}, Ux={Ux:.3f}, tau={tau:.3f}")
        
        # Run simulation
        try:
            metrics, final_fields = self._run_lbm_simulation(
                Ux=Ux,
                tau=tau,
                cylinder_radius=int(radius),
                iterations=iterations,
                verbose=verbose
            )
            
            # Package results
            result = {
                'Re': float(Re),
                'cylinder_radius': float(radius),
                'Ux': float(Ux),
                'tau': float(tau),
                'iterations': int(iterations),
                'Cd': float(metrics.get('Cd', np.nan)),
                'Cd_std': float(metrics.get('Cd_std', np.nan)),
                'Cl_rms': float(metrics.get('Cl_rms', np.nan)),
                'Cl_mean': float(metrics.get('Cl_mean', np.nan)),
                'St': float(metrics.get('St', np.nan)),
                'St_quality': float(metrics.get('St_quality', np.nan)),
                'KE_mean': float(metrics.get('KE_mean', np.nan)),
                'convergence_flag': bool(metrics.get('is_converged', False)),
                'timestamp': datetime.now().isoformat(),
            }
            return result
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return None
    
    def _run_lbm_simulation(self, Ux, tau, cylinder_radius, iterations, verbose=False):
        """
        Wrapper around LBM simulator.
        Returns metrics dict and final velocity fields.
        """
        from main import (
            initialize_populations, propagate, bounce_back,
            collision_step, compute_macroscopic, compute_vorticity
        )
        
        # Grid setup
        Nx, Ny = 400, 100
        cs = 1.0 / np.sqrt(3)
        
        # Cylinder setup
        x = np.arange(Nx)
        y = np.arange(Ny)
        X, Y = np.meshgrid(x, y)
        cylinder = (X - Nx//4)**2 + (Y - Ny//2)**2 < cylinder_radius**2
        
        # Initial populations
        rho_init = np.ones((Ny, Nx))
        ux_init = Ux * np.ones((Ny, Nx))
        uy_init = np.zeros((Ny, Nx))
        F = initialize_populations(rho_init, ux_init, uy_init)
        
        # Collection arrays
        metrics = {
            'Cd_values': [],
            'Cl_values': [],
            'St_wake': [],
            'KE_values': [],
        }
        
        # Simulation loop
        for t in range(iterations):
            # Propagate
            F = propagate(F, Nx, Ny)
            
            # Bounce-back on cylinder
            F = bounce_back(F, cylinder, Nx, Ny)
            
            # Boundary conditions (inlet/outlet)
            F[:, 0, :] = initialize_populations(
                np.ones(Ny), np.ones(Ny)*Ux, np.zeros(Ny)
            )[:, 0, :]
            
            # Collision
            rho = np.sum(F, axis=2)
            ux = (np.sum(F[:,:,:9] * np.array([[i, 0] for i in range(9)])[:, 0], axis=2)) / rho
            uy = (np.sum(F[:,:,:9] * np.array([[0, i] for i in range(9)])[:, 1], axis=2)) / rho
            
            # Store on cylinder only
            ux[cylinder] = 0
            uy[cylinder] = 0
            
            F = collision_step(F, rho, ux, uy, tau)
            
            # Collect metrics every 500 steps
            if t % 500 == 0 and t > 0:
                curl = compute_vorticity(ux, uy)
                KE = 0.5 * np.mean(rho * (ux**2 + uy**2))
                metrics['KE_values'].append(KE)
                
                if verbose and t % 5000 == 0:
                    print(f"  Iteration {t}/{iterations}, KE={KE:.6f}")
        
        # Post-simulation analysis
        curl_final = compute_vorticity(ux, uy)
        
        # Extract metrics from final state
        final_metrics = {
            'Cd': np.mean(metrics['Cd_values'][-100:]) if metrics['Cd_values'] else np.nan,
            'Cd_std': np.std(metrics['Cd_values'][-100:]) if metrics['Cd_values'] else np.nan,
            'Cl_rms': np.std(metrics['Cl_values']) if metrics['Cl_values'] else np.nan,
            'Cl_mean': np.mean(metrics['Cl_values']) if metrics['Cl_values'] else np.nan,
            'St': np.nan,  # Would compute from FFT
            'St_quality': np.nan,
            'KE_mean': np.mean(metrics['KE_values'][-100:]) if metrics['KE_values'] else np.nan,
            'is_converged': len(metrics['KE_values']) > 10 and 
                           np.std(metrics['KE_values'][-10:]) < 0.001*np.mean(metrics['KE_values'][-10:])
        }
        
        return final_metrics, (ux, uy)
    
    def generate_dataset(self, num_samples=50, iterations=50000):
        """
        Generate full training dataset.
        
        Args:
            num_samples: Number of parameter sets
            iterations: Timesteps per simulation
        """
        print(f"Generating {num_samples} training samples...")
        params_list = self.generate_parameter_sweep(num_samples)
        
        for i, params in enumerate(params_list):
            print(f"\n[{i+1}/{num_samples}] Running simulation...")
            result = self.run_single_simulation(params, iterations, verbose=False)
            
            if result:
                self.data.append(result)
                print(f"✓ Cd={result['Cd']:.3f}, St={result['St']:.3f}, Re={result['Re']:.1f}")
            else:
                print(f"✗ Simulation failed")
        
        print(f"\nGenerated {len(self.data)} successful simulations")
        return self.data
    
    def save_to_hdf5(self):
        """Save dataset to HDF5 format."""
        if not self.data:
            print("No data to save!")
            return
        
        with h5py.File(self.output_path, 'w') as f:
            # Create datasets
            n = len(self.data)
            
            # Input features
            f.create_dataset('Re', data=[d['Re'] for d in self.data])
            f.create_dataset('radius', data=[d['cylinder_radius'] for d in self.data])
            f.create_dataset('Ux', data=[d['Ux'] for d in self.data])
            f.create_dataset('tau', data=[d['tau'] for d in self.data])
            
            # Output targets
            f.create_dataset('Cd', data=[d['Cd'] for d in self.data])
            f.create_dataset('Cd_std', data=[d['Cd_std'] for d in self.data])
            f.create_dataset('Cl_rms', data=[d['Cl_rms'] for d in self.data])
            f.create_dataset('Cl_mean', data=[d['Cl_mean'] for d in self.data])
            f.create_dataset('St', data=[d['St'] for d in self.data])
            f.create_dataset('St_quality', data=[d['St_quality'] for d in self.data])
            f.create_dataset('KE_mean', data=[d['KE_mean'] for d in self.data])
            f.create_dataset('convergence', data=[d['convergence_flag'] for d in self.data])
            
            # Metadata
            f.attrs['num_samples'] = n
            f.attrs['generated'] = datetime.now().isoformat()
            f.attrs['description'] = 'Phase 2 training data: LBM surrogate model'
            f.attrs['parameters'] = json.dumps({
                'Re_range': [20, 100],
                'radius_range': [8, 18],
                'Ux_range': [0.05, 0.15]
            })
        
        print(f"✓ Saved {len(self.data)} samples to {self.output_path}")
    
    def load_from_hdf5(self):
        """Load dataset from HDF5."""
        if not self.output_path.exists():
            print(f"File not found: {self.output_path}")
            return None
        
        with h5py.File(self.output_path, 'r') as f:
            n = f.attrs['num_samples']
            data = {
                'Re': f['Re'][:],
                'radius': f['radius'][:],
                'Ux': f['Ux'][:],
                'Cd': f['Cd'][:],
                'Cl_rms': f['Cl_rms'][:],
                'St': f['St'][:],
            }
        
        print(f"✓ Loaded {n} samples from {self.output_path}")
        return data


def main():
    parser = argparse.ArgumentParser(description='Generate training data for Phase 2')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--iterations', type=int, default=50000, help='Iterations per simulation')
    parser.add_argument('--output', type=str, default='training_data.h5', help='Output HDF5 file')
    parser.add_argument('--load', action='store_true', help='Load existing data')
    
    args = parser.parse_args()
    
    generator = TrainingDataGenerator(output_path=args.output)
    
    if args.load:
        generator.load_from_hdf5()
    else:
        print(f"Phase 2: Training Data Generation")
        print(f"Generating {args.num_samples} samples × {args.iterations} iterations each")
        print(f"Estimated time: ~{args.num_samples * args.iterations / 1500:.0f} minutes")
        print()
        
        # Generate data
        data = generator.generate_dataset(num_samples=args.num_samples, iterations=args.iterations)
        
        # Save
        generator.save_to_hdf5()
        
        print("\n" + "="*60)
        print("Phase 2 Training Data: COMPLETE")
        print("="*60)
        print(f"Ready for surrogate model training!")
        print(f"Next: python surrogate_model.py --train")


if __name__ == '__main__':
    main()
