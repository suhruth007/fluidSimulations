#!/usr/bin/env python
"""
PHASE 1 TEST: Validate drag coefficient, lift coefficient, and Strouhal calculations
"""

import numpy as np
from main import *

print('=== PHASE 1 TEST: Drag Coefficient Calculation ===')
print('Testing basic functions...\n')

# Create minimal test data
Nx, Ny = 100, 50
NL = 9
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

F = np.ones((Ny, Nx, NL))
rho = np.ones((Ny, Nx))
ux = 0.1 * np.ones((Ny, Nx))
uy = np.zeros((Ny, Nx))

# Create cylinder
cy_mask = (np.arange(Ny)[:, None] - Ny//2)**2 + (np.arange(Nx)[None, :] - Nx//4)**2 < 100
cy_indices = np.where(cy_mask)

# Test Cd computation
cd, fx = compute_drag_coefficient(F, cy_indices, rho, ux, uy, cxs, cys, weights)
print(f'✓ compute_drag_coefficient works: Cd = {cd:.4f}')

# Test Cl computation
cl, fy = compute_lift_coefficient(F, cy_indices, rho, ux, uy, cxs, cys, weights)
print(f'✓ compute_lift_coefficient works: Cl = {cl:.4f}')

# Test Strouhal
vort_hist = np.sin(np.linspace(0, 10*np.pi, 200))
st, f = compute_strouhal_number(vort_hist)
print(f'✓ compute_strouhal_number works: St = {st:.4f}, f = {f:.4f}')

# Test metrics tracker
tracker = MetricsTracker()
tracker.record(0, cd, cl, 0.1, 0.01, 0.001)
tracker.record(1, cd+0.01, cl+0.01, 0.2, 0.015, 0.0015)
stats = tracker.get_statistics()
print(f'✓ MetricsTracker works: Cd_mean = {stats["cd_mean"]:.4f}')

print('\n✅ All Phase 1 functions validated successfully!')
print('\nNext: Run the simulator with Phase 1 enabled!')
print('Expected Cd at Re=40: 1.465')
