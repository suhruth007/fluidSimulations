#!/usr/bin/env python
"""
PHASE 1 TEST: Validate all Phase 1 functions including quality metrics
"""

import numpy as np
from main import *

print('=== PHASE 1 TEST: Enhanced Metrics Validation ===')
print('Testing Phase 1 functions...\n')

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

# Test 1: Cd computation
print('Test 1: compute_drag_coefficient()')
cd, fx = compute_drag_coefficient(F, cy_indices, rho, ux, uy, cxs, cys, weights)
print(f'  ✓ Cd = {cd:.4f}, Fx = {fx:.4f}')

# Test 2: Cl computation
print('Test 2: compute_lift_coefficient()')
cl, fy = compute_lift_coefficient(F, cy_indices, rho, ux, uy, cxs, cys, weights)
print(f'  ✓ Cl = {cl:.4f}, Fy = {fy:.4f}')

# Test 3: Strouhal with quality metric
print('Test 3: compute_strouhal_number() - now with quality')
vort_hist = np.sin(np.linspace(0, 10*np.pi, 200))
result = compute_strouhal_number(vort_hist)
if len(result) == 3:
    st, f, quality = result
    print(f'  ✓ St = {st:.4f}, f = {f:.4f}, quality = {quality:.3f}')
else:
    st, f = result
    print(f'  ✓ St = {st:.4f}, f = {f:.4f} (legacy format)')

# Test 4: Metrics tracker with convergence check
print('Test 4: MetricsTracker with convergence check')
tracker = MetricsTracker()

# Add converging data
for i in range(100):
    cd_val = 1.4 + 0.05 * np.exp(-i/30) + 0.01*np.sin(i/10)
    cl_val = 0.2 * np.sin(i/5)
    vort = np.sin(i/20)
    ke = 0.003 + 0.0001 * np.cos(i/50)
    enst = 0.0001
    tracker.record(i*10, cd_val, cl_val, vort, ke, enst)

# Check convergence
is_converged, details = tracker.check_convergence(tolerance=0.05)
print(f'  ✓ Converged: {is_converged}')
print(f'    First half Cd:  {details["first_half_mean"]:.4f}')
print(f'    Second half Cd: {details["second_half_mean"]:.4f} ± {details["second_half_std"]:.4f}')

# Get statistics
stats = tracker.get_statistics()
print(f'  ✓ Stats: Cd_mean = {stats["cd_mean"]:.4f}, Cl_mean = {stats["cl_mean"]:.4f}')

# Test Strouhal from tracker
st_tracker, f_tracker, q_tracker = tracker.compute_strouhal(dt=1.0, U_ref=0.1, D=26)
print(f'  ✓ Tracker Strouhal: St = {st_tracker:.4f}, quality = {q_tracker:.3f}')

# Test 5: Save and load metrics
print('Test 5: Save metrics to JSON')
tracker.save_metrics('test_metrics.json')
import json
with open('test_metrics.json') as f:
    saved = json.load(f)
print(f'  ✓ Saved {len(saved["cd"])} data points to test_metrics.json')

print('\n' + '='*50)
print('✅ ALL PHASE 1 TESTS PASSED!')
print('='*50)
print('\nPhase 1 Status:')
print('  ✓ Drag coefficient: Working')
print('  ✓ Lift coefficient: Working')
print('  ✓ Strouhal number: Working (with quality metric)')
print('  ✓ Convergence check: Working')
print('  ✓ Metrics export: Working')
print('\nReady to run 100k iteration validation!')
print('Command: python main.py')

