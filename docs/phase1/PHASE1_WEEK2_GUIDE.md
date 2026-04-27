# Phase 1 Week 2: Strouhal Validation & Enhanced Analysis

**Status**: Core metrics (Cd/Cl) complete ✓ | Advanced analysis (St, convergence) complete ✓  
**Duration**: 4 hours hands-on work  
**Target**: Validate Strouhal number with 100k iteration test

---

## What's New This Week

### 1. **Enhanced Strouhal FFT** ⭐
```python
# Now returns signal quality metric (0-1)
st, f, quality = compute_strouhal_number(vorticity_history)
```

**Improvements**:
- Hanning window (reduces spectral leakage)
- 4000-point FFT (better frequency resolution)
- Quality metric shows reliability
- Better frequency range search

**Quality interpretation**:
- `quality > 0.5` - Excellent resolution (100k+ iterations)
- `quality 0.3-0.5` - Good (50k iterations)
- `quality < 0.3` - Weak signal (need longer simulation)

### 2. **Automatic Convergence Checking**
```python
is_converged, details = metrics.check_convergence(tolerance=0.02)
```

Returns:
- `is_converged` - Boolean (Cd std < tolerance)
- `details` - Dict with half-by-half analysis

### 3. **Analysis Script** 📊
```bash
python analyze_phase1.py
```

Generates:
- **phase1_analysis.png** - 4-panel convergence/oscillation plots
- **Console report** - Detailed validation statistics
- **Convergence check** - Automatic PASS/FAIL against benchmarks

---

## Phase 1 Week 2 Workflow

### Step 1: Run 100k Iteration Test (15-20 min)

```bash
python main.py
```

**Settings**:
- Iterations: **100000** (Key change from 50k)
- Color scheme: Red-Blue (bwr)
- Click: **Start Simulation**

**What to expect**:
```
Running LBM simulation...
Iteration 5000/100000...
Iteration 10000/100000...
...
=== PHASE 1 METRICS ===
Drag Coefficient (Cd):  1.4650 ± 0.0089
Lift Coefficient (Cl):  -0.0015 ± 0.3421
Strouhal Number (St):   0.1687
Dominant Frequency:     0.0065 Hz
Signal Quality (FFT):   45.2%

=== BENCHMARK COMPARISON (Re=40) ===
Cd: Expected ≈ 1.465  | Your result: 1.4650 | Error: 0.00%
St: Expected ≈ 0.17   | Your result: 0.1687 | Valid: ✓

✅ PHASE 1 VALIDATION SUCCESSFUL!
```

**Time estimates**:
- Setup: 1 min
- Run: 15-20 min (depends on CPU)
- Dialog display: 1 min
- Total: ~20 minutes

### Step 2: Analyze Results (5 min)

```bash
python analyze_phase1.py
```

**Output files**:
- `phase1_analysis.png` - 4-panel convergence plot
- Console output:
  ```
  ============================================================
  PHASE 1 VALIDATION REPORT
  ============================================================

  📊 CONVERGENCE ANALYSIS:
    Transient Cd mean:     1.4520
    Steady-state Cd mean:  1.4650 ± 0.0089
    Convergence rate:      0.89%
    Converged:             ✓ YES

  🌊 OSCILLATION ANALYSIS:
    Cl mean:               -0.0015
    Cl RMS:                0.2140
    Cl amplitude:          0.8321
    Oscillating:           ✓ YES

  ⚡ ENERGY ANALYSIS:
    Mean KE:               0.003421
    KE std:                0.000012
    Energy settled:        ✓ YES

  🎯 BENCHMARK COMPARISON:
    Expected Cd:           1.465
    Your Cd:               1.4650
    Error:                 0.00%
    Acceptable:            ✓ YES (<2%)

  ============================================================
  ✅ PHASE 1 VALIDATION: PASSED
  ============================================================
  ```

### Step 3: Inspect Output Files

**phase1_metrics.json**:
```json
{
  "time_steps": [50, 100, 150, ...],
  "cd": [1.420, 1.442, 1.463, ...],
  "cl": [-0.05, 0.12, -0.08, ...],
  "ke": [0.00340, 0.00342, ...],
  "enstrophy": [0.000123, ...]
}
```

**phase1_analysis.png**:
- Panel 1: Cd convergence (should flatten out to 1.465)
- Panel 2: Cl oscillations (should oscillate with ~0.17 frequency)
- Panel 3: Kinetic energy settling
- Panel 4: Steady-state Cd histogram

---

## Expected Results (100k iterations)

### Success Criteria ✓

| Metric | Expected | Your Target | Status |
|--------|----------|-------------|--------|
| **Cd** | 1.465 ± 0.005 | 1.460-1.470 | ✓ Pass if < 2% error |
| **Cl RMS** | ~0.35-0.40 | >0.1 | ✓ Pass if oscillating |
| **St** | 0.16-0.20 | 0.160-0.200 | ✓ Pass if in range |
| **Quality** | >0.30 | >0.20 | ✓ Pass if >0.3 |
| **KE settle** | <0.1% change | Stable last 50k | ✓ Pass if settled |

### Typical Progression

**First 5k iterations** (Transient):
- Cd: 1.20 → 1.35 (settling from initial conditions)
- Cl: Barely oscillating
- St: Noisy

**5k-50k iterations** (Transition):
- Cd: 1.35 → 1.46 (converging)
- Cl: Starts clear oscillations
- St: Frequency resolves

**50k-100k iterations** (Steady state):
- Cd: Stable ±0.01 around 1.465
- Cl: Clear periodic oscillations
- St: Sharp FFT peak, quality >0.4

---

## Troubleshooting

### Cd Too Low (<1.40)
**Cause**: Grid too coarse or Re not 40  
**Fix**: Check grid size (400×100) and velocity (U_ref=0.1)  
**Action**: Verify in main.py lines 90-95

### Cd Too High (>1.50)
**Cause**: Numerical diffusion or grid aligned  
**Fix**: Often resolves with longer simulation  
**Action**: Run 150k iterations if still high

### Cl Not Oscillating
**Cause**: Vortex shedding not started  
**Fix**: Simulation too short (need 50k+ for vortices to form)  
**Action**: Run 100k iterations minimum

### St = 0 (No peak found)
**Cause**: FFT window too small or wrong frequency range  
**Fix**: Already improved (4000 points, better window)  
**Action**: Run test_phase1.py to verify FFT code

### Quality < 0.2
**Cause**: Not enough data cycles for FFT  
**Fix**: Run 150k-200k iterations for better resolution  
**Action**: Extend simulation if convergence check passes

---

## Code Changes Summary

### Enhanced `compute_strouhal_number()`
```python
# Now includes:
- Hanning window for spectral filtering
- 4000-point FFT (vs 2000)
- Quality metric (energy ratio)
- Better frequency range search
```

### New `check_convergence()` Method
```python
# MetricsTracker method that returns:
- is_converged: Boolean
- details: Dict with statistics
```

### Updated Validation Logic
```python
# In run_simulation() post-processing:
- Checks cd_error < 2% against benchmark
- Validates st within 0.16-0.20 range
- Checks quality > 0.3
- Prints PASS/FAIL status
```

---

## Phase 1 Week 2 Checklist

- [ ] Run `python main.py` with 100000 iterations
- [ ] Wait for simulation to complete (~20 min)
- [ ] Check dialog shows metrics
- [ ] Run `python analyze_phase1.py`
- [ ] Verify plot generated: `phase1_analysis.png`
- [ ] Check console shows ✅ VALIDATION PASSED
- [ ] Verify all 4 criteria met:
  - [ ] Cd within 2% of 1.465
  - [ ] Cl oscillating (RMS > 0.1)
  - [ ] St between 0.16-0.20
  - [ ] Quality > 0.3

---

## Next Steps (Week 3)

Once validation passes, you'll:

1. **Test VTK Export**
   ```bash
   python -c "from main import export_to_vtk; ..."
   ```
   - Export to ParaView format
   - Visualize velocity/vorticity fields

2. **Fine-tune Grid**
   - Validate 400×100 adequacy
   - Test coarser (300×75) vs finer (500×125)

3. **Grid Convergence Study**
   - Run 3 grid sizes
   - Measure convergence order
   - Verify Richardson extrapolation

---

## Pro Tips 💡

1. **Keep phase1_metrics.json** - Use for plotting later
2. **Run during off-peak** - 20 min simulation, don't interrupt
3. **Monitor CPU** - If <50% utilized, consider parallel optimization next week
4. **Save plots** - phase1_analysis.png proves validation
5. **Check git** - All Phase 1 docs excluded via .gitignore

---

## Resources

- **Code**: See compute_strouhal_number() at line 292
- **Metrics**: Check phase1_metrics.json structure
- **Analysis**: Run analyze_phase1.py anytime after simulation
- **Benchmark**: Reference Williamson (1989) Re=40

---

## Ready? Let's Go! 🚀

```bash
# Command to start Week 2 validation:
python main.py
# → Set iterations to 100000
# → Click Start
# → Wait ~20 minutes
# → Then: python analyze_phase1.py
```

Good luck! Report back with your Cd and St values. 🎯
