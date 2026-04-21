# Project Structure & File Organization

**Last Updated**: April 21, 2026  
**Version**: 2.0 (Professional Structure)

---

## Directory Tree

```
fluidSimulations/
│
├── README.md                          # Main project overview
├── LICENSE                            # License file
├── setup.py                           # Package installer (future)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── ARCHITECTURE.md                    # 🆕 System architecture document
├── DEVELOPMENT_HISTORY.md             # 🆕 Project tracker & JIRA
├── PROJECT_STRUCTURE.md               # 🆕 This file
│
├── src/                               # 🆕 Source code (main code)
│   ├── __init__.py                    # Package initializer
│   ├── simulator.py                   # Phase 1: LBM simulator (future refactor)
│   ├── metrics.py                     # Phase 1: Aerodynamic metrics (future)
│   ├── training_data.py               # Phase 2: Data generation
│   ├── surrogate_model.py             # Phase 2: Neural network model
│   └── optimization.py                # Phase 3: Genetic algorithm (future)
│
├── tests/                             # 🆕 Test suite
│   ├── __init__.py                    # Test package initializer
│   ├── test_phase1.py                 # Unit tests for Phase 1
│   ├── test_surrogate.py              # Validation for Phase 2
│   ├── test_integration.py            # E2E tests (future)
│   └── conftest.py                    # Pytest configuration (future)
│
├── data/                              # 🆕 Data directory
│   ├── training_data.h5               # Generated training dataset
│   ├── test_samples.h5                # Test set (fixed)
│   ├── benchmarks/                    # Reference data
│   │   ├── williamson_1989.json       # Literature benchmark
│   │   └── phase1_validation.json     # Phase 1 results
│   └── .gitkeep                       # Ensure folder tracked
│
├── models/                            # 🆕 Trained models
│   ├── surrogate_v1.pth               # Best surrogate model weights
│   ├── flow_predictor_v1.pth          # Phase 3 flow prediction (future)
│   ├── metadata/                      # Model metadata
│   │   ├── surrogate_v1.json          # Normalization + metrics
│   │   └── training_history.json      # Training logs
│   └── .gitkeep                       # Ensure folder tracked
│
├── notebooks/                         # 🆕 Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Explore training data
│   ├── 02_surrogate_training.ipynb    # Train surrogate model
│   ├── 03_optimization_demo.ipynb     # Design optimization example
│   └── README.md                      # Notebook guide
│
├── docs/                              # 🆕 Documentation
│   ├── ARCHITECTURE.md                # System architecture (link)
│   ├── DEVELOPMENT_HISTORY.md         # Project history (link)
│   ├── PROJECT_STRUCTURE.md           # File organization (link)
│   ├── INSTALLATION.md                # Setup & installation guide
│   ├── API_REFERENCE.md               # Complete API documentation
│   ├── PHASE1_GUIDE.md                # Phase 1 detailed guide
│   ├── PHASE2_GUIDE.md                # Phase 2 guide (future)
│   ├── PHASE3_GUIDE.md                # Phase 3 guide (future)
│   ├── TROUBLESHOOTING.md             # Common issues & solutions
│   └── QUICKSTART.md                  # Quick start for users
│
├── .github/                           # 🆕 GitHub configuration
│   └── workflows/                     # CI/CD pipelines
│       ├── tests.yml                  # Automated testing
│       ├── lint.yml                   # Code quality checks
│       └── deploy.yml                 # Release automation
│
├── main.py                            # GUI application entry point
├── generate_training_data.py          # Phase 2: Data generation script
├── surrogate_model.py                 # Phase 2: Model training script
├── test_surrogate.py                  # Phase 2: Validation script
├── phase1_metrics.json                # Generated: Phase 1 results
├── training_data.h5                   # Generated: Training dataset
├── best_surrogate_model.pth           # Generated: Trained model
└── surrogate_metadata.json            # Generated: Model parameters
```

---

## Folder Descriptions

### `/src/` - Source Code (Main Implementation)
**Purpose**: Core application code  
**Status**: 🆕 Created for Phase 2, will be reorganized further  
**Contents**:
```
Currently (Phase 2):
- surrogate_model.py       Training data generation + NN model
- training_data.py         Data pipeline implementation

Future (Phase 3+):
- simulator.py             Refactored Phase 1 LBM simulator
- metrics.py               Refactored Phase 1 metrics
- optimization.py          Genetic algorithm for design
- flow_predictor.py        CNN/LSTM for flow fields
```

**Guidelines**:
- Pure Python modules (importable)
- No GUI code (except for command-line interfaces)
- Well-documented with docstrings
- Comprehensive error handling

**Example Usage**:
```python
from src.surrogate_model import AerodynamicSurrogate
model = AerodynamicSurrogate.load('models/surrogate_v1.pth')
prediction = model.predict(Re=40, radius=13, Ux=0.1)
```

---

### `/tests/` - Test Suite
**Purpose**: Automated testing  
**Coverage**: Unit, integration, regression tests  
**Contents**:
```
test_phase1.py          Unit tests for simulator & metrics
test_surrogate.py       Validation for surrogate models
test_integration.py     End-to-end tests (Phase 3)
conftest.py             Pytest configuration (future)
```

**Running Tests**:
```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_surrogate.py

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v
```

---

### `/data/` - Data Directory
**Purpose**: Store datasets and benchmarks  
**Status**: 🆕 Created for Phase 2  
**Contents**:
```
training_data.h5           Generated training dataset (50-100 samples)
test_samples.h5            Fixed test set for validation
benchmarks/
  ├── williamson_1989.json Literature reference data
  └── phase1_validation.json Phase 1 benchmark results
```

**HDF5 File Structure**:
```
training_data.h5
├── Re                   Reynolds numbers [N,]
├── radius               Cylinder radius [N,]
├── Ux                   Inlet velocity [N,]
├── Cd                   Drag coefficient [N,] (target)
├── Cl_rms               Lift RMS [N,] (target)
├── St                   Strouhal number [N,] (target)
└── metadata
    ├── num_samples: 50
    ├── generated: 2026-04-21
    └── parameters: JSON
```

**Important**: Never commit HDF5 files (large, binary). Update .gitignore.

---

### `/models/` - Trained Models
**Purpose**: Store neural network weights and metadata  
**Status**: 🆕 Created for Phase 2  
**Contents**:
```
surrogate_v1.pth                Latest surrogate model (PyTorch format)
metadata/
  ├── surrogate_v1.json         Normalization parameters
  └── training_history.json     Training loss, metrics
```

**Model Format**:
```
.pth file:  PyTorch state_dict (model weights only)
.json file: Metadata (normalization, training info)
```

**Versioning Strategy**:
```
v1: Initial training (50 samples)
v2: With 100 samples
v3: Hyperparameter tuning
etc.

Always keep best model:
- best_surrogate_model.pth
- surrogate_metadata.json
```

**Important**: Do NOT commit model files (50-200MB). Use .gitignore.

---

### `/notebooks/` - Jupyter Notebooks
**Purpose**: Interactive exploration and visualization  
**Status**: 🆕 Created for Phase 2  
**Contents**:
```
01_data_exploration.ipynb     Analyze training dataset distribution
02_surrogate_training.ipynb   Train and visualize surrogate model
03_optimization_demo.ipynb    Design optimization examples
README.md                     How to use notebooks
```

**Running Notebooks**:
```bash
jupyter notebook
# Opens browser to notebook interface
```

**Notebook Best Practices**:
- One topic per notebook
- Clear markdown sections
- Output saved for review
- Don't commit `.ipynb_checkpoints/`

---

### `/docs/` - Documentation
**Purpose**: User and developer guides  
**Status**: 🆕 Created for Phase 2  
**Contents**:
```
ARCHITECTURE.md          System design & modules (comprehensive)
DEVELOPMENT_HISTORY.md   Project tracker & JIRA (comprehensive)
PROJECT_STRUCTURE.md     This file - file organization
INSTALLATION.md          Setup & dependency installation
API_REFERENCE.md         Function documentation & examples
PHASE1_GUIDE.md          Phase 1 detailed workflow
PHASE2_GUIDE.md          Phase 2 guide (future)
PHASE3_GUIDE.md          Phase 3 guide (future)
TROUBLESHOOTING.md       Common issues & solutions
QUICKSTART.md            5-minute quick start
```

**Documentation Guidelines**:
- Markdown format (.md)
- Clear headings and sections
- Code examples for all features
- Links to other docs
- Keep updated with code changes

---

### `/.github/workflows/` - CI/CD Pipelines
**Purpose**: Automated testing and deployment  
**Status**: 🆕 Created for Phase 2 (empty, to be configured)  
**Contents**:
```
tests.yml          Run pytest on every push
lint.yml           Code quality checks (flake8, black)
deploy.yml         Build & publish releases
```

**GitHub Actions Configuration** (example):
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

---

## Root-Level Files

### `README.md`
**Purpose**: Project overview for new users  
**Audience**: General public, new contributors  
**Contents**:
- Quick description
- Feature highlights
- Installation instructions
- Quick start example
- Links to other docs

### `ARCHITECTURE.md`
**Purpose**: System design for developers  
**Audience**: Developers, system architects  
**Contents**:
- Module structure
- Data flow
- Design decisions
- Performance targets
- Scalability roadmap

### `DEVELOPMENT_HISTORY.md`
**Purpose**: Project timeline and tracker (JIRA-like)  
**Audience**: Project managers, developers  
**Contents**:
- Phase timeline
- Achievements by phase
- Backlog & issues
- Lessons learned
- Risk assessment
- Next actions

### `PROJECT_STRUCTURE.md`
**Purpose**: File organization (this file)  
**Audience**: Developers  
**Contents**:
- Directory tree
- Folder descriptions
- File naming conventions
- Organization principles

### `requirements.txt`
**Purpose**: Python dependencies  
**Format**:
```
# Phase 1: Core
numpy>=1.20
matplotlib>=3.0
numba>=0.55

# Phase 2: ML
torch>=1.9
h5py>=3.0
scipy>=1.7

# Development
pytest>=6.0
jupyter>=1.0
black>=21.0
```

### `.gitignore`
**Purpose**: Files to exclude from git  
**Should Include**:
```
# Data files (large)
*.h5
data/training_data.h5
data/test_samples.h5

# Model files (large)
*.pth
*.pt
models/

# Generated files
*.json (model metadata)
phase1_metrics.json

# Python
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/

# Virtual env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## File Naming Conventions

### Python Source Files
```
module_name.py          Lowercase with underscores
example: surrogate_model.py, training_data.py

Class Naming:           PascalCase
example: class AerodynamicSurrogate

Function Naming:        snake_case
example: def run_simulation()

Variable Naming:        snake_case
example: iterations_per_second = 120
```

### Data Files
```
training_data.h5       Training dataset
test_samples.h5        Test set
phase1_metrics.json    Phase 1 results
```

### Model Files
```
surrogate_v1.pth       Model weights (version #)
surrogate_v1.json      Model metadata
training_history.json  Training logs
```

### Documentation
```
UPPERCASE.md           Main docs (ARCHITECTURE, README)
lowercase.md           Specific guides (phase1_guide.md)
                       or phase1 guides (PHASE1_GUIDE.md)
```

---

## Git Organization

### Repository Structure
```
main (production)
  ├── Stable releases
  ├── All tests passing
  └── Well-documented

develop (development)
  ├── Active development
  ├── New features
  └── May have failing tests
```

### Branch Naming
```
feature/phase3-flow-prediction
bugfix/normalization-issue
docs/update-readme
release/v2.0
```

### Commit Messages
```
Short (50 chars):
Phase 2: ML/AI Surrogate Models Implementation

Long:
- Add generate_training_data.py: Training data pipeline
- Add surrogate_model.py: Neural network model
- Add test_surrogate.py: Validation tests
- Update main.py: GUI prediction panel
- Update README.md: Phase 2 documentation
```

---

## Development Workflow

### Adding a New Feature
```
1. Create branch:    git checkout -b feature/my-feature
2. Make changes      Implement + test in src/
3. Add tests         Create tests/test_my_feature.py
4. Commit:           git commit -m "Add my feature"
5. Push:             git push origin feature/my-feature
6. Create PR         Submit pull request
7. Merge to main:    After review + tests pass
```

### Adding Data Files
```
1. Generate data        python generate_training_data.py
2. Verify format        Check HDF5 structure
3. Save to /data/       Move training_data.h5 to data/
4. Update .gitignore    Don't commit large files
5. Document             Add notes to DEVELOPMENT_HISTORY.md
```

### Adding Documentation
```
1. Write in Markdown    Clear sections + examples
2. Save to /docs/       Organize by topic
3. Link from README     Update main README
4. Commit              git add docs/, git commit
```

---

## Size Guidelines

### File Sizes
```
.py files:      500-1000 lines (max 2000)
.md files:      500-2000 lines
.ipynb files:   Clean output, <50MB
HDF5 files:     >50MB (don't commit)
.pth files:     >50MB (don't commit)
```

### Performance
```
Imports:        <1 second per script
Tests:          <30 seconds total
Documentation:  <5 seconds to render
```

---

## Integration Points

### Phase 1 ↔ Phase 2
```
Input:  LBM simulator output (vorticity, velocity, metrics)
Output: Training data for neural networks
File:   training_data.h5 in /data/
```

### Phase 2 ↔ Phase 3 (Planned)
```
Input:  Trained surrogate model (surrogate_v1.pth)
Output: Optimized parameters
File:   results stored in /models/
```

### GUI ↔ Models
```
main.py → surrogate_model.py: Load model for predictions
main.py → training_data.py: Generate new data
Results: Display in GUI, save to JSON
```

---

## Maintenance & Cleanup

### Regular Tasks
```
Weekly:
  - Review new files in /data/
  - Check model accuracy metrics
  - Update DEVELOPMENT_HISTORY.md

Monthly:
  - Archive old model versions
  - Clean up test data
  - Review documentation
  - Update .gitignore as needed

Quarterly:
  - Major documentation review
  - Refactor old code
  - Performance optimization
```

### Disk Space Management
```
Raw simulation data:     Don't keep (regenerate as needed)
Training datasets:       Keep best version only
Model checkpoints:       Keep last 3 versions
Results/metrics:         Archive old results

Total storage budget:    ~2GB per phase
```

---

## Scalability Considerations

### As Project Grows
```
Phase 1: Single file (main.py) ✅
Phase 2: Multiple files (src/) ✅
Phase 3: Organize by feature (src/phase3/)
Phase 4: Consider package structure (setup.py)
```

### Data Growth
```
Phase 1: Minimal data
Phase 2: 50-100 training samples (~500MB) → /data/
Phase 3: 1000+ samples (~5GB) → Separate /datasets/
Phase 4: Distributed storage (S3, cloud)
```

### Computing Power
```
Local: Current setup (CPU-based)
GPU Ready: PyTorch CUDA support
Cloud Ready: Docker containers
Distributed: Future with Kubernetes
```

---

## Template: Adding a New Component

### When Adding Phase 3 Feature

**1. Create source file**
```
src/flow_predictor.py
  - class FlowFieldPredictor(nn.Module)
  - def train()
  - def predict()
```

**2. Create test file**
```
tests/test_flow_predictor.py
  - test_model_initialization()
  - test_forward_pass()
  - test_prediction_shape()
```

**3. Create documentation**
```
docs/PHASE3_GUIDE.md
  - Quick start
  - Architecture diagram
  - API reference
  - Examples
```

**4. Update main docs**
```
ARCHITECTURE.md           Add Phase 3 architecture
DEVELOPMENT_HISTORY.md    Add Phase 3 timeline
PROJECT_STRUCTURE.md      Add new files
README.md                 Mention new feature
```

**5. Create notebook**
```
notebooks/04_flow_prediction_demo.ipynb
  - Load data
  - Train model
  - Visualize results
```

**6. Commit everything**
```
git add src/ tests/ docs/ notebooks/
git commit -m "Phase 3: Flow field prediction implementation"
```

---

## Quick Reference

### Common Commands
```bash
# Run simulator
python main.py

# Generate training data
python generate_training_data.py --num_samples 50

# Train surrogate
python surrogate_model.py --train --data data/training_data.h5

# Run tests
pytest tests/ -v

# Run Jupyter
jupyter notebook

# Check project size
du -sh .
```

### Important Paths
```
Source code:     src/
Tests:           tests/
Documentation:   docs/
Data:            data/
Models:          models/
Notebooks:       notebooks/
Entry point:     main.py
```

---

**Document Version**: 2.0  
**Last Updated**: April 21, 2026  
**Next Review**: After Phase 3 start

