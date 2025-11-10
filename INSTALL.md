# Installation Guide

## For Users

### From PyPI (when published)
```bash
pip install distribution-regressor
```

### From Source
```bash
git clone https://github.com/yourusername/distribution-regressor.git
cd distribution-regressor
pip install .
```

## For Developers

### Clone and Install in Development Mode
```bash
git clone https://github.com/yourusername/distribution-regressor.git
cd distribution-regressor
pip install -e ".[dev]"
```

This installs:
- The package in editable mode (changes reflect immediately)
- Development dependencies (pytest, black, flake8)

### Run Tests
```bash
pytest
```

### Run Examples
```bash
python examples/basic_usage.py
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- lightgbm >= 3.0.0

## Quick Test

After installation, test with:

```python
from distribution_regressor import DistributionRegressor
import numpy as np

# Generate simple data
X = np.random.randn(100, 3)
y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

# Train model
model = DistributionRegressor(n_estimators=50, verbose=0)
model.fit(X[:80], y[:80])

# Predict
y_pred = model.predict(X[80:])
print(f"Predictions: {y_pred}")
print("✅ Installation successful!")
```

