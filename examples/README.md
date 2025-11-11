# Examples

This directory contains example scripts demonstrating various capabilities of DistributionRegressor.

## Running the Examples

First, install the package:
```bash
pip install distribution-regressor
```

Then run any example:
```bash
python examples/basic_usage.py
```

## Available Examples

### 1. `basic_usage.py`
**Introduction to DistributionRegressor**

A complete introduction covering:
- Training the model
- Point predictions (mean)
- Negative log-likelihood evaluation
- Prediction intervals
- Quantile predictions
- Sampling from distributions

**Best for:** First-time users wanting a quick overview

```bash
python examples/basic_usage.py
```

---

### 2. `uncertainty_quantification.py`
**Capturing Heteroscedastic Uncertainty**

Demonstrates how DistributionRegressor automatically captures varying uncertainty:
- Heteroscedastic data (noise varies with input)
- Multiple prediction intervals (50%, 90%, 99%)
- Quantile predictions
- Visualization of uncertainty bands

**Best for:** Understanding how the model quantifies uncertainty

**Output:** Creates `uncertainty_quantification.png`

```bash
python examples/uncertainty_quantification.py
```

---

### 3. `distribution_visualization.py`
**Visualizing Full Probability Distributions**

Shows the complete predicted probability distributions:
- Full distribution plots for individual predictions
- True values vs predicted distributions
- Prediction intervals overlaid on distributions
- Sampling from predicted distributions

**Best for:** Understanding what "distributional regression" means

**Output:** Creates `distribution_visualization.png`

```bash
python examples/distribution_visualization.py
```

---

### 4. `comparison_with_standard_regression.py`
**DistributionRegressor vs Standard LightGBM**

Direct comparison showing advantages over standard regression:
- Point prediction accuracy (MSE, MAE)
- Probabilistic evaluation (NLL)
- Prediction interval coverage
- Uncertainty visualization

**Best for:** Understanding when to use DistributionRegressor over standard methods

**Output:** Creates `comparison_with_standard_regression.png`

```bash
python examples/comparison_with_standard_regression.py
```

---

## Example Output

Each example prints informative output to the console and some create visualizations saved as PNG files.

### Console Output Example
```
============================================================
Training DistributionRegressor
============================================================
[Training progress shown here]

============================================================
Making Predictions
============================================================

MSE: 0.2847
NLL: 1.2534
90% interval coverage: 89.5%

Quantiles shape: (200, 3)
First 5 medians: [0.234, -0.512, 1.823, -0.098, 0.445]
```

### Visualization Output
- **uncertainty_quantification.png** - Prediction intervals and quantiles
- **distribution_visualization.png** - Individual probability distributions
- **comparison_with_standard_regression.png** - Comparison with standard regression

## Quick Test

To verify your installation works:

```python
import numpy as np
from distribution_regressor import DistributionRegressor

# Simple test
X = np.random.randn(100, 3)
y = X[:, 0] + np.random.randn(100) * 0.1

model = DistributionRegressor(n_estimators=50, verbose=0)
model.fit(X[:80], y[:80])

print("Prediction:", model.predict(X[80:])[:5])
print("✓ Installation successful!")
```

## Customization

All examples use default or standard parameters. Feel free to experiment with:
- `negative_type`: "hard" or "soft"
- `k_neg`: Number of negative samples (50-200)
- `neg_sampler`: "uniform", "normal", "mix", or "stratified"
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Learning rate for gradient boosting

See the main README for full parameter documentation.

## Need Help?

- **Documentation:** https://github.com/guyko81/DistributionRegressor#readme
- **Issues:** https://github.com/guyko81/DistributionRegressor/issues
- **PyPI:** https://pypi.org/project/distribution-regressor/

