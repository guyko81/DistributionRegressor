# DistributionRegressor

Nonparametric distributional regression using LightGBM. Predicts full probability distributions p(y|x) instead of just point estimates.

## Overview

`DistributionRegressor` provides a robust way to predict complete probability distributions over continuous targets. Unlike standard regression that outputs a single value, this package allows you to:

- **Predict full probability distributions** (arbitrary shapes: multimodal, skewed, etc.)
- **Quantify uncertainty** with natural confidence intervals
- **Obtain point predictions** (mean, mode/peak, quantiles)

It uses a **Single-Model Soft-Target** approach:
1. **Discretizes the target space** into a grid (the "canvas").
2. **Expands the dataset** by crossing samples with grid points.
3. **Trains a single LightGBM regressor** on "soft targets" (Gaussian-smoothed probabilities) to learn the plausibility of each grid point.

This approach is **fast, stable, and requires minimal tuning**.

## Installation

```bash
pip install distribution-regressor
```

## Quick Start

```python
import numpy as np
from distribution_regressor import DistributionRegressor

# 1. Initialize
model = DistributionRegressor(
    n_bins=50,              # Resolution of the distribution grid
    n_estimators=100,       # Number of boosting trees
    sigma='auto'            # 'auto' automatically estimates noise level
)

# 2. Train
# X: (n_samples, n_features), y: (n_samples,)
model.fit(X_train, y_train)

# 3. Predict Points
y_mean = model.predict(X_test)               # Mean (Expected Value)
y_mode = model.predict_mode(X_test)          # Mode (Most likely value / Peak)
y_median = model.predict_quantile(X_test, 0.5)

# 4. Predict Intervals & Uncertainty
# 10th and 90th percentiles (80% confidence interval)
lower = model.predict_quantile(X_test, 0.1)
upper = model.predict_quantile(X_test, 0.9)

# 5. Predict Full Distribution
grid, dists = model.predict_distribution(X_test)
# grid: (n_bins,) - The y-values
# dists: (n_samples, n_bins) - Probability density for each sample
```

## Key Parameters

```python
DistributionRegressor(
    n_bins=50,              # Number of grid points (higher = more resolution, more RAM)
    sigma='auto',           # Kernel width for soft targets. 'auto' or float.
                            # Controls how "smeared" the probability is around true y.
    output_smoothing=1.0,   # Gaussian smoothing on predicted distribution (0.0 to disable)
    n_estimators=100,       # LightGBM trees
    learning_rate=0.1,      # Learning rate
    random_state=42,        # Seed
    **kwargs                # Passed to LGBMRegressor (e.g., max_depth, num_leaves)
)
```

## How It Works

The model treats regression as a "soft classification" problem over a continuous grid.

1. **Grid Creation**: A grid of `n_bins` points is created covering the range of `y`.
2. **Soft Targets**: For each training sample `(x_i, y_i)`, we assign a probability score to every grid point `g_j` based on its distance to `y_i` (Gaussian kernel). Points close to `y_i` get high scores; far points get low scores.
3. **Single Model**: The dataset is expanded so the model sees inputs `(x_i, g_j)` and predicts the soft score.
4. **Prediction**: At inference, for a new `x`, we query the model for all grid points `g_j` to reconstruct the distribution curve.

## Example Visualization

```python
import matplotlib.pyplot as plt

# Predict distribution for a single sample
grid, dists = model.predict_distribution(X_test[0:1])

plt.plot(grid, dists[0], label='Predicted PDF')
plt.axvline(y_test[0], color='r', linestyle='--', label='True Value')
plt.legend()
plt.show()
```

## Citation

```bibtex
@software{distributionregressor2025,
  title={DistributionRegressor: Nonparametric Distributional Regression},
  author={Gabor Gulyas},
  year={2025},
  url={https://github.com/guyko81/DistributionRegressor}
}
```

## License

MIT License
