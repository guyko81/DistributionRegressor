# DistributionRegressor

Nonparametric distributional regression using LightGBM. Predicts full probability distributions p(y|x) instead of just point estimates.

**[Documentation](https://guyko81.github.io/DistributionRegressor/)** | **[PyPI](https://pypi.org/project/distribution-regressor/)** | **[Examples](https://guyko81.github.io/DistributionRegressor/#examples)**

## Overview

`DistributionRegressor` provides a robust way to predict complete probability distributions over continuous targets. Unlike standard regression that outputs a single value, this package allows you to:

- **Predict full probability distributions** (arbitrary shapes: multimodal, skewed, etc.)
- **Quantify uncertainty** with natural confidence intervals
- **Obtain point predictions** (mean, mode/peak, quantiles)

It uses a **CDF-based** approach:
1. **Discretizes the target space** into a grid of threshold points.
2. **Learns the conditional CDF** F(τ|x) = P(Y ≤ τ | X = x) using binary targets and logistic loss.
3. **Enforces monotonicity** via LightGBM's monotone constraints on the threshold feature.
4. **Recovers the PMF** by differencing the predicted CDF.

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
grids, dists, offsets = model.predict_distribution(X_test)
# grids: (n_samples, n_bins) - Per-sample grid points
# dists: (n_samples, n_bins) - Probability mass for each sample
```

## Key Parameters

```python
DistributionRegressor(
    n_bins=50,              # Number of grid points (higher = more resolution, more RAM)
    use_base_model=False,   # If True, learns residual CDF around a base LGBM prediction
    monte_carlo_training=False,  # If True, sample grid points instead of full expansion
    mc_samples=5,           # MC sample points per observation (when MC enabled)
    n_estimators=100,       # LightGBM trees
    learning_rate=0.1,      # Learning rate
    random_state=42,        # Seed
    **kwargs                # Passed to LGBMRegressor (e.g., max_depth, num_leaves)
)
```

## How It Works

The model learns the conditional CDF using binary classification:

1. **Grid Creation**: A grid of `n_bins` threshold points is created covering the range of `y`.
2. **Binary Targets**: For each training sample `(x_i, y_i)` and threshold `τ_j`, the target is `z_ij = 1{y_i ≤ τ_j}` — simply whether `y_i` falls below the threshold.
3. **Single Model**: A single LightGBM model is trained with cross-entropy loss on `(x_i, τ_j) → z_ij`, with a monotone increasing constraint on `τ_j` to ensure a valid CDF.
4. **Prediction**: At inference, the model predicts F(τ|x) for all grid points, then differences the CDF to recover the probability mass function.

## Example Visualization

```python
import matplotlib.pyplot as plt

# Predict distribution for a single sample
grids, dists, offsets = model.predict_distribution(X_test[0:1])

plt.plot(grids[0], dists[0], label='Predicted PMF')
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
