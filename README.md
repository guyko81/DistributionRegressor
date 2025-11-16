# DistributionRegressor

Nonparametric distributional regression using LightGBM. Predicts full probability distributions p(y|x) instead of just point estimates.

## Overview

`DistributionRegressor` provides multiple methods to predict complete probability distributions over continuous targets. Unlike traditional regression that outputs a single value, this package provides:

- Full probability distributions p(y|x)
- Quantiles and prediction intervals
- Uncertainty estimates
- Point predictions (mean, median, mode)

**No parametric assumptions** - the distribution shape is learned from data.

## Recommended: DistributionRegressorRandomForest

The **Random Forest Single-Model approach** (`DistributionRegressorRandomForest`) is the **recommended method** for most use cases:

### Why This Method?

- ⚡ **Significantly Faster Training**: Trains only K models instead of K×N models (where K=estimators, N=bins)
- 🎯 **Optimized Architecture**: Uses bin_id as a feature to train one classifier per estimator
- 🌲 **Ensemble Power**: Combines multiple random binning strategies like Random Forest
- 📊 **Full Distributions**: Predicts complete probability distributions, not just point estimates
- 🔧 **Simple & Robust**: Fewer hyperparameters to tune, stable performance
- 🎨 **Flexible**: Works with any number of bins, automatically adapts to data range

### Key Features

- 📊 **Full Distribution Estimation**: Complete probability distributions, not just point predictions
- 🎯 **Multiple Prediction Types**: Mean, median, mode, quantiles, intervals, standard deviation
- 🌲 **LightGBM Backend**: Fast, efficient gradient boosting with categorical feature support
- ⚡ **Optimized Training**: Single-model-per-estimator design for speed
- 🔄 **Random Binning**: Each estimator uses random bin boundaries for robust ensemble
- 📈 **Uncertainty Quantification**: Natural uncertainty estimates from learned distributions
- 🔀 **Parallel Processing**: Built-in parallelization support with n_jobs parameter

## Installation

```bash
pip install distribution-regressor
```

This will automatically install all dependencies (numpy, pandas, scikit-learn, lightgbm, joblib).

## Quick Start

```python
import numpy as np
from distribution_regressor import DistributionRegressorRandomForest

# Your data
X_train, y_train = ...  
X_test, y_test = ...    

# Create and train (recommended method)
model = DistributionRegressorRandomForest(
    n_estimators=100,      # Number of trees in forest
    n_bins=10,             # Bins per tree
    n_jobs=-1,             # Use all CPUs
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)                       # Mean predictions
y_median = model.predict_median(X_test)              # Median predictions
y_std = model.predict_std(X_test)                    # Standard deviation
intervals = model.predict_interval(X_test, 0.95)     # 95% prediction intervals
quantiles = model.predict_quantile(X_test, [0.1, 0.5, 0.9])  # Multiple quantiles

# Full distribution
y_values, distributions = model.predict_distribution(X_test, resolution=100)
```

## How It Works

### DistributionRegressorRandomForest (Recommended)

Uses a **Random Forest philosophy with random binning** to build distribution ensembles:

1. **Random Binning**: Each estimator generates random bin boundaries over the target range
2. **Single Binary Classifier**: For each estimator, train ONE binary classifier with bin_id as a feature
   - Input: (X, bin_id) → Output: probability that y falls in that bin
   - This reduces model count from K×N to K (K=estimators, N=bins)
3. **Ensemble Predictions**: Average predictions across all estimators for robust distributions

**Key Innovation**: Instead of training N separate classifiers per estimator (one per bin), we train a single classifier that takes bin_id as an input feature. This dramatically speeds up training while maintaining prediction quality.

### Architecture Comparison

| Approach | Models Trained | Speed | Memory |
|----------|---------------|-------|--------|
| Naive (N classifiers/estimator) | K×N | Slow | High |
| **Single-Model (this package)** | **K** | **Fast** | **Low** |

For K=100 estimators and N=10 bins: **1000 models → 100 models** (10× fewer!)

## Model Parameters

### DistributionRegressorRandomForest (Recommended)

```python
DistributionRegressorRandomForest(
    # Ensemble configuration
    n_estimators=100,               # Number of trees/estimators in forest
    n_bins=10,                      # Number of bins per tree
    
    # Parallelization
    n_jobs=None,                    # Parallel jobs (-1 = all CPUs, None = 1)
    
    # LightGBM parameters (per classifier)
    lgbm_params={
        'learning_rate': 0.1,       # Boosting learning rate
        'max_depth': 7,             # Maximum tree depth
        'n_estimators': 100,        # Boosting rounds per classifier
        'subsample': 0.8,           # Row sampling fraction
        'colsample_bytree': 0.8,    # Feature sampling fraction
        'reg_alpha': 0.0,           # L1 regularization
        'reg_lambda': 0.0,          # L2 regularization
        'max_bin': 255,             # Max number of bins for features
    },
    
    # Random state
    random_state=42                 # Seed for reproducibility
)
```

**Simple defaults work well!** Just set `n_estimators`, `n_bins`, and `n_jobs` for most cases.

## API Reference

### Training

```python
model.fit(X, y)
```

**Args:**
- `X`: (n_samples, n_features) training features (numpy array or pandas DataFrame)
- `y`: (n_samples,) training targets

The model is scikit-learn compatible and works with numpy arrays or pandas DataFrames.

### Predictions

#### 1. Point Predictions

```python
# Mean (expected value)
y_pred = model.predict(X, resolution=100)
y_mean = model.predict_mean(X, resolution=100)  # Explicit

# Median
y_median = model.predict_median(X, resolution=100)

# Mode (peak of distribution)
y_mode = model.predict_mode(X, resolution=100)
```

**Args:**
- `X`: (n_samples, n_features) input features
- `resolution`: Number of points in y-grid for computing distributions (default=100)

#### 2. Quantiles

```python
quantiles = model.predict_quantile(X, q=0.5, resolution=100)
```

**Args:**
- `q`: float or array of floats in [0, 1]
- Returns shape (n_samples,) if q is scalar, (n_samples, len(q)) if q is array

**Examples:**
```python
# Median
median = model.predict_quantile(X_test, q=0.5)

# Multiple quantiles
q = model.predict_quantile(X_test, q=[0.1, 0.5, 0.9])

# Deciles
q = model.predict_quantile(X_test, q=np.arange(0.1, 1.0, 0.1))
```

#### 3. Prediction Intervals

```python
intervals = model.predict_interval(X, confidence=0.95, resolution=100)
```

**Args:**
- `confidence`: Confidence level (0.95 = 95% interval from 2.5% to 97.5% quantiles)

**Returns:** (n_samples, 2) array of [lower, upper] bounds

**Examples:**
```python
# 95% prediction interval
intervals = model.predict_interval(X_test, confidence=0.95)
lower, upper = intervals[:, 0], intervals[:, 1]

# Check coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"Coverage: {coverage:.1%}")

# 90% interval
intervals = model.predict_interval(X_test, confidence=0.90)
```

#### 4. Full Distribution

```python
y_values, distributions = model.predict_distribution(X, resolution=100)
```

**Returns:**
- `y_values`: (resolution,) array of y-grid points
- `distributions`: (n_samples, resolution) array of probability densities

**Example:**
```python
y_values, distributions = model.predict_distribution(X_test[:1], resolution=200)

import matplotlib.pyplot as plt
plt.plot(y_values, distributions[0])
plt.axvline(y_test[0], color='r', linestyle='--', label='True')
plt.xlabel('y')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
```

#### 5. Standard Deviation

```python
stds = model.predict_std(X, resolution=100)
```

Returns the standard deviation of the predicted distribution for each sample.

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from distribution_regressor import DistributionRegressorRandomForest

# Generate data with heteroscedastic noise
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
y_true = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2]**2
noise = (0.5 + 0.3*np.abs(X[:, 0])) * np.random.randn(n)
y = y_true + noise

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Distribution Regressor
model = DistributionRegressorRandomForest(
    n_estimators=100,
    n_bins=10,
    n_jobs=-1,              # Use all CPUs
    random_state=42
)

model.fit(X_train, y_train)

# Point predictions
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Prediction intervals
intervals = model.predict_interval(X_test, confidence=0.90)
lower, upper = intervals[:, 0], intervals[:, 1]
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.1%}")

# Quantiles
q = model.predict_quantile(X_test, q=[0.1, 0.5, 0.9])
print(f"10th, 50th, 90th quantiles shape: {q.shape}")

# Standard deviations
stds = model.predict_std(X_test)
print(f"Mean predicted std: {stds.mean():.3f}")

# Visualize distribution for one example
idx = 0
y_values, distributions = model.predict_distribution(X_test[idx:idx+1], resolution=200)

plt.figure(figsize=(10, 6))
plt.plot(y_values, distributions[0], 'b-', lw=2, label='Predicted Distribution')
plt.axvline(y_test[idx], color='r', linestyle='--', lw=2, 
            label=f'True: {y_test[idx]:.2f}')
plt.axvline(y_pred[idx], color='g', linestyle='--', lw=2,
            label=f'Mean: {y_pred[idx]:.2f}')
plt.axvspan(lower[idx], upper[idx], alpha=0.2, 
            color='gray', label='90% interval')
plt.xlabel('y value')
plt.ylabel('Probability Density')
plt.title('Predicted Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Advanced Usage

### Custom LightGBM Parameters

```python
model = DistributionRegressorRandomForest(
    n_estimators=100,
    n_bins=10,
    lgbm_params={
        'learning_rate': 0.05,      # Slower learning
        'max_depth': 5,             # Shallower trees
        'reg_alpha': 0.1,           # L1 regularization
        'reg_lambda': 0.1,          # L2 regularization
        'num_leaves': 31,
        'min_child_samples': 5
    },
    n_jobs=-1,
    random_state=42
)
```

### Higher Resolution Distributions

```python
# Use more points for smoother distributions
y_values, distributions = model.predict_distribution(X_test, resolution=500)

# All prediction methods accept resolution parameter
y_pred = model.predict(X_test, resolution=200)
quantiles = model.predict_quantile(X_test, q=[0.1, 0.9], resolution=200)
```

### Working with DataFrames

```python
import pandas as pd

# Works seamlessly with pandas
df_train = pd.DataFrame(X_train, columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
df_test = pd.DataFrame(X_test, columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])

model = DistributionRegressorRandomForest(n_estimators=100, random_state=42)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)
```

## Comparison with Other Methods

| Method | Distributions | Uncertainty | Parametric | Speed | Complexity |
|--------|--------------|-------------|------------|-------|------------|
| Linear Regression | ✗ | ✗ | Gaussian | ⚡⚡⚡ | Low |
| Random Forest | ✗ | Limited | ✗ | ⚡⚡ | Low |
| Gradient Boosting | ✗ | ✗ | ✗ | ⚡⚡⚡ | Low |
| Quantile Regression | Limited | ✓ | ✗ | ⚡⚡ | Medium |
| NGBoost | ✓ | ✓ | ✓ (Gaussian) | ⚡⚡ | Medium |
| **DistributionRegressorRandomForest** | **✓** | **✓** | **✗ (Non-parametric)** | **⚡⚡** | **Low** |

### Advantages

✅ **No distributional assumptions** - learns any distribution shape  
✅ **Flexible uncertainty** - captures heteroscedastic noise naturally  
✅ **One model, many predictions** - mean, median, mode, quantiles, intervals, std dev from one model  
✅ **Fast training** - optimized single-model-per-estimator design  
✅ **Simple** - few hyperparameters, sensible defaults  
✅ **Parallel** - built-in multi-core support with n_jobs  
✅ **Robust ensemble** - random binning like Random Forest  

### Limitations

⚠️ **Grid-based** - evaluates distributions on a grid (adjustable resolution)  
⚠️ **Memory** - more memory than single-point regressors (but less than naive approaches)  

## Best Practices

### 1. Number of Estimators

- **n_estimators=100**: Good default for most cases
- **n_estimators=50**: Faster training, still reasonable
- **n_estimators=200+**: Better distributions, slower training

### 2. Number of Bins

- **n_bins=10**: Good default, balances resolution and speed
- **n_bins=5-7**: Faster, coarser distributions
- **n_bins=15-20**: Finer distributions, slower training

### 3. Parallelization

```python
# Use all CPUs for significant speedup
model = DistributionRegressorRandomForest(n_jobs=-1)
```

### 4. Resolution

- **resolution=100**: Good default for most predictions
- **resolution=200-500**: Smoother distributions for visualization
- **resolution=50**: Faster inference when precision isn't critical

## Troubleshooting

### Poor Calibration (intervals don't match nominal coverage)

**Solution:**
- Increase `n_estimators` (more trees)
- Adjust `n_bins` (try 15-20 for finer resolution)
- Tune LGBM regularization

### Slow Training

**Solution:**
- Reduce `n_estimators` (try 50)
- Reduce `n_bins` (try 5-7)
- Use `n_jobs=-1` for parallelization
- Lower LGBM `n_estimators` in `lgbm_params`

### Slow Predictions

**Solution:**
- Reduce `resolution` parameter (try 50)
- Reduce `n_estimators` (fewer trees)
- Use `n_jobs=-1` for parallel predictions

### Memory Issues

**Solution:**
- Reduce `resolution` in predictions
- Process test set in smaller batches
- Reduce `n_estimators`

## Citation

If you use this in your research, please cite:

```bibtex
@software{distributionregressor2025,
  title={DistributionRegressor: Nonparametric Distributional Regression},
  author={Gabor Gulyas},
  year={2025},
  url={https://github.com/guyko81/DistributionRegressor}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Work

- [NGBoost](https://github.com/stanfordmlgroup/ngboost) - Natural Gradient Boosting for probabilistic prediction
- [LightGBM](https://github.com/microsoft/LightGBM) - Efficient gradient boosting framework
- [Conformalized Quantile Regression](https://github.com/yromano/cqr) - Distribution-free prediction intervals

## Alternative Methods

The package also includes the original contrastive learning approach (`DistributionRegressor`) with hard/soft negative sampling. See the source code for details. However, **DistributionRegressorRandomForest is recommended** for most use cases due to its simplicity, speed, and robust performance.

## Changelog

### Version 1.1.0 (2025)
- **NEW**: `DistributionRegressorRandomForest` - optimized random forest approach (recommended)
- Single-model-per-estimator design for 10× faster training
- Built-in parallelization with n_jobs parameter
- Simplified API and hyperparameters

### Version 1.0.0 (2025)
- Initial release with `DistributionRegressor`
- Contrastive learning with hard/soft negative modes
- Comprehensive prediction API
- Scikit-learn compatibility
