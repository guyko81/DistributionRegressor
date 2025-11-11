# DistributionRegressor

Nonparametric distributional regression using LightGBM. Predicts full probability distributions p(y|x) instead of just point estimates.

## Overview

`DistributionRegressor` learns to predict complete probability distributions over continuous targets using contrastive learning. Unlike traditional regression that outputs a single value, this method provides:

- Full probability distributions p(y|x)
- Quantiles and prediction intervals
- Uncertainty estimates
- Point predictions (mean, median, mode)

**No parametric assumptions** - the distribution shape is learned from data.

### Key Features

- 📊 **Full Distribution Estimation**: Complete probability distributions, not just point predictions
- 🎯 **Multiple Prediction Types**: Mean, median, mode, quantiles, intervals, CDF/PDF, sampling
- 🔄 **Two Training Modes**: Hard labels (0/1) or soft labels (distance-weighted)
- 🌲 **LightGBM Backend**: Fast, efficient gradient boosting with categorical feature support
- 🎨 **Flexible Negative Sampling**: Multiple strategies (uniform, normal, stratified, mix)
- 📈 **Uncertainty Quantification**: Natural uncertainty estimates from learned distributions

## Installation

```bash
pip install numpy scikit-learn lightgbm scipy
```

## Quick Start

```python
import numpy as np
from contrastive_LGBM_regression import DistributionRegressor

# Your data
X_train, y_train = ...  
X_test, y_test = ...    

# Create and train
model = DistributionRegressor(
    negative_type="soft",
    n_estimators=200,
    learning_rate=0.1,
    k_neg=100,
    random_state=42
)

model.fit(X_train, y_train, X_test, y_test)

# Predictions
y_pred = model.predict(X_test)                      # Point predictions
lower, upper = model.predict_interval(X_test, 0.1)  # 90% interval [5%, 95%]
quantiles = model.predict_quantiles(X_test)         # Quantiles
samples = model.sample_y(X_test, n_samples=1000)   # Draw samples
```

## How It Works

`DistributionRegressor` uses **contrastive learning** to learn a scoring function s(x, y):

1. **Positive Pairs**: True training examples (x_i, y_i)
2. **Negative Pairs**: Synthetic pairs (x_i, y_neg) where y_neg ≠ y_i  
3. **Learn to Score**: Train model to give high scores to positives, low to negatives
4. **Convert to Distribution**: p(y|x) ∝ exp(s(x,y)) via softmax

### Training Modes

| Aspect | Hard Negatives | Soft Negatives |
|--------|---------------|----------------|
| Internal Training | Binary labels (0/1) | Continuous labels (distance-weighted) |
| LGBM Model | LGBMClassifier | LGBMRegressor |
| Labels | True=1, all negatives=0 | True=1, negatives decay with distance |
| Scaling | Min-Max | StandardScaler (X), z-score (y) |
| Best For | Sharp, peaked distributions | Smooth, spread-out distributions |
| Training Signal | All negatives equal | Negatives weighted by distance |

**Soft negatives** compute labels based on distance:
```python
z = (y_neg - y_true) / soft_label_std
plausibility = 2 * min(CDF(z), 1 - CDF(z))
target = logit(plausibility)
```

## Model Parameters

```python
DistributionRegressor(
    # Training mode
    negative_type="hard",           # "hard" or "soft"
    
    # LightGBM parameters
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    subsample=0.8,
    colsample=0.8,
    lgbm_params=None,               # Additional LGBM params
    
    # Negative sampling
    k_neg=100,                      # Negatives per positive sample
    neg_sampler="mix",              # "uniform", "normal", "mix", "stratified"
    neg_std=2.0,                    # Std for normal sampling (scaled space)
    soft_label_std=1.0,             # Std for soft labels (soft mode only)
    uniform_margin=0.25,            # Margin for uniform sampling
    stratified_global_frac=0.5,     # Global/local mix for stratified
    
    # Features
    use_interactions=True,          # Add interaction features
    categorical_features=None,      # List of categorical column indices
    
    # Training
    early_stopping_rounds=50,
    verbose=1,
    random_state=42
)
```

## API Reference

### Training

```python
model.fit(X, y, X_val=None, y_val=None)
```

**Args:**
- `X`: (n, d) training features
- `y`: (n,) training targets  
- `X_val`, `y_val`: Optional validation data for early stopping

### Predictions

#### 1. Point Predictions

```python
y_pred = model.predict(X, method="mean", y_grid=None)
```

**Args:**
- `method`: "mean" (expected value), "median", or "mode"
- `y_grid`: Optional grid (auto-generated if None)

**Examples:**
```python
y_mean = model.predict(X_test)  # Default: mean
y_median = model.predict(X_test, method="median")
y_mode = model.predict(X_test, method="mode")
```

#### 2. Quantiles

```python
quantiles = model.predict_quantiles(X, qs=[0.05, 0.5, 0.95], y_grid=None)
```

**Returns:** (n, len(qs)) array

**Examples:**
```python
# 5th, 50th, 95th percentiles
q = model.predict_quantiles(X_test, qs=[0.05, 0.5, 0.95])

# Deciles
q = model.predict_quantiles(X_test, qs=np.arange(0.1, 1.0, 0.1))
```

#### 3. Prediction Intervals

```python
lower, upper = model.predict_interval(X, alpha=0.1, y_grid=None)
```

**Args:**
- `alpha`: Significance level (0.1 = 90% interval)

**Examples:**
```python
# 90% prediction interval (5th to 95th percentile)
lower, upper = model.predict_interval(X_test, alpha=0.1)

# 95% prediction interval (2.5th to 97.5th percentile)
lower, upper = model.predict_interval(X_test, alpha=0.05)

# Check coverage
lower, upper = model.predict_interval(X_test, alpha=0.1)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.1%}")
```

#### 4. Distribution Functions

**Predict full CDF:**
```python
cdf, y_grid = model.predict_cdf(X, y_grid=None)
# Returns: (n, n_grid) array of cumulative probabilities
```

**Predict full PDF:**
```python
pdf, y_grid = model.predict_pdf(X, y_grid=None)
# Returns: (n, n_grid) array of probability densities
```

**Example:**
```python
pdf, y_grid = model.predict_pdf(X_test[:1])

import matplotlib.pyplot as plt
plt.plot(y_grid, pdf[0])
plt.axvline(y_test[0], color='r', linestyle='--', label='True')
plt.xlabel('y')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
```

#### 5. Sampling

```python
samples = model.sample_y(X, n_samples=1000, y_grid=None, random_state=None)
```

**Returns:** (n, n_samples) array

**Examples:**
```python
# Draw 1000 samples per test point
samples = model.sample_y(X_test, n_samples=1000, random_state=42)

# Compute statistics
sample_means = samples.mean(axis=1)
sample_stds = samples.std(axis=1)
sample_q95 = np.percentile(samples, 95, axis=1)
```

### Evaluation

#### Negative Log-Likelihood

```python
nll, nll_per_sample = model.negative_log_likelihood(X, y_true, y_grid=None, bandwidth=None)
```

Evaluates the quality of predicted probability distributions by measuring how well they assign probability mass to the true target values. This is the standard evaluation metric for probabilistic regression models.

**Args:**
- `X`: (n, d) feature array
- `y_true`: (n,) true target values
- `y_grid`: Optional grid of y-values (auto-generated if None)
- `bandwidth`: Optional kernel bandwidth for probability smoothing

**Returns:**
- `nll`: Mean negative log-likelihood across all samples
- `nll_per_sample`: (n,) array of per-sample NLL values

Lower values indicate better probabilistic predictions.

**Examples:**
```python
# Evaluate on test set
nll, nll_per_sample = model.negative_log_likelihood(X_test, y_test)

# With bandwidth smoothing for improved robustness
nll_smooth, _ = model.negative_log_likelihood(X_test, y_test, bandwidth=0.1)

# Analyze per-sample performance
worst_predictions = np.argsort(nll_per_sample)[-10:]
```

#### Model Scoring

```python
score = model.score(X, y_true, y_grid=None, bandwidth=None)
```

Returns the negative of negative log-likelihood for scikit-learn compatibility. Higher values indicate better model performance. This enables integration with scikit-learn's model selection utilities.

**Example:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring=None)
```

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from contrastive_LGBM_regression import DistributionRegressor

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

# Train with soft negatives
model = DistributionRegressor(
    negative_type="soft",
    n_estimators=200,
    learning_rate=0.1,
    k_neg=100,
    soft_label_std=1.0,
    neg_sampler="stratified",
    early_stopping_rounds=30,
    verbose=1,
    random_state=42
)

model.fit(X_train, y_train, X_test, y_test)

# Point predictions
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Prediction intervals (90% = 5th to 95th percentile)
lower, upper = model.predict_interval(X_test, alpha=0.1)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.1%}")

# Quantiles
q = model.predict_quantiles(X_test, qs=[0.1, 0.5, 0.9])

# Visualize distribution for one example
idx = 0
pdf, y_grid = model.predict_pdf(X_test[idx:idx+1])

plt.figure(figsize=(10, 6))
plt.plot(y_grid, pdf[0], 'b-', lw=2, label='Predicted PDF')
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

# Sample from distribution
samples = model.sample_y(X_test[:5], n_samples=5000, random_state=42)
for i in range(5):
    print(f"Sample {i}: mean={samples[i].mean():.3f}, "
          f"std={samples[i].std():.3f}, true={y_test[i]:.3f}")
```

## Advanced Usage

### Categorical Features

```python
# Specify categorical columns by index
model = DistributionRegressor(
    categorical_features=[0, 3, 5],  # Columns 0, 3, 5 are categorical
    negative_type="soft"
)

# No one-hot encoding needed - LightGBM handles them natively
model.fit(X_train, y_train)
```

### Negative Sampling Strategies

```python
# 1. Normal: Local sampling around true values
model = DistributionRegressor(
    neg_sampler="normal",
    neg_std=2.0  # Controls spread
)

# 2. Uniform: Global sampling across full range
model = DistributionRegressor(
    neg_sampler="uniform",
    uniform_margin=0.25  # Extra margin
)

# 3. Stratified: Mix of local + global (recommended)
model = DistributionRegressor(
    neg_sampler="stratified",
    stratified_global_frac=0.5,  # 50% global, 50% local
    neg_std=2.0,
    uniform_margin=0.25
)

# 4. Mix: Half normal, half uniform
model = DistributionRegressor(
    neg_sampler="mix"
)
```

### Custom LightGBM Parameters

```python
model = DistributionRegressor(
    negative_type="soft",
    lgbm_params={
        'reg_alpha': 0.1,           # L1 regularization
        'reg_lambda': 0.1,          # L2 regularization
        'min_gain_to_split': 0.01,
        'num_leaves': 31
    }
)
```

### Custom Grid

```python
# Define your own evaluation grid
y_grid = np.linspace(y_min, y_max, 2000)

# Use for all predictions
y_pred = model.predict(X_test, y_grid=y_grid)
quantiles = model.predict_quantiles(X_test, y_grid=y_grid)
samples = model.sample_y(X_test, y_grid=y_grid)
```

## Comparison with Other Methods

| Method | Distributions | Uncertainty | Parametric | Speed |
|--------|--------------|-------------|------------|-------|
| Linear Regression | ✗ | ✗ | Gaussian | ⚡⚡⚡ |
| Random Forest | ✗ | Limited | ✗ | ⚡⚡ |
| Gradient Boosting | ✗ | ✗ | ✗ | ⚡⚡⚡ |
| Quantile Regression | Limited | ✓ | ✗ | ⚡⚡ |
| NGBoost | ✓ | ✓ | ✓ (Gaussian) | ⚡⚡ |
| **DistributionRegressor** | **✓** | **✓** | **✗ (Non-parametric)** | **⚡⚡** |

### Advantages

✅ **No distributional assumptions** - learns any distribution shape  
✅ **Flexible uncertainty** - captures heteroscedastic noise naturally  
✅ **One model, many predictions** - mean, quantiles, intervals, samples all from one model  
✅ **Efficient** - leverages fast LightGBM  
✅ **Categorical support** - native handling without encoding  

### Limitations

⚠️ **Grid-based** - requires evaluating on a grid (slower for many predictions)  
⚠️ **Memory** - storing distributions can be memory-intensive  
⚠️ **Tuning** - requires careful selection of negative sampling strategy  

## Best Practices

### 1. Choose Training Mode

**Soft negatives** (recommended):
- Smooth, continuous distributions
- Better training signal
- Slightly slower

**Hard negatives**:
- Sharp, concentrated distributions  
- Faster training
- Simpler objective

### 2. Tune Negative Sampling

- **k_neg**: 100-200 for better distributions, 50-100 for speed
- **neg_std**: Smaller (1.0-2.0) for local, larger (3.0+) for global
- **neg_sampler**: "stratified" usually works best

### 3. Grid Resolution

- Default 1000 points usually sufficient
- Increase for highly peaked distributions
- Decrease for faster inference

### 4. Always Use Validation

```python
model.fit(X_train, y_train, X_val, y_val)  # For early stopping
```

## Troubleshooting

### Poor Calibration (intervals don't match nominal coverage)

**Solution:**
- Increase `k_neg` (more negatives)
- Try different `neg_sampler` strategies
- Adjust `soft_label_std` (for soft mode)

### Slow Training

**Solution:**
- Reduce `k_neg`
- Fewer `n_estimators`
- Lower `subsample` and `colsample`

### Overconfident Predictions

**Solution:**
- Increase `soft_label_std` (soft mode)
- Use "stratified" or "uniform" sampling
- Add regularization: `lgbm_params={'reg_alpha': 0.1, 'reg_lambda': 0.1}`

### Memory Issues

**Solution:**
- Reduce grid size in predictions
- Process in batches
- Use smaller `k_neg` during training

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

## Changelog

### Version 1.0.0 (2025)
- Initial release
- Hard and soft training modes
- Comprehensive prediction API (mean, quantiles, intervals, CDF/PDF, sampling)
- Negative log-likelihood evaluation metric
- Scikit-learn compatible scoring function
- Categorical feature support
- Multiple negative sampling strategies
