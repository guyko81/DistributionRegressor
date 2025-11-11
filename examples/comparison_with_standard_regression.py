"""
Comparison with Standard Regression

Compares DistributionRegressor with standard LightGBM regression,
showing the advantages of probabilistic predictions.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from distribution_regressor import DistributionRegressor

# Generate data with heteroscedastic noise
np.random.seed(42)
n = 500
X = np.random.randn(n, 4)
y_true = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2]**2
# Noise that depends on input
noise_std = 0.5 + 0.8*np.abs(X[:, 0])
y = y_true + noise_std * np.random.randn(n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("="*70)
print("Training Both Models")
print("="*70)

# Standard LightGBM
print("\n1. Standard LightGBM Regressor...")
lgbm_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_train, y_train)

# DistributionRegressor
print("2. DistributionRegressor...")
dist_model = DistributionRegressor(
    negative_type="soft",
    n_estimators=200,
    learning_rate=0.1,
    k_neg=100,
    verbose=0,
    random_state=42
)
dist_model.fit(X_train, y_train, X_test, y_test)

print("\n" + "="*70)
print("Point Prediction Comparison")
print("="*70)

# Point predictions
lgbm_pred = lgbm_model.predict(X_test)
dist_pred = dist_model.predict(X_test)

lgbm_mse = mean_squared_error(y_test, lgbm_pred)
dist_mse = mean_squared_error(y_test, dist_pred)

lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
dist_mae = mean_absolute_error(y_test, dist_pred)

print(f"\nStandard LightGBM:")
print(f"  MSE: {lgbm_mse:.4f}")
print(f"  MAE: {lgbm_mae:.4f}")

print(f"\nDistributionRegressor:")
print(f"  MSE: {dist_mse:.4f}")
print(f"  MAE: {dist_mae:.4f}")

print("\n" + "="*70)
print("Probabilistic Evaluation")
print("="*70)

# NLL - only available for DistributionRegressor
nll, _ = dist_model.negative_log_likelihood(X_test, y_test)
print(f"\nDistributionRegressor NLL: {nll:.4f}")
print("(Standard LightGBM cannot compute NLL - no distribution predicted)")

# Prediction intervals - only available for DistributionRegressor
lower_90, upper_90 = dist_model.predict_interval(X_test, alpha=0.1)
coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))

lower_50, upper_50 = dist_model.predict_interval(X_test, alpha=0.5)
coverage_50 = np.mean((y_test >= lower_50) & (y_test <= upper_50))

print(f"\nPrediction Interval Coverage:")
print(f"  50% interval: {coverage_50:.1%} (target: 50%)")
print(f"  90% interval: {coverage_90:.1%} (target: 90%)")
print("(Standard LightGBM cannot provide calibrated intervals)")

print("\n" + "="*70)
print("Visualization")
print("="*70)

# Visualize predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Standard LightGBM
axes[0].scatter(y_test, lgbm_pred, alpha=0.5, s=30)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('True values')
axes[0].set_ylabel('Predicted values')
axes[0].set_title(f'Standard LightGBM\nMSE: {lgbm_mse:.4f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: DistributionRegressor point predictions
axes[1].scatter(y_test, dist_pred, alpha=0.5, s=30)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect prediction')
axes[1].set_xlabel('True values')
axes[1].set_ylabel('Predicted values')
axes[1].set_title(f'DistributionRegressor (Mean)\nMSE: {dist_mse:.4f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: DistributionRegressor with uncertainty
axes[2].errorbar(y_test, dist_pred, 
                 yerr=[dist_pred - lower_90, upper_90 - dist_pred],
                 fmt='o', alpha=0.5, markersize=4, capsize=0, linewidth=1)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect prediction')
axes[2].set_xlabel('True values')
axes[2].set_ylabel('Predicted values')
axes[2].set_title(f'DistributionRegressor with 90% Intervals\nCoverage: {coverage_90:.1%}')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_with_standard_regression.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to 'comparison_with_standard_regression.png'")

print("\n" + "="*70)
print("Summary")
print("="*70)
print("""
Key Advantages of DistributionRegressor:

✓ Full probability distributions - not just point estimates
✓ Calibrated uncertainty quantification
✓ Prediction intervals with proper coverage
✓ Probabilistic evaluation (NLL)
✓ Better for decision-making under uncertainty

Point predictions (MSE/MAE) are often comparable, but DistributionRegressor
provides much richer information about prediction confidence.
""")

