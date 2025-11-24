"""
Basic usage example for DistributionRegressor.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from distribution_regressor import DistributionRegressor

# Generate synthetic data with heteroscedastic noise
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
y_true = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2]**2
noise = (0.5 + 0.3*np.abs(X[:, 0])) * np.random.randn(n)
y = y_true + noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Training DistributionRegressor (Soft Target)")
print("=" * 60)

# Create and train model
model = DistributionRegressor(
    n_bins=50,              # Resolution of the distribution
    n_estimators=200,       # Number of trees
    learning_rate=0.1,
    sigma='auto',           # Automatic noise estimation
    output_smoothing=1.0,   # Smooth the output distribution
    random_state=42
)

model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("Making Predictions")
print("=" * 60)

# Point predictions
y_pred = model.predict(X_test)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")

# Prediction intervals
intervals = model.predict_interval(X_test, confidence=0.90)
lower, upper = intervals[:, 0], intervals[:, 1]
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.1%}")

# Quantiles
quantiles = model.predict_quantile(X_test, q=[0.1, 0.5, 0.9])
print(f"\nQuantiles shape: {quantiles.shape}")
print(f"First 5 medians: {quantiles[:5, 1]}")

# Standard Deviation (Uncertainty)
stds = model.predict_std(X_test)
print(f"\nMean Predicted Std Dev: {stds.mean():.3f}")

# Full Distribution
print(f"\nPredicting full distribution for first sample...")
grid, dists = model.predict_distribution(X_test[:1])
print(f"Grid shape: {grid.shape}")
print(f"Distribution shape: {dists.shape}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
