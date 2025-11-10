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
print("Training DistributionRegressor")
print("=" * 60)

# Create and train model
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

print("\n" + "=" * 60)
print("Making Predictions")
print("=" * 60)

# Point predictions
y_pred = model.predict(X_test)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")

# Prediction intervals
lower, upper = model.predict_interval(X_test, alpha=0.1)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"90% interval coverage: {coverage:.1%}")

# Quantiles
quantiles = model.predict_quantiles(X_test, qs=[0.1, 0.5, 0.9])
print(f"\nQuantiles shape: {quantiles.shape}")
print(f"First 5 medians: {quantiles[:5, 1]}")

# Samples
samples = model.sample_y(X_test[:5], n_samples=1000, random_state=42)
print(f"\nSample statistics (first 5 test points):")
for i in range(5):
    print(f"  Point {i}: mean={samples[i].mean():.3f}, "
          f"std={samples[i].std():.3f}, true={y_test[i]:.3f}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)

