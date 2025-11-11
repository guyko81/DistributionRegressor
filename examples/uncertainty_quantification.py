"""
Uncertainty Quantification Example

Demonstrates how DistributionRegressor captures uncertainty in predictions,
especially useful for heteroscedastic data where noise varies across input space.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from distribution_regressor import DistributionRegressor

# Generate data with heteroscedastic noise (varying uncertainty)
np.random.seed(42)
n = 500
X = np.linspace(-3, 3, n).reshape(-1, 1)

# True function with input-dependent noise
y_true = np.sin(X.ravel()) * 2
noise_std = 0.2 + 0.5 * np.abs(X.ravel())  # Noise increases away from zero
y = y_true + noise_std * np.random.randn(n)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training model to capture uncertainty...")
model = DistributionRegressor(
    negative_type="soft",
    n_estimators=200,
    learning_rate=0.1,
    k_neg=100,
    neg_sampler="stratified",
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train, X_test, y_test)

# Make predictions
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict(X_plot)

# Get prediction intervals
lower_50, upper_50 = model.predict_interval(X_plot, alpha=0.5)  # 50% interval
lower_90, upper_90 = model.predict_interval(X_plot, alpha=0.1)  # 90% interval
lower_99, upper_99 = model.predict_interval(X_plot, alpha=0.01) # 99% interval

# Get quantiles
quantiles = model.predict_quantiles(X_plot, qs=[0.05, 0.25, 0.5, 0.75, 0.95])

print("\nPrediction Statistics:")
print(f"Mean prediction: {y_pred.mean():.3f}")
print(f"Median 90% interval width: {np.median(upper_90 - lower_90):.3f}")

# Visualize uncertainty
plt.figure(figsize=(14, 5))

# Plot 1: Prediction intervals
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.3, s=10, label='Training data')
plt.plot(X_plot, y_pred, 'r-', linewidth=2, label='Mean prediction')
plt.fill_between(X_plot.ravel(), lower_99, upper_99, alpha=0.2, color='blue', label='99% interval')
plt.fill_between(X_plot.ravel(), lower_90, upper_90, alpha=0.3, color='blue', label='90% interval')
plt.fill_between(X_plot.ravel(), lower_50, upper_50, alpha=0.4, color='blue', label='50% interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Prediction Intervals\n(Notice wider intervals where noise is higher)')
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Quantile predictions
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, alpha=0.3, s=10, label='Training data')
plt.plot(X_plot, quantiles[:, 2], 'r-', linewidth=2, label='Median (50th)')
plt.plot(X_plot, quantiles[:, 0], 'b--', linewidth=1.5, label='5th percentile')
plt.plot(X_plot, quantiles[:, 4], 'b--', linewidth=1.5, label='95th percentile')
plt.plot(X_plot, quantiles[:, 1], 'g:', linewidth=1.5, label='25th percentile')
plt.plot(X_plot, quantiles[:, 3], 'g:', linewidth=1.5, label='75th percentile')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Quantile Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('uncertainty_quantification.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to 'uncertainty_quantification.png'")

# Analyze uncertainty at different points
test_points = np.array([[-2.5], [0.0], [2.5]])
for point in test_points:
    lower, upper = model.predict_interval(point.reshape(1, -1), alpha=0.1)
    width = upper[0] - lower[0]
    print(f"\nAt X = {point[0]:5.2f}: 90% interval width = {width:.3f}")

print("\nKey insight: The model automatically captures that predictions")
print("are more uncertain in regions with higher noise variance!")

