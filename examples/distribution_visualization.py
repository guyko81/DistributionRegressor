"""
Distribution Visualization Example

Visualizes the full predicted probability distributions for individual predictions,
showing how DistributionRegressor goes beyond point estimates.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from distribution_regressor import DistributionRegressor

# Generate synthetic regression data
np.random.seed(42)
X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = DistributionRegressor(
    negative_type="soft",
    n_estimators=200,
    learning_rate=0.1,
    k_neg=100,
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train, X_test, y_test)

# Select a few test points to visualize
n_examples = 6
example_idx = np.random.choice(len(X_test), n_examples, replace=False)
X_examples = X_test[example_idx]
y_examples = y_test[example_idx]

# Get predicted distributions
y_grid = np.linspace(y_test.min() - 20, y_test.max() + 20, 200)
distributions = model.predict_distribution(X_examples, y_grid)

# Get point predictions and intervals
y_pred = model.predict(X_examples)
lower, upper = model.predict_interval(X_examples, alpha=0.1)

# Visualize distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(n_examples):
    ax = axes[i]
    
    # Plot the distribution
    ax.fill_between(y_grid, distributions[i], alpha=0.3, color='blue', label='Distribution')
    ax.plot(y_grid, distributions[i], 'b-', linewidth=2)
    
    # Mark key points
    ax.axvline(y_examples[i], color='green', linestyle='--', linewidth=2, label=f'True: {y_examples[i]:.1f}')
    ax.axvline(y_pred[i], color='red', linestyle='-', linewidth=2, label=f'Mean: {y_pred[i]:.1f}')
    ax.axvline(lower[i], color='orange', linestyle=':', linewidth=1.5, label='90% interval')
    ax.axvline(upper[i], color='orange', linestyle=':', linewidth=1.5)
    
    # Styling
    ax.set_xlabel('y value')
    ax.set_ylabel('Probability density')
    ax.set_title(f'Example {i+1}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Highlight the area between interval bounds
    mask = (y_grid >= lower[i]) & (y_grid <= upper[i])
    ax.fill_between(y_grid[mask], distributions[i][mask], alpha=0.2, color='orange')

plt.suptitle('Predicted Probability Distributions for Individual Test Points', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'distribution_visualization.png'")

# Sample from distributions
print("\n" + "="*60)
print("Sampling from Predicted Distributions")
print("="*60)

samples = model.sample_y(X_examples[:3], n_samples=1000, random_state=42)

for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"  True value:      {y_examples[i]:7.2f}")
    print(f"  Mean prediction: {y_pred[i]:7.2f}")
    print(f"  Sample mean:     {samples[i].mean():7.2f}")
    print(f"  Sample std:      {samples[i].std():7.2f}")
    print(f"  Sample range:    [{samples[i].min():7.2f}, {samples[i].max():7.2f}]")

print("\n" + "="*60)
print("Key Insight")
print("="*60)
print("Each prediction is a full probability distribution, not just a point!")
print("This enables:")
print("  • Uncertainty quantification")
print("  • Risk assessment")
print("  • Multi-modal predictions")
print("  • Confidence-based decision making")

