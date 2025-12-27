"""
Distribution Visualization Example

Visualizes the full predicted probability distributions for individual predictions,
showing how DistributionRegressor goes beyond point estimates.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    n_bins=100,
    n_estimators=2000,
    learning_rate=0.05,
    sigma='auto',
    output_smoothing=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# Select a few test points to visualize
n_examples = 6
example_idx = np.random.choice(len(X_test), n_examples, replace=False)
X_examples = X_test[example_idx]
y_examples = y_test[example_idx]

# Get predicted distributions
y_grid, distributions = model.predict_distribution(X_examples)

# Get point predictions and intervals
y_pred = model.predict(X_examples)
lower = model.predict_quantile(X_examples, 0.05)
upper = model.predict_quantile(X_examples, 0.95)

# Visualize distributions with plotly
fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Example {i+1}' for i in range(n_examples)])

for i in range(n_examples):
    row = i // 3 + 1
    col = i % 3 + 1
    
    x_vals = y_grid.tolist()
    y_vals = distributions[i].tolist()
    
    # Distribution fill
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, fill='tozeroy', fillcolor='rgba(0,100,255,0.3)',
        line=dict(color='blue', width=2), name='Distribution', showlegend=(i==0)
    ), row=row, col=col)
    
    # True value
    fig.add_vline(x=float(y_examples[i]), line=dict(color='green', dash='dash', width=2), row=row, col=col)
    # Mean prediction
    fig.add_vline(x=float(y_pred[i]), line=dict(color='red', width=2), row=row, col=col)
    # 90% interval
    fig.add_vline(x=float(lower[i]), line=dict(color='orange', dash='dot', width=1.5), row=row, col=col)
    fig.add_vline(x=float(upper[i]), line=dict(color='orange', dash='dot', width=1.5), row=row, col=col)

fig.update_layout(
    title='Predicted Probability Distributions for Individual Test Points',
    height=600, width=1000, showlegend=True
)
fig.update_xaxes(title_text='y value')
fig.update_yaxes(title_text='Probability density')
fig.show()

