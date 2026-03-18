"""
Generate all figures for the DistributionRegressor documentation site.

Run from the project root:
    python docs/generate_figures.py

Produces PNGs in docs/img/:
  - rossmann_timeseries.png   (zoomed to last 5 weeks of test)
  - rossmann_densities.png    (open days only - promo vs normal)
  - california_ngboost.png    (DistReg vs NGBoost Normal on California Housing)
  - baseline_comparison.png   (calibration + RMSE)
  - baseline_multimodal.png   (direct vs baseline on California Housing)
  - mc_comparison.png         (full grid vs MC)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from distribution_regressor import DistributionRegressor

IMG_DIR = os.path.join(os.path.dirname(__file__), 'img')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

BLUE = '#2563eb'
RED = '#ef4444'
GREEN = '#22c55e'
ORANGE = '#f59e0b'


# ===================================================================
# HELPERS
# ===================================================================
def load_rossmann():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'data', 'train.csv')
    df = pd.read_csv(data_path, dayfirst=True)
    df = df[df['Store'] == 1].copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Dayofweek'] = df['Date'].dt.dayofweek
    df['Dayofyear'] = df['Date'].dt.dayofyear
    df['Weekofyear'] = df['Date'].dt.isocalendar().week
    df['Closed'] = df['Dayofweek'] == 6

    features = ['Store', 'Dayofweek', 'Dayofyear', 'Weekofyear', 'Year', 'Month',
                'Day', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Closed']
    target = 'Sales'

    df['train_mask'] = df['Date'] < pd.to_datetime('2015-06-01')
    train = df[df['train_mask']]
    test = df[~df['train_mask']]
    return df, train, test, features, target


def load_california():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = data.data, data.target
    # Target is median house value in $100k (sklearn default)
    # Keep original scale for comparable NLL values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_density(ax, grids, dists, idx=0):
    """Helper to plot a single density from grid/dist arrays."""
    g = grids[idx]
    d = dists[idx]
    bw = g[1] - g[0]
    density = d / bw
    return g, density


# ===================================================================
# 1. ROSSMANN TIME SERIES (zoomed to last 5 weeks of test)
# ===================================================================
def fig_rossmann_timeseries(model, df, test, features, target):
    print("Generating rossmann_timeseries.png ...")

    # Zoom to last 5 weeks of test data
    test_sorted = test.sort_values('Date')
    end_date = test_sorted['Date'].max()
    start_date = end_date - pd.Timedelta(weeks=5)
    zoom = test_sorted[test_sorted['Date'] >= start_date]

    p10 = model.predict_quantile(zoom[features], 0.1)
    p25 = model.predict_quantile(zoom[features], 0.25)
    p50 = model.predict_quantile(zoom[features], 0.5)
    p75 = model.predict_quantile(zoom[features], 0.75)
    p90 = model.predict_quantile(zoom[features], 0.9)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    dates = zoom['Date'].values

    ax.fill_between(dates, p10, p90, alpha=0.12, color=RED, label='10th-90th pctile')
    ax.fill_between(dates, p25, p75, alpha=0.25, color=RED, label='25th-75th pctile')
    ax.plot(dates, p50, color=RED, linewidth=2, label='Median', alpha=0.85)
    ax.plot(dates, zoom[target].values, color=BLUE, linewidth=2.5, label='Actual Sales', alpha=0.9)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Rossmann Store #1 - Predicted Sales Distribution (Test Period)')
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'rossmann_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===================================================================
# 2. ROSSMANN DENSITIES (includes closed day + promo + normal)
# ===================================================================
def fig_rossmann_densities(model, test, features, target):
    print("Generating rossmann_densities.png ...")

    # Mix: 1 closed (Sunday), 2 promo, 3 normal
    closed_test = test[test['Closed'] == True]
    open_test = test[test['Closed'] == False].copy()
    promos = open_test[open_test['Promo'] == 1]
    normals = open_test[open_test['Promo'] == 0]

    sample_indices = []
    if len(closed_test) > 0:
        sample_indices.append(closed_test.index[0])
    if len(promos) >= 2:
        sample_indices.extend(promos.index[:2].tolist())
    if len(normals) >= 3:
        sample_indices.extend(normals.index[:3].tolist())
    sample_indices = sample_indices[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    model.set_output_smoothing(output_smoothing=1, atom_values=0)

    for ax, idx in zip(axes.flat, sample_indices):
        row = test.loc[[idx]]
        grids, dists, offsets = model.predict_distribution(row[features])
        true_y = row[target].values[0]
        date_str = row['Date'].dt.strftime('%Y-%m-%d').values[0]
        is_promo = row['Promo'].values[0]
        is_closed = row['Closed'].values[0]

        g, density = plot_density(ax, grids, dists)

        ax.fill_between(g, density, alpha=0.3, color=BLUE)
        ax.plot(g, density, color=BLUE, linewidth=1.5)
        ax.axvline(true_y, color=RED, linestyle='--', linewidth=1.5, alpha=0.8)

        if is_closed:
            label = date_str + ' (Closed)'
        elif is_promo:
            label = date_str + ' (Promo)'
        else:
            label = date_str
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Sales')
        ax.set_ylabel('Density')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

    model.set_output_smoothing(output_smoothing=0)

    plt.suptitle('Predicted p(Sales | x) for Selected Test Days', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'rossmann_densities.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===================================================================
# 3. California Housing: DistReg vs NGBoost Normal
# ===================================================================
def fig_california_ngboost():
    print("Generating california_ngboost.png ...")
    X_train, X_test, y_train, y_test = load_california()

    # Fit DistributionRegressor
    dr = DistributionRegressor(
        n_bins=200, n_estimators=1000, learning_rate=0.05,
        subsample=0.8, random_state=42, use_base_model=False,
    )
    dr.fit(X_train, y_train)

    # Fit NGBoost Normal
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from scipy.stats import norm

    ngb = NGBRegressor(
        Dist=Normal, n_estimators=1000, learning_rate=0.05,
        minibatch_frac=0.8, random_state=42, verbose=False,
    )
    ngb.fit(X_train, y_train)
    ngb_dists = ngb.pred_dist(X_test)

    np.random.seed(42)
    sample_idx = np.random.choice(len(y_test), 6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    dr.set_output_smoothing(output_smoothing=1)

    for ax, idx in zip(axes.flat, sample_idx):
        X_i = X_test[idx:idx+1]
        true_y = y_test[idx]

        # DistReg density
        grids, dists, _ = dr.predict_distribution(X_i)
        g, density = plot_density(ax, grids, dists)
        ax.fill_between(g, density, alpha=0.25, color=BLUE)
        ax.plot(g, density, color=BLUE, linewidth=1.8, label='DistReg')

        # NGBoost Normal density
        ngb_loc = ngb_dists.params['loc'][idx]
        ngb_scale = ngb_dists.params['scale'][idx]
        y_plot = np.linspace(g.min(), g.max(), 500)
        ngb_pdf = norm.pdf(y_plot, loc=ngb_loc, scale=ngb_scale)
        ax.plot(y_plot, ngb_pdf, color=GREEN, linewidth=1.8, linestyle='--', label='NGBoost (Normal)')

        ax.axvline(true_y, color=RED, linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'true = {true_y:.2f}')
        ax.set_title(f'Test sample {idx}', fontsize=10)
        ax.set_xlabel('House Value ($100k)')
        ax.set_ylabel('Density')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

    dr.set_output_smoothing(output_smoothing=0)

    axes.flat[0].legend(fontsize=8)
    plt.suptitle('DistributionRegressor vs NGBoost (Normal) on California Housing', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'california_ngboost.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===================================================================
# 4. BASELINE COMPARISON (Calibration + RMSE) - Rossmann
# ===================================================================
def fig_baseline_comparison():
    print("Generating baseline_comparison.png ...")
    _, train, test, features, target = load_rossmann()

    common = dict(n_bins=100, n_estimators=1000, learning_rate=0.01,
                  random_state=42)

    model_direct = DistributionRegressor(use_base_model=False, **common)
    model_direct.fit(train[features], train[target])

    model_baseline = DistributionRegressor(use_base_model=True, **common)
    model_baseline.fit(train[features], train[target])

    y_true = test[target].values
    X_test = test[features]

    levels = np.arange(0.1, 1.0, 0.1)
    cov_direct = []
    cov_baseline = []

    for lvl in levels:
        iv = model_direct.predict_interval(X_test, confidence=lvl)
        cov_direct.append(((y_true >= iv[:, 0]) & (y_true <= iv[:, 1])).mean())
        iv = model_baseline.predict_interval(X_test, confidence=lvl)
        cov_baseline.append(((y_true >= iv[:, 0]) & (y_true <= iv[:, 1])).mean())

    from sklearn.metrics import mean_squared_error
    rmse_direct = np.sqrt(mean_squared_error(y_true, model_direct.predict(X_test)))
    rmse_baseline = np.sqrt(mean_squared_error(y_true, model_baseline.predict(X_test)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
    ax1.plot(levels, cov_direct, 'o-', color=BLUE, linewidth=2, markersize=6,
             label='Direct (no baseline)')
    ax1.plot(levels, cov_baseline, 's-', color=ORANGE, linewidth=2, markersize=6,
             label='With baseline')
    ax1.set_xlabel('Nominal Coverage')
    ax1.set_ylabel('Empirical Coverage')
    ax1.set_title('Calibration Plot')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')

    bars = ax2.bar(['Direct\n(no baseline)', 'With\nbaseline'], [rmse_direct, rmse_baseline],
                   color=[BLUE, ORANGE], width=0.5, alpha=0.85)
    for bar, val in zip(bars, [rmse_direct, rmse_baseline]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Point Prediction Accuracy')
    ax2.set_ylim(0, max(rmse_direct, rmse_baseline) * 1.15)

    plt.suptitle('Baseline vs Direct Mode - Rossmann Test Set', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'baseline_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===================================================================
# 5. BASELINE COMPARISON on California Housing - distribution shape
# ===================================================================
def fig_baseline_multimodal():
    print("Generating baseline_multimodal.png ...")
    X_train, X_test, y_train, y_test = load_california()

    common = dict(n_bins=200, n_estimators=1000, learning_rate=0.05,
                  subsample=0.8, random_state=42)

    model_direct = DistributionRegressor(use_base_model=False, **common)
    model_direct.fit(X_train, y_train)

    model_baseline = DistributionRegressor(use_base_model=True, **common)
    model_baseline.fit(X_train, y_train)

    np.random.seed(42)
    sample_idx = np.random.choice(len(y_test), 6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    model_direct.set_output_smoothing(output_smoothing=1)
    model_baseline.set_output_smoothing(output_smoothing=1)

    for ax, idx in zip(axes.flat, sample_idx):
        X_i = X_test[idx:idx+1]
        true_y = y_test[idx]

        # Direct model
        grids_d, dists_d, _ = model_direct.predict_distribution(X_i)
        g_d, dens_d = plot_density(ax, grids_d, dists_d)
        ax.plot(g_d, dens_d, color=BLUE, linewidth=1.8, label='Direct')
        ax.fill_between(g_d, dens_d, alpha=0.2, color=BLUE)

        # Baseline model
        grids_b, dists_b, _ = model_baseline.predict_distribution(X_i)
        g_b, dens_b = plot_density(ax, grids_b, dists_b)
        ax.plot(g_b, dens_b, color=ORANGE, linewidth=1.8, linestyle='--', label='Baseline')

        ax.axvline(true_y, color=RED, linestyle=':', linewidth=1.2, alpha=0.7)
        ax.set_title(f'Test sample {idx}', fontsize=10)
        ax.set_xlabel('House Value ($100k)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

    model_direct.set_output_smoothing(output_smoothing=0)
    model_baseline.set_output_smoothing(output_smoothing=0)

    axes.flat[0].legend(fontsize=9)
    plt.suptitle('Direct vs Baseline Mode - California Housing', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'baseline_multimodal.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===================================================================
# 6. MONTE CARLO vs FULL GRID
# ===================================================================
def fig_mc_comparison():
    print("Generating mc_comparison.png ...")
    _, train, test, features, target = load_rossmann()

    base_params = dict(n_bins=100, n_estimators=1000, learning_rate=0.01,
                       random_state=42, use_base_model=False)

    configs = [
        ('Full grid', dict(monte_carlo_training=False)),
        ('MC freq=100', dict(monte_carlo_training=True, mc_samples=20, mc_resample_freq=100)),
        ('MC freq=10', dict(monte_carlo_training=True, mc_samples=20, mc_resample_freq=10)),
    ]
    colors = [BLUE, ORANGE, GREEN]

    models = {}
    for label, kw in configs:
        m = DistributionRegressor(**{**base_params, **kw})
        m.fit(train[features], train[target])
        m.set_output_smoothing(output_smoothing=1, atom_values=0)
        models[label] = m

    np.random.seed(123)
    open_test = test[test['Closed'] == False]
    sample_indices = open_test.index[np.random.choice(len(open_test), 6, replace=False)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, idx in zip(axes.flat, sample_indices):
        row = test.loc[[idx]]
        true_y = row[target].values[0]
        date_str = row['Date'].dt.strftime('%Y-%m-%d').values[0]

        for (label, m), color in zip(models.items(), colors):
            grids, dists, _ = m.predict_distribution(row[features])
            g, dens = plot_density(ax, grids, dists)
            ax.plot(g, dens, color=color, linewidth=1.2, label=label, alpha=0.85)

        ax.axvline(true_y, color=RED, linestyle=':', linewidth=1.2, alpha=0.6)
        ax.set_title(date_str, fontsize=10)
        ax.set_xlabel('Sales')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

    for m in models.values():
        m.set_output_smoothing(output_smoothing=0)

    axes.flat[0].legend(fontsize=9)
    plt.suptitle('Full Grid vs MC Resampling (K=20)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'mc_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===================================================================
# MAIN
# ===================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating documentation figures")
    print("=" * 60)

    # Train Rossmann model once, reuse for timeseries + densities
    _, train, test, features, target = load_rossmann()
    df, _, _, _, _ = load_rossmann()

    print("Training Rossmann model (default settings) ...")
    rossmann_model = DistributionRegressor(
        n_bins=100, n_estimators=1000, learning_rate=0.01,
        random_state=42,
    )
    rossmann_model.fit(train[features], train[target])

    fig_rossmann_timeseries(rossmann_model, df, test, features, target)
    fig_rossmann_densities(rossmann_model, test, features, target)
    fig_baseline_comparison()
    fig_mc_comparison()
    fig_california_ngboost()
    fig_baseline_multimodal()

    print("=" * 60)
    print(f"All figures saved to {IMG_DIR}")
    print("=" * 60)
