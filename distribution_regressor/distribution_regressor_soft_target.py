"""
Soft-Target Distribution Regressor (CatBoost Custom Loss)

A scikit-learn-compatible regressor that predicts full probability distributions.
It uses a CatBoost model with a custom multi-output loss function that spreads
soft targets across neighboring bins via Gaussian kernel.

This approach allows for:
1. Arbitrary distribution shapes (multimodal, skewed, etc.)
2. "Dragging" behavior via the sigma parameter (smoothing neighbors)
3. Efficient training without dataset expansion (native multi-output)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.ndimage import gaussian_filter1d


class SoftTargetCrossEntropyLoss:
    """
    Custom CatBoost objective for soft-target cross-entropy.
    
    Each sample's target y is converted to a soft distribution over n_bins
    using a Gaussian kernel centered at y. The loss is cross-entropy between
    the predicted distribution (after softmax) and the soft target.
    """
    
    def __init__(self, grid, sigma):
        """
        Parameters
        ----------
        grid : array of shape (n_bins,)
            Grid points for the distribution.
        sigma : float or array of shape (n_samples,)
            Standard deviation for Gaussian kernel.
        """
        self.grid = np.asarray(grid)
        self.n_bins = len(grid)
        self.sigma = sigma
        self.y_values_ = None  # Will store original y values during fit
        self.soft_targets_ = None  # Precomputed soft targets
        
    def set_targets(self, y):
        """Store original y values and precompute soft targets."""
        self.y_values_ = np.asarray(y)
        self.soft_targets_ = self._compute_soft_targets(self.y_values_)
        
    def _compute_soft_targets(self, y):
        """Compute soft target distribution for each sample."""
        y = np.asarray(y)
        
        # Broadcast: (n_samples, n_bins)
        diff_sq = (y[:, None] - self.grid[None, :]) ** 2
        
        # Handle sigma broadcasting
        if np.ndim(self.sigma) == 0:
            sigma_sq = self.sigma ** 2
        else:
            sigma_sq = (self.sigma ** 2)[:, None]
        
        targets = np.exp(-diff_sq / (2 * sigma_sq))
        # Normalize to sum to 1
        targets = targets / (targets.sum(axis=1, keepdims=True) + 1e-10)
        return targets
    
    def _softmax(self, approxes):
        """Apply softmax across dimensions for each sample."""
        # approxes: list of n_bins arrays, each of length n_samples
        # Convert to (n_samples, n_bins)
        raw = np.column_stack(approxes)
        # Numerical stability
        raw = raw - raw.max(axis=1, keepdims=True)
        exp_raw = np.exp(raw)
        probs = exp_raw / (exp_raw.sum(axis=1, keepdims=True) + 1e-10)
        return probs
    
    def calc_ders_multi(self, approxes, target, weight):
        """
        Calculate gradients and hessians for multi-class objective.
        
        Parameters
        ----------
        approxes : list of length n_bins
            Each element is array of predictions for that dimension.
        target : array of shape (n_samples,)
            Class indices (bin indices) - we use precomputed soft targets instead.
        weight : array or None
            Sample weights.
            
        Returns
        -------
        list of tuples [(grad_0, hess_0), (grad_1, hess_1), ...]
            Gradients and hessians for each class/bin.
        """
        n_samples = len(target)
        
        # Compute softmax probabilities: (n_samples, n_bins)
        probs = self._softmax(approxes)
        
        # Use precomputed soft targets (indexed by sample order)
        soft_targets = self.soft_targets_[:n_samples]
        
        # Cross-entropy gradient: p - t (for softmax + cross-entropy)
        # Hessian diagonal approximation: p * (1 - p)
        grads = probs - soft_targets
        hess = probs * (1 - probs) + 1e-6  # Small epsilon for stability
        
        # Apply weights if provided
        if weight is not None:
            weight = np.asarray(weight)[:, None]
            grads = grads * weight
            hess = hess * weight
        
        # Return as list of (grad, hess) tuples per class
        result = []
        for dim in range(self.n_bins):
            result.append((grads[:, dim], hess[:, dim]))
        
        return result


class DistributionRegressorSoftTarget(BaseEstimator, RegressorMixin):
    """
    Predicts probability distributions using CatBoost with custom soft-target loss.
    
    Parameters
    ----------
    n_bins : int, default=50
        Number of grid points to discretize the target variable range.
    
    sigma : float or str, default=1.0
        The standard deviation of the Gaussian kernel used to generate soft targets.
        - If float: Constant spread around the true y.
        - If 'auto': Data-driven estimation based on residual standard deviation.
    
    output_smoothing : float, default=1.0
        Standard deviation for Gaussian smoothing of the output distribution.
    
    n_estimators : int, default=100
        Number of boosting trees.
    
    learning_rate : float, default=0.1
        Boosting learning rate.
        
    random_state : int or None, default=None
        Random seed.
        
    **kwargs : dict
        Additional parameters passed to CatBoostRegressor.
    """
    
    def __init__(
        self,
        n_bins=50,
        sigma='auto',
        output_smoothing=1.0,
        n_estimators=100,
        learning_rate=0.1,
        random_state=None,
        **kwargs
    ):
        self.n_bins = n_bins
        self.sigma = sigma
        self.output_smoothing = output_smoothing
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.catboost_kwargs = kwargs
        
    def _validate_params(self):
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model using CatBoost with custom soft-target loss.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.
        """
        self._validate_params()
        
        # 1. Prepare Data
        self._is_dataframe = isinstance(X, pd.DataFrame)
        if self._is_dataframe:
            self.feature_names_in_ = X.columns.tolist()
            X_array = X.values
            y_array = np.asarray(y)
        else:
            X_array = X
            y_array = y
            
        X_array, y_array = check_X_y(X_array, y_array, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X_array.shape[1]
        n_samples = X_array.shape[0]
        
        # Validate sample_weight
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != n_samples:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} samples, expected {n_samples}")

        # 2. Define the Grid
        y_min = float(np.min(y_array))
        y_max = float(np.max(y_array))
        self.grid_ = np.linspace(y_min, y_max, self.n_bins)
        grid_step = (self.grid_[-1] - self.grid_[0]) / (self.n_bins - 1)
        
        # 3. Resolve Sigma
        if self.sigma == 'auto':
            # Fit a quick baseline model to estimate residual variance
            baseline = CatBoostRegressor(
                iterations=min(self.n_estimators, 100),
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbose=0
            )
            baseline.fit(X_array, y_array, sample_weight=sample_weight)
            y_pred = baseline.predict(X_array)
            residuals = y_array - y_pred
            std_resid = np.std(residuals)
            
            # Sigma is at least half a grid step
            self.sigma_val_ = max(std_resid, grid_step * 0.5)
        else:
            self.sigma_val_ = float(self.sigma)
        
        # 4. Create custom loss with grid and sigma
        self.loss_ = SoftTargetCrossEntropyLoss(self.grid_, self.sigma_val_)
        
        # 5. Store y values in loss for soft target computation
        self.loss_.set_targets(y_array)
        
        # 6. Convert y to bin indices (closest bin) for CatBoost class labels
        bin_indices = np.argmin(np.abs(y_array[:, None] - self.grid_[None, :]), axis=1)
        
        # 7. Configure CatBoost Classifier with custom multi-class loss
        params = {
            'iterations': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'verbose': 0,
            'loss_function': self.loss_,
            'classes_count': self.n_bins,
        }
        params.update(self.catboost_kwargs)
        
        self.model_ = CatBoostClassifier(**params)
        
        # Create Pool with class labels (bin indices)
        train_pool = Pool(
            data=X_array,
            label=bin_indices,
            weight=sample_weight
        )
        
        self.model_.fit(train_pool)

        return self

    def predict_distribution(self, X):
        """
        Returns grid points and probability distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to predict distributions.
        
        Returns
        -------
        grid : array of shape (n_bins,)
            Grid points over the target variable range.
        
        distributions : array of shape (n_samples, n_bins)
            Probability distribution for each sample at each grid point.
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        X_array = check_array(X_array, accept_sparse=False)
        
        # Predict raw scores: (n_samples, n_bins)
        # Use prediction_type='RawFormulaVal' to get raw logits before softmax
        raw_scores = self.model_.predict(X_array, prediction_type='RawFormulaVal')
        
        # Ensure 2D
        if raw_scores.ndim == 1:
            raw_scores = raw_scores.reshape(-1, self.n_bins)
        
        # Apply softmax to get probabilities
        raw_scores = raw_scores - raw_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(raw_scores)
        distributions = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)
        
        # Apply output smoothing if enabled
        if self.output_smoothing > 0:
            distributions = gaussian_filter1d(distributions, sigma=self.output_smoothing, axis=1)
            # Renormalize
            distributions = distributions / (distributions.sum(axis=1, keepdims=True) + 1e-10)
        
        return self.grid_, distributions

    def predict(self, X):
        """Default: Predict Mean"""
        return self.predict_mean(X)

    def predict_mean(self, X):
        """
        Predict the mean of the distribution for each sample.
        """
        grid, dists = self.predict_distribution(X)
        return np.sum(dists * grid, axis=1)

    def predict_mode(self, X):
        """
        Predict the mode (peak) of the distribution for each sample.
        """
        grid, dists = self.predict_distribution(X)
        max_indices = np.argmax(dists, axis=1)
        return grid[max_indices]

    def predict_quantile(self, X, q=0.5):
        """
        Predict quantile(s) of the distribution for each sample.
        """
        grid, dists = self.predict_distribution(X)
        cdfs = np.cumsum(dists, axis=1)
        
        q_arr = np.asarray(q)
        is_scalar = q_arr.ndim == 0
        if is_scalar:
            q_arr = q_arr.reshape(1)
            
        n_samples = len(dists)
        n_quantiles = len(q_arr)
        quantiles = np.zeros((n_samples, n_quantiles))
        
        for i in range(n_samples):
            indices = np.searchsorted(cdfs[i], q_arr)
            indices = np.clip(indices, 0, self.n_bins - 1)
            quantiles[i] = grid[indices]
            
        if is_scalar:
            return quantiles.flatten()
        return quantiles

    def predict_interval(self, X, confidence=0.95):
        """
        Predict prediction interval for each sample.
        """
        alpha = 1.0 - confidence
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0
        return self.predict_quantile(X, q=[q_low, q_high])

    def predict_std(self, X):
        """
        Predict standard deviation of the distribution for each sample.
        """
        grid, dists = self.predict_distribution(X)
        means = np.sum(dists * grid, axis=1, keepdims=True)
        variance = np.sum(dists * (grid - means)**2, axis=1)
        return np.sqrt(variance)
