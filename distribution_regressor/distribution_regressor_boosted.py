"""
Distribution Regressor with Custom Boosting (Boosted Version)

A scikit-learn-compatible regressor that predicts full probability distributions
over the target variable range using custom gradient boosting with LGBMRegressors.

APPROACH: Instead of training K LGBMClassifiers (each with N trees), this version
trains N individual trees total using custom boosting in logit space. Each iteration:
1. Generates random bin boundaries over the y-range
2. Trains a single LGBMRegressor tree to predict logit-space residuals
3. Accumulates predictions on a fine grid (default 10,000 points)
4. Uses learning rate for gradual improvement
5. Supports averaging in probability or logit space when calculating baseline values

This reduces total tree count from K*N to N while maintaining distribution quality.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from lightgbm import LGBMRegressor
from scipy.special import logit, expit
import warnings


class DistributionRegressorBoosted(BaseEstimator, RegressorMixin):
    """
    Distribution Regressor using custom boosting with individual trees.
    
    Each boosting iteration trains a single tree on logit-space residuals with 
    random binning boundaries to build probability distributions over the target 
    variable range.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations (total trees).
    
    n_bins : int, default=10
        Number of bins per iteration for discretizing the target variable.
    
    grid_resolution : int, default=10000
        Number of grid points for the probability distribution over y-range.
    
    logit_eps : float, default=1e-5
        Epsilon value for logit calculations to avoid logit(0) and logit(1).
    
    baseline_mode : str, default='mean'
        How to calculate baseline from grid predictions:
        - 'mean': Simple average over grid points in bin range
        - 'weighted': Weighted average by grid point density
    
    averaging_space : str, default='probability'
        Space in which to perform averaging when calculating baseline values:
        - 'probability': Average in probability space (convert logits to probabilities,
          average, then convert back to logits)
        - 'logit': Average directly in logit space
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    learning_rate : float, default=0.1
        Boosting learning rate (shrinkage).
    
    max_depth : int, default=7
        Maximum tree depth for each LGBMRegressor.
    
    num_leaves : int, default=31
        Maximum number of leaves in one tree.
    
    subsample : float, default=0.8
        Fraction of samples for bagging.
    
    colsample_bytree : float, default=0.8
        Fraction of features for each tree.
    
    reg_alpha : float, default=0.0
        L1 regularization term.
    
    reg_lambda : float, default=0.0
        L2 regularization term.
    
    max_bin : int, default=255
        Max number of bins for LGBM feature bucketing.
    
    min_child_samples : int, default=1
        Minimum number of data needed in a child leaf.
    
    **kwargs : dict
        Additional LightGBM parameters.
    
    Attributes
    ----------
    estimators_ : list of dict
        List of dictionaries, one per iteration, containing:
        - 'tree': Trained LGBMRegressor (single tree)
        - 'bin_boundaries': Array of bin edges
        - 'n_bins': Number of bins for this iteration
    
    grid_ : array
        Grid of y values used for probability distribution.
    
    y_min_ : float
        Minimum value of target variable in training data.
    
    y_max_ : float
        Maximum value of target variable in training data.
    
    n_features_in_ : int
        Number of features in training data.
    """
    
    def __init__(
        self,
        n_estimators=100,
        n_bins=10,
        grid_resolution=10000,
        logit_eps=1e-5,
        baseline_mode='mean',
        averaging_space='probability',
        random_state=None,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_bin=255,
        min_child_samples=1,
        **kwargs
    ):
        # Model-specific parameters
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.grid_resolution = grid_resolution
        self.logit_eps = logit_eps
        self.baseline_mode = baseline_mode
        self.averaging_space = averaging_space
        self.random_state = random_state
        
        # LightGBM parameters
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.min_child_samples = min_child_samples
        
        # Store any additional kwargs
        self.lgbm_kwargs = kwargs
    
    def _validate_params(self):
        """Validate input parameters."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        
        if self.n_bins <= 1:
            raise ValueError("n_bins must be greater than 1")
        
        if self.grid_resolution <= 1:
            raise ValueError("grid_resolution must be greater than 1")
        
        if self.logit_eps <= 0 or self.logit_eps >= 0.5:
            raise ValueError("logit_eps must be in (0, 0.5)")
        
        if self.baseline_mode not in ['mean', 'weighted']:
            raise ValueError("baseline_mode must be 'mean' or 'weighted'")
        
        if self.averaging_space not in ['probability', 'logit']:
            raise ValueError("averaging_space must be 'probability' or 'logit'")
    
    def _generate_random_bins(self, y_values, rng):
        """
        Generate fully random bin boundaries within the y range.
        
        Parameters
        ----------
        y_values : array
            Training target values
        rng : numpy random generator
            Random number generator
        
        Returns array of bin edges including min and max.
        """
        # Get min and max of y values
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        
        # Add small epsilon to avoid edge case issues
        epsilon = (y_max - y_min) * 1e-6
        y_min -= epsilon
        y_max += epsilon
        
        # Generate random boundaries between y_min and y_max
        random_boundaries = rng.uniform(y_min, y_max, size=self.n_bins - 1)
        random_boundaries = np.sort(random_boundaries)
        
        # Add endpoints (min and max)
        bin_boundaries = np.concatenate([[y_min], random_boundaries, [y_max]])
        
        # Remove duplicate boundaries (can happen if random values are very close)
        bin_boundaries = np.unique(bin_boundaries)
        
        return bin_boundaries
    
    def _bin_target(self, y, bin_boundaries):
        """Convert continuous target to discrete classes based on bin boundaries."""
        # Use digitize to assign each y value to a bin
        # bins are [y_min, edge1), [edge1, edge2), ..., [edge_{n-1}, y_max]
        # digitize returns 1-indexed, so we subtract 1
        classes = np.digitize(y, bin_boundaries[1:-1], right=False)
        return classes
    
    def fit(self, X, y):
        """
        Build a custom boosted distribution regressor from the training set (X, y).
        
        Each boosting iteration trains a single tree on logit-space residuals with
        random binning boundaries.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or pandas DataFrame.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate parameters
        self._validate_params()
        
        # Check if X is a DataFrame and store feature names
        self._is_dataframe = isinstance(X, pd.DataFrame)
        if self._is_dataframe:
            self.feature_names_in_ = X.columns.tolist()
            X_array = X.values
            y_array = np.asarray(y)
        else:
            X_array = X
            y_array = y
        
        # Validate input
        X_array, y_array = check_X_y(X_array, y_array, accept_sparse=False, dtype=np.float64)
        
        # Store training information
        self.n_features_in_ = X_array.shape[1]
        n_samples = len(X_array)
        
        # Store y range for predictions (with epsilon)
        y_min = float(np.min(y_array))
        y_max = float(np.max(y_array))
        epsilon = (y_max - y_min) * 1e-6
        self.y_min_ = y_min - epsilon
        self.y_max_ = y_max + epsilon
        
        # Create fine grid over y-range
        self.grid_ = np.linspace(self.y_min_, self.y_max_, self.grid_resolution)
        
        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)
        
        # Initialize list to store estimators
        self.estimators_ = []
        
        # Always create DataFrame with feature names for consistency
        if self._is_dataframe:
            feature_names = self.feature_names_in_
        else:
            # Create generic feature names for numpy arrays
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        # Initialize accumulated predictions in logit space (starts at 0)
        accumulated_logits = np.zeros((n_samples, self.grid_resolution))
        
        # Custom boosting: train N trees (one per iteration)
        for iter_idx in range(self.n_estimators):
            # Generate random bin boundaries
            bin_boundaries = self._generate_random_bins(y_array, rng)
            n_bins = len(bin_boundaries) - 1
            
            # Determine which bin contains true y for each sample
            y_binned = self._bin_target(y_array, bin_boundaries)
            
            # Create expanded dataset (n_samples × n_bins rows)
            X_expanded = np.repeat(X_array, n_bins, axis=0)
            bin_ids = np.tile(np.arange(n_bins), n_samples)
            
            # Calculate targets in logit space
            y_binned_expanded = np.repeat(y_binned, n_bins)
            is_true_bin = (bin_ids == y_binned_expanded).astype(float)
            
            # Target: logit(1 - eps) for true bin, logit(0 + eps) for others
            target_logit = np.where(
                is_true_bin == 1,
                logit(1 - self.logit_eps),
                logit(self.logit_eps)
            )
            
            # Calculate baseline from accumulated predictions - VECTORIZED
            # Pre-compute which bin each grid point belongs to
            # Use digitize: returns bin index for each grid point
            grid_bin_indices = np.digitize(self.grid_, bin_boundaries[1:-1], right=False)
            grid_bin_indices = np.clip(grid_bin_indices, 0, n_bins - 1)
            
            # Create baseline array
            baseline = np.zeros(n_samples * n_bins)
            
            # For each bin, calculate mean of accumulated values for grid points in that bin
            if self.averaging_space == 'probability':
                # Average in probability space
                # Convert logits to probabilities
                accumulated_probs = expit(accumulated_logits)
                
                for bin_idx in range(n_bins):
                    bin_mask = (grid_bin_indices == bin_idx)
                    if np.sum(bin_mask) > 0:
                        # Calculate mean of probabilities across grid points in this bin
                        # Shape: (n_samples,)
                        bin_means_prob = np.mean(accumulated_probs[:, bin_mask], axis=1)
                        # Convert back to logit space
                        # Clip to avoid logit(0) and logit(1)
                        bin_means_prob = np.clip(bin_means_prob, self.logit_eps, 1 - self.logit_eps)
                        bin_means_logit = logit(bin_means_prob)
                        # Assign to baseline for all samples for this bin
                        baseline[bin_idx::n_bins] = bin_means_logit
            else:
                # Average in logit space (original behavior)
                for bin_idx in range(n_bins):
                    bin_mask = (grid_bin_indices == bin_idx)
                    if np.sum(bin_mask) > 0:
                        # Calculate mean across grid points in this bin for all samples at once
                        # Shape: (n_samples,)
                        bin_means = np.mean(accumulated_logits[:, bin_mask], axis=1)
                        # Assign to baseline for all samples for this bin
                        baseline[bin_idx::n_bins] = bin_means
            
            # Calculate residuals
            residuals = target_logit - baseline
            
            # Create DataFrame with bin_id as additional feature
            X_df_expanded = pd.DataFrame(X_expanded, columns=feature_names)
            X_df_expanded['bin_id'] = bin_ids
            X_df_expanded['bin_id'] = pd.Categorical(X_df_expanded['bin_id'])
            
            # Prepare LGBM parameters
            tree_params = {
                'n_estimators': 1,  # Single tree per iteration
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'num_leaves': self.num_leaves,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'max_bin': self.max_bin,
                'min_child_samples': self.min_child_samples,
                'verbose': -1,
                'force_col_wise': True,
            }
            
            # Add any additional kwargs
            tree_params.update(self.lgbm_kwargs)
            
            # Train single LGBMRegressor tree
            tree = LGBMRegressor(
                random_state=None if self.random_state is None else rng.integers(0, 2**31),
                **tree_params
            )
            
            tree.fit(X_df_expanded, residuals)
            
            # Predict on expanded dataset
            tree_predictions = tree.predict(X_df_expanded)
            
            # Map predictions back to fine grid and accumulate - VECTORIZED
            tree_predictions_reshaped = tree_predictions.reshape(n_samples, n_bins)
            
            # For each bin, spread its prediction over grid points in that bin
            for bin_idx in range(n_bins):
                bin_mask = (grid_bin_indices == bin_idx)
                if np.sum(bin_mask) > 0:
                    # Add prediction for this bin to all grid points in the bin
                    # Broadcasting: (n_samples, 1) * learning_rate -> (n_samples, n_grid_points_in_bin)
                    accumulated_logits[:, bin_mask] += (
                        self.learning_rate * tree_predictions_reshaped[:, bin_idx:bin_idx+1]
                    )
            
            # Store the estimator information
            self.estimators_.append({
                'tree': tree,
                'bin_boundaries': bin_boundaries,
                'n_bins': n_bins
            })
        
        return self
    
    def _map_probs_to_distribution(self, bin_probs, bin_boundaries, y_values):
        """
        Map bin probabilities to continuous distribution over y_values.
        
        Parameters
        ----------
        bin_probs : array of shape (n_bins,)
            Probability for each bin.
        
        bin_boundaries : array of shape (n_bins + 1,)
            Bin edges for all bins.
        
        y_values : array of shape (resolution,)
            Grid of y values to evaluate distribution on.
        
        Returns
        -------
        distribution : array of shape (resolution,)
            Probability density at each y_value.
        """
        distribution = np.zeros(len(y_values))
        
        # For each bin, spread its probability uniformly over its bin range
        for i, prob in enumerate(bin_probs):
            if prob > 0:
                # Find which y_values fall in this bin
                lower = bin_boundaries[i]
                upper = bin_boundaries[i + 1]
                
                # Find indices of y_values in this range
                mask = (y_values >= lower) & (y_values < upper)
                
                # Handle the last bin (include upper boundary)
                if i == len(bin_boundaries) - 2:
                    mask = (y_values >= lower) & (y_values <= upper)
                
                # Spread probability uniformly
                n_points_in_bin = np.sum(mask)
                if n_points_in_bin > 0:
                    distribution[mask] += prob / n_points_in_bin
        
        return distribution
    
    def predict_distribution(self, X):
        """
        Predict probability distribution over target variable for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to predict distributions. Can be numpy array or pandas DataFrame.
        
        Returns
        -------
        y_values : array of shape (grid_resolution,)
            Grid of y values (same as model's grid_).
        
        distributions : array of shape (n_samples, grid_resolution)
            Probability distribution for each sample at each y_value.
        """
        check_is_fitted(self)
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_array = check_array(X_array, accept_sparse=False, dtype=np.float64)
        
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, but model was trained with "
                f"{self.n_features_in_} features"
            )
        
        n_samples = X_array.shape[0]
        
        # Always create DataFrame with feature names for consistency
        if self._is_dataframe:
            feature_names = self.feature_names_in_
        else:
            # Create generic feature names for numpy arrays
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        # Initialize accumulated predictions in logit space (starts at 0)
        accumulated_logits = np.zeros((n_samples, self.grid_resolution))
        
        # Accumulate predictions from each tree
        for estimator_info in self.estimators_:
            tree = estimator_info['tree']
            bin_boundaries = estimator_info['bin_boundaries']
            n_bins = estimator_info['n_bins']
            
            # Create expanded dataset with all possible bin_ids for each sample
            X_expanded = np.repeat(X_array, n_bins, axis=0)
            bin_ids = np.tile(np.arange(n_bins), n_samples)
            
            # Create DataFrame with bin_id as additional column
            X_df_expanded = pd.DataFrame(X_expanded, columns=feature_names)
            X_df_expanded['bin_id'] = bin_ids
            X_df_expanded['bin_id'] = pd.Categorical(X_df_expanded['bin_id'])
            
            # Get tree predictions (in logit space)
            tree_predictions = tree.predict(X_df_expanded)
            
            # Reshape to (n_samples, n_bins)
            tree_predictions_reshaped = tree_predictions.reshape(n_samples, n_bins)
            
            # Map predictions back to fine grid - VECTORIZED
            # Pre-compute which bin each grid point belongs to
            grid_bin_indices = np.digitize(self.grid_, bin_boundaries[1:-1], right=False)
            grid_bin_indices = np.clip(grid_bin_indices, 0, n_bins - 1)
            
            # For each bin, spread its prediction over grid points in that bin
            for bin_idx in range(n_bins):
                bin_mask = (grid_bin_indices == bin_idx)
                if np.sum(bin_mask) > 0:
                    # Add prediction for this bin to all grid points in the bin
                    accumulated_logits[:, bin_mask] += (
                        self.learning_rate * tree_predictions_reshaped[:, bin_idx:bin_idx+1]
                    )
        
        # Convert from logit space to probabilities
        distributions = expit(accumulated_logits)
        
        # Normalize to sum to 1 for each sample
        for i in range(n_samples):
            total = np.sum(distributions[i])
            if total > 0:
                distributions[i] /= total
        
        return self.grid_, distributions
    
    def predict_mean(self, X):
        """
        Predict the mean of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        means : array of shape (n_samples,)
            Predicted means.
        """
        y_values, distributions = self.predict_distribution(X)
        
        # Compute expected value
        means = np.sum(distributions * y_values[np.newaxis, :], axis=1)
        
        return means
    
    def predict(self, X):
        """
        Predict using the mean of the distribution (default point estimate).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values.
        """
        return self.predict_mean(X)
    
    def predict_median(self, X):
        """
        Predict the median of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        medians : array of shape (n_samples,)
            Predicted medians.
        """
        return self.predict_quantile(X, q=0.5)
    
    def predict_mode(self, X):
        """
        Predict the mode (peak) of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        modes : array of shape (n_samples,)
            Predicted modes.
        """
        y_values, distributions = self.predict_distribution(X)
        
        # Find the peak of each distribution
        peak_indices = np.argmax(distributions, axis=1)
        modes = y_values[peak_indices]
        
        return modes
    
    def predict_quantile(self, X, q=0.5):
        """
        Predict quantile(s) of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        q : float or array-like of floats
            Quantile(s) to compute, in [0, 1].
        
        Returns
        -------
        quantiles : array
            Predicted quantiles.
            - If q is scalar: shape (n_samples,)
            - If q is array: shape (n_samples, len(q))
        """
        y_values, distributions = self.predict_distribution(X)
        
        n_samples = distributions.shape[0]
        q = np.atleast_1d(q)
        
        quantiles = np.zeros((n_samples, len(q)))
        
        for i in range(n_samples):
            # Compute cumulative distribution
            cdf = np.cumsum(distributions[i])
            cdf /= cdf[-1]  # Normalize
            
            # Interpolate to find quantiles
            for j, quantile_level in enumerate(q):
                quantiles[i, j] = np.interp(quantile_level, cdf, y_values)
        
        # Return scalar if single quantile was requested
        if len(q) == 1:
            return quantiles[:, 0]
        
        return quantiles
    
    def predict_std(self, X):
        """
        Predict the standard deviation of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        stds : array of shape (n_samples,)
            Predicted standard deviations.
        """
        y_values, distributions = self.predict_distribution(X)
        
        # Compute mean
        means = np.sum(distributions * y_values[np.newaxis, :], axis=1)
        
        # Compute variance
        variances = np.sum(
            distributions * (y_values[np.newaxis, :] - means[:, np.newaxis])**2,
            axis=1
        )
        
        stds = np.sqrt(variances)
        
        return stds
    
    def predict_interval(self, X, confidence=0.95):
        """
        Predict confidence/prediction interval for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        confidence : float, default=0.95
            Confidence level for the interval.
        
        Returns
        -------
        intervals : array of shape (n_samples, 2)
            Lower and upper bounds of the interval for each sample.
        """
        alpha = 1 - confidence
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        quantiles = self.predict_quantile(X, q=[lower_q, upper_q])
        
        return quantiles

