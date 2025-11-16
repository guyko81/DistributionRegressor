"""
Distribution Regressor Random Forest

A scikit-learn-compatible regressor that predicts full probability distributions
over the target variable range using an ensemble of LGBMClassifiers with random
binning strategies.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from lightgbm import LGBMClassifier
import warnings


class DistributionRegressorRandomForest(BaseEstimator, RegressorMixin):
    """
    Distribution Regressor using Random Forest philosophy with LGBMClassifier.
    
    Each tree uses random binning boundaries to predict probability distributions
    over the target variable range.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees/turns in the forest.
    
    n_bins : int, default=10
        Number of bins per tree for discretizing the target variable.
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    lgbm_params : dict or None, default=None
        Dictionary of parameters to pass to each LGBMClassifier. Common parameters include:
        - n_estimators: Number of boosting rounds (default=100 if not specified)
        - num_leaves: Maximum tree leaves (default=31)
        - max_depth: Maximum tree depth (default=-1, no limit)
        - learning_rate: Boosting learning rate (default=0.1)
        - subsample: Fraction of samples for bagging (default=1.0)
        - colsample_bytree: Fraction of features for each tree (default=1.0)
        - min_child_samples: Minimum data in one leaf (default=20)
        Note: Do not pass 'random_state' here; it's controlled by the main parameter.
    
    Attributes
    ----------
    estimators_ : list of dict
        List of dictionaries, one per tree, containing:
        - 'classifier': Trained LGBMClassifier
        - 'bin_boundaries': Array of bin edges
    
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
        random_state=None,
        lgbm_params=None
    ):
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.random_state = random_state
        self.lgbm_params = lgbm_params if lgbm_params is not None else {}
    
    def _validate_params(self):
        """Validate input parameters."""
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        
        if self.n_bins <= 1:
            raise ValueError("n_bins must be greater than 1")
    
    def _generate_random_bins(self, y_values, rng):
        """
        Generate random bin boundaries using quantiles to ensure all bins have data.
        
        Parameters
        ----------
        y_values : array
            Training target values
        rng : numpy random generator
            Random number generator
        
        Returns array of bin edges including min and max.
        """
        # Use random quantiles to ensure each bin has data
        # Generate random quantile points between 0 and 1
        random_quantiles = rng.uniform(0, 1, size=self.n_bins - 1)
        random_quantiles = np.sort(random_quantiles)
        
        # Get bin boundaries at these quantiles
        bin_boundaries = np.quantile(y_values, random_quantiles)
        
        # Add endpoints (min and max)
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        
        # Add small epsilon to avoid edge case issues
        epsilon = (y_max - y_min) * 1e-6
        y_min -= epsilon
        y_max += epsilon
        
        bin_boundaries = np.concatenate([[y_min], bin_boundaries, [y_max]])
        
        # Remove duplicate boundaries (can happen if data is sparse)
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
        Build a forest of distribution regressors from the training set (X, y).
        
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
        
        # Store y range for predictions (with epsilon)
        y_min = float(np.min(y_array))
        y_max = float(np.max(y_array))
        epsilon = (y_max - y_min) * 1e-6
        self.y_min_ = y_min - epsilon
        self.y_max_ = y_max + epsilon
        
        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)
        
        # Initialize list to store estimators
        self.estimators_ = []
        
        # Always create DataFrame with feature names for consistency
        # This prevents LGBM feature name warnings
        if self._is_dataframe:
            feature_names = self.feature_names_in_
        else:
            # Create generic feature names for numpy arrays
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        X_df = pd.DataFrame(X_array, columns=feature_names)
        
        # Train each tree
        for i in range(self.n_estimators):
            # Generate random bin boundaries using quantiles
            bin_boundaries = self._generate_random_bins(y_array, rng)
            
            # Bin the target variable
            y_binned = self._bin_target(y_array, bin_boundaries)
            
            # Train LGBMClassifier
            # Set verbosity to -1 to suppress output
            lgbm_params = self.lgbm_params.copy()
            if 'verbose' not in lgbm_params:
                lgbm_params['verbose'] = -1
            # Disable feature name validation to avoid warnings
            if 'force_col_wise' not in lgbm_params:
                lgbm_params['force_col_wise'] = True
            
            # Set default n_estimators if not provided
            if 'n_estimators' not in lgbm_params:
                lgbm_params['n_estimators'] = 100
            
            # Ensure min_child_samples is set to avoid empty leaf errors
            if 'min_child_samples' not in lgbm_params:
                lgbm_params['min_child_samples'] = 1
            
            # Override min_child_weight if set to 0 (causes errors)
            if lgbm_params.get('min_child_weight', 1e-3) == 0.0:
                lgbm_params['min_child_weight'] = 1e-3
            
            # Remove random_state if user provided it (we control it)
            if 'random_state' in lgbm_params:
                warnings.warn(
                    "Found 'random_state' in lgbm_params. This will be ignored. "
                    "The random_state is controlled by the main model parameter.",
                    UserWarning
                )
                lgbm_params = {k: v for k, v in lgbm_params.items() if k != 'random_state'}
            
            classifier = LGBMClassifier(
                random_state=None if self.random_state is None else rng.integers(0, 2**31),
                **lgbm_params
            )
            
            classifier.fit(X_df, y_binned)
            
            # Store the estimator information
            self.estimators_.append({
                'classifier': classifier,
                'bin_boundaries': bin_boundaries
            })
        
        return self
    
    def _map_probs_to_distribution(self, class_probs, bin_boundaries, y_values, classifier):
        """
        Map class probabilities to continuous distribution over y_values.
        
        Parameters
        ----------
        class_probs : array of shape (n_classes_seen,)
            Probability for each class seen during training.
        
        bin_boundaries : array of shape (n_bins + 1,)
            Bin edges for all bins.
        
        y_values : array of shape (resolution,)
            Grid of y values to evaluate distribution on.
        
        classifier : LGBMClassifier
            The trained classifier (needed to get which classes were actually seen).
        
        Returns
        -------
        distribution : array of shape (resolution,)
            Probability density at each y_value.
        """
        distribution = np.zeros(len(y_values))
        
        # IMPORTANT: LGBMClassifier only predicts for classes it saw during training
        # classifier.classes_ contains the actual class labels
        # class_probs corresponds to these classes, NOT to all bins
        seen_classes = classifier.classes_
        
        # For each seen class, spread its probability uniformly over its bin range
        for idx, class_label in enumerate(seen_classes):
            prob = class_probs[idx]
            if prob > 0:
                # class_label is the bin index
                i = class_label
                
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
    
    def predict_distribution(self, X, resolution=100):
        """
        Predict probability distribution over target variable for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to predict distributions. Can be numpy array or pandas DataFrame.
        
        resolution : int, default=100
            Number of points in the y_values grid.
        
        Returns
        -------
        y_values : array of shape (resolution,)
            Grid of y values.
        
        distributions : array of shape (n_samples, resolution)
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
        
        # Create y_values grid
        y_values = np.linspace(self.y_min_, self.y_max_, resolution)
        
        # Initialize distributions
        distributions = np.zeros((n_samples, resolution))
        
        # Always create DataFrame with feature names for consistency
        # This prevents LGBM feature name warnings
        if self._is_dataframe:
            feature_names = self.feature_names_in_
        else:
            # Create generic feature names for numpy arrays
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        X_df = pd.DataFrame(X_array, columns=feature_names)
        
        # Accumulate predictions from each tree
        for estimator_info in self.estimators_:
            classifier = estimator_info['classifier']
            bin_boundaries = estimator_info['bin_boundaries']
            
            # Get class probabilities
            class_probs = classifier.predict_proba(X_df)
            
            # Map to distribution for each sample
            for i in range(n_samples):
                tree_distribution = self._map_probs_to_distribution(
                    class_probs[i], bin_boundaries, y_values, classifier
                )
                distributions[i] += tree_distribution
        
        # Average across trees
        distributions /= self.n_estimators
        
        # Normalize (in case of numerical issues)
        for i in range(n_samples):
            total = np.sum(distributions[i])
            if total > 0:
                distributions[i] /= total
        
        return y_values, distributions
    
    def predict_mean(self, X, resolution=100):
        """
        Predict the mean of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        means : array of shape (n_samples,)
            Predicted means.
        """
        y_values, distributions = self.predict_distribution(X, resolution=resolution)
        
        # Compute expected value
        means = np.sum(distributions * y_values[np.newaxis, :], axis=1)
        
        return means
    
    def predict(self, X, resolution=100):
        """
        Predict using the mean of the distribution (default point estimate).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values.
        """
        return self.predict_mean(X, resolution=resolution)
    
    def predict_median(self, X, resolution=100):
        """
        Predict the median of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        medians : array of shape (n_samples,)
            Predicted medians.
        """
        return self.predict_quantile(X, q=0.5, resolution=resolution)
    
    def predict_mode(self, X, resolution=100):
        """
        Predict the mode (peak) of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        modes : array of shape (n_samples,)
            Predicted modes.
        """
        y_values, distributions = self.predict_distribution(X, resolution=resolution)
        
        # Find the peak of each distribution
        peak_indices = np.argmax(distributions, axis=1)
        modes = y_values[peak_indices]
        
        return modes
    
    def predict_quantile(self, X, q=0.5, resolution=100):
        """
        Predict quantile(s) of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        q : float or array-like of floats
            Quantile(s) to compute, in [0, 1].
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        quantiles : array
            Predicted quantiles.
            - If q is scalar: shape (n_samples,)
            - If q is array: shape (n_samples, len(q))
        """
        y_values, distributions = self.predict_distribution(X, resolution=resolution)
        
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
    
    def predict_std(self, X, resolution=100):
        """
        Predict the standard deviation of the distribution for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        stds : array of shape (n_samples,)
            Predicted standard deviations.
        """
        y_values, distributions = self.predict_distribution(X, resolution=resolution)
        
        # Compute mean
        means = np.sum(distributions * y_values[np.newaxis, :], axis=1)
        
        # Compute variance
        variances = np.sum(
            distributions * (y_values[np.newaxis, :] - means[:, np.newaxis])**2,
            axis=1
        )
        
        stds = np.sqrt(variances)
        
        return stds
    
    def predict_interval(self, X, confidence=0.95, resolution=100):
        """
        Predict confidence/prediction interval for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        confidence : float, default=0.95
            Confidence level for the interval.
        
        resolution : int, default=100
            Number of points for distribution computation.
        
        Returns
        -------
        intervals : array of shape (n_samples, 2)
            Lower and upper bounds of the interval for each sample.
        """
        alpha = 1 - confidence
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        quantiles = self.predict_quantile(X, q=[lower_q, upper_q], resolution=resolution)
        
        return quantiles

