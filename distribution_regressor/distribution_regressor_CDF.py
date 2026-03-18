"""
CDF-based Distribution Regressor (Single LGBM)

Learns the conditional CDF F(τ|x) = P(Y ≤ τ | X = x) using binary targets
z = 1{y ≤ τ} and logistic loss. Recovers the PMF by differencing the CDF.

Every observation provides dense signal at every threshold:
  - all thresholds below y: target = 0
  - all thresholds above y: target = 1
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMRegressor
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold


class DistributionRegressorCDF(BaseEstimator, RegressorMixin):
    """
    Predicts probability distributions by learning the conditional CDF
    F(τ|x) = P(Y ≤ τ | X = x) with a single LightGBM model trained on
    binary threshold targets under logistic loss.

    Parameters
    ----------
    n_bins : int, default=50
        Number of grid points to discretize the target variable range.

    use_base_model : bool, default=False
        If True, trains a base LGBMRegressor for point predictions and learns
        the residual CDF via OOF residuals. If False, learns the CDF of raw y.

    monte_carlo_training : bool, default=False
        If True, use Monte Carlo sampling of threshold points (K per sample).
        If False, use full grid expansion.

    mc_samples : int, default=5
        Number of threshold sample points per observation when
        monte_carlo_training=True.

    mc_resample_freq : int, default=100
        Resample MC grid points every this many trees. E.g. 1 means every
        tree gets fresh grid points, 100 means resample every 100 trees.
        Lower = better grid coverage but more overhead. Only used when
        monte_carlo_training=True.

    output_smoothing : float, default=0
        Gaussian smoothing sigma (in grid-index units) applied to the CDF
        before deriving any outputs. Set to 0 to disable. Can be changed
        after training via set_output_smoothing().

    atom_values : float, array-like, or None, default=None
        Discrete mass points whose CDF jumps are preserved through smoothing.
        Only relevant when output_smoothing > 0.

    n_estimators : int, default=100
        Number of boosting trees.

    learning_rate : float, default=0.1
        Boosting learning rate.

    random_state : int or None, default=None
        Random seed.

    **kwargs : dict
        Additional parameters passed to LGBMRegressor.
    """

    def __init__(
        self,
        n_bins=50,
        use_base_model=False,
        n_estimators=100,
        learning_rate=0.1,
        monte_carlo_training=False,
        mc_samples=5,
        mc_resample_freq=100,
        output_smoothing=0,
        atom_values=None,
        random_state=None,
        **kwargs
    ):
        self.n_bins = n_bins
        self.use_base_model = use_base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.monte_carlo_training = monte_carlo_training
        self.mc_samples = mc_samples
        self.mc_resample_freq = mc_resample_freq
        self.output_smoothing = output_smoothing
        self.atom_values = atom_values
        self.random_state = random_state
        self.lgbm_kwargs = kwargs

    def set_output_smoothing(self, output_smoothing=0, atom_values=None):
        """
        Set or update output smoothing after training.

        Parameters
        ----------
        output_smoothing : float, default=0
            Gaussian smoothing sigma (in grid-index units) applied to the CDF.
            Set to 0 to disable smoothing.
        atom_values : float, array-like, or None
            Discrete mass points whose CDF jumps are preserved through smoothing.

        Returns
        -------
        self
        """
        self.output_smoothing = output_smoothing
        self.atom_values = atom_values
        return self

    def _validate_params(self):
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")

    def _to_dataframe(self, X):
        """Convert X to DataFrame if it isn't one already, using stored feature names."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def _expand_with_grid_points(self, X_df, grid_points, K):
        """
        Repeat each row of X_df K times and append grid_point column.
        Preserves original dtypes.
        """
        idx = np.repeat(np.arange(len(X_df)), K)
        X_expanded = X_df.iloc[idx].reset_index(drop=True)
        X_expanded['grid_point'] = grid_points
        return X_expanded

    def _per_sample_grids(self, base_preds):
        """Compute per-sample residual grids bounded to [y_min, y_max] in absolute space."""
        r_min = np.maximum(self.grid_[0], self.y_min_ - base_preds)
        r_max = np.minimum(self.grid_[-1], self.y_max_ - base_preds)
        t = np.linspace(0, 1, self.n_bins)
        return r_min[:, None] + t[None, :] * (r_max - r_min)[:, None]

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model by learning F(τ|x) = P(Y ≤ τ | X = x).

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
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]

        X_df = self._to_dataframe(X)
        y_array = np.asarray(y, dtype=np.float64)
        self.y_min_ = float(np.min(y_array))
        self.y_max_ = float(np.max(y_array))
        self.n_features_in_ = X_df.shape[1]
        n_samples = X_df.shape[0]

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        full_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'verbose': -1
        }
        full_params.update(self.lgbm_kwargs)

        # 2. Base model + OOF residuals (optional)
        if self.use_base_model:
            self.base_model_ = LGBMRegressor(**full_params)
            self.base_model_.fit(X_df, y_array, sample_weight=sample_weight)

            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            oof_predictions = np.zeros(n_samples)
            for train_idx, val_idx in kf.split(X_df):
                fold_model = LGBMRegressor(**full_params)
                fold_sw = sample_weight[train_idx] if sample_weight is not None else None
                fold_model.fit(X_df.iloc[train_idx], y_array[train_idx], sample_weight=fold_sw)
                oof_predictions[val_idx] = fold_model.predict(X_df.iloc[val_idx])

            target_for_grid = y_array - oof_predictions
            X_df = X_df.copy()
            X_df['_base_pred'] = oof_predictions
        else:
            self.base_model_ = None
            target_for_grid = y_array

        # 3. Define the Grid
        t_min = float(np.min(target_for_grid))
        t_max = float(np.max(target_for_grid))
        self.grid_ = np.linspace(t_min, t_max, self.n_bins)

        # ------------------------------------------------------------------
        # 4. Training Expansion + Fitting
        # ------------------------------------------------------------------
        if self.use_base_model:
            r_min = np.maximum(t_min, self.y_min_ - oof_predictions)
            r_max = np.minimum(t_max, self.y_max_ - oof_predictions)

        use_mc = self.monte_carlo_training
        K = self.mc_samples if use_mc else self.n_bins
        freq = self.mc_resample_freq if use_mc else self.n_estimators
        rng = np.random.default_rng(self.random_state) if use_mc else None

        # Monotone constraint: CDF must be non-decreasing in grid_point (last col)
        n_cols = X_df.shape[1] + 1  # +1 for grid_point column
        mono_constraints = [0] * (n_cols - 1) + [1]

        base_params = {
            'objective': 'cross_entropy',
            'metric': 'cross_entropy',
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'monotone_constraints': mono_constraints,
            'verbose': -1
        }
        base_params.update(self.lgbm_kwargs)

        self.model_ = None
        trees_built = 0

        while trees_built < self.n_estimators:
            # Sample or build grid points
            if use_mc:
                if self.use_base_model:
                    g_points = (r_min[:, None]
                                + rng.uniform(size=(n_samples, K))
                                * (r_max - r_min)[:, None])
                    g_points = np.clip(g_points, r_min[:, None], r_max[:, None])
                else:
                    g_points = rng.uniform(t_min, t_max, size=(n_samples, K))
                grid_points_flat = g_points.ravel()
            else:
                if self.use_base_model:
                    grid_points_flat = self._per_sample_grids(oof_predictions).ravel()
                else:
                    grid_points_flat = np.tile(self.grid_, n_samples)

            X_expanded = self._expand_with_grid_points(X_df, grid_points_flat, K)
            y_targets = (np.repeat(target_for_grid, K) <= grid_points_flat).astype(np.float64)
            sw_expanded = np.repeat(sample_weight, K) if sample_weight is not None else None

            n_trees = min(freq, self.n_estimators - trees_built)

            round_params = {**base_params, 'n_estimators': n_trees}
            round_model = LGBMRegressor(**round_params)
            round_model.fit(X_expanded, y_targets, sample_weight=sw_expanded,
                            init_model=self.model_)
            self.model_ = round_model
            trees_built += n_trees

        return self

    def _predict_cdf(self, X):
        """
        Predict CDF values on the full grid with monotonicity enforcement
        and optional smoothing. This is the single source of truth for the
        predicted CDF — all other methods derive from this.

        Smoothing is controlled by self.output_smoothing and self.atom_values,
        set at construction or via set_output_smoothing().

        Returns
        -------
        grids : array of shape (n_samples, n_bins)
            Per-sample grid points (residual space if base model is used).
        cdf_matrix : array of shape (n_samples, n_bins)
            Monotone CDF values in [0, 1].
        """
        check_is_fitted(self)

        X_df = self._to_dataframe(X)
        n_samples = X_df.shape[0]

        if self.base_model_ is not None:
            base_preds = self.base_model_.predict(X_df)
            X_df = X_df.copy()
            X_df['_base_pred'] = base_preds
            grids = self._per_sample_grids(base_preds)
        else:
            grids = np.tile(self.grid_, (n_samples, 1))

        grid_points_flat = grids.ravel()
        X_expanded = self._expand_with_grid_points(X_df, grid_points_flat, self.n_bins)

        pred_cdf = self.model_.predict(X_expanded)
        cdf_matrix = pred_cdf.reshape(n_samples, self.n_bins)

        # Enforce monotonicity via cumulative maximum
        cdf_matrix = np.maximum.accumulate(cdf_matrix, axis=1)
        cdf_matrix = np.clip(cdf_matrix, 0.0, 1.0)

        # Optional smoothing
        if self.output_smoothing > 0:
            cdf_matrix = self._smooth_cdf(grids, cdf_matrix, self.output_smoothing, self.atom_values)

        return grids, cdf_matrix

    def _smooth_cdf(self, grids, cdf_matrix, sigma, atom_values=None):
        """
        Smooth the CDF with Gaussian filter, optionally preserving jumps at
        atom values (discrete mass points).

        Parameters
        ----------
        grids : array of shape (n_samples, n_bins)
        cdf_matrix : array of shape (n_samples, n_bins)
        sigma : float
            Gaussian smoothing width (in grid-index units).
        atom_values : float, array-like, or None
            Values whose CDF jumps should be preserved through smoothing.

        Returns
        -------
        cdf_smoothed : array of shape (n_samples, n_bins)
        """
        cdf_matrix = cdf_matrix.copy()
        n_samples, n_bins = cdf_matrix.shape

        if atom_values is not None:
            atom_arr = np.atleast_1d(atom_values)
            bin_width = float(np.median(np.diff(grids[0])))
            atol = bin_width / 2

            # Find atom positions and save their CDF jumps
            atom_jumps = {}
            for atom_val in atom_arr:
                dists = np.abs(grids - atom_val)
                closest_idx = np.argmin(dists, axis=1)
                within_tol = dists[np.arange(n_samples), closest_idx] <= atol
                rows = np.arange(n_samples)[within_tol]
                cols = closest_idx[within_tol]
                prev_cols = np.maximum(cols - 1, 0)
                jumps = np.maximum(cdf_matrix[rows, cols] - cdf_matrix[rows, prev_cols], 0.0)
                atom_jumps[atom_val] = (rows, cols, jumps)

                # Remove jumps before smoothing (flatten CDF at atoms)
                for j in range(n_bins):
                    mask = cols <= j
                    if mask.any():
                        cdf_matrix[rows[mask], j] -= jumps[mask]

            # Smooth the continuous part
            gaussian_filter1d(cdf_matrix, sigma=sigma, axis=1, output=cdf_matrix)

            # Restore atom jumps
            for atom_val, (rows, cols, jumps) in atom_jumps.items():
                for j in range(n_bins):
                    mask = cols <= j
                    if mask.any():
                        cdf_matrix[rows[mask], j] += jumps[mask]
        else:
            gaussian_filter1d(cdf_matrix, sigma=sigma, axis=1, output=cdf_matrix)

        return np.clip(cdf_matrix, 0.0, 1.0)

    def _cdf_to_pmf(self, cdf_matrix):
        """
        Derive a normalized PMF by differencing a CDF.

        Parameters
        ----------
        cdf_matrix : array of shape (n_samples, n_bins)

        Returns
        -------
        pmf : array of shape (n_samples, n_bins)
        """
        pmf = np.empty_like(cdf_matrix)
        pmf[:, 0] = cdf_matrix[:, 0]
        pmf[:, 1:] = np.diff(cdf_matrix, axis=1)
        pmf = np.maximum(pmf, 0.0)

        row_sums = pmf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        pmf /= row_sums

        return pmf

    def _to_absolute(self, X, values):
        """Shift residual-space values to absolute space if base model is used."""
        if self.base_model_ is not None:
            return values + self.base_model_.predict(self._to_dataframe(X))
        return values

    def _to_residual(self, X, y):
        """Convert absolute y values to residual space if base model is used."""
        y_array = np.asarray(y, dtype=np.float64)
        if self.base_model_ is not None:
            return y_array - self.base_model_.predict(self._to_dataframe(X))
        return y_array

    def _interpolate_on_grid(self, grids, matrix, y_residual):
        """
        Per-sample linear interpolation of a grid-based matrix at given y values.

        Returns interpolated values, with 0 below support and 1 above for CDF,
        or the caller handles out-of-range.
        """
        bin_widths = grids[:, 1] - grids[:, 0]
        y_clipped = np.clip(y_residual, grids[:, 0], grids[:, -1])
        frac_idx = (y_clipped - grids[:, 0]) / bin_widths
        idx_lo = np.clip(np.floor(frac_idx).astype(int), 0, self.n_bins - 2)
        idx_hi = idx_lo + 1
        alpha = frac_idx - idx_lo

        rows = np.arange(len(y_residual))
        return (1 - alpha) * matrix[rows, idx_lo] + alpha * matrix[rows, idx_hi]

    def predict_distribution(self, X, cumulative=False):
        """
        Returns grid points and probability distribution for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to predict distributions.
        cumulative : bool, default=False
            If False, return the PMF (probability mass per bin).
            If True, return the CDF (cumulative distribution function).

        Returns
        -------
        grids : array of shape (n_samples, n_bins)
            Per-sample grid points in absolute space.
        distributions : array of shape (n_samples, n_bins)
            PMF or CDF for each sample, depending on `cumulative`.
        base_offsets : array of shape (n_samples,)
            Per-sample base model prediction. Zeros when use_base_model=False.
        """
        grids, cdf_matrix = self._predict_cdf(X)
        dist = cdf_matrix if cumulative else self._cdf_to_pmf(cdf_matrix)
        base_offsets = self.base_model_.predict(self._to_dataframe(X)) if self.base_model_ is not None else np.zeros(len(grids))
        absolute_grids = grids + base_offsets[:, None]
        return absolute_grids, dist, base_offsets

    def predict(self, X):
        """Default: Predict Mean"""
        return self.predict_mean(X)

    def predict_mean(self, X):
        """
        Predict the mean of the distribution for each sample.
        Computes E[Y|X] = base_pred + sum(residual_grid * residual_pmf).
        """
        grids, cdf_matrix = self._predict_cdf(X)
        pmf = self._cdf_to_pmf(cdf_matrix)
        residual_means = np.sum(grids * pmf, axis=1)
        return self._to_absolute(X, residual_means)

    def predict_mode(self, X):
        """
        Predict the mode (peak) of the distribution for each sample.
        """
        grids, cdf_matrix = self._predict_cdf(X)
        pmf = self._cdf_to_pmf(cdf_matrix)
        max_indices = np.argmax(pmf, axis=1)
        modes = grids[np.arange(len(grids)), max_indices]
        return self._to_absolute(X, modes)

    def predict_quantile(self, X, q=0.5):
        """
        Predict quantile(s) using the CDF directly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        q : float or array-like, default=0.5

        Returns
        -------
        quantiles : array of shape (n_samples,) or (n_samples, n_quantiles)
        """
        grids, cdf_matrix = self._predict_cdf(X)

        q_arr = np.atleast_1d(q)
        is_scalar = np.ndim(q) == 0

        # Find first grid point where CDF >= q
        mask = cdf_matrix[:, None, :] >= q_arr[None, :, None]
        indices = mask.argmax(axis=2)
        rows = np.arange(len(grids))[:, None]
        quantiles = grids[rows, indices]

        if self.base_model_ is not None:
            base_pred = self.base_model_.predict(self._to_dataframe(X))
            quantiles = quantiles + base_pred[:, None]

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
        grids, cdf_matrix = self._predict_cdf(X)
        pmf = self._cdf_to_pmf(cdf_matrix)
        means = np.sum(grids * pmf, axis=1)
        expected_sq = np.sum(grids ** 2 * pmf, axis=1)
        variance = expected_sq - means ** 2
        return np.sqrt(np.maximum(variance, 0.0))

    def pdf(self, X, y, eps=1e-10):
        """
        Evaluate the predicted probability density at given y values.
        Returns eps for y outside the grid support.
        """
        grids, cdf_matrix = self._predict_cdf(X)
        pmf = self._cdf_to_pmf(cdf_matrix)

        bin_widths = grids[:, 1] - grids[:, 0]
        density_grid = pmf / bin_widths[:, None]

        y_residual = self._to_residual(X, y)
        densities = self._interpolate_on_grid(grids, density_grid, y_residual)

        # Zero density outside support
        out_of_range = (y_residual < grids[:, 0]) | (y_residual > grids[:, -1])
        densities[out_of_range] = 0.0

        return np.maximum(densities, eps)

    def logpdf(self, X, y, eps=1e-10):
        """
        Evaluate the log probability density at given y values.
        """
        return np.log(self.pdf(X, y, eps=eps))

    def cdf(self, X, y):
        """
        Evaluate the predicted CDF at given y values.
        Returns 0 for y below support, 1 for y above support.
        """
        grids, cdf_matrix = self._predict_cdf(X)

        y_residual = self._to_residual(X, y)
        cdf_vals = self._interpolate_on_grid(grids, cdf_matrix, y_residual)

        # Out-of-range handling
        cdf_vals = np.where(y_residual < grids[:, 0], 0.0, cdf_vals)
        cdf_vals = np.where(y_residual > grids[:, -1], 1.0, cdf_vals)

        return np.clip(cdf_vals, 0.0, 1.0)

    def ppf(self, X, q=0.5):
        """
        Percent point function (inverse CDF / quantile function).
        """
        return self.predict_quantile(X, q=q)

    def nll(self, X, y, eps=1e-10):
        """
        Compute mean negative log-likelihood.
        """
        return -np.mean(self.logpdf(X, y, eps=eps))
