import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import check_random_state
from scipy.stats import norm
from scipy.special import logit

def _softmax_stable(logits, axis=-1):
    """Numerically stable softmax function."""
    z = logits - np.max(logits, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)

def _ensure_2d(x):
    """Ensure input is a 2D array."""
    x = np.asarray(x)
    return x if x.ndim == 2 else x.reshape(-1, 1)

@dataclass
class DistributionRegressor:
    """
    Nonparametric distributional regression using LightGBM.

    Learns full probability distributions p(y|x) for regression tasks without 
    parametric assumptions. Uses contrastive learning with positive (true) and 
    negative (synthetic) pairs to learn a scoring function, which is then 
    converted to distributions via softmax.
    
    Two training modes:
    - "hard" negatives: Binary labels (0/1) with LGBMClassifier
    - "soft" negatives: Continuous plausibility labels with LGBMRegressor,
      weighted by distance from true value
    """
    # LGBM parameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 4
    min_samples_leaf: int = 20   # Maps to min_child_samples in LGBM
    subsample: float = 0.8       # Maps to bagging_fraction
    colsample: float = 0.8       # Maps to feature_fraction
    lgbm_params: Optional[Dict] = None # For other LGBM params like reg_alpha, etc.

    # Negative sampling strategy
    negative_type: Literal["hard", "soft"] = "hard"  # Hard=classifier, Soft=regressor
    k_neg: int = 100             # Number of negative samples per positive sample
    neg_sampler: Literal["uniform", "normal", "mix", "stratified"] = "mix"
    neg_std: float = 2.0         # Std dev for normal sampler (IN SCALED SPACE)
    soft_label_std: float = 1.0  # Std dev for soft label calculation (IN SCALED SPACE, only for soft negatives)
    uniform_margin: float = 0.25 # Extend global y-range for uniform negatives
    # Stratified sampling: fraction of negatives from global uniform vs local
    stratified_global_frac: float = 0.5  # 50% global, 50% local by default

    # Feature map
    use_interactions: bool = True
    # NEW: Specify categorical features by their column index
    categorical_features: Optional[List[int]] = None

    # Training configuration
    early_stopping_rounds: Optional[int] = 50
    verbose: int = 1
    random_state: Optional[int] = 42

    # Internal attributes (filled in after fit)
    lgbm_model_: Union[lgb.LGBMClassifier, lgb.LGBMRegressor] = None
    # Scalers for numerical X and y
    num_x_scaler_: Union[StandardScaler, MinMaxScaler] = None
    y_min_: float = None  # For min-max scaling (hard negatives)
    y_max_: float = None  # For min-max scaling (hard negatives)
    y_mean_: float = None  # For z-score scaling (soft negatives)
    y_std_: float = None   # For z-score scaling (soft negatives)
    # Feature indices
    num_feature_idx_: List[int] = None
    cat_feature_idx_: List[int] = None
    lgbm_cat_idx_: List[int] = None
    # y_train_bounds_ are stored in scaled space
    y_train_bounds_: Tuple[float, float] = None
    evals_result_: dict = None
    best_iteration_: Optional[int] = None

    def _phi(self, X_num_scaled, X_cat, y_scaled_vec):
        """
        Feature map phi(x, y) operating on split/scaled inputs.
        The final feature order is [scaled_num_X | cat_X | y_features | interaction_features].
        """
        y = y_scaled_vec.reshape(-1, 1)

        # Base parts: original X features
        base_parts = []
        has_num = X_num_scaled is not None and X_num_scaled.shape[1] > 0
        has_cat = X_cat is not None and X_cat.shape[1] > 0
        if has_num:
            base_parts.append(X_num_scaled)
        if has_cat:
            base_parts.append(X_cat)

        # Features derived from y
        # Numerically stable features that distinguish y=0 from nearby values
        eps = 1e-8  # For numerical stability
        y_sq = y * y
        y_parts = [
            y, 
            y_sq, 
            np.abs(y),
            np.tanh(y),                              # Bounded sigmoid-like, always stable
            1.0 / (1.0 + np.abs(y) + eps),          # Safe reciprocal, max at y=0
            -np.log1p(y_sq),                         # -log(1 + y²) = log(1/(1+y²)), stable Gaussian-like
            np.log1p(np.abs(y)),                     # Stable log, distinguishes small values
            np.sin(y * np.pi),
            np.cos(y * np.pi),
        ]

        # Interaction features (only from numerical X)
        interaction_parts = []
        if has_num:
            interaction_parts.append(X_num_scaled.mean(axis=1, keepdims=True) * y)
            if self.use_interactions:
                xy = X_num_scaled * y
                x_minus_y = X_num_scaled - y
                interaction_parts.extend([
                    xy,      # Elementwise product
                    (xy**2), # Squared interactions
                    x_minus_y, # x - y
                    (x_minus_y**2), # Squared x - y
                ])

        all_parts = base_parts + y_parts + interaction_parts
        return np.hstack(all_parts)

    def _prepare_and_scale_X(self, X: np.ndarray, is_fitting: bool = False):
        """
        Splits X into numerical and categorical parts, and scales the numerical part.
        If is_fitting, it also fits the scaler and sets feature indices.
        Uses StandardScaler for soft negatives, MinMaxScaler for hard negatives.
        """
        X = _ensure_2d(X)
        if is_fitting:
            all_indices = set(range(X.shape[1]))
            self.cat_feature_idx_ = self.categorical_features if self.categorical_features else []
            self.num_feature_idx_ = sorted(list(all_indices - set(self.cat_feature_idx_)))

        X_cat = X[:, self.cat_feature_idx_]
        # Ensure numerical part is float for scaler; this can raise a helpful ValueError.
        X_num = X[:, self.num_feature_idx_].astype(float, copy=False)

        if is_fitting:
            # Use StandardScaler for soft negatives, MinMaxScaler for hard negatives
            if self.negative_type == "soft":
                self.num_x_scaler_ = StandardScaler().fit(X_num)
            else:
                self.num_x_scaler_ = MinMaxScaler().fit(X_num)
            # Define categorical indices for the transformed feature space (Phi)
            n_num = len(self.num_feature_idx_)
            n_cat = len(self.cat_feature_idx_)
            self.lgbm_cat_idx_ = list(range(n_num, n_num + n_cat)) if n_cat > 0 else []

        X_num_s = self.num_x_scaler_.transform(X_num)
        return X_num_s, X_cat

    def _build_candidates(self, y_true_scaled, rng):
        """
        Generates candidates in the SCALED y-space.
        
        Supports multiple sampling strategies:
        - normal: Samples around each true value with Gaussian noise
        - uniform: Samples uniformly from the global range
        - mix: Combines normal and uniform
        - stratified: Combines local (normal) and global (full-range uniform) sampling
                     to ensure both fine-grained and coarse-grained discrimination
        """
        y_true = np.asarray(y_true_scaled).ravel()
        n = len(y_true)
        C = self.k_neg + 1
        Yc = np.empty((n, C), dtype=float)

        Yc[:, 0] = y_true # First column is the positive target

        if self.neg_sampler == "normal":
            negs = rng.normal(loc=y_true[:, None], scale=self.neg_std, size=(n, self.k_neg))
            negs += rng.normal(scale=1e-6, size=negs.shape) # break ties
        elif self.neg_sampler == "uniform":  # uniform
            y_min, y_max = y_true.min(), y_true.max()
            span = y_max - y_min
            lo = y_min - self.uniform_margin * span
            hi = y_max + self.uniform_margin * span
            negs = rng.uniform(lo, hi, size=(n, self.k_neg))
        elif self.neg_sampler == "stratified":
            # STRATIFIED: Mix of local (around true value) and global (full range) negatives
            # This ensures the model learns both fine-grained and coarse-grained discrimination
            n_global = int(self.k_neg * self.stratified_global_frac)
            n_local = self.k_neg - n_global
            
            # Local negatives: around the true value (fine-grained discrimination)
            if n_local > 0:
                local_negs = rng.normal(loc=y_true[:, None], scale=self.neg_std, size=(n, n_local))
                local_negs += rng.normal(scale=1e-6, size=local_negs.shape)
            
            # Global negatives: uniform across ENTIRE training range (coarse-grained discrimination)
            # Use 0 to 1 since we're in scaled space, with margin
            if n_global > 0:
                global_negs = rng.uniform(-self.uniform_margin, 1.0 + self.uniform_margin, size=(n, n_global))
            
            # Combine
            if n_local > 0 and n_global > 0:
                negs = np.concatenate([local_negs, global_negs], axis=1)
            elif n_local > 0:
                negs = local_negs
            else:
                negs = global_negs
        else: # mix: half normal, half uniform (original behavior)
            y_min, y_max = y_true.min(), y_true.max()
            span = y_max - y_min
            lo = y_min - self.uniform_margin * span
            hi = y_max + self.uniform_margin * span
            negs = rng.normal(loc=y_true[:, None], scale=self.neg_std, size=(n, self.k_neg//2))
            negs += rng.normal(scale=1e-6, size=negs.shape) # break ties
            negs = np.concatenate([negs, rng.uniform(lo, hi, size=(n, self.k_neg//2))], axis=1)
        Yc[:, 1:] = negs
        return Yc

    def _calculate_plausibility_scores(self, Yc_scaled: np.ndarray) -> np.ndarray:
        """
        Calculates plausibility scores based on distances in SCALED y-space.
        Soft targets are limited at y_train min/max bounds (in scaled space).
        Only used when negative_type="soft".
        """
        scores = np.ones(Yc_scaled.shape, dtype=float)
        if Yc_scaled.shape[1] <= 1: # No negatives
            return scores.ravel()

        y_true_scaled = Yc_scaled[:, 0]
        y_negs_scaled = Yc_scaled[:, 1:]

        z_scores = (y_negs_scaled - y_true_scaled[:, None]) / self.soft_label_std
        cdf_vals = norm.cdf(z_scores)

        soft_labels_neg = 2 * np.minimum(cdf_vals, 1 - cdf_vals)

        if self.y_train_bounds_ is not None:
            y_min_scaled, y_max_scaled = self.y_train_bounds_
            outside_bounds = (y_negs_scaled < y_min_scaled) | (y_negs_scaled > y_max_scaled)
            soft_labels_neg[outside_bounds] = 0.0

        scores[:, 1:] = soft_labels_neg
        return scores.ravel()

    def _scores(self, X_phi):
        """
        Calculate raw scores (logits) from the fitted LGBM model.
        For hard negatives: logit from classifier
        For soft negatives: predicted logit from regressor
        """
        if self.lgbm_model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        if self.negative_type == "soft":
            # Regressor directly predicts the logit
            return self.lgbm_model_.predict(X_phi)
        else:
            # Classifier: get raw scores (logits)
            return self.lgbm_model_.predict(X_phi, raw_score=True)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the distributional regression model.
        
        Learns to score (x, y) pairs via contrastive learning:
        - Hard negatives: Binary labels (true=1, negatives=0)
        - Soft negatives: Continuous plausibility scores
        
        Args:
            X: (n, d) training features
            y: (n,) training targets
            X_val, y_val: Optional validation data for early stopping
        
        Returns:
            self
        """
        rng = check_random_state(self.random_state)
        y = np.asarray(y).ravel().astype(float)

        # 1. Split X, scale numerical features, and setup indices
        Xs_num, Xs_cat = self._prepare_and_scale_X(X, is_fitting=True)

        # 2. Scale TARGETS (different for hard vs soft)
        if self.negative_type == "soft":
            # Use z-score normalization for soft negatives
            self.y_mean_ = y.mean()
            self.y_std_ = y.std()
            if self.y_std_ < 1e-8: 
                self.y_std_ = 1.0
            y_s = (y - self.y_mean_) / self.y_std_
        else:
            # Use min-max scaling for hard negatives
            self.y_min_ = y.min()
            self.y_max_ = y.max()
            span = self.y_max_ - self.y_min_
            if span < 1e-8:
                span = 1.0
            y_s = (y - self.y_min_) / span
        
        self.y_train_bounds_ = (float(y_s.min()), float(y_s.max()))

        # 3. Build training data
        C = self.k_neg + 1
        Yc_train_s = self._build_candidates(y_s, rng)
        Xs_train_rep_num = np.repeat(Xs_num, C, axis=0)
        Xs_train_rep_cat = np.repeat(Xs_cat, C, axis=0)
        Phi_train = self._phi(Xs_train_rep_num, Xs_train_rep_cat, Yc_train_s.ravel())

        # Create targets based on negative_type
        if self.negative_type == "soft":
            # Soft negatives: plausibility scores -> logit
            plausibility_train = self._calculate_plausibility_scores(Yc_train_s)
            # Clip to avoid inf/-inf in logit (handles 0.0 and 1.0 values)
            plausibility_train = np.clip(plausibility_train, 1e-6, 1 - 1e-6)
            target_train = logit(plausibility_train)
        else:
            # Hard negatives: binary labels (1 for true, 0 for negatives)
            labels_train = np.zeros(Yc_train_s.shape, dtype=int)
            labels_train[:, 0] = 1
            target_train = labels_train.ravel()

        # 4. Build validation data (if provided)
        eval_set = []
        callbacks = []
        if X_val is not None and y_val is not None:
            y_val = np.asarray(y_val).ravel().astype(float)
            Xs_val_num, Xs_val_cat = self._prepare_and_scale_X(X_val, is_fitting=False)
            
            # Scale validation y using the same scaling as training
            if self.negative_type == "soft":
                y_val_s = (y_val - self.y_mean_) / self.y_std_
            else:
                span = self.y_max_ - self.y_min_
                if span < 1e-8:
                    span = 1.0
                y_val_s = (y_val - self.y_min_) / span

            val_rng = check_random_state(rng.randint(0, 10**9))
            Yc_val_s = self._build_candidates(y_val_s, val_rng)
            Xs_val_rep_num = np.repeat(Xs_val_num, C, axis=0)
            Xs_val_rep_cat = np.repeat(Xs_val_cat, C, axis=0)
            Phi_val = self._phi(Xs_val_rep_num, Xs_val_rep_cat, Yc_val_s.ravel())

            # Create validation targets based on negative_type
            if self.negative_type == "soft":
                plausibility_val = self._calculate_plausibility_scores(Yc_val_s)
                # Clip to avoid inf/-inf in logit (handles 0.0 and 1.0 values)
                plausibility_val = np.clip(plausibility_val, 1e-6, 1 - 1e-6)
                target_val = logit(plausibility_val)
            else:
                labels_val = np.zeros(Yc_val_s.shape, dtype=int)
                labels_val[:, 0] = 1
                target_val = labels_val.ravel()

            eval_set = [(Phi_val, target_val)]
            if self.early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(
                    self.early_stopping_rounds, verbose=self.verbose > 0
                ))

        # 5. Configure and train the LGBM model (classifier or regressor)
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_samples_leaf,
            'bagging_fraction': self.subsample if self.subsample < 1.0 else 1.0,
            'feature_fraction': self.colsample if self.colsample < 1.0 else 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'bagging_freq': 1 if self.subsample < 1.0 else 0,
        }
        if self.lgbm_params:
            params.update(self.lgbm_params)

        # Choose model type based on negative_type
        if self.negative_type == "soft":
            # For soft negatives, use objective regression_l2 explicitly
            params['objective'] = 'regression_l2'
            self.lgbm_model_ = lgb.LGBMRegressor(**params)
            eval_metric = 'l2'
        else:
            # For hard negatives, use classifier
            self.lgbm_model_ = lgb.LGBMClassifier(**params)
            eval_metric = 'logloss'
        
        lgbm_verbose_level = 10 if self.verbose > 0 else -1
        callbacks.append(lgb.log_evaluation(period=lgbm_verbose_level))

        self.lgbm_model_.fit(
            Phi_train,
            target_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            categorical_feature=self.lgbm_cat_idx_,
            callbacks=callbacks
        )

        self.evals_result_ = self.lgbm_model_.evals_result_
        self.best_iteration_ = self.lgbm_model_.best_iteration_
        return self

    # ------------------------ public API ------------------------

    def _compute_scores(self, X, y):
        """
        Internal method: compute scores for (x, y) pairs.
        
        Args:
            X: (n, d) feature array
            y: (n,) or (n, C) target values
            
        Returns:
            scores: (n, C) array of scores (higher = more plausible)
        """
        if self.num_x_scaler_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        Xs_num, Xs_cat = self._prepare_and_scale_X(X, is_fitting=False)
        y_unscaled = np.asarray(y, dtype=float)
        if y_unscaled.ndim == 0:
            y_unscaled = np.full((Xs_num.shape[0],), y_unscaled)
        y_unscaled = y_unscaled.reshape(Xs_num.shape[0], -1)

        # Scale y using the appropriate method
        if self.negative_type == "soft":
            y_s = (y_unscaled - self.y_mean_) / self.y_std_
        else:
            span = self.y_max_ - self.y_min_
            if span < 1e-8:
                span = 1.0
            y_s = (y_unscaled - self.y_min_) / span

        n, C = y_s.shape
        X_rep_num = np.repeat(Xs_num, C, axis=0)
        X_rep_cat = np.repeat(Xs_cat, C, axis=0)

        Phi = self._phi(X_rep_num, X_rep_cat, y_s.reshape(-1))
        scores = self._scores(Phi).reshape(n, C)
        return scores

    def predict_distribution(self, X, y_grid: np.ndarray):
        """
        Predict probability distribution p(y|x) over a grid of y values.

        Args:
            X: (n, d) feature array
            y_grid: (G,) array of y-values to evaluate

        Returns:
            probs: (n, G) array of probabilities (each row sums to 1)
        """
        y_grid = np.asarray(y_grid, dtype=float).ravel()
        n = _ensure_2d(X).shape[0]

        Y_grid_tiled = np.tile(y_grid, (n, 1))
        scores = self._compute_scores(X, Y_grid_tiled)
        return _softmax_stable(scores, axis=1)

    def predict_expected_value(self, X, y_grid: np.ndarray):
        """Calculate the expected value of y from the predicted distribution."""
        probs = self.predict_distribution(X, y_grid)
        return (probs * y_grid[None, :]).sum(axis=1)

    def predict_proba_answer(self, X, y_star: np.ndarray, y_grid: np.ndarray, bin_width: Optional[float]=None):
        """
        Calculate the probability that the proposed answer y_star is correct.

        This is approximated by the probability mass at or around y_star on the y_grid.

        Args:
            X: (n, d) feature array.
            y_star: (n,) array of proposed target values.
            y_grid: (G,) array of y-values over which the distribution is defined.
            bin_width: If provided, sum the mass for grid points within
                       y_star +/- bin_width/2. Otherwise, use the nearest grid point.
        """
        y_star = np.asarray(y_star).ravel()
        probs = self.predict_distribution(X, y_grid)

        if bin_width is None:
            # Find the nearest grid point for each y_star
            idx = np.abs(y_grid[None, :] - y_star[:, None]).argmin(axis=1)
            return probs[np.arange(len(y_star)), idx]
        else:
            # Sum probability mass within the bin
            half = bin_width / 2.0
            mask = (y_grid[None, :] >= y_star[:, None] - half) & (y_grid[None, :] <= y_star[:, None] + half)
            return np.sum(probs * mask, axis=1)

    def _create_default_grid(self, n_points: int = 1000, margin: float = 0.2):
        """
        Create a default y-grid based on training data bounds.
        
        Args:
            n_points: Number of grid points
            margin: Extra margin beyond training range (as fraction of range)
        
        Returns:
            y_grid: (n_points,) array
        """
        if self.negative_type == "soft":
            # For soft negatives, use y_mean_ and y_std_
            y_min = self.y_mean_ - 4 * self.y_std_
            y_max = self.y_mean_ + 4 * self.y_std_
        else:
            # For hard negatives, use y_min_ and y_max_
            span = self.y_max_ - self.y_min_
            y_min = self.y_min_ - margin * span
            y_max = self.y_max_ + margin * span
        
        return np.linspace(y_min, y_max, n_points)

    def predict(self, X, method: Literal["mean", "median", "mode"] = "mean", y_grid: Optional[np.ndarray] = None):
        """
        Point predictions from the predicted distribution.
        
        Args:
            X: (n, d) feature array
            method: "mean" for expected value, "median" for 50th percentile, "mode" for argmax
            y_grid: Optional y-grid. If None, creates a default grid.
        
        Returns:
            (n,) array of predictions
        """
        if y_grid is None:
            y_grid = self._create_default_grid()
        
        if method == "mean":
            return self.predict_expected_value(X, y_grid)
        elif method == "median":
            return self.predict_quantiles(X, qs=[0.5], y_grid=y_grid)[:, 0]
        elif method == "mode":
            probs = self.predict_distribution(X, y_grid)
            mode_idx = np.argmax(probs, axis=1)
            return y_grid[mode_idx]
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'mean', 'median', or 'mode'.")

    def predict_quantiles(self, X, qs: Union[List[float], np.ndarray] = [0.05, 0.5, 0.95], 
                         y_grid: Optional[np.ndarray] = None):
        """
        Predict quantiles of the conditional distribution p(y|x).
        
        Args:
            X: (n, d) feature array
            qs: List or array of quantile levels in [0, 1]
            y_grid: Optional y-grid. If None, creates a default grid.
        
        Returns:
            (n, len(qs)) array of quantile predictions
        """
        if y_grid is None:
            y_grid = self._create_default_grid()
        
        qs = np.asarray(qs).ravel()
        probs = self.predict_distribution(X, y_grid)
        
        # Compute CDF
        cdf = np.cumsum(probs, axis=1)
        
        # Find quantiles by interpolating the inverse CDF
        n = probs.shape[0]
        quantiles = np.zeros((n, len(qs)))
        
        for i, q in enumerate(qs):
            for j in range(n):
                # Find the first index where CDF >= q
                idx = np.searchsorted(cdf[j], q)
                if idx == 0:
                    quantiles[j, i] = y_grid[0]
                elif idx >= len(y_grid):
                    quantiles[j, i] = y_grid[-1]
                else:
                    # Linear interpolation between grid points
                    if cdf[j, idx-1] == cdf[j, idx]:
                        quantiles[j, i] = y_grid[idx]
                    else:
                        alpha = (q - cdf[j, idx-1]) / (cdf[j, idx] - cdf[j, idx-1])
                        quantiles[j, i] = y_grid[idx-1] + alpha * (y_grid[idx] - y_grid[idx-1])
        
        return quantiles

    def predict_interval(self, X, alpha: float = 0.1, y_grid: Optional[np.ndarray] = None):
        """
        Predict a (1-alpha) prediction interval.
        
        Args:
            X: (n, d) feature array
            alpha: Significance level (e.g., 0.1 for 90% interval)
            y_grid: Optional y-grid. If None, creates a default grid.
        
        Returns:
            lower: (n,) array of lower bounds
            upper: (n,) array of upper bounds
        """
        qs = [alpha / 2, 1 - alpha / 2]
        intervals = self.predict_quantiles(X, qs=qs, y_grid=y_grid)
        return intervals[:, 0], intervals[:, 1]

    def predict_cdf(self, X, y_grid: Optional[np.ndarray] = None):
        """
        Predict the cumulative distribution function (CDF) values.
        
        Args:
            X: (n, d) feature array
            y_grid: Optional y-grid. If None, creates a default grid.
        
        Returns:
            cdf: (n, len(y_grid)) array of CDF values
            y_grid: The y_grid used
        """
        if y_grid is None:
            y_grid = self._create_default_grid()
        
        probs = self.predict_distribution(X, y_grid)
        cdf = np.cumsum(probs, axis=1)
        return cdf, y_grid

    def predict_pdf(self, X, y_grid: Optional[np.ndarray] = None):
        """
        Predict the probability density/mass function (PDF/PMF) values.
        This is an alias for predict_distribution for consistency.
        
        Args:
            X: (n, d) feature array
            y_grid: Optional y-grid. If None, creates a default grid.
        
        Returns:
            pdf: (n, len(y_grid)) array of probability densities
            y_grid: The y_grid used
        """
        if y_grid is None:
            y_grid = self._create_default_grid()
        
        pdf = self.predict_distribution(X, y_grid)
        return pdf, y_grid

    def sample_y(self, X, n_samples: int = 1000, y_grid: Optional[np.ndarray] = None, 
                 random_state: Optional[int] = None):
        """
        Sample from the conditional distribution p(y|x).
        
        Args:
            X: (n, d) feature array
            n_samples: Number of samples to draw for each input
            y_grid: Optional y-grid. If None, creates a default grid.
            random_state: Random seed for reproducibility
        
        Returns:
            samples: (n, n_samples) array of sampled y values
        """
        if y_grid is None:
            y_grid = self._create_default_grid()
        
        rng = check_random_state(random_state)
        probs = self.predict_distribution(X, y_grid)
        
        n = probs.shape[0]
        samples = np.zeros((n, n_samples))
        
        for i in range(n):
            # Sample indices according to the probability distribution
            sampled_indices = rng.choice(len(y_grid), size=n_samples, p=probs[i])
            samples[i] = y_grid[sampled_indices]
        
        return samples