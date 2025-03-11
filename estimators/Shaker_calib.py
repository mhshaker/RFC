import numpy as np
from scipy.stats import beta
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import brier_score_loss
from skopt import gp_minimize
from skopt.space import Real
from joblib import Parallel, delayed


def convert_prob_2D(prob1D):
    prob1D = np.clip(prob1D, 0, 1)  # Ensure valid probability range
    prob_second_class = 1.0 - prob1D
    return np.column_stack((prob_second_class, prob1D))

class Shaker_calib(BaseEstimator, RegressorMixin):
    def __init__(self, initial_noise_level=0.1, noise_sample=1000, global_noise=True, n_jobs=-1, seed=0):
        """
        Parameters:
        - initial_noise_level: Initial noise level for optimization.
        - noise_sample: Number of noise samples for averaging predictions.
        - global_noise: If True, use a single noise level for all features. Otherwise, learn per-feature noise levels.
        - n_jobs: Number of parallel jobs for computation (default: use all available cores).
        """
        self.initial_noise_level = initial_noise_level
        self.noise_sample = noise_sample
        self.global_noise = global_noise
        self.n_jobs = n_jobs
        self.noise_levels_ = None
        self.a_ = None
        self.b_ = None
        self.seed = seed
        self.forward = False
    
    def _add_noise(self, X, noise_levels):
        """Adds Gaussian noise to X using either global or per-feature noise levels."""
        return X + np.random.normal(0, noise_levels, X.shape)
    
    def _get_noise_preds(self, X, model, noise_levels):
        """Generates noisy predictions by averaging over multiple noise samples in parallel."""
        
        def single_sample_prediction(_):
            return model.predict_proba(self._add_noise(X, noise_levels))
        
        ns_predictions = Parallel(n_jobs=self.n_jobs)(delayed(single_sample_prediction)(i) for i in range(self.noise_sample))
        
        return np.mean(np.stack(ns_predictions), axis=0)
    
    def _beta_transform(self, p, a, b):
        """Apply Beta calibration transformation."""
        return beta.cdf(p, a, b)
    
    def _determine_optimization_params(self, n_features):
        """Dynamically determine the number of optimization calls based on feature count."""
        base_calls = 50
        base_random_starts = 20
        
        if self.global_noise:
            return base_calls, base_random_starts  # Global noise is 1D, so no need for more calls
        
        # Scale optimization calls based on the number of features
        n_calls = min(max(base_calls, 10 * n_features), 300)
        n_random_starts = min(max(base_random_starts, n_features), 100)
        
        return n_calls, n_random_starts
    
    def _optimize_noise_levels(self, X, y, model):
        """Optimize noise levels using Bayesian Optimization."""
        n_features = X.shape[1]
        n_calls, n_random_starts = self._determine_optimization_params(n_features)

        def objective(params):
            noise_levels = params[0] if self.global_noise else np.array(params)
            ns_predictions_calib = self._get_noise_preds(X, model, noise_levels)
            return brier_score_loss(y, ns_predictions_calib[:, 1])
        
        space = [Real(0.001, 1, "log-uniform")] if self.global_noise else [Real(0.001, 1, "log-uniform") for _ in range(X.shape[1])]
        
        result = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_random_starts, random_state=self.seed)
        
        self.noise_levels_ = result.x[0] if self.global_noise else np.array(result.x)
    
    def _optimize_beta_calibration(self, X, y, model):
        """Optimize Beta calibration parameters using Bayesian Optimization."""
        
        def objective(params):
            a, b = params
            ns_predictions_calib = self._get_noise_preds(X, model, self.noise_levels_)
            p_calibrated = self._beta_transform(ns_predictions_calib[:, 1], a, b)
            return brier_score_loss(y, p_calibrated)
        
        space = [Real(0.5, 5, "uniform"), Real(0.5, 5, "uniform")]
        
        result = gp_minimize(objective, space, n_calls=30, random_state=self.seed)
        
        self.a_, self.b_ = result.x
    
    def fit(self, X, y, model):
        """Optimize noise levels first, then optimize Beta calibration parameters."""
        self._optimize_noise_levels(X, y, model)
        self._optimize_beta_calibration(X, y, model)

        p_calib = self.predict(X, model)
        p_model = model.predict_proba(X)

        bs_calib = brier_score_loss(y, p_calib[:,1])
        bs_model = brier_score_loss(y, p_model[:,1])

        if bs_calib >= bs_model:
            self.forward = True

        return self
    
    def predict(self, X, model):
        if self.forward:
            return model.predict_proba(X)
        else:
            """Generate calibrated predictions using the learned noise levels and Beta calibration."""
            if self.noise_levels_ is None or self.a_ is None or self.b_ is None:
                raise ValueError("Model has not been fitted yet.")
            ns_predictions_calib = self._get_noise_preds(X, model, self.noise_levels_)
            return convert_prob_2D(self._beta_transform(ns_predictions_calib[:, 1], self.a_, self.b_))
