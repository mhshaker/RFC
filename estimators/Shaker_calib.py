import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import brier_score_loss
from skopt import gp_minimize
from skopt.space import Real
from joblib import Parallel, delayed

class Shaker_calib(BaseEstimator, RegressorMixin):
    def __init__(self, initial_r=1.0, initial_noise_level=0.1, noise_sample=1000, n_jobs=-1):
        self.initial_r = initial_r
        self.initial_noise_level = initial_noise_level
        self.noise_sample = noise_sample
        self.r_ = None
        self.noise_level_ = None
        self.n_jobs = n_jobs  # Number of parallel jobs
    
    def _transform_probs(self, p, r):
        """Apply the transformation p^r / (p^r + (1 - p)^r)."""
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid numerical issues
        p_r = np.power(p, r)
        return p_r / (p_r + np.power(1 - p, r))
    
    def _add_noise(self, X, noise_level):
        """Vectorized noise addition."""
        noise = np.random.normal(0.0, noise_level, size=(self.noise_sample,) + X.shape)
        return X[None, :, :] + noise  # Shape: (noise_sample, X.shape[0], X.shape[1])
    
    def _get_noise_preds(self, X, model, noise_level):
        """Generates noisy predictions in parallel."""
        X_noisy = self._add_noise(X, noise_level)  # Shape: (noise_sample, n_samples, n_features)

        def predict_single(noisy_X):
            return model.predict_proba(noisy_X)[:, 1]  # Extract probabilities for class 1

        ns_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_single)(X_noisy[i]) for i in range(self.noise_sample)
        )

        return np.mean(ns_predictions, axis=0)  # Shape: (n_samples,)

    def fit(self, X, y, model, n_calls=50):
        """Optimize noise_level and r using Bayesian Optimization."""
        
        def objective(params):
            noise_level, r = params
            ns_predictions_calib = self._get_noise_preds(X, model, noise_level)
            p_transformed = self._transform_probs(ns_predictions_calib, r)
            return brier_score_loss(y, p_transformed)

        space = [Real(0.001, 0.5, "log-uniform"), Real(0.1, 10, "log-uniform")]

        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        self.noise_level_, self.r_ = result.x
        return self

    def predict(self, X, model):
        """Transform probabilities using the learned r and noise level."""
        if self.r_ is None or self.noise_level_ is None:
            raise ValueError("Model has not been fitted yet.")
        ns_predictions_calib = self._get_noise_preds(X, model, self.noise_level_)
        return self._transform_probs(ns_predictions_calib, self.r_)