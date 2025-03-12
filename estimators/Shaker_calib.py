import numpy as np
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
    def __init__(self, initial_noise_level=0.1, noise_sample=1000, n_jobs=-1, seed=0):
        """
        Parameters:
        - initial_noise_level: Initial noise level for optimization.
        - noise_sample: Number of noise samples for averaging predictions.
        - n_jobs: Number of parallel jobs for computation (default: use all available cores).
        """
        self.initial_noise_level = initial_noise_level
        self.noise_sample = noise_sample
        self.n_jobs = n_jobs
        self.noise_level_ = None
        self.r_ = None
        self.seed = seed
        self.forward = False
    
    def _add_noise(self, X, noise_level):
        """Adds Gaussian noise to X using a single global noise level."""
        return X + np.random.normal(0, noise_level, X.shape)
    
    def _get_noise_preds(self, X, model, noise_level):
        """Generates noisy predictions by averaging over multiple noise samples in parallel."""
        
        def single_sample_prediction(_):
            return model.predict_proba(self._add_noise(X, noise_level))
        
        ns_predictions = Parallel(n_jobs=self.n_jobs)(delayed(single_sample_prediction)(i) for i in range(self.noise_sample))
        
        return np.mean(np.stack(ns_predictions), axis=0)
    
    def _transform_probs(self, p, r):
        """Apply the transformation p^r / (p^r + (1 - p)^r)"""
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid numerical issues
        p_r = np.power(p, r)
        q_r = np.power(1 - p, r)
        return p_r / (p_r + q_r)
    
    def _optimize_noise_level(self, X, y, model):
        """Optimize the global noise level using Bayesian Optimization."""
        
        def objective(params):
            noise_level = params[0]
            ns_predictions_calib = self._get_noise_preds(X, model, noise_level)
            return brier_score_loss(y, ns_predictions_calib[:, 1])
        
        space = [Real(0.001, 1, "log-uniform")]
        
        result = gp_minimize(objective, space, n_calls=50, n_initial_points=20, random_state=self.seed)
        
        self.noise_level_ = result.x[0]
    
    def _optimize_r(self, X, y, model):
        """Optimize the parameter r using Bayesian Optimization."""
        
        def objective(params):
            r = params[0]
            ns_predictions_calib = self._get_noise_preds(X, model, self.noise_level_)
            p_calibrated = self._transform_probs(ns_predictions_calib[:, 1], r)
            return brier_score_loss(y, p_calibrated)
        
        space = [Real(0.5, 5, "uniform")]
        
        result = gp_minimize(objective, space, n_calls=30, random_state=self.seed)
        
        self.r_ = result.x[0]
    
    def fit(self, X, y, model):
        """Optimize noise level first, then optimize the r parameter."""
        self._optimize_noise_level(X, y, model)
        self._optimize_r(X, y, model)

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
            """Generate calibrated predictions using the learned noise level and r parameter."""
            if self.noise_level_ is None or self.r_ is None:
                raise ValueError("Model has not been fitted yet.")
            ns_predictions_calib = self._get_noise_preds(X, model, self.noise_level_)
            return convert_prob_2D(self._transform_probs(ns_predictions_calib[:, 1], self.r_))
