import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.calibration import _SigmoidCalibration
from old.CalibrationM import convert_prob_2D
from sklearn.metrics import brier_score_loss

class CRF_calib(BaseEstimator, ClassifierMixin):
    
    def __init__(self, step=1, learning_method='sig_brior'):

        self.step = step
        self.learning_method = learning_method


    def fit(self, X, y): # X,y are calibration dataset

        # [Goal] set the value of "r"
        # learn _SigmoidCalibration "a" and "b"
        # optimize _SigmoidCalibration with brier_score_loss (from sklearn.metrics import brier_score_loss)
        if self.learning_method == 'brier_opt':
            low = 0
            up = 1
            r_list = np.arange(low, up + self.step, self.step)

            r_opt = 0
            min_brier_score = 10000000
            for r in r_list:
                y_p = self.predict_training(X,r)
                brier_score = brier_score_loss(y, y_p[:,1])
                if brier_score < min_brier_score:
                    min_brier_score = brier_score
                    r_opt = r
            self.r = r_opt

        elif self.learning_method == "sig":
            self.sig = _SigmoidCalibration().fit(X, y)
        elif self.learning_method == "sig_brior":
            low = 0
            up = 50
            a_list = np.arange(low, up + self.step, self.step)
            b_list = np.arange(low, up + self.step, self.step)

            a_opt = 0
            b_opt = 0
            min_brier_score = 10000000
            for a in a_list:
                for b in b_list:
                    r = 1/(1 + np.exp(a*X + b))
                    y_p = self.predict_training(X,r)
                    brier_score = brier_score_loss(y, y_p[:,1])
                    if brier_score < min_brier_score:
                        min_brier_score = brier_score
                        a_opt = a
                        b_opt = b
            self.a = a_opt
            self.b = b_opt

        
        return self

    def predict_training(self, X, r):
        X_2d = convert_prob_2D(X)        

        max_idx = np.argmax(X_2d, axis=1)
        min_idx = np.argmin(X_2d, axis=1)

        max_indices = list(range(len(X_2d))), max_idx
        min_indices = list(range(len(X_2d))), min_idx

        X_2d[max_indices] += r * (1- X_2d[max_indices])
        X_2d[min_indices] *= (1-r)

        y = X_2d

        return y

    def predict(self, X):

        # print("X", X)

        if self.learning_method == 'sig':
            r = self.sig.predict(X)
        elif self.learning_method == 'brier_opt':
            r = self.r
        elif self.learning_method == 'sig_brior':
            r = 1/(1 + np.exp(self.a * X + self.b))

        X_2d = convert_prob_2D(X)        

        max_idx = np.argmax(X_2d, axis=1)
        min_idx = np.argmin(X_2d, axis=1)

        neq_idx = (np.where(X_2d[:,0] != X_2d[:,1]))[0]

        # print("max_idx", max_idx)
        # print("min_idx", min_idx)

        max_idx = max_idx[neq_idx]
        min_idx = min_idx[neq_idx]

        # print("max_idx", max_idx)
        # print("min_idx", min_idx)

        max_indices = neq_idx, max_idx
        min_indices = neq_idx, min_idx

        # print("eq_idx", neq_idx)
        # print("max_indices", max_indices)
        # print("min_indices", min_indices)

        r = r[neq_idx]
        # print("r", r)

        X_2d[max_indices] += r * (1- X_2d[max_indices])
        X_2d[min_indices] *= (1-r)

        y = X_2d

        return y
