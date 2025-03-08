import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def convert_prob_2D(prob1D):
    prob_second_class = np.ones(len(prob1D)) - prob1D
    prob2D = np.concatenate((prob_second_class.reshape(-1,1), prob1D.reshape(-1,1)), axis=1)
    return prob2D


class Elkan_calib(BaseEstimator, ClassifierMixin):
    
    def fit(self, y_train, y_calib, class_to_calib=1):
        self.b_train =  len(np.where(y_train==class_to_calib)) / len(y_train)
        self.b_calib =  len(np.where(y_calib==class_to_calib)) / len(y_calib)
        return self


    def predict(self, X):

        elkan_p = (self.b_calib * X - self.b_train * self.b_calib * X) / (self.b_calib * X + self.b_train - self.b_train * X - self.b_train * self.b_calib)
        elkan_p = np.clip(elkan_p, 0, 1)
        probs = convert_prob_2D(elkan_p)

        return probs
