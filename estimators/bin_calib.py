import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def find_nearest_index(arr, X):
    sorted_arr = np.sort(arr)
    absolute_diff = np.abs(sorted_arr - X)
    nearest_index = np.argmin(absolute_diff)
    return nearest_index

class Bin_calib():

    def __init__(self, bins=30):

        self.bins = bins

    def fit(self, X, y, xc, yc): # X,y are calibration or test dataset - X is prob from x_train/x_calib after passing through RF
        self.prob_true, self.prob_pred = calibration_curve(y, X, n_bins=self.bins)
        # print("prob_true", self.prob_true)
        # print("prob_pred", self.prob_pred)
        # prob_true_c, prob_pred_c = calibration_curve(yc, xc, n_bins=self.bins)
        # plt.scatter(self.prob_true, self.prob_pred)
        # plt.scatter(prob_true_c, prob_pred_c)
        # plt.show()
        return self

    def predict(self, X):
        calib_prob = []
        for x in X:
            index = find_nearest_index(self.prob_pred, x)
            calib_prob.append(self.prob_true[index])
        calib_prob = np.array(calib_prob)
        return calib_prob