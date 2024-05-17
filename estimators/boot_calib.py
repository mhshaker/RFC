from estimators.IR_RF_estimator import IR_RF
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

def find_nearest_index(arr, X):
    sorted_arr = np.sort(arr)
    absolute_diff = np.abs(sorted_arr - X)
    nearest_index = np.argmin(absolute_diff)
    return nearest_index


class Boot_calib():

    def __init__(self, bootstrap_size=100, boot_count=100, bins=30):

        self.bootstrap_size = bootstrap_size
        self.boot_count = boot_count
        self.bins = bins

    def fit(self, x_train, y_train, model):
        ens = []
        p_all = np.zeros(1)
        y_all = np.zeros(1)
        for boot_index in range(self.boot_count):
            model.random_state = boot_index * 100 # change the random seed to fit again
            # model = IR_RF(n_estimators=10  , oob_score=False, max_depth= 6, random_state=boot_index) # changing the RF params
            model.fit(x_train, y_train)
            p = model.predict_proba(x_train)
            if len(p_all) == 0:
                p_all = p
                y_all = y_train
            else:
                p_all = np.concatenate((p[:,1], p_all))
                y_all = np.concatenate((y_train, y_all))
            ens.append(p.copy())

        bin = int(len(p_all)/20)
        if bin > 100:
            bin = 100
        if bin < 10:
            bin = 10
        self.prob_true, self.prob_pred = calibration_curve(y_all, p_all, n_bins=bin)
        return self

    def predict_boot(self, X):

        b = []
        for boot_index in range(self.boot_count):
            tree_index = np.random.randint(low=0, high=X.shape[1], size=self.bootstrap_size)
            boot = X[:,tree_index,:]
            b.append(boot.copy())
        b = np.array(b)
        b = np.mean(b, axis=2) # average trees in each bootstrap
        b = np.mean(b, axis=0) # average all the bootstraps

        return b

    def predict_ens(self, x_test, x_train, y_train, model, param_change=False):

        ens = []
        params = model.get_params().copy()

        depth_extention = np.random.randint(low=-1, high=1, size=self.boot_count)

        for boot_index, de in zip(range(self.boot_count), depth_extention):
            params["random_state"] = boot_index * 100
            if param_change:
                params['n_estimators'] += de
            rf_ens = IR_RF(**params).fit(x_train, y_train)
            p = rf_ens.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)
        b = np.mean(ens, axis=0) # average all the ens

        return b

    def predict_ens_params(self, x_test, x_train, y_train, hyper_params, seed):

        ens = []

        for i, params in enumerate(hyper_params):
            rf_ens = IR_RF(**params, random_state=i*100).fit(x_train, y_train)
            p = rf_ens.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)
        b = np.mean(ens, axis=0) # average all the ens

        return b



    def predict_ens_boot(self, x_test, x_train, y_train, model, param_change=False):

        ens = []
        params = model.get_params().copy()
        params_org = model.get_params().copy()

        depth_extention = np.random.randint(low=-1, high=1, size=self.boot_count)

        for boot_index, de in zip(range(self.boot_count), depth_extention):
            model.random_state = boot_index * 100 # change the random seed to fit again
            if param_change:
                model.max_depth = params['max_depth'] + de
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)

        RF_index = np.random.randint(low=0, high=ens.shape[0], size=self.bootstrap_size)
        b = ens[RF_index,:,:]
        b = np.mean(b, axis=0) # average all the bootstraps

        model.set_params(**params_org)

        return b



    def true_prob_ens(self, x_test, y_test, x_train, y_train, model, param_change=False):

        ens = []
        params = model.get_params().copy()
        params_org = model.get_params().copy()

        depth_extention = np.random.randint(low=-1, high=1, size=self.boot_count)

        for boot_index, de in zip(range(self.boot_count), depth_extention):
            model.random_state = boot_index * 100 # change the random seed to fit again
            if param_change:
                model.max_depth = params['max_depth'] + de
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)

        true_prob = []
        for i, y in enumerate(y_test):
            ens_prob_single_point = ens[:,i,:]
            ens_decision_single_point = np.argmax(ens_prob_single_point,axis=1)
            acc = accuracy_score(np.repeat(y, len(ens_decision_single_point)), ens_decision_single_point)
            true_prob.append(acc)
            # print("true prob ", ens_prob_single_point)
            # print("true deci ", ens_decision_single_point)
            # print("y_test", np.repeat(y, len(ens_decision_single_point)))
            # print("acc ", acc)
            # print("---------------------------------")
        true_prob = np.array(true_prob)
        # print("true_prob ", true_prob)

        model.set_params(**params_org)

        return true_prob



    def predict_largeRF(self, x_test, x_train, y_train, model):

        params = model.get_params().copy()
        params['n_estimators'] = params['n_estimators'] * self.boot_count

        rf_l = IR_RF(**params).fit(x_train, y_train)
        p = rf_l.predict_proba(x_test)

        return p


    def predict_ensbin(self, x_test, x_train, y_train, model):

        ens = []
        for boot_index in range(self.boot_count):
            model.random_state = boot_index * 100 # change the random seed to fit again
            # model = IR_RF(n_estimators=10  , oob_score=False, max_depth= 6, random_state=boot_index) # changing the RF params
            model.fit(x_train, y_train)
            p = model.predict_proba(x_test)
            ens.append(p.copy())
        ens = np.array(ens)
        b = np.mean(ens, axis=0) # average all the ens

        return b

    def predict_ens2(self, X):
        calib_prob = []
        for x in X:
            index = find_nearest_index(self.prob_pred, x)
            if index + 1 < len(self.prob_pred):
                p = (self.prob_true[index] + self.prob_true[index+1]) / 2
            else:
                p = self.prob_true[index]
            calib_prob.append(p)
        calib_prob = np.array(calib_prob)
        return calib_prob