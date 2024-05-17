
### Impots
import time
import os
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from estimators.CRF_estimator import CRF_calib
from estimators.Elkan_estimator import Elkan_calib
from estimators.Venn_estimator import Venn_calib
from estimators.VA_estimator import VA_calib
from estimators.TLR_estimator import treeLR_calib
from estimators.boot_calib import Boot_calib
from estimators.bin_calib import Bin_calib
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from betacal import BetaCalibration
from sklearn.linear_model import LinearRegression
from scipy.stats import binned_statistic
from old.CalibrationM import confidance_ECE, convert_prob_2D
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import scipy
from sklearn.metrics import mutual_info_score

tvec = np.linspace(0.01, 0.99, 990)

def kl_divergence(p, q,epsilon=1e-10):
    """
    Calculate KL divergence between two arrays of probability distributions.
    """
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    
    kl_values = np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=1)
    return np.mean(kl_values)

def split_train_calib_test(name, X, y, test_size, calib_size, runs=1, tp=np.full(10,-1)):
    ### spliting data to train calib and test
    data_runs = []
    for orig_seed in range(runs):
        
        for i in range(1000, 1100): # the for loop is to make sure the calib train and test split all consist of both classes of the binary dataset
            data = {"name": name }
            seed = i + orig_seed
            x_train_calib, x_test, y_train_calib, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)
            x_train, x_calib, y_train, y_calib = train_test_split(x_train_calib, y_train_calib, test_size=calib_size, shuffle=True, random_state=seed)

            # data["test_index"] = np.where(np.isin(X, x_test))[0]
            data["X"] = X
            data["y"] = y
            data["x_train_calib"], data["x_test"] = x_train_calib, x_test
            data["y_train_calib"], data["y_test"] = y_train_calib, y_test
            data["x_train"], data["x_calib"] = x_train, x_calib
            data["y_train"], data["y_calib"] = y_train, y_calib


            if tp.sum() != -10:
                _, _, tp_train_calib, tp_test = train_test_split(X, tp, test_size=test_size, shuffle=True, random_state=seed)
                _, _, tp_train, tp_calib = train_test_split(x_train_calib, tp_train_calib, test_size=calib_size, shuffle=True, random_state=seed)
                data["tp"] = tp
                data["tp_train_calib"], data["tp_test"] = tp_train_calib, tp_test
                data["tp_train"], data["tp_calib"] = tp_train, tp_calib
            
            if len(np.unique(data["y_calib"])) > 1 and len(np.unique(data["y_test"])) > 1 and len(np.unique(data["y_train"])) > 1:
                break
        data_runs.append(data)

    return data_runs

def CV_split_train_calib_test(name, X, y, folds=10, runs=0, tp=np.full(10,-1)):
    
    data_runs = []

    for seed in range(runs): # by setting seed to the range of runs all folds of every CV with different seed will be in data_runs
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        for train_calib_index, test_index in skf.split(X, y):
            data = {"name": name }
            # data["test_index"] = test_index
            data["X"] = X
            data["y"] = y
            data["x_train_calib"], data["x_test"] = X[train_calib_index], X[test_index]
            data["y_train_calib"], data["y_test"] = y[train_calib_index], y[test_index]
            if tp.sum() != -10:
                data["tp"] = tp
                data["tp_train_calib"], data["tp_test"] = tp[train_calib_index], tp[test_index]

            skf2 = StratifiedKFold(n_splits=folds-1, shuffle=True, random_state=seed)
            train_index, calib_index = next(skf2.split(data["x_train_calib"], data["y_train_calib"]))
            data["x_train"], data["x_calib"] = data["x_train_calib"][train_index], data["x_train_calib"][calib_index]
            data["y_train"], data["y_calib"] = data["y_train_calib"][train_index], data["y_train_calib"][calib_index]
            if tp.sum() != -10:
                data["tp_train"], data["tp_calib"] = data["tp_train_calib"][train_index], data["tp_train_calib"][calib_index]
            
            data_runs.append(data) # data is one fold of the CV

    return data_runs


def calibration(data, params, seed=0):
    data_name = data["name"]
    calib_methods = params["calib_methods"] 
    metrics = params["metrics"]
    # the retuen is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
    # the structure of the keys in the dict is data_calibMethod_metric
    results_dict = {}

    # train model - hyper opt
    time_rf_opt_calib_s = time.time()
    rf = IR_RF(random_state=seed)
    RS = RandomizedSearchCV(rf, params["search_space"], scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
    if params["oob"] == False:
        RS.fit(data["x_train"], data["y_train"])
    else:
        RS.fit(data["x_train_calib"], data["y_train_calib"])
    time_rf_opt_calib = time.time() - time_rf_opt_calib_s

    # RF = rf_best
    RF = RS.best_estimator_

    # get main random forest calibration probs
    if params["oob"] == False:
        rf_p_calib = RF.predict_proba(data["x_calib"], params["laplace"])
        y_p_calib = data["y_calib"]
    else:
        rf_p_calib = RF.oob_decision_function_
        y_p_calib = data["y_train_calib"]

    # get test probs from main RF - to later be used as input for calibration methods to predict
    rf_p_test = RF.predict_proba(data["x_test"], params["laplace"])

    results_dict[data["name"] + "_RF_prob"] = rf_p_test
    if "CL" in metrics:
        results_dict[data["name"] + "_RF_prob_c"] = RF.predict_proba(data["X"], params["laplace"]) # prob c is on all X data
    # results_dict[data["name"] + "_RF_prob_calib"] = rf_p_calib


    # all input probs to get the fit calib model

    ### full data (train + calib)
    method = "RF_d"
    if method in calib_methods:
        time_rf_d_s = time.time()
        rf_d = IR_RF(n_estimators=params["search_space"]["n_estimators"][0], random_state=seed).fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rf_d_s

        rf_d_p_test = rf_d.predict_proba(data["x_test"], params["laplace"])
        results_dict[f"{data_name}_{method}_prob"] = rf_d_p_test
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = rf_d.predict_proba(data["X"], params["laplace"])

    method = "RF_large"
    if method in calib_methods:
        time_rfl_s = time.time()
        # RF_large_p_test_fd = bc.predict_largeRF(data["x_test"], data["x_train_calib"], data["y_train_calib"], RF)

        # best_rf_params = RF.get_params().copy()
        # best_rf_params['n_estimators'] = best_rf_params['n_estimators'] * params["boot_count"]

        rf_l = IR_RF(n_estimators=params["search_space"]["n_estimators"][0]* params["boot_count"], random_state=seed).fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rfl_s

        # rf_l = IR_RF(**best_rf_params).fit(data["x_train_calib"], data["y_train_calib"])
        RF_large_p_test_fd = rf_l.predict_proba(data["x_test"], params["laplace"])

        results_dict[f"{data_name}_{method}_prob"] = RF_large_p_test_fd
        if "CL" in metrics:
            # results_dict[f"{data_name}_{method}_prob_c"] = bc.predict_largeRF(data["X"], data["x_train_calib"], data["y_train_calib"], RF)
            results_dict[f"{data_name}_{method}_prob_c"] = rf_l.predict_proba(data["X"], params["laplace"])

    method = "RF_opt"
    if method in calib_methods:
        time_rf_opt_s = time.time()
        # train model - hyper opt with x_train_calib
        RS.fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rf_opt_s
        RF_opt = RS.best_estimator_

        rff_p_test = RF_opt.predict_proba(data["x_test"]) # params["laplace"]
        results_dict[f"{data_name}_{method}_prob"] = rff_p_test
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = RF_opt.predict_proba(data["X"]) # params["laplace"] 
    

    method = "RF_opt_CT"
    if method in calib_methods:
        time_rf_ct = time.time()

        ct_param_space = params["search_space"].copy()
        ct_param_space["min_samples_leaf"] = params["curt_v"]

        rf = IR_RF(random_state=seed)
        RS_ct_2 = RandomizedSearchCV(rf, ct_param_space, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS_ct_2.fit(data["x_train_calib"], data["y_train_calib"])
        RF_ct = RS_ct_2.best_estimator_

        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rf_ct
        rfct_p_test = RF_ct.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = rfct_p_test
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = RF_ct.predict_proba(data["X"]) 




    method = "Platt"
    if method in calib_methods:
        time_platt_s = time.time()
        plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_platt_s + time_rf_opt_calib

        plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = plat_p_test
        results_dict[f"{data_name}_{method}_fit"] = plat_calib.predict(tvec)
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = convert_prob_2D(plat_calib.predict(results_dict[data["name"] + "_RF_prob_c"][:,1]))

    # ISO calibration on RF
    method = "ISO"
    if method in calib_methods:
        time_iso_s = time.time()
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_iso_s + time_rf_opt_calib
        
        iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = iso_p_test
        results_dict[f"{data_name}_{method}_fit"] = iso_calib.predict(tvec)
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = convert_prob_2D(iso_calib.predict(results_dict[data["name"] + "_RF_prob_c"][:,1]))

    # CRF calibrator
    method = "PPA"
    if method in calib_methods:
        time_crf_s = time.time()
        crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_crf_s + time_rf_opt_calib
        
        crf_p_test = crf_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = crf_p_test
        results_dict[f"{data_name}_{method}_fit"] = crf_calib.predict(tvec)[:,1]
        if "CL" in metrics: 
            results_dict[f"{data_name}_{method}_prob_c"] = crf_calib.predict(results_dict[data["name"] + "_RF_prob_c"][:,1])

    # Venn abers
    method = "VA"
    if method in calib_methods:
        time_va_s = time.time()
        VA = VA_calib().fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_va_s + time_rf_opt_calib
        
        va_p_test = convert_prob_2D(VA.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = va_p_test
        results_dict[f"{data_name}_{method}_fit"] = VA.predict(tvec)
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = convert_prob_2D(VA.predict(results_dict[data["name"] + "_RF_prob_c"][:,1]))


    # Beta calibration
    method = "Beta"
    if method in calib_methods:
        time_beta_s = time.time()
        beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_beta_s + time_rf_opt_calib
        
        beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = beta_p_test
        results_dict[f"{data_name}_{method}_fit"] = beta_calib.predict(tvec)
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = convert_prob_2D(beta_calib.predict(results_dict[data["name"] + "_RF_prob_c"][:,1]))

    # tree LR calib
    method = "tlr"
    if method in calib_methods:
        time_tlr_s = time.time()
        tlr_calib = treeLR_calib().fit(RF, data["x_train"] ,data["y_train"], data["x_calib"], data["y_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_tlr_s + time_rf_opt_calib
        
        tlr_p_test = tlr_calib.predict(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = tlr_p_test
        # results_dict[data["name"] + "_tlr_fit"] = tlr_calib.predict(convert_prob_2D(tvec))[:,1]
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = tlr_calib.predict(data["X"])

    method = "CT"
    if method in calib_methods:
        time_rf_ct = time.time()
        # prepare the grid space
        ct_params = {
                    "n_estimators": params["search_space"]["n_estimators"],
                    "curt_v":  params["curt_v"],
                    }        
        rf_ct = IR_RF(random_state=seed)
        RS_ct = RandomizedSearchCV(rf_ct, ct_params, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        # if params["oob"] == False:
        #     RS_ct.fit(data["x_train"], data["y_train"])
        # else:
        RS_ct.fit(data["x_train_calib"], data["y_train_calib"])
        rf_ct = RS_ct.best_estimator_
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rf_ct

        rf_ct_p_test = rf_ct.predict_proba(data["x_test"], params["laplace"])
        
        results_dict[f"{data_name}_{method}_prob"] = rf_ct_p_test
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = rf_ct.predict_proba(data["X"], params["laplace"])

    # Elkan calibration
    method = "Elkan"
    if method in calib_methods:
        time_elkan_s = time.time()
        elkan_calib = Elkan_calib().fit(data["y_train"], data["y_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_elkan_s + time_rf_opt_calib
        
        elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = elkan_p_test
        results_dict[f"{data_name}_{method}_fit"] = elkan_calib.predict(tvec)[:,1]

    # RF ranking + ISO
    method = "Rank"
    if method in calib_methods:
        time_rank_s = time.time()
        x_calib_rank = RF.rank(data["x_calib"], class_to_rank=1, train_rank=True)
        x_test_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)

        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, data["y_calib"]) 
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rank_s + time_rf_opt_calib
        
        rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        results_dict[f"{data_name}_{method}_prob"] = rank_p_test
        # tvec_rank = RF.rank_refrence(data["x_test"], class_to_rank=1)
        # results_dict[data["name"] + "_Rank_fit"] = iso_rank.predict(tvec_rank)


    ### models

    method = "DT"
    if method in calib_methods:
        time_dt_opt_s = time.time()
        search_space_dt = {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ['best', 'random'],
                "max_depth": np.arange(2, 100).tolist(),
                "min_samples_split": np.arange(2, 11).tolist(),
                "min_samples_leaf": np.arange(1, 11).tolist(),
                "max_features": ['sqrt', 'log2', None],
                }
        dt = DecisionTreeClassifier(random_state=seed)
        RS_dt = RandomizedSearchCV(dt, search_space_dt, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS_dt.fit(data["x_train_calib"], data["y_train_calib"])
        dt = RS_dt.best_estimator_
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_dt_opt_s

        dt_p_test = dt.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = dt_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = dt.predict_proba(data["X"])

    method = "LR"
    if method in calib_methods:
        time_lr_opt_s = time.time()
        search_space_lr = {
                "penalty": ['l2', None],
                "C": [0.001, 1, 10, 100],
                "solver": ['newton-cholesky', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                "max_iter":  [100, 500, 1000],
                "intercept_scaling": [0.1, 1, 10],
                }
        lr = LogisticRegression(random_state=seed)
        RS_lr = RandomizedSearchCV(lr, search_space_lr, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS_lr.fit(data["x_train_calib"], data["y_train_calib"])
        lr = RS_lr.best_estimator_
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_lr_opt_s
        
        lr_p_test = lr.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = lr_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = lr.predict_proba(data["X"])

    method = "LR_d"
    if method in calib_methods:
        time_lr_opt_s = time.time()
        lr = LogisticRegression(random_state=seed)
        lr.fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_lr_opt_s
        
        lr_p_test = lr.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = lr_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = lr.predict_proba(data["X"])


    method = "SVM"
    if method in calib_methods:
        time_svm_opt_s = time.time()
        search_space_svm = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 1, 10, 100],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto', [0.1, 1, 10]],
                'coef0': [0, 1, 2],
                'shrinking': [True, False],
                'class_weight': [None, 'balanced'],
                'max_iter': [1000, 5000, 10000],
                # 'random_seed': list(range(101)),
                'decision_function_shape': ['ovo', 'ovr'],
                'tol': [1e-4, 1e-3, 1e-2],
                'probability': [True]
            }

        svm = SVC(random_state=seed)
        RS_svm = RandomizedSearchCV(svm, search_space_svm, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS_svm.fit(data["x_train_calib"], data["y_train_calib"])
        svm = RS_svm.best_estimator_
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_svm_opt_s

        svm_p_test = svm.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = svm_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = svm.predict_proba(data["X"])

    method = "SVM_d"
    if method in calib_methods:
        time_svm_opt_s = time.time()

        svm = SVC(random_state=seed, probability=True)
        svm.fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_svm_opt_s
        svm_p_test = svm.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = svm_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = svm.predict_proba(data["X"])

    method = "NN"
    if method in calib_methods:
        time_nn_opt_s = time.time()
        search_space_nn = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [200, 300, 500],
        }

        nn = MLPClassifier(random_state=seed)
        RS_nn = RandomizedSearchCV(nn, search_space_nn, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS_nn.fit(data["x_train_calib"], data["y_train_calib"])
        nn = RS_nn.best_estimator_
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_nn_opt_s

        nn_p_test = nn.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = nn_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = nn.predict_proba(data["X"])


    method = "GNB"
    if method in calib_methods:
        time_gnb_opt_s = time.time()
        search_space_gnb = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        }
        gnb = GaussianNB()
        RS_gnb = RandomizedSearchCV(gnb, search_space_gnb, scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=10, random_state=seed)
        RS_gnb.fit(data["x_train_calib"], data["y_train_calib"])
        gnb = RS_gnb.best_estimator_
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_gnb_opt_s

        gnb_p_test = gnb.predict_proba(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = gnb_p_test
        
        if "CL" in metrics:
            results_dict[f"{data_name}_{method}_prob_c"] = gnb.predict_proba(data["X"])




    if "time" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_time"] = results_dict[f"{data_name}_{method}_runtime"]

    if "acc" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_acc"] = accuracy_score(data["y_test"], np.argmax(results_dict[f"{data_name}_{method}_prob"],axis=1))

    if "auc" in metrics:
        for method in calib_methods:
            fpr, tpr, thresholds = roc_curve(data["y_test"], results_dict[data["name"] + "_" + method +"_prob"][:,1])
            results_dict[f"{data_name}_{method}_auc"] = auc(fpr, tpr)

    if "ece" in metrics:
        for method in calib_methods:
            pt, pp = calibration_curve(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            results_dict[f"{data_name}_{method}_ece"] = mean_squared_error(pt, pp)
            # results_dict[f"{data_name}_{method}_ece"] = 0 # will be updated in the plot function
            # results_dict[f"{data_name}_{method}_ece"] = confidance_ECE(results_dict[f"{data_name}_{method}_prob"], data["y_test"], bins=params["ece_bins"])

    if "brier" in metrics:
        for method in calib_methods: 
            results_dict[f"{data_name}_{method}_brier"] = brier_score_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "logloss" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_logloss"] = log_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "tce" in metrics:
        for method in calib_methods:
            # results_dict[f"{data_name}_{method}_tce"] = mean_squared_error(data["tp_test"], results_dict[f"{data_name}_{method}_prob"][:,1]) # mean squared error for TCE
            
            true_prob_ = np.concatenate((data["tp_test"].reshape(-1,1), (1-data["tp_test"]).reshape(-1,1)), axis=1)
            pred_prob_ = np.concatenate((results_dict[f"{data_name}_{method}_prob"][:,1].reshape(-1,1), (1-results_dict[f"{data_name}_{method}_prob"][:,1]).reshape(-1,1)), axis=1)
            results_dict[f"{data_name}_{method}_tce"] = kl_divergence(true_prob_, pred_prob_)  # TCE is the KL divergence with epsilon to avoid inf problem       


    if "BS" in metrics:

        # u = np.unique(results_dict[f"{data_name}_RF_prob_c"][:,1])
        # c_method = data["tp"].copy()
        # for v in u:
        #     e_index = np.argwhere(results_dict[f"{data_name}_RF_prob_c"][:,1] == v)
        #     e_labels_mean = data["tp"][e_index].mean()
        #     c_method[e_index] = e_labels_mean
        
        # c_method_test = c_method[data["test_index"]]


        for method in calib_methods:
            # calculate C_test
            u = np.unique(results_dict[f"{data_name}_{method}_prob_c"][:,1])
            c_method = data["tp"].copy()
            for v in u:
                e_index = np.argwhere(results_dict[f"{data_name}_{method}_prob_c"][:,1] == v)
                e_labels_mean = data["tp"][e_index].mean()
                c_method[e_index] = e_labels_mean
            
            # c_method_test = c_method[data["test_index"]]
            
            CL = mean_squared_error(results_dict[f"{data_name}_{method}_prob_c"][:,1], c_method) # S,C
            GL = mean_squared_error(c_method, data["tp"]) # C, Q
            IL = mean_squared_error(data["tp"], data["y"]) # Q, Y
            BS = mean_squared_error(data["y"], results_dict[f"{data_name}_{method}_prob_c"][:,1])
            results_dict[f"{data_name}_{method}_CL"] = CL
            results_dict[f"{data_name}_{method}_GL"] = GL
            results_dict[f"{data_name}_{method}_IL"] = IL
            results_dict[f"{data_name}_{method}_BS"] = CL + GL + IL
            results_dict[f"{data_name}_{method}_BS2"] = BS

    if params["plot"] == False:
        filtered_res_dict = {key: value for key, value in results_dict.items() if "_prob" not in key}
        filtered_res_dict = {key: value for key, value in filtered_res_dict.items() if "_fit" not in key}
        return filtered_res_dict
    else:
        for key in results_dict.keys():
            if "_prob" in key or "_fit" in key:  
                results_dict[key] = results_dict[key].tolist()
        return results_dict

def update_runs(ref_dict, new_dict):

    # calib results for every run for the same dataset is aggregated in this function 
    # (ex. acc of every run will be available as an array with the data_calibMethod_metric key)
    # the _prob and _decision keys in the returned result is not meaningfull

    if ref_dict == {}:
        res_dict = new_dict.copy()
        for k in new_dict.keys():
            res_dict[k] = [res_dict[k]]
        return res_dict
    
    res_dict = ref_dict.copy()
    # print("new_dict keys", len(new_dict.keys()))
    # print("res_dict keys", len(res_dict.keys()))
    # print("---------------------------------")
    for k in ref_dict.keys():
        res_dict[k] = ref_dict[k] + [new_dict[k]]

    return res_dict    

def mean_and_ranking_table(results_dict, metrics, calib_methods, data_list, save_results=False, mean_and_rank=True, std=False):

    # save results as txt
    df_dict = {}
    res = ""
    for metric in metrics:
        txt = "Data"
        for method in calib_methods:
            txt += "," + method

        for data in data_list:
            data = str(data)
            txt += "\n"+ data
            for method in calib_methods:
                txt += "," + str(np.array(results_dict[data+ "_" + method + "_"+ metric]).mean())

        txt_data = StringIO(txt)
        df = pd.read_csv(txt_data, sep=",")
        df.set_index('Data', inplace=True)

        if std:
            txt = "Data"
            for method in calib_methods:
                txt += "," + method + "_std"

            for data in data_list:
                data = str(data)
                txt += "\n"+ data
                for method in calib_methods:
                    txt += "," + str(np.array(results_dict[data+ "_" + method + "_"+ metric]).std())

            txt_data = StringIO(txt)
            df_std = pd.read_csv(txt_data, sep=",")
            df_std.set_index('Data', inplace=True)
            df_dict[metric + "_std"] = df_std
        if mean_and_rank:
            mean_res = df.mean()
            if metric == "acc":
                df_rank = df.rank(axis=1, ascending = False)
            else:
                df_rank = df.rank(axis=1, ascending = True)

            mean_rank = df_rank.mean()
            df.loc["Mean"] = mean_res
            df.loc["Rank"] = mean_rank
        df_dict[metric] = df
        if save_results:
            df.to_csv(f"./results/Synthetic/{data}_DataCalib_{metric}.csv",index=True)
        # print("---------------------------------", metric)
        # print(df)
        res += f"--------------------------------- {metric}\n"
        res += str(df)
        
    return df_dict

def make_table(results_dict, metrics, calib_methods, data_list, mean_and_rank=True):

    df_dict = {}

    for metric in metrics:
        df_dict[metric] = pd.DataFrame(columns=calib_methods)
        # df_dict[metric] = {}
        
        for data in data_list:
            data_df = pd.DataFrame(columns=calib_methods)
            for method in calib_methods:
                data_df[method] = np.array(results_dict[data+ "_" + method + "_"+ metric])
            df_dict[metric] = pd.concat([df_dict[metric], data_df], ignore_index=True)
            # df_dict[metric][data] = data_df
        
        if mean_and_rank:
            mean_res = df_dict[metric].mean()
            if metric == "ece" or metric == "brier" or metric == "tce" or metric == "logloss":
                df_rank = df_dict[metric].rank(axis=1, ascending = True)
            else:
                df_rank = df_dict[metric].rank(axis=1, ascending = False)

            mean_rank = df_rank.mean()
            df_dict[metric].loc["Mean"] = mean_res
            df_dict[metric].loc["Rank"] = mean_rank

        
    return df_dict

def exp_mean_rank_through_time(exp_df_all, exp_df, exp_value, value="rank", exp_test="Calibration"):
    value_index = -1
    if value == "mean":
        value_index = -2
    for k in exp_df.keys():
        table = exp_df[k]
        calib_values = table.iloc[value_index].to_dict()
        calib_values[exp_test] = exp_value
        exp_df_all[k] = pd.concat([exp_df_all[k], (pd.DataFrame([calib_values]))])
    return exp_df_all

def find_nearest_index(arr, X):
    sorted_arr = np.sort(arr)
    absolute_diff = np.abs(sorted_arr - X)
    nearest_index = np.argmin(absolute_diff)
    return nearest_index

def predict_bin(prob_true, prob_pred, Y):
    calib_prob = []
    for y in Y:
        index = find_nearest_index(prob_pred, y)
        p = prob_true[index]
        calib_prob.append(p)
    calib_prob = np.array(calib_prob)
    return calib_prob

def plot_probs(exp_data_name, probs_runs, data_runs, params, ref_plot_name="RF", hist_plot=False, calib_plot=False, corrct_ece=True):

    calib_methods = params["calib_methods"]

    # concatinate all runs
    for method in calib_methods:

        all_run_probs = np.zeros(1)
        for prob in probs_runs[f"{exp_data_name}_{method}_prob"]:
            if len(all_run_probs) == 1:
                all_run_probs = prob
            else:
                all_run_probs = np.concatenate((all_run_probs, prob))

        all_run_probs_ref = np.zeros(1)
        for prob in probs_runs[f"{exp_data_name}_{ref_plot_name}_prob"]:
            if len(all_run_probs_ref) == 1:
                all_run_probs_ref = prob
            else:
                all_run_probs_ref = np.concatenate((all_run_probs_ref, prob))

        all_run_y = np.zeros(1)
        for data in data_runs:
            if len(all_run_y) == 1:
                all_run_y = data["y_test"]
            else:
                all_run_y = np.concatenate((all_run_y, data["y_test"]))

        if "synthetic" in params["data_name"]:
            all_run_tp = np.zeros(1)
            for data in data_runs:
                if len(all_run_tp) == 1:
                    all_run_tp = data["tp_test"]
                else:
                    all_run_tp = np.concatenate((all_run_tp, data["tp_test"]))
        
        plt.plot([0, 1], [0, 1], linestyle='--')
        colors = ['black', 'red']
        colors_mean = ['orange', 'blue']
        if "synthetic" in params["data_name"]:
            plt.scatter(all_run_tp, all_run_probs[:,1], marker='.', c=[colors[c] for c in all_run_y.astype(int)]) # Calibrated probs
            plt.scatter(all_run_tp, all_run_probs_ref[:,1], marker='.', c=[colors[c] for c in all_run_y.astype(int)], alpha=0.1) # faded RF probs
        else:
            prob_true, prob_pred = calibration_curve(all_run_y, all_run_probs[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            plt.scatter(prob_true, prob_pred, marker='.', c='darkblue') # Calibrated probs
            prob_true_ref, prob_pred_ref = calibration_curve(all_run_y, all_run_probs_ref[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            plt.scatter(prob_true_ref, prob_pred_ref, marker='.', alpha=0.2, c="gray") # Calibrated probs
            
        # plt.scatter(data["tp_train"], probs[f"{exp_data_name}_{ref_plot_name}_prob_train"][:,1], marker='.', c=[colors[c] for c in data["y_train"].astype(int)]) # RF train probs 


        # ################## Just to test vertical and horisantal averaging
        # bin_means, bin_edges, binnumber = binned_statistic(data["tp_test"], probs[f"{exp_data_name}_{ref_plot_name}_prob"][:,1], bins=100) # Mean of the calibrated probs
        # true_binded = (bin_edges[:-1] + bin_edges[1:])/2
        # plt.scatter(true_binded, bin_means, label='binned statistic of data')
        # v_tce = mean_squared_error((bin_edges[:-1] + bin_edges[1:])/2, bin_means)

        # y = predict_bin(true_binded, bin_means, probs[f"{exp_data_name}_{method}_prob"][:,1])
        # plt.scatter(data["tp_test"], y, marker='.', c=[colors[c] for c in all_run_y.astype(int)]) # Calibrated probs


        # bin_means, bin_edges, binnumber = binned_statistic(probs[f"{exp_data_name}_{method}_prob"][:,1], data["tp_test"], bins=100) # Horizantal Mean of the calibrated probs
        # plt.scatter((bin_edges[:-1] + bin_edges[1:])/2, bin_means, label='binned statistic of data')
        # h_tce = mean_squared_error((bin_edges[:-1] + bin_edges[1:])/2, bin_means)
        # ##################
        if "synthetic" in params["data_name"]:
            calib_tce = mean_squared_error(all_run_tp, all_run_probs[:,1]) # calculate TCE to add to the calib method plot
            calib_tce_ref = mean_squared_error(all_run_tp, all_run_probs_ref[:,1]) # calculate TCE to add to the calib method plot
            tce_txt = f" (TCE {calib_tce:0.5f})"
            tce_ref = f" (TCE {calib_tce_ref:0.5f})"
        else:
            # print("exp_data_name", exp_data_name)
            # print("method", method)
            # print("prob shape", probs[f"{exp_data_name}_{method}_prob"].shape)
            calib_ece = mean_squared_error(prob_true, prob_pred) # confidance_ECE(all_run_probs, all_run_y, bins=params["ece_bins"])
            calib_ece_ref = mean_squared_error(prob_true_ref, prob_pred_ref) # confidance_ECE(all_run_probs_ref, all_run_y, bins=params["ece_bins"])
            if corrct_ece:
                probs_runs[f"{exp_data_name}_{method}_ece"] = [calib_ece]

            ece_txt = f" (ECE {calib_ece:0.5f})"
            ece_ref = f" (ECE {calib_ece_ref:0.5f})"
        # calib_tce = mean_squared_error(data["tp_test"], y) # calculate TCE to add to the calib method plot
        
        if (method == "ISO" or method == "CRF" or method == "Line" or method == "Platt" or method =="Beta" or method =="VA") and calib_plot:
            plt.plot(probs_runs[f"{exp_data_name}_{method}_fit"][0], tvec, c="blue")
        plt.xlabel(f"True probability")
        plt.ylabel("Predicted probability")

        # Add legend
        if "synthetic" in params["data_name"]:
            red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
            black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
            calib_patch = plt.plot([],[], marker='_', markersize=15, color='blue', linestyle='')[0]
            if method == ref_plot_name:
                plt.legend((red_patch, black_patch, calib_patch), ('Class 1', 'Class 0', method + tce_txt), loc='upper left')
            else:
                plt.legend((red_patch, black_patch, calib_patch), ('Class 1', 'Class 0', method + tce_txt+ " RF"+ tce_ref), loc='upper left')
        else:
            orchid_patch = plt.plot([],[], marker='o', markersize=10, color='darkblue', linestyle='')[0]
            gray_patch = plt.plot([],[], marker='o', markersize=10, color='gray', linestyle='')[0]
            calib_patch = plt.plot([],[], marker='_', markersize=15, color='blue', linestyle='')[0]
            plt.legend((orchid_patch, gray_patch, calib_patch), (method + ece_txt, 'RF' + ece_ref, method + " fit"), loc='upper left')
            
        path = f"./results/{params['exp_name']}/plots/{method}"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.title(params['exp_name'] + f" {exp_data_name}")

        if "synthetic" in params["data_name"]:
            plt.savefig(f"{path}/{exp_data_name}_{method}_s.pdf", format='pdf', transparent=True)
        else:
            plt.savefig(f"{path}/{exp_data_name}_{method}.pdf", format='pdf', transparent=True)
        plt.close()

        if hist_plot:
            plt.hist(all_run_probs[:,1], bins=50)
            plt.xlabel(f"probability output of {method}")
            plt.savefig(f"{path}/{method}_{exp_data_name}_hist.pdf", format='pdf', transparent=True)
            plt.close()


def vialin_plot(results_dict, metrics, calib_methods, data_list):

    # save results as txt
    df_dict = {}
    for data in data_list:
        for metric in metrics:
            df = pd.DataFrame(columns=calib_methods)
            for method in calib_methods:
                df[method] = np.array(results_dict[data+ "_" + method + "_"+ metric])
            # print("df", df.head())
            # fig, ax1 = plt.subplots()
            # ax1.violinplot(df, showmeans=True) 
            # ax1.set_xticks(np.arange(len(calib_methods)+1), labels=[""]+ calib_methods)
            # # Rotate the tick labels by 90 degrees
            # plt.xticks(rotation = 90) 
            # plt.savefig(f"results/vialin_plot/{data}_{metric}.pdf", format='pdf', transparent=True)
            # plt.close() 

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the violins with the means and standard errors
            violins = ax.violinplot(df,
                                    positions=np.arange(len(calib_methods)),
                                    showmedians=True)

            # Customization
            ax.set_xticks(np.arange(len(calib_methods)))
            ax.set_xticklabels(calib_methods)
            ax.set_xlabel('Groups')
            ax.set_ylabel('Values')
            ax.set_title('Violin Plot with Standard Error')
            plt.xticks(rotation = 45)
            # # Add error bars for each group
            # for i, (mean, se) in enumerate(zip(means, std_errors)):
            #     ax.errorbar(i, mean, yerr=se, color='black', capsize=5, linewidth=1.5, marker='o')

            plt.grid(True)
            plt.savefig(f"results/vialin_plot/{data}_{metric}.pdf", format='pdf', transparent=True)
            plt.close() 
    return df_dict