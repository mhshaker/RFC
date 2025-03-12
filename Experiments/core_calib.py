
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
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from estimators.IR_RF_estimator import IR_RF
from estimators.CRF_estimator import CRF_calib
from estimators.Shaker_calib import Shaker_calib
from estimators.Elkan_estimator import Elkan_calib
from estimators.VA_estimator import VA_calib
from estimators.TLR_estimator import treeLR_calib
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from betacal import BetaCalibration
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import random


tvec = np.linspace(0.01, 0.99, 990)

def convert_prob_2D(prob1D):
    prob_second_class = np.ones(len(prob1D)) - prob1D
    prob2D = np.concatenate((prob_second_class.reshape(-1,1), prob1D.reshape(-1,1)), axis=1)
    return prob2D


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
    seed = int(seed)
    data_name = data["name"]
    calib_methods = params["calib_methods"] 
    metrics = params["metrics"]
    # the retuen is a dict with all the metrics results as well as RF probs and every calibration method decision for every test data point
    # the structure of the keys in the dict is data_calibMethod_metric
    results_dict = {}

    # train model - hyper opt
    if any(method in ["Platt", "ISO", "Beta", "VA", "PPA", "Shaker"] for method in calib_methods):
        time_rf_opt_calib_s = time.time()
        random.seed(seed)
        np.random.seed(seed)
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

    # all input probs to get the fit calib model

    ### full data (train + calib)
    method = "RF_d"
    if method in calib_methods:
        time_rf_d_s = time.time()
        random.seed(seed)
        np.random.seed(seed)
        rf_d = IR_RF(n_estimators=params["search_space"]["n_estimators"][0], random_state=seed).fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rf_d_s

        rf_d_p_test = rf_d.predict_proba(data["x_test"], params["laplace"])
        results_dict[f"{data_name}_{method}_prob"] = rf_d_p_test

    method = "RF_opt"
    if method in calib_methods:
        time_rf_opt_s = time.time()
        random.seed(seed)
        np.random.seed(seed)
        # train model - hyper opt with x_train_calib
        rf = IR_RF(random_state=seed)
        RS_rf_opt = RandomizedSearchCV(rf, params["search_space"], scoring=["neg_brier_score"], refit="neg_brier_score", cv=params["opt_cv"], n_iter=params["opt_n_iter"], random_state=seed)
        RS_rf_opt.fit(data["x_train"], data["y_train"])

        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rf_opt_s
        RF_opt = RS_rf_opt.best_estimator_
        rff_p_test = RF_opt.predict_proba(data["x_test"], params["laplace"]) # 

        results_dict[f"{data_name}_{method}_prob"] = rff_p_test

    method = "RF_large"
    if method in calib_methods:
        time_rfl_s = time.time()

        random.seed(seed)
        np.random.seed(seed)
        rf_l = IR_RF(n_estimators=params["search_space"]["n_estimators"][0]* params["boot_count"], random_state=seed).fit(data["x_train_calib"], data["y_train_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rfl_s

        RF_large_p_test_fd = rf_l.predict_proba(data["x_test"], params["laplace"])

        results_dict[f"{data_name}_{method}_prob"] = RF_large_p_test_fd    

    method = "Platt"
    if method in calib_methods:
        time_platt_s = time.time()
        plat_calib = _SigmoidCalibration().fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_platt_s + time_rf_opt_calib

        plat_p_test = convert_prob_2D(plat_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = plat_p_test
        results_dict[f"{data_name}_{method}_fit"] = plat_calib.predict(tvec)

    # ISO calibration on RF
    method = "ISO"
    if method in calib_methods:
        time_iso_s = time.time()
        iso_calib = IsotonicRegression(out_of_bounds='clip').fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_iso_s + time_rf_opt_calib
        
        iso_p_test = convert_prob_2D(iso_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = iso_p_test
        results_dict[f"{data_name}_{method}_fit"] = iso_calib.predict(tvec)

    # CRF calibrator
    method = "PPA"
    if method in calib_methods:
        time_crf_s = time.time()
        crf_calib = CRF_calib(learning_method="sig_brior").fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_crf_s + time_rf_opt_calib
        
        crf_p_test = crf_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = crf_p_test
        results_dict[f"{data_name}_{method}_fit"] = crf_calib.predict(tvec)[:,1]

    # Shaker calibrator
    method = "Shaker"
    if method in calib_methods:
        time_shaker_s = time.time()
        shaker_calib = Shaker_calib(seed=seed)
        shaker_calib.fit(data["x_calib"], y_p_calib, RF)

        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_shaker_s + time_rf_opt_calib

        shaker_p_test = shaker_calib.predict(data["x_test"], RF)
        results_dict[f"{data_name}_{method}_prob"] = shaker_p_test

    # Venn abers
    method = "VA"
    if method in calib_methods:
        time_va_s = time.time()
        VA = VA_calib().fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_va_s + time_rf_opt_calib
        
        va_p_test = convert_prob_2D(VA.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = va_p_test
        results_dict[f"{data_name}_{method}_fit"] = VA.predict(tvec)

    # Beta calibration
    method = "Beta"
    if method in calib_methods:
        time_beta_s = time.time()
        beta_calib = BetaCalibration(parameters="abm").fit(rf_p_calib[:,1], y_p_calib)
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_beta_s + time_rf_opt_calib
        
        beta_p_test = convert_prob_2D(beta_calib.predict(rf_p_test[:,1]))
        results_dict[f"{data_name}_{method}_prob"] = beta_p_test
        results_dict[f"{data_name}_{method}_fit"] = beta_calib.predict(tvec)


    method = "CT"
    if method in calib_methods:
        time_rf_ct = time.time()
        # prepare the grid space
        ct_params = {
                    "n_estimators": params["search_space"]["n_estimators"],
                    "curt_v":  params["curt_v"],
                    }  
        random.seed(seed)
        np.random.seed(seed)
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

    # Elkan calibration
    method = "Elkan"
    if method in calib_methods:
        time_elkan_s = time.time()
        elkan_calib = Elkan_calib().fit(data["y_train"], data["y_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_elkan_s + time_rf_opt_calib
        
        elkan_p_test = elkan_calib.predict(rf_p_test[:,1])
        results_dict[f"{data_name}_{method}_prob"] = elkan_p_test
        results_dict[f"{data_name}_{method}_fit"] = elkan_calib.predict(tvec)[:,1]

    # tree LR calib
    method = "tlr"
    if method in calib_methods:
        time_tlr_s = time.time()
        tlr_calib = treeLR_calib().fit(RF, data["x_train"] ,data["y_train"], data["x_calib"], data["y_calib"])
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_tlr_s + time_rf_opt_calib
        
        tlr_p_test = tlr_calib.predict(data["x_test"])
        results_dict[f"{data_name}_{method}_prob"] = tlr_p_test

    # RF ranking + ISO
    method = "Rank"
    if method in calib_methods:
        time_rank_s = time.time()
        x_calib_rank = RF.rank(data["x_calib"], class_to_rank=1, train_rank=True, laplace=params["laplace"])
        x_test_rank = RF.rank_refrence(data["x_test"], class_to_rank=1, laplace=params["laplace"])

        iso_rank = IsotonicRegression(out_of_bounds='clip').fit(x_calib_rank, data["y_calib"]) 
        results_dict[f"{data_name}_{method}_runtime"] = time.time() - time_rank_s + time_rf_opt_calib
        
        rank_p_test = convert_prob_2D(iso_rank.predict(x_test_rank))
        results_dict[f"{data_name}_{method}_prob"] = rank_p_test

    if "time" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_time"] = results_dict[f"{data_name}_{method}_runtime"]

    if "acc" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_acc"] = accuracy_score(data["y_test"], np.argmax(results_dict[f"{data_name}_{method}_prob"],axis=1))

    if "ece" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_ece"] = expected_calibration_error(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1], n_bins=params["ece_bins"])

    if "brier" in metrics:
        for method in calib_methods: 
            results_dict[f"{data_name}_{method}_brier"] = brier_score_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "logloss" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_logloss"] = log_loss(data["y_test"], results_dict[f"{data_name}_{method}_prob"][:,1])

    if "tce_mse" in metrics:
        for method in calib_methods:
            results_dict[f"{data_name}_{method}_tce_mse"] = mean_squared_error(data["tp_test"][:,1], results_dict[f"{data_name}_{method}_prob"][:,1]) # mean squared error for TCE
    
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

def calculate_entropy(arr):
    # Flatten the array to ensure we are working with a 1D array
    arr = arr.flatten()
    
    # Count the occurrences of each value in the array
    _, counts = np.unique(arr, return_counts=True)
    
    # Normalize the counts to get probabilities
    probabilities = counts / counts.sum()
    
    # Calculate the entropy using the Shannon entropy formula
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def plot_probs(exp_data_name, probs_runs, data_runs, params, ref_plot_name="RF", hist_plot=False, calib_plot=False, corrct_ece=False):

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
            # plt.scatter(all_run_tp, all_run_probs_ref[:,1], marker='.', c=[colors[c] for c in all_run_y.astype(int)], alpha=0.1) # faded RF probs
        else:
            prob_true, prob_pred = calibration_curve(all_run_y, all_run_probs[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            plt.scatter(prob_true, prob_pred, marker='.', c='darkblue') # Calibrated probs
            prob_true_ref, prob_pred_ref = calibration_curve(all_run_y, all_run_probs_ref[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])
            plt.scatter(prob_true_ref, prob_pred_ref, marker='.', alpha=0.2, c="gray") # Calibrated probs
            
        if "synthetic" in params["data_name"]:
            calib_tce = mean_squared_error(all_run_tp, all_run_probs[:,1]) # calculate TCE to add to the calib method plot
            calib_tce_ref = mean_squared_error(all_run_tp, all_run_probs_ref[:,1]) # calculate TCE to add to the calib method plot
            tce_txt = f" (TCE {calib_tce:0.5f})"
            tce_ref = f" (TCE {calib_tce_ref:0.5f})"
        else:
            calib_ece = mean_squared_error(prob_true, prob_pred) # confidance_ECE(all_run_probs, all_run_y, bins=params["ece_bins"])
            calib_ece_ref = mean_squared_error(prob_true_ref, prob_pred_ref) # confidance_ECE(all_run_probs_ref, all_run_y, bins=params["ece_bins"])
            if corrct_ece:
                probs_runs[f"{exp_data_name}_{method}_ece"] = [calib_ece]

            ece_txt = f" (ECE {calib_ece:0.5f})"
            ece_ref = f" (ECE {calib_ece_ref:0.5f})"
        
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

        exp_index = find_closest_index(params['exp_values'], float(exp_data_name))

        if "synthetic" in params["data_name"]:
            plt.savefig(f"{path}/{method}_RLs{exp_index}.pdf", format='pdf', transparent=True)
        else:
            plt.savefig(f"{path}/{method}_RL_{exp_index}.pdf", format='pdf', transparent=True)
        plt.close()

        if hist_plot:
            plt.hist(all_run_probs[:,1], bins=50)
            plt.xlim(0, 1)
            plt.ylim(0, len(all_run_probs[:,1]) / 3)

            plt.xlabel(f"probability output of {method}")
            plt.savefig(f"{path}/{method}_hist_{exp_index}.pdf", format='pdf', transparent=True)
            plt.close()
            if "synthetic" in params["data_name"]:
                plt.xlim(0, 1)
                plt.ylim(0, len(all_run_probs[:,1]) / 3)
                plt.hist(all_run_tp, bins=50)
                plt.xlabel(f"True probability")
                plt.savefig(f"{path}/TP_hist_{exp_index}.pdf", format='pdf', transparent=True)
                plt.close()

def find_closest_index(my_list, value):
    try:
        index = my_list.index(value)
        return index
    except ValueError:
        # Find the closest value
        closest_index = min(range(len(my_list)), key=lambda i: abs(my_list[i] - value))
        return closest_index


def plot_ece(exp_data_name, probs_runs, data_runs, params, corrct_ece=False):

    path = f"./results/{params['exp_name']}/plots/ECE"
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.title(params['exp_name'] + f" {exp_data_name}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    
    fig, ax = plt.subplots(figsize=(12, 6))

    legend_markers = []
    legend_labels = []
    # Set limits for x and y axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Set aspect of the plot to be equal
    ax.set_aspect('equal')

    calib_methods = params["calib_methods"]
    num_squares = params["ece_bins"]
    side_length = 1 / num_squares

    for i in range(num_squares):
        x = i * side_length
        y = i * side_length
        square = patches.Rectangle((x, y), side_length, side_length, edgecolor='black', facecolor='none')
        ax.add_patch(square)

    # concatinate all runs
    for method in calib_methods:

        all_run_probs = np.zeros(1)
        for prob in probs_runs[f"{exp_data_name}_{method}_prob"]:
            if len(all_run_probs) == 1:
                all_run_probs = prob
            else:
                all_run_probs = np.concatenate((all_run_probs, prob))

        all_run_y = np.zeros(1)
        for data in data_runs:
            if len(all_run_y) == 1:
                all_run_y = data["y_test"]
            else:
                all_run_y = np.concatenate((all_run_y, data["y_test"]))        


        prob_true, prob_pred = calibration_curve(all_run_y, all_run_probs[:,1], n_bins=params["ece_bins"], strategy=params["bin_strategy"])

        for i in range(num_squares):
            x = i * side_length
            y = i * side_length

            i = np.where((prob_pred > x) & (prob_pred <= x+side_length))
            for b_acc in prob_true[i]:
                ax.plot([x, x+side_length], [b_acc, b_acc], color=params["calib_method_colors"][method]) # label=plot_label


        calib_ece = mean_squared_error(prob_true, prob_pred) # confidance_ECE(all_run_probs, all_run_y, bins=params["ece_bins"])
        if corrct_ece:
            probs_runs[f"{exp_data_name}_{method}_ece"] = [calib_ece]

        ent = calculate_entropy(all_run_probs[:,1])
        plot_label =  f"{method} ECE {calib_ece:0.3f} Entropy {ent:0.3f}"
        marker = plt.plot([],[], marker='_', markersize=15, color=params["calib_method_colors"][method], linestyle='')[0]
        legend_markers.append(marker)
        legend_labels.append(plot_label)

    plt.legend(legend_markers, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

    # plt.legend()
    plt.xlabel(f"Bin Confidance")
    plt.ylabel("Bin Accuracy")
    plt.savefig(f"{path}/{exp_data_name}_ECE_all.pdf", format='pdf', transparent=True)
    plt.close()


def vialin_plot(results_dict, metrics, calib_methods, data_list):

    # save results as txt
    df_dict = {}
    for data in data_list:
        for metric in metrics:
            df = pd.DataFrame(columns=calib_methods)
            for method in calib_methods:
                df[method] = np.array(results_dict[data+ "_" + method + "_"+ metric])

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

            plt.grid(True)
            plt.savefig(f"results/vialin_plot/{data}_{metric}.pdf", format='pdf', transparent=True)
            plt.close() 
    return df_dict


def expected_calibration_error(pred_prob, true_labels, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).

    Parameters:
    -----------
    pred_prob : np.ndarray
        A 1D NumPy array of predicted probabilities (confidence scores).

    true_labels : np.ndarray
        A 1D NumPy array of true binary labels (0 or 1).

    n_bins : int, optional (default=10)
        Number of n_bins used to compute the ECE.

    Returns:
    --------
    float
        The Expected Calibration Error.

    Example:
    --------
    >>> pred_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.95])
    >>> true_labels = np.array([0, 0, 1, 1, 1])
    >>> sq_ece = expected_calibration_error(pred_prob, true_labels, n_bins=10)
    >>> print(f"ECE: {sq_ece:.4f}")
    """

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)  # Define bin edges
    bin_indices = np.digitize(pred_prob, bin_edges, right=True) - 1  # Assign n_bins
    
    ece = 0.0  # Initialize ECE
    for i in range(n_bins):
        mask = bin_indices == i  # Get indices for current bin
        if np.any(mask):  # If bin is not empty
            avg_confidence = np.mean(pred_prob[mask])  # Mean predicted probability
            avg_accuracy = np.mean(true_labels[mask])  # Mean actual accuracy
            bin_size = np.sum(mask) / len(pred_prob)  # Proportion of samples in bin
            
            ece += np.abs(avg_confidence - avg_accuracy) * bin_size  # Weighted error

    return ece
