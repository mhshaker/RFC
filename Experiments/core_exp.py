# imports
import sys
import os
sys.path.append('../../') # to access the files in higher directories
sys.path.append('../') # to access the files in higher directories
import Data.data_provider as dp
# import core_calib as cal
from Experiments import cal
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


np.random.seed(0)


def run_exp(exp_key, exp_values, params):

    exp_res = {}
    data_list = []
    
    for exp_param in exp_values: 
        params[exp_key] = exp_param
        if exp_key == "n_estimators" or exp_key == "max_depth":
            params["search_space"][exp_key] = [exp_param]
        # Data
        exp_data_name = str(exp_param) # data_name + "_" + 
        data_list.append(exp_data_name)

        res_runs = {} # results for each data set will be saved in here.

        # load data for different runs
        data_runs = load_data_runs(params, exp_data_name, params["path"], exp_key) # "../../"

        # to change the calib set size (only for calib size experiment)
        if exp_key == "calib_size":
            for data in data_runs:    
                calib_size = int(params["calib_size"] / 100 * len(data["x_calib"]))
                for start_index in range(len(data["x_calib"]) - calib_size): # the for is to find a subset of calib data such that it contains all the class lables
                    if len(np.unique(data["y_calib"][start_index : start_index+calib_size])) > 1: 
                        data["x_calib"] = data["x_calib"][start_index : start_index+calib_size]
                        data["y_calib"] = data["y_calib"][start_index : start_index+calib_size]
                        break
                # print(f"data size train {len(data['x_train'])} test {len(data['x_test'])} calib {len(data['x_calib'])}")
        # for data in data_folds/randomsplits running the same dataset multiple times - res_list is a list of all the results on given metrics
        res_list = Parallel(n_jobs=-1)(delayed(cal.calibration)(data, params, seed) for data, params, seed in zip(data_runs, np.repeat(params, len(data_runs)), np.arange(len(data_runs))))
        
        for res in res_list: # res_runs is a dict of all the metrics which are a list of results of multiple runs 
            res_runs = cal.update_runs(res_runs, res) # calib results for every run for the same dataset is aggregated in res_runs (ex. acc of every run as an array)

        if params["plot"]:
            plot_reliability_diagram(params, exp_data_name, res_runs, data_runs)
                
        exp_res.update(res_runs) # merge results of all datasets together

        print(f"exp_param {exp_param} done")

    return exp_res, data_list
        
def load_data_runs(params, exp_data_name, real_data_path=".", exp_key=""):
    data_runs = []
    if "synthetic" in params["data_name"]:
        if params["data_name"] == "synthetic":
            X, y, tp = dp.make_classification_gaussian_with_true_prob(params["data_size"], 
                                                                    params["n_features"], 
                                                                    class1_mean_min = params["class1_mean_min"], 
                                                                    class1_mean_max = params["class1_mean_max"],
                                                                    class2_mean_min = params["class2_mean_min"], 
                                                                    class2_mean_max = params["class2_mean_max"], 
                                                                    class1_cov_min = params["class1_cov_min"], 
                                                                    class1_cov_max = params["class1_cov_max"],
                                                                    class2_cov_min = params["class2_cov_min"], 
                                                                    class2_cov_max = params["class2_cov_max"], 
                                                                    bais_accuracy=params["bais_accuracy"])
            # X, y, tp = dp.make_classification_mixture_gaussian_with_true_prob(params["data_size"], 
            #                                                         params["n_features"], 
            #                                                         4)

        elif params["data_name"] == "synthetic2":
            X_temp, _ = make_classification(n_samples=params["data_size"], 
                        n_features= params["n_features"], 
                        n_informative= params["n_informative"], 
                        n_redundant= params["n_redundant"], 
                        n_repeated= params["n_repeated"], 
                        n_classes=2, 
                        n_clusters_per_class=1, 
                        weights=None, 
                        flip_y=0.05, 
                        class_sep=1.0, 
                        hypercube=True, 
                        shift=0.0, 
                        scale=1.0, 
                        shuffle=True, 
                        random_state=params["seed"])
            X, y, tp = dp.x_y_q(X_temp, n_copy = params["n_copy"], seed = params["seed"])

        elif params["data_name"] == "synthetic3":
            X, y, tp = dp.make_classification_with_true_prob_3(params["data_size"], params["n_features"])

        if params["plot_data"]:
            colors = ['black', 'red']
            plt.scatter(X[:,0], X[:,1], marker='.', c=[colors[c] for c in y.astype(int)]) # Calibrated probs
            red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
            black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
            plt.legend((red_patch, black_patch), ('Class 1', 'Class 0'), loc='upper left')
            plt.xlabel("X_0")
            plt.ylabel("X_1")

            path = f"./results/{params['exp_name']}"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}/data.pdf", format='pdf', transparent=True)
            plt.close()

        if params["split"] == "CV":
            data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],params["runs"],tp)
        elif params["split"] == "random_split":
            data_folds = cal.split_train_calib_test(exp_data_name, X,y,params["test_split"], params["calib_split"],params["runs"],tp)
    else:
        X, y = dp.load_data(params["data_name"], real_data_path)
        if params["split"] == "CV":
            data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],params["runs"])
        elif params["split"] == "random_split":
            data_folds = cal.split_train_calib_test(exp_data_name, X,y,params["test_split"], params["calib_split"],params["runs"])
    for data in data_folds:    
        data_runs.append(data)
    
    return data_runs

def plot_reliability_diagram(params, exp_data_name, res_runs, data_runs):
    cal.plot_probs(exp_data_name, res_runs, data_runs, params, "RF", False, True, False) 
    
    if params["data_name"] != "synthetic2":
        tmp = params["data_name"]
        params["data_name"] = tmp + "ece"
        cal.plot_probs(exp_data_name, res_runs, data_runs, params, "RF", False, True, False) 
        params["data_name"] = tmp
