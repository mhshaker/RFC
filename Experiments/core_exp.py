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
from sklearn.manifold import TSNE
import random
from sklearn.preprocessing import MinMaxScaler
import re

np.random.seed(0)


def run_exp(exp_key, exp_values, params):

    exp_res = {}
    data_list = []
    
    for exp_param in exp_values: 
        params[exp_key] = exp_param
        if exp_key == "n_estimators" or exp_key == "max_depth":
            params["search_space"][exp_key] = [exp_param]
        # if exp_key == "Gaussian Mean Delta":
        #     overlap_value = 0.25   
        #     params["class2_mean_min"] -= overlap_value #(exp_param/200)
        #     params["class2_mean_max"] -= overlap_value #(exp_param/200)
        # Data
        exp_data_name = str(exp_param) # data_name + "_" + 
        data_list.append(exp_data_name)

        res_runs = {} # results for each data set will be saved in here.

        # load data for different runs
        data_runs = load_data_runs(params, exp_data_name, params["path"], exp_param) # "../../"

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
        res_list = Parallel(n_jobs=params["n_jobs"])(delayed(cal.calibration)(data, params, seed) for data, params, seed in zip(data_runs, np.repeat(params, len(data_runs)), np.arange(len(data_runs))))
        
        for res in res_list: # res_runs is a dict of all the metrics which are a list of results of multiple runs 
            res_runs = cal.update_runs(res_runs, res) # calib results for every run for the same dataset is aggregated in res_runs (ex. acc of every run as an array)

        if params["plot"]:
            plot_reliability_diagram(params, exp_data_name, res_runs, data_runs)
                
        exp_res.update(res_runs) # merge results of all datasets together

        print(f"exp_param {exp_param} done")

    return exp_res, data_list
        
def load_data_runs(params, exp_data_name, real_data_path=".", exp_key=""):
    data_runs = []
    od = 0
    if "synthetic" in params["data_name"]:
        if params["data_name"] == "synthetic_g":
            X, y, tp, od = dp.make_classification_gaussian_with_true_prob(params["data_size"], 
                                                                    params["n_features"], 
                                                                    class1_mean_min = params["class1_mean_min"], 
                                                                    class1_mean_max = params["class1_mean_max"],
                                                                    class2_mean_min = params["class2_mean_min"], 
                                                                    class2_mean_max = params["class2_mean_max"], 
                                                                    class1_cov_min = params["class1_cov_min"], 
                                                                    class1_cov_max = params["class1_cov_max"],
                                                                    class2_cov_min = params["class2_cov_min"], 
                                                                    class2_cov_max = params["class2_cov_max"]
                                                                    )
        elif "synthetic_nn" in params["data_name"]:
            s_data_seed = re.findall(r'\d+', params["data_name"])
            if s_data_seed:
                s_data_seed = int(s_data_seed[0])
                # Set all random seeds
                random.seed(s_data_seed)
                np.random.seed(s_data_seed)

                # Define a random number of hidden layers
                num_layers = random.randint(1, 5)
                hidden_layers = [random.randint(16, 128) for _ in range(num_layers)]

                generator = dp.SyntheticDataGenerator(num_features=params["n_features"], num_classes=2, hidden_layers=hidden_layers, seed=s_data_seed)
                X, y, tp = generator.generate_data(num_samples=params["data_size"], temperature=0.1, mask_ratio=0, x_grid=False)

            else:
                generator = dp.SyntheticDataGenerator(num_features=params["n_features"], num_classes=2, hidden_layers=[64,32],seed=params["seed"])
                X, y, tp = generator.generate_data(num_samples=params["data_size"], temperature=0.1, mask_ratio=0, x_grid=False)


        if params["plot_data"]:
            colors = ['black', 'red']
            path = f"./results/{params['exp_name']}"
            if not os.path.exists(path):
                os.makedirs(path)
            if params["n_features"] == 2:
                plt.scatter(X[:,0], X[:,1], c=[colors[c] for c in y.astype(int)]) # Calibrated probs,  marker='.'
                red_patch = plt.plot([],[], marker='o', markersize=10, color='red', linestyle='')[0]
                black_patch = plt.plot([],[], marker='o', markersize=10, color='black', linestyle='')[0]
                plt.legend((red_patch, black_patch), ('Class 1', 'Class 0'), loc='upper left')
                plt.xlabel("X_0")
                plt.ylabel("X_1")

                plt.savefig(f"{path}/data_{exp_key}.pdf", format='pdf', transparent=True)
                plt.close()
            else:
                visualize_tsne(X, y, path, od, params)
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)


        if params["split"] == "CV":
            random.seed(params["seed"])
            np.random.seed(params["seed"])
            data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],params["runs"],tp)
        elif params["split"] == "random_split":
            random.seed(params["seed"])
            np.random.seed(params["seed"])
            data_folds = cal.split_train_calib_test(exp_data_name, X,y,params["test_split"], params["calib_split"],params["runs"],tp)
    else:
        X, y = dp.load_data(params["data_name"], real_data_path)
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        if params["split"] == "CV":
            random.seed(params["seed"])
            np.random.seed(params["seed"])
            data_folds = cal.CV_split_train_calib_test(exp_data_name, X,y,params["cv_folds"],params["runs"])
        elif params["split"] == "random_split":
            random.seed(params["seed"])
            np.random.seed(params["seed"])
            data_folds = cal.split_train_calib_test(exp_data_name, X,y,params["test_split"], params["calib_split"],params["runs"])
    for data in data_folds:    
        data_runs.append(data)
    
    return data_runs

def plot_reliability_diagram(params, exp_data_name, res_runs, data_runs):
    cal.plot_probs(exp_data_name, res_runs, data_runs, params, "RF", True, True, False) 
    cal.plot_ece(exp_data_name, res_runs, data_runs, params, False) 
    
    if params["data_name"] != "synthetic2":
        tmp = params["data_name"]
        params["data_name"] = tmp + "ece"
        cal.plot_probs(exp_data_name, res_runs, data_runs, params, "RF", False) 
        params["data_name"] = tmp


# Function to visualize high-dimensional data using t-SNE
def visualize_tsne(X, labels, path, exp_key, params, perplexity=30, learning_rate=200, n_iter=1000, random_state=0, return_image=False):
    """
    Visualizes high-dimensional data using t-SNE.
    
    Parameters:
    X (numpy.ndarray): High-dimensional data (n_samples, n_features).
    labels (numpy.ndarray): Labels for each point, optional, used for coloring.
    perplexity (float): The perplexity parameter for t-SNE.
    learning_rate (float): The learning rate for t-SNE.
    n_iter (int): Number of iterations for optimization.
    random_state (int): Random state for reproducibility.
    """
    
    # Apply t-SNE to reduce data to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, 
                n_iter=n_iter, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    # Plotting the t-SNE result
    # plt.figure(figsize=(8, 6))
    colors = ['red', 'black']
    if labels is not None:
        # If labels are provided, color the points by their labels
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter(X_tsne[labels == label, 0], X_tsne[labels == label, 1], label="Class " + str(int(label)), c= colors[int(label)]) # alpha=0.7
        plt.legend()
    else:
        # If no labels are provided, plot all points in the same color
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    
    plt.title(f"Gaussian Bhattacharyya distance {exp_key:.2f}")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    exp_index = cal.find_closest_index(params['exp_values'], float(exp_key))


    if return_image == False:
        plt.savefig(f"{path}/data_{exp_index}.pdf", format='pdf', transparent=True)
        plt.close()
    else:
        return plt
