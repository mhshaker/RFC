import os
import time
import random
import string
import json
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import core_calib as cal

import Data.data_provider as dp


def generate_readable_short_id(name=""):
    timestamp = int(time.time())  # Get current timestamp
    random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))  # Generate a random 4-character code
    if name == "":
        short_id = f"{timestamp}_{random_chars}"
    else:
        short_id = f"{timestamp}_{name}"
    return short_id

def save_params(params):
    path = f"./results/{params['exp_name']}"
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = f"{path}/params.json"

    # Open the file for writing
    with open(file_name, "w") as json_file:
        json.dump(params, json_file)

def save_results(calib_results_dict, exp_name):
    path = f"./results/{exp_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = f"{path}/results.json"

    # Open the file for writing
    with open(file_name, "w") as json_file:
        json.dump(calib_results_dict, json_file)

def read_results(file_address):
    file_name = f"{file_address}/results.json"
    with open(file_name, "r") as json_file:
        loaded_data = json.load(json_file)

    file_name = f"{file_address}/params.json"
    with open(file_name, "r") as json_file:
        params = json.load(json_file)

    tables = cal.mean_and_ranking_table(loaded_data, 
                                    params["metrics"], 
                                    params["calib_methods"], 
                                    params["exp_values"], 
                                    mean_and_rank=False, 
                                    std=False)
    return loaded_data, tables, params



def save_metrics_to_csv(tables, metrics, exp_name):
    path = f"./results/{exp_name}/metrics"
    if not os.path.exists(path):
        os.makedirs(path)

    for metric in metrics:
        tables[metric].round(5).to_csv(f"{path}/{metric}.csv") 

    for metric in metrics:
        tables[metric+ "_std"].round(5).to_csv(f"{path}/{metric}_std.csv")

def save_metrics_to_latex(tables, metrics, exp_name, sava_std=True):
    path = f"./results/{exp_name}/metrics"
    if not os.path.exists(path):
        os.makedirs(path)

    for metric in metrics:
        tables[metric].round(5).to_latex(f"{path}/{metric}.txt") 

    if sava_std:
        for metric in metrics:
            tables[metric+ "_std"].round(5).to_latex(f"{path}/{metric}_std.txt")

def res_statistics(tables, metrics, path, colors=None):
    path += "/statistics_nf"

    if not os.path.exists(path):
        os.makedirs(path)

    # tables = cal.mean_and_ranking_table(calib_results_dict, 
    #                                     params["metrics"],
    #                                     calib_methods, 
    #                                     params["exp_values"], 
    #                                     mean_and_rank=True, 
    #                                     std=True)
    
    # tables = cal.make_table(calib_results_dict, 
    #                         params["metrics"],
    #                         calib_methods, 
    #                         params["exp_values"])

    for metric in metrics:
        print("metric", metric)
        df = pd.DataFrame(tables[metric])
        avg_rank = df.loc["Rank"]
        df = df.drop(["Mean", "Rank"])

        # Perform the Friedman test
        statistic, p_value = friedmanchisquare(*df.values.T)

        print(f"Friedman Test Statistic: {statistic}")
        print(f"P-value: {p_value}")

        # Check for significance (you can choose your significance level, e.g., 0.05)
        alpha = 0.05
        if True:
            print("The differences between groups are significant.")

            # posthoc_res = sp.posthoc_conover_friedman(df)

            posthoc_res = sp.posthoc_nemenyi_friedman(df)

            # w_df = np.array(df.T)
            # posthoc_res = sp.posthoc_wilcoxon(w_df, p_adjust="hommel")
            # posthoc_res.columns = df.columns
            # posthoc_res = posthoc_res.set_index(df.columns)

            sp.sign_plot(posthoc_res)
            plt.savefig(f"{path}/sign_plot_{metric}.pdf", format='pdf', transparent=True)
            plt.close() 

            plt.figure(figsize=(10, 4), dpi=100)
            # plt.title(f"Critical difference diagram of average score ranks ({metric})")
            if colors != None:    
                sp.critical_difference_diagram(avg_rank, posthoc_res, color_palette=colors)
            else:
                sp.critical_difference_diagram(avg_rank, posthoc_res)

            # path = f"./results/{params['exp_name']}/statistics"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}/CD_{metric}.pdf", format='pdf', transparent=True)
            plt.close() 
        else:
            print("The differences between groups are not significant.")


def res_statistics_ttest(tables, metrics, path, colors=None, extention="_L"):
    path += "/statistics_nf"

    if not os.path.exists(path):
        os.makedirs(path)

    for metric in metrics:
        # print("metric", metric)
        df = pd.DataFrame(tables[metric])

        columns_list = df.columns.tolist()
        for i in range(int(len(columns_list)/2)):
            method_df = df[[columns_list[i], columns_list[i]+extention]]
            avg_rank = method_df.loc["Rank"]
            method_df = method_df.drop(["Mean", "Rank"])

            df_list = [list(method_df[columns_list[i]]), list(method_df[columns_list[i]+extention])]
                        
            ttest_res = sp.posthoc_ttest(df_list, p_adjust = 'holm')

            print(f"{metric} {columns_list[i]} {ttest_res.iloc[0, 1]}")
            if ttest_res.iloc[0, 1] < 0.05:
                
                if avg_rank[1] < avg_rank[0]:
                    print(f"{metric} {columns_list[i]} PPP")
                else:
                    print(f"{metric} {columns_list[i]} NNN")
            else:
                # print(f"{metric} {columns_list[i]} FFF")
                pass
            # print("---------------------------------")

def add_rank_mean(tabels):
    for metric in tabels.keys():
        mean_res = tabels[metric].mean()
        if metric == "ece" or metric == "brier" or metric == "tce" or metric == "logloss":
            df_rank = tabels[metric].rank(axis=1, ascending = True)
        else:
            df_rank = tabels[metric].rank(axis=1, ascending = False)

        mean_rank = df_rank.mean()
        tabels[metric].loc["Mean"] = mean_res
        tabels[metric].loc["Rank"] = mean_rank
    return tabels

from collections import Counter

def real_data_info(data_list):
    info = pd.DataFrame()

    data_size = []
    data_features = []
    data_mjc = []
    for data_name in data_list:
        X, y = dp.load_data(data_name, "../../")
        if len(y) > 0:
            # Sample dataset
            data = {
                'class': y
            }

            # Convert the data dictionary into a DataFrame
            df = pd.DataFrame(data)

            # Calculate majority class percentage
            class_counts = Counter(df['class'])
            total_samples = len(df)
            majority_class = class_counts.most_common(1)[0][0]
            majority_class_count = class_counts[majority_class]
            majority_class_percentage = (majority_class_count / total_samples) * 100

            data_size.append(len(y))
            data_features.append(X.shape[1])
            data_mjc.append(majority_class_percentage)
            # print(f"{data_name} done")
    
    info["Name"] = data_list
    info["instances"] = data_size
    info["features"] = data_features
    info["major class"] = data_mjc

    return info

## Kendalltau test ##

# import scipy.stats as stats
# import numpy as np

# tce_ranks = np.array(tables["tce"].loc["Rank"])
# ece_ranks = np.array(tables["ece"].loc["Rank"])
# brier_ranks = np.array(tables["BS"].loc["Rank"])
# logloss_ranks = np.array(tables["logloss"].loc["Rank"])
# acc_ranks = np.array(tables["acc"].loc["Rank"])

# tau, p_value = stats.kendalltau(tce_ranks, brier_ranks)
# print(f"tau {tau} p_value {p_value}")