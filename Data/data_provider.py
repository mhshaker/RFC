
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import resample
from sklearn import preprocessing
from scipy.io import arff
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal

def unpickle(file): # for reading the CIFAR dataset
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def load_data(data_name, path="."):   

	df = pd.read_csv(f'{path}/Data/{data_name}.csv')
	features = np.array(df.drop("Class", axis=1))
	target = np.array(df["Class"])

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	if "cifar10" == data_name or "mnist" in data_name:
		features = features.astype('float32')
		features /= 255

	if "digits" == data_name:
		features = features.astype('float32')
		features /= 16
	if "cifar10small" == data_name:
		features = features.astype('float32')
		features /= 256

	return features, target

def load_ood(data_name, split=0.3, calibration_set=True, seed=1):
	features, target = load_data(data_name)
	np.random.seed(seed)
	classes = np.unique(target)
	selected_id = np.random.choice(classes,int(len(classes)/2),replace=False) # select id classes
	selected_id_index = np.argwhere(np.isin(target,selected_id)) # get index of all id instances
	selected_ood_index = np.argwhere(np.isin(target,selected_id,invert=True)) # get index of all not selected classes (OOD)

	target_id    = target[selected_id_index].reshape(-1)
	features_id  = features[selected_id_index].reshape(-1, features.shape[1])
	target_ood   = target[selected_ood_index].reshape(-1)
	features_ood = features[selected_ood_index].reshape(-1, features.shape[1])    

	x_train, x_test_id, y_train, y_test_id  = split_data(features_id, target_id,   split=split, seed=seed)
	_      , x_test_ood, _     , y_test_ood = split_data(features_ood, target_ood, split=split, seed=seed)

	if calibration_set:
		x_test_id, x_calib, y_test_id, y_calib = train_test_split(x_test_id, y_test_id, test_size=0.5, shuffle=True, random_state=1)
	else:
		x_calib, y_calib = 0, 0

	minlen = len(x_test_id)
	if len(x_test_ood) < minlen:
		minlen = len(x_test_ood)

	y_test_idoodmix = np.concatenate((np.zeros(minlen), np.ones(minlen)), axis=0)
	x_test_idoodmix = np.concatenate((x_test_id[:minlen], x_test_ood[:minlen]), axis=0)

	return x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood, x_test_idoodmix, y_test_idoodmix, x_calib, y_calib

def split_data(features, target, split, seed=1):
   x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=split, shuffle=True, random_state=seed, stratify=target)
   return x_train, x_test, y_train, y_test

def balance_dataset(df):
	# Separate majority and minority classes
	y = df.Class.unique()
	df_loss = df[df.Class==y[0]]
	df_win  = df[df.Class==y[1]]
	
	max_len = len(df_loss)
	if len(df_win) > max_len:
		max_len = len(df_win)
	# Upsample minority class
	df_upsampled_win = resample(df_win, 
                                replace=True,           # sample with replacement
                                n_samples=max_len,      # to match majority class
                                random_state=123)       # reproducible results
	 
	# Combine majority class with upsampled minority class
	df_balance = pd.concat([df_loss, df_upsampled_win])
	return df_balance

def load_arff_data(name="adult", convert_to_int=True, type="binary", log=True):
	dataset = arff.load(open(f'/home/mhshaker/projects/uncertainty/Data/{name}.arff', 'r'))
	df = pd.DataFrame(dataset['data'])
	df = df.sample(frac=1).reset_index(drop=True)
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if log:
		print(f"Data name = {name}")
		print(df.head())
		print(df['target'].value_counts())

	if convert_to_int:
		for column in df:
			df[column] = df[column].astype("category").cat.codes
	# print(df.head())
	# exit()
	features = df.drop("target", axis=1)
	# print(features)
	features = preprocessing.scale(features)
	# print(features)

	# features_names = list(features.columns)
	# target_names = ["class1","class2"]

	return features, df.target #features_names, target_names

def load_arff_2(data_name):
	data = arff.loadarff(f"./Data/{data_name}.arff")
	df = pd.DataFrame(data[0])
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if data_name == "MagicTelescope":
		df.drop("ID", axis=1, inplace=True)

	features = df.drop("target", axis=1)
	target = df.target

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	# print(features.head())

	return np.array(features), np.array(target)


def x_y_q(X, n_copy=50, seed=0): # create true probability with repeating X instances n_copy times with different labels assigned by a random choice with prob p drawn from uniform dirstribution
	np.random.seed(seed)
	n_features = X.shape[1]
	n_samples = len(X)

	P = np.random.uniform(0,1,n_samples)

	XX = []
	yy = []
	PP = []
	for x, p in zip(X, P):
		y_r = np.random.choice([0,1], n_copy, p=[1-p, p])
		x_r = np.full((n_copy,n_features), x)

		u , counts = np.unique(y_r, return_counts=True)
		# print(f"u {u} counts {counts}")
		if len(counts) > 1:
			e_p = float(counts[1] / (counts[1] + counts[0]))
		else:
			e_p = float(u[0])
		# print(f"e_p type {type(e_p)} e_p {e_p}")
		# print("---------------------------------")
		# p_r = np.full(n_copy, p)
		# print("p_r", p_r)

		p_r = np.full(n_copy, e_p)
		# print("e_p", p_r)
		# print("r\n", r)
		# print("y", y_r)
		# print("---------------------------------")
		yy.append(y_r)
		XX.append(x_r)
		PP.append(p_r)

	XX = np.array(XX).reshape(-1, n_features)
	yy = np.array(yy).reshape(-1)
	PP = np.array(PP).reshape(-1)

	return XX, yy, PP

def get_pre_x(n_features):

	
	# only for parameters 

	# "class1_mean_min":0, 
    # "class1_mean_max":1,
    # "class2_mean_min":2, 
    # "class2_mean_max":3, 

    # "class1_cov_min":1, 
    # "class1_cov_max":2,
    # "class2_cov_min":1, 
    # "class2_cov_max":2.5, 
	

	pre_x = {
		2: 0.0,
		4: 0.25,
		6: 0.47,
		8: 0.56,
		10: 0.7,
		12: 0.72,
		14: 0.74,
		16: 0.78,
		18: 0.79,
		20: 0.88,
		22: 0.86,
		24: 0.87,
		26: 0.93,
		28: 0.86,
		30: 0.95,
		32: 0.94,
		34: 0.94,
		36: 0.967,

		38: 0.799,
		40: 0.199,
		42: 0.659,
		44: 0.789,
		46: 0.909,
		48: 0.989,
	}

	return pre_x[n_features]

def make_classification_gaussian_with_true_prob(n_samples, 
						n_features, 
						class1_mean_min=0, 
						class1_mean_max=1, 
						class1_cov_min=1, 
						class1_cov_max=2, 
						class2_mean_min=0, 
						class2_mean_max=1, 
						class2_cov_min=1, 
						class2_cov_max=2, 
						seed=0,
						bais_accuracy= 0): #0.76 only for #features exp
	n_samples = int(n_samples / 2)
	# Synthetic data with n_features dimentions and n_classes classes

	np.random.seed(seed)
	
	x_list = np.arange(0, 1, 0.001)
	x_list = np.round(x_list, decimals=3)

	for x in x_list:
		xx = x
		if x == 0 and bais_accuracy != 0:
			xx = get_pre_x(n_features)
		np.random.seed(int(xx* 100)) # change to xx later
		if n_features > 36 and bais_accuracy != 0:
			xx = 1

		mean1 = np.random.uniform(class1_mean_min + xx, class1_mean_max + xx, n_features) #[0, 2, 3, -1, 9]
		cov1 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov1, np.random.uniform(class1_cov_min,class1_cov_max,n_features))

		mean2 = np.random.uniform(class2_mean_min - xx, class2_mean_max - xx,n_features) # [-1, 3, 0, 2, 3]
		cov2 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov2, np.random.uniform(class2_cov_min,class2_cov_max,n_features))

		x1 = np.random.multivariate_normal(mean1, cov1, n_samples)
		x2 = np.random.multivariate_normal(mean2, cov2, n_samples)

		X = np.concatenate([x1, x2])
		true_prob = multivariate_normal.pdf(X, mean2, cov2) * 0.5 / (0.5 * multivariate_normal.pdf(X, mean1, cov1) + 0.5 * multivariate_normal.pdf(X, mean2, cov2))
		y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])


		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
		clf = RandomForestClassifier(n_estimators=100, random_state=seed)  

		# print(">>> f", n_features, " x ", xx)
  
		clf.fit(x_train, y_train)
		accuracy = clf.score(x_test, y_test)
		print(f"{n_features}: {xx} ACC {accuracy}")

		if bais_accuracy == 0:
			break
		if accuracy < bais_accuracy and accuracy > bais_accuracy - 0.05:
			# print(f"{n_features}: {x},")
			# if xx != get_pre_x(n_features):
			# 	print("pre_x did not work")
			break

	return X, y, true_prob

def make_classification_mixture_gaussian_with_true_prob(n_samples, 
						n_features,
						n_clusters, 
						same_cov = True,
						seed=0,
						bais_accuracy= 0): #0.76 only for #features exp
	n_samples = int(n_samples / 2)
	# Synthetic data with n_features dimentions and n_classes classes

	np.random.seed(seed)
	
	means1 = []
	covariances1 = []
	weights1 = []

	means2 = []
	covariances2 = []
	weights2 = []

	for i in range(n_clusters):
		mean_min_max = np.random.randint(1, 20, size=2)
		cov_min_max = np.random.randint(1, 5, size=2)
		means1.append(np.random.uniform(mean_min_max.min(), mean_min_max.max(), n_features)) # [0, 2, 3, -1, 9]
		cov1 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov1, np.random.uniform(cov_min_max.min(), cov_min_max.max(),n_features))
		covariances1.append(cov1)
		weights1.append(1 / n_clusters)  # Equal weights1 for all components

		mean_min_max = np.random.randint(1, 20, size=2)
		cov2_min_max = np.random.randint(1, 5, size=2)
		if same_cov:
			cov2_min_max = cov_min_max
		means2.append(np.random.uniform(mean_min_max.min(), mean_min_max.max(), n_features)) # [0, 2, 3, -1, 9]
		cov2 = np.zeros((n_features,n_features))
		np.fill_diagonal(cov2, np.random.uniform(cov2_min_max.min(), cov2_min_max.max(),n_features))
		covariances2.append(cov2)
		weights2.append(1 / n_clusters)  # Equal weights for all components

	data1 = []
	for mean, covariance in zip(means1, covariances1):
		data1.append(np.random.multivariate_normal(mean, covariance, n_samples))

	data2 = []
	for mean, covariance in zip(means2, covariances2):
		data2.append(np.random.multivariate_normal(mean, covariance, n_samples))

	x1 = np.concatenate(data1, axis=0)
	x2 = np.concatenate(data2, axis=0)
	X = np.concatenate([x1, x2])

	# Calculate the PDF of the mixture Gaussian distributions
	mixture_pdf1 = np.zeros(len(X))
	for mean, covariance, weight in zip(means1, covariances1, weights1):
		mixture_pdf1 += multivariate_normal(mean=mean, cov=covariance).pdf(X) * weight 

	mixture_pdf2 = np.zeros(len(X))
	for mean, covariance, weight in zip(means2, covariances2, weights2):
		mixture_pdf2 += multivariate_normal(mean=mean, cov=covariance).pdf(X) * weight 

	# Calculate the probabilities of each sample belonging to class 1
	prob_class1 = mixture_pdf1 * 0.5  # Class 1 prior probability is 0.5
	prob_class2 = mixture_pdf2 * 0.5  # Class 2 prior probability is 0.5

	# Split true probabilities into two parts, one for each class
	true_prob = prob_class2 / (prob_class1 + prob_class2)

	y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])

	return X, y, true_prob

from sklearn.datasets import make_regression

def make_classification_with_true_prob2(n_samples, n_features, n_classes=2, seed=0):
	X, tp = make_regression(n_samples, n_features, tail_strength=0) # make regression data
	y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold
	return X, y, tp

def make_classification_with_true_prob3(n_samples, w=2, noise_mu=0, noise_sigma=0.1, seed=0):
	# y = x.w + noise
	n = np.random.normal(noise_mu, noise_sigma, n_samples)
	x = np.random.uniform(-1,1,n_samples)
	tp = x * w + n

	y = np.where(tp>0, 1, 0) # create classification labels by setting a threshold
	return x.reshape(-1, 1), y, x

def make_classification_with_true_prob_logestic(n_samples, n_features, mean_true_prob=0.8, std_true_prob= 0.2, seed=0):
	np.random.seed(seed)

	true_prob = np.random.normal(loc=mean_true_prob, scale=std_true_prob, size=n_samples)
	true_prob = np.where(true_prob<0, 0, true_prob) # clip negetive values
	true_prob = np.where(true_prob>1, 1, true_prob) # clip greater than 1

	logit = np.log(true_prob / (1-true_prob))

	mean1 = np.random.uniform(-1,1,n_features) #[0, 2, 3, -1, 9]
	cov1 = np.zeros((n_features,n_features))
	np.fill_diagonal(cov1, np.random.uniform(0,1,n_features))
	X = np.random.multivariate_normal(mean1, cov1, n_samples)

	beta_cof = np.random.uniform(-1,1,n_features)

	alpha = logit - np.mean(beta_cof * X, axis=1)

	y = []
	for tp in true_prob:
		y.append(np.random.binomial(1, tp , 1))
	# logit = alpha + beta_cof * X
	# true_prob = 1/(1 + np.exp(-logit))

	return X, y, true_prob 

def make_classification_with_true_prob_3(n_samples, n_features, seed=0):
	# Set random seed for reproducibility
	np.random.seed(seed)

	# Generate synthetic dataset
	# X, Y = make_classification(n_samples=n_samples, n_features=n_features, random_state=seed)
	X, Y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=2, n_clusters_per_class=1, random_state=42)

	# Generate true probabilities for each instance

	weights = np.random.rand(X.shape[1])
	bias = np.random.rand()
	linear_combination = np.dot(X, weights) + bias
	# Normalize to [0, 1] using sigmoid function
	true_probabilities = 1 / (1 + np.exp(-linear_combination))


	return X, Y, true_probabilities 
