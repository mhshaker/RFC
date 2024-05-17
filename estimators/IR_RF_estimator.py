import numpy as np
from sklearn.ensemble import RandomForestClassifier

def curtailment(tree, x_data, curt_v, laplace_smoothing):
    tree_prob = tree.predict_proba(x_data)
    decision_path_array = tree.decision_path(x_data)

    decision_nodes_array = []
    for data_index in range(len(x_data)):
        # print("---------------------------------")
        # print(decision_path_array)
        # print("---------------------------------")
        node_index = decision_path_array.indices[
            decision_path_array.indptr[data_index] : decision_path_array.indptr[data_index + 1]
        ]

        for leaf_index in node_index[::-1]: # start parsing the decision path from the end
            leaf_values = tree.tree_.value[leaf_index]
            leaf_samples = np.array(leaf_values).sum()
            if leaf_samples >= curt_v:
                for i,v in enumerate(leaf_values[0]):
                    L = laplace_smoothing
                    tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
                break # once statistically enough samples are found, stop parsing back the decision path

        decision_nodes_array.append(node_index)
    return tree_prob

def tree_laplace_corr(tree, x_data, laplace_smoothing, return_counts=False):
    tree_prob = tree.predict_proba(x_data)
    leaf_index_array = tree.apply(x_data)
    for data_index, leaf_index in enumerate(leaf_index_array):
        leaf_values = tree.tree_.value[leaf_index]
        leaf_samples = np.array(leaf_values).sum()
        for i,v in enumerate(leaf_values[0]):
            L = laplace_smoothing
            # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
            if return_counts:
                tree_prob[data_index][i] = v + L
            else:
                tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
    return tree_prob

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def convert_prob_2D(prob1D):
    prob_second_class = np.ones(len(prob1D)) - prob1D
    prob2D = np.concatenate((prob_second_class.reshape(-1,1), prob1D.reshape(-1,1)), axis=1)
    return prob2D

class IR_RF(RandomForestClassifier):

    def __init__(
                    self,
                    n_estimators=100,
                    criterion="gini",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    max_leaf_nodes=None,
                    bootstrap=True,
                    oob_score=False,
                    class_weight=None,
                    curt_v=0, 
                    laplace=0,
                    **kwargs
                ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.bootstrap = bootstrap

        self.curt_v = curt_v
        self.laplace = laplace


    def predict_proba(self, X, laplace=-1, return_tree_prob=False, classifier_tree=False): # normal random forest predic_proba with the addition of Laplace
        if laplace == -1:
            L = self.laplace
        else:
            L = laplace
        prob_matrix  = []
        for estimator in self.estimators_:
            if classifier_tree:
                tree_output = estimator.predict(X) # only the desision
            elif self.curt_v > 0:
                tree_output = curtailment(estimator, X, self.curt_v, L) # probability
            else:
                tree_output = tree_laplace_corr(estimator,X, L) # probability
            prob_matrix.append(tree_output)
        prob_matrix = np.array(prob_matrix)
        
        if classifier_tree:
            return convert_prob_2D(prob_matrix.mean(axis=0))
        else:
            prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
            if return_tree_prob:
                return prob_matrix
            else:
                return prob_matrix.mean(axis=1)

    def tree_leafcounts(self, X, laplace=1):
        leafcount_matrix  = []
        for estimator in self.estimators_:
            tree_prob = tree_laplace_corr(estimator,X, laplace, return_counts=True)
            leafcount_matrix.append(tree_prob)
        leafcount_matrix = np.array(leafcount_matrix)
        leafcount_matrix = leafcount_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
        return leafcount_matrix

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def rank(self, X, class_to_rank=1, laplace=1, train_rank=False, return_tree_rankings=False):
        probs = self.predict_proba(X, laplace=laplace, return_tree_prob=True)[:,:,class_to_rank]
        count = self.tree_leafcounts(X, laplace=laplace)[:,:,class_to_rank] # not probs but the number of instances in the leafs of the trees
        probs *= count

        order = np.argsort(probs, axis=0, kind="stable")
        order = order.transpose([1,0]) 
        ranks = order.argsort(axis=1)
        
        if train_rank:
            probs = probs.transpose([1,0])
            self.refrence_prob = np.sort(probs, axis=1) # np.sort(probs, axis=0, kind="stable")

        IR_RF_rank = ranks.sum(axis=0) / self.n_estimators

        if return_tree_rankings:
            return ranks
        else:
            return IR_RF_rank
        
    def rank_refrence(self, X, class_to_rank=1, laplace=1): # find where X fals in the ranking of calib data

        probs = self.predict_proba(X, laplace=laplace, return_tree_prob=True)[:,:,class_to_rank]
        count = self.tree_leafcounts(X, laplace=laplace)[:,:,class_to_rank] # not probs but the number of instances in the leafs of the trees
        probs *= count

        ranks = []
        for prob in probs: # loop through data
            data_rank = 0
            for tree_prob, ref_prob in zip(prob, self.refrence_prob): # loop through trees
                # print("refrence tree prob", ref_prob)
                # print("refrence tree prob", np.unique(ref_prob))
                # print("X tree prob", tree_prob)
                # print("nearest", find_nearest_index(np.unique(ref_prob), tree_prob))
                # exit()
                data_rank += find_nearest_index(ref_prob, tree_prob)
            ranks.append(data_rank/self.n_estimators)
        return np.array(ranks)

    def rank_fa(self, X, class_to_rank=1): # rank fa
        favor = self.tree_leafcounts(X, laplace=1)[:,:,class_to_rank] # not favor but the number of instances in the leafs of the trees
        prob_arg_sort = np.argsort(favor, axis=0, kind="stable")

        prob_arg_sort = prob_arg_sort.transpose([1,0]) 
        favor = favor.transpose([1,0]) 
        prob_rank = favor.copy()

        for index, (y_r, y_a) in enumerate(zip(prob_rank, prob_arg_sort)):
            for i in range(len(y_r)):
                y_r[y_a[i]] = i
            prob_rank[index] = y_r

        IR_RF_rank_favor = prob_rank.sum(axis=0)

        against = self.tree_leafcounts(X, laplace=1)[:,:,class_to_rank-1] # not probs but the number of instances in the leafs of the trees
        prob_arg_sort = np.argsort(-against, axis=0, kind="stable")

        prob_arg_sort = prob_arg_sort.transpose([1,0]) 
        against = against.transpose([1,0]) 
        prob_rank = against.copy()

        for index, (y_r, y_a) in enumerate(zip(prob_rank, prob_arg_sort)):
            for i in range(len(y_r)):
                y_r[y_a[i]] = i
            prob_rank[index] = y_r

        IR_RF_rank_against = prob_rank.sum(axis=0)

        IR_RF_rank = IR_RF_rank_favor + np.flip(IR_RF_rank_against)
        return IR_RF_rank

