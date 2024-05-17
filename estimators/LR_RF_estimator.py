from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier

from sklearn.ensemble._forest import (_generate_sample_indices, _get_n_samples_bootstrap, ForestClassifier)


from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel

from sklearn.base import is_classifier
from sklearn.base import ClassifierMixin, MultiOutputMixin, RegressorMixin, TransformerMixin

from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
# from ._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    _check_feature_names_in,
)
from sklearn.utils.validation import _num_samples
from sklearn.utils._param_validation import Interval, StrOptions


__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "RandomTreesEmbedding",
]

MAX_INT = np.iinfo(np.int32).max

def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))


    # print("full data shape ", X.shape)


    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    # print("selected index shape ", np.unique(indices).shape)
    return [tree, indices] ############################################################### added code to sklearn to return tree data indexes


class BaseForest_data(BaseForest):

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._validate_estimator()
        if isinstance(self, (RandomForestRegressor, ExtraTreesRegressor)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features=1.0` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestRegressors and ExtraTreesRegressors.",
                    FutureWarning,
                )
        elif isinstance(self, (RandomForestClassifier, ExtraTreesClassifier)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features='sqrt'` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestClassifiers and ExtraTreesClassifiers.",
                    FutureWarning,
                )

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees_indexes = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )
            trees_indexes = np.array(trees_indexes, dtype="O") ######################################### added code to seperate trees and indexes (from the function return)
            trees = trees_indexes[:,0]
            indexes = trees_indexes[:,1]

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self, indexes ################################## also include indexes

class ForestClassifier_data(BaseForest_data, ForestClassifier):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight)

class RandomForestClassifier_data(ForestClassifier_data):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight)

class LR_RF(RandomForestClassifier_data):
    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha


    def _set_oob_score_and_attributes(self, X, y):
        """Compute and set the OOB score and attributes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        """
        self.oob_decision_function_ = super()._compute_oob_predictions(X, y)
        if self.oob_decision_function_.shape[-1] == 1:
            # drop the n_outputs axis if there is a single output
            self.oob_decision_function_ = self.oob_decision_function_.squeeze(axis=-1)
        self.oob_score_ = accuracy_score(
            y, np.argmax(self.oob_decision_function_, axis=1)
        )

    #################################################### my code

    def fit(self, X, y, y_rank, sample_weight=None):
        # convert lables to ranking
        # gnb = GaussianNB()
        # y_pred = gnb.fit(X, y).predict_proba(X)
        # y_argsort = np.argsort(y_pred, axis=1, kind="stable")
        # target_rank = y_pred.copy()
        # for index, (y_r, y_a) in enumerate(zip(target_rank, y_argsort)):
        #     for i in range(len(y_r)):
        #         y_r[y_a[i]] = i
        #     target_rank[index] = y_r

        # self.y_rank = target_rank
        # y_top_rank = np.argmax(self.y_rank, axis=1)

        # self, indexes = super().fit(X, y, sample_weight)
        self, indexes = super().fit(X, y, sample_weight)

        # get neighbor rankings
        borda_dict_list = []
        for estimator, est_data_index in zip(self.estimators_, indexes):
            borda_dict_list.append(self.create_neighbors_ranking(estimator,X[est_data_index], y_rank[est_data_index]))

        self.borda_dict_list = borda_dict_list

        return self

    def create_neighbors_ranking(self, tree, x_tree, y_train_rank_tree):
        tree_borda_dict = {}
        leaf_index_array = tree.apply(x_tree)
        leaf_nodes = np.unique(leaf_index_array, return_counts=False)
        for leaf_node in leaf_nodes:
            leaf_nodes_borda = y_train_rank_tree[np.argwhere(leaf_index_array==leaf_node)].sum(axis=0)
            tree_borda_dict[leaf_node] = np.array(leaf_nodes_borda).reshape(-1)

        return tree_borda_dict

    def get_ranking(self, tree, x_test, borda):
        leaf_index_array = tree.apply(x_test)
        rankings = []
        for leaf_index in leaf_index_array:
            rankings.append(borda[leaf_index]) 
        rankings = np.array(rankings)
        return rankings

    def predict_proba(self, X):
        # return super().predict_proba(X)
        # first level aggregation
        tree_rankings = []
        for i, (estimator, borda) in enumerate(zip(self.estimators_, self.borda_dict_list)):
            tree_rankings.append(self.get_ranking(estimator, X, borda))
        tree_rankings = np.array(tree_rankings)

        # second level aggregation
        RF_rankings = tree_rankings.sum(axis=0)

        # convert ranking to prob
        ranking_sum = RF_rankings.sum(axis=1)
        LR_RF_probs = RF_rankings / ranking_sum[:, np.newaxis]

        return LR_RF_probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def convert_to_ranking(features, target):
    gnb = GaussianNB()
    y_pred = gnb.fit(features, target).predict_proba(features)
    y_argsort = np.argsort(y_pred, axis=1, kind="stable")
    target_rank = y_pred.copy()
    for index, (y_r, y_a) in enumerate(zip(target_rank, y_argsort)):
        for i in range(len(y_r)):
            y_r[y_a[i]] = i
        target_rank[index] = y_r
    y_top_rank = np.argmax(target_rank, axis=1)
    return target_rank, y_top_rank
