import numpy as np
from sklearn.ensemble import RandomForestClassifier

class BootstrappedRandomForest:
    def __init__(self, base_forest, perturbation=0.1, n_bootstrap_samples=10, random_state=None):
        """
        :param base_forest: A trained RandomForestClassifier or RandomForestRegressor
        :param perturbation: Fraction of trees to sample (0.0 = no perturbation, 1.0 = full bootstrapping)
        :param n_bootstrap_samples: Number of times to resample for stabilization
        :param random_state: Random seed for reproducible results (int or None)
        """
        self.base_forest = base_forest
        self.perturbation = perturbation
        self.n_bootstrap_samples = n_bootstrap_samples
        self.random_state = random_state

    def predict_proba(self, X):
        """Generate perturbed probability predictions."""
        # Set random seed for reproducibility if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_trees = len(self.base_forest.estimators_)
        n_sampled_trees = max(1, int(self.perturbation * n_trees))  # Control perturbation level

        probas = []
        for _ in range(self.n_bootstrap_samples):
            sampled_trees = np.random.choice(self.base_forest.estimators_, size=n_sampled_trees, replace=True)
            tree_probas = np.mean([tree.predict_proba(X) for tree in sampled_trees], axis=0)
            probas.append(tree_probas)

        return np.mean(probas, axis=0)  # Average over multiple bootstrap samples


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create perturbed forest wrapper
perturbed_rf = BootstrappedRandomForest(rf, perturbation=0.8, n_bootstrap_samples=5, random_state=42)

# Get perturbed probabilities
perturbed_probs = perturbed_rf.predict_proba(X_test)
print(perturbed_probs[:5])  # Print perturbed probabilities for first 5 samples