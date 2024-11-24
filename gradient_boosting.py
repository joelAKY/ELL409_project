import tree_regressor
import numpy as np
from sklearn.metrics import mean_squared_error

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        for _ in range(self.n_estimators):
            print(f"print run : {_}")
            residuals = y - y_pred
            tree = tree_regressor.TreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.build_tree(X, residuals)
            self.trees.append(tree)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
    