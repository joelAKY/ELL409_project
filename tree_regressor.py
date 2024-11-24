import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from collections import deque
import multiprocessing

class TreeNode:
    def __init__(self, best_feature, best_threshold, left, right, value, depth):
        self.best_feature = best_feature
        self.best_threshold = best_threshold
        self.left = left
        self.right = right
        self.value = value
        self.depth = depth

class TreeRegressor:
    def __init__(self, min_samples_split=10, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def build_tree(self, X, y, depth=0):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return TreeNode(best_feature=None, best_threshold=None, left=None, right=None, value=np.mean(y), depth=depth)

        best_feature, best_threshold, best_score, splits = self._find_best_split(X, y)
        if best_feature is None:
            return TreeNode(best_feature=None, best_threshold=None, left=None, right=None, value=np.mean(y), depth=depth)

        left_tree = self._build_tree(splits["left_X"], splits["left_y"], depth + 1)
        right_tree = self._build_tree(splits["right_X"], splits["right_y"], depth + 1)
        return TreeNode(best_feature=best_feature, best_threshold=best_threshold, left=left_tree, right=right_tree, value=np.mean(y), depth=depth)
    
    def _find_best_split(self, X, y):
        best_score = float("inf")
        best_feature = None
        best_threshold = None
        best_splits = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) == 1:
                continue
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])
                score = (
                    np.sum((y[left_mask] - left_mean) ** 2)
                    + np.sum((y[right_mask] - right_mean) ** 2)
                )

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = {
                        "left_X": X[left_mask],
                        "left_y": y[left_mask],
                        "right_X": X[right_mask],
                        "right_y": y[right_mask],
                    }
        #print(f"Best feature: {best_feature}, Best threshold: {best_threshold}, Best score: {best_score}, Best splits: {best_splits}")
        return best_feature, best_threshold, best_score, best_splits
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        if node.left is None and node.right is None:
            return node.value
        
        if x[node.best_feature] <= node.best_threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
        
    def print_tree(self):
        queue = deque([self.tree])
        while queue:
            current_node = queue.popleft()
            print(f"Feature: {current_node.best_feature}, Threshold: {current_node.best_threshold}, Value: {current_node.value}, depth: {current_node.depth}")
            if current_node.left is not None and current_node.right is not None:
                for child in [current_node.left, current_node.right]:
                    queue.append(child)
        
class RandomForestClassifier():

    def __init__(self, num_trees = 10, max_depth=10, min_samples_split = 10):

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        print(f"fitting with {num_trees} trees")

    def fit(self, X, y):
        #args = [((self.bootstrap_data(X, y)), self.max_depth, self.min_samples_split) for i in range(self.num_trees)]
        #pool = multiprocessing.Pool(processes=self.num_trees)
        #self.trees = pool.starmap(fit_tree, args)

        
        self.trees = []
        np.random.seed(42)
        for i in range(self.num_trees):
            X_bootstrap, y_bootstrap = self.bootstrap_data(X, y)
            tree = TreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.build_tree(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        

    def predict(self, X):
     
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        
        predictions = np.array(predictions).mean(axis=0)
        return predictions
    
    

    def bootstrap_data(self, X, y):
        n_samples = X.shape[0]
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[bootstrap_indices], y[bootstrap_indices]

def fit_tree(data, max_depth, min_samples_split):
    tree = TreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    X = data[0]
    y = data[1]
    tree.build_tree(X, y)
    return tree