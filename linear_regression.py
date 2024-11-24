import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epoch = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        m = X.shape[0]
        for _ in range(self.epoch):
            print(f"Epoch: {_}")
            y_pred = self.predict(X)
            error = y - y_pred
            print("Nan check", np.isnan(y).any())
            print("Nan check", np.isnan(y_pred).any())
            print("Nan check", np.isinf(X).any())
            print("inf check", np.isinf(self.weights).any())
            self.weights += self.learning_rate * np.dot(X.T, error)/m
            self.bias += self.learning_rate * np.sum(error)/m
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    
    

    