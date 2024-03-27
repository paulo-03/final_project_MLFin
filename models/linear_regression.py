"""
This python script propose a class containing all useful method in a linear regression.
"""
import numpy as np


class OLS:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x = x_train
        self.y = y_train
        self.len = len(x_train)
        self.weights = None

    def fit(self, ridge=0, lasso=0):
        NotImplementedError

    def predict(self):
        NotImplementedError
