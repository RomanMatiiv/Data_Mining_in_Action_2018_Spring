from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


TREE_PARAMS_DICT = {'max_depth': 2}
TAU = 0.05


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # y это 0 или 1
            # a - сырое предсказание
            # f(a) = 1 / (1 + exp(-a)) - преобразование в вероятность
            # f'(a) = - exp(a) / (1 + exp(-a))^2 = - f(a) (1 - f(a))
            # log loss это (y log f(a) + (1 - y) log(1 - f(a)))

            # d/da (y log f(a) + (1 - y) log(1 - f(a))) = f'(a) (y/f(a) - (1 - y) / (1 - f(a)))
            fa = 1. / (1 + exp(-curr_pred))
            grad = - fa * (1. - fa) * (y_data / fa - (1. - y_data) / (1. - fa))
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, - grad)
            self.estimators.append(algo)
            curr_pred += self.tau * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        return res > 0.5
