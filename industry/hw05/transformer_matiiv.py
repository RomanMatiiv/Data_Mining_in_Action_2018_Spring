# coding=utf-8
import numpy as np
from sklearn.base import TransformerMixin
from collections import Counter

from sklearn.preprocessing import OneHotEncoder


LR_PARAMS_DICT = {'C': 6.4578947368421051, 'dual': False, 'fit_intercept': False, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 1e-07, 'warm_start': False}


class CustomTransformer(TransformerMixin):
    
    def fit(self, X, y):
        self.ohe=OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(X)
        
        return self
    
    def transform(self, X):

        X_new=self.ohe.transform(X)
        return X_new
