# coding=utf-8
import numpy as np
from sklearn.base import TransformerMixin
from collections import Counter

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder


LR_PARAMS_DICT = {}


class CustomTransformer(TransformerMixin):
    
    def fit(self, X, y):
        self.ohe=OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(X)
        
        return self
    
    def transform(self, X):

        X_new=self.ohe.transform(X)
        return X_new
