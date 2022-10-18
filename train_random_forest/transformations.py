'''This module contains the necessary transformation classes
used in the inference pipeline.

Author: Mikel Sagardia
Date: 2022-10-18
'''
from sklearn.base import BaseEstimator, TransformerMixin

class MeanImputer(BaseEstimator, TransformerMixin):
    """Mean missing value imputer.
    The mean is used to fill NA values of the passed variables.
    The variables need to be numerical."""
    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
        self.imputer_dict_ = dict()

    def fit(self, X, y=None):
        '''Learn and persist mean values in a dictionary.'''
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        '''Replace the NA values with the learned mean values.'''
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class ModeImputer(BaseEstimator, TransformerMixin):
    """Mode missing value imputer.
    The mode is used to fill NA values of the passed variables.
    The variables need to be categorical or integers."""
    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
        self.imputer_dict_ = dict()

    def fit(self, X, y=None):
        '''Learn and persist mean values in a dictionary.'''
        self.imputer_dict_ = X[self.variables].mode().iloc[0].to_dict()
        return self

    def transform(self, X):
        '''Replace the NA values with the learned mode values.'''
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

