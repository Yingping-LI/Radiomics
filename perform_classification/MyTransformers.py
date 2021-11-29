#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("../")
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from utils.harmonizationUtils import neuroComBat_harmonization, neuroComBat_harmonization_FromTraning

class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    
    References: https://stackoverflow.com/questions/62191643/is-there-a-way-to-force-simpleimputer-to-return-a-pandas-dataframe
    """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)
    
    
class ComBatTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to incorporate the ComBat harmonization into the Pipeline to avoid data leakage.
    """
    
    def __init__(self, feature_columns, setting_label_column, ComBat_method, ref_batch):  
        #print('ComBatTransformer.__init__()...\n')
        self.feature_columns = feature_columns
        self.setting_label_column = setting_label_column
        self.ComBat_method=ComBat_method
        self.ref_batch=ref_batch
    
    def fit(self, X, y = None):
        #print('ComBatTransformer.fit()...\n')
        dataframe = X.copy() # creating a copy to avoid changes to original dataset
        _, self.estimates, _=neuroComBat_harmonization(dataframe, self.feature_columns, self.setting_label_column, self.ComBat_method, self.ref_batch)
        return self

    def transform(self, X, y = None):
        #print('ComBatTransformer.transform()...\n')
        dataframe = X.copy() # creating a copy to avoid changes to original dataset
        harmonized_data, _=neuroComBat_harmonization_FromTraning(dataframe, self.feature_columns, self.setting_label_column, self.estimates)
        
        return harmonized_data
    

class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Select columns from a data frame;
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy=X.copy()
        return X_copy[self.columns]
    
    
    
class DeleteCorrColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Delete the correlated feature columns;
    
    See reference:
    https://stackoverflow.com/questions/66221834/how-to-create-a-custom-python-class-to-be-used-in-pipeline-for-dropping-highly-c
    """
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.correlated_columns = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        
        corr_matrix = X.corr()
        correlated_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
                    
        self.correlated_features = correlated_features
        return self

    def transform(self, X, y=None):
        
        X_=(pd.DataFrame(X)).drop(labels=self.correlated_features, axis=1)
       
        return X_
    
    