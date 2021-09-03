#!/usr/bin/env python
# coding: utf-8
from neuroCombat import neuroCombat

'''
Public codes for ComBat methods.
https://github.com/Jfortin1/neuroCombat
'''
def neuroComBat_harmonization(dataframe, feature_columns, setting_label_column, method):
    
    features=dataframe[feature_columns]
    features=transpose_dataframe(features)
    setting_labels=dataframe[setting_label_column].to_frame()

    if method=='parametric_ComBat':
        harmonized_features=neuroCombat(dat=features,  covars=setting_labels,  batch_col=setting_label_column,
           categorical_cols=None,
           continuous_cols=None,
           eb=True,
           parametric=True,
           mean_only=False,
           ref_batch=None)
        
    elif method=='nonParametric_ComBat':
        harmonized_features=neuroCombat(dat=features,  covars=setting_labels,  batch_col=setting_label_column,
           categorical_cols=None,
           continuous_cols=None,
           eb=True,
           parametric=False,
           mean_only=False,
           ref_batch=None)
        
    elif method=='noEB_ComBat':
        harmonized_features=neuroCombat(dat=features,  covars=setting_labels,  batch_col=setting_label_column,
           categorical_cols=None,
           continuous_cols=None,
           eb=False,
           parametric=True,
           mean_only=False,
           ref_batch=None)
        
    else:
        raise ValueError('Undefined harmonization method!')
    
    
    harmonized_features=pd.DataFrame(harmonized_features, index=features.index, columns=features.columns) 
    harmonized_data=dataframe.copy()
    harmonized_data[feature_columns]=transpose_dataframe(harmonized_features)
    
    return harmonized_data


import pandas as pd    
def transpose_dataframe(df):
    transp_df=pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    
    return transp_df


