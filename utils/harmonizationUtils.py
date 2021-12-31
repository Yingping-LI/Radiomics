#!/usr/bin/env python
# coding: utf-8
from neuroCombat import neuroCombat, neuroCombatFromTraining

'''
Public codes for ComBat methods.
https://github.com/Jfortin1/neuroCombat
'''
def neuroComBat_harmonization(dataframe, feature_columns, batch_col, categorical_cols, continuous_cols, method, ref_batch):
    
    features=dataframe[feature_columns]
    features=transpose_dataframe(features)
    
    covars_columns_list=[batch_col]
    if categorical_cols is not None:
        covars_columns_list=covars_columns_list+categorical_cols
    if continuous_cols is not None:
        covars_columns_list=covars_columns_list+continuous_cols
        
    covars_columns=dataframe[covars_columns_list]

    if method=='parametric_ComBat':
        harmozationResults=neuroCombat(dat=features,  covars=covars_columns,  batch_col=batch_col,
           categorical_cols=categorical_cols,
           continuous_cols=continuous_cols,
           eb=True,
           parametric=True,
           mean_only=False,
           ref_batch=ref_batch)
        
    elif method=='nonParametric_ComBat':
        harmozationResults=neuroCombat(dat=features,  covars=covars_columns,  batch_col=batch_col,
           categorical_cols=categorical_cols,
           continuous_cols=continuous_cols,
           eb=True,
           parametric=False,
           mean_only=False,
           ref_batch=ref_batch)
        
    elif method=='noEB_ComBat':
        harmozationResults=neuroCombat(dat=features,  covars=covars_columns,  batch_col=batch_col,
           categorical_cols=categorical_cols,
           continuous_cols=continuous_cols,
           eb=False,
           parametric=True,
           mean_only=False,
           ref_batch=ref_batch)
        
    else:
        raise ValueError('Undefined harmonization method!')
    
    harmonized_features=harmozationResults["data"]
    estimates=harmozationResults["estimates"]
    info=harmozationResults["info"]
    
    harmonized_features=pd.DataFrame(harmonized_features, index=features.index, columns=features.columns) 
    harmonized_data=dataframe.copy()
    harmonized_data[feature_columns]=transpose_dataframe(harmonized_features)
    
    return harmonized_data, estimates, info


def neuroComBat_harmonization_FromTraning(dataframe, feature_columns, batch_col, estimates):
    print("neuroComBat_harmonization_FromTraning...")
    
    features=dataframe[feature_columns]
    features=transpose_dataframe(features)

    harmozationResults=neuroCombatFromTraining(dat=features, batch=dataframe[batch_col].values, estimates=estimates)
        
    harmonized_features=harmozationResults["data"]
    estimates=harmozationResults["estimates"]
    
    harmonized_features=pd.DataFrame(harmonized_features, index=features.index, columns=features.columns) 
    harmonized_data=dataframe.copy()
    harmonized_data[feature_columns]=transpose_dataframe(harmonized_features)
    
    return harmonized_data, estimates

import pandas as pd    
def transpose_dataframe(df):
    transp_df=pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    
    return transp_df


