#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os

'''
Basic Settings for the code
'''
global_basic_settings={
    "experiment_class": "TCGA_MGMT", #"BraTS2021",  "TCGA_IDH",  "TCGA_MGMT"
    ##"501.01_BraTS2021-segNiiData_base", "601.01_BraTS2021-dcmToNiiData_base",  "701.01_BraTS2021-segNiiData-zscore_base"
    "task_name":"TCGA_4.01_isMGMTMethylated_base", 
    "features_for_TCGA": "extracted_features", #"extracted_features", "public_features"
    "feature_selection_method":"AnovaTest", #"RFECV","RFE", AnovaTest, SelectFromModel
    "use_randomSearchCV":True, #False, True
    "harmonization_method": "withoutComBat", # withoutComBat, "parametric_ComBat", nonParametric_ComBat, noEB_ComBat
    "harmonization_label": "is_3T",      #"Tissue.source.site", "is_3T"
    "random_seed": 2021,
}


def get_basic_settings():
   
    return global_basic_settings
